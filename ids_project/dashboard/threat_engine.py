"""
Rule-based threat classification from NetFlow / CICFlowMeter features.
Mimics NIDS-style alerts: family, type, human-readable detail, and indicators.
"""
from __future__ import annotations

import re
from typing import Any

import pandas as pd

NORMAL_LABELS = {"", "no label", "normal", "benign"}


def _to_float(value: Any, default: float = 0.0) -> float:
    num = pd.to_numeric(value, errors="coerce")
    if pd.isna(num):
        return default
    return float(num)


def _to_int(value: Any, default: int = 0) -> int:
    num = pd.to_numeric(value, errors="coerce")
    if pd.isna(num):
        return default
    return int(num)


# OWASP / common service ports often targeted in attacks
_SENSITIVE_TCP_PORTS = {
    22: "SSH",
    23: "Telnet",
    25: "SMTP",
    53: "DNS",
    110: "POP3",
    143: "IMAP",
    445: "SMB",
    1433: "MSSQL",
    3306: "MySQL",
    3389: "RDP",
    5432: "PostgreSQL",
    5900: "VNC",
    8080: "HTTP-Alt",
}

# CICIDS2017-style attack labels (substring match)
_KNOWN_ATTACK_PATTERNS = [
    (r"ddos", "DDoS", "Denial of Service"),
    (r"dos\b", "DoS", "Denial of Service"),
    (r"port.?scan|scan", "Network scan", "Reconnaissance"),
    (r"brute|ftp-patator|ssh-patator|web attack.*brute", "Brute-force", "Credential attack"),
    (r"injection|xss|sql", "Web attack", "Application-layer attack"),
    (r"bot|infiltration", "Malware / botnet", "Lateral movement or C2"),
    (r"heartbleed", "Heartbleed", "SSL/TLS vulnerability exploit"),
]


def _match_dataset_attack(raw_label: str) -> tuple[str | None, str | None, str]:
    """Return (threat_type, threat_family, description) if dataset label is a known attack."""
    s = raw_label.lower().strip()
    for pattern, ttype, family in _KNOWN_ATTACK_PATTERNS:
        if re.search(pattern, s, re.I):
            return ttype, family, f"Traffic labeled in dataset as: {raw_label.strip()}"
    if s and s not in NORMAL_LABELS:
        return "Labeled anomaly", "Unknown", f"Dataset label indicates non-benign class: {raw_label.strip()}"
    return None, None, ""


def categorize_flow(row: Any) -> dict:
    """
    Classify a single flow row (Series or dict with CIC column names).

    Returns:
        is_threat: bool
        label: short title for list views
        threat_type: machine slug e.g. PORT_SCAN, SYN_FLOOD
        threat_family: e.g. Reconnaissance, Denial of Service
        threat_detail: full explanation for analysts
        indicators: list of observable facts
        severity: low | medium | high
    """
    if hasattr(row, "get"):
        get = row.get
    else:
        get = lambda k, d=None: row[k] if k in row else d

    raw_label = str(get("Label", "")).strip()
    raw_lower = raw_label.lower()

    # Known attack names from training data (CICIDS, etc.)
    if raw_lower not in NORMAL_LABELS:
        ttype, family, desc = _match_dataset_attack(raw_label)
        if ttype:
            return {
                "is_threat": True,
                "label": f"{ttype} (dataset)",
                "threat_type": ttype.replace(" ", "_").upper()[:64],
                "threat_family": family or "Malware",
                "threat_detail": desc,
                "indicators": [f"Label: {raw_label}", "Source: benchmark dataset classification"],
                "severity": "high" if "ddos" in raw_lower or "dos" in raw_lower else "medium",
            }

    # Flow features (CICFlowMeter names)
    flow_pkts_per_s = _to_float(get("Flow Pkts/s", 0))
    flow_byts_per_s = _to_float(get("Flow Byts/s", 0))
    syn_flag_cnt = _to_int(get("SYN Flag Cnt", 0))
    ack_flag_cnt = _to_int(get("ACK Flag Cnt", 0))
    rst_flag_cnt = _to_int(get("RST Flag Cnt", 0))
    fin_flag_cnt = _to_int(get("FIN Flag Cnt", 0))
    psh_flag_cnt = _to_int(get("PSH Flag Cnt", 0))
    dst_port = _to_int(get("Dst Port", 0))
    src_port = _to_int(get("Src Port", 0))
    tot_fwd_pkts = _to_int(get("Tot Fwd Pkts", 0))
    tot_bwd_pkts = _to_int(get("Tot Bwd Pkts", 0))
    flow_duration = _to_float(get("Flow Duration", 0))
    totlen_fwd = _to_float(get("TotLen Fwd Pkts", 0))
    totlen_bwd = _to_float(get("TotLen Bwd Pkts", 0))

    protocol = get("Protocol", "")
    try:
        proto_num = int(float(protocol))
    except (TypeError, ValueError):
        proto_num = -1

    indicators: list[str] = []
    score = 0

    # --- Heuristic signals ---
    if flow_pkts_per_s > 1000:
        score += 2
        indicators.append(f"High packet rate: {flow_pkts_per_s:.0f} pkt/s (possible flood or scan)")
    if flow_byts_per_s > 500_000:
        score += 1
        indicators.append(f"High byte rate: {flow_byts_per_s / 1e6:.2f} MB/s")

    syn_no_ack = syn_flag_cnt >= 1 and ack_flag_cnt == 0
    if syn_no_ack:
        score += 2
        indicators.append(f"SYN without ACK: SYN={syn_flag_cnt}, ACK={ack_flag_cnt} (incomplete handshake)")

    if rst_flag_cnt >= 1:
        score += 1
        indicators.append(f"RST flags: {rst_flag_cnt} (connection resets / possible scan or probing)")

    service_name = _SENSITIVE_TCP_PORTS.get(dst_port) or _SENSITIVE_TCP_PORTS.get(src_port)
    if service_name and tot_fwd_pkts > 10:
        score += 1
        indicators.append(f"Sensitive service touch: port {dst_port or src_port} ({service_name})")

    if flow_duration < 100 and tot_fwd_pkts > 20:
        score += 1
        indicators.append(f"Short flow ({flow_duration:.0f} µs) with many forward pkts ({tot_fwd_pkts}) — burst pattern")

    if syn_flag_cnt >= 3:
        indicators.append(f"Multiple SYNs: {syn_flag_cnt} (typical of SYN scan or flood attempts)")

    # UDP flood hint
    if proto_num == 17 and flow_pkts_per_s > 800:
        score += 2
        indicators.append("UDP with sustained high packet rate")

    # ICMP anomaly
    if proto_num in (1, 58) and flow_pkts_per_s > 100:
        score += 2
        indicators.append("ICMP traffic at high rate (possible ping flood or scan)")

    is_threat = score >= 2

    # Primary type (priority: DoS / scan / sensitive / generic)
    threat_type = "ANOMALOUS_FLOW"
    threat_family = "Policy violation"
    title = "Anomalous flow (composite signals)"
    severity = "low"

    if is_threat:
        if proto_num in (1, 58) and flow_pkts_per_s > 100:
            threat_type = "ICMP_FLOOD"
            threat_family = "Denial of Service"
            title = "ICMP flood or ICMP-based scan"
            severity = "high" if flow_pkts_per_s > 500 else "medium"
        elif proto_num == 17 and flow_pkts_per_s > 800:
            threat_type = "UDP_FLOOD"
            threat_family = "Denial of Service"
            title = "UDP flood-like traffic"
            severity = "high"
        elif syn_no_ack and syn_flag_cnt >= 2:
            threat_type = "SYN_SCAN"
            threat_family = "Reconnaissance"
            title = "TCP SYN scan or SYN flood pattern"
            severity = "high" if flow_pkts_per_s > 2000 else "medium"
        elif flow_pkts_per_s > 3000:
            threat_type = "HIGH_RATE_FLOOD"
            threat_family = "Denial of Service"
            title = "High-rate volumetric anomaly (possible DoS)"
            severity = "high"
        elif service_name and tot_fwd_pkts > 10:
            threat_type = "SENSITIVE_SERVICE_ACCESS"
            threat_family = "Initial Access"
            title = f"Sensitive service access ({service_name})"
            severity = "medium"
        elif rst_flag_cnt >= 2 and tot_fwd_pkts > 5:
            threat_type = "CONNECTION_ANOMALY"
            threat_family = "Reconnaissance"
            title = "Abnormal TCP behavior (RST-heavy)"
            severity = "low"

    detail_parts = [
        f"Classification: {title} ({threat_family}).",
        "Observed indicators:",
    ]
    for ind in indicators[:8]:
        detail_parts.append(f" • {ind}")
    if not indicators:
        detail_parts.append(" • (No strong single indicator; composite score triggered.)")

    detail_parts.append(
        f"Recommendation: correlate by source IP and time window; verify against asset inventory and firewall logs."
    )

    threat_detail = "\n".join(detail_parts)

    return {
        "is_threat": is_threat,
        "label": title if is_threat else "Normal",
        "threat_type": threat_type if is_threat else "",
        "threat_family": threat_family if is_threat else "",
        "threat_detail": threat_detail if is_threat else "",
        "indicators": indicators if is_threat else [],
        "severity": severity if is_threat else "none",
    }


def normalize_or_infer_label(row: Any) -> str:
    """Backward-compatible: return display label string only."""
    r = categorize_flow(row)
    if not r["is_threat"]:
        return "Normal"
    return r["label"]


def alert_severity_from_stored_fields(
    label: str | None,
    threat_type: str | None,
    threat_family: str | None = None,
) -> str:
    """Map persisted TrafficLog fields to high | medium | low for dashboard badges."""
    lb = (label or "").lower()
    tt = (threat_type or "").lower()
    tf = (threat_family or "").lower()
    blob = f"{lb} {tt} {tf}"
    if any(
        k in tt
        for k in (
            "ddos",
            "icmp_flood",
            "udp_flood",
            "high_rate",
            "flood",
        )
    ):
        return "high"
    if any(k in blob for k in ("ddos", "dos", "flood", "exploit", "ransom", "injection", "heartbleed")):
        return "high"
    if any(k in tt for k in ("syn_scan", "ml_anomaly", "sensitive_service", "connection_anomaly")):
        return "medium"
    if any(k in blob for k in ("scan", "probe", "brute", "web attack", "credential", "reconnaissance")):
        return "medium"
    if "anomaly" in lb or "ml" in lb:
        return "medium"
    return "low"
