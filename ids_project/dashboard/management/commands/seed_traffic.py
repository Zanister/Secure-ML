from pathlib import Path

import pandas as pd
from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone
from datetime import timedelta

from dashboard.models import TrafficLog
from dashboard.threat_engine import categorize_flow


class Command(BaseCommand):
    help = "Seed dash_trafficlog from CICIDS-style CSV data."

    def add_arguments(self, parser):
        parser.add_argument(
            "--csv",
            default="/app/data_Capture/testdata.csv",
            help="Path to CSV file (default: /app/data_Capture/testdata.csv)",
        )
        parser.add_argument(
            "--truncate",
            action="store_true",
            help="Delete existing rows before seeding.",
        )
        parser.add_argument(
            "--limit",
            type=int,
            default=0,
            help="Optional row limit for quick testing.",
        )
        parser.add_argument(
            "--keep-original-timestamps",
            action="store_true",
            help="Keep timestamps from CSV instead of shifting them near now.",
        )

    def handle(self, *args, **options):
        csv_path = Path(options["csv"])
        if not csv_path.exists():
            raise CommandError(f"CSV not found: {csv_path}")

        if options["truncate"]:
            deleted, _ = TrafficLog.objects.all().delete()
            self.stdout.write(self.style.WARNING(f"Deleted existing rows: {deleted}"))

        df = pd.read_csv(csv_path, low_memory=False)
        if options["limit"] and options["limit"] > 0:
            df = df.head(options["limit"])

        if "Label" not in df.columns:
            df["Label"] = "Normal"

        def to_int(value):
            num = pd.to_numeric(value, errors="coerce")
            if pd.isna(num):
                return None
            return int(num)

        def to_float(value):
            num = pd.to_numeric(value, errors="coerce")
            if pd.isna(num):
                return None
            return float(num)

        ts = pd.to_datetime(
            df.get("Timestamp", timezone.now()),
            errors="coerce",
            dayfirst=True,
            format="%d/%m/%Y %I:%M:%S %p",
        ).fillna(timezone.now())

        if not options["keep_original_timestamps"] and len(ts) > 0:
            # Shift historical CICIDS data near current time so dashboard 24h filters show data.
            min_ts = ts.min()
            now_anchor = timezone.now() - timedelta(hours=1)
            ts = ts.apply(lambda x: now_anchor + (x - min_ts))

        objs = []
        for i, row in df.iterrows():
            info = categorize_flow(row)
            lbl = info["label"] if info["is_threat"] else "Normal"
            detection_source = "RULE_ENGINE"
            confidence = 0.87 if info["is_threat"] else 0.98
            objs.append(
                TrafficLog(
                    timestamp=ts.loc[i].to_pydatetime() if hasattr(ts.loc[i], "to_pydatetime") else timezone.now(),
                    src_ip=str(row.get("Src IP", "")) or None,
                    dst_ip=str(row.get("Dst IP", "")) or None,
                    src_port=to_int(row.get("Src Port")),
                    dst_port=to_int(row.get("Dst Port")),
                    protocol=str(row.get("Protocol", "")) or None,
                    label=lbl,
                    threat_type=info.get("threat_type") or None,
                    threat_family=info.get("threat_family") or None,
                    threat_detail=info.get("threat_detail") or None,
                    detection_source=detection_source,
                    confidence=confidence,
                    flow_duration=to_int(row.get("Flow Duration")),
                    tot_fwd_pkts=to_int(row.get("Tot Fwd Pkts")),
                    tot_bwd_pkts=to_int(row.get("Tot Bwd Pkts")),
                    fwd_pkts_per_sec=to_float(row.get("Fwd Pkts/s")),
                    bwd_pkts_per_sec=to_float(row.get("Bwd Pkts/s")),
                    flow_byts_per_sec=to_float(row.get("Flow Byts/s")),
                    flow_pkts_per_sec=to_float(row.get("Flow Pkts/s")),
                    flow_iat_mean=to_float(row.get("Flow IAT Mean")),
                    flow_iat_std=to_float(row.get("Flow IAT Std")),
                    totlen_fwd_pkts=to_float(row.get("TotLen Fwd Pkts")),
                    totlen_bwd_pkts=to_float(row.get("TotLen Bwd Pkts")),
                    fwd_pkt_len_max=to_float(row.get("Fwd Pkt Len Max")),
                    fwd_pkt_len_min=to_float(row.get("Fwd Pkt Len Min")),
                    fwd_pkt_len_mean=to_float(row.get("Fwd Pkt Len Mean")),
                    fwd_pkt_len_std=to_float(row.get("Fwd Pkt Len Std")),
                    bwd_pkt_len_max=to_float(row.get("Bwd Pkt Len Max")),
                    bwd_pkt_len_min=to_float(row.get("Bwd Pkt Len Min")),
                    bwd_pkt_len_mean=to_float(row.get("Bwd Pkt Len Mean")),
                    bwd_pkt_len_std=to_float(row.get("Bwd Pkt Len Std")),
                    flow_iat_max=to_float(row.get("Flow IAT Max")),
                    flow_iat_min=to_float(row.get("Flow IAT Min")),
                    fwd_iat_tot=to_float(row.get("Fwd IAT Tot")),
                    fwd_iat_mean=to_float(row.get("Fwd IAT Mean")),
                    fwd_iat_std=to_float(row.get("Fwd IAT Std")),
                    fwd_iat_max=to_float(row.get("Fwd IAT Max")),
                    fwd_iat_min=to_float(row.get("Fwd IAT Min")),
                )
            )

        TrafficLog.objects.bulk_create(objs, batch_size=1000)
        self.stdout.write(self.style.SUCCESS(f"Seeded rows: {len(objs)}"))
