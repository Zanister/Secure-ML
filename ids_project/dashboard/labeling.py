"""Backward-compatible exports; logic lives in threat_engine."""
from dashboard.threat_engine import categorize_flow, normalize_or_infer_label

__all__ = ["categorize_flow", "normalize_or_infer_label"]
