"""Thin wrapper so pipeline code can resolve threat labels (Django app on path in Docker)."""
import sys
from pathlib import Path


def _engine():
    ids_project = Path(__file__).resolve().parents[1] / "ids_project"
    p = str(ids_project)
    if p not in sys.path:
        sys.path.insert(0, p)
    from dashboard.threat_engine import categorize_flow as cf, normalize_or_infer_label as nf

    return cf, nf


def normalize_or_infer_label(row):
    _, nf = _engine()
    return nf(row)


def categorize_flow(row):
    cf, _ = _engine()
    return cf(row)
