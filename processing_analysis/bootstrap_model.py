from __future__ import annotations

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier


def ensure_baseline_model() -> Path:
    model_path = Path(os.getenv("FALLBACK_MODEL_PATH", "/app/models/baseline-ids.joblib"))
    anomaly_model_path = Path(os.getenv("ANOMALY_MODEL_PATH", "/app/models/baseline-anomaly.joblib"))
    csv_path = Path(os.getenv("MODEL_BOOTSTRAP_CSV", "/app/data_Capture/testdata.csv"))

    if model_path.exists() and anomaly_model_path.exists():
        print(f"[bootstrap-model] Existing models found: {model_path}, {anomaly_model_path}")
        return model_path
    if not csv_path.exists():
        raise FileNotFoundError(f"Training CSV not found: {csv_path}")

    df = pd.read_csv(csv_path, low_memory=False)
    if df.empty or "Label" not in df.columns:
        raise RuntimeError("Training CSV is empty or missing Label column.")

    y = (
        pd.Series(df["Label"])
        .astype(str)
        .str.lower()
        .apply(lambda v: 0 if v in {"normal", "benign", "no label"} else 1)
        .to_numpy()
    )
    x = df.drop(
        columns=["Flow ID", "Timestamp", "Src IP", "Dst IP", "Src Port", "Dst Port", "Protocol", "Label"],
        errors="ignore",
    )
    x = x.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    x = x.clip(lower=-1e12, upper=1e12).astype("float32").to_numpy()
    if x.size == 0:
        raise RuntimeError("No numeric features found for baseline model.")

    model = RandomForestClassifier(
        n_estimators=180,
        max_depth=18,
        min_samples_split=4,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    model.fit(x, y)

    # Unsupervised anomaly model for unknown attack patterns.
    benign_x = x[y == 0]
    if len(benign_x) < 64:
        benign_x = x
    anomaly_model = IsolationForest(
        n_estimators=180,
        contamination=0.08,
        random_state=42,
        n_jobs=-1,
    )
    anomaly_model.fit(benign_x)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    joblib.dump(anomaly_model, anomaly_model_path)
    print(f"[bootstrap-model] Trained and saved baseline model: {model_path}")
    print(f"[bootstrap-model] Trained and saved anomaly model: {anomaly_model_path}")
    return model_path


if __name__ == "__main__":
    ensure_baseline_model()
