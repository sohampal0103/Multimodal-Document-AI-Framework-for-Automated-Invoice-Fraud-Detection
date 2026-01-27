import argparse
from pathlib import Path

import numpy as np
import pandas as pd
try:
    import xgboost as xgb
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "Failed to import xgboost. On macOS you likely need OpenMP: `brew install libomp`. "
        f"Original error: {exc}"
    )
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate an XGBoost model on a CSV dataset.")
    p.add_argument("--data", default="data/synthetic/features.csv")
    p.add_argument("--model", default="models/xgb_fraud.json")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument(
        "--features",
        default=None,
        help="Comma-separated feature columns. Default: all numeric columns except label.",
    )
    args = p.parse_args()

    df = pd.read_csv(args.data)

    if "label" not in df.columns:
        raise SystemExit("Missing required column: label")

    if args.features:
        feature_cols = [c.strip() for c in args.features.split(",") if c.strip()]
    else:
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        feature_cols = [c for c in numeric_cols if c != "label"]

    if not feature_cols:
        raise SystemExit("No feature columns selected")
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing feature columns in CSV: {missing}")

    X = df[feature_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X = X.to_numpy(dtype=float)
    y = df["label"].to_numpy(dtype=int)

    booster = xgb.Booster()
    booster.load_model(args.model)

    dmat = xgb.DMatrix(X)
    prob = booster.predict(dmat)
    pred = (prob >= args.threshold).astype(int)

    acc = accuracy_score(y, pred)
    prec = precision_score(y, pred, zero_division=0)
    rec = recall_score(y, pred, zero_division=0)
    f1 = f1_score(y, pred, zero_division=0)
    auc = roc_auc_score(y, prob)
    cm = confusion_matrix(y, pred)

    print(f"accuracy: {acc:.4f}")
    print(f"precision: {prec:.4f}")
    print(f"recall: {rec:.4f}")
    print(f"f1: {f1:.4f}")
    print(f"roc_auc: {auc:.4f}")
    print("confusion_matrix [ [tn fp] [fn tp] ]:")
    print(cm)


if __name__ == "__main__":
    main()
