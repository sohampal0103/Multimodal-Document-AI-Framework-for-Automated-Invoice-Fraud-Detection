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
from sklearn.model_selection import train_test_split


def main() -> None:
    p = argparse.ArgumentParser(description="Train an XGBoost fraud classifier from a CSV.")
    p.add_argument("--data", default="data/synthetic/features.csv", help="Input CSV")
    p.add_argument("--out", default="models/xgb_fraud.json", help="Output model path")
    p.add_argument(
        "--features",
        default=None,
        help="Comma-separated feature columns. Default: all numeric columns except label.",
    )
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()

    df = pd.read_csv(args.data)
    if "label" not in df.columns:
        raise SystemExit("Missing required column: label")

    if args.features:
        feature_cols = [c.strip() for c in args.features.split(",") if c.strip()]
    else:
        # Default to all numeric columns except label.
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

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": 4,
        "eta": 0.1,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "seed": args.seed,
    }

    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=300,
        evals=[(dtrain, "train"), (dtest, "test")],
        verbose_eval=False,
        early_stopping_rounds=30,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(out)
    print(f"Saved model: {out}")


if __name__ == "__main__":
    main()
