import argparse
from itertools import combinations

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score


def train_eval(X, y, seed: int = 7):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
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
        "seed": seed,
    }

    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=300,
        evals=[(dtest, "test")],
        verbose_eval=False,
        early_stopping_rounds=30,
    )

    prob = booster.predict(dtest)
    pred = (prob >= 0.5).astype(int)
    return float(f1_score(y_test, pred)), float(roc_auc_score(y_test, prob))


def main() -> None:
    p = argparse.ArgumentParser(description="Run simple feature ablations for XGBoost.")
    p.add_argument("--data", default="data/real/features.csv")
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()

    df = pd.read_csv(args.data)
    feature_cols = [c for c in ["tamper_score", "has_signature", "has_logo", "has_qr"] if c in df.columns]
    if "label" not in df.columns:
        raise SystemExit("Dataset must contain a 'label' column")

    y = df["label"].to_numpy(dtype=int)

    print("features,f1,roc_auc")
    for r in range(1, len(feature_cols) + 1):
        for subset in combinations(feature_cols, r):
            X = df[list(subset)].to_numpy(dtype=float)
            f1, auc = train_eval(X, y, seed=args.seed)
            print(f"{'+'.join(subset)},{f1:.4f},{auc:.4f}")


if __name__ == "__main__":
    main()
