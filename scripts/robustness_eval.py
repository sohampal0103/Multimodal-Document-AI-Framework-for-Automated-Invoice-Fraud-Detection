import argparse

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score


def perturb(df: pd.DataFrame, noise_std: float, flip_p: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = df.copy()

    if "tamper_score" in out.columns and noise_std > 0:
        out["tamper_score"] = np.clip(
            out["tamper_score"].to_numpy(dtype=float) + rng.normal(0, noise_std, size=len(out)),
            0.0,
            1.0,
        )

    for col in ["has_signature", "has_logo", "has_qr"]:
        if col in out.columns and flip_p > 0:
            flips = rng.random(len(out)) < flip_p
            vals = out[col].to_numpy(dtype=int)
            vals[flips] = 1 - vals[flips]
            out[col] = vals

    return out


def train_eval(train_df: pd.DataFrame, test_df: pd.DataFrame, features, seed: int = 7):
    X_train = train_df[list(features)].to_numpy(dtype=float)
    y_train = train_df["label"].to_numpy(dtype=int)
    X_test = test_df[list(features)].to_numpy(dtype=float)
    y_test = test_df["label"].to_numpy(dtype=int)

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
    p = argparse.ArgumentParser(description="Robustness eval via feature perturbations.")
    p.add_argument("--data", default="data/real/features.csv")
    p.add_argument("--noise-std", type=float, default=0.05)
    p.add_argument("--flip-p", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()

    df = pd.read_csv(args.data)
    if "label" not in df.columns:
        raise SystemExit("Dataset must contain a 'label' column")

    features = [c for c in ["tamper_score", "has_signature", "has_logo", "has_qr"] if c in df.columns]
    if not features:
        raise SystemExit("No recognized features found")

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=args.seed, stratify=df["label"])
    clean_f1, clean_auc = train_eval(train_df, test_df, features, seed=args.seed)

    pert_test = perturb(test_df, noise_std=args.noise_std, flip_p=args.flip_p, seed=args.seed)
    rob_f1, rob_auc = train_eval(train_df, pert_test, features, seed=args.seed)

    print("setting,f1,roc_auc")
    print(f"clean,{clean_f1:.4f},{clean_auc:.4f}")
    print(f"perturbed(noise_std={args.noise_std},flip_p={args.flip_p}),{rob_f1:.4f},{rob_auc:.4f}")


if __name__ == "__main__":
    main()
