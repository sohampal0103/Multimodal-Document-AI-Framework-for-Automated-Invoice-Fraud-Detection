import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def main() -> None:
    p = argparse.ArgumentParser(description="Generate a synthetic invoice-fraud feature dataset.")
    p.add_argument("--out", default="data/synthetic/features.csv", help="Output CSV path")
    p.add_argument("--n", type=int, default=2000, help="Number of rows")
    p.add_argument("--seed", type=int, default=7, help="RNG seed")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)

    tamper = rng.uniform(0, 0.6, size=args.n)
    has_signature = rng.integers(0, 2, size=args.n)
    has_logo = rng.integers(0, 2, size=args.n)
    has_qr = rng.integers(0, 2, size=args.n)

    # Hidden generating process (ground truth).
    z = -1.25 + 5.0 * tamper + 0.75 * (1 - has_signature) + 0.35 * (1 - has_logo) + 0.15 * (1 - has_qr)
    prob = sigmoid(z)
    y = rng.binomial(1, prob)

    df = pd.DataFrame(
        {
            "tamper_score": tamper,
            "has_signature": has_signature,
            "has_logo": has_logo,
            "has_qr": has_qr,
            "label": y,
        }
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    meta = {
        "rows": int(args.n),
        "seed": int(args.seed),
        "schema": list(df.columns),
    }
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Wrote {out} and {out.with_suffix('.meta.json')}")


if __name__ == "__main__":
    main()
