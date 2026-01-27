import argparse
import json
from pathlib import Path

import pandas as pd

from src.cv_forgery.forgery import forgery_score
from src.preprocessing.preprocess import preprocess_image
from src.yolo_detection.detect import detect_layout


def main() -> None:
    p = argparse.ArgumentParser(
        description="Build a feature CSV from invoices + labels for classifier training."
    )
    p.add_argument("--invoices", required=True, help="Directory of invoice files (images/PDFs)")
    p.add_argument(
        "--labels",
        required=True,
        help="CSV with columns: filename,label (label 0=genuine,1=fraud)",
    )
    p.add_argument("--out", default="data/real/features.csv")
    p.add_argument(
        "--yolo-weights",
        default=None,
        help="Optional YOLO weights path (best.pt) for layout detection",
    )
    args = p.parse_args()

    invoices_dir = Path(args.invoices)
    labels_df = pd.read_csv(args.labels)
    if not {"filename", "label"}.issubset(labels_df.columns):
        raise SystemExit("labels CSV must contain columns: filename,label")

    rows = []
    for _, row in labels_df.iterrows():
        fname = str(row["filename"])
        label = int(row["label"])
        path = invoices_dir / fname
        if not path.exists():
            raise SystemExit(f"Missing invoice file: {path}")

        img = preprocess_image(str(path))
        tamper = float(forgery_score(img))
        layout = detect_layout(img, weights_path=args.yolo_weights)

        feats = {
            "tamper_score": tamper,
            "has_signature": int(bool(layout.get("Signature", False))),
            "has_logo": int(bool(layout.get("Logo", False))),
            "has_qr": int(bool(layout.get("QR Code", False))),
            "label": label,
            "file": fname,
        }
        rows.append(feats)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Wrote features: {out}")


if __name__ == "__main__":
    main()
