"""Batch-mode scoring: run the fraud pipeline on a folder and export a CSV report.

This is useful for:
- evaluating a trained model on a directory of invoices
- producing an audit-friendly report (probability + flags)

Example:
  python scripts/batch_score_invoices.py --invoices data/public_benchmark/_smoke_cord10/invoices_test \
    --out reports/scores.csv --threshold 0.5 --skip-donut

Notes:
- Uses the same preprocessing/layout/forgery logic as the Streamlit app.
- For model-based scoring, provide --model or set XGB_MODEL_PATH.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

# Allow running as `python scripts/batch_score_invoices.py` without installing as a package.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.cv_forgery.forgery import forgery_score
from src.document_ai.extractor import extract_invoice_data
from src.preprocessing.preprocess import preprocess_image
from src.yolo_detection.detect import detect_layout
from src.classifier.fraud_classifier import predict_fraud


def _iter_invoice_files(root: Path) -> List[Path]:
    exts = {".png", ".jpg", ".jpeg", ".pdf"}
    files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    return sorted(files)


def main() -> None:
    p = argparse.ArgumentParser(description="Batch score a folder of invoices and export a report CSV.")
    p.add_argument("--invoices", required=True, help="Folder containing invoice images/PDFs")
    p.add_argument("--out", default="reports/batch_scores.csv", help="Output CSV path")
    p.add_argument("--threshold", type=float, default=0.5, help="Decision threshold")
    p.add_argument("--model", default=None, help="Optional XGBoost model path (overrides XGB_MODEL_PATH)")
    p.add_argument("--yolo-weights", default=None, help="Optional YOLO weights path")
    p.add_argument("--skip-donut", action="store_true", help="Skip Donut extraction (faster/offline)")
    p.add_argument("--include-extracted", action="store_true", help="Include extracted JSON as a string column")
    args = p.parse_args()

    invoices_dir = Path(args.invoices)
    if not invoices_dir.exists():
        raise SystemExit(f"Missing invoices dir: {invoices_dir}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.skip_donut:
        os.environ["DONUT_DISABLE"] = "1"

    model_path = args.model or os.getenv("XGB_MODEL_PATH")
    yolo_weights = args.yolo_weights or os.getenv("YOLO_WEIGHTS_PATH")

    files = _iter_invoice_files(invoices_dir)
    if not files:
        raise SystemExit(f"No invoice files found under: {invoices_dir}")

    rows: List[Dict[str, Any]] = []
    for f in files:
        try:
            img = preprocess_image(str(f))
            tamper = float(forgery_score(img))
            layout = detect_layout(img, weights_path=yolo_weights)

            extracted: Dict[str, Any] = {}
            if not args.skip_donut:
                try:
                    extracted = extract_invoice_data(img)
                except Exception:
                    extracted = {}

            features = {
                "tamper_score": float(tamper),
                "has_logo": bool(layout.get("Logo", False)),
                "has_signature": bool(layout.get("Signature", False)),
                "has_qr": bool(layout.get("QR Code", False)),
            }
            prob = float(predict_fraud(features, model_path=model_path))
            pred = 1 if prob >= float(args.threshold) else 0

            anomalies = []
            if tamper > 0.25:
                anomalies.append("Visual tampering detected")
            if not features["has_signature"]:
                anomalies.append("Missing signature")
            if not features["has_logo"]:
                anomalies.append("Missing logo")

            row: Dict[str, Any] = {
                "file": str(f.relative_to(invoices_dir)),
                "prob": prob,
                "pred": pred,
                "threshold": float(args.threshold),
                "tamper_score": features["tamper_score"],
                "has_logo": int(features["has_logo"]),
                "has_signature": int(features["has_signature"]),
                "has_qr": int(features["has_qr"]),
                "anomalies": ";".join(anomalies),
            }
            if args.include_extracted:
                row["extracted"] = str(extracted)

            rows.append(row)
        except Exception as exc:
            rows.append(
                {
                    "file": str(f.relative_to(invoices_dir)),
                    "error": str(exc),
                }
            )

    # Write CSV with union of keys.
    fieldnames = sorted({k for r in rows for k in r.keys()})
    with out_path.open("w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote report: {out_path} ({len(rows)} files)")


if __name__ == "__main__":
    main()
