import argparse
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np
import pandas as pd

from src.cv_forgery.forgery import forgery_score
from src.document_ai.extractor import extract_invoice_data
from src.preprocessing.preprocess import preprocess_image
from src.yolo_detection.detect import detect_layout


_NUM_RE = re.compile(r"[-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?|[-+]?\d+(?:\.\d+)?")


def _flatten(obj: Any, prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            out.update(_flatten(v, key))
        return out
    if isinstance(obj, list):
        for i, v in enumerate(obj):
            key = f"{prefix}[{i}]"
            out.update(_flatten(v, key))
        return out
    out[prefix] = obj
    return out


def _find_first_number(text: str) -> float:
    m = _NUM_RE.search(text)
    if not m:
        return float("nan")
    s = m.group(0).replace(",", "")
    try:
        return float(s)
    except Exception:
        return float("nan")


def _extract_vl_features(extracted: Dict[str, Any]) -> Dict[str, float]:
    """Convert Donut JSON output into numeric features.

    This is intentionally heuristic so it works even when output schema varies.
    """
    flat = _flatten(extracted)
    keys_joined = " ".join(map(str, flat.keys())).lower()
    vals_joined = " ".join(map(lambda x: str(x).lower(), flat.values()))

    def has_kw(*kws: str) -> float:
        for kw in kws:
            if kw in keys_joined or kw in vals_joined:
                return 1.0
        return 0.0

    raw_str = str(extracted)

    # Try to find likely totals/tax fields.
    total_val = float("nan")
    tax_val = float("nan")
    for k, v in flat.items():
        kk = str(k).lower()
        vv = str(v)
        if total_val != total_val and any(t in kk for t in ["total", "amount", "grand"]):
            total_val = _find_first_number(vv)
        if tax_val != tax_val and any(t in kk for t in ["tax", "gst", "vat"]):
            tax_val = _find_first_number(vv)

    # Fallback: look in the whole string.
    if total_val != total_val:
        total_val = _find_first_number(raw_str)

    return {
        "vl_has_any": 1.0 if bool(extracted) else 0.0,
        "vl_num_keys": float(len(flat)),
        "vl_raw_len": float(len(raw_str)),
        "vl_has_vendor": has_kw("vendor", "seller", "supplier"),
        "vl_has_invoice_no": has_kw("invoice number", "invoice no", "inv no", "invoice #"),
        "vl_has_date": has_kw("date", "invoice date"),
        "vl_has_total": has_kw("total", "grand total", "amount"),
        "vl_has_tax": has_kw("tax", "gst", "vat"),
        "vl_currency_inr": 1.0 if ("₹" in raw_str or "inr" in vals_joined) else 0.0,
        "vl_total_value": float(total_val) if total_val == total_val else float("nan"),
        "vl_tax_value": float(tax_val) if tax_val == tax_val else float("nan"),
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description="Build a feature CSV from invoices + fraud labels using Donut (weak supervision) + CV/layout features."
    )
    p.add_argument("--invoices", required=True, help="Directory of invoice files (images/PDFs)")
    p.add_argument(
        "--labels",
        required=True,
        help="CSV with columns: filename,label (label 0=genuine,1=fraud)",
    )
    p.add_argument("--out", default="data/real/features_vl.csv")
    p.add_argument(
        "--yolo-weights",
        default=None,
        help="Optional YOLO weights path (best.pt) for layout detection",
    )
    p.add_argument(
        "--skip-donut",
        action="store_true",
        help="Skip Donut extraction (faster/offline).",
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

        extracted: Dict[str, Any] = {}
        if not args.skip_donut:
            try:
                extracted = extract_invoice_data(img)
            except Exception:
                extracted = {}

        vl = _extract_vl_features(extracted)

        feats = {
            "tamper_score": tamper,
            "has_signature": int(bool(layout.get("Signature", False))),
            "has_logo": int(bool(layout.get("Logo", False))),
            "has_qr": int(bool(layout.get("QR Code", False))),
            **vl,
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
