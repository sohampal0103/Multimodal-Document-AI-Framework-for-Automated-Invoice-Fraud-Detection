"""Public benchmark runner: CORD base + synthetic fraud labels (Option A).

This script creates a reproducible, fully-public benchmark by:
1) Downloading CORD via HuggingFace datasets
2) Sampling N documents
3) Generating realistic-imbalance fraud labels (default: 10%)
4) Applying *invoice-like* tamper operations (optionally multi-step)
5) Exporting train/val/test folders + labels.csv
6) Building features (re-using scripts/build_feature_dataset_vl.py)
7) Training XGBoost + tuning threshold on val + evaluating on test

Paper-friendly framing:
- Base documents are fully public (CORD)
- Fraud labels are defined by controlled manipulations (unambiguous ground truth)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Tuple

import numpy as np
from PIL import Image, ImageDraw

try:
    import xgboost as xgb
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "Failed to import xgboost. On macOS you likely need OpenMP: `brew install libomp`. "
        f"Original error: {exc}"
    )

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass(frozen=True)
class BenchmarkConfig:
    dataset: str
    dataset_config: str | None
    n: int
    fraud_rate: float
    seed: int
    test_size: float
    val_size: float
    severity: float
    tamper_ops: Tuple[str, ...]
    multi_step: bool
    max_steps: int
    skip_donut: bool
    threshold_metric: str
    fp_cost: float
    fn_cost: float


def _require_datasets() -> None:
    try:
        import datasets  # noqa: F401
    except Exception as exc:
        raise SystemExit(
            "Missing dependency: datasets. Install training deps: `pip install -r requirements-train.txt`. "
            f"Original error: {exc}"
        )


def _load_cord_dataset():
    """Load the CORD dataset without decoding every image up front."""
    _require_datasets()
    from datasets import load_dataset  # type: ignore

    ds = load_dataset("naver-clova-ix/cord-v2")
    # CORD uses the split name "validation" (not "val").
    splits = [s for s in ["train", "validation", "test"] if s in ds]
    if not splits:
        raise SystemExit("CORD dataset did not contain expected splits")
    return ds, splits


def _sample_cord_images(ds, splits: List[str], *, rng: random.Random, n: int) -> List[Image.Image]:
    """Sample n images from CORD (with replacement if needed) and decode only those."""
    # Build a small index pool (split, idx).
    pool: List[Tuple[str, int]] = []
    for s in splits:
        try:
            ln = len(ds[s])
        except Exception:
            ln = 0
        for i in range(ln):
            pool.append((s, i))

    if not pool:
        raise SystemExit("Failed to index any items from CORD dataset")

    chosen = rng.sample(pool, k=n) if len(pool) >= n else [rng.choice(pool) for _ in range(n)]
    out: List[Image.Image] = []
    for split, idx in chosen:
        item = ds[split][idx]
        img = item.get("image")
        if img is None:
            continue
        if isinstance(img, Image.Image):
            out.append(img.convert("RGB"))
        else:
            out.append(Image.fromarray(np.array(img)).convert("RGB"))

    if len(out) < max(1, int(0.8 * n)):
        raise SystemExit(f"Failed to decode enough images from CORD (wanted {n}, got {len(out)})")
    # If a few were missing, pad by re-sampling.
    while len(out) < n:
        split, idx = rng.choice(pool)
        img = ds[split][idx].get("image")
        if img is None:
            continue
        out.append(img.convert("RGB") if isinstance(img, Image.Image) else Image.fromarray(np.array(img)).convert("RGB"))
    return out


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _rand_rect(rng: random.Random, w: int, h: int, severity: float) -> Tuple[int, int, int, int]:
    # Rect size scales with severity.
    s = 0.08 + 0.22 * _clamp01(severity)
    rw = max(8, int(w * (s + rng.random() * s)))
    rh = max(8, int(h * (s + rng.random() * s)))

    # Bias towards bottom half (signatures/totals often there).
    x0 = rng.randint(0, max(0, w - rw))
    y0 = rng.randint(int(h * 0.45), max(int(h * 0.45), h - rh))
    return x0, y0, x0 + rw, y0 + rh


def _tamper_occlude(img: Image.Image, rng: random.Random, severity: float) -> Image.Image:
    w, h = img.size
    x0, y0, x1, y1 = _rand_rect(rng, w, h, severity)
    out = img.copy()
    draw = ImageDraw.Draw(out)
    # White box + faint gray border.
    draw.rectangle([x0, y0, x1, y1], fill=(255, 255, 255), outline=(220, 220, 220))
    return out


def _tamper_copy_move(img: Image.Image, rng: random.Random, severity: float) -> Image.Image:
    w, h = img.size
    src = _rand_rect(rng, w, h, severity)
    dst = _rand_rect(rng, w, h, severity)

    patch = img.crop(src)
    out = img.copy()
    out.paste(patch, (dst[0], dst[1]))
    return out


def _tamper_stamp(img: Image.Image, rng: random.Random, severity: float) -> Image.Image:
    # Create a scribble-like stamp/signature patch and overlay it.
    w, h = img.size
    pw = max(40, int(w * (0.12 + 0.18 * _clamp01(severity))))
    ph = max(18, int(h * (0.04 + 0.08 * _clamp01(severity))))

    patch = Image.new("RGBA", (pw, ph), (0, 0, 0, 0))
    d = ImageDraw.Draw(patch)

    # Draw random polyline.
    npts = 10 + int(10 * _clamp01(severity))
    pts = []
    for i in range(npts):
        x = int((i / max(1, npts - 1)) * (pw - 1))
        y = rng.randint(0, ph - 1)
        pts.append((x, y))

    thickness = 2 if severity < 0.6 else 3
    d.line(pts, fill=(0, 0, 0, 200), width=thickness)

    # Place near bottom quarter.
    x = rng.randint(0, max(0, w - pw))
    y = rng.randint(int(h * 0.70), max(int(h * 0.70), h - ph))

    out = img.convert("RGBA")
    out.alpha_composite(patch, (x, y))
    return out.convert("RGB")


def _tamper_recompress(img: Image.Image, rng: random.Random, severity: float) -> Image.Image:
    # JPEG round-trip. Lower quality with higher severity.
    q = int(85 - 45 * _clamp01(severity) - rng.random() * 10)
    q = max(25, min(95, q))

    from io import BytesIO

    buf = BytesIO()
    img.save(buf, format="JPEG", quality=q, optimize=True)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _tamper_amount_edit(img: Image.Image, rng: random.Random, severity: float) -> Image.Image:
    """Simulate editing the total/amount field by overwriting a region with new digits."""
    w, h = img.size
    # Prefer bottom-right area.
    bw = max(50, int(w * (0.18 + 0.12 * _clamp01(severity))))
    bh = max(18, int(h * (0.05 + 0.05 * _clamp01(severity))))
    x0 = rng.randint(max(0, w - bw - 8), max(0, w - bw))
    y0 = rng.randint(int(h * 0.60), max(int(h * 0.60), h - bh))
    out = img.copy()
    d = ImageDraw.Draw(out)
    d.rectangle([x0, y0, x0 + bw, y0 + bh], fill=(255, 255, 255), outline=(220, 220, 220))

    # Draw new numeric string. (We avoid font dependencies; PIL uses a default bitmap font.)
    digits = "".join(str(rng.randint(0, 9)) for _ in range(rng.randint(3, 6)))
    if rng.random() < 0.6:
        digits = digits[:-2] + "." + digits[-2:]
    text = digits
    d.text((x0 + 3, y0 + 2), text, fill=(0, 0, 0))
    return out


def _tamper_vendor_swap(img: Image.Image, donor: Image.Image, rng: random.Random, severity: float) -> Image.Image:
    """Simulate vendor/header swap by transplanting a top header strip from another document."""
    w, h = img.size
    header_h = max(20, int(h * (0.12 + 0.08 * _clamp01(severity))))

    base = img.copy()
    donor_r = donor.resize((w, h), resample=Image.BILINEAR)
    patch = donor_r.crop((0, 0, w, header_h))

    # Slight x offset to mimic imperfect alignment.
    xoff = rng.randint(-int(w * 0.03), int(w * 0.03))
    base.paste(patch, (xoff, 0))
    return base


def _tamper_lineitem_add_remove(img: Image.Image, rng: random.Random, severity: float) -> Image.Image:
    """Simulate line-item manipulation by duplicating or erasing a horizontal band."""
    w, h = img.size
    band_h = max(12, int(h * (0.03 + 0.03 * _clamp01(severity))))
    y0 = rng.randint(int(h * 0.30), int(h * 0.70))
    y0 = min(max(0, y0), max(0, h - band_h))

    out = img.copy()
    if rng.random() < 0.5:
        # Remove: white-out the band.
        d = ImageDraw.Draw(out)
        d.rectangle([0, y0, w, y0 + band_h], fill=(255, 255, 255))
    else:
        # Add: copy the band and paste it slightly below.
        patch = img.crop((0, y0, w, y0 + band_h))
        dy = rng.randint(band_h, min(band_h * 3, max(1, h - (y0 + band_h) - 1)))
        out.paste(patch, (0, min(h - band_h, y0 + dy)))
    return out


def _tamper_signature_transplant(img: Image.Image, donor: Image.Image, rng: random.Random, severity: float) -> Image.Image:
    """Simulate signature transplant by pasting a bottom-region patch from another document."""
    w, h = img.size
    pw = max(60, int(w * (0.25 + 0.10 * _clamp01(severity))))
    ph = max(22, int(h * (0.06 + 0.04 * _clamp01(severity))))

    donor_r = donor.resize((w, h), resample=Image.BILINEAR)
    # Take donor patch near bottom.
    sx0 = rng.randint(0, max(0, w - pw))
    sy0 = rng.randint(int(h * 0.70), max(int(h * 0.70), h - ph))
    patch = donor_r.crop((sx0, sy0, sx0 + pw, sy0 + ph))

    out = img.copy()
    tx0 = rng.randint(0, max(0, w - pw))
    ty0 = rng.randint(int(h * 0.70), max(int(h * 0.70), h - ph))
    out.paste(patch, (tx0, ty0))
    return out


def _tamper_qr_replace(img: Image.Image, rng: random.Random, severity: float) -> Image.Image:
    """Simulate QR replacement by overlaying a QR-like block pattern."""
    w, h = img.size
    size = max(40, int(min(w, h) * (0.10 + 0.10 * _clamp01(severity))))
    x0 = rng.randint(0, max(0, w - size))
    y0 = rng.randint(int(h * 0.25), max(int(h * 0.25), h - size))

    # Create a simple grid pattern.
    grid_n = 21
    cell = max(1, size // grid_n)
    patch = Image.new("RGB", (grid_n * cell, grid_n * cell), (255, 255, 255))
    d = ImageDraw.Draw(patch)
    for gy in range(grid_n):
        for gx in range(grid_n):
            if rng.random() < (0.42 + 0.10 * _clamp01(severity)):
                d.rectangle([gx * cell, gy * cell, (gx + 1) * cell, (gy + 1) * cell], fill=(0, 0, 0))

    out = img.copy()
    out.paste(patch, (x0, y0))
    return out


_TAMPER_FUNCS = {
    "occlude": _tamper_occlude,
    "copy_move": _tamper_copy_move,
    "stamp": _tamper_stamp,
    "recompress": _tamper_recompress,
    "amount_edit": _tamper_amount_edit,
    "lineitem_add_remove": _tamper_lineitem_add_remove,
    "qr_replace": _tamper_qr_replace,
}


def _apply_tamper(
    img: Image.Image,
    donor_pool: List[Image.Image],
    rng: random.Random,
    severity: float,
    ops: Tuple[str, ...],
    *,
    multi_step: bool,
    max_steps: int,
) -> Tuple[Image.Image, str]:
    """Apply one or more tamper operations and return (tampered_image, tamper_op_label)."""
    available = [op for op in ops if op in _TAMPER_FUNCS or op in {"vendor_swap", "signature_transplant"}]
    if not available:
        raise SystemExit(
            f"No valid tamper ops selected. Valid: {sorted(_TAMPER_FUNCS.keys()) + ['vendor_swap','signature_transplant']}"
        )

    steps = 1
    if multi_step:
        # More steps as severity increases.
        p2 = _clamp01(0.15 + 0.70 * _clamp01(severity))
        if rng.random() < p2:
            steps += 1
        if severity > 0.80 and rng.random() < 0.35:
            steps += 1
    steps = max(1, min(int(max_steps), steps))

    out = img
    chosen: List[str] = []

    for _ in range(steps):
        op = rng.choice(available)
        donor = rng.choice(donor_pool) if donor_pool else img

        if op == "vendor_swap":
            out = _tamper_vendor_swap(out, donor, rng, severity)
        elif op == "signature_transplant":
            out = _tamper_signature_transplant(out, donor, rng, severity)
        else:
            out = _TAMPER_FUNCS[op](out, rng, severity)
        chosen.append(op)

    # Common post-process: recompress to simulate real-world sharing/printing artifacts.
    if "recompress" not in chosen and rng.random() < _clamp01(0.15 + 0.50 * _clamp01(severity)):
        out = _tamper_recompress(out, rng, severity)
        chosen.append("recompress")

    return out, "+".join(chosen)


def _train_val_threshold(
    y_true: np.ndarray,
    prob: np.ndarray,
    *,
    metric: str,
    fp_cost: float,
    fn_cost: float,
) -> float:
    metric = metric.lower().strip()
    if metric not in {"f1", "cost"}:
        raise SystemExit("--threshold-metric must be one of: f1, cost")

    best_t = 0.5
    best_score = -1e18

    # Coarse grid is fine; you can refine later.
    for t in np.linspace(0.01, 0.99, 99):
        pred = (prob >= t).astype(int)
        if metric == "f1":
            score = f1_score(y_true, pred, zero_division=0)
        else:
            tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
            score = -(fp_cost * fp + fn_cost * fn)
        if score > best_score:
            best_score = score
            best_t = float(t)
    return best_t


def _ece(y_true: np.ndarray, prob: np.ndarray, *, n_bins: int = 10) -> Tuple[float, List[Dict[str, float]]]:
    """Expected Calibration Error + reliability data."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    rel: List[Dict[str, float]] = []
    n = len(y_true)
    for i in range(n_bins):
        lo, hi = float(bins[i]), float(bins[i + 1])
        mask = (prob >= lo) & (prob < hi) if i < n_bins - 1 else (prob >= lo) & (prob <= hi)
        cnt = int(mask.sum())
        if cnt == 0:
            rel.append({"bin_lo": lo, "bin_hi": hi, "count": 0.0, "mean_prob": float("nan"), "frac_pos": float("nan")})
            continue
        mean_p = float(prob[mask].mean())
        frac_pos = float(y_true[mask].mean())
        ece += (cnt / max(1, n)) * abs(frac_pos - mean_p)
        rel.append({"bin_lo": lo, "bin_hi": hi, "count": float(cnt), "mean_prob": mean_p, "frac_pos": frac_pos})
    return float(ece), rel


def _metrics_dict(
    *,
    y_true: np.ndarray,
    prob: np.ndarray,
    threshold: float,
    fp_cost: float,
    fn_cost: float,
) -> Dict[str, object]:
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    ece, rel = _ece(y_true, prob, n_bins=10)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, prob)) if len(np.unique(y_true)) > 1 else float("nan"),
        "pr_auc": float(average_precision_score(y_true, prob)) if len(np.unique(y_true)) > 1 else float("nan"),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "expected_cost": float(fp_cost * fp + fn_cost * fn),
        "ece": float(ece),
        "reliability": rel,
    }


def _bootstrap_ci(values: List[float], *, rng: np.random.Generator, n_boot: int = 5000) -> Dict[str, float]:
    if not values:
        return {"mean": float("nan"), "std": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    arr = np.array(values, dtype=float)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    if len(arr) == 1:
        return {"mean": mean, "std": std, "ci_low": mean, "ci_high": mean}
    boots = []
    for _ in range(n_boot):
        sample = rng.choice(arr, size=len(arr), replace=True)
        boots.append(float(sample.mean()))
    ci_low, ci_high = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
    return {"mean": mean, "std": std, "ci_low": ci_low, "ci_high": ci_high}


def _write_labels_csv(path: Path, rows: Iterable[Tuple[str, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "label"])
        for fname, label in rows:
            w.writerow([fname, int(label)])


def _run(cmd: List[str], cwd: Path) -> None:
    env = os.environ.copy()
    # Ensure imports like `from src...` work when running scripts from subprocess.
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(cwd) + (os.pathsep + existing if existing else "")

    proc = subprocess.run(cmd, cwd=str(cwd), check=False, env=env)
    if proc.returncode != 0:
        raise SystemExit(f"Command failed ({proc.returncode}): {' '.join(cmd)}")


def main() -> None:
    p = argparse.ArgumentParser(description="CORD public benchmark + synthetic fraud labels (realistic imbalance).")
    p.add_argument("--out", default="data/public_benchmark/cord_fraud", help="Output directory")
    p.add_argument("--n", type=int, default=1000, help="Total samples to export")
    p.add_argument("--fraud-rate", type=float, default=0.10, help="Fraction of fraud samples (realistic imbalance)")
    p.add_argument(
        "--seeds",
        default=None,
        help="Optional comma-separated list of seeds for multi-run aggregation (e.g. '7,11,13,17,19').",
    )
    p.add_argument("--seed", type=int, default=7, help="Seed (used if --seeds not provided)")
    p.add_argument("--test-size", type=float, default=0.20)
    p.add_argument("--val-size", type=float, default=0.10, help="Validation fraction (from remaining after test)")
    p.add_argument("--severity", type=float, default=0.55, help="Tamper severity in [0,1]")
    p.add_argument(
        "--tamper-ops",
        default="amount_edit,vendor_swap,lineitem_add_remove,signature_transplant,qr_replace,copy_move,occlude,recompress",
        help=(
            "Comma-separated tamper ops. Valid: "
            + ",".join(sorted(list(_TAMPER_FUNCS.keys()) + ["vendor_swap", "signature_transplant"]))
        ),
    )
    p.add_argument(
        "--multi-step",
        action="store_true",
        help="Enable multi-step tampering (e.g., vendor_swap+recompress).",
    )
    p.add_argument("--max-steps", type=int, default=3)
    p.add_argument(
        "--threshold-metric",
        default="f1",
        choices=["f1", "cost"],
        help="How to choose decision threshold on validation split.",
    )
    p.add_argument("--fp-cost", type=float, default=1.0, help="False-positive cost (used if threshold-metric=cost)")
    p.add_argument("--fn-cost", type=float, default=10.0, help="False-negative cost (used if threshold-metric=cost)")
    p.add_argument(
        "--use-donut",
        action="store_true",
        help="Include Donut-derived features (slower; downloads model on first run).",
    )
    p.add_argument(
        "--yolo-weights",
        default=None,
        help="Optional YOLO weights path for layout detection.",
    )
    args = p.parse_args()

    ds, splits = _load_cord_dataset()
    if args.n <= 0:
        raise SystemExit("--n must be > 0")

    seeds: List[int]
    if args.seeds:
        seeds = [int(s.strip()) for s in str(args.seeds).split(",") if s.strip()]
        if not seeds:
            raise SystemExit("--seeds was provided but parsed to empty")
    else:
        seeds = [int(args.seed)]

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    all_run_metrics: List[Dict[str, object]] = []
    for seed in seeds:
        run_dir = out_root if len(seeds) == 1 else (out_root / f"seed_{seed}")
        run_dir.mkdir(parents=True, exist_ok=True)

        rng = random.Random(seed)

        sampled = _sample_cord_images(ds, splits, rng=rng, n=int(args.n))

        fraud_rate = _clamp01(float(args.fraud_rate))
        n_fraud_target = max(1, int(round(args.n * fraud_rate)))

        # Split indices by document to avoid leakage.
        idx = list(range(args.n))
        rng.shuffle(idx)
        n_test = max(1, int(round(args.n * float(args.test_size))))
        remaining = idx[n_test:]
        n_val = max(1, int(round(len(remaining) * float(args.val_size))))
        test_idx = set(idx[:n_test])
        val_idx = set(remaining[:n_val])

        train_dir = run_dir / "invoices_train"
        val_dir = run_dir / "invoices_val"
        test_dir = run_dir / "invoices_test"
        for d in [train_dir, val_dir, test_dir]:
            d.mkdir(parents=True, exist_ok=True)

        ops = tuple([s.strip() for s in str(args.tamper_ops).split(",") if s.strip()])
        fraud_indices = set(rng.sample(range(args.n), k=min(n_fraud_target, args.n)))

        labels_train: List[Tuple[str, int]] = []
        labels_val: List[Tuple[str, int]] = []
        labels_test: List[Tuple[str, int]] = []
        tamper_meta: List[Dict[str, object]] = []

        for i, img in enumerate(sampled):
            is_fraud = i in fraud_indices
            if i in test_idx:
                split = "test"
                target_dir = test_dir
            elif i in val_idx:
                split = "val"
                target_dir = val_dir
            else:
                split = "train"
                target_dir = train_dir

            fname = f"cord_{i:06d}.png"
            out_img = img
            tamper_op = "clean"
            if is_fraud:
                out_img, tamper_op = _apply_tamper(
                    img,
                    donor_pool=sampled,
                    rng=rng,
                    severity=float(args.severity),
                    ops=ops,
                    multi_step=bool(args.multi_step),
                    max_steps=int(args.max_steps),
                )

            out_path = target_dir / fname
            out_img.save(out_path)

            row = (fname, 1 if is_fraud else 0)
            if split == "test":
                labels_test.append(row)
            elif split == "val":
                labels_val.append(row)
            else:
                labels_train.append(row)

            tamper_meta.append({"file": fname, "split": split, "label": int(row[1]), "tamper_op": tamper_op})

        labels_train_path = run_dir / "labels_train.csv"
        labels_val_path = run_dir / "labels_val.csv"
        labels_test_path = run_dir / "labels_test.csv"
        _write_labels_csv(labels_train_path, labels_train)
        _write_labels_csv(labels_val_path, labels_val)
        _write_labels_csv(labels_test_path, labels_test)

        cfg = BenchmarkConfig(
            dataset="naver-clova-ix/cord-v2",
            dataset_config=None,
            n=int(args.n),
            fraud_rate=float(fraud_rate),
            seed=int(seed),
            test_size=float(args.test_size),
            val_size=float(args.val_size),
            severity=float(_clamp01(float(args.severity))),
            tamper_ops=ops,
            multi_step=bool(args.multi_step),
            max_steps=int(args.max_steps),
            skip_donut=not bool(args.use_donut),
            threshold_metric=str(args.threshold_metric),
            fp_cost=float(args.fp_cost),
            fn_cost=float(args.fn_cost),
        )
        (run_dir / "generation_config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
        (run_dir / "tamper_metadata.json").write_text(json.dumps(tamper_meta, indent=2), encoding="utf-8")

        # Build features.
        root = Path(__file__).resolve().parents[1]
        py = sys.executable
        features_train = run_dir / "features_train.csv"
        features_val = run_dir / "features_val.csv"
        features_test = run_dir / "features_test.csv"

        build_cmd_base = [py, "scripts/build_feature_dataset_vl.py"]
        if args.yolo_weights:
            build_cmd_base = [py, "scripts/build_feature_dataset_vl.py", "--yolo-weights", str(args.yolo_weights)]

        def build_split(inv_dir: Path, labels_path: Path, out_csv: Path) -> None:
            cmd = build_cmd_base + ["--invoices", str(inv_dir), "--labels", str(labels_path), "--out", str(out_csv)]
            if not args.use_donut:
                cmd.append("--skip-donut")
            else:
                os.environ.pop("DONUT_DISABLE", None)
            _run(cmd, cwd=root)

        print(f"[seed {seed}] Building features (train/val/test)...")
        build_split(train_dir, labels_train_path, features_train)
        build_split(val_dir, labels_val_path, features_val)
        build_split(test_dir, labels_test_path, features_test)

        # Train XGBoost with early stopping on val.
        import pandas as pd

        df_train = pd.read_csv(features_train)
        df_val = pd.read_csv(features_val)
        df_test = pd.read_csv(features_test)

        def feature_cols(df) -> List[str]:
            numeric = df.select_dtypes(include=["number"]).columns.tolist()
            return [c for c in numeric if c != "label"]

        feats = feature_cols(df_train)
        if not feats:
            raise SystemExit("No numeric features found in features_train.csv")

        def to_xy(df) -> Tuple[np.ndarray, np.ndarray]:
            X = df[feats].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
            y = df["label"].to_numpy(dtype=int)
            return X, y

        Xtr, ytr = to_xy(df_train)
        Xva, yva = to_xy(df_val)
        Xte, yte = to_xy(df_test)

        dtrain = xgb.DMatrix(Xtr, label=ytr)
        dval = xgb.DMatrix(Xva, label=yva)
        dtest = xgb.DMatrix(Xte, label=yte)

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
            num_boost_round=400,
            evals=[(dtrain, "train"), (dval, "val")],
            verbose_eval=False,
            early_stopping_rounds=40,
        )

        model_path = run_dir / "xgb_model.json"
        booster.save_model(model_path)

        prob_val = booster.predict(dval)
        threshold = _train_val_threshold(
            yva,
            prob_val,
            metric=str(args.threshold_metric),
            fp_cost=float(args.fp_cost),
            fn_cost=float(args.fn_cost),
        )

        prob_test = booster.predict(dtest)
        overall = _metrics_dict(y_true=yte, prob=prob_test, threshold=threshold, fp_cost=float(args.fp_cost), fn_cost=float(args.fn_cost))

        # Per-tamper-op breakdown (test only).
        file_to_op: Dict[str, str] = {m["file"]: str(m["tamper_op"]) for m in tamper_meta if m["split"] == "test"}
        # Features CSV has a `file` column; use it to align.
        if "file" in df_test.columns:
            ops_series = df_test["file"].map(lambda f: file_to_op.get(str(f), "unknown"))
        else:
            ops_series = None

        per_type: Dict[str, Dict[str, object]] = {}
        if ops_series is not None:
            for op_name in sorted(set(map(str, ops_series.tolist()))):
                mask = ops_series == op_name
                if int(mask.sum()) < 2:
                    continue
                per_type[op_name] = _metrics_dict(
                    y_true=yte[mask.to_numpy()],
                    prob=prob_test[mask.to_numpy()],
                    threshold=threshold,
                    fp_cost=float(args.fp_cost),
                    fn_cost=float(args.fn_cost),
                )

        metrics = {
            "seed": seed,
            "n": int(args.n),
            "fraud_rate": float(fraud_rate),
            "severity": float(_clamp01(float(args.severity))),
            "feature_columns": feats,
            "threshold_metric": str(args.threshold_metric),
            "chosen_threshold": float(threshold),
            "overall_test": overall,
            "per_tamper_op_test": per_type,
        }

        (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        # Human-readable summary.
        summary = (
            f"seed={seed}\n"
            f"threshold={threshold:.3f} (tuned on val via {args.threshold_metric})\n"
            f"test_accuracy={overall['accuracy']:.4f}\n"
            f"test_f1={overall['f1']:.4f}\n"
            f"test_roc_auc={overall['roc_auc']:.4f}\n"
            f"test_pr_auc={overall['pr_auc']:.4f}\n"
            f"test_ece={overall['ece']:.4f}\n"
            f"test_expected_cost={overall['expected_cost']:.2f} (fp_cost={args.fp_cost}, fn_cost={args.fn_cost})\n"
            f"confusion={overall['confusion']}\n"
        )
        (run_dir / "metrics.txt").write_text(summary, encoding="utf-8")
        print(summary.strip())
        print(f"[seed {seed}] Wrote outputs to: {run_dir}")

        all_run_metrics.append(metrics)

    # Aggregate across seeds.
    if len(seeds) > 1:
        rng_np = np.random.default_rng(1337)
        def collect(metric_key: str) -> List[float]:
            vals: List[float] = []
            for m in all_run_metrics:
                v = m["overall_test"].get(metric_key)
                if v is None:
                    continue
                try:
                    fv = float(v)
                except Exception:
                    continue
                if fv == fv:
                    vals.append(fv)
            return vals

        agg = {
            "n_seeds": len(seeds),
            "seeds": seeds,
            "roc_auc": _bootstrap_ci(collect("roc_auc"), rng=rng_np),
            "pr_auc": _bootstrap_ci(collect("pr_auc"), rng=rng_np),
            "f1": _bootstrap_ci(collect("f1"), rng=rng_np),
            "accuracy": _bootstrap_ci(collect("accuracy"), rng=rng_np),
            "ece": _bootstrap_ci(collect("ece"), rng=rng_np),
        }
        (out_root / "summary.json").write_text(json.dumps(agg, indent=2), encoding="utf-8")
        print(f"Wrote multi-seed summary to: {out_root / 'summary.json'}")


if __name__ == "__main__":
    main()
