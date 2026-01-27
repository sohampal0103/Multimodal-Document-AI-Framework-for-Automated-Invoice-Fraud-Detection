# Invoice Fraud Detection (Streamlit)

This repository contains a runnable end-to-end demo pipeline:

- **Preprocessing**: loads images (and optionally PDFs) and normalizes to grayscale
- **Document AI (stub)**: placeholder invoice field extraction
- **CV forgery score**: simple edge-based tamper score
- **Layout detection**: heuristic fallback (optionally supports custom YOLOv8 weights)
- **Fraud classifier**: deterministic baseline (optionally supports a trained XGBoost model)

> Note: The extraction/detection models are intentionally lightweight so the project runs reliably. For a publication-quality system you’ll want to plug in real trained models and evaluate on a dataset.

## Quick start (teammates)

### 1) Clone the repo

SSH (recommended):

```zsh
git clone git@github.com:sohampal0103/Multimodal-Document-AI-Framework-for-Automated-Invoice-Fraud-Detection.git
cd Multimodal-Document-AI-Framework-for-Automated-Invoice-Fraud-Detection
```

This repo includes the sample benchmark data under `data/public_benchmark/` (including the `invoices_*` image folders), so you can run the full pipeline end-to-end.

### 2) Create a virtual environment + install deps

macOS / Linux:

```zsh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

Windows (PowerShell):

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -r requirements.txt
```

### 3) Run the app

```zsh
streamlit run app.py
```

Open the URL shown in the terminal (usually `http://localhost:8501`).

## Prerequisites (macOS)

- Python **3.12** (recommended)
- (Optional, only for PDFs) Poppler
- (Recommended, for XGBoost) OpenMP runtime (`libomp`)

Install Poppler (PDF → image conversion):

```zsh
brew install poppler
```

Install OpenMP runtime (required by XGBoost on macOS):

```zsh
brew install libomp
```

## Setup (recommended: venv)

From the project root:

```zsh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

Quick sanity check:

```zsh
python -c "import cv2; print('cv2 OK', cv2.__version__)"
```

## Run

```zsh
streamlit run app.py
```

Open the URL shown in the terminal (usually `http://localhost:8501`).

## Optional: custom model assets

### YOLOv8 layout model

If you have a trained YOLOv8 model for invoice elements (logo/signature/QR), set:

```zsh
export YOLO_WEIGHTS_PATH="/absolute/path/to/weights.pt"
```

The pipeline will call `detect_layout(image, weights_path=YOLO_WEIGHTS_PATH)`.

### XGBoost classifier

If you have a trained XGBoost model saved as JSON, set:

```zsh
export XGB_MODEL_PATH="/absolute/path/to/xgb_fraud.json"
```

The pipeline will call `predict_fraud(features, model_path=XGB_MODEL_PATH)`.

## Donut extractor (real model)

`src/document_ai/extractor.py` uses HuggingFace **Donut** by default (OCR-free).

Default model:
- `naver-clova-ix/donut-base-finetuned-cord-v2` (trained on receipts, not invoices)

You can change it with:

```zsh
export DONUT_MODEL_ID="your-org/your-donut-invoice-checkpoint"
```

First run downloads weights (internet required).

## Training + evaluation (for experiments)

These scripts make it easy to train/evaluate a baseline fraud classifier from a feature CSV.

### 1) Generate a synthetic dataset (demo only)

```zsh
python scripts/make_synthetic_dataset.py --out data/synthetic/features.csv --n 2000
```

### 2) Train XGBoost and save to JSON

```zsh
python scripts/train_xgb.py --data data/synthetic/features.csv --out models/xgb_fraud.json
```

### 3) Evaluate metrics

```zsh
python scripts/evaluate.py --data data/synthetic/features.csv --model models/xgb_fraud.json
```

### Expected dataset schema

Your real dataset CSV should include:
- `tamper_score` (float)
- `has_signature` (0/1)
- `has_logo` (0/1)
- `has_qr` (0/1)
- `label` (0 = genuine, 1 = fraud)

## Research workflow (fine-tuning + detection)

### A) Fine-tune Donut on invoices

1) Prepare JSONL files as documented in `data/real/README.md`:
- `data/real/donut/train.jsonl`
- `data/real/donut/val.jsonl`

2) Install training dependencies:

```zsh
pip install -r requirements-train.txt
```

3) Fine-tune:

```zsh
python scripts/finetune_donut.py \
	--train data/real/donut/train.jsonl \
	--val data/real/donut/val.jsonl \
	--out models/donut-invoice
```

4) Run app using your fine-tuned model:

```zsh
export DONUT_MODEL_ID="models/donut-invoice"
streamlit run app.py
```

If you want to skip Donut (offline/lightweight):

```zsh
export DONUT_DISABLE=1
```

### B) Train YOLOv8 for layout detection

1) Prepare a YOLO dataset (see `data/real/README.md`) and `data/yolo_layout/data.yaml`.

2) Train:

```zsh
python scripts/train_yolo_layout.py --data data/yolo_layout/data.yaml
```

3) Use trained weights in the app:

```zsh
export YOLO_WEIGHTS_PATH="runs/layout/yolov8-layout/weights/best.pt"
streamlit run app.py
```

### C) Build features from labeled invoices + run ablations/robustness

1) Create a labels CSV: `filename,label` and a folder of invoices.

```zsh
python scripts/build_feature_dataset.py \
	--invoices /path/to/invoices \
	--labels /path/to/labels.csv \
	--out data/real/features.csv \
	--yolo-weights "$YOLO_WEIGHTS_PATH"
```

## Best workflow for Option A (files + fraud labels)

With **Option A** you only have a binary fraud label per invoice (no per-field ground truth), so the most practical approach is:

1) Use Donut **as a feature generator** (weak supervision) + CV/layout features
2) Train an XGBoost fraud classifier on those features

### 1) Build VL+CV feature CSV

```zsh
python scripts/build_feature_dataset_vl.py \
	--invoices /path/to/invoices \
	--labels /path/to/labels.csv \
	--out data/real/features_vl.csv \
	--yolo-weights "$YOLO_WEIGHTS_PATH"
```

If you want a fast/offline run, skip Donut:

```zsh
python scripts/build_feature_dataset_vl.py --invoices /path/to/invoices --labels /path/to/labels.csv --out data/real/features_vl.csv --skip-donut
```

### 2) Train + evaluate on the resulting features

By default, training/evaluation uses **all numeric columns except `label`**.

```zsh
python scripts/train_xgb.py --data data/real/features_vl.csv --out models/xgb_fraud.json
python scripts/evaluate.py --data data/real/features_vl.csv --model models/xgb_fraud.json
```

You can also specify exact feature columns:

```zsh
python scripts/train_xgb.py --data data/real/features_vl.csv --out models/xgb_fraud.json --features tamper_score,has_signature,has_logo,has_qr,vl_has_total,vl_total_value
```

2) Ablation table:

```zsh
python scripts/ablation_eval.py --data data/real/features.csv
```

3) Robustness via feature perturbations:

```zsh
python scripts/robustness_eval.py --data data/real/features.csv --noise-std 0.05 --flip-p 0.05
```

## Public benchmark (paper-friendly): CORD + synthetic fraud (realistic imbalance)

Public datasets rarely include **fraud / not-fraud** labels. A simple, defensible setup is:

- Use a fully public document dataset (CORD)
- Define "fraud" as **controlled tampering operations** (occlusion, copy-move, stamp overlay, recompression)
- Train/evaluate on clean vs tampered documents with a realistic fraud rate (default: **10%**)

1) Install training deps (for HuggingFace `datasets`):

```zsh
pip install -r requirements-train.txt
```

2) Run the benchmark end-to-end (downloads CORD on first run):

```zsh
python scripts/run_public_benchmark_cord.py --n 1000 --fraud-rate 0.10 --out data/public_benchmark/cord10
```

This writes:
- `data/public_benchmark/cord10/invoices_train/` + `labels_train.csv`
- `data/public_benchmark/cord10/invoices_test/` + `labels_test.csv`
- `data/public_benchmark/cord10/features_train.csv` + `features_test.csv`
- `data/public_benchmark/cord10/xgb_model.json`
- `data/public_benchmark/cord10/metrics.txt`

By default it uses `--skip-donut` (faster, no model download). To include Donut-derived features:

```zsh
python scripts/run_public_benchmark_cord.py --n 1000 --fraud-rate 0.10 --out data/public_benchmark/cord10 --use-donut
```

### Benchmark options (more realistic + paper-grade metrics)

- Invoice-like tamper ops (taxonomy is recorded per sample in `tamper_metadata.json`):

```zsh
python scripts/run_public_benchmark_cord.py \
	--n 2000 --fraud-rate 0.10 --severity 0.6 --multi-step \
	--tamper-ops amount_edit,vendor_swap,lineitem_add_remove,signature_transplant,qr_replace,copy_move,occlude,recompress \
	--out data/public_benchmark/cord10_real
```

- Validation split + threshold tuning:
	- The script creates `invoices_train/`, `invoices_val/`, `invoices_test/`
	- It tunes the decision threshold on **val** and reports **ROC-AUC, PR-AUC, ECE** on test

- Cost-sensitive threshold selection (good for fraud imbalance):

```zsh
python scripts/run_public_benchmark_cord.py --n 2000 --fraud-rate 0.10 \
	--threshold-metric cost --fp-cost 1 --fn-cost 10 \
	--out data/public_benchmark/cord10_cost
```

- Cross-seed stability (writes `summary.json` with bootstrap CIs):

```zsh
python scripts/run_public_benchmark_cord.py --n 2000 --fraud-rate 0.10 \
	--seeds 7,11,13,17,19 --out data/public_benchmark/cord10_multiseed
```

## Batch mode (score a folder → report CSV)

Run the fraud pipeline on a directory and export a report:

```zsh
python scripts/batch_score_invoices.py \
	--invoices /path/to/invoices \
	--out reports/scores.csv \
	--threshold 0.5 \
	--skip-donut
```

To use a trained XGBoost model, either set `XGB_MODEL_PATH` or pass `--model`.