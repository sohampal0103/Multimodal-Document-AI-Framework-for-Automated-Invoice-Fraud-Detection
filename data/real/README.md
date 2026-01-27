# Real datasets (templates)

This folder documents the expected dataset formats for training/fine-tuning.

## 1) Donut fine-tuning dataset (JSONL)

Create JSONL files:

- `data/real/donut/train.jsonl`
- `data/real/donut/val.jsonl`

Each line is a JSON object:

```json
{"image": "path/to/image.png", "ground_truth": {"vendor": "...", "invoice_number": "...", "date": "YYYY-MM-DD", "total": "...", "tax": "..."}}
```

Notes:
- `image` is a path (relative to repo root or absolute).
- `ground_truth` must be a JSON object (nested objects/lists are allowed).
- Your fine-tuned model will learn to generate this JSON.

## 2) YOLOv8 layout dataset

Use standard YOLO format:

- Images: `data/yolo_layout/images/{train,val}/...`
- Labels: `data/yolo_layout/labels/{train,val}/...`

Label files are YOLO txt format (one box per line):

```
<class_id> <x_center> <y_center> <width> <height>
```

All coordinates are normalized to [0, 1].

Create `data/yolo_layout/data.yaml` with your class names. See `data/yolo_layout/data.yaml`.
