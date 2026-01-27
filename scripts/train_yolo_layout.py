import argparse
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description="Train YOLOv8 for invoice layout elements.")
    p.add_argument("--data", default="data/yolo_layout/data.yaml", help="YOLO dataset YAML")
    p.add_argument("--model", default="yolov8n.pt", help="Base model")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--project", default="runs/layout")
    p.add_argument("--name", default="yolov8-layout")
    args = p.parse_args()

    try:
        from ultralytics import YOLO
    except Exception as exc:  # pragma: no cover
        raise SystemExit(f"Ultralytics not installed: {exc}")

    model = YOLO(args.model)
    results = model.train(
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        project=args.project,
        name=args.name,
    )

    # Ultralytics saves weights under runs/.../weights/best.pt
    out_dir = Path(args.project) / args.name / "weights"
    print(f"Training complete. Weights in: {out_dir}")


if __name__ == "__main__":
    main()
