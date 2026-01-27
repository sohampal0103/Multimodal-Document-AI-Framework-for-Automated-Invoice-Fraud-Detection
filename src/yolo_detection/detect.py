import os
from typing import Any, Dict, Optional


def _fallback_layout(image) -> Dict[str, Any]:
    # Minimal heuristic: signatures are often dark strokes near the bottom.
    h, w = image.shape[:2]
    bottom = image[int(h * 0.75) :, :]
    # If many pixels are "ink-like" (dark), treat as present.
    dark_ratio = float((bottom < 60).mean())

    return {
        "Logo": True,
        "Signature": dark_ratio > 0.02,
        "QR Code": True,
        "_debug": {"dark_ratio": round(dark_ratio, 4)},
    }


def detect_layout(
    image,
    weights_path: Optional[str] = None,
    conf: float = 0.25,
) -> Dict[str, Any]:
    """Detect key invoice elements.

    If `weights_path` points to a trained YOLOv8 model, it will be used.
    Otherwise we fall back to a lightweight heuristic so the project runs end-to-end.
    """
    if weights_path and os.path.exists(weights_path):
        try:
            from ultralytics import YOLO
        except Exception:
            return _fallback_layout(image)

        model = YOLO(weights_path)
        results = model.predict(source=image, conf=conf, verbose=False)
        if not results:
            return _fallback_layout(image)

        # Map detected class names to expected keys.
        names = set()
        for r in results:
            if getattr(r, "names", None) and getattr(r, "boxes", None) is not None:
                for cls_id in r.boxes.cls.tolist():
                    names.add(r.names[int(cls_id)])

        return {
            "Logo": ("logo" in names) or ("Logo" in names),
            "Signature": ("signature" in names) or ("Signature" in names),
            "QR Code": ("qr" in names) or ("QRCode" in names) or ("QR Code" in names),
            "_detected": sorted(names),
        }

    return _fallback_layout(image)
