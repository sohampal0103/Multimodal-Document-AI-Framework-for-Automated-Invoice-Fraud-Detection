import os

import cv2
import numpy as np


def _read_image_any(path: str):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        try:
            from pdf2image import convert_from_path
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "PDF support requires pdf2image. Install via `pip install pdf2image`."
            ) from exc

        try:
            pages = convert_from_path(path, first_page=1, last_page=1)
        except Exception as exc:
            raise RuntimeError(
                "Failed to convert PDF to image. On macOS, install Poppler: `brew install poppler`."
            ) from exc

        if not pages:
            raise ValueError("PDF contains no pages")

        # Convert PIL -> OpenCV BGR
        pil_img = pages[0].convert("RGB")
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        return img

    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not read image file: {path}")
    return img


def preprocess_image(path: str):
    """Loads an invoice from disk and returns a normalized grayscale image."""
    img_bgr = _read_image_any(path)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return gray
