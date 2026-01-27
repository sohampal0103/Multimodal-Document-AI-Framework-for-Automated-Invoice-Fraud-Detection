"""Invoice field extraction via a Vision-Language model.

This module uses Hugging Face Donut (OCR-free document understanding) by default.

Environment variables:
- `DONUT_MODEL_ID`: HF model id to use.
  Default: `naver-clova-ix/donut-base-finetuned-cord-v2`
- `HF_HOME` / `TRANSFORMERS_CACHE`: optional model cache location.

Notes:
- First run will download model weights (internet required).
- The default model is trained on receipts (CORD); for invoices you should fine-tune.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Dict

import numpy as np


def _to_pil_rgb(image: Any):
    try:
        from PIL import Image
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Pillow is required: `pip install pillow`") from exc

    if isinstance(image, Image.Image):
        return image.convert("RGB")

    if isinstance(image, np.ndarray):
        arr = image
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(f"Unsupported image array shape: {arr.shape}")
        return Image.fromarray(arr.astype(np.uint8), mode="RGB")

    raise TypeError(f"Unsupported image type: {type(image)}")


@lru_cache(maxsize=1)
def _load_donut():
    model_id = os.getenv("DONUT_MODEL_ID", "naver-clova-ix/donut-base-finetuned-cord-v2")

    try:
        from transformers import DonutProcessor, VisionEncoderDecoderModel
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Missing Transformers dependencies. Install with `pip install transformers` (and torch)."
        ) from exc

    processor = DonutProcessor.from_pretrained(model_id)
    model = VisionEncoderDecoderModel.from_pretrained(model_id)

    # Keep runtime stable on CPU unless CUDA/MPS is explicitly available.
    try:
        import torch

        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        model.to(device)
    except Exception:
        device = None

    model.eval()
    return processor, model, device


def _normalize_keys(obj: Any) -> Any:
    """Best-effort key normalization for downstream pipeline/UI."""
    if isinstance(obj, dict):
        out: Dict[str, Any] = {}
        for k, v in obj.items():
            key = str(k).strip().replace("_", " ").title()
            out[key] = _normalize_keys(v)
        return out
    if isinstance(obj, list):
        return [_normalize_keys(x) for x in obj]
    return obj


def extract_invoice_data(image: Any) -> Dict[str, Any]:
    """Extracts invoice-like key fields from an image.

    Returns a dict. For best results, fine-tune Donut on an invoice dataset and set
    `DONUT_MODEL_ID` to your fine-tuned checkpoint.
    """
    if os.getenv("DONUT_DISABLE") == "1":
        return {}

    pil = _to_pil_rgb(image)

    try:
        processor, model, device = _load_donut()
    except Exception as exc:
        # Clear message for offline environments.
        raise RuntimeError(
            "Failed to load Donut model. Ensure you have internet for first download and "
            "that `transformers`, `torch`, and `sentencepiece` are installed."
        ) from exc

    import torch

    pixel_values = processor(pil, return_tensors="pt").pixel_values
    if device is not None:
        pixel_values = pixel_values.to(device)

    # Task prompt is model-specific. CORD-style models expect this.
    task_prompt = "<s_cord-v2>"
    decoder_input_ids = processor.tokenizer(
        task_prompt, add_special_tokens=False, return_tensors="pt"
    ).input_ids
    if device is not None:
        decoder_input_ids = decoder_input_ids.to(device)

    with torch.no_grad():
        outputs = model.generate(
            pixel_values,
            decoder_input_ids=decoder_input_ids,
            max_length=768,
            early_stopping=True,
            num_beams=1,
            bad_words_ids=[[processor.tokenizer.unk_token_id]],
            return_dict_in_generate=True,
        )

    seq = outputs.sequences[0]
    decoded = processor.batch_decode([seq], skip_special_tokens=True)[0]
    decoded = decoded.replace(processor.tokenizer.eos_token, "").replace(
        processor.tokenizer.pad_token, ""
    )

    # Convert Donut tokens into structured JSON if possible.
    try:
        structured = processor.token2json(decoded)
        normalized = _normalize_keys(structured)
        if isinstance(normalized, dict) and normalized:
            return normalized
    except Exception:
        pass

    # Fallback: at least surface the raw decoded string.
    return {"Raw": decoded.strip()}
