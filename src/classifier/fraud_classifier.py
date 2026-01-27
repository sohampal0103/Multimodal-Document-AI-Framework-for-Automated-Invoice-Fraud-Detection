import math
import os
from typing import Any, Dict, Optional


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _heuristic_score(features: Dict[str, Any]) -> float:
    tamper = float(features.get("tamper_score", 0.0))
    has_signature = bool(features.get("has_signature", True))
    has_logo = bool(features.get("has_logo", True))
    has_qr = bool(features.get("has_qr", True))

    # Simple, deterministic baseline. Replace with a trained model for publication.
    z = -1.25
    z += 5.0 * tamper
    z += 0.75 * (0.0 if has_signature else 1.0)
    z += 0.35 * (0.0 if has_logo else 1.0)
    z += 0.15 * (0.0 if has_qr else 1.0)
    return float(_sigmoid(z))


def predict_fraud(features: Dict[str, Any], model_path: Optional[str] = None) -> float:
    """Returns a fraud probability in [0, 1].

    If `model_path` points to an XGBoost JSON model, it will be used.
    Otherwise a deterministic heuristic baseline is used.
    """
    if model_path and os.path.exists(model_path):
        try:
            import xgboost as xgb
            import numpy as np
        except Exception:
            return round(_heuristic_score(features), 2)

        booster = xgb.Booster()
        booster.load_model(model_path)

        # Keep feature ordering stable.
        ordered = [
            float(features.get("tamper_score", 0.0)),
            float(bool(features.get("has_signature", True))),
            float(bool(features.get("has_logo", True))),
            float(bool(features.get("has_qr", True))),
        ]
        dmat = xgb.DMatrix(np.array([ordered], dtype=float))
        pred = float(booster.predict(dmat)[0])
        return round(max(0.0, min(1.0, pred)), 2)

    return round(_heuristic_score(features), 2)
