from src.preprocessing.preprocess import preprocess_image
from src.document_ai.extractor import extract_invoice_data
from src.cv_forgery.forgery import forgery_score
from src.yolo_detection.detect import detect_layout
from src.classifier.fraud_classifier import predict_fraud

import os

def run_pipeline(path: str):
    img = preprocess_image(path)

    data = extract_invoice_data(img)
    tamper_score = forgery_score(img)
    yolo_weights = os.getenv("YOLO_WEIGHTS_PATH")
    layout = detect_layout(img, weights_path=yolo_weights)

    features = {
        "tamper_score": float(tamper_score),
        "has_logo": bool(layout.get("Logo", False)),
        "has_signature": bool(layout.get("Signature", False)),
        "has_qr": bool(layout.get("QR Code", False)),
    }
    xgb_model = os.getenv("XGB_MODEL_PATH")
    prob = predict_fraud(features, model_path=xgb_model)

    anomalies = []
    if tamper_score > 0.25:
        anomalies.append("Visual tampering detected")
    if not features["has_signature"]:
        anomalies.append("Missing signature")
    if not features["has_logo"]:
        anomalies.append("Missing logo")
    return data, prob, anomalies
