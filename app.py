"""
app.py
──────
Flask application for Fraud Detection inference.

Routes
------
GET  /              → Demo UI
POST /predict       → JSON inference endpoint
POST /predict-form  → Form submission (returns JSON, called by UI)
GET  /health        → Health check
GET  /model-info    → Model metadata
"""

import os
import json
import logging
from flask import Flask, request, jsonify, render_template

from model.loader    import load_model
from model.predictor import predict

# ─────────────────────────────────────────────────────────────────────────────
# App setup
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Load model once at startup
# ─────────────────────────────────────────────────────────────────────────────

logger.info("Loading fraud detection model...")
try:
    MODEL, METADATA = load_model()
    logger.info(
        f"Model loaded ✅  |  threshold={METADATA['threshold']:.4f}"
        f"  |  features={len(METADATA['feature_names'])}"
    )
    MODEL_LOADED = True
except FileNotFoundError as e:
    logger.warning(f"Model checkpoint not found — running in DEMO mode.\n{e}")
    MODEL        = None
    METADATA     = None
    MODEL_LOADED = False


# ─────────────────────────────────────────────────────────────────────────────
# Demo pre-built test cases (shown in the UI)
# ─────────────────────────────────────────────────────────────────────────────

SAMPLE_CASES = [
    {
        "label":    "🔴 High-Risk Transfer",
        "scenario": "Large transfer draining account to zero",
        "data": {
            "step": 1, "type": "TRANSFER", "amount": 181.0,
            "oldbalanceOrg": 181.0, "newbalanceOrig": 0.0,
            "oldbalanceDest": 0.0,  "newbalanceDest": 0.0,
            "isFlaggedFraud": 0,    "nameDest": "C1234567890",
        },
    },
    {
        "label":    "🔴 Suspicious Cash-Out",
        "scenario": "Full account drain via CASH_OUT",
        "data": {
            "step": 12, "type": "CASH_OUT", "amount": 229133.94,
            "oldbalanceOrg": 229133.94, "newbalanceOrig": 0.0,
            "oldbalanceDest": 8990000.0, "newbalanceDest": 9219133.94,
            "isFlaggedFraud": 0,         "nameDest": "C9876543210",
        },
    },
    {
        "label":    "🟢 Normal Payment",
        "scenario": "Routine bill payment, partial balance used",
        "data": {
            "step": 200, "type": "PAYMENT", "amount": 1500.0,
            "oldbalanceOrg": 50000.0, "newbalanceOrig": 48500.0,
            "oldbalanceDest": 0.0,    "newbalanceDest": 0.0,
            "isFlaggedFraud": 0,      "nameDest": "M123456789",
        },
    },
    {
        "label":    "🟢 Normal Cash-In",
        "scenario": "Salary deposit",
        "data": {
            "step": 50, "type": "CASH_IN", "amount": 3200.0,
            "oldbalanceOrg": 1000.0, "newbalanceOrig": 4200.0,
            "oldbalanceDest": 0.0,   "newbalanceDest": 0.0,
            "isFlaggedFraud": 0,     "nameDest": "C5551234567",
        },
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the demo UI."""
    return render_template(
        "index.html",
        sample_cases=SAMPLE_CASES,
        model_loaded=MODEL_LOADED,
        threshold=METADATA["threshold"] if METADATA else "N/A",
        pr_auc=METADATA["val_pr_auc"]   if METADATA else "N/A",
        feature_count=len(METADATA["feature_names"]) if METADATA else 16,
    )


@app.route("/predict", methods=["POST"])
def predict_api():
    """
    JSON API endpoint.

    Request body (JSON):
    {
        "step": 1,
        "type": "TRANSFER",
        "amount": 181.0,
        "oldbalanceOrg": 181.0,
        "newbalanceOrig": 0.0,
        "oldbalanceDest": 0.0,
        "newbalanceDest": 0.0,
        "isFlaggedFraud": 0,
        "nameDest": "C1234567890"      // optional
    }
    """
    if not MODEL_LOADED:
        return jsonify({"error": "Model not loaded. Place fraud_dl_model.pt in the app root."}), 503

    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "No JSON body received."}), 400

    required = ["step", "type", "amount",
                "oldbalanceOrg", "newbalanceOrig",
                "oldbalanceDest", "newbalanceDest"]
    missing = [f for f in required if f not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    try:
        result = predict(data, MODEL, METADATA)
        logger.info(
            f"Prediction: prob={result['fraud_probability']:.4f}"
            f"  fraud={result['is_fraud']}  risk={result['risk_level']}"
        )
        return jsonify(result), 200

    except Exception as e:
        logger.exception("Prediction error")
        return jsonify({"error": str(e)}), 500


@app.route("/predict-form", methods=["POST"])
def predict_form():
    """Handles form submission from the demo UI (also returns JSON)."""
    if not MODEL_LOADED:
        return jsonify({"error": "Model not loaded."}), 503

    form = request.form
    try:
        data = {
            "step":           int(form.get("step", 1)),
            "type":           form.get("type", "TRANSFER"),
            "amount":         float(form.get("amount", 0)),
            "oldbalanceOrg":  float(form.get("oldbalanceOrg", 0)),
            "newbalanceOrig": float(form.get("newbalanceOrig", 0)),
            "oldbalanceDest": float(form.get("oldbalanceDest", 0)),
            "newbalanceDest": float(form.get("newbalanceDest", 0)),
            "isFlaggedFraud": int(form.get("isFlaggedFraud", 0)),
            "nameDest":       form.get("nameDest", ""),
        }
        result = predict(data, MODEL, METADATA)
        return jsonify(result), 200

    except ValueError as e:
        return jsonify({"error": f"Invalid input: {e}"}), 400
    except Exception as e:
        logger.exception("Form prediction error")
        return jsonify({"error": str(e)}), 500


@app.route("/health")
def health():
    return jsonify({
        "status":       "ok",
        "model_loaded": MODEL_LOADED,
        "device":       str(next(MODEL.parameters()).device) if MODEL_LOADED else "N/A",
    })


@app.route("/model-info")
def model_info():
    if not MODEL_LOADED:
        return jsonify({"error": "Model not loaded."}), 503
    return jsonify({
        "threshold":     METADATA["threshold"],
        "val_pr_auc":    METADATA["val_pr_auc"],
        "feature_names": METADATA["feature_names"],
        "feature_count": len(METADATA["feature_names"]),
    })


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)
