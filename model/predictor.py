"""
predictor.py
────────────
End-to-end inference pipeline.

Steps
-----
1. Accept raw transaction dict (matches dataset columns)
2. Re-apply the same feature engineering as the notebook
3. Scale with the saved scaler parameters
4. Run model forward pass
5. Return fraud probability + label + explanation hints
"""

import torch
import numpy as np
from typing import Dict, Any

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transaction type encoding  (must match notebook)
TYPE_MAP = {
    "CASH_IN":  0,
    "CASH_OUT": 1,
    "DEBIT":    2,
    "PAYMENT":  3,
    "TRANSFER": 4,
}


# ─────────────────────────────────────────────────────────────────────────────
# Feature Engineering
# ─────────────────────────────────────────────────────────────────────────────

def engineer_features(raw: Dict[str, Any]) -> Dict[str, float]:
    """
    Reproduce the GPU feature-engineering from the notebook on CPU.

    Parameters
    ----------
    raw : dict with keys:
        step, type, amount,
        oldbalanceOrg, newbalanceOrig,
        oldbalanceDest, newbalanceDest,
        isFlaggedFraud

    Returns
    -------
    dict of 16 engineered features (same order as FEATURES list)
    """
    step           = float(raw["step"])
    amount         = float(raw["amount"])
    old_orig       = float(raw["oldbalanceOrg"])
    new_orig       = float(raw["newbalanceOrig"])
    old_dest       = float(raw["oldbalanceDest"])
    new_dest       = float(raw["newbalanceDest"])
    is_flagged     = float(raw.get("isFlaggedFraud", 0))
    tx_type        = raw["type"].upper()

    type_enc       = float(TYPE_MAP.get(tx_type, -1))

    diff_orig      = new_orig - old_orig
    diff_dest      = new_dest - old_dest
    err_orig       = abs(new_orig + amount - old_orig)
    err_dest       = abs(old_dest + amount - new_dest)
    amount_ratio   = amount / (old_orig + 1.0)
    zeroed         = float(new_orig == 0 and old_orig > 0)
    dest_merchant  = float(str(raw.get("nameDest", "")).startswith("M"))
    is_risky       = float(type_enc in (1.0, 4.0))   # CASH_OUT or TRANSFER

    return {
        "step":               step,
        "type_enc":           type_enc,
        "amount":             amount,
        "oldbalanceOrg":      old_orig,
        "newbalanceOrig":     new_orig,
        "oldbalanceDest":     old_dest,
        "newbalanceDest":     new_dest,
        "isFlaggedFraud":     is_flagged,
        "diffbalanceOrig":    diff_orig,
        "diffbalanceDest":    diff_dest,
        "errorBalanceOrig":   err_orig,
        "errorBalanceDest":   err_dest,
        "amount_ratio_orig":  amount_ratio,
        "orig_balance_zeroed": zeroed,
        "dest_is_merchant":   dest_merchant,
        "is_risky_type":      is_risky,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Scaler (manual — avoids sklearn dep at inference time)
# ─────────────────────────────────────────────────────────────────────────────

def scale_features(
    feature_vector: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    return (feature_vector - mean) / scale


# ─────────────────────────────────────────────────────────────────────────────
# Main predict function
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict(
    raw_transaction: Dict[str, Any],
    model,
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Run end-to-end inference on a single raw transaction dict.

    Returns
    -------
    {
        fraud_probability : float   [0, 1]
        is_fraud          : bool
        confidence        : str     'High' | 'Medium' | 'Low'
        risk_level        : str     'CRITICAL' | 'HIGH' | 'MEDIUM' | 'LOW'
        features          : dict    engineered feature values
        threshold         : float
    }
    """
    # 1. Feature engineering
    features = engineer_features(raw_transaction)

    # 2. Ordered feature vector
    feat_names = metadata["feature_names"]
    vec = np.array([features[f] for f in feat_names], dtype="float32")

    # 3. Scale
    vec_scaled = scale_features(vec, metadata["scaler_mean"], metadata["scaler_scale"])

    # 4. Forward pass
    tensor = torch.tensor(vec_scaled, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    prob   = float(torch.sigmoid(model(tensor)).cpu().item())

    # 5. Decision
    threshold = metadata["threshold"]
    is_fraud  = prob >= threshold

    # 6. Confidence & risk tier
    gap = abs(prob - threshold)
    if gap > 0.30:
        confidence = "High"
    elif gap > 0.10:
        confidence = "Medium"
    else:
        confidence = "Low"

    if prob >= 0.75:
        risk_level = "CRITICAL"
    elif prob >= 0.50:
        risk_level = "HIGH"
    elif prob >= 0.25:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    return {
        "fraud_probability": round(prob, 6),
        "fraud_probability_pct": round(prob * 100, 2),
        "is_fraud":          is_fraud,
        "confidence":        confidence,
        "risk_level":        risk_level,
        "threshold":         round(threshold, 4),
        "features":          {k: round(v, 4) for k, v in features.items()},
    }
