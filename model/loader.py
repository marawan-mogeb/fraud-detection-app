"""
loader.py
─────────
Loads the saved PyTorch checkpoint (fraud_dl_model.pt) once at
startup and exposes a ready-to-use model + metadata dict.
"""

import torch
import numpy as np
from pathlib import Path

from model.architecture import FraudDetectionNet

CHECKPOINT_PATH = Path("fraud_dl_model.pt")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(checkpoint_path: Path = CHECKPOINT_PATH):
    """
    Load checkpoint saved by the training notebook.

    Returns
    -------
    model       : FraudDetectionNet in eval mode on DEVICE
    metadata    : dict with threshold, feature_names, scaler arrays
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {checkpoint_path}.\n"
            "Run the training notebook and copy fraud_dl_model.pt here."
        )

    ckpt = torch.load(checkpoint_path, map_location=DEVICE)

    # Reconstruct model from saved config
    model = FraudDetectionNet(**ckpt["model_config"]).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    metadata = {
        "threshold":     ckpt["best_threshold"],
        "feature_names": ckpt["feature_names"],
        "scaler_mean":   np.array(ckpt["scaler_mean"], dtype="float32"),
        "scaler_scale":  np.array(ckpt["scaler_scale"], dtype="float32"),
        "val_pr_auc":    ckpt.get("val_pr_auc", None),
    }

    return model, metadata
