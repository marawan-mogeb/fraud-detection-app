# 🛡️ FraudGuard — Real-Time Fraud Detection API

A production-ready Flask web application that serves the deep-learning fraud detection model trained in the Kaggle notebook.  
Runs on CPU or GPU, exposes a clean REST API, and ships with an interactive demo UI.

---

## Architecture

```
fraud-detection-app/
├── app.py                  ← Flask routes (predict, health, model-info)
├── model/
│   ├── __init__.py
│   ├── architecture.py     ← FraudDetectionNet + ResidualBlock (PyTorch)
│   ├── loader.py           ← Loads checkpoint, returns model + metadata
│   └── predictor.py        ← Feature engineering + inference pipeline
├── static/
│   ├── css/
│   │   └── style.css       ← All application styles
│   └── js/
│       └── app.js          ← Client-side logic (form, predictions, UI)
├── templates/
│   └── index.html          ← Interactive demo UI
├── requirements.txt
└── README.md
```

---

## Setup & Reproducibility

### 1 — Prerequisites

| Requirement | Version |
|---|---|
| Python | ≥ 3.10 |
| pip | ≥ 23 |
| (optional) CUDA | ≥ 11.8 for GPU inference |

### 2 — Clone / copy the project

```bash
git clone <your-repo-url>
cd fraud-detection-app
```

### 3 — Install dependencies

```bash
# Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Install
pip install -r requirements.txt
```

> **GPU users:** replace the `torch` line in `requirements.txt` with the CUDA wheel:
> ```
> torch==2.3.1+cu121 --index-url https://download.pytorch.org/whl/cu121
> ```

### 4 — Add the model checkpoint

Copy `fraud_dl_model.pt` (saved by the training notebook) into the project root:

```bash
cp /path/to/fraud_dl_model.pt ./fraud_dl_model.pt
```

The checkpoint must contain these keys (produced by the notebook's Section 9.11):

```python
{
    'model_state_dict' : ...,
    'model_config'     : {'input_dim': 16, 'hidden_dim': 256, 'dropout': 0.3},
    'best_threshold'   : 0.xxx,
    'feature_names'    : [...],   # list of 16 feature names
    'scaler_mean'      : [...],
    'scaler_scale'     : [...],
    'val_pr_auc'       : 0.xxx,
}
```

### 5 — Run the app

**Development:**
```bash
python app.py
# → http://localhost:5000
```

**Production (Gunicorn):**
```bash
gunicorn -w 2 -b 0.0.0.0:5000 app:app
```

---

## API Reference

### `POST /predict`

Accepts a JSON transaction and returns a fraud prediction.

**Request:**
```json
{
    "step":           1,
    "type":           "TRANSFER",
    "amount":         181.0,
    "oldbalanceOrg":  181.0,
    "newbalanceOrig": 0.0,
    "oldbalanceDest": 0.0,
    "newbalanceDest": 0.0,
    "isFlaggedFraud": 0,
    "nameDest":       "C1234567890"
}
```

**Response:**
```json
{
    "fraud_probability":     0.947821,
    "fraud_probability_pct": 94.78,
    "is_fraud":              true,
    "confidence":            "High",
    "risk_level":            "CRITICAL",
    "threshold":             0.312,
    "features": {
        "errorBalanceOrig":   0.0,
        "orig_balance_zeroed": 1.0,
        "is_risky_type":      1.0,
        ...
    }
}
```

**cURL example:**
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "step": 1, "type": "TRANSFER", "amount": 181.0,
    "oldbalanceOrg": 181.0, "newbalanceOrig": 0.0,
    "oldbalanceDest": 0.0,  "newbalanceDest": 0.0,
    "isFlaggedFraud": 0
  }'
```

### `GET /health`
```json
{ "status": "ok", "model_loaded": true, "device": "cuda:0" }
```

### `GET /model-info`
Returns threshold, PR-AUC, and feature names from the checkpoint.

---

## Feature Engineering

The predictor automatically replicates the notebook's GPU feature engineering on CPU:

| Feature | Formula |
|---|---|
| `diffbalanceOrig` | `newbalanceOrig − oldbalanceOrg` |
| `diffbalanceDest` | `newbalanceDest − oldbalanceDest` |
| `errorBalanceOrig` | `\|newbalanceOrig + amount − oldbalanceOrg\|` |
| `errorBalanceDest` | `\|oldbalanceDest + amount − newbalanceDest\|` |
| `amount_ratio_orig` | `amount / (oldbalanceOrg + 1)` |
| `orig_balance_zeroed` | `1` if account drained to zero |
| `dest_is_merchant` | `1` if `nameDest` starts with `M` |
| `type_enc` | integer encoding of transaction type |
| `is_risky_type` | `1` if CASH_OUT or TRANSFER |

---

## Risk Levels

| Level | Fraud Probability |
|---|---|
| 🔴 CRITICAL | ≥ 75% |
| 🟠 HIGH     | 50 – 74% |
| 🟡 MEDIUM   | 25 – 49% |
| 🟢 LOW      | < 25% |

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `PORT` | `5000` | Port to listen on |
| `FLASK_DEBUG` | `false` | Enable debug mode |

---

## Notes

- The model checkpoint is **not** included in the repository (binary artifact, ~20 MB).  
  Generate it by running Section 9.11 of the training notebook on Kaggle.
- When the checkpoint is absent the app starts in **Demo Mode** — the UI and API endpoints remain accessible but return a `503` on inference.
- Accuracy ≠ reliability for fraud detection. Focus on **PR-AUC** and **Recall**.
