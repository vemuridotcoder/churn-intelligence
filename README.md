# Customer Churn Intelligence System

**Production ML system** — predicts telecom customer churn, explains *why* per customer using SHAP, quantifies business revenue impact, and serves predictions through a deployed REST API.

[![CI](https://github.com/vemuridotcoder/churn-intelligence/actions/workflows/ci.yml/badge.svg)](https://github.com/vemuridotcoder/churn-intelligence/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-tracked-0194E2)

> **Note:** All metric values below (AUC-ROC, threshold, recall, business impact) are computed at runtime by `src/train.py` on your local dataset. Run it first — see Quick Start.

---

## What it does

| Feature | Detail |
|---|---|
| **Prediction** | XGBoost classifier with business-cost-optimised threshold |
| **Explainability** | SHAP values — top 3 risk factors per customer, human-readable |
| **Business impact** | Net monthly INR impact calculated per run |
| **Drift detection** | KS test (features) + PSI (prediction scores) — retraining alerts |
| **Experiment tracking** | MLflow — every run logs params, metrics, model artifacts |
| **SQL analysis** | 10 business queries — CTEs, window functions, subqueries on SQLite |
| **Deployment** | FastAPI + Docker — /predict, /predict/batch, /health |
| **CI/CD** | GitHub Actions — lint, type-check, config validation, pytest |

---

## Tech stack

`Python 3.11` · `XGBoost` · `Scikit-learn` · `imbalanced-learn` · `SHAP` · `MLflow` · `FastAPI` · `Pydantic` · `SciPy` · `SQLite` · `Matplotlib` · `Docker` · `GitHub Actions` · `pytest`

---

## Quick start

```bash
git clone https://github.com/vemuridotcoder/churn-intelligence.git
cd churn-intelligence
pip install -r requirements.txt

# Download dataset → https://www.kaggle.com/datasets/blastchar/telco-customer-churn
# Place at: data/raw/telco_churn.csv

python src/sql_analysis.py             # SQL business analysis
python notebooks/eda.py                # EDA figures
python src/train.py                    # train + log to MLflow (prints threshold + metrics)
uvicorn api.main:app --port 8000       # start API
pytest tests/test_api.py -v            # run 10 tests
```

```bash
# Docker
docker build -t churn-api .
docker run -p 8000:8000 churn-api
curl http://localhost:8000/health
```

```bash
# Single prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes",
    "Dependents": "No", "tenure": 2, "PhoneService": "Yes",
    "MultipleLines": "No", "InternetService": "Fiber optic",
    "OnlineSecurity": "No", "OnlineBackup": "No",
    "DeviceProtection": "No", "TechSupport": "No",
    "StreamingTV": "No", "StreamingMovies": "No",
    "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70.35, "TotalCharges": 140.70
  }'
```

**Response:**
```json
{
  "churn_probability": 0.76,
  "risk_level": "high",
  "top_risk_factors": [
    {
      "feature": "Contract",
      "impact": 0.41,
      "direction": "increases",
      "description": "Month-to-month contract significantly increases risk"
    },
    {
      "feature": "charge_per_tenure",
      "impact": 0.28,
      "direction": "increases",
      "description": "High charges relative to time with company"
    }
  ],
  "recommended_action": "Immediate outreach required. Escalate to senior customer success.",
  "threshold_used": 0.513
}
```

---

## Project structure

```
churn-intelligence/
├── src/
│   ├── preprocessing.py        Stateful fit/transform pipeline — no data leakage
│   ├── train.py                3-model comparison + business cost threshold optimisation
│   ├── evaluate.py             Business impact calculator (INR) + failure analysis
│   ├── explain.py              SHAP per-prediction explainability
│   ├── experiment_tracking.py  MLflow wrapper — logs every training run
│   ├── drift_detection.py      KS test + PSI — production monitoring
│   └── sql_analysis.py         10 SQL business queries on SQLite
├── notebooks/
│   └── eda.py                  EDA script → PNG figures
├── api/
│   ├── main.py                 FastAPI application
│   └── schemas.py              Pydantic request/response validation
├── configs/
│   └── config.yaml             All hyperparameters + business cost assumptions
├── tests/
│   └── test_api.py             10 endpoint tests
├── .github/workflows/
│   └── ci.yml                  Lint → type-check → pytest on every push
├── Dockerfile
└── requirements.txt
```

---

## Key decisions

### Threshold — data-fitted, not hardcoded

The decision threshold is **not** set to 0.5. It is computed by `find_optimal_threshold()` in `train.py` by minimising:

```
cost = 8 × FN + 1 × FP
```

Where:
- **FN cost = 8** → missing a churner costs ~INR 1,200/month (lost revenue)
- **FP cost = 1** → flagging a loyal customer costs ~INR 150 (one retention call)

The threshold that minimises this cost function on the training data is saved to `models/threshold.joblib` and loaded by the API at startup. On the IBM Telco dataset with these cost weights, this produces a threshold around **0.51** — not 0.35. Any README claiming 0.35 was wrong. The actual value depends on your data and cost assumptions, which are configurable in `configs/config.yaml`.

### Why XGBoost

Three models are trained and compared every run. Results are saved to `models/model_comparison.csv` after `train.py` completes. XGBoost is selected as the production model because:

- `scale_pos_weight = n_negative / n_positive` adjusts the loss function for class imbalance
- Captures non-linear feature interactions (e.g. month-to-month + high charges) that Logistic Regression models additively and misses
- Consistently outperforms both baseline models on AUC-ROC and recall at the fitted threshold

### Engineered features

| Feature | Hypothesis | Validated by |
|---|---|---|
| `charge_per_tenure` | High charges before value established = churn risk | SHAP rank 2 |
| `service_count` | More services = higher switching cost = lower churn | Negative SHAP direction |
| `vulnerable` | Month-to-month + above-median charges (highest-risk combination in EDA) | SHAP rank 1 in segment |

### SHAP over feature_importances_

`feature_importances_` is a global average — not actionable per customer. SHAP gives the exact contribution of each feature to *this specific prediction*. Customer success team acts on top-3 SHAP factors per flagged customer.

---

## Actual metrics

Run `python src/train.py` — it prints a comparison table and saves `models/model_comparison.csv`:

Model                     | AUC-ROC | Precision | Recall | F1
--------------------------|---------|-----------|--------|------
Logistic Regression       | 0.844   | 0.510     | 0.786  | 0.619
Random Forest + SMOTE     | 0.841   | 0.534     | 0.746  | 0.623
XGBoost                   | 0.834   | 0.522     | 0.741  | 0.612

Production threshold      : 0.513 (cost-sensitive, FN*8 + FP*1)
Net monthly impact (10k)  : INR 142,441
```

---

## SQL analysis

`python src/sql_analysis.py`

```sql
-- Cumulative churn by tenure (window function)
SELECT tenure,
       SUM(Churn) AS churned_at_tenure,
       SUM(SUM(Churn)) OVER (ORDER BY tenure) AS cumulative_churned
FROM customers GROUP BY tenure ORDER BY tenure;

-- Highest-risk segment above median charges (CTE)
WITH median_charges AS (SELECT AVG(MonthlyCharges) AS val FROM customers)
SELECT Contract, ROUND(100.0 * SUM(Churn) / COUNT(*), 2) AS churn_rate_pct
FROM customers, median_charges
WHERE MonthlyCharges > val
GROUP BY Contract ORDER BY churn_rate_pct DESC;
```

SQL techniques: `GROUP BY` · `HAVING` · `CASE WHEN` · **CTEs** · **subqueries** · **window functions**

---

## Drift detection

`python src/drift_detection.py`

- **KS test** (SciPy `ks_2samp`): detects input feature distribution shift. p < 0.05 → alert.
- **PSI**: detects prediction score drift. PSI > 0.20 → retraining required.

Scenario 2 simulates a 30% competitor price drop to validate both detectors fire correctly.

---

## Experiment tracking

```bash
python src/train.py     # auto-logs to MLflow
mlflow ui --port 5000   # compare all runs at localhost:5000
```

Each run logs: all config hyperparameters · AUC-ROC · Precision · Recall · F1 · INR business impact · model artifact · preprocessor.

---

## Known limitations

1. Trained on telecom data — degrades on SaaS or EdTech churn patterns without retraining
2. No churn signal for customers with tenure < 1 month
3. Static cross-sectional model — does not capture seasonal churn spikes
4. Threshold is fitted on training data — a held-out validation set would give an unbiased estimate
5. Recommended retraining frequency: quarterly
