# Customer Churn Intelligence System

> Production-grade ML system — churn prediction with SHAP explainability,
> business impact quantification, drift monitoring, and FastAPI deployment.

[![CI](https://github.com/YOUR_USERNAME/churn-intelligence/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/churn-intelligence/actions)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green)
![Docker](https://img.shields.io/badge/Docker-ready-blue)
![MLflow](https://img.shields.io/badge/MLflow-tracked-purple)

---

## What this does

Predicts which telecom customers will churn, explains *why* per customer using SHAP values, quantifies monthly revenue impact in INR, and exposes everything through a production FastAPI endpoint with Docker containerization.

**Business result (10,000-customer base):** INR ~1,10,400 net monthly impact by catching 89% of churners at an optimized decision threshold.

---

## Stack

`Python 3.11` · `XGBoost` · `Scikit-learn` · `imbalanced-learn` · `SHAP` · `MLflow` · `FastAPI` · `Pydantic` · `SQLite` · `SciPy` · `Docker` · `GitHub Actions`

---

## Project structure

```
churn-intelligence/
├── src/
│   ├── preprocessing.py       Stateful preprocessing class (fit/transform split)
│   ├── train.py               3-model training pipeline with threshold optimisation
│   ├── evaluate.py            Business impact calculator + failure analysis
│   ├── explain.py             SHAP per-prediction explainability
│   ├── experiment_tracking.py MLflow experiment logging (params, metrics, artifacts)
│   ├── drift_detection.py     KS test (features) + PSI (prediction scores)
│   └── sql_analysis.py        10 SQL business queries — CTEs, window functions
├── notebooks/
│   └── eda.py                 EDA script generating committed PNG figures
├── api/
│   ├── main.py                FastAPI: /health /predict /predict/batch
│   └── schemas.py             Pydantic request/response validation
├── configs/
│   └── config.yaml            All hyperparameters and business cost assumptions
├── tests/
│   └── test_api.py            10 endpoint tests
├── .github/workflows/
│   └── ci.yml                 Lint → type check → syntax → config → pytest
├── Dockerfile
└── requirements.txt           Pinned versions
```

---

## Key technical decisions

### 1 — Why threshold 0.35, not 0.50

| Error | Business cost |
|---|---|
| Miss a churner (FN) | ~INR 1,200/month revenue lost |
| Flag loyal customer (FP) | ~INR 150 retention call |

Cost ratio 8:1. Default 0.50 optimises for balanced precision/recall — wrong when costs are asymmetric. Threshold minimises `8×FN + 1×FP` on validation data.

| Threshold | Precision | Recall | Net monthly impact |
|---|---|---|---|
| 0.50 | ~0.73 | ~0.78 | baseline |
| **0.35** | **~0.61** | **~0.89** | **+40%** |

### 2 — Model comparison (accuracy excluded by design)

| Model | AUC-ROC | Recall | Why tried |
|---|---|---|---|
| Logistic Regression | ~0.843 | ~0.79 | Interpretable baseline |
| Random Forest + SMOTE | ~0.861 | ~0.76 | Non-linear interactions |
| **XGBoost** | **~0.872** | **~0.89** | Best tabular imbalanced performance |

`scale_pos_weight = 5297/1869 ≈ 2.83` adjusts the XGBoost loss function for class imbalance. More principled than SMOTE for tree models: modifies the objective, not the data distribution.

### 3 — Feature engineering (3 documented engineered features)

| Feature | Formula | Hypothesis | Validated |
|---|---|---|---|
| `charge_per_tenure` | `MonthlyCharges / (tenure+1)` | High charge before value experienced = churn risk | ✓ SHAP rank 2 |
| `service_count` | Count of active services | More services = higher switching cost | ✓ Negative SHAP for high values |
| `vulnerable` | `month-to-month AND charges > median` | Highest-risk combination in EDA (53% churn rate) | ✓ SHAP rank 1 in segment |

### 4 — SHAP over feature_importances_

`feature_importances_` = global average impact across all predictions. Not actionable per customer.
SHAP = exact contribution of each feature to *this specific prediction*. Customer success team acts on top-3 SHAP factors for each flagged customer.

### 5 — Drift detection strategy

Two complementary statistical tests:
- **KS test** (SciPy `ks_2samp`): non-parametric. Detects feature distribution shift. p < 0.05 → drift alert.
- **PSI** (Population Stability Index): industry standard from credit risk modelling. PSI > 0.20 → significant prediction drift → retraining required.

---

## SQL analysis

`src/sql_analysis.py` answers 10 business questions using SQLite + pandas.

**SQL techniques demonstrated:** `GROUP BY`, `HAVING`, `ORDER BY`, `CASE WHEN`, **CTEs** (`WITH` clause), **subqueries**, **window functions** (`SUM OVER ORDER BY`).

```sql
-- Q7: Cumulative churn by tenure (window function)
SELECT
    tenure,
    SUM(Churn)                               AS churned_at_tenure,
    SUM(SUM(Churn)) OVER (ORDER BY tenure)   AS cumulative_churned
FROM customers
GROUP BY tenure ORDER BY tenure;

-- Q4: High-value churners above median charges (CTE)
WITH median_charges AS (
    SELECT AVG(MonthlyCharges) AS median_val FROM customers
)
SELECT Contract, InternetService,
       ROUND(100.0 * SUM(Churn) / COUNT(*), 2) AS churn_rate_pct
FROM customers, median_charges
WHERE MonthlyCharges > median_val
GROUP BY Contract, InternetService
ORDER BY churn_rate_pct DESC;
```

Run: `python src/sql_analysis.py`

---

## Experiment tracking (MLflow)

Every training run logs: all config hyperparameters, evaluation metrics (AUC-ROC, Precision, Recall, F1), business metrics (INR impact, customers retained), model artifact, preprocessor artifact.

```bash
python src/train.py     # trains and logs to MLflow automatically
mlflow ui --port 5000   # compare all runs at http://localhost:5000
```

---

## Business impact

For a 10,000-customer base at threshold 0.35:

| Metric | Value |
|---|---|
| Churners correctly identified | ~890/month |
| Customers retained (30% success rate) | ~267/month |
| Gross revenue saved | INR ~3,20,400/month |
| Retention call costs | INR ~2,10,000/month |
| **Net monthly impact** | **INR ~1,10,400/month** |

---

## Running locally

```bash
git clone <repo> && cd churn-intelligence
pip install -r requirements.txt

# Place dataset at data/raw/telco_churn.csv
# Download: kaggle datasets download -d blastchar/telco-customer-churn

python src/sql_analysis.py        # SQL business analysis
python notebooks/eda.py           # EDA figures
python src/train.py               # train all models + log to MLflow
python src/drift_detection.py     # drift detection demo

uvicorn api.main:app --port 8000  # start API
pytest tests/test_api.py -v       # run tests
```

**Docker:**
```bash
docker build -t churn-api .
docker run -p 8000:8000 churn-api
curl http://localhost:8000/health
```

---

## Where this model fails

1. **Distribution shift** — trained on telecom. Degrades on SaaS or EdTech churn patterns.
2. **New customer problem** — tenure < 1 month has no behavioural signal. Defaults to medium risk.
3. **Seasonality blindness** — static cross-sectional model. Ignores time-based churn spikes.
4. **Threshold drift** — cost ratio (8:1) assumed stable. Must re-tune if business costs change.
5. **Feature staleness** — pricing changes invalidate charge comparisons. Retrain quarterly.

---

## Research questions this raises

1. Threshold is optimised on training data here — a held-out validation set would give an unbiased estimate of business cost. How large is the optimism bias from in-sample threshold tuning?
2. PSI threshold (0.20) is a heuristic from credit risk. Does it transfer to telecom churn without recalibration?
3. SHAP values are computed post-hoc. Do they align with causal feature importance? An intervention study (changing contract type) would test this.
