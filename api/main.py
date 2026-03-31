"""
main.py — Churn Intelligence API
==================================
Production-grade FastAPI application for churn prediction.

Endpoints:
  GET  /health              — liveness check
  POST /predict             — single customer prediction with SHAP explanation
  POST /predict/batch       — batch predictions (up to 100 customers)

Design decisions:
- Models loaded at startup (not per-request) — avoids 2s cold start per call
- SHAP explainer loaded separately — can be disabled for high-throughput scenarios
- Threshold loaded from disk — can be updated without redeploying the model
- Pydantic schemas validate inputs at boundary — invalid data never reaches model
"""

import os
import sys
import logging
import joblib
import numpy as np
import pandas as pd
import yaml
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from explain import ChurnExplainer
from api.schemas import (
    CustomerFeatures, PredictionResponse, RiskFactor, RiskLevel,
    BatchRequest, BatchResponse, HealthResponse
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Global model state ──────────────────────────────────────────────────────
# Loaded once at startup, shared across all requests
_model = None
_preprocessor = None
_explainer = None
_threshold = None
_feature_names = None
_config = None


def load_models():
    """Load all artifacts. Called once at application startup."""
    global _model, _preprocessor, _explainer, _threshold, _feature_names, _config

    required_paths = {
        "model": "models/xgboost_model.joblib",
        "preprocessor": "models/preprocessor.joblib",
        "threshold": "models/threshold.joblib",
        "feature_names": "models/feature_names.joblib",
    }

    missing = [p for p in required_paths.values() if not os.path.exists(p)]
    if missing:
        raise RuntimeError(
            f"Model artifacts missing. Run 'python src/train.py' first.\n"
            f"Missing: {missing}"
        )

    _model = joblib.load(required_paths["model"])
    _preprocessor = joblib.load(required_paths["preprocessor"])
    _threshold = joblib.load(required_paths["threshold"])
    _feature_names = joblib.load(required_paths["feature_names"])

    with open("configs/config.yaml") as f:
        _config = yaml.safe_load(f)

    # SHAP explainer is optional — API works without it (no explanations)
    shap_path = "models/shap_explainer.joblib"
    if os.path.exists(shap_path):
        _explainer = ChurnExplainer(shap_path, required_paths["feature_names"])
        logger.info("SHAP explainer loaded")
    else:
        logger.warning("SHAP explainer not found — predictions will have no explanations")

    logger.info(f"Models loaded. Threshold: {_threshold:.3f}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan: load models on startup, cleanup on shutdown."""
    load_models()
    yield
    logger.info("API shutting down")


# ── Application ─────────────────────────────────────────────────────────────
app = FastAPI(
    title="Churn Intelligence API",
    description=(
        "Predicts customer churn probability with SHAP-based explanations. "
        "Returns risk level, top 3 risk factors, and recommended retention action."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helper functions ────────────────────────────────────────────────────────

def get_risk_level(probability: float) -> RiskLevel:
    bands = _config["threshold"]["risk_bands"]
    if probability < bands["low"]:
        return RiskLevel.low
    elif probability < bands["medium"]:
        return RiskLevel.medium
    return RiskLevel.high


def get_recommendation(risk_level: RiskLevel) -> str:
    actions = {
        RiskLevel.low: "No action needed. Monitor next billing cycle.",
        RiskLevel.medium: "Send personalized retention offer. Assign to standard queue.",
        RiskLevel.high: "Immediate outreach required. Escalate to senior customer success.",
    }
    return actions[risk_level]


def predict_single(customer: CustomerFeatures) -> PredictionResponse:
    """
    Core prediction logic. Called by both /predict and /predict/batch.

    Steps:
    1. Convert Pydantic model to DataFrame (preprocessor expects DataFrame)
    2. Preprocess using training-fitted preprocessor (no refit)
    3. Get calibrated probability from XGBoost
    4. Apply business threshold (not 0.5)
    5. Get SHAP explanation if explainer is available
    6. Return structured response
    """
    # Step 1: DataFrame conversion
    input_df = pd.DataFrame([customer.dict()])

    # Step 2: Preprocess
    try:
        X = _preprocessor.transform(input_df)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Preprocessing failed: {str(e)}")

    # Step 3: Predict probability
    probability = float(_model.predict_proba(X)[0][1])

    # Step 4: Risk classification using business threshold
    risk_level = get_risk_level(probability)

    # Step 5: SHAP explanation
    risk_factors = []
    if _explainer is not None:
        raw_factors = _explainer.explain(X.values, top_n=3)
        risk_factors = [RiskFactor(**f) for f in raw_factors]
    else:
        # Fallback: use feature importances when SHAP unavailable
        importances = _model.feature_importances_
        top_3 = np.argsort(importances)[-3:][::-1]
        for idx in top_3:
            if idx < len(_feature_names):
                risk_factors.append(RiskFactor(
                    feature=_feature_names[idx],
                    impact=float(importances[idx]),
                    direction="increases",
                    description=f"{_feature_names[idx]} is a key predictor"
                ))

    return PredictionResponse(
        churn_probability=round(probability, 4),
        risk_level=risk_level,
        top_risk_factors=risk_factors,
        recommended_action=get_recommendation(risk_level),
        threshold_used=round(_threshold, 4),
    )


# ── Endpoints ───────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """
    Liveness check. Returns model status and configuration.
    Every production system has a health endpoint.
    Load balancers and orchestration systems (Kubernetes) ping this.
    """
    return HealthResponse(
        status="healthy",
        model_version="1.0.0",
        threshold=round(_threshold, 4),
        model_loaded=_model is not None,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(customer: CustomerFeatures):
    """
    Predict churn probability for a single customer.

    Returns:
    - churn_probability: 0-1 score
    - risk_level: low / medium / high (based on business cost threshold)
    - top_risk_factors: top 3 SHAP-derived reasons for the prediction
    - recommended_action: what customer success team should do

    Threshold is {threshold} (not 0.5) — see README for cost justification.
    """
    return predict_single(customer)


@app.post("/predict/batch", response_model=BatchResponse, tags=["Prediction"])
async def predict_batch(request: BatchRequest):
    """
    Predict churn for up to 100 customers in one request.
    More efficient than 100 individual /predict calls.
    Returns aggregate risk counts alongside individual predictions.
    """
    predictions = [predict_single(customer) for customer in request.customers]

    risk_counts = {level: 0 for level in RiskLevel}
    for pred in predictions:
        risk_counts[pred.risk_level] += 1

    return BatchResponse(
        predictions=predictions,
        total_high_risk=risk_counts[RiskLevel.high],
        total_medium_risk=risk_counts[RiskLevel.medium],
        total_low_risk=risk_counts[RiskLevel.low],
        batch_size=len(predictions),
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
