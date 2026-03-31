"""
test_api.py — API endpoint tests
==================================
Run: pytest tests/test_api.py -v

Tests verify:
1. Health endpoint returns correct structure
2. Single prediction returns valid probability and risk level
3. Batch prediction aggregates correctly
4. Invalid input is rejected with 422 (not 500)
5. Edge cases: new customer (tenure=0), maximum charges
"""

import pytest
from fastapi.testclient import TestClient
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

SAMPLE_CUSTOMER = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 2,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70.35,
    "TotalCharges": 140.70,
}

LOW_RISK_CUSTOMER = {
    **SAMPLE_CUSTOMER,
    "tenure": 60,
    "Contract": "Two year",
    "MonthlyCharges": 25.0,
    "TotalCharges": 1500.0,
    "OnlineSecurity": "Yes",
    "TechSupport": "Yes",
}


@pytest.fixture
def client():
    """
    Returns test client. Requires trained model artifacts.
    If artifacts missing, tests skip gracefully with a message.
    """
    try:
        from api.main import app
        return TestClient(app)
    except RuntimeError as e:
        pytest.skip(f"Model artifacts not found. Run train.py first. ({e})")


def test_health_endpoint(client):
    """Health endpoint must return 200 with required fields."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "threshold" in data
    assert data["model_loaded"] is True
    assert 0 < data["threshold"] < 1, "Threshold must be between 0 and 1"


def test_predict_returns_valid_probability(client):
    """Prediction must return probability in [0, 1] range."""
    response = client.post("/predict", json=SAMPLE_CUSTOMER)
    assert response.status_code == 200
    data = response.json()
    assert 0 <= data["churn_probability"] <= 1
    assert data["risk_level"] in ["low", "medium", "high"]
    assert len(data["top_risk_factors"]) <= 3
    assert data["recommended_action"] != ""


def test_high_risk_customer_flagged(client):
    """Month-to-month + high charges + short tenure should be high/medium risk."""
    response = client.post("/predict", json=SAMPLE_CUSTOMER)
    assert response.status_code == 200
    data = response.json()
    # This specific customer profile is empirically high-risk in Telco data
    assert data["risk_level"] in ["high", "medium"], (
        f"Expected high/medium risk for high-risk profile, got {data['risk_level']}"
    )


def test_low_risk_customer(client):
    """Long tenure + annual contract + security services = low/medium risk."""
    response = client.post("/predict", json=LOW_RISK_CUSTOMER)
    assert response.status_code == 200
    data = response.json()
    assert data["risk_level"] in ["low", "medium"]


def test_new_customer_tenure_zero(client):
    """New customers with tenure=0 must not cause errors."""
    new_customer = {**SAMPLE_CUSTOMER, "tenure": 0, "TotalCharges": 0.0}
    response = client.post("/predict", json=new_customer)
    assert response.status_code == 200


def test_invalid_input_rejected(client):
    """Negative MonthlyCharges must be rejected with 422, not 500."""
    invalid = {**SAMPLE_CUSTOMER, "MonthlyCharges": -50.0}
    response = client.post("/predict", json=invalid)
    assert response.status_code == 422, (
        "Invalid input must return 422 Unprocessable Entity, not 500"
    )


def test_missing_field_rejected(client):
    """Incomplete request must be rejected with 422."""
    incomplete = {"tenure": 5, "MonthlyCharges": 70.0}
    response = client.post("/predict", json=incomplete)
    assert response.status_code == 422


def test_batch_prediction(client):
    """Batch endpoint must return correct count and aggregates."""
    batch = {"customers": [SAMPLE_CUSTOMER, LOW_RISK_CUSTOMER]}
    response = client.post("/predict/batch", json=batch)
    assert response.status_code == 200
    data = response.json()
    assert data["batch_size"] == 2
    assert len(data["predictions"]) == 2
    total = data["total_high_risk"] + data["total_medium_risk"] + data["total_low_risk"]
    assert total == 2, "Risk counts must sum to batch size"


def test_batch_size_limit(client):
    """Batches over 100 must be rejected."""
    oversized = {"customers": [SAMPLE_CUSTOMER] * 101}
    response = client.post("/predict/batch", json=oversized)
    assert response.status_code == 422


def test_risk_factors_have_required_fields(client):
    """Each risk factor must have feature, impact, direction, description."""
    response = client.post("/predict", json=SAMPLE_CUSTOMER)
    assert response.status_code == 200
    data = response.json()
    for factor in data["top_risk_factors"]:
        assert "feature" in factor
        assert "impact" in factor
        assert "direction" in factor
        assert "description" in factor
        assert factor["direction"] in ["increases", "decreases", "unknown"]
