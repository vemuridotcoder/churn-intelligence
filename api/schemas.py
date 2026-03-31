"""
schemas.py — API request and response models
=============================================
Pydantic models enforce type validation at the API boundary.
Invalid requests are rejected with clear error messages before
reaching the model — not after.
"""

from pydantic import BaseModel, Field, validator
from typing import Optional
from enum import Enum


class RiskLevel(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"


class CustomerFeatures(BaseModel):
    """
    Input schema for a single churn prediction request.
    All fields match the IBM Telco dataset column names.
    """
    gender: str = Field(..., example="Female")
    SeniorCitizen: int = Field(..., ge=0, le=1, example=0)
    Partner: str = Field(..., example="Yes")
    Dependents: str = Field(..., example="No")
    tenure: int = Field(..., ge=0, le=72, example=12)
    PhoneService: str = Field(..., example="Yes")
    MultipleLines: str = Field(..., example="No")
    InternetService: str = Field(..., example="Fiber optic")
    OnlineSecurity: str = Field(..., example="No")
    OnlineBackup: str = Field(..., example="Yes")
    DeviceProtection: str = Field(..., example="No")
    TechSupport: str = Field(..., example="No")
    StreamingTV: str = Field(..., example="No")
    StreamingMovies: str = Field(..., example="No")
    Contract: str = Field(..., example="Month-to-month")
    PaperlessBilling: str = Field(..., example="Yes")
    PaymentMethod: str = Field(..., example="Electronic check")
    MonthlyCharges: float = Field(..., ge=0, example=70.35)
    TotalCharges: float = Field(..., ge=0, example=843.45)

    @validator("tenure")
    def tenure_non_negative(cls, v):
        if v < 0:
            raise ValueError("tenure cannot be negative")
        return v

    @validator("MonthlyCharges", "TotalCharges")
    def charges_non_negative(cls, v):
        if v < 0:
            raise ValueError("Charges cannot be negative")
        return v

    class Config:
        schema_extra = {
            "example": {
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
        }


class RiskFactor(BaseModel):
    feature: str
    impact: float
    direction: str
    description: str


class PredictionResponse(BaseModel):
    churn_probability: float = Field(..., ge=0, le=1)
    risk_level: RiskLevel
    top_risk_factors: list[RiskFactor]
    recommended_action: str
    threshold_used: float
    model_version: str = "1.0.0"


class BatchRequest(BaseModel):
    customers: list[CustomerFeatures]

    @validator("customers")
    def batch_size_limit(cls, v):
        if len(v) > 100:
            raise ValueError("Batch size limited to 100 customers per request")
        return v


class BatchResponse(BaseModel):
    predictions: list[PredictionResponse]
    total_high_risk: int
    total_medium_risk: int
    total_low_risk: int
    batch_size: int


class HealthResponse(BaseModel):
    status: str
    model_version: str
    threshold: float
    model_loaded: bool
