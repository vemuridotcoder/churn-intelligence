"""
explain.py — Churn Intelligence System
========================================
SHAP-based explainability for individual predictions.

Why SHAP over feature_importances_:
- feature_importances_ = global average impact across all predictions
- SHAP = exact contribution of each feature to THIS specific prediction
- Actionable: customer success team addresses the top 3 risk factors
- Theoretically sound: Shapley values guarantee consistent, fair attribution
  rooted in cooperative game theory

Used by the API to populate the 'top_risk_factors' response field.
"""

import numpy as np
import joblib
import logging

logger = logging.getLogger(__name__)


class ChurnExplainer:
    """
    Wraps a SHAP TreeExplainer for per-prediction explanations.

    At inference time, returns the top N features driving churn probability
    for a specific customer — both magnitude and direction.
    """

    def __init__(self, explainer_path: str, feature_names_path: str):
        self.explainer = joblib.load(explainer_path)
        self.feature_names = joblib.load(feature_names_path)

    def explain(self, X, top_n: int = 3) -> list[dict]:
        """
        Return top_n features driving churn probability for one customer.

        Args:
            X: preprocessed feature array, shape (1, n_features)
            top_n: number of risk factors to return

        Returns:
            List of dicts with feature name, SHAP value, and direction.

        SHAP value interpretation:
        - Positive value → this feature INCREASES churn probability
        - Negative value → this feature DECREASES churn probability
        - Magnitude → how much this feature moves the prediction
        """
        try:
            shap_values = self.explainer.shap_values(X)

            # For binary classification, TreeExplainer returns values for class 1 (churn)
            if isinstance(shap_values, list):
                sv = shap_values[1][0]
            else:
                sv = shap_values[0]

            # Sort by absolute magnitude — biggest movers first
            top_indices = np.argsort(np.abs(sv))[-top_n:][::-1]

            risk_factors = []
            for idx in top_indices:
                direction = "increases" if sv[idx] > 0 else "decreases"
                risk_factors.append({
                    "feature": self.feature_names[idx],
                    "impact": round(float(sv[idx]), 4),
                    "direction": direction,
                    "description": self._human_readable(
                        self.feature_names[idx], sv[idx]
                    ),
                })

            return risk_factors

        except Exception as e:
            logger.warning(f"SHAP explanation failed: {e}")
            return [{"feature": "unknown", "impact": 0.0, "direction": "unknown",
                     "description": "Explanation unavailable"}]

    def _human_readable(self, feature: str, shap_value: float) -> str:
        """
        Convert feature name + SHAP value into a sentence a human can act on.
        Customer success teams read this, not data scientists.
        """
        direction = "high" if shap_value > 0 else "low"
        templates = {
            "Contract": f"Month-to-month contract significantly increases risk"
                        if shap_value > 0 else "Long-term contract reduces churn risk",
            "MonthlyCharges": f"{'High' if shap_value > 0 else 'Low'} monthly charges affecting retention",
            "tenure": (
                "Short tenure — customer has not experienced full value yet"
                if shap_value > 0
                else "Long tenure — loyal established customer"
            ),
            "charge_per_tenure": "High charges relative to time with company — value not established",
            "TechSupport": f"{'No' if shap_value > 0 else 'Active'} tech support subscription",
            "OnlineSecurity": f"{'No' if shap_value > 0 else 'Active'} online security subscription",
            "InternetService": f"Internet service type influencing satisfaction",
            "vulnerable": "High-risk combination: month-to-month contract + high charges",
            "service_count": f"{'Low' if shap_value > 0 else 'High'} service adoption — "
                             f"{'low switching costs' if shap_value > 0 else 'high switching costs'}",
            "tenure_risk_score": f"{'Early-stage' if shap_value > 0 else 'Established'} customer relationship",
        }
        return templates.get(feature, f"{feature} is a {direction} contributor to churn risk")
