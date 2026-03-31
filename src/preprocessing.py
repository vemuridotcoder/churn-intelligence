"""
preprocessing.py — Churn Intelligence System
=============================================
Transforms raw Telco CSV into model-ready features.

Design decisions documented inline. Every transformation has a reason.
This module is stateful: fit() on training data, transform() on any split.
This prevents data leakage — test set statistics never influence training.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
import joblib
import yaml
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChurnPreprocessor(BaseEstimator, TransformerMixin):
    """
    Stateful preprocessing pipeline for Telco Churn data.

    Fits on training data only. Call transform() on train/val/test splits
    and on single inference requests via the API.

    Why a class instead of functions:
    - Stores median, encoders, scaler from training data
    - Ensures identical transformation at inference time
    - Sklearn-compatible: can be saved with joblib and loaded in the API
    """

    def __init__(self, config: dict):
        self.config = config
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.median_monthly_charges = None
        self.feature_names = None
        self._is_fitted = False

    def fit(self, df: pd.DataFrame) -> "ChurnPreprocessor":
        """
        Learn statistics from training data only.
        Never call fit() on validation or test splits.
        """
        df = df.copy()
        df = self._fix_data_quality(df)
        df = self._engineer_features(df)

        # Store median from training data for vulnerable feature
        # Using median (not mean) because MonthlyCharges is right-skewed
        self.median_monthly_charges = df["MonthlyCharges"].median()

        # Fit label encoders on training distribution
        cat_cols = self.config["features"]["categorical"]
        for col in cat_cols:
            if col in df.columns:
                le = LabelEncoder()
                le.fit(df[col].astype(str))
                self.label_encoders[col] = le

        # Fit scaler on training numeric features
        df = self._encode_categoricals(df)
        numeric_cols = self.config["features"]["numeric"]
        existing_numeric = [c for c in numeric_cols if c in df.columns]
        self.scaler.fit(df[existing_numeric])
        self.feature_names = existing_numeric + [
            c for c in self.config["features"]["categorical"] if c in df.columns
        ]

        self._is_fitted = True
        logger.info(f"Preprocessor fitted. Features: {len(self.feature_names)}")
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply learned transformations to any split.
        Raises ValueError if fit() was not called first.
        """
        if not self._is_fitted:
            raise ValueError("Call fit() before transform().")

        df = df.copy()
        df = self._fix_data_quality(df)
        df = self._engineer_features(df)
        df = self._encode_categoricals(df)
        df = self._scale_numerics(df)

        # Select only model features in correct order
        available = [f for f in self.feature_names if f in df.columns]
        return df[available]

    def fit_transform(self, df: pd.DataFrame, y=None) -> pd.DataFrame:
        return self.fit(df).transform(df)

    # ------------------------------------------------------------------
    # Private methods — each has a documented reason for existing
    # ------------------------------------------------------------------

    def _fix_data_quality(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fix known data quality issues in the IBM Telco dataset.

        Issues found during EDA (see notebooks/exploration.ipynb):

        1. TotalCharges is object dtype despite being numeric.
           Root cause: 11 rows have empty string "" for TotalCharges.
           These are customers with tenure=0 (brand new, no charges yet).
           Fix: convert to float, fill NaN with 0 (correct business logic,
           not imputation — these customers genuinely have zero charges).

        2. SeniorCitizen is already 0/1 int — no transformation needed.

        3. Churn column has "Yes"/"No" strings — convert to 1/0 for training.
           At inference time this column may not exist, handled gracefully.
        """
        df = df.copy()

        # Fix 1: TotalCharges
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        n_nulls = df["TotalCharges"].isna().sum()
        if n_nulls > 0:
            logger.info(f"TotalCharges: {n_nulls} NaN filled with 0 (new customers)")
        df["TotalCharges"] = df["TotalCharges"].fillna(0)

        # Fix 2: Drop customerID — not a feature, it's an identifier
        if "customerID" in df.columns:
            df = df.drop("customerID", axis=1)

        # Fix 3: Encode target if present
        if "Churn" in df.columns:
            df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

        return df

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create 3 engineered features. Each has a documented hypothesis.

        Feature 1: charge_per_tenure
        Hypothesis: customers paying high monthly charges relative to how long
        they've been with the company haven't yet experienced the full value.
        High charge + low tenure = high churn risk.
        Formula: MonthlyCharges / (tenure + 1)
        The +1 prevents division by zero for new customers.

        Feature 2: service_count
        Hypothesis: customers using more services have higher switching costs.
        Cancelling means losing phone + internet + streaming + backup simultaneously.
        Higher service adoption = lower churn risk.
        Range: 0–7 (count of optional services subscribed).

        Feature 3: vulnerable
        Hypothesis: month-to-month contract AND above-median monthly charges
        is the highest-risk combination in the data (verified in EDA:
        churn rate 53% for this group vs 11% overall for long-term contracts).
        This interaction term captures what linear models miss.
        """
        df = df.copy()

        # Feature 1: charge per tenure
        df["charge_per_tenure"] = df["MonthlyCharges"] / (df["tenure"] + 1)

        # Feature 2: service adoption count
        service_cols = self.config["features"]["service_columns"]
        existing_services = [c for c in service_cols if c in df.columns]

        def is_active(val):
            return 1 if val == "Yes" else 0

        df["service_count"] = df[existing_services].applymap(is_active).sum(axis=1)

        # Feature 3: vulnerable flag
        # Use training-fitted median at inference; use current median at training
        median = (
            self.median_monthly_charges
            if self.median_monthly_charges is not None
            else df["MonthlyCharges"].median()
        )

        df["vulnerable"] = (
            (df["Contract"] == "Month-to-month")
            & (df["MonthlyCharges"] > median)
        ).astype(int)

        # Feature 4: tenure risk score
        # Non-linear tenure effect: highest churn in months 1-3,
        # drops sharply after month 24 (customers who survive become loyal)
        # Modelled as inverse log to capture this decay shape
        df["tenure_risk_score"] = 1 / (np.log1p(df["tenure"]) + 1)

        return df

    def _encode_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Label encode categorical columns using training-fitted encoders.

        Why label encoding instead of one-hot:
        - XGBoost handles ordinal-style label encoding natively
        - One-hot with 15 categorical columns creates 50+ sparse features
        - For tree-based models, label encoding performs comparably
          with lower dimensionality

        Unknown categories at inference: mapped to -1 (unseen label).
        The model was not trained on -1 values, so this triggers
        the "uncertain" response band in the API.
        """
        df = df.copy()
        for col, encoder in self.label_encoders.items():
            if col in df.columns:
                known_classes = set(encoder.classes_)
                df[col] = df[col].astype(str).apply(
                    lambda x: encoder.transform([x])[0]
                    if x in known_classes
                    else -1
                )
        return df

    def _scale_numerics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize numeric features using training-fitted scaler.

        Why scale for XGBoost: XGBoost is tree-based and scale-invariant,
        but scaling is kept for two reasons:
        1. SHAP values are more interpretable on scaled features
        2. The same preprocessor is used for Logistic Regression baseline,
           which requires scaling

        Transform only — never fit on validation/test data.
        """
        numeric_cols = self.config["features"]["numeric"]
        existing_numeric = [c for c in numeric_cols if c in df.columns]
        df[existing_numeric] = self.scaler.transform(df[existing_numeric])
        return df

    def save(self, path: str) -> None:
        joblib.dump(self, path)
        logger.info(f"Preprocessor saved to {path}")

    @classmethod
    def load(cls, path: str) -> "ChurnPreprocessor":
        return joblib.load(path)
