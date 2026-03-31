"""
drift_detection.py — Churn Intelligence System
================================================
Statistical drift detection for production monitoring.

Two types of drift this module detects:

1. DATA DRIFT (covariate shift):
   The distribution of input features changes over time.
   Example: a competitor promotion causes MonthlyCharges distribution to shift.
   Detection: KS test (Kolmogorov-Smirnov) on each numeric feature.

2. PREDICTION DRIFT:
   The distribution of model output probabilities changes over time.
   Example: model starts predicting much higher churn probabilities in January.
   Detection: PSI (Population Stability Index) on prediction scores.

Why these two tests:
- KS test: non-parametric, no distribution assumption, exact p-value.
  Standard for detecting feature distribution shift in production ML.
- PSI: industry standard in credit risk modeling (originated at banks).
  PSI > 0.2 = significant shift, model retraining required.
  PSI 0.1-0.2 = moderate shift, monitor closely.
  PSI < 0.1 = stable.

Run:
    python src/drift_detection.py
    (requires models/ artifacts from train.py)
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from scipy import stats
from dataclasses import dataclass, asdict
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class DriftReport:
    """Structured output from drift detection."""
    feature: str
    test: str                  # "ks" or "psi"
    statistic: float
    p_value: Optional[float]   # None for PSI (no p-value)
    drift_detected: bool
    severity: str              # "none" / "moderate" / "significant"
    action: str


class KSFeatureDriftDetector:
    """
    Kolmogorov-Smirnov test for numeric feature drift.

    The KS test measures the maximum difference between two empirical
    cumulative distribution functions. A small p-value means the two
    samples are unlikely to come from the same distribution.

    Threshold: p-value < 0.05 signals drift (5% significance level).
    """

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        self.reference_distributions = {}

    def fit(self, X_reference: pd.DataFrame, numeric_features: list[str]):
        """Store reference (training) distributions."""
        for feature in numeric_features:
            if feature in X_reference.columns:
                self.reference_distributions[feature] = X_reference[feature].dropna().values
        logger.info(f"KS detector fitted on {len(self.reference_distributions)} features")

    def detect(self, X_current: pd.DataFrame) -> list[DriftReport]:
        """
        Compare current feature distributions against reference.
        Returns one DriftReport per feature.
        """
        reports = []
        for feature, reference in self.reference_distributions.items():
            if feature not in X_current.columns:
                continue

            current = X_current[feature].dropna().values
            if len(current) < 20:
                logger.warning(f"Too few samples for KS test on {feature}: {len(current)}")
                continue

            ks_stat, p_value = stats.ks_2samp(reference, current)
            drift_detected = p_value < self.alpha

            if p_value >= self.alpha:
                severity = "none"
                action = "No action needed."
            elif p_value >= 0.01:
                severity = "moderate"
                action = f"Monitor {feature} closely. Consider retraining if trend continues."
            else:
                severity = "significant"
                action = f"Significant drift in {feature}. Retrain model on recent data."

            reports.append(DriftReport(
                feature=feature,
                test="ks",
                statistic=round(float(ks_stat), 4),
                p_value=round(float(p_value), 6),
                drift_detected=drift_detected,
                severity=severity,
                action=action,
            ))

        return reports


class PSIPredictionDriftDetector:
    """
    Population Stability Index for prediction score drift.

    PSI measures how much a score distribution has shifted between
    a reference period and a current period.

    PSI = sum((current_pct - reference_pct) * ln(current_pct / reference_pct))

    PSI thresholds (industry standard from credit risk modeling):
    < 0.10: No significant change — model is stable
    0.10–0.20: Moderate change — monitor, consider investigation
    > 0.20: Significant change — model retraining required

    Why PSI for predictions specifically:
    - KS test works on raw feature values.
    - PSI is designed for score distributions (bounded 0-1).
    - More interpretable for business stakeholders.
    """

    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins
        self.reference_bins = None
        self.bin_edges = None

    def fit(self, reference_scores: np.ndarray):
        """Compute reference score distribution."""
        self.bin_edges = np.linspace(0, 1, self.n_bins + 1)
        counts, _ = np.histogram(reference_scores, bins=self.bin_edges)
        # Add small epsilon to avoid log(0)
        self.reference_bins = (counts + 1e-6) / (len(reference_scores) + 1e-6 * self.n_bins)
        logger.info(f"PSI detector fitted on {len(reference_scores)} reference scores")

    def detect(self, current_scores: np.ndarray) -> DriftReport:
        """Compute PSI between reference and current score distributions."""
        if self.reference_bins is None:
            raise ValueError("Call fit() before detect()")

        counts, _ = np.histogram(current_scores, bins=self.bin_edges)
        current_bins = (counts + 1e-6) / (len(current_scores) + 1e-6 * self.n_bins)

        psi = float(np.sum(
            (current_bins - self.reference_bins) * np.log(current_bins / self.reference_bins)
        ))

        if psi < 0.10:
            severity = "none"
            drift_detected = False
            action = "Model predictions are stable."
        elif psi < 0.20:
            severity = "moderate"
            drift_detected = True
            action = "Prediction distribution shifting. Investigate input features."
        else:
            severity = "significant"
            drift_detected = True
            action = "Major prediction shift. Retrain model immediately."

        return DriftReport(
            feature="prediction_score",
            test="psi",
            statistic=round(psi, 4),
            p_value=None,
            drift_detected=drift_detected,
            severity=severity,
            action=action,
        )


class DriftMonitor:
    """
    Orchestrates full drift monitoring.
    In production: called by a scheduled job or triggered by /predict volume.
    """

    def __init__(self, numeric_features: list[str]):
        self.ks_detector = KSFeatureDriftDetector(alpha=0.05)
        self.psi_detector = PSIPredictionDriftDetector(n_bins=10)
        self.numeric_features = numeric_features
        self._fitted = False

    def fit_reference(
        self,
        X_reference: pd.DataFrame,
        reference_scores: np.ndarray,
    ):
        """Fit detectors on training/reference data."""
        self.ks_detector.fit(X_reference, self.numeric_features)
        self.psi_detector.fit(reference_scores)
        self._fitted = True

    def check(
        self,
        X_current: pd.DataFrame,
        current_scores: np.ndarray,
    ) -> dict:
        """
        Run all drift checks. Returns summary report.
        Call this on a batch of recent production predictions.
        """
        if not self._fitted:
            raise ValueError("Call fit_reference() before check()")

        feature_reports = self.ks_detector.detect(X_current)
        prediction_report = self.psi_detector.detect(current_scores)
        all_reports = feature_reports + [prediction_report]

        n_drifted = sum(1 for r in all_reports if r.drift_detected)
        significant = [r for r in all_reports if r.severity == "significant"]

        summary = {
            "total_checks": len(all_reports),
            "drift_detected_count": n_drifted,
            "significant_drift": len(significant) > 0,
            "retraining_required": len(significant) > 0,
            "reports": [asdict(r) for r in all_reports],
        }

        if summary["retraining_required"]:
            logger.warning(
                f"DRIFT ALERT: {len(significant)} significant drifts detected. "
                f"Features: {[r.feature for r in significant]}. "
                "Retraining recommended."
            )
        else:
            logger.info(f"Drift check complete. {n_drifted}/{len(all_reports)} checks flagged.")

        return summary

    def save_report(self, report: dict, path: str = "evaluation/drift_report.json"):
        os.makedirs("evaluation", exist_ok=True)
        with open(path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Drift report saved to {path}")


def demo_drift_detection():
    """
    Demonstrates drift detection using simulated production data.
    In production: replace simulated_current with real prediction batches.
    """
    import joblib
    import yaml

    if not os.path.exists("models/xgboost_model.joblib"):
        print("Run train.py first to generate model artifacts.")
        return

    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)

    df = pd.read_csv(config["data"]["raw_path"])
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    numeric_features = ["MonthlyCharges", "TotalCharges", "tenure"]

    model = joblib.load("models/xgboost_model.joblib")
    preprocessor = joblib.load("models/preprocessor.joblib")
    X_processed = preprocessor.transform(df)
    reference_scores = model.predict_proba(X_processed)[:, 1]

    monitor = DriftMonitor(numeric_features=numeric_features)
    monitor.fit_reference(df[numeric_features], reference_scores)

    # Simulate two scenarios:
    print("\n=== Scenario 1: Stable data (no drift expected) ===")
    stable_sample = df[numeric_features].sample(500, random_state=99)
    stable_scores = reference_scores[:500]
    report_stable = monitor.check(stable_sample, stable_scores)
    print(f"Drift detected: {report_stable['drift_detected_count']}/{report_stable['total_checks']} checks")
    print(f"Retraining required: {report_stable['retraining_required']}")

    print("\n=== Scenario 2: Shifted data (competitor promotion — charges dropped 30%) ===")
    shifted = df[numeric_features].copy().sample(500, random_state=42)
    shifted["MonthlyCharges"] = shifted["MonthlyCharges"] * 0.70  # 30% price drop
    shifted_scores = np.clip(reference_scores[:500] * 0.5, 0, 1)  # scores shift too
    report_shifted = monitor.check(shifted, shifted_scores)
    print(f"Drift detected: {report_shifted['drift_detected_count']}/{report_shifted['total_checks']} checks")
    print(f"Retraining required: {report_shifted['retraining_required']}")

    for r in report_shifted["reports"]:
        if r["drift_detected"]:
            print(f"  [{r['severity'].upper()}] {r['feature']}: {r['test']}={r['statistic']:.4f} — {r['action']}")

    monitor.save_report(report_shifted)


if __name__ == "__main__":
    demo_drift_detection()
