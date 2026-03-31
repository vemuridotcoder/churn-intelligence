"""
experiment_tracking.py — Churn Intelligence System
=====================================================
MLflow experiment tracking wrapper.

Logs every training run with:
- All hyperparameters from config.yaml
- Evaluation metrics (AUC-ROC, Precision, Recall, F1)
- Business metrics (net revenue impact)
- Model artifacts (serialized model + preprocessor)
- Threshold decision rationale

Run MLflow UI after training:
    mlflow ui --port 5000
    Open: http://localhost:5000

Why MLflow over manual logging:
- Reproducibility: every run is versioned with exact parameters
- Comparison: side-by-side view of all experiments
- Deployment: MLflow Model Registry integrates with serving infrastructure
- Industry standard: used at Databricks, Airbnb, LinkedIn
"""

import os
import json
import logging
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from datetime import datetime

logger = logging.getLogger(__name__)

EXPERIMENT_NAME = "churn-prediction"


def setup_experiment() -> str:
    """Create or retrieve MLflow experiment. Returns experiment ID."""
    mlflow.set_tracking_uri("mlruns")  # local file-based tracking
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        experiment_id = mlflow.create_experiment(
            EXPERIMENT_NAME,
            tags={
                "project": "churn-intelligence",
                "dataset": "IBM Telco Customer Churn",
                "problem_type": "binary_classification",
            }
        )
    else:
        experiment_id = experiment.experiment_id
    mlflow.set_experiment(EXPERIMENT_NAME)
    return experiment_id


class ExperimentTracker:
    """
    Context manager wrapping an MLflow run.

    Usage:
        with ExperimentTracker("XGBoost", config) as tracker:
            model = train_xgboost(X_train, y_train, config)
            tracker.log_metrics({"auc_roc": 0.87, "recall": 0.89})
            tracker.log_model(model, "xgboost")
    """

    def __init__(self, run_name: str, config: dict):
        self.run_name = run_name
        self.config = config
        self.run = None

    def __enter__(self):
        setup_experiment()
        self.run = mlflow.start_run(run_name=self.run_name)
        self._log_config()
        logger.info(f"MLflow run started: {self.run_name} (id={self.run.info.run_id[:8]})")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            mlflow.set_tag("status", "FAILED")
            mlflow.set_tag("error", str(exc_val))
        else:
            mlflow.set_tag("status", "SUCCESS")
        mlflow.end_run()
        logger.info(f"MLflow run ended: {self.run_name}")

    def _log_config(self):
        """Log all config values as MLflow parameters."""
        # Flatten nested config into dot-notation keys
        def flatten(d, prefix=""):
            items = {}
            for k, v in d.items():
                key = f"{prefix}.{k}" if prefix else k
                if isinstance(v, dict):
                    items.update(flatten(v, key))
                elif isinstance(v, (int, float, str, bool)):
                    items[key] = v
            return items

        flat_config = flatten(self.config)
        # MLflow has a 250-param limit per run; log most important ones
        important_keys = [
            "threshold.default",
            "models.xgboost.n_estimators",
            "models.xgboost.max_depth",
            "models.xgboost.learning_rate",
            "models.random_forest.n_estimators",
            "data.test_size",
            "business.monthly_revenue_per_customer",
            "business.retention_call_cost",
            "business.retention_success_rate",
        ]
        for key in important_keys:
            if key in flat_config:
                mlflow.log_param(key, flat_config[key])

        mlflow.set_tag("run_timestamp", datetime.now().isoformat())

    def log_metrics(self, metrics: dict, step: int = None):
        """Log evaluation metrics."""
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(key, value, step=step)

    def log_business_impact(self, impact: dict):
        """Log business-level metrics separately for executive reporting."""
        business_metrics = {
            "business.net_revenue_impact_inr": impact.get("net_revenue_impact_inr", 0),
            "business.customers_retained": impact.get("customers_retained", 0),
            "business.missed_revenue_inr": impact.get("missed_revenue_inr", 0),
            "business.churners_correctly_identified": impact.get("churners_correctly_identified", 0),
        }
        for key, value in business_metrics.items():
            mlflow.log_metric(key, value)

    def log_threshold_decision(self, threshold: float, fn_cost: int, fp_cost: int):
        """Log threshold decision with business rationale."""
        mlflow.log_param("threshold.optimized", threshold)
        mlflow.log_param("threshold.fn_cost_weight", fn_cost)
        mlflow.log_param("threshold.fp_cost_weight", fp_cost)
        mlflow.set_tag(
            "threshold.rationale",
            f"FN={fn_cost}x more expensive than FP={fp_cost}x. "
            f"Threshold {threshold:.3f} minimizes total business cost."
        )

    def log_model(self, model, model_name: str, preprocessor=None):
        """Log trained model as MLflow artifact."""
        if "xgboost" in model_name.lower():
            mlflow.xgboost.log_model(model, model_name)
        else:
            mlflow.sklearn.log_model(model, model_name)

        if preprocessor is not None:
            import joblib, tempfile
            with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
                joblib.dump(preprocessor, f.name)
                mlflow.log_artifact(f.name, "preprocessor")

        logger.info(f"Model logged: {model_name}")

    def log_comparison_table(self, comparison_df):
        """Log model comparison CSV as artifact."""
        import tempfile
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as f:
            comparison_df.to_csv(f.name, index=False)
            mlflow.log_artifact(f.name, "model_comparison")
