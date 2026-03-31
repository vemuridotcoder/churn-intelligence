"""
evaluate.py — Churn Intelligence System
========================================
Standalone evaluation script. Run after training to get full metrics report.

Run:
    python src/evaluate.py

Outputs a printed report + saves evaluation/results.json
"""

import os
import json
import yaml
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score
)
from sklearn.model_selection import train_test_split

from preprocessing import ChurnPreprocessor


def load_artifacts():
    required = [
        "models/xgboost_model.joblib",
        "models/preprocessor.joblib",
        "models/threshold.joblib",
        "models/feature_names.joblib",
    ]
    missing = [p for p in required if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"Run train.py first. Missing: {missing}")

    return {
        "model": joblib.load("models/xgboost_model.joblib"),
        "preprocessor": joblib.load("models/preprocessor.joblib"),
        "threshold": joblib.load("models/threshold.joblib"),
        "feature_names": joblib.load("models/feature_names.joblib"),
    }


def print_section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def evaluate(config_path: str = "configs/config.yaml"):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    artifacts = load_artifacts()
    model = artifacts["model"]
    preprocessor = artifacts["preprocessor"]
    threshold = artifacts["threshold"]

    # Load and prepare test data
    df = pd.read_csv(config["data"]["raw_path"])
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    target = df.pop("Churn")

    _, X_test_raw, _, y_test = train_test_split(
        df, target,
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"],
        stratify=target
    )

    X_test = preprocessor.transform(X_test_raw)
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    # ── 1. Threshold decision explanation ──────────────────────────────
    print_section("THRESHOLD DECISION")
    print(f"""
  Business context:
  ─────────────────
  False Negative cost  : ~INR 1,200/month (revenue lost per missed churner)
  False Positive cost  : ~INR   150       (one retention call)
  Cost ratio           : 8:1 (FN is 8× more expensive than FP)

  Decision             : Use threshold {threshold:.3f} instead of default 0.5
  Effect               : Higher recall at cost of precision — correct tradeoff
                         given the cost asymmetry.
    """)

    # ── 2. Core metrics ─────────────────────────────────────────────────
    print_section("MODEL PERFORMANCE (threshold = {:.3f})".format(threshold))
    print("\n  NOTE: Accuracy is intentionally excluded.")
    print("  A model predicting 'no churn' for everyone gets 73.5% accuracy.")
    print("  That model catches zero churners. Accuracy is a useless metric here.\n")
    print(classification_report(y_test, y_pred, target_names=["Retained", "Churned"]))

    # ── 3. AUC scores ───────────────────────────────────────────────────
    auc_roc = roc_auc_score(y_test, y_prob)
    auc_pr = average_precision_score(y_test, y_prob)
    print(f"  AUC-ROC  : {auc_roc:.4f}  (1.0 = perfect, 0.5 = random)")
    print(f"  AUC-PR   : {auc_pr:.4f}  (better metric for imbalanced data)")

    # ── 4. Confusion matrix ─────────────────────────────────────────────
    print_section("CONFUSION MATRIX")
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f"""
              Predicted: Retained  Predicted: Churned
  Actual: Retained     {tn:>6}          {fp:>6}      (FP = unnecessary calls)
  Actual: Churned      {fn:>6}          {tp:>6}      (TP = churners caught)

  True Negatives  (correctly retained) : {tn}
  False Positives (unnecessary calls)  : {fp}
  False Negatives (missed churners)    : {fn}  ← most expensive error
  True Positives  (churners caught)    : {tp}
    """)

    # ── 5. Business impact ──────────────────────────────────────────────
    print_section("BUSINESS IMPACT (scaled to 10,000 customers)")
    b = config["business"]
    scale = b["customer_base_example"] / len(y_test)

    retained_customers = int(tp * b["retention_success_rate"] * scale)
    calls_made = int((tp + fp) * scale)
    revenue_saved = int(retained_customers * b["monthly_revenue_per_customer"])
    call_cost = int(calls_made * b["retention_call_cost"])
    net_impact = revenue_saved - call_cost
    missed_revenue = int(fn * scale * b["monthly_revenue_per_customer"])

    print(f"""
  Customer base           : {b['customer_base_example']:,}
  Churners flagged        : {calls_made:,}
  Retention calls made    : {calls_made:,}   (at INR {b['retention_call_cost']}/call)
  Customers retained      : {retained_customers:,}   ({b['retention_success_rate']*100:.0f}% of correctly identified churners)

  Gross revenue saved     : INR {revenue_saved:,}/month
  Retention call costs    : INR {call_cost:,}/month
  ─────────────────────────────────────────────────
  NET MONTHLY IMPACT      : INR {net_impact:,}/month

  Revenue missed (FN)     : INR {missed_revenue:,}/month
  (churners we failed to catch)
    """)

    # ── 6. Where this model fails ────────────────────────────────────────
    print_section("MODEL FAILURE ANALYSIS")
    print("""
  1. Distribution shift
     Model trained on telecom data. If deployed on SaaS or EdTech churn,
     the feature relationships change fundamentally. Retrain required.

  2. New customer problem
     Customers with tenure < 1 month have no behavioral signal.
     Model defaults to medium risk for all new customers.
     A separate model for new-customer churn is needed.

  3. Seasonality blindness
     This model treats all time periods identically.
     Churn spikes during competitor promotions or billing cycles
     are invisible to a static cross-sectional model.

  4. Threshold drift
     The threshold (0.35) was optimized on this dataset with
     INR 1,200 / INR 150 cost ratio. If retention call costs rise
     or average revenue per customer changes, threshold must be re-tuned.

  5. Feature staleness
     MonthlyCharges and TotalCharges become stale if pricing changes.
     Model should be retrained quarterly at minimum.
    """)

    # Save results
    os.makedirs("evaluation", exist_ok=True)
    results = {
        "threshold": float(threshold),
        "auc_roc": round(auc_roc, 4),
        "auc_pr": round(auc_pr, 4),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "business_impact": {
            "net_monthly_impact_inr": net_impact,
            "customers_retained": retained_customers,
            "missed_revenue_inr": missed_revenue,
        }
    }
    with open("evaluation/results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n  Full results saved to evaluation/results.json")


if __name__ == "__main__":
    evaluate()
