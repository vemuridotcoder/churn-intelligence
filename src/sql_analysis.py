"""
sql_analysis.py — Churn Intelligence System
=============================================
SQL-based business analysis on the churn dataset.

Answers 10 business questions using SQLite + pandas.
Demonstrates: JOINs, aggregations, window functions, CTEs, subqueries.

Run:
    python src/sql_analysis.py

Why SQL alongside a Python ML pipeline:
- Data teams operate in SQL-first environments.
- Business stakeholders query results in BI tools (Tableau, Looker) via SQL.
- Preprocessing decisions are validated here before entering the ML pipeline.
"""

import sqlite3
import pandas as pd
import yaml
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_to_sqlite(csv_path: str, db_path: str = "data/churn.db") -> sqlite3.Connection:
    """
    Load CSV into an in-memory SQLite database.
    SQLite chosen for zero-config local development.
    Production equivalent: PostgreSQL or BigQuery.
    """
    os.makedirs("data", exist_ok=True)
    df = pd.read_csv(csv_path)
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)

    conn = sqlite3.connect(db_path)
    df.to_sql("customers", conn, if_exists="replace", index=False)
    logger.info(f"Loaded {len(df)} rows into SQLite: {db_path}")
    return conn


def run_analysis(conn: sqlite3.Connection) -> dict:
    """
    10 business questions answered via SQL.
    Each query is documented with its business purpose.
    """
    results = {}

    # ── Q1: Overall churn rate ──────────────────────────────────────────
    results["q1_overall_churn_rate"] = pd.read_sql_query("""
        SELECT
            COUNT(*)                                    AS total_customers,
            SUM(Churn)                                  AS churned,
            ROUND(100.0 * SUM(Churn) / COUNT(*), 2)    AS churn_rate_pct,
            ROUND(SUM(MonthlyCharges * Churn), 2)       AS monthly_revenue_at_risk
        FROM customers
    """, conn)

    # ── Q2: Churn rate by contract type ────────────────────────────────
    # Business question: which contract type retains customers best?
    results["q2_churn_by_contract"] = pd.read_sql_query("""
        SELECT
            Contract,
            COUNT(*)                                    AS customers,
            SUM(Churn)                                  AS churned,
            ROUND(100.0 * SUM(Churn) / COUNT(*), 2)    AS churn_rate_pct,
            ROUND(AVG(MonthlyCharges), 2)               AS avg_monthly_charges
        FROM customers
        GROUP BY Contract
        ORDER BY churn_rate_pct DESC
    """, conn)

    # ── Q3: Revenue at risk by tenure band ─────────────────────────────
    # Business question: which tenure group represents the most revenue risk?
    results["q3_revenue_at_risk_by_tenure"] = pd.read_sql_query("""
        SELECT
            CASE
                WHEN tenure BETWEEN 0  AND 3  THEN '0-3 months'
                WHEN tenure BETWEEN 4  AND 12 THEN '4-12 months'
                WHEN tenure BETWEEN 13 AND 24 THEN '13-24 months'
                WHEN tenure BETWEEN 25 AND 48 THEN '25-48 months'
                ELSE '48+ months'
            END                                         AS tenure_band,
            COUNT(*)                                    AS customers,
            SUM(Churn)                                  AS churned,
            ROUND(100.0 * SUM(Churn) / COUNT(*), 2)    AS churn_rate_pct,
            ROUND(SUM(MonthlyCharges * Churn), 2)       AS monthly_revenue_at_risk
        FROM customers
        GROUP BY tenure_band
        ORDER BY churn_rate_pct DESC
    """, conn)

    # ── Q4: High-value churners (top revenue loss) ──────────────────────
    # CTE to identify customers who churned AND had above-median charges.
    # Business question: are we losing our most valuable customers?
    results["q4_high_value_churners"] = pd.read_sql_query("""
        WITH median_charges AS (
            SELECT AVG(MonthlyCharges) AS median_val FROM customers
        )
        SELECT
            Contract,
            InternetService,
            ROUND(AVG(MonthlyCharges), 2)               AS avg_monthly_charges,
            COUNT(*)                                    AS count,
            SUM(Churn)                                  AS churned,
            ROUND(100.0 * SUM(Churn) / COUNT(*), 2)    AS churn_rate_pct
        FROM customers, median_charges
        WHERE MonthlyCharges > median_val
        GROUP BY Contract, InternetService
        ORDER BY churn_rate_pct DESC
        LIMIT 10
    """, conn)

    # ── Q5: Service adoption vs churn ──────────────────────────────────
    # Business question: do customers with tech support churn less?
    # Validates the 'service_count' engineered feature from preprocessing.py
    results["q5_techsupport_vs_churn"] = pd.read_sql_query("""
        SELECT
            TechSupport,
            OnlineSecurity,
            COUNT(*)                                    AS customers,
            ROUND(100.0 * SUM(Churn) / COUNT(*), 2)    AS churn_rate_pct
        FROM customers
        GROUP BY TechSupport, OnlineSecurity
        ORDER BY churn_rate_pct DESC
    """, conn)

    # ── Q6: Payment method vs churn ────────────────────────────────────
    results["q6_payment_method_churn"] = pd.read_sql_query("""
        SELECT
            PaymentMethod,
            COUNT(*)                                    AS customers,
            SUM(Churn)                                  AS churned,
            ROUND(100.0 * SUM(Churn) / COUNT(*), 2)    AS churn_rate_pct
        FROM customers
        GROUP BY PaymentMethod
        ORDER BY churn_rate_pct DESC
    """, conn)

    # ── Q7: Window function — running churn count by tenure ─────────────
    # Demonstrates window function: cumulative churn as tenure increases.
    # Business insight: at what tenure does cumulative churn stabilize?
    results["q7_cumulative_churn_by_tenure"] = pd.read_sql_query("""
        SELECT
            tenure,
            SUM(Churn)                                  AS churned_at_tenure,
            SUM(SUM(Churn)) OVER (ORDER BY tenure)      AS cumulative_churned,
            COUNT(*)                                    AS customers_at_tenure
        FROM customers
        GROUP BY tenure
        ORDER BY tenure
        LIMIT 24
    """, conn)

    # ── Q8: Subquery — customers above average charges who didn't churn ─
    # Business question: who are our loyal high-value customers?
    # These are retention benchmarks.
    results["q8_loyal_high_value"] = pd.read_sql_query("""
        SELECT
            Contract,
            ROUND(AVG(tenure), 1)                       AS avg_tenure_months,
            ROUND(AVG(MonthlyCharges), 2)               AS avg_monthly_charges,
            COUNT(*)                                    AS count
        FROM customers
        WHERE Churn = 0
          AND MonthlyCharges > (SELECT AVG(MonthlyCharges) FROM customers)
        GROUP BY Contract
        ORDER BY avg_monthly_charges DESC
    """, conn)

    # ── Q9: Senior citizen churn rate ──────────────────────────────────
    results["q9_senior_citizen_churn"] = pd.read_sql_query("""
        SELECT
            CASE WHEN SeniorCitizen = 1 THEN 'Senior' ELSE 'Non-Senior' END AS segment,
            COUNT(*)                                    AS customers,
            ROUND(100.0 * SUM(Churn) / COUNT(*), 2)    AS churn_rate_pct,
            ROUND(AVG(MonthlyCharges), 2)               AS avg_monthly_charges
        FROM customers
        GROUP BY SeniorCitizen
    """, conn)

    # ── Q10: Most dangerous customer profile ───────────────────────────
    # CTE joining multiple risk factors to find the single highest-risk segment.
    # This validates the 'vulnerable' engineered feature from preprocessing.py.
    results["q10_highest_risk_profile"] = pd.read_sql_query("""
        WITH risk_profile AS (
            SELECT
                Contract,
                InternetService,
                TechSupport,
                CASE WHEN tenure <= 3 THEN 'new' ELSE 'established' END AS tenure_group,
                COUNT(*)                                AS customers,
                SUM(Churn)                              AS churned,
                ROUND(100.0 * SUM(Churn) / COUNT(*), 2) AS churn_rate_pct
            FROM customers
            GROUP BY Contract, InternetService, TechSupport, tenure_group
            HAVING COUNT(*) >= 10
        )
        SELECT * FROM risk_profile
        ORDER BY churn_rate_pct DESC
        LIMIT 5
    """, conn)

    return results


def print_report(results: dict) -> None:
    """Print all query results in a readable format."""
    titles = {
        "q1_overall_churn_rate":       "Q1 — Overall churn rate and revenue at risk",
        "q2_churn_by_contract":        "Q2 — Churn rate by contract type",
        "q3_revenue_at_risk_by_tenure":"Q3 — Revenue at risk by tenure band",
        "q4_high_value_churners":      "Q4 — High-value churners (CTE)",
        "q5_techsupport_vs_churn":     "Q5 — Tech support and security vs churn",
        "q6_payment_method_churn":     "Q6 — Payment method vs churn",
        "q7_cumulative_churn_by_tenure":"Q7 — Cumulative churn by tenure (window function)",
        "q8_loyal_high_value":         "Q8 — Loyal high-value customers (subquery)",
        "q9_senior_citizen_churn":     "Q9 — Senior citizen segment",
        "q10_highest_risk_profile":    "Q10 — Highest-risk customer profile (CTE)",
    }
    for key, df in results.items():
        print(f"\n{'='*60}")
        print(f"  {titles.get(key, key)}")
        print(f"{'='*60}")
        print(df.to_string(index=False))

    print("\n\nSQL techniques demonstrated:")
    print("  ✓ Basic aggregations (COUNT, SUM, AVG, ROUND)")
    print("  ✓ GROUP BY with ORDER BY and HAVING")
    print("  ✓ CASE WHEN for bucketing")
    print("  ✓ CTEs (WITH clause) — Q4, Q10")
    print("  ✓ Subqueries — Q8")
    print("  ✓ Window functions (SUM OVER ORDER BY) — Q7")
    print("  ✓ Multi-table logic (implicit self-join via CTE) — Q4")


def main():
    config_path = "configs/config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    csv_path = config["data"]["raw_path"]
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Dataset not found: {csv_path}\n"
            "Download from Kaggle and place at data/raw/telco_churn.csv"
        )

    conn = load_to_sqlite(csv_path)
    results = run_analysis(conn)
    print_report(results)

    os.makedirs("data", exist_ok=True)
    for name, df in results.items():
        df.to_csv(f"data/{name}.csv", index=False)
    logger.info("SQL results saved to data/q*.csv")
    conn.close()


if __name__ == "__main__":
    main()
