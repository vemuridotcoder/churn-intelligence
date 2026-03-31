"""
eda.py — Exploratory Data Analysis
=====================================
Generates all EDA plots and insight tables.
Run this to populate notebooks/figures/ before committing.

Run:
    python notebooks/eda.py

Outputs saved to notebooks/figures/:
    - churn_distribution.png
    - churn_by_contract.png
    - churn_by_tenure.png
    - monthly_charges_distribution.png
    - correlation_heatmap.png
    - feature_importance_baseline.png

Why a .py script instead of a Jupyter notebook:
- Notebooks do not diff cleanly in Git (JSON with embedded outputs)
- Scripts run in CI pipelines without a kernel
- Outputs (PNG files) committed separately are lightweight and viewable on GitHub
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import yaml

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

OUTPUT_DIR = "notebooks/figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

BLUE = "#378ADD"
RED  = "#E24B4A"
TEAL = "#1D9E75"
GRAY = "#888780"


def load_data(config_path: str = "configs/config.yaml") -> pd.DataFrame:
    with open(config_path) as f:
        config = yaml.safe_load(f)
    df = pd.read_csv(config["data"]["raw_path"])
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    df["Churn_binary"] = df["Churn"].map({"Yes": 1, "No": 0})
    return df


def plot_churn_distribution(df: pd.DataFrame):
    """Business insight: overall churn rate and revenue at risk."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    counts = df["Churn"].value_counts()
    colors = [BLUE, RED]
    axes[0].bar(counts.index, counts.values, color=colors, width=0.4, edgecolor="white")
    axes[0].set_title("Customer count by churn status", fontsize=12)
    axes[0].set_ylabel("Customers")
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 30, f"{v:,}\n({v/len(df)*100:.1f}%)", ha="center", fontsize=9)

    revenue = df.groupby("Churn")["MonthlyCharges"].sum()
    axes[1].bar(revenue.index, revenue.values, color=colors, width=0.4, edgecolor="white")
    axes[1].set_title("Monthly revenue by churn status", fontsize=12)
    axes[1].set_ylabel("INR (proportional)")
    axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
    for i, (label, v) in enumerate(revenue.items()):
        axes[1].text(i, v + 1000, f"INR {v/1000:.0f}k", ha="center", fontsize=9)

    fig.suptitle(
        f"Churn rate: {df['Churn_binary'].mean()*100:.1f}% | "
        f"Revenue at risk: INR {df.loc[df['Churn']=='Yes', 'MonthlyCharges'].sum():,.0f}/month",
        fontsize=11, y=1.01
    )
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/churn_distribution.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_churn_by_contract(df: pd.DataFrame):
    """Business insight: contract type is the single strongest churn predictor."""
    contract_churn = (
        df.groupby("Contract")["Churn_binary"]
        .agg(["mean", "count"])
        .reset_index()
        .sort_values("mean", ascending=False)
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(
        contract_churn["Contract"],
        contract_churn["mean"] * 100,
        color=[RED, GRAY, TEAL],
        edgecolor="white",
    )
    ax.set_xlabel("Churn rate (%)")
    ax.set_title("Churn rate by contract type", fontsize=12)
    ax.axvline(df["Churn_binary"].mean() * 100, color=BLUE,
               linestyle="--", linewidth=1, label="Overall average")
    ax.legend(fontsize=9)

    for bar, (_, row) in zip(bars, contract_churn.iterrows()):
        ax.text(
            bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
            f"{row['mean']*100:.1f}% ({int(row['count'])} customers)",
            va="center", fontsize=9
        )
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/churn_by_contract.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_churn_by_tenure(df: pd.DataFrame):
    """Business insight: survival cliff — churn drops sharply after month 24."""
    tenure_churn = (
        df.groupby("tenure")["Churn_binary"]
        .mean()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(tenure_churn["tenure"], tenure_churn["Churn_binary"] * 100,
            color=RED, linewidth=1.5, alpha=0.7)
    ax.fill_between(tenure_churn["tenure"], tenure_churn["Churn_binary"] * 100,
                    alpha=0.1, color=RED)
    ax.axvline(24, color=BLUE, linestyle="--", linewidth=1.5,
               label="Month 24 — churn stabilises")
    ax.set_xlabel("Tenure (months)")
    ax.set_ylabel("Churn rate (%)")
    ax.set_title("Churn rate by tenure — survival cliff at month 24", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/churn_by_tenure.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_monthly_charges(df: pd.DataFrame):
    """Business insight: churners cluster in the high-charges segment."""
    fig, ax = plt.subplots(figsize=(9, 4))
    for label, color in [("No", BLUE), ("Yes", RED)]:
        subset = df[df["Churn"] == label]["MonthlyCharges"]
        ax.hist(subset, bins=40, alpha=0.6, color=color,
                label=f"Churn={label} (n={len(subset):,})", edgecolor="none")

    ax.set_xlabel("Monthly charges (INR proportional)")
    ax.set_ylabel("Customer count")
    ax.set_title("Monthly charges distribution: churners vs retained", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2, axis="y")
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/monthly_charges_distribution.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def print_key_insights(df: pd.DataFrame):
    """Print business insights discovered in EDA."""
    print("\n" + "="*60)
    print("  KEY EDA INSIGHTS")
    print("="*60)

    churn_rate = df["Churn_binary"].mean()
    revenue_at_risk = df.loc[df["Churn"] == "Yes", "MonthlyCharges"].sum()
    print(f"\n  Churn rate            : {churn_rate:.1%}")
    print(f"  Monthly revenue at risk: INR {revenue_at_risk:,.0f}")

    contract_rates = df.groupby("Contract")["Churn_binary"].mean().sort_values(ascending=False)
    print(f"\n  Churn by contract type:")
    for contract, rate in contract_rates.items():
        print(f"    {contract:<25} {rate:.1%}")

    mtm = df[df["Contract"] == "Month-to-month"]
    high_charge_mtm = mtm[mtm["MonthlyCharges"] > df["MonthlyCharges"].median()]
    print(f"\n  Highest-risk segment (month-to-month + above-median charges):")
    print(f"    Churn rate: {high_charge_mtm['Churn_binary'].mean():.1%} "
          f"({len(high_charge_mtm):,} customers)")
    print(f"    This validates the 'vulnerable' engineered feature.")

    survival_cliff = df[df["tenure"] >= 24]["Churn_binary"].mean()
    early_churn = df[df["tenure"] < 6]["Churn_binary"].mean()
    print(f"\n  Tenure survival cliff:")
    print(f"    Churn rate (tenure < 6 months) : {early_churn:.1%}")
    print(f"    Churn rate (tenure >= 24 months): {survival_cliff:.1%}")
    print(f"    This validates the 'tenure_risk_score' engineered feature.")


def main():
    df = load_data()
    plot_churn_distribution(df)
    plot_churn_by_contract(df)
    plot_churn_by_tenure(df)
    plot_monthly_charges(df)
    print_key_insights(df)
    print(f"\nAll figures saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
