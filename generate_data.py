import pandas as pd
import numpy as np
import os

np.random.seed(42)
os.makedirs("data", exist_ok=True)

# --- Monthly Financial Data (2 years) ---
months = pd.date_range(start="2023-01-01", periods=24, freq="MS")
base_revenue = 1_200_000
revenue = []
expenses = []
cogs = []
cash_inflow = []
cash_outflow = []

for i, m in enumerate(months):
    seasonal = 1 + 0.08 * np.sin(2 * np.pi * i / 12)
    trend = 1 + 0.004 * i
    rev = base_revenue * seasonal * trend * np.random.uniform(0.95, 1.05)
    # Inject anomaly in month 10 (Oct 2023) and month 22 (Oct 2024)
    exp_mult = 1.42 if i in [9, 21] else np.random.uniform(0.62, 0.70)
    exp = rev * exp_mult
    cog = rev * np.random.uniform(0.58, 0.64)
    ci = rev * np.random.uniform(0.90, 1.05)
    co = exp * np.random.uniform(0.88, 0.98)
    revenue.append(round(rev, 2))
    expenses.append(round(exp, 2))
    cogs.append(round(cog, 2))
    cash_inflow.append(round(ci, 2))
    cash_outflow.append(round(co, 2))

monthly_df = pd.DataFrame({
    "date": months,
    "revenue": revenue,
    "cogs": cogs,
    "operating_expenses": expenses,
    "cash_inflow": cash_inflow,
    "cash_outflow": cash_outflow,
})
monthly_df["gross_profit"] = monthly_df["revenue"] - monthly_df["cogs"]
monthly_df["net_profit"] = monthly_df["revenue"] - monthly_df["cogs"] - monthly_df["operating_expenses"]
monthly_df["net_cashflow"] = monthly_df["cash_inflow"] - monthly_df["cash_outflow"]
monthly_df.to_csv("data/monthly_financials.csv", index=False)

# --- Balance Sheet (Quarterly) ---
quarters = pd.date_range(start="2023-01-01", periods=8, freq="QS")
bs_data = []
current_assets = 1_800_000
total_equity = 2_200_000
for i, q in enumerate(quarters):
    ca = current_assets * (1 + 0.015 * i) * np.random.uniform(0.97, 1.03)
    cl = ca / (1.7 - 0.04 * i)  # current ratio declining
    inv = ca * 0.35
    quick_assets = ca - inv
    total_assets = ca * 2.4 * np.random.uniform(0.98, 1.02)
    te = total_equity * (1 + 0.01 * i)
    total_debt = total_assets - te
    bs_data.append({
        "quarter": q.strftime("%Y-Q") + str((i % 4) + 1),
        "current_assets": round(ca, 2),
        "current_liabilities": round(cl, 2),
        "inventory": round(inv, 2),
        "quick_assets": round(quick_assets, 2),
        "total_assets": round(total_assets, 2),
        "total_equity": round(te, 2),
        "total_debt": round(total_debt, 2),
        "net_profit": round(monthly_df[monthly_df["date"].dt.quarter == (i % 4) + 1]["net_profit"].mean() * 3, 2),
    })

bs_df = pd.DataFrame(bs_data)
bs_df.to_csv("data/balance_sheet.csv", index=False)

# --- Budget vs Actual ---
categories = ["Sales Revenue", "Operating Expenses", "Payroll", "Marketing", "COGS", "Net Profit"]
budget =     [4_800_000,       2_560_000,            3_560_000,  840_000,    2_880_000,  3_080_000]
actual_mult = [1.15,           1.28,                 1.16,       0.93,       1.03,       0.79]
actual = [round(b * m, 2) for b, m in zip(budget, actual_mult)]
var_df = pd.DataFrame({
    "category": categories,
    "budget": budget,
    "actual": actual,
    "variance": [round(a - b, 2) for a, b in zip(actual, budget)],
    "variance_pct": [round((a - b) / b * 100, 2) for a, b in zip(actual, budget)],
    "type": ["Revenue", "Expense", "Expense", "Expense", "Expense", "Profit"],
})
var_df["status"] = var_df.apply(
    lambda r: "Favourable" if (r["type"] == "Revenue" and r["variance"] > 0)
              or (r["type"] in ["Expense"] and r["variance"] < 0)
              or (r["type"] == "Profit" and r["variance"] > 0)
              else "Unfavourable", axis=1
)
var_df.to_csv("data/variance.csv", index=False)

print("Data generated successfully.")
print(f"  monthly_financials.csv — {len(monthly_df)} rows")
print(f"  balance_sheet.csv      — {len(bs_df)} rows")
print(f"  variance.csv           — {len(var_df)} rows")
