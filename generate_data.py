"""
Pick n Pay Financial Data Generator
Based on real published annual results:
  FY2023: Revenue R108.57B | Net Profit R1.17B  | OpEx R20.2B | GPM ~19.0%
  FY2024: Revenue R114.94B | Net Loss  -R3.19B  | OpEx R22.6B | GPM  18.1%
All values in ZAR millions unless noted.
"""

import pandas as pd
import numpy as np
import os

np.random.seed(42)
os.makedirs("data", exist_ok=True)

# ── Monthly Financial Data (Jan 2023 – Dec 2024) ──────────────────────────────
months = pd.date_range(start="2023-01-01", periods=24, freq="MS")

# Annual targets (ZAR millions)
ANNUAL_REVENUE    = {2023: 108_570, 2024: 114_940}
ANNUAL_OPEX       = {2023: 20_200,  2024: 22_600}
ANNUAL_FIN_COSTS  = {2023: 700,     2024: 1_390}   # finance & other below-line
GROSS_MARGIN      = {2023: 0.190,   2024: 0.181}

revenue, expenses, fin_costs_list = [], [], []
cogs_list, cash_inflow, cash_outflow = [], [], []

for i, m in enumerate(months):
    yr = m.year
    seasonal = 1.0 + 0.10 * np.sin(2 * np.pi * (m.month - 3) / 12)
    if m.month == 12: seasonal *= 1.18   # festive season
    if m.month == 1:  seasonal *= 0.87   # January slowdown

    rev = (ANNUAL_REVENUE[yr] / 12) * seasonal * np.random.uniform(0.97, 1.03)
    cog = rev * (1 - GROSS_MARGIN[yr]) * np.random.uniform(0.99, 1.01)

    base_exp = ANNUAL_OPEX[yr] / 12
    # Cost spike Q3 2024 (Jul–Sep) — fuel/logistics/load-shedding pressures
    if yr == 2024 and m.month in [7, 8, 9]:
        exp = base_exp * 1.20 * np.random.uniform(0.98, 1.02)
    else:
        exp = base_exp * seasonal * 0.82 * np.random.uniform(0.97, 1.03)

    fc = (ANNUAL_FIN_COSTS[yr] / 12) * np.random.uniform(0.90, 1.10)

    ci = rev  * np.random.uniform(0.85, 0.94)   # retail cash lag
    co = (cog + exp) * np.random.uniform(0.88, 0.97)

    revenue.append(round(rev, 2))
    cogs_list.append(round(cog, 2))
    expenses.append(round(exp, 2))
    fin_costs_list.append(round(fc, 2))
    cash_inflow.append(round(ci, 2))
    cash_outflow.append(round(co, 2))

monthly_df = pd.DataFrame({
    "date":               months,
    "revenue":            revenue,
    "cogs":               cogs_list,
    "operating_expenses": expenses,
    "finance_costs":      fin_costs_list,
    "cash_inflow":        cash_inflow,
    "cash_outflow":       cash_outflow,
})
monthly_df["gross_profit"] = monthly_df["revenue"] - monthly_df["cogs"]
monthly_df["net_profit"]   = (monthly_df["revenue"] - monthly_df["cogs"]
                               - monthly_df["operating_expenses"]
                               - monthly_df["finance_costs"])
monthly_df["net_cashflow"] = monthly_df["cash_inflow"] - monthly_df["cash_outflow"]
monthly_df.to_csv("data/monthly_financials.csv", index=False)

# ── Balance Sheet (Quarterly) ─────────────────────────────────────────────────
# Targets derived from published 2024 ratios:
#   Current Ratio 1.05 | Quick Ratio 0.40 | Inventory Turnover ~8.8x
#   Starting 2023 Q1 at healthier levels, deteriorating through 2024

quarters = pd.date_range(start="2023-01-01", periods=8, freq="QS")
bs_data = []

START_CL         = 15_000   # R15B current liabilities Q1 2023
START_CR, END_CR = 1.30,  1.05
START_QR, END_QR = 0.55,  0.40
TOTAL_ASSETS_BASE = 34_000

for i, q in enumerate(quarters):
    t = i / 7
    cr = START_CR - t * (START_CR - END_CR) + np.random.uniform(-0.02, 0.02)
    qr = START_QR - t * (START_QR - END_QR) + np.random.uniform(-0.01, 0.01)

    cl = START_CL * (1 + 0.025 * i) * np.random.uniform(0.98, 1.02)
    ca = cr * cl
    inv = (cr - qr) * cl
    quick_assets = ca - inv

    total_assets = TOTAL_ASSETS_BASE * (1 + 0.015 * i) * np.random.uniform(0.98, 1.02)

    # Equity: growing slightly in 2023, eroding in 2024 after R3.19B loss
    te = 12_000 + 300 * i if i < 4 else 13_200 - 950 * (i - 3)
    total_debt = total_assets - te

    # Pull matching quarterly actuals from monthly data
    q_mask = ((monthly_df["date"].dt.year == q.year) &
              (monthly_df["date"].dt.quarter == ((i % 4) + 1)))
    q_rows = monthly_df[q_mask]
    q_net_profit = q_rows["net_profit"].sum() if len(q_rows) else 0
    q_cogs       = q_rows["cogs"].sum()       if len(q_rows) else 0
    q_rev        = q_rows["revenue"].sum()    if len(q_rows) else 0

    inv_turnover   = round((q_cogs * 4) / inv, 2)        if inv > 0          else 0
    asset_turnover = round((q_rev  * 4) / total_assets, 2) if total_assets > 0 else 0

    bs_data.append({
        "quarter":             q.strftime("%Y-Q") + str((i % 4) + 1),
        "current_assets":      round(ca, 2),
        "current_liabilities": round(cl, 2),
        "inventory":           round(inv, 2),
        "quick_assets":        round(quick_assets, 2),
        "total_assets":        round(total_assets, 2),
        "total_equity":        round(te, 2),
        "total_debt":          round(total_debt, 2),
        "net_profit":          round(q_net_profit, 2),
        "inventory_turnover":  inv_turnover,
        "asset_turnover":      asset_turnover,
    })

bs_df = pd.DataFrame(bs_data)
bs_df.to_csv("data/balance_sheet.csv", index=False)

# ── Budget vs Actual + Year-on-Year (ZAR millions) ───────────────────────────
categories   = ["Sales Revenue", "Gross Profit", "Operating Expenses", "Finance Costs", "Net Profit / Loss"]
actual_2024  = [ 114_940,         20_800,          22_600,               1_390,           -3_190]
budget_2024  = [ 112_000,         22_960,          20_500,                 700,            2_500]  # pre-crisis budget
actual_2023  = [ 108_570,         20_628,          20_200,                 700,            1_170]
cat_type     = ["Revenue",       "Profit",        "Expense",            "Expense",       "Profit"]

var_df = pd.DataFrame({
    "category":    categories,
    "actual_2024": actual_2024,
    "budget_2024": budget_2024,
    "actual_2023": actual_2023,
    "variance":    [round(a - b, 2) for a, b in zip(actual_2024, budget_2024)],
    "variance_pct":[round((a - b) / abs(b) * 100, 2) for a, b in zip(actual_2024, budget_2024)],
    "yoy_change":  [round(a4 - a3, 2) for a4, a3 in zip(actual_2024, actual_2023)],
    "yoy_pct":     [round((a4 - a3) / abs(a3) * 100, 2) for a4, a3 in zip(actual_2024, actual_2023)],
    "type":        cat_type,
})
var_df["status"] = var_df.apply(
    lambda r: "Favourable"
    if (r["type"] == "Revenue" and r["variance"] > 0)
    or (r["type"] == "Expense" and r["variance"] < 0)
    or (r["type"] == "Profit"  and r["variance"] > 0)
    else "Unfavourable", axis=1
)
var_df.to_csv("data/variance.csv", index=False)

# ── Annual Revenue Trend (2021–2025) ─────────────────────────────────────────
trend_df = pd.DataFrame({
    "year":       [2021,   2022,   2023,    2024,    2025],
    "revenue":    [94_660, 99_630, 108_570, 114_940, 121_560],
    "net_profit": [None,   None,   1_170,  -3_190,   None],
})
trend_df.to_csv("data/revenue_trend.csv", index=False)

print("Pick n Pay data generated successfully.")
print(f"  monthly_financials.csv — {len(monthly_df)} rows")
print(f"  balance_sheet.csv      — {len(bs_df)} rows")
print(f"  variance.csv           — {len(var_df)} rows")
print(f"  revenue_trend.csv      — {len(trend_df)} rows")
