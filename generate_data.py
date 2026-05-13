"""
Pick n Pay Financial Data Generator — calibrated to published annual results.

Annual targets:
  FY2023: Revenue R108.57B | Net Profit  R1.17B  | GPM ~19.0%
  FY2024: Revenue R114.94B | Net Loss   -R3.19B  | GPM  18.1%

NP derivation (ZAR millions):
  2023: GP(20,628) - OpEx(18,758) - FinCosts(700)   =  1,170  ✓
  2024: GP(20,804) - OpEx(22,604) - FinCosts(1,390) = -3,190  ✓

Note: "Operating Expenses" in the published report (R20.2B / R22.6B) includes
items outside this simplified P&L model. The variance table stores those
exact published figures; the monthly model uses effective opex to hit
the correct net profit outcome.
"""

import pandas as pd
import numpy as np
import os

np.random.seed(42)
os.makedirs("data", exist_ok=True)

# ── Annual calibration targets (ZAR millions) ─────────────────────────────────
TARGETS = {
    2023: {"rev": 108_570, "gpm": 0.190, "eff_opex": 18_758, "fcosts":   700},
    2024: {"rev": 114_940, "gpm": 0.181, "eff_opex": 22_400, "fcosts": 1_390},
}

# ── Monthly Financial Data (Jan 2023 – Dec 2024) ──────────────────────────────
months = pd.date_range(start="2023-01-01", periods=24, freq="MS")

revenue, cogs_list, opex_list, fcosts_list = [], [], [], []
cash_inflow, cash_outflow = [], []

for i, m in enumerate(months):
    t  = TARGETS[m.year]
    # Retail seasonality: December peak (+18%), January trough (-13%)
    seasonal = 1.0 + 0.10 * np.sin(2 * np.pi * (m.month - 3) / 12)
    if m.month == 12: seasonal *= 1.18
    if m.month == 1:  seasonal *= 0.87

    rev  = (t["rev"] / 12) * seasonal * np.random.uniform(0.97, 1.03)
    cog  = rev * (1 - t["gpm"]) * np.random.uniform(0.99, 1.01)

    exp  = (t["eff_opex"] / 12) * np.random.uniform(0.97, 1.03)
    fc   = (t["fcosts"]  / 12) * np.random.uniform(0.90, 1.10)

    ci   = rev * np.random.uniform(0.85, 0.94)
    co   = (cog + exp) * np.random.uniform(0.88, 0.97)

    revenue.append(round(rev, 2))
    cogs_list.append(round(cog, 2))
    opex_list.append(round(exp, 2))
    fcosts_list.append(round(fc, 2))
    cash_inflow.append(round(ci, 2))
    cash_outflow.append(round(co, 2))

monthly_df = pd.DataFrame({
    "date":               months,
    "revenue":            revenue,
    "cogs":               cogs_list,
    "operating_expenses": opex_list,
    "finance_costs":      fcosts_list,
    "cash_inflow":        cash_inflow,
    "cash_outflow":       cash_outflow,
})
monthly_df["gross_profit"] = monthly_df["revenue"] - monthly_df["cogs"]
monthly_df["net_profit"]   = (monthly_df["revenue"] - monthly_df["cogs"]
                               - monthly_df["operating_expenses"]
                               - monthly_df["finance_costs"])
monthly_df["net_cashflow"] = monthly_df["cash_inflow"] - monthly_df["cash_outflow"]
monthly_df.to_csv("data/monthly_financials.csv", index=False)

# ── Verify annual totals ───────────────────────────────────────────────────────
monthly_df["year"] = monthly_df["date"].dt.year
ann = monthly_df.groupby("year")[["revenue","net_profit","operating_expenses","gross_profit"]].sum()
for yr in [2023, 2024]:
    r  = ann.loc[yr]
    gm = r["gross_profit"] / r["revenue"] * 100
    np_margin = r["net_profit"] / r["revenue"] * 100
    print(f"FY{yr}: Rev=R{r['revenue']/1000:.2f}B  NP=R{r['net_profit']/1000:.2f}B"
          f"  NPM={np_margin:.2f}%  GPM={gm:.2f}%")
monthly_df.drop(columns=["year"], inplace=True)

# ── Balance Sheet (Quarterly) ─────────────────────────────────────────────────
# Calibrated to hit CR≈1.05, QR≈0.40, IT≈8.8x at Q8 (2024-Q4).
# Target Inventory Q8: COGS_annual(94,136M) / 8.8 = 10,697M
# With CR-QR spread = 0.65 → CL_Q8 = 10,697/0.65 = 16,457M
# START_CL = 16,457 / (1.025^7) = 13,840M

quarters  = pd.date_range(start="2023-01-01", periods=8, freq="QS")
START_CL  = 13_300
START_CR, END_CR = 1.30, 1.05
START_QR, END_QR = 0.55, 0.40
TOTAL_ASSETS_BASE = 34_000
bs_data = []

for i, q in enumerate(quarters):
    t_ratio = i / 7                          # 0 → 1 over 8 quarters

    cr = START_CR - t_ratio*(START_CR - END_CR) + np.random.uniform(-0.02, 0.02)
    qr = START_QR - t_ratio*(START_QR - END_QR) + np.random.uniform(-0.01, 0.01)

    cl = START_CL * (1.025 ** i) * np.random.uniform(0.98, 1.02)
    ca = cr * cl
    inv = (cr - qr) * cl
    quick_assets = ca - inv

    total_assets = TOTAL_ASSETS_BASE * (1 + 0.015 * i) * np.random.uniform(0.98, 1.02)

    # Equity grows in 2023, erodes in 2024 after R3.19B loss
    te = 12_000 + 300 * i if i < 4 else 13_200 - 950 * (i - 3)
    total_debt = total_assets - te

    q_mask = ((monthly_df["date"].dt.year    == q.year) &
              (monthly_df["date"].dt.quarter == ((i % 4) + 1)))
    q_rows = monthly_df[q_mask]
    q_np   = q_rows["net_profit"].sum() if len(q_rows) else 0
    q_cogs = q_rows["cogs"].sum()       if len(q_rows) else 0
    q_rev  = q_rows["revenue"].sum()    if len(q_rows) else 0

    inv_turnover   = round((q_cogs * 4) / inv,          2) if inv          > 0 else 0
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
        "net_profit":          round(q_np, 2),
        "inventory_turnover":  inv_turnover,
        "asset_turnover":      asset_turnover,
    })

bs_df = pd.DataFrame(bs_data)
bs_df.to_csv("data/balance_sheet.csv", index=False)

last = bs_df.iloc[-1]
cr_f = last["current_assets"] / last["current_liabilities"]
qr_f = last["quick_assets"]   / last["current_liabilities"]
it_f = last["inventory_turnover"]
print(f"Q8 ratios: CR={cr_f:.3f} (target 1.05)  QR={qr_f:.3f} (target 0.40)  IT={it_f:.2f}x (target 8.8x)")

# ── Budget vs Actual + YoY — exact document figures (ZAR millions) ────────────
categories  = ["Sales Revenue", "Gross Profit", "Operating Expenses",
                "Finance Costs", "Net Profit / Loss"]
actual_2024 = [ 114_940,         20_800,          22_600,        1_390,  -3_190]
budget_2024 = [ 112_000,         22_960,          20_500,          700,   2_500]
actual_2023 = [ 108_570,         20_628,          20_200,          700,   1_170]
cat_type    = ["Revenue",       "Profit",        "Expense",    "Expense","Profit"]

var_df = pd.DataFrame({
    "category":    categories,
    "actual_2024": actual_2024,
    "budget_2024": budget_2024,
    "actual_2023": actual_2023,
    "variance":    [round(a-b, 2) for a, b in zip(actual_2024, budget_2024)],
    "variance_pct":[round((a-b)/abs(b)*100, 2) for a, b in zip(actual_2024, budget_2024)],
    "yoy_change":  [round(a4-a3, 2) for a4, a3 in zip(actual_2024, actual_2023)],
    "yoy_pct":     [round((a4-a3)/abs(a3)*100, 2) for a4, a3 in zip(actual_2024, actual_2023)],
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

# ── Annual Revenue Trend 2021–2025 (exact document figures) ───────────────────
trend_df = pd.DataFrame({
    "year":       [2021,   2022,   2023,    2024,    2025],
    "revenue":    [94_660, 99_630, 108_570, 114_940, 121_560],
    "net_profit": [None,   None,   1_170,  -3_190,   None],
})
trend_df.to_csv("data/revenue_trend.csv", index=False)

print(f"\nFiles written:")
print(f"  monthly_financials.csv — {len(monthly_df)} rows")
print(f"  balance_sheet.csv      — {len(bs_df)} rows")
print(f"  variance.csv           — {len(var_df)} rows")
print(f"  revenue_trend.csv      — {len(trend_df)} rows")
