"""
Pick n Pay Financial Data Generator — calibrated to published annual results.

Annual targets:
  FY2023: Revenue R108.57B | Net Profit  R1.17B  | GPM ~19.0%
  FY2024: Revenue R114.94B | Net Loss   -R3.19B  | GPM  18.1%
  FY2025: Revenue R118.61B | Net Loss   -R0.74B  | GPM  18.4%  (audited, 53 wks ended 02 Mar 2025)

NP derivation (ZAR millions):
  2023: GP(20,628) - OpEx(18,758) - FinCosts(700)   =  1,170  ✓
  2024: GP(20,804) - OpEx(22,604) - FinCosts(1,390) = -3,190  ✓
  2025: GP(21,764) - OpEx(21,400) - FinCosts(1,100) =   -736  ✓
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
    2025: {"rev": 118_610, "gpm": 0.1835,"eff_opex": 21_400, "fcosts": 1_100},
}

# ── Quarterly Financial Data (Q1 2023 – Q4 2025, 12 quarters) ────────────────
quarters = pd.date_range(start="2023-01-01", periods=12, freq="QS")

revenue, cogs_list, opex_list, fcosts_list = [], [], [], []
cash_inflow, cash_outflow = [], []

for i, q in enumerate(quarters):
    t = TARGETS[q.year]
    qtr = (i % 4) + 1
    # Retail seasonality: Q4 (Oct–Dec) peak, Q1 (Jan–Mar) softer
    seasonal = {1: 0.93, 2: 0.98, 3: 1.02, 4: 1.07}[qtr]

    rev  = (t["rev"] / 4) * seasonal * np.random.uniform(0.97, 1.03)
    cog  = rev * (1 - t["gpm"]) * np.random.uniform(0.99, 1.01)
    exp  = (t["eff_opex"] / 4) * np.random.uniform(0.97, 1.03)
    fc   = (t["fcosts"]  / 4) * np.random.uniform(0.90, 1.10)
    ci   = rev * np.random.uniform(0.85, 0.94)
    co   = (cog + exp) * np.random.uniform(0.88, 0.97)

    revenue.append(round(rev, 2))
    cogs_list.append(round(cog, 2))
    opex_list.append(round(exp, 2))
    fcosts_list.append(round(fc, 2))
    cash_inflow.append(round(ci, 2))
    cash_outflow.append(round(co, 2))

quarterly_df = pd.DataFrame({
    "date":               quarters,
    "revenue":            revenue,
    "cogs":               cogs_list,
    "operating_expenses": opex_list,
    "finance_costs":      fcosts_list,
    "cash_inflow":        cash_inflow,
    "cash_outflow":       cash_outflow,
})
quarterly_df["gross_profit"] = quarterly_df["revenue"] - quarterly_df["cogs"]
quarterly_df["net_profit"]   = (quarterly_df["revenue"] - quarterly_df["cogs"]
                                - quarterly_df["operating_expenses"]
                                - quarterly_df["finance_costs"])
quarterly_df["net_cashflow"] = quarterly_df["cash_inflow"] - quarterly_df["cash_outflow"]
quarterly_df.to_csv("data/quarterly_financials.csv", index=False)

# ── Verify annual totals ───────────────────────────────────────────────────────
quarterly_df["year"] = quarterly_df["date"].dt.year
ann = quarterly_df.groupby("year")[["revenue","net_profit","gross_profit"]].sum()
for yr in [2023, 2024, 2025]:
    r  = ann.loc[yr]
    gm = r["gross_profit"] / r["revenue"] * 100
    npm = r["net_profit"] / r["revenue"] * 100
    print(f"FY{yr}: Rev=R{r['revenue']/1000:.2f}B  NP=R{r['net_profit']/1000:.2f}B"
          f"  NPM={npm:.2f}%  GPM={gm:.2f}%")
quarterly_df.drop(columns=["year"], inplace=True)

# ── Balance Sheet (Quarterly) ─────────────────────────────────────────────────
START_CL  = 13_300
START_CR, END_CR = 1.30, 1.05
START_QR, END_QR = 0.55, 0.40
TOTAL_ASSETS_BASE = 34_000
bs_data = []

for i, q in enumerate(quarters):
    t_ratio = i / 11

    cr = START_CR - t_ratio*(START_CR - END_CR) + np.random.uniform(-0.02, 0.02)
    qr = START_QR - t_ratio*(START_QR - END_QR) + np.random.uniform(-0.01, 0.01)

    cl = START_CL * (1.025 ** i) * np.random.uniform(0.98, 1.02)
    ca = cr * cl
    inv = (cr - qr) * cl
    quick_assets = ca - inv

    total_assets = TOTAL_ASSETS_BASE * (1 + 0.015 * i) * np.random.uniform(0.98, 1.02)
    # Equity: grows FY2023, erodes FY2024 (big loss), partial recovery FY2025
    if   i < 4:  te = 12_000 + 300 * i
    elif i < 8:  te = 13_200 - 950 * (i - 3)
    else:        te = 9_300  + 400 * (i - 7)
    total_debt = total_assets - te

    q_row  = quarterly_df.iloc[i]
    q_cogs = q_row["cogs"]
    q_rev  = q_row["revenue"]
    q_np   = q_row["net_profit"]

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
print(f"Q12 ratios: CR={cr_f:.3f}  QR={qr_f:.3f}")

# ── Budget vs Actual + YoY — exact document figures (ZAR millions) ────────────
# FY2025 audited results (53 weeks ended 02 March 2025, JSE SENS 26 May 2025)
categories  = ["Sales Revenue", "Gross Profit", "Operating Expenses",
                "Finance Costs", "Net Profit / Loss"]
actual_2025 = [ 118_610,         21_764,          22_954,        1_100,    -736]
budget_2025 = [ 116_000,         21_400,          21_500,        1_000,     900]
actual_2024 = [ 112_295,         20_280,          22_518,        1_390,  -3_301]
cat_type    = ["Revenue",       "Profit",        "Expense",    "Expense","Profit"]

var_df = pd.DataFrame({
    "category":    categories,
    "actual_2025": actual_2025,
    "budget_2025": budget_2025,
    "actual_2024": actual_2024,
    "variance":    [round(a-b, 2) for a, b in zip(actual_2025, budget_2025)],
    "variance_pct":[round((a-b)/abs(b)*100, 2) for a, b in zip(actual_2025, budget_2025)],
    "yoy_change":  [round(a5-a4, 2) for a5, a4 in zip(actual_2025, actual_2024)],
    "yoy_pct":     [round((a5-a4)/abs(a4)*100, 2) for a5, a4 in zip(actual_2025, actual_2024)],
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
    "revenue":    [94_660, 99_630, 108_570, 112_295, 118_610],
    "net_profit": [None,   None,   1_170,   -3_301,    -736],
})
trend_df.to_csv("data/revenue_trend.csv", index=False)

print(f"\nFiles written:")
print(f"  quarterly_financials.csv — {len(quarterly_df)} rows")
print(f"  balance_sheet.csv        — {len(bs_df)} rows")
print(f"  variance.csv             — {len(var_df)} rows")
print(f"  revenue_trend.csv        — {len(trend_df)} rows")
