"""Validation script — simulates all tab logic without Streamlit."""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from sklearn.ensemble import IsolationForest
from scipy import stats

monthly  = pd.read_csv("data/monthly_financials.csv", parse_dates=["date"])
balance  = pd.read_csv("data/balance_sheet.csv")
variance = pd.read_csv("data/variance.csv")
trend    = pd.read_csv("data/revenue_trend.csv")
print("CSVs loaded OK")

# ── calc_ratios ───────────────────────────────────────────────────────────────
def calc_ratios(bs):
    df = bs.copy()
    df["current_ratio"]      = (df["current_assets"] / df["current_liabilities"]).round(2)
    df["quick_ratio"]        = (df["quick_assets"]   / df["current_liabilities"]).round(2)
    df["debt_to_equity"]     = (df["total_debt"]      / df["total_equity"]).round(2)
    df["roa"]                = (df["net_profit"]       / df["total_assets"] * 100).round(2)
    df["working_capital"]    = (df["current_assets"] - df["current_liabilities"]).round(0)
    df["inventory_turnover"] = df["inventory_turnover"].round(2)
    df["asset_turnover"]     = df["asset_turnover"].round(2)
    return df

ratios_df = calc_ratios(balance)
latest_r  = ratios_df.iloc[-1]
print("calc_ratios OK")

# ── health score ──────────────────────────────────────────────────────────────
recent = monthly.tail(3)
cr  = latest_r["current_ratio"]
qr  = latest_r["quick_ratio"]
de  = latest_r["debt_to_equity"]
gpm = (recent["gross_profit"] / recent["revenue"]).mean() * 100
npm = (recent["net_profit"]   / recent["revenue"]).mean() * 100
liq_score  = min(100, max(0, (cr / 2.0) * 50 + (qr / 1.5) * 50))
prof_score = min(100, max(0, gpm * 1.2 + npm * 2.5))
solv_score = min(100, max(0, 100 - (de - 1.0) * 25))
asset_turn = recent["revenue"].mean() / latest_r["total_assets"]
eff_score  = min(100, max(0, asset_turn * 100))
ncf_avg = recent["net_cashflow"].mean()
rev_avg = recent["revenue"].mean()
cf_score = min(100, max(0, 50 + (ncf_avg / rev_avg) * 200))
weights = {"liquidity":0.20,"profitability":0.25,"solvency":0.20,"efficiency":0.20,"cashflow":0.15}
scores  = {"liquidity":liq_score,"profitability":prof_score,"solvency":solv_score,
           "efficiency":eff_score,"cashflow":cf_score}
overall  = sum(scores[k]*weights[k] for k in scores)
category = "Healthy" if overall >= 75 else "Moderate Risk" if overall >= 50 else "High Risk"
print(f"health score OK: {overall:.1f} — {category}")
health = {"scores":scores,"overall":round(overall,1),"category":category,
          "gpm":round(gpm,1),"npm":round(npm,1),"roa":round(latest_r["roa"],1),
          "color":"#dc2626"}

# ── anomaly detection ─────────────────────────────────────────────────────────
features   = monthly[["revenue","operating_expenses","net_cashflow","net_profit"]].copy()
z_scores   = np.abs(stats.zscore(features))
anomaly_df = monthly.copy()
anomaly_df["max_zscore"]  = z_scores.max(axis=1).round(2)
anomaly_df["zscore_flag"] = anomaly_df["max_zscore"] > 2.0
iso = IsolationForest(contamination=0.1, random_state=42)
anomaly_df["iso_flag"] = iso.fit_predict(features) == -1
anomaly_df["anomaly"]  = anomaly_df["zscore_flag"] | anomaly_df["iso_flag"]
n_anomalies = anomaly_df["anomaly"].sum()
print(f"anomaly detection OK — flagged: {n_anomalies}")

# ── Tab 2 efficiency metrics ──────────────────────────────────────────────────
it_delta = round(latest_r["inventory_turnover"] - ratios_df.iloc[-2]["inventory_turnover"], 2)
at_delta = round(latest_r["asset_turnover"]     - ratios_df.iloc[-2]["asset_turnover"],     2)
wc_val   = latest_r["working_capital"]
gpm_calc = monthly["gross_profit"].tail(3).sum() / monthly["revenue"].tail(3).sum() * 100
print(f"Tab2 efficiency OK: IT={latest_r['inventory_turnover']}, AT={latest_r['asset_turnover']}, WC={wc_val:.0f}, GPM={gpm_calc:.1f}%")

# ── Tab 3 variance rows ───────────────────────────────────────────────────────
rev_row = variance[variance["category"] == "Sales Revenue"].iloc[0]
exp_row = variance[variance["category"] == "Operating Expenses"].iloc[0]
pnl_row = variance[variance["category"] == "Net Profit / Loss"].iloc[0]
gp_row  = variance[variance["category"] == "Gross Profit"].iloc[0]
print(f"Tab3 variance rows OK: rev={rev_row['variance']/1000:.1f}B, pnl={pnl_row['variance']/1000:.1f}B")

# variance color column (used in waterfall)
variance["color"] = variance.apply(
    lambda r: "#16a34a" if r["status"] == "Favourable" else "#dc2626", axis=1)
print("variance color OK")

# ── Tab 3 display_var copy (renamed columns) ──────────────────────────────────
display_var = variance.copy()
for col in ["actual_2024","budget_2024","actual_2023"]:
    display_var[col] = display_var[col].apply(lambda x: f"R{x/1000:+.2f}B")
display_var["variance"]     = display_var["variance"].apply(lambda x: f"R{x/1000:+.2f}B")
display_var["variance_pct"] = display_var["variance_pct"].apply(lambda x: f"{x:+.1f}%")
display_var["yoy_change"]   = display_var["yoy_change"].apply(lambda x: f"R{x/1000:+.2f}B")
display_var["yoy_pct"]      = display_var["yoy_pct"].apply(lambda x: f"{x:+.1f}%")
print("display_var formatting OK")

# ── Tab 5 risk logic ──────────────────────────────────────────────────────────
def risk_level(score, thresholds):
    if thresholds[0]: return "HIGH"
    if thresholds[1]: return "MEDIUM"
    return "LOW"

risks = {
    "Liquidity Risk":     risk_level(1, [cr < 1.2, cr < 1.5]),
    "Cash Flow Risk":     risk_level(1, [anomaly_df["net_cashflow"].tail(3).mean() < 0,
                                         anomaly_df["net_cashflow"].tail(3).mean() < 50000]),
    "Solvency Risk":      risk_level(1, [de > 2.5, de > 1.8]),
    "Profitability Risk": risk_level(1, [health["npm"] < 5, health["npm"] < 12]),
    "Anomaly / Fraud":    risk_level(1, [n_anomalies > 5, n_anomalies > 2]),
    "Operational Risk":   risk_level(1, [abs(variance["variance_pct"]).mean() > 25,
                                         abs(variance["variance_pct"]).mean() > 12]),
}
print("Tab5 risks OK:", risks)

# ── Tab 5 forecast ────────────────────────────────────────────────────────────
last_inflow  = monthly["cash_inflow"].tail(3).mean()
last_outflow = monthly["cash_outflow"].tail(3).mean()
forecast_months = pd.date_range(
    start=monthly["date"].max() + pd.DateOffset(months=1), periods=6, freq="MS")
inflow_f  = [last_inflow  * (1 + 0.015*i) * np.random.uniform(0.97, 1.03) for i in range(1,7)]
outflow_f = [last_outflow * (1 + 0.018*i) * np.random.uniform(0.97, 1.03) for i in range(1,7)]
net_f     = [i - o for i, o in zip(inflow_f, outflow_f)]
print(f"Tab5 forecast OK: {len(net_f)} months, net range [{min(net_f):.0f}, {max(net_f):.0f}]")

# ── Tab 1 rolling score ───────────────────────────────────────────────────────
rolling = monthly.copy()
rolling["gpm"] = rolling["gross_profit"] / rolling["revenue"] * 100
rolling["npm"] = rolling["net_profit"]   / rolling["revenue"] * 100
rolling["cf_ratio"] = rolling["net_cashflow"] / rolling["revenue"] * 100
rolling["gpm_norm"] = ((rolling["gpm"] - 15) / 10).clip(0, 1)
rolling["npm_norm"] = ((rolling["npm"] +  5) / 10).clip(0, 1)
rolling["cf_norm"]  = ((rolling["cf_ratio"] + 10) / 15).clip(0, 1)
rolling["rolling_score"] = (
    rolling["gpm_norm"] * 30 +
    rolling["npm_norm"] * 50 +
    rolling["cf_norm"]  * 20
).clip(10, 90).round(1)
print(f"Tab1 rolling score OK: range [{rolling['rolling_score'].min():.1f}, {rolling['rolling_score'].max():.1f}]")

print("\n=== ALL CHECKS PASSED ===")
