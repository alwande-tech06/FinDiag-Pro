"""
FinDiag Pro — AI-Enhanced Organisational Financial Health Diagnostics
Pathway B: Performance Monitoring & Early Warning Systems
Durban University of Technology — Managerial Finance PBL
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.ensemble import IsolationForest
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FinDiag Pro — Financial Health Diagnostics",
    #page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Tailwind CSS CDN + Streamlit overrides ────────────────────────────────────
st.markdown("""
<link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
<style>
    div[data-testid="metric-container"] {
        background: #f8fafc; border-radius: 10px;
        padding: 0.75rem; border: 1px solid #e2e8f0;
    }
    .stMarkdown > div { padding-top: 0 !important; }
</style>
""", unsafe_allow_html=True)

# ── Data loaders ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    monthly  = pd.read_csv("data/monthly_financials.csv", parse_dates=["date"])
    balance  = pd.read_csv("data/balance_sheet.csv")
    variance = pd.read_csv("data/variance.csv")
    trend    = pd.read_csv("data/revenue_trend.csv")
    return monthly, balance, variance, trend

monthly_df, balance_df, variance_df, trend_df = load_data()

# ── Financial ratio calculations ─────────────────────────────────────────────
def calc_ratios(bs: pd.DataFrame) -> pd.DataFrame:
    df = bs.copy()
    df["current_ratio"]       = (df["current_assets"] / df["current_liabilities"]).round(2)
    df["quick_ratio"]         = (df["quick_assets"]   / df["current_liabilities"]).round(2)
    df["debt_to_equity"]      = (df["total_debt"]      / df["total_equity"]).round(2)
    df["roa"]                 = (df["net_profit"]       / df["total_assets"] * 100).round(2)
    df["working_capital"]     = (df["current_assets"] - df["current_liabilities"]).round(0)
    # Efficiency ratios (pre-calculated in CSV from annualised quarterly figures)
    df["inventory_turnover"]  = df["inventory_turnover"].round(2)
    df["asset_turnover"]      = df["asset_turnover"].round(2)
    return df

# ── Health score engine ───────────────────────────────────────────────────────
def compute_health_score(monthly: pd.DataFrame, ratios: pd.DataFrame) -> dict:
    latest = ratios.iloc[-1]
    recent = monthly.tail(3)

    # Liquidity (20%)
    cr = latest["current_ratio"]
    qr = latest["quick_ratio"]
    liq_score = min(100, max(0, (cr / 2.0) * 50 + (qr / 1.5) * 50))

    # Profitability (25%)
    gpm = (recent["gross_profit"] / recent["revenue"]).mean() * 100
    npm = (recent["net_profit"]   / recent["revenue"]).mean() * 100
    prof_score = min(100, max(0, gpm * 1.2 + npm * 2.5))

    # Solvency (20%)
    de = latest["debt_to_equity"]
    solv_score = min(100, max(0, 100 - (de - 1.0) * 25))

    # Efficiency (20%)
    asset_turn = (recent["revenue"].mean() / latest["total_assets"])
    eff_score  = min(100, max(0, asset_turn * 100))

    # Cash Flow (15%)
    ncf_avg = recent["net_cashflow"].mean()
    rev_avg = recent["revenue"].mean()
    cf_score = min(100, max(0, 50 + (ncf_avg / rev_avg) * 200))

    weights = {"liquidity": 0.20, "profitability": 0.25, "solvency": 0.20, "efficiency": 0.20, "cashflow": 0.15}
    scores  = {"liquidity": liq_score, "profitability": prof_score, "solvency": solv_score,
               "efficiency": eff_score, "cashflow": cf_score}
    overall = sum(scores[k] * weights[k] for k in scores)

    if overall >= 75:
        category, color = "Healthy", "#16a34a"
    elif overall >= 50:
        category, color = "Moderate Risk", "#d97706"
    else:
        category, color = "High Risk", "#dc2626"

    return {"scores": scores, "weights": weights, "overall": round(overall, 1),
            "category": category, "color": color,
            "gpm": round(gpm, 1), "npm": round(npm, 1), "roa": round(latest["roa"], 1)}

# ── Anomaly detection ─────────────────────────────────────────────────────────
def detect_anomalies(monthly: pd.DataFrame):
    features = monthly[["revenue", "operating_expenses", "net_cashflow", "net_profit"]].copy()

    # Z-score
    z_scores = np.abs(stats.zscore(features))
    monthly = monthly.copy()
    monthly["max_zscore"] = z_scores.max(axis=1).round(2)
    monthly["zscore_flag"] = monthly["max_zscore"] > 2.0

    # Isolation Forest
    iso = IsolationForest(contamination=0.1, random_state=42)
    monthly["iso_flag"] = iso.fit_predict(features) == -1

    monthly["anomaly"] = monthly["zscore_flag"] | monthly["iso_flag"]
    return monthly

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Dashboard Controls")
    st.markdown("---")

    uploaded = st.file_uploader("📂 Upload Financial CSV", type=["csv", "xlsx"],
                                help="Upload your own financial data (monthly format)")
    if uploaded:
        try:
            if uploaded.name.endswith(".xlsx"):
                monthly_df = pd.read_excel(uploaded, parse_dates=["date"])
            else:
                monthly_df = pd.read_csv(uploaded, parse_dates=["date"])
            st.success("✅ Custom data loaded!")
        except Exception as e:
            st.error(f"Error reading file: {e}")

    st.markdown("---")
    period_options = ["All periods", "Last 12 months", "Last 6 months"]
    selected_period = st.selectbox("📅 Analysis Period", period_options)
    if selected_period == "Last 12 months":
        monthly_df = monthly_df.tail(12).reset_index(drop=True)
    elif selected_period == "Last 6 months":
        monthly_df = monthly_df.tail(6).reset_index(drop=True)

    zscore_threshold = st.slider("🔍 Z-Score Anomaly Threshold", 1.5, 3.5, 2.0, 0.1)

    st.markdown("---")
    st.markdown("**Organisation**")
    org_name = st.text_input("Name", value="Pick n Pay Stores Ltd")
    fiscal_year = st.selectbox("Fiscal Year", ["FY 2024", "FY 2023", "FY 2022"])

    st.markdown("---")
    st.markdown("**About FinDiag Pro**")
    st.caption("AI-Enhanced Financial Health Diagnostics\nDurban University of Technology")

# ── Computed data ─────────────────────────────────────────────────────────────
ratios_df   = calc_ratios(balance_df)
health      = compute_health_score(monthly_df, ratios_df)
anomaly_df  = detect_anomalies(monthly_df)
latest_r    = ratios_df.iloc[-1]

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="bg-gradient-to-r from-blue-900 to-blue-600 rounded-2xl px-8 py-5 mb-4 shadow-lg flex items-center justify-between">
  <div>
    <h1 class="text-white text-2xl font-bold tracking-tight m-0 p-0">&#128202; FinDiag Pro &mdash; Financial Health Diagnostics</h1>
    <p class="text-blue-200 text-sm mt-1 m-0">{org_name} &nbsp;&bull;&nbsp; {fiscal_year} &nbsp;&bull;&nbsp; AI-Enhanced Early-Warning &amp; Decision Support System</p>
  </div>
  <div class="text-right">
    <span class="bg-blue-800 text-blue-100 text-xs font-semibold px-3 py-1 rounded-full uppercase tracking-wider">Live Dashboard</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    " Health Scorecard",
    " Ratio Analysis",
    " Variance Monitor",
    " Anomaly Detection",
    " Early Warning"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — HEALTH SCORECARD
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Financial Health Scorecard")
    st.caption("Composite financial health index based on liquidity, profitability, solvency, efficiency and cash flow.")

    # Top KPIs
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Overall Score", f"{health['overall']}/100",
                delta=f"{'▼' if health['overall'] < 70 else '▲'} Q-o-Q")
    col2.metric("Risk Category", health["category"])
    col3.metric("Gross Profit Margin", f"{health['gpm']}%")
    col4.metric("Net Profit Margin", f"{health['npm']}%")
    col5.metric("Return on Assets", f"{health['roa']}%")

    st.markdown("---")
    c1, c2 = st.columns([1, 1.6])

    # Gauge chart
    with c1:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=health["overall"],
            title={"text": "Overall Financial Health Score", "font": {"size": 14}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar":  {"color": health["color"], "thickness": 0.25},
                "steps": [
                    {"range": [0,  50], "color": "#fef2f2"},
                    {"range": [50, 75], "color": "#fffbeb"},
                    {"range": [75,100], "color": "#f0fdf4"},
                ],
                "threshold": {"line": {"color": health["color"], "width": 3},
                              "thickness": 0.75, "value": health["overall"]},
            },
            number={"font": {"size": 40, "color": health["color"]}},
        ))
        fig_gauge.update_layout(height=280, margin=dict(t=50, b=10, l=20, r=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

        badge = {
            "Healthy":       "bg-green-100 text-green-800 border border-green-300",
            "Moderate Risk": "bg-yellow-100 text-yellow-800 border border-yellow-300",
            "High Risk":     "bg-red-100 text-red-800 border border-red-300",
        }[health["category"]]
        st.markdown(f"""
        <div class="text-center mt-1">
          <span class="{badge} px-4 py-1 rounded-full text-base font-semibold">{health["category"]}</span>
        </div>""", unsafe_allow_html=True)

    # Radar / dimension breakdown
    with c2:
        dims   = list(health["scores"].keys())
        vals   = [round(v, 1) for v in health["scores"].values()]
        labels = ["Liquidity","Profitability","Solvency","Efficiency","Cash Flow"]

        fig_radar = go.Figure(go.Scatterpolar(
            r=vals + [vals[0]],
            theta=labels + [labels[0]],
            fill="toself",
            fillcolor="rgba(24,95,165,0.15)",
            line=dict(color="#185FA5", width=2),
            marker=dict(size=6, color="#185FA5"),
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0,100],
                                       tickfont=dict(size=10))),
            showlegend=False, height=280,
            margin=dict(t=30, b=10, l=40, r=40),
            title=dict(text="Dimension Breakdown", font=dict(size=13), x=0.5),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        # Dimension bars (Tailwind)
        for lbl, val in zip(labels, vals):
            bar_cls  = "bg-green-500" if val >= 70 else "bg-yellow-400" if val >= 50 else "bg-red-500"
            text_cls = "text-green-700" if val >= 70 else "text-yellow-700" if val >= 50 else "text-red-700"
            st.markdown(f"""
            <div class="flex items-center gap-2 mb-1 text-xs">
              <span class="w-24 text-gray-500 font-medium">{lbl}</span>
              <div class="flex-1 bg-gray-100 rounded-full h-2">
                <div class="{bar_cls} h-2 rounded-full" style="width:{val}%"></div>
              </div>
              <span class="w-8 text-right font-bold {text_cls}">{val}</span>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Health score trend
    if len(monthly_df) >= 6:
        rolling = monthly_df.copy()
        rolling["gpm"] = rolling["gross_profit"] / rolling["revenue"] * 100
        rolling["npm"] = rolling["net_profit"]   / rolling["revenue"] * 100
        rolling["cf_ratio"] = rolling["net_cashflow"] / rolling["revenue"] * 100
        # Normalised scoring calibrated for retail margins (PnP GPM 18-20%, NPM -5% to +5%)
        rolling["gpm_norm"] = ((rolling["gpm"] - 15) / 10).clip(0, 1)        # 15-25% → 0-1
        rolling["npm_norm"] = ((rolling["npm"] +  5) / 10).clip(0, 1)        # -5–+5% → 0-1
        rolling["cf_norm"]  = ((rolling["cf_ratio"] + 10) / 15).clip(0, 1)   # -10–+5% → 0-1
        rolling["rolling_score"] = (
            rolling["gpm_norm"] * 30 +   # up to 30 pts — gross efficiency
            rolling["npm_norm"] * 50 +   # up to 50 pts — bottom-line health
            rolling["cf_norm"]  * 20     # up to 20 pts — cash generation
        ).clip(10, 90).round(1)

        fig_trend = px.line(rolling, x="date", y="rolling_score",
                            title="Financial Health Score Trend",
                            labels={"rolling_score": "Score", "date": "Month"},
                            color_discrete_sequence=["#185FA5"])
        fig_trend.add_hline(y=75, line_dash="dash", line_color="#16a34a",
                            annotation_text="Healthy ≥ 75")
        fig_trend.add_hline(y=50, line_dash="dash", line_color="#dc2626",
                            annotation_text="High Risk < 50")
        fig_trend.update_traces(line_width=2.5)
        fig_trend.update_layout(height=280, margin=dict(t=50,b=20,l=10,r=10))
        st.plotly_chart(fig_trend, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — RATIO ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Automated Ratio Analysis Dashboard")
    st.caption("Key financial ratios calculated from balance sheet data with trend visualisation and benchmark comparison.")

    # Key ratio KPI cards — row 1
    c1, c2, c3, c4 = st.columns(4)
    cr_delta  = round(latest_r["current_ratio"]  - ratios_df.iloc[-2]["current_ratio"],  2)
    qr_delta  = round(latest_r["quick_ratio"]    - ratios_df.iloc[-2]["quick_ratio"],    2)
    de_delta  = round(latest_r["debt_to_equity"] - ratios_df.iloc[-2]["debt_to_equity"], 2)
    roa_delta = round(latest_r["roa"]            - ratios_df.iloc[-2]["roa"],            2)

    c1.metric("Current Ratio",   f"{latest_r['current_ratio']:.2f}",  f"{cr_delta:+.2f} QoQ",
              delta_color="normal")
    c2.metric("Quick Ratio",     f"{latest_r['quick_ratio']:.2f}",    f"{qr_delta:+.2f} QoQ",
              delta_color="normal")
    c3.metric("Debt-to-Equity",  f"{latest_r['debt_to_equity']:.2f}", f"{de_delta:+.2f} QoQ",
              delta_color="inverse")
    c4.metric("ROA",             f"{latest_r['roa']:.1f}%",           f"{roa_delta:+.1f}% QoQ",
              delta_color="normal")

    # Efficiency ratios — row 2
    c5, c6, c7, c8 = st.columns(4)
    it_delta  = round(latest_r["inventory_turnover"] - ratios_df.iloc[-2]["inventory_turnover"], 2)
    at_delta  = round(latest_r["asset_turnover"]     - ratios_df.iloc[-2]["asset_turnover"],     2)
    wc_val    = latest_r["working_capital"]

    c5.metric("Inventory Turnover", f"{latest_r['inventory_turnover']:.2f}x", f"{it_delta:+.2f} QoQ",
              delta_color="normal")
    c6.metric("Asset Turnover",     f"{latest_r['asset_turnover']:.2f}x",     f"{at_delta:+.2f} QoQ",
              delta_color="normal")
    c7.metric("Working Capital",    f"R{wc_val/1e3:,.1f}B",
              delta="Declining" if wc_val < 3_000 else "Stable",
              delta_color="inverse" if wc_val < 3_000 else "normal")
    c8.metric("Gross Profit Margin",
              f"{(monthly_df['gross_profit'].tail(3).sum()/monthly_df['revenue'].tail(3).sum()*100):.1f}%",
              "18.1% FY2024 actual")

    st.markdown("---")
    c1, c2 = st.columns(2)

    with c1:
        # Liquidity ratios trend
        fig_liq = go.Figure()
        fig_liq.add_trace(go.Scatter(x=ratios_df["quarter"], y=ratios_df["current_ratio"],
                                     name="Current Ratio", mode="lines+markers",
                                     line=dict(color="#185FA5", width=2.5), marker=dict(size=7)))
        fig_liq.add_trace(go.Scatter(x=ratios_df["quarter"], y=ratios_df["quick_ratio"],
                                     name="Quick Ratio", mode="lines+markers",
                                     line=dict(color="#1D9E75", width=2.5, dash="dash"),
                                     marker=dict(size=7)))
        fig_liq.add_hline(y=1.5, line_dash="dot", line_color="#dc2626",
                          annotation_text="Current Ratio benchmark")
        fig_liq.add_hline(y=1.0, line_dash="dot", line_color="#f59e0b",
                          annotation_text="Quick Ratio minimum")
        fig_liq.update_layout(title="Liquidity Ratios — Quarterly Trend",
                              height=280, margin=dict(t=50,b=20),
                              legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig_liq, use_container_width=True)

    with c2:
        # Debt-to-equity trend
        fig_de = go.Figure()
        fig_de.add_trace(go.Bar(x=ratios_df["quarter"], y=ratios_df["debt_to_equity"],
                                name="Debt-to-Equity",
                                marker_color=["#dc2626" if v > 2 else "#f59e0b" if v > 1.5
                                              else "#16a34a" for v in ratios_df["debt_to_equity"]]))
        fig_de.add_hline(y=2.0, line_dash="dash", line_color="#dc2626",
                         annotation_text="High risk threshold")
        fig_de.update_layout(title="Debt-to-Equity Ratio", height=280,
                             margin=dict(t=50,b=20), showlegend=False)
        st.plotly_chart(fig_de, use_container_width=True)

    # Profit margins
    monthly_df["gpm"] = (monthly_df["gross_profit"] / monthly_df["revenue"] * 100).round(2)
    monthly_df["npm"] = (monthly_df["net_profit"]   / monthly_df["revenue"] * 100).round(2)

    fig_margin = make_subplots(specs=[[{"secondary_y": False}]])
    fig_margin.add_trace(go.Scatter(x=monthly_df["date"], y=monthly_df["gpm"],
                                    name="Gross Profit Margin %",
                                    line=dict(color="#1D9E75", width=2.5),
                                    fill="tozeroy", fillcolor="rgba(29,158,117,0.07)"))
    fig_margin.add_trace(go.Scatter(x=monthly_df["date"], y=monthly_df["npm"],
                                    name="Net Profit Margin %",
                                    line=dict(color="#185FA5", width=2.5),
                                    fill="tozeroy", fillcolor="rgba(24,95,165,0.07)"))
    fig_margin.update_layout(title="Profit Margin Trends — Monthly", height=280,
                             margin=dict(t=50,b=20),
                             yaxis_title="Margin %",
                             legend=dict(orientation="h", y=-0.3))
    st.plotly_chart(fig_margin, use_container_width=True)

    # Efficiency ratio trend charts
    c1, c2 = st.columns(2)
    with c1:
        fig_it = go.Figure()
        fig_it.add_trace(go.Scatter(x=ratios_df["quarter"], y=ratios_df["inventory_turnover"],
                                    name="Inventory Turnover", mode="lines+markers",
                                    line=dict(color="#7C3AED", width=2.5), marker=dict(size=7)))
        fig_it.add_hline(y=8.0, line_dash="dot", line_color="#dc2626",
                         annotation_text="Industry min ~8x")
        fig_it.update_layout(title="Inventory Turnover — Quarterly", height=250,
                             margin=dict(t=50, b=20), yaxis_title="Turns (x)",
                             showlegend=False)
        st.plotly_chart(fig_it, use_container_width=True)

    with c2:
        fig_at = go.Figure()
        fig_at.add_trace(go.Scatter(x=ratios_df["quarter"], y=ratios_df["asset_turnover"],
                                    name="Asset Turnover", mode="lines+markers",
                                    line=dict(color="#0891B2", width=2.5), marker=dict(size=7)))
        fig_at.update_layout(title="Asset Turnover — Quarterly", height=250,
                             margin=dict(t=50, b=20), yaxis_title="Turns (x)",
                             showlegend=False)
        st.plotly_chart(fig_at, use_container_width=True)

    # Ratio interpretation table
    st.markdown("#### Ratio Interpretation & Recommendations")
    roa_status  = "✅ Good" if latest_r["roa"] > 0 else "🔴 Loss"
    roa_interp  = ("Positive asset returns." if latest_r["roa"] > 0
                   else "Negative — assets generating a loss. FY2024 reflects R3.19B net loss.")
    ratio_interp = [
        {"Ratio": "Current Ratio",
         "Value": f"{latest_r['current_ratio']:.2f}",
         "Benchmark": "≥ 1.5", "Status": "🔴 Weak",
         "Interpretation": "At ~1.05 — barely above 1.0. Working capital buffer is very thin."},
        {"Ratio": "Quick Ratio",
         "Value": f"{latest_r['quick_ratio']:.2f}",
         "Benchmark": "≥ 1.0", "Status": "🔴 Risky",
         "Interpretation": "At ~0.40 — PnP cannot cover current liabilities with liquid assets alone."},
        {"Ratio": "Debt-to-Equity",
         "Value": f"{latest_r['debt_to_equity']:.2f}",
         "Benchmark": "≤ 1.5", "Status": "🔴 High",
         "Interpretation": "Leverage is very high due to equity erosion from FY2024 loss."},
        {"Ratio": "ROA",
         "Value": f"{latest_r['roa']:.1f}%",
         "Benchmark": "≥ 5%", "Status": roa_status,
         "Interpretation": roa_interp},
        {"Ratio": "Inventory Turnover",
         "Value": f"{latest_r['inventory_turnover']:.2f}x",
         "Benchmark": "≥ 8.0x", "Status": "⚠️ Watch",
         "Interpretation": "Declining turns signal slower stock movement — risk of excess inventory."},
        {"Ratio": "Asset Turnover",
         "Value": f"{latest_r['asset_turnover']:.2f}x",
         "Benchmark": "≥ 3.0x (retail)", "Status": "⚠️ Monitor",
         "Interpretation": "Revenue generated per rand of assets. Declining trend warrants attention."},
    ]
    st.dataframe(pd.DataFrame(ratio_interp), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — VARIANCE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Variance Analysis Monitoring System")
    st.caption("Compares budgeted vs actual financial performance. Flags unfavourable variances and overspend alerts.")

    # Summary KPIs
    rev_row = variance_df[variance_df["category"] == "Sales Revenue"].iloc[0]
    exp_row = variance_df[variance_df["category"] == "Operating Expenses"].iloc[0]
    pnl_row = variance_df[variance_df["category"] == "Net Profit / Loss"].iloc[0]
    gp_row  = variance_df[variance_df["category"] == "Gross Profit"].iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Revenue vs Budget",      f"R{rev_row['variance']/1000:+.1f}B", f"{rev_row['variance_pct']:+.1f}%")
    c2.metric("OpEx vs Budget",         f"R{exp_row['variance']/1000:+.1f}B", f"{exp_row['variance_pct']:+.1f}%",
              delta_color="inverse")
    c3.metric("Gross Profit vs Budget", f"R{gp_row['variance']/1000:+.1f}B",  f"{gp_row['variance_pct']:+.1f}%")
    c4.metric("Net Profit vs Budget",   f"R{pnl_row['variance']/1000:+.1f}B", f"{pnl_row['variance_pct']:+.1f}%")

    st.markdown("---")

    # Variance table — Budget vs Actual 2024
    st.markdown("##### Budget vs Actual — FY2024 (ZAR millions)")
    display_var = variance_df.copy()
    for col in ["actual_2024", "budget_2024", "actual_2023"]:
        display_var[col] = display_var[col].apply(lambda x: f"R{x/1000:+.2f}B")
    display_var["variance"]     = display_var["variance"].apply(lambda x: f"R{x/1000:+.2f}B")
    display_var["variance_pct"] = display_var["variance_pct"].apply(lambda x: f"{x:+.1f}%")
    display_var["yoy_change"]   = display_var["yoy_change"].apply(lambda x: f"R{x/1000:+.2f}B")
    display_var["yoy_pct"]      = display_var["yoy_pct"].apply(lambda x: f"{x:+.1f}%")
    display_var = display_var.rename(columns={
        "category": "Category", "budget_2024": "Budget 2024", "actual_2024": "Actual 2024",
        "actual_2023": "Actual 2023", "variance": "Budget Variance",
        "variance_pct": "Variance %", "yoy_change": "YoY Change",
        "yoy_pct": "YoY %", "type": "Type", "status": "Status"
    })
    st.dataframe(display_var[["Category","Type","Budget 2024","Actual 2024","Budget Variance",
                               "Variance %","Actual 2023","YoY Change","YoY %","Status"]],
                 use_container_width=True, hide_index=True)

    st.markdown("---")
    c1, c2 = st.columns(2)

    with c1:
        # Grouped bar: budget vs actual vs prior year
        fig_var = go.Figure()
        fig_var.add_trace(go.Bar(name="Budget 2024",
                                 x=variance_df["category"],
                                 y=variance_df["budget_2024"] / 1000,
                                 marker_color="rgba(24,95,165,0.55)",
                                 marker_line=dict(color="#185FA5", width=1)))
        fig_var.add_trace(go.Bar(name="Actual 2024",
                                 x=variance_df["category"],
                                 y=variance_df["actual_2024"] / 1000,
                                 marker_color="rgba(220,38,38,0.55)",
                                 marker_line=dict(color="#dc2626", width=1)))
        fig_var.add_trace(go.Bar(name="Actual 2023",
                                 x=variance_df["category"],
                                 y=variance_df["actual_2023"] / 1000,
                                 marker_color="rgba(29,158,117,0.45)",
                                 marker_line=dict(color="#1D9E75", width=1)))
        fig_var.update_layout(barmode="group", title="Budget vs Actual vs Prior Year (R billions)",
                              height=300, margin=dict(t=50,b=20),
                              yaxis_title="R billions",
                              legend=dict(orientation="h", y=-0.3),
                              xaxis_tickangle=-20)
        st.plotly_chart(fig_var, use_container_width=True)

    with c2:
        # Variance waterfall
        variance_df["color"] = variance_df.apply(
            lambda r: "#16a34a" if r["status"] == "Favourable" else "#dc2626", axis=1)
        fig_wfall = go.Figure(go.Bar(
            x=variance_df["category"],
            y=variance_df["variance"] / 1000,
            marker_color=variance_df["color"],
            text=[f"R{v/1000:+.1f}B" for v in variance_df["variance"]],
            textposition="outside",
        ))
        fig_wfall.add_hline(y=0, line_color="gray", line_width=1)
        fig_wfall.update_layout(title="Budget Variance Analysis (R billions)",
                                height=300, margin=dict(t=50,b=20),
                                yaxis_title="Variance (R billions)", showlegend=False,
                                xaxis_tickangle=-20)
        st.plotly_chart(fig_wfall, use_container_width=True)

    # Annual revenue trend (real PnP data)
    fig_trend_annual = go.Figure()
    fig_trend_annual.add_trace(go.Bar(
        x=trend_df["year"].astype(str), y=trend_df["revenue"] / 1000,
        name="Revenue", marker_color="rgba(24,95,165,0.55)"))
    fig_trend_annual.update_layout(
        title="Pick n Pay Annual Revenue Trend (2021–2025, R billions)",
        height=260, margin=dict(t=50,b=20), yaxis_title="R billions", showlegend=False)
    st.plotly_chart(fig_trend_annual, use_container_width=True)

    # Monthly revenue vs expenses
    fig_rev = go.Figure()
    fig_rev.add_trace(go.Bar(x=monthly_df["date"], y=monthly_df["revenue"] / 1000,
                             name="Revenue", marker_color="rgba(24,95,165,0.4)"))
    fig_rev.add_trace(go.Scatter(x=monthly_df["date"],
                                 y=monthly_df["operating_expenses"] / 1000,
                                 name="Operating Expenses",
                                 line=dict(color="#dc2626", width=2, dash="dash")))
    fig_rev.update_layout(title="Monthly Revenue vs Operating Expenses (R millions)", height=280,
                          margin=dict(t=50,b=20), yaxis_title="R millions",
                          legend=dict(orientation="h", y=-0.3))
    st.plotly_chart(fig_rev, use_container_width=True)

    # Automated alerts
    unfav = variance_df[variance_df["status"] == "Unfavourable"]
    if not unfav.empty:
        st.markdown("#### ⚠️ Automated Variance Alerts")
        for _, row in variance_df.iterrows():
            if row["status"] == "Unfavourable":
                pct = abs(row["variance_pct"])
                cls  = "bg-red-50 border-red-500"    if pct > 15 else "bg-yellow-50 border-yellow-400"
                icon = "🔴" if pct > 15 else "🟡"
                note = "Immediate review recommended." if pct > 15 else "Monitor closely."
                st.markdown(f"""
                <div class="border-l-4 {cls} px-4 py-3 rounded-r-lg my-2">
                  <span class="font-semibold">{icon} {row['category']}</span>
                  <span class="text-gray-600 text-sm"> &mdash; Unfavourable variance of
                  R{abs(row['variance'])/1000:.2f}B ({pct:.1f}% vs budget). {note}</span>
                </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — ANOMALY DETECTION
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Financial Anomaly Detection Model")
    st.caption("Z-Score analysis and Isolation Forest identify unusual financial patterns, expense spikes, and potential fraud indicators.")

    # Re-compute with sidebar threshold
    features = anomaly_df[["revenue","operating_expenses","net_cashflow","net_profit"]]
    z = np.abs(stats.zscore(features))
    anomaly_df["max_zscore"] = z.max(axis=1).round(2)
    anomaly_df["zscore_flag"] = anomaly_df["max_zscore"] > zscore_threshold
    anomaly_df["anomaly"] = anomaly_df["zscore_flag"] | anomaly_df["iso_flag"]

    n_anomalies = anomaly_df["anomaly"].sum()
    max_z       = anomaly_df["max_zscore"].max()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Anomalies Detected", int(n_anomalies), "This period")
    c2.metric("Max Z-Score", f"{max_z:.2f}", "Threshold: " + str(zscore_threshold))
    c3.metric("Model", "Z-Score + IsoForest")
    c4.metric("Fraud Risk Signal", "Low" if n_anomalies < 4 else "Medium")

    st.markdown("---")
    c1, c2 = st.columns(2)

    with c1:
        # Expense scatter with anomaly highlights
        colors = ["#dc2626" if a else "#185FA5" for a in anomaly_df["anomaly"]]
        sizes  = [14 if a else 8 for a in anomaly_df["anomaly"]]
        fig_sc = go.Figure()
        fig_sc.add_trace(go.Scatter(
            x=anomaly_df["date"], y=anomaly_df["operating_expenses"] / 1e3,
            mode="markers",
            marker=dict(color=colors, size=sizes, line=dict(width=1, color="white")),
            text=[f"Z={z:.2f}" for z in anomaly_df["max_zscore"]],
            hovertemplate="<b>%{x|%b %Y}</b><br>Expenses: R%{y:.0f}K<br>%{text}<extra></extra>",
        ))
        fig_sc.update_layout(title="Operating Expense Anomalies (Z-Score)",
                             height=300, margin=dict(t=50,b=20),
                             yaxis_title="Expenses (R000)")
        st.plotly_chart(fig_sc, use_container_width=True)

        st.markdown("""
        <div class="flex gap-4 text-xs text-gray-500 mt-1">
          <span class="flex items-center gap-1">&#128309; Normal</span>
          <span class="flex items-center gap-1">&#128308; Anomaly (|Z| &gt; threshold)</span>
        </div>""", unsafe_allow_html=True)

    with c2:
        # Z-score bar chart
        bar_colors = ["#dc2626" if z > zscore_threshold else "#94a3b8"
                      for z in anomaly_df["max_zscore"]]
        fig_z = go.Figure(go.Bar(
            x=anomaly_df["date"].dt.strftime("%b %y"),
            y=anomaly_df["max_zscore"],
            marker_color=bar_colors,
            text=[f"{z:.2f}" for z in anomaly_df["max_zscore"]],
            textposition="outside",
        ))
        fig_z.add_hline(y=zscore_threshold, line_dash="dash", line_color="#dc2626",
                        annotation_text=f"Threshold ({zscore_threshold})")
        fig_z.update_layout(title="Max Z-Score Per Month", height=300,
                            margin=dict(t=50,b=20), yaxis_title="Z-Score",
                            showlegend=False)
        st.plotly_chart(fig_z, use_container_width=True)

    # Flagged anomalies table
    flagged = anomaly_df[anomaly_df["anomaly"]].copy()
    if not flagged.empty:
        st.markdown("#### Flagged Anomalous Records")
        flagged["date"] = flagged["date"].dt.strftime("%b %Y")
        flagged["revenue"] = flagged["revenue"].apply(lambda x: f"R{x/1e3:.0f}K")
        flagged["operating_expenses"] = flagged["operating_expenses"].apply(lambda x: f"R{x/1e3:.0f}K")
        flagged["net_profit"] = flagged["net_profit"].apply(lambda x: f"R{x/1e3:.0f}K")
        flagged["net_cashflow"] = flagged["net_cashflow"].apply(lambda x: f"R{x/1e3:.0f}K")
        flagged["anomaly_type"] = flagged.apply(
            lambda r: "Z-Score + IsoForest" if r["zscore_flag"] and r["iso_flag"]
                      else "Z-Score" if r["zscore_flag"] else "Isolation Forest", axis=1)
        st.dataframe(flagged[["date","revenue","operating_expenses","net_profit",
                               "net_cashflow","max_zscore","anomaly_type"]].rename(columns={
            "date": "Month", "revenue": "Revenue", "operating_expenses": "Op. Expenses",
            "net_profit": "Net Profit", "net_cashflow": "Net Cash Flow",
            "max_zscore": "Max Z-Score", "anomaly_type": "Detection Method"
        }), use_container_width=True, hide_index=True)

    # Multi-feature line
    fig_multi = make_subplots(rows=2, cols=2, subplot_titles=[
        "Revenue", "Operating Expenses", "Net Profit", "Net Cash Flow"])
    for i, (col, name, color) in enumerate([
        ("revenue", "Revenue", "#185FA5"),
        ("operating_expenses", "Op. Expenses", "#dc2626"),
        ("net_profit", "Net Profit", "#1D9E75"),
        ("net_cashflow", "Cash Flow", "#f59e0b"),
    ]):
        r, c = i // 2 + 1, i % 2 + 1
        fig_multi.add_trace(go.Scatter(
            x=anomaly_df["date"], y=anomaly_df[col] / 1e3,
            name=name, line=dict(color=color, width=1.8),
            mode="lines+markers",
            marker=dict(color=["#dc2626" if a else color for a in anomaly_df["anomaly"]],
                        size=[10 if a else 4 for a in anomaly_df["anomaly"]])
        ), row=r, col=c)
    fig_multi.update_layout(height=400, showlegend=False,
                             title_text="Financial Features — Anomalies Highlighted in Red",
                             margin=dict(t=60, b=20))
    st.plotly_chart(fig_multi, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — EARLY WARNING
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("Early-Warning Financial Risk Dashboard")
    st.caption("Real-time risk indicators, automated alerts and 6-month cash flow forecast to support proactive managerial decisions.")

    # Risk scoring
    def risk_level(score, thresholds):
        if thresholds[0]: return "🔴 HIGH"
        if thresholds[1]: return "🟡 MEDIUM"
        return "🟢 LOW"

    liq  = latest_r["current_ratio"]
    qr   = latest_r["quick_ratio"]
    de   = latest_r["debt_to_equity"]

    risks = {
        "Liquidity Risk":     risk_level(1, [liq < 1.2, liq < 1.5]),
        "Cash Flow Risk":     risk_level(1, [anomaly_df["net_cashflow"].tail(3).mean() < 0,
                                             anomaly_df["net_cashflow"].tail(3).mean() < 50000]),
        "Solvency Risk":      risk_level(1, [de > 2.5, de > 1.8]),
        "Profitability Risk": risk_level(1, [health["npm"] < 5, health["npm"] < 12]),
        "Anomaly / Fraud":    risk_level(1, [n_anomalies > 5, n_anomalies > 2]),
        "Operational Risk":   risk_level(1, [abs(variance_df["variance_pct"]).mean() > 25,
                                             abs(variance_df["variance_pct"]).mean() > 12]),
    }

    cols = st.columns(3)
    for i, (label, level) in enumerate(risks.items()):
        bg   = "bg-red-50 border border-red-200"   if "HIGH"   in level else \
               "bg-yellow-50 border border-yellow-200" if "MEDIUM" in level else \
               "bg-green-50 border border-green-200"
        tcls = "text-red-800"    if "HIGH"   in level else \
               "text-yellow-800" if "MEDIUM" in level else \
               "text-green-800"
        cols[i % 3].markdown(f"""
        <div class="{bg} rounded-xl p-4 text-center mb-2 shadow-sm">
          <div class="text-2xl">{level.split()[0]}</div>
          <div class="text-xs font-bold {tcls} mt-1 uppercase tracking-wide">{label}</div>
          <div class="text-xs {tcls} font-medium mt-0.5">{level.split()[-1]}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    c1, c2 = st.columns([1.2, 1])

    # Active alerts
    with c1:
        st.markdown("#### Active Alerts")
        alerts = []
        if qr < 1.0:
            alerts.append(("critical", "Quick Ratio below 1.0",
                           f"At {qr:.2f} — organisation cannot cover all current liabilities with liquid assets. Review short-term credit."))
        if de > 1.8:
            alerts.append(("warning", f"Debt-to-Equity elevated at {de:.2f}",
                           "Approaching high-risk threshold of 2.0. Limit new debt obligations."))
        if liq < 1.5:
            alerts.append(("warning", "Current Ratio below benchmark",
                           f"At {liq:.2f} vs 1.5 benchmark. Working capital buffer is shrinking."))
        unfav_count = (variance_df["status"] == "Unfavourable").sum()
        if unfav_count >= 3:
            alerts.append(("critical", f"{unfav_count} unfavourable budget variances",
                           "Operating expenses and payroll are significantly over budget."))
        if n_anomalies > 2:
            alerts.append(("warning", f"{int(n_anomalies)} financial anomalies detected",
                           "Expense spikes and cash flow abnormalities require investigation."))
        alerts.append(("info", "Revenue growth remains positive",
                       f"YoY revenue growth is a key stabilising factor for the organisation."))

        for level, title, desc in alerts:
            cls  = "bg-red-50 border-red-500"    if level == "critical" else \
                   "bg-yellow-50 border-yellow-400" if level == "warning"  else \
                   "bg-blue-50 border-blue-400"
            icon = "🔴" if level == "critical" else "🟡" if level == "warning" else "🔵"
            st.markdown(f"""
            <div class="border-l-4 {cls} px-4 py-3 rounded-r-xl my-2">
              <p class="font-semibold text-sm m-0">{icon} {title}</p>
              <p class="text-xs text-gray-500 m-0 mt-1">{desc}</p>
            </div>""", unsafe_allow_html=True)

    # Risk radar
    with c2:
        risk_vals = [
            min(100, max(0, 100 - (liq - 1) * 50)),
            min(100, max(0, 100 - health["npm"] * 3)),
            min(100, max(0, (de / 3) * 100)),
            min(100, max(0, 100 - health["gpm"] * 1.5)),
            min(100, max(0, (1 - anomaly_df["net_cashflow"].tail(3).mean() /
                             anomaly_df["revenue"].tail(3).mean()) * 100)),
            min(100, max(0, abs(variance_df["variance_pct"]).mean() * 3)),
        ]
        dim_labels = ["Liquidity","Profitability","Leverage","Margin","Cash Flow","Variance"]
        fig_rr = go.Figure(go.Scatterpolar(
            r=risk_vals + [risk_vals[0]],
            theta=dim_labels + [dim_labels[0]],
            fill="toself",
            fillcolor="rgba(220,38,38,0.12)",
            line=dict(color="#dc2626", width=2),
            marker=dict(size=6, color="#dc2626"),
        ))
        fig_rr.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=False, height=310,
            title=dict(text="Risk Radar", font=dict(size=13), x=0.5),
            margin=dict(t=50, b=20, l=30, r=30),
        )
        st.plotly_chart(fig_rr, use_container_width=True)

    st.markdown("---")

    # 6-month cash flow forecast
    last_inflow  = monthly_df["cash_inflow"].tail(3).mean()
    last_outflow = monthly_df["cash_outflow"].tail(3).mean()
    forecast_months = pd.date_range(start=monthly_df["date"].max() + pd.DateOffset(months=1),
                                    periods=6, freq="MS")
    inflow_f  = [last_inflow  * (1 + 0.015 * i) * np.random.uniform(0.97, 1.03) for i in range(1, 7)]
    outflow_f = [last_outflow * (1 + 0.018 * i) * np.random.uniform(0.97, 1.03) for i in range(1, 7)]
    net_f     = [i - o for i, o in zip(inflow_f, outflow_f)]

    fig_fcst = go.Figure()
    fig_fcst.add_trace(go.Scatter(x=forecast_months, y=[v/1e3 for v in inflow_f],
                                  name="Projected Inflow", line=dict(color="#1D9E75", width=2.5),
                                  fill="tozeroy", fillcolor="rgba(29,158,117,0.08)"))
    fig_fcst.add_trace(go.Scatter(x=forecast_months, y=[v/1e3 for v in outflow_f],
                                  name="Projected Outflow", line=dict(color="#dc2626", width=2.5, dash="dash"),
                                  fill="tozeroy", fillcolor="rgba(220,38,38,0.06)"))
    fig_fcst.add_trace(go.Bar(x=forecast_months, y=[v/1e3 for v in net_f],
                              name="Net Cash Flow",
                              marker_color=["#16a34a" if v > 0 else "#dc2626" for v in net_f],
                              opacity=0.5))
    fig_fcst.update_layout(title="6-Month Cash Flow Forecast",
                           height=300, margin=dict(t=50, b=20),
                           yaxis_title="R thousands",
                           legend=dict(orientation="h", y=-0.3))
    st.plotly_chart(fig_fcst, use_container_width=True)

    # Summary recommendations
    st.markdown("#### Strategic Recommendations for Management")
    recs = [
        ("🔴", "Improve liquidity immediately",
         f"Quick ratio is {qr:.2f}. Accelerate debtor collections, review stock holding, and renegotiate short-term credit lines."),
        ("🟡", "Control operating expenditure",
         "Operating expenses are R180K+ over budget. Conduct a cost audit and freeze discretionary spending."),
        ("🟡", "Monitor leverage levels",
         f"D/E at {de:.2f}. Avoid additional long-term debt until the ratio falls below 1.5."),
        ("🟢", "Leverage revenue growth",
         f"Revenue growing at {health['gpm']:.1f}% gross margin. Channel growth into cash reserves to rebuild liquidity."),
    ]
    for icon, title, desc in recs:
        cls = "bg-red-50 border-red-500"    if icon == "🔴" else \
              "bg-yellow-50 border-yellow-400" if icon == "🟡" else \
              "bg-green-50 border-green-500"
        st.markdown(f"""
        <div class="border-l-4 {cls} px-4 py-3 rounded-r-xl my-2 shadow-sm">
          <p class="font-semibold text-sm m-0">{icon} {title}</p>
          <p class="text-xs text-gray-600 m-0 mt-1">{desc}</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class="mt-6 border-t border-gray-200 pt-4 flex items-center justify-between text-xs text-gray-400">
      <span>FinDiag Pro v1.0 &mdash; AI-Enhanced Financial Diagnostics</span>
      <span>Managerial Finance PBL &bull; Pathway B &bull; Durban University of Technology</span>
    </div>""", unsafe_allow_html=True)
