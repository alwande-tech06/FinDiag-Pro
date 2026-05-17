"""
FinDiag Pro v2.0 — AI-Enhanced Organisational Financial Health Diagnostics
Pathway B: Performance Monitoring & Early Warning Systems
Durban University of Technology — Managerial Finance PBL
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots
from sklearn.ensemble import IsolationForest
from scipy import stats
from fpdf import FPDF
import warnings
import io
import json
import hashlib
import os
from datetime import datetime
warnings.filterwarnings("ignore")

# ── Global Plotly theme ───────────────────────────────────────────────────────
_t = go.layout.Template()
_t.layout = go.Layout(
    paper_bgcolor="#ffffff",
    plot_bgcolor="#f8fafb",
    font=dict(family="Inter, system-ui, sans-serif", color="#334155", size=11),
    title=dict(font=dict(size=13, color="#003087"), x=0.01),
    xaxis=dict(gridcolor="#EDF2F7", linecolor="#E2E8F0", tickcolor="#94a3b8",
               showgrid=True, zeroline=False, color="#64748b"),
    yaxis=dict(gridcolor="#EDF2F7", linecolor="#E2E8F0", tickcolor="#94a3b8",
               showgrid=True, zeroline=False, color="#64748b"),
    legend=dict(bgcolor="rgba(255,255,255,0.95)", bordercolor="#E2E8F0",
                borderwidth=1, font=dict(size=11, color="#334155")),
    colorway=["#003087","#e31837","#0D9488","#D97706","#6366f1","#64748B"],
)
pio.templates["findiag"] = _t
pio.templates.default   = "plotly+findiag"

C_BLUE   = "#003087"
C_TEAL   = "#0D9488"
C_RED    = "#e31837"
C_AMBER  = "#D97706"
C_INDIGO = "#6366f1"
C_SLATE  = "#64748B"

# ── Page config — must be first Streamlit call ────────────────────────────────
st.set_page_config(
    page_title="Pick n Pay Financial Health System",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# AUTH UTILITIES
# ══════════════════════════════════════════════════════════════════════════════
USERS_FILE = "data/users.json"
AUDIT_FILE = "data/audit_log.csv"

_DEFAULT_USERS = [
    {"email": "admin@findiag.com",   "name": "Admin User",       "role": "Admin",
     "password": hashlib.sha256(b"Admin@123").hexdigest()},
    {"email": "analyst@findiag.com", "name": "Jane Analyst",     "role": "Analyst",
     "password": hashlib.sha256(b"Analyst@123").hexdigest()},
    {"email": "viewer@findiag.com",  "name": "Executive Viewer", "role": "Viewer",
     "password": hashlib.sha256(b"Viewer@123").hexdigest()},
]

def _hash(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()

def load_users() -> list:
    os.makedirs("data", exist_ok=True)
    if not os.path.exists(USERS_FILE):
        with open(USERS_FILE, "w") as f:
            json.dump(_DEFAULT_USERS, f, indent=2)
        return _DEFAULT_USERS[:]
    with open(USERS_FILE) as f:
        return json.load(f)

def save_users(users: list):
    os.makedirs("data", exist_ok=True)
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)

def authenticate(email: str, pw: str):
    hashed = _hash(pw)
    for u in load_users():
        if u["email"].lower() == email.lower() and u["password"] == hashed:
            return u
    return None

def log_action(email: str, action: str):
    os.makedirs("data", exist_ok=True)
    row = pd.DataFrame([{
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "user": email, "action": action,
    }])
    exists = os.path.exists(AUDIT_FILE)
    row.to_csv(AUDIT_FILE, mode="a", header=not exists, index=False)

def load_audit() -> pd.DataFrame:
    if not os.path.exists(AUDIT_FILE):
        return pd.DataFrame(columns=["timestamp", "user", "action"])
    return pd.read_csv(AUDIT_FILE)

def register_user(name: str, email: str, pw: str, role: str = "Viewer") -> tuple:
    users = load_users()
    if any(u["email"].lower() == email.lower() for u in users):
        return False, "An account with this email already exists."
    users.append({"name": name, "email": email, "role": role, "password": _hash(pw)})
    save_users(users)
    log_action(email, f"Account registered — Role: {role}")
    return True, "Account created successfully!"

# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════
_defaults = {"authenticated": False, "user": None, "page": "landing",
             "logged_access": False, "auth_tab": "login"}
for _k, _v in _defaults.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ══════════════════════════════════════════════════════════════════════════════
# LANDING PAGE
# ══════════════════════════════════════════════════════════════════════════════
if not st.session_state.authenticated and st.session_state.page == "landing":
    st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Barlow:wght@400;600;700;900&family=Barlow+Condensed:wght@600;700;900&display=swap" rel="stylesheet">
    <style>
    #MainMenu, footer, [data-testid="stHeader"] { visibility: hidden; }
    [data-testid="stAppViewContainer"] {
        background-color: #04071a !important;
        background-image:
            radial-gradient(ellipse 80% 60% at 50% -10%, rgba(0,48,135,0.45) 0%, transparent 70%),
            radial-gradient(ellipse 50% 40% at 80% 80%, rgba(227,24,55,0.2) 0%, transparent 60%),
            linear-gradient(rgba(255,255,255,0.025) 1px, transparent 1px),
            linear-gradient(90deg, rgba(255,255,255,0.025) 1px, transparent 1px) !important;
        background-size: auto, auto, 50px 50px, 50px 50px !important;
        background-position: center top, right bottom, 0 0, 0 0 !important;
    }
    .block-container { padding-top: 0 !important; max-width: 900px; }
    * { font-family: 'Barlow', sans-serif !important; }
    @keyframes pulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50%       { opacity: 0.6; transform: scale(0.8); }
    }
    .stButton > button[kind="primary"], .stButton > button {
        background: linear-gradient(135deg, #003087 0%, #003087 100%) !important;
        border: none !important;
        box-shadow: 0 6px 24px rgba(0,48,135,0.5) !important;
        color: #fff !important;
        font-weight: 700 !important;
        border-radius: 8px !important;
        font-size: 1rem !important;
        letter-spacing: 0.03em !important;
        font-family: 'Barlow', sans-serif !important;
    }
    </style>""", unsafe_allow_html=True)

    st.markdown("""
    <div style="padding:4rem 2rem 0.5rem;text-align:center;">
      <div style="display:inline-flex;align-items:center;gap:8px;
                  background:rgba(0,48,135,0.25);border:1px solid rgba(0,48,135,0.5);
                  border-radius:50px;padding:6px 18px;margin-bottom:2rem;">
        <span style="display:inline-block;width:8px;height:8px;border-radius:50%;
                     background:#4d8bff;animation:pulse 2s infinite;flex-shrink:0;"></span>
        <span style="color:#7ab3ff;font-size:0.75rem;font-weight:600;letter-spacing:0.12em;
                     font-family:'Barlow Condensed',sans-serif;text-transform:uppercase;">
          Live Financial Intelligence
        </span>
      </div>
      <h1 style="font-family:'Barlow Condensed',sans-serif;font-size:3.6rem;font-weight:900;
                 margin:0 0 0.5rem;letter-spacing:-0.01em;color:#e8edf8;line-height:1.08;">
        Pick n Pay &nbsp;
        <span style="color:#4d8bff;">Financial</span>&nbsp;
        <span style="color:#e31837;">Health</span>
        System
      </h1>
      <p style="color:#8694b5;font-size:1rem;max-width:560px;line-height:1.75;margin:0.8rem auto 2.5rem;">
        Real-time financial monitoring, anomaly detection and early warning alerts
        to support executive decision-making at Pick n Pay Stores Ltd.
      </p>
    </div>""", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    for col, title, desc in [
        (c1, "Health Scorecard",  "5-dimension composite financial health index"),
        (c2, "Anomaly Detection", "Z-Score + Isolation Forest AI models"),
        (c3, "Early Warning",     "Automated risk alerts & 6-month forecast"),
        (c4, "Variance Monitor",  "Budget vs actual with automated alerts"),
    ]:
        col.markdown(f"""
        <div style="background:#111e36;border:1px solid rgba(255,255,255,0.08);
                    border-radius:14px;padding:1.4rem 1.2rem;text-align:left;min-height:120px;">
          <div style="color:#e8edf8;font-weight:700;font-size:0.88rem;
                      font-family:'Barlow Condensed',sans-serif;letter-spacing:0.04em;">{title}</div>
          <div style="color:#8694b5;font-size:0.78rem;margin-top:6px;line-height:1.55;">{desc}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div style="display:flex;justify-content:center;gap:0;margin:2rem auto 0;max-width:680px;
                background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.07);
                border-radius:14px;overflow:hidden;">
      <div style="flex:1;text-align:center;padding:1.2rem 0.5rem;border-right:1px solid rgba(255,255,255,0.07);">
        <div style="color:#e8edf8;font-size:1.45rem;font-weight:900;
                    font-family:'Barlow Condensed',sans-serif;">R114.9B</div>
        <div style="color:#8694b5;font-size:0.68rem;text-transform:uppercase;
                    letter-spacing:0.1em;margin-top:3px;">Revenue FY2024</div>
      </div>
      <div style="flex:1;text-align:center;padding:1.2rem 0.5rem;border-right:1px solid rgba(255,255,255,0.07);">
        <div style="color:#e8edf8;font-size:1.45rem;font-weight:900;
                    font-family:'Barlow Condensed',sans-serif;">5</div>
        <div style="color:#8694b5;font-size:0.68rem;text-transform:uppercase;
                    letter-spacing:0.1em;margin-top:3px;">Modules</div>
      </div>
      <div style="flex:1;text-align:center;padding:1.2rem 0.5rem;border-right:1px solid rgba(255,255,255,0.07);">
        <div style="color:#e8edf8;font-size:1.45rem;font-weight:900;
                    font-family:'Barlow Condensed',sans-serif;">3</div>
        <div style="color:#8694b5;font-size:0.68rem;text-transform:uppercase;
                    letter-spacing:0.1em;margin-top:3px;">User Roles</div>
      </div>
      <div style="flex:1;text-align:center;padding:1.2rem 0.5rem;">
        <div style="color:#00c896;font-size:1.45rem;font-weight:900;
                    font-family:'Barlow Condensed',sans-serif;">Live</div>
        <div style="color:#8694b5;font-size:0.68rem;text-transform:uppercase;
                    letter-spacing:0.1em;margin-top:3px;">Real-Time</div>
      </div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:2rem'></div>", unsafe_allow_html=True)
    _, mid, _ = st.columns([2, 1, 2])
    with mid:
        if st.button("Get Started →", use_container_width=True, type="primary"):
            st.session_state.page = "login"
            st.rerun()

    st.markdown("""
    <div style="text-align:center;padding:1.8rem 0 1rem;">
      <p style="color:#3d4f6e;font-size:0.7rem;letter-spacing:0.1em;text-transform:uppercase;">
        Durban University of Technology &nbsp;|&nbsp; Managerial Finance PBL &nbsp;|&nbsp; Pathway B
      </p>
    </div>""", unsafe_allow_html=True)
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# LOGIN / REGISTER PAGE
# ══════════════════════════════════════════════════════════════════════════════
if not st.session_state.authenticated and st.session_state.page == "login":
    st.markdown('<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Barlow+Condensed:wght@700;800;900&display=swap" rel="stylesheet">', unsafe_allow_html=True)
    st.markdown("""<style>
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
[data-testid="stHeader"] {visibility:hidden;}
[data-testid="stToolbar"] {visibility:hidden;}
[data-testid="stAppViewContainer"] {
    background-color:#f0f4f8 !important;
}
.block-container {
    background:#ffffff !important;
    border:1.5px solid rgba(0,48,135,0.12) !important;
    border-radius:16px !important;
    max-width:460px !important;
    padding:2.5rem 2.4rem 2rem !important;
    box-shadow:0 4px 24px rgba(0,48,135,0.1),0 1px 4px rgba(0,0,0,0.05) !important;
    margin-top:6vh !important;
}
.stButton > button {
    border-radius:10px !important;
    font-weight:600 !important;
    font-size:0.9rem !important;
    height:44px !important;
}
.stButton > button[kind="primary"] {
    background:#003087 !important;
    border:none !important;
    color:#ffffff !important;
}
.stButton > button[kind="secondary"] {
    background:#ffffff !important;
    border:1.5px solid rgba(0,48,135,0.2) !important;
    color:#003087 !important;
}
div[data-testid="stFormSubmitButton"] > button {
    background:#003087 !important;
    border:none !important;
    border-radius:11px !important;
    color:#fff !important;
    font-weight:700 !important;
    font-size:1rem !important;
    height:50px !important;
    box-shadow:0 4px 16px rgba(0,48,135,0.3) !important;
    margin-top:0.2rem !important;
}
div[data-testid="stFormSubmitButton"] > button:hover {
    background:#e31837 !important;
    box-shadow:0 4px 16px rgba(227,24,55,0.3) !important;
}
.stTextInput label {
    color:#5a6a8a !important;
    font-size:0.79rem !important;
    font-weight:600 !important;
    letter-spacing:0.06em !important;
    text-transform:uppercase !important;
}
.stTextInput > div > div > input {
    background:#f8f9fb !important;
    border:1.5px solid rgba(0,48,135,0.15) !important;
    border-radius:10px !important;
    color:#1a2744 !important;
    padding:0.65rem 1rem !important;
    font-size:0.95rem !important;
}
.stTextInput > div > div > input:focus {
    border-color:#003087 !important;
    box-shadow:0 0 0 3px rgba(0,48,135,0.1) !important;
    background:#ffffff !important;
}
.stTextInput > div > div > input::placeholder {color:#a8b5cc !important;}
.stSelectbox label {
    color:#5a6a8a !important;
    font-size:0.79rem !important;
    font-weight:600 !important;
    letter-spacing:0.06em !important;
    text-transform:uppercase !important;
}
.stSelectbox > div > div {
    background:#f8f9fb !important;
    border:1.5px solid rgba(0,48,135,0.15) !important;
    border-radius:10px !important;
    color:#1a2744 !important;
}
.stCheckbox label p {color:#5a6a8a !important;font-size:0.85rem !important;}
[data-testid="stForm"] {background:transparent !important;border:none !important;padding:0 !important;}
hr {border-color:rgba(0,48,135,0.1) !important;margin:1rem 0 !important;}
[data-testid="InputInstructions"] {display:none !important;}
</style>""", unsafe_allow_html=True)

    # ── App name ──────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="text-align:center;margin-bottom:1.6rem;">
      <div style="color:#003087;font-family:'Barlow Condensed',sans-serif;
                  font-size:1.7rem;font-weight:900;letter-spacing:-0.01em;">
        Pick n Pay Financial Health System
      </div>
      <div style="color:#5a6a8a;font-size:0.82rem;margin-top:4px;">
        Pick n Pay &middot; Financial Health System
      </div>
    </div>""", unsafe_allow_html=True)

    # ── Tab switcher ──────────────────────────────────────────────────────────
    tc1, tc2 = st.columns(2)
    with tc1:
        if st.button("Sign In", use_container_width=True,
                     type="primary" if st.session_state.auth_tab == "login" else "secondary",
                     key="tab_signin"):
            st.session_state.auth_tab = "login"
            st.rerun()
    with tc2:
        if st.button("Register", use_container_width=True,
                     type="primary" if st.session_state.auth_tab == "register" else "secondary",
                     key="tab_register"):
            st.session_state.auth_tab = "register"
            st.rerun()

    st.markdown("<hr>", unsafe_allow_html=True)

    # ══ SIGN IN FORM ══════════════════════════════════════════════════════════
    if st.session_state.auth_tab == "login":
        st.markdown("""
        <div style="margin-bottom:1.1rem;">
          <div style="color:#1a2744;font-size:1.3rem;font-weight:700;">Welcome back</div>
          <div style="color:#5a6a8a;font-size:0.83rem;margin-top:3px;">
            Enter your credentials to access your account
          </div>
        </div>""", unsafe_allow_html=True)

        with st.form("login_form"):
            li_email = st.text_input("Email address", placeholder="you@findiag.com")
            li_pw    = st.text_input("Password", type="password", placeholder="Enter your password")
            rc1, rc2 = st.columns([1.2, 1])
            with rc1:
                st.checkbox("Remember me")
            with rc2:
                st.markdown("""<div style="text-align:right;padding-top:7px;">
                  <span style="color:#003087;font-size:0.81rem;font-weight:600;
                               cursor:pointer;">Forgot password?</span>
                </div>""", unsafe_allow_html=True)
            li_submit = st.form_submit_button("Sign In", use_container_width=True)

        if li_submit:
            if not li_email or not li_pw:
                st.error("Please enter both email and password.")
            else:
                user = authenticate(li_email, li_pw)
                if user:
                    st.session_state.authenticated = True
                    st.session_state.user = user
                    st.session_state.page = "dashboard"
                    st.session_state.logged_access = False
                    log_action(user["email"], f"Login — Role: {user['role']}")
                    st.rerun()
                else:
                    st.error("Incorrect email or password. Please try again.")

        st.markdown("""
        <div style="text-align:center;margin-top:1.1rem;font-size:0.83rem;">
          <span style="color:#5a6a8a;">Don't have an account?&nbsp;</span>
        </div>""", unsafe_allow_html=True)

    # ══ REGISTER FORM ═════════════════════════════════════════════════════════
    else:
        st.markdown("""
        <div style="margin-bottom:1.1rem;">
          <div style="color:#e8edf8;font-size:1.3rem;font-weight:700;">Create an account</div>
          <div style="color:#8694b5;font-size:0.83rem;margin-top:3px;">
            Fill in the details below to get started
          </div>
        </div>""", unsafe_allow_html=True)

        with st.form("register_form"):
            rg_name  = st.text_input("Full Name", placeholder="e.g. Jane Smith")
            rg_email = st.text_input("Email Address", placeholder="you@company.com")
            rg_role  = st.selectbox("Access Level",
                                    ["Viewer — read dashboards",
                                     "Analyst — full analysis tools"])
            rg_pw    = st.text_input("Password", type="password",
                                     placeholder="At least 6 characters")
            rg_pw2   = st.text_input("Confirm Password", type="password",
                                     placeholder="Repeat password")
            rg_submit = st.form_submit_button("Create Account", use_container_width=True)

        if rg_submit:
            role_map = {"Viewer — read dashboards": "Viewer",
                        "Analyst — full analysis tools": "Analyst"}
            chosen_role = role_map.get(rg_role, "Viewer")
            if not all([rg_name.strip(), rg_email.strip(), rg_pw, rg_pw2]):
                st.error("Please fill in all fields.")
            elif len(rg_pw) < 6:
                st.error("Password must be at least 6 characters.")
            elif rg_pw != rg_pw2:
                st.error("Passwords do not match.")
            else:
                ok, msg = register_user(rg_name.strip(), rg_email.strip(),
                                        rg_pw, chosen_role)
                if ok:
                    st.success(f"{msg} Please sign in to continue.")
                    st.session_state.auth_tab = "login"
                    st.rerun()
                else:
                    st.error(msg)

        st.markdown("""
        <div style="text-align:center;margin-top:1.1rem;font-size:0.83rem;">
          <span style="color:#8694b5;">Already have an account?&nbsp;</span>
        </div>""", unsafe_allow_html=True)

    # ── Footer row ────────────────────────────────────────────────────────────
    st.markdown("<div style='height:0.2rem'></div>", unsafe_allow_html=True)
    if st.button("← Back to Home", use_container_width=True, type="secondary"):
        st.session_state.page = "landing"
        st.rerun()
    st.markdown("""
    <div style="text-align:center;margin-top:0.9rem;">
      <span style="color:#243350;font-size:0.67rem;letter-spacing:0.07em;">
        Durban University of Technology &nbsp;&middot;&nbsp; Pick n Pay Financial Health System
      </span>
    </div>""", unsafe_allow_html=True)
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# DASHBOARD — authenticated users only below this line
# ══════════════════════════════════════════════════════════════════════════════
_user = st.session_state.user
role  = _user["role"]

# ── Data loaders ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    monthly  = pd.read_csv("data/monthly_financials.csv", parse_dates=["date"])
    balance  = pd.read_csv("data/balance_sheet.csv")
    variance = pd.read_csv("data/variance.csv")
    trend    = pd.read_csv("data/revenue_trend.csv")
    return monthly, balance, variance, trend

monthly_df, balance_df, variance_df, trend_df = load_data()

# ── Financial functions ───────────────────────────────────────────────────────
def calc_ratios(bs: pd.DataFrame) -> pd.DataFrame:
    df = bs.copy()
    df["current_ratio"]      = (df["current_assets"] / df["current_liabilities"]).round(2)
    df["quick_ratio"]        = (df["quick_assets"]   / df["current_liabilities"]).round(2)
    df["debt_to_equity"]     = (df["total_debt"]      / df["total_equity"]).round(2)
    df["roa"]                = (df["net_profit"]       / df["total_assets"] * 100).round(2)
    df["working_capital"]    = (df["current_assets"] - df["current_liabilities"]).round(0)
    df["inventory_turnover"] = df["inventory_turnover"].round(2)
    df["asset_turnover"]     = df["asset_turnover"].round(2)
    return df

def compute_health_score(monthly: pd.DataFrame, ratios: pd.DataFrame) -> dict:
    latest = ratios.iloc[-1]
    recent = monthly.tail(3)
    cr = latest["current_ratio"];  qr = latest["quick_ratio"]
    liq_score  = min(100, max(0, (cr / 2.0) * 50 + (qr / 1.5) * 50))
    gpm = (recent["gross_profit"] / recent["revenue"]).mean() * 100
    npm = (recent["net_profit"]   / recent["revenue"]).mean() * 100
    prof_score = min(100, max(0, gpm * 1.2 + npm * 2.5))
    de = latest["debt_to_equity"]
    solv_score = min(100, max(0, 100 - (de - 1.0) * 25))
    asset_turn = recent["revenue"].mean() / latest["total_assets"]
    eff_score  = min(100, max(0, asset_turn * 100))
    ncf_avg = recent["net_cashflow"].mean();  rev_avg = recent["revenue"].mean()
    cf_score = min(100, max(0, 50 + (ncf_avg / rev_avg) * 200))
    weights = {"liquidity":0.20,"profitability":0.25,"solvency":0.20,
               "efficiency":0.20,"cashflow":0.15}
    scores  = {"liquidity":liq_score,"profitability":prof_score,"solvency":solv_score,
               "efficiency":eff_score,"cashflow":cf_score}
    overall = sum(scores[k] * weights[k] for k in scores)
    if overall >= 75:   category, color = "Healthy",       "#16a34a"
    elif overall >= 50: category, color = "Moderate Risk", "#d97706"
    else:               category, color = "High Risk",     "#dc2626"
    return {"scores":scores,"weights":weights,"overall":round(overall,1),
            "category":category,"color":color,
            "gpm":round(gpm,1),"npm":round(npm,1),"roa":round(latest["roa"],1)}

def detect_anomalies(monthly: pd.DataFrame) -> pd.DataFrame:
    features = monthly[["revenue","operating_expenses","net_cashflow","net_profit"]].copy()
    z_scores = np.abs(stats.zscore(features.to_numpy()))
    monthly  = monthly.copy()
    monthly["max_zscore"]  = z_scores.max(axis=1).round(2)
    monthly["zscore_flag"] = monthly["max_zscore"] > 2.0
    iso = IsolationForest(contamination=0.1, random_state=42)
    monthly["iso_flag"] = iso.fit_predict(features) == -1
    monthly["anomaly"]  = monthly["zscore_flag"] | monthly["iso_flag"]
    return monthly

def build_excel(monthly, balance, variance, anomaly) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        monthly.to_excel(xw, sheet_name="Monthly Financials", index=False)
        balance.to_excel(xw, sheet_name="Balance Sheet", index=False)
        variance.to_excel(xw, sheet_name="Variance Analysis", index=False)
        anomaly[anomaly["anomaly"]].to_excel(xw, sheet_name="Anomalies", index=False)
    return buf.getvalue()

def build_pdf(health, latest_r, org, fy) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 22)
    pdf.set_text_color(11, 31, 58)
    pdf.cell(0, 12, "Pick n Pay Financial Health System", ln=True, align="C")
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(100, 116, 139)
    pdf.cell(0, 7, "AI-Enhanced Financial Health Diagnostics Report", ln=True, align="C")
    pdf.cell(0, 7, f"{org}  |  {fy}  |  Generated: {datetime.now():%Y-%m-%d %H:%M}", ln=True, align="C")
    pdf.ln(6)
    pdf.set_draw_color(226, 232, 240)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(5)

    def section(title):
        pdf.set_font("Helvetica", "B", 13)
        pdf.set_text_color(11, 31, 58)
        pdf.cell(0, 9, title, ln=True)
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(71, 85, 105)

    def row(label, value):
        pdf.cell(80, 7, label, border=0)
        pdf.cell(0, 7, str(value), ln=True)

    section("1. Overall Financial Health")
    row("Health Score:", f"{health['overall']} / 100")
    row("Risk Category:", health["category"])
    row("Gross Profit Margin:", f"{health['gpm']}%")
    row("Net Profit Margin:", f"{health['npm']}%")
    row("Return on Assets:", f"{health['roa']}%")
    pdf.ln(4)

    section("2. Dimension Scores")
    labels = {"liquidity":"Liquidity","profitability":"Profitability",
              "solvency":"Solvency","efficiency":"Efficiency","cashflow":"Cash Flow"}
    for k, lbl in labels.items():
        row(f"  {lbl}:", f"{round(health['scores'][k],1)} / 100")
    pdf.ln(4)

    section("3. Key Financial Ratios")
    row("Current Ratio:", f"{latest_r['current_ratio']:.2f}  (benchmark >= 1.5)")
    row("Quick Ratio:", f"{latest_r['quick_ratio']:.2f}  (benchmark >= 1.0)")
    row("Debt-to-Equity:", f"{latest_r['debt_to_equity']:.2f}  (benchmark <= 1.5)")
    row("Inventory Turnover:", f"{latest_r['inventory_turnover']:.2f}x  (benchmark >= 8x)")
    row("Asset Turnover:", f"{latest_r['asset_turnover']:.2f}x")
    row("Working Capital:", f"R{latest_r['working_capital']/1e3:,.1f}B")
    pdf.ln(4)

    section("4. Risk Assessment")
    cr = latest_r["current_ratio"];  de = latest_r["debt_to_equity"]
    risk_items = [
        ("Liquidity Risk",     "HIGH" if cr < 1.2 else "MEDIUM" if cr < 1.5 else "LOW"),
        ("Solvency Risk",      "HIGH" if de > 2.5 else "MEDIUM" if de > 1.8 else "LOW"),
        ("Profitability Risk", "HIGH" if health["npm"] < 5 else "MEDIUM" if health["npm"] < 12 else "LOW"),
    ]
    for label, level in risk_items:
        row(f"  {label}:", level)
    pdf.ln(4)

    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(148, 163, 184)
    pdf.cell(0, 6, "Durban University of Technology | Managerial Finance PBL | Pathway B", ln=True, align="C")
    return bytes(pdf.output())

# ── Computed data ─────────────────────────────────────────────────────────────
ratios_df   = calc_ratios(balance_df)
health      = compute_health_score(monthly_df, ratios_df)
anomaly_df  = detect_anomalies(monthly_df)
latest_r    = ratios_df.iloc[-1]
n_anomalies = anomaly_df["anomaly"].sum()
variance_df["color"] = variance_df.apply(
    lambda r: "#16a34a" if r["status"] == "Favourable" else "#dc2626", axis=1)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style="background:rgba(255,255,255,0.12);border:1px solid rgba(255,255,255,0.2);
                border-radius:10px;padding:10px 14px;margin-bottom:12px;">
      <p style="color:#ffffff;font-weight:800;font-size:0.9rem;margin:0;">{_user['name']}</p>
      <p style="color:rgba(255,255,255,0.65);font-size:0.72rem;margin:3px 0 0;">
        <span style="color:#ffffff;font-weight:700;">{role}</span>
        &nbsp;&middot;&nbsp; {_user['email']}
      </p>
    </div>""", unsafe_allow_html=True)

    st.markdown("### Dashboard Controls")
    st.markdown("---")

    if role in ["Admin", "Analyst"]:
        uploaded = st.file_uploader("Upload Financial CSV", type=["csv", "xlsx"])
        if uploaded:
            try:
                monthly_df = (pd.read_excel(uploaded, parse_dates=["date"])
                              if uploaded.name.endswith(".xlsx")
                              else pd.read_csv(uploaded, parse_dates=["date"]))
                log_action(_user["email"], f"Uploaded: {uploaded.name}")
                st.success("Custom data loaded!")
            except Exception as e:
                st.error(f"Error: {e}")
        st.markdown("---")

    period = st.selectbox("Analysis Period", ["All periods","Last 12 months","Last 6 months"])
    if period == "Last 12 months":
        monthly_df = monthly_df.tail(12).reset_index(drop=True)
    elif period == "Last 6 months":
        monthly_df = monthly_df.tail(6).reset_index(drop=True)

    zscore_threshold = (st.slider("Z-Score Threshold", 1.5, 3.5, 2.0, 0.1)
                        if role in ["Admin", "Analyst"] else 2.0)

    st.markdown("---")
    st.markdown("**Organisation**")
    org_name    = st.text_input("Name", value="Pick n Pay Stores Ltd",
                                 disabled=(role == "Viewer"))
    fiscal_year = st.selectbox("Fiscal Year", ["FY 2024","FY 2023","FY 2022"],
                                disabled=(role == "Viewer"))

    if role in ["Admin", "Analyst"]:
        st.markdown("---")
        st.markdown("**Export**")
        excel_bytes = build_excel(monthly_df, balance_df, variance_df, anomaly_df)
        st.download_button("Download Excel Report", data=excel_bytes,
                           file_name=f"FinDiag_{datetime.now():%Y%m%d}.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           use_container_width=True)
        pdf_bytes = build_pdf(health, latest_r, org_name, fiscal_year)
        st.download_button("Download PDF Report", data=pdf_bytes,
                           file_name=f"FinDiag_{datetime.now():%Y%m%d}.pdf",
                           mime="application/pdf",
                           use_container_width=True)

    st.markdown("---")
    if st.button("Sign Out", use_container_width=True):
        log_action(_user["email"], "Logout")
        st.session_state.update({"authenticated":False,"user":None,
                                  "page":"login","logged_access":False})
        st.rerun()

    st.markdown("---")
    st.caption("Pick n Pay Financial Health System\nDurban University of Technology")

# ── Log access once per session ───────────────────────────────────────────────
if not st.session_state.logged_access:
    log_action(_user["email"], f"Dashboard viewed — {role}")
    st.session_state.logged_access = True

# ── Dashboard theme — Pick n Pay official colours ─────────────────────────────
st.markdown("""<style>
[data-testid="stAppViewContainer"] {
    background-color:#f0f4f8 !important;
}
[data-testid="stSidebar"] {
    background-color:#003087 !important;
    border-right:none !important;
}
[data-testid="stSidebar"] section {background-color:#003087 !important;}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] h3 {color:rgba(255,255,255,0.9) !important;}
[data-testid="stSidebar"] .stCaption p {color:rgba(255,255,255,0.55) !important;}
[data-testid="stSidebar"] hr {border-color:rgba(255,255,255,0.15) !important;}
[data-testid="stSidebar"] .stTextInput > div > div > input {
    background:#ffffff !important;
    border:1.5px solid rgba(255,255,255,0.4) !important;
    color:#1a2744 !important;
    border-radius:8px !important;
    font-weight:500 !important;
}
[data-testid="stSidebar"] .stTextInput > div > div > input::placeholder {
    color:#94a3b8 !important;
}
[data-testid="stSidebar"] .stSelectbox > div > div {
    background:rgba(255,255,255,0.12) !important;
    border:1px solid rgba(255,255,255,0.2) !important;
    color:#ffffff !important;
    border-radius:8px !important;
}
[data-testid="stSidebar"] .stSlider label {color:rgba(255,255,255,0.7) !important;}
[data-testid="stSidebar"] .stButton > button {
    background:rgba(255,255,255,0.15) !important;
    border:1px solid rgba(255,255,255,0.25) !important;
    color:#ffffff !important;
    border-radius:8px !important;
}
[data-testid="stSidebar"] .stDownloadButton > button {
    background:rgba(255,255,255,0.15) !important;
    border:1px solid rgba(255,255,255,0.25) !important;
    color:#ffffff !important;
    border-radius:8px !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploader"] button {
    background:#c8d9f5 !important;
    border:none !important;
    color:#000000 !important;
    font-weight:700 !important;
    border-radius:8px !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploader"] small {
    color:rgba(255,255,255,0.6) !important;
}
.stTabs [data-testid="stTab"] {
    background:transparent !important;
    color:#64748b !important;
    font-weight:600 !important;
}
.stTabs [aria-selected="true"] {
    color:#003087 !important;
    border-bottom-color:#003087 !important;
}
.stTabs [data-testid="stTabsTabList"] {
    border-bottom:2px solid rgba(0,48,135,0.12) !important;
    background:#ffffff !important;
    border-radius:8px 8px 0 0 !important;
    padding:0 0.5rem !important;
}
[data-testid="stMetric"] {
    background:#ffffff !important;
    border:1px solid rgba(0,48,135,0.1) !important;
    border-radius:12px !important;
    padding:1rem !important;
    box-shadow:0 2px 8px rgba(0,48,135,0.06) !important;
}
[data-testid="stMetricLabel"] p {color:#64748b !important;}
[data-testid="stMetricValue"] {color:#1a2744 !important;white-space:normal !important;overflow:visible !important;text-overflow:unset !important;word-break:break-word !important;}
[data-testid="stMetricDelta"] {color:#16a34a !important;}
.stDataFrame {background:#ffffff !important;border-radius:10px !important;border:1px solid rgba(0,48,135,0.08) !important;}
h1,h2,h3,h4 {color:#003087 !important;}
p {color:#334155 !important;}
.stSubheader {color:#003087 !important;}
</style>""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="background:#003087;border-radius:12px;padding:1.2rem 1.8rem;margin-bottom:1rem;
     box-shadow:0 4px 16px rgba(0,48,135,0.25);">
  <div style="display:flex;justify-content:space-between;align-items:center;">
    <div style="display:flex;align-items:center;gap:14px;">
      <div style="background:#ffffff;border-radius:8px;padding:4px 10px;
                  font-size:1.1rem;font-weight:900;color:#003087;line-height:1.2;letter-spacing:-0.02em;">
        <span style="color:#e31837;">P</span>ick n<span style="color:#e31837;"> P</span>ay
      </div>
      <div>
        <div style="color:#ffffff;font-size:1.2rem;font-weight:800;letter-spacing:-0.01em;">
          PnP Financial Health System
        </div>
        <div style="color:rgba(255,255,255,0.7);font-size:0.72rem;letter-spacing:0.1em;
                    text-transform:uppercase;font-weight:500;">
          Financial Health Diagnostics
        </div>
      </div>
    </div>
    <div style="text-align:right;">
      <div style="color:#ffffff;font-weight:700;font-size:0.92rem;">{org_name}</div>
      <div style="color:rgba(255,255,255,0.65);font-size:0.72rem;margin:2px 0 6px;">
        {fiscal_year} &nbsp;&bull;&nbsp; Decision Support Platform
      </div>
      <span style="background:#e31837;color:#ffffff;font-size:0.62rem;font-weight:700;
                   padding:3px 10px;border-radius:999px;letter-spacing:0.1em;">
        &#9679; LIVE
      </span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB CONTENT FUNCTIONS
# All functions read from module-level variables set above.
# ══════════════════════════════════════════════════════════════════════════════

def render_health_scorecard():
    st.subheader("Financial Health Scorecard")
    st.caption("Composite financial health index based on liquidity, profitability, solvency, efficiency and cash flow.")

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Overall Score", f"{health['overall']} / 100",
                delta=f"{'▼' if health['overall'] < 70 else '▲'} Q-o-Q")
    col2.metric("Risk Category", health["category"])
    col3.metric("Gross Profit Margin", f"{health['gpm']}%")
    col4.metric("Net Profit Margin", f"{health['npm']}%")
    col5.metric("Return on Assets", f"{health['roa']}%")

    st.markdown("---")
    c1, c2 = st.columns([1, 1.6])

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

        _badge_styles = {
            "Healthy":       "background:#DCFCE7;color:#14532D;border:1.5px solid #86EFAC;",
            "Moderate Risk": "background:#FEF3C7;color:#78350F;border:1.5px solid #FCD34D;",
            "High Risk":     "background:#FEE2E2;color:#7F1D1D;border:1.5px solid #FCA5A5;",
        }
        st.markdown(f"""
        <div style="text-align:center;margin-top:6px;">
          <span style="{_badge_styles[health['category']]}
                        padding:5px 18px;border-radius:999px;font-size:0.9rem;
                        font-weight:700;letter-spacing:0.03em;">
            {health["category"]}
          </span>
        </div>""", unsafe_allow_html=True)

    with c2:
        dims   = list(health["scores"].keys())
        vals   = [round(v, 1) for v in health["scores"].values()]
        labels = ["Liquidity","Profitability","Solvency","Efficiency","Cash Flow"]

        fig_radar = go.Figure(go.Scatterpolar(
            r=vals + [vals[0]], theta=labels + [labels[0]],
            fill="toself", fillcolor="rgba(29,78,216,0.15)",
            line=dict(color=C_BLUE, width=2),
            marker=dict(size=6, color=C_BLUE),
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0,100], tickfont=dict(size=10))),
            showlegend=False, height=280,
            margin=dict(t=30, b=10, l=40, r=40),
            title=dict(text="Dimension Breakdown", font=dict(size=13), x=0.5),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        for lbl, val in zip(labels, vals):
            bar_col  = "#16A34A" if val >= 70 else "#D97706" if val >= 50 else "#e31837"
            text_col = "#15803D" if val >= 70 else "#92400E" if val >= 50 else "#991B1B"
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;font-size:12px;">
              <span style="width:90px;color:#64748B;font-weight:600;">{lbl}</span>
              <div style="flex:1;background:#E2E8F0;border-radius:999px;height:8px;">
                <div style="width:{val}%;background:{bar_col};height:8px;
                     border-radius:999px;"></div>
              </div>
              <span style="width:30px;text-align:right;font-weight:800;
                           color:{text_col};font-size:11px;">{val}</span>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    if len(monthly_df) >= 6:
        rolling = monthly_df.copy()
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

        fig_trend = px.line(rolling, x="date", y="rolling_score",
                            title="Financial Health Score Trend",
                            labels={"rolling_score":"Score","date":"Month"},
                            color_discrete_sequence=[C_BLUE])
        fig_trend.add_hline(y=75, line_dash="dash", line_color="#16a34a",
                            annotation_text="Healthy >= 75")
        fig_trend.add_hline(y=50, line_dash="dash", line_color="#dc2626",
                            annotation_text="High Risk < 50")
        fig_trend.update_traces(line_width=2.5)
        fig_trend.update_layout(height=280, margin=dict(t=50,b=20,l=10,r=10))
        st.plotly_chart(fig_trend, use_container_width=True)


def render_ratio_analysis():
    st.subheader("Automated Ratio Analysis Dashboard")
    st.caption("Key financial ratios calculated from balance sheet data with trend visualisation and benchmark comparison.")

    c1, c2, c3, c4 = st.columns(4)
    cr_delta  = round(latest_r["current_ratio"]  - ratios_df.iloc[-2]["current_ratio"],  2)
    qr_delta  = round(latest_r["quick_ratio"]    - ratios_df.iloc[-2]["quick_ratio"],    2)
    de_delta  = round(latest_r["debt_to_equity"] - ratios_df.iloc[-2]["debt_to_equity"], 2)
    roa_delta = round(latest_r["roa"]            - ratios_df.iloc[-2]["roa"],            2)

    c1.metric("Current Ratio",  f"{latest_r['current_ratio']:.2f}",  f"{cr_delta:+.2f} QoQ")
    c2.metric("Quick Ratio",    f"{latest_r['quick_ratio']:.2f}",    f"{qr_delta:+.2f} QoQ")
    c3.metric("Debt-to-Equity", f"{latest_r['debt_to_equity']:.2f}", f"{de_delta:+.2f} QoQ",
              delta_color="inverse")
    c4.metric("ROA",            f"{latest_r['roa']:.1f}%",           f"{roa_delta:+.1f}% QoQ")

    c5, c6, c7, c8 = st.columns(4)
    it_delta = round(latest_r["inventory_turnover"] - ratios_df.iloc[-2]["inventory_turnover"], 2)
    at_delta = round(latest_r["asset_turnover"]     - ratios_df.iloc[-2]["asset_turnover"],     2)
    wc_val   = latest_r["working_capital"]

    c5.metric("Inventory Turnover", f"{latest_r['inventory_turnover']:.2f}x", f"{it_delta:+.2f} QoQ")
    c6.metric("Asset Turnover",     f"{latest_r['asset_turnover']:.2f}x",     f"{at_delta:+.2f} QoQ")
    c7.metric("Working Capital",    f"R{wc_val/1e3:,.1f}B",
              delta="Declining" if wc_val < 3_000 else "Stable",
              delta_color="inverse" if wc_val < 3_000 else "normal")
    c8.metric("Gross Profit Margin",
              f"{(monthly_df['gross_profit'].tail(3).sum()/monthly_df['revenue'].tail(3).sum()*100):.1f}%",
              "18.1% FY2024 actual")

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        fig_liq = go.Figure()
        fig_liq.add_trace(go.Scatter(x=ratios_df["quarter"], y=ratios_df["current_ratio"],
                                     name="Current Ratio", mode="lines+markers",
                                     line=dict(color=C_BLUE, width=2.5), marker=dict(size=7)))
        fig_liq.add_trace(go.Scatter(x=ratios_df["quarter"], y=ratios_df["quick_ratio"],
                                     name="Quick Ratio", mode="lines+markers",
                                     line=dict(color=C_TEAL, width=2.5, dash="dash"),
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

    monthly_df["gpm"] = (monthly_df["gross_profit"] / monthly_df["revenue"] * 100).round(2)
    monthly_df["npm"] = (monthly_df["net_profit"]   / monthly_df["revenue"] * 100).round(2)

    fig_margin = make_subplots(specs=[[{"secondary_y": False}]])
    fig_margin.add_trace(go.Scatter(x=monthly_df["date"], y=monthly_df["gpm"],
                                    name="Gross Profit Margin %",
                                    line=dict(color=C_TEAL, width=2.5),
                                    fill="tozeroy", fillcolor="rgba(13,148,136,0.07)"))
    fig_margin.add_trace(go.Scatter(x=monthly_df["date"], y=monthly_df["npm"],
                                    name="Net Profit Margin %",
                                    line=dict(color=C_BLUE, width=2.5),
                                    fill="tozeroy", fillcolor="rgba(29,78,216,0.07)"))
    fig_margin.update_layout(title="Profit Margin Trends — Monthly", height=280,
                             margin=dict(t=50,b=20), yaxis_title="Margin %",
                             legend=dict(orientation="h", y=-0.3))
    st.plotly_chart(fig_margin, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        fig_it = go.Figure()
        fig_it.add_trace(go.Scatter(x=ratios_df["quarter"], y=ratios_df["inventory_turnover"],
                                    name="Inventory Turnover", mode="lines+markers",
                                    line=dict(color=C_INDIGO, width=2.5), marker=dict(size=7)))
        fig_it.add_hline(y=8.0, line_dash="dot", line_color="#dc2626",
                         annotation_text="Industry min ~8x")
        fig_it.update_layout(title="Inventory Turnover — Quarterly", height=250,
                             margin=dict(t=50,b=20), yaxis_title="Turns (x)", showlegend=False)
        st.plotly_chart(fig_it, use_container_width=True)

    with c2:
        fig_at = go.Figure()
        fig_at.add_trace(go.Scatter(x=ratios_df["quarter"], y=ratios_df["asset_turnover"],
                                    name="Asset Turnover", mode="lines+markers",
                                    line=dict(color=C_TEAL, width=2.5), marker=dict(size=7)))
        fig_at.update_layout(title="Asset Turnover — Quarterly", height=250,
                             margin=dict(t=50,b=20), yaxis_title="Turns (x)", showlegend=False)
        st.plotly_chart(fig_at, use_container_width=True)

    st.markdown("#### Ratio Interpretation & Recommendations")
    roa_status = "Good" if latest_r["roa"] > 0 else "Loss"
    roa_interp = ("Positive asset returns."
                  if latest_r["roa"] > 0
                  else "Negative — assets generating a loss. FY2024 reflects R3.19B net loss.")
    st.dataframe(pd.DataFrame([
        {"Ratio":"Current Ratio","Value":f"{latest_r['current_ratio']:.2f}",
         "Benchmark":"≥ 1.5","Status":"Weak",
         "Interpretation":"At ~1.05 — barely above 1.0. Working capital buffer is very thin."},
        {"Ratio":"Quick Ratio","Value":f"{latest_r['quick_ratio']:.2f}",
         "Benchmark":"≥ 1.0","Status":"Risky",
         "Interpretation":"At ~0.40 — PnP cannot cover current liabilities with liquid assets alone."},
        {"Ratio":"Debt-to-Equity","Value":f"{latest_r['debt_to_equity']:.2f}",
         "Benchmark":"≤ 1.5","Status":"High",
         "Interpretation":"Leverage is very high due to equity erosion from FY2024 loss."},
        {"Ratio":"ROA","Value":f"{latest_r['roa']:.1f}%",
         "Benchmark":"≥ 5%","Status":roa_status,"Interpretation":roa_interp},
        {"Ratio":"Inventory Turnover","Value":f"{latest_r['inventory_turnover']:.2f}x",
         "Benchmark":"≥ 8.0x","Status":"Watch",
         "Interpretation":"Declining turns signal slower stock movement."},
        {"Ratio":"Asset Turnover","Value":f"{latest_r['asset_turnover']:.2f}x",
         "Benchmark":"≥ 3.0x (retail)","Status":"Monitor",
         "Interpretation":"Revenue generated per rand of assets. Declining trend warrants attention."},
    ]), use_container_width=True, hide_index=True)


def render_variance_monitor():
    st.subheader("Variance Analysis Monitoring System")
    st.caption("Compares budgeted vs actual financial performance. Flags unfavourable variances and overspend alerts.")

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
    st.markdown("##### Budget vs Actual — FY2024 (ZAR millions)")
    display_var = variance_df.copy()
    for col in ["actual_2024","budget_2024","actual_2023"]:
        display_var[col] = display_var[col].apply(lambda x: f"R{x/1000:+.2f}B")
    display_var["variance"]     = display_var["variance"].apply(lambda x: f"R{x/1000:+.2f}B")
    display_var["variance_pct"] = display_var["variance_pct"].apply(lambda x: f"{x:+.1f}%")
    display_var["yoy_change"]   = display_var["yoy_change"].apply(lambda x: f"R{x/1000:+.2f}B")
    display_var["yoy_pct"]      = display_var["yoy_pct"].apply(lambda x: f"{x:+.1f}%")
    display_var = display_var.rename(columns={
        "category":"Category","budget_2024":"Budget 2024","actual_2024":"Actual 2024",
        "actual_2023":"Actual 2023","variance":"Budget Variance",
        "variance_pct":"Variance %","yoy_change":"YoY Change","yoy_pct":"YoY %",
        "type":"Type","status":"Status",
    })
    st.dataframe(display_var[["Category","Type","Budget 2024","Actual 2024","Budget Variance",
                               "Variance %","Actual 2023","YoY Change","YoY %","Status"]],
                 use_container_width=True, hide_index=True)

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        fig_var = go.Figure()
        fig_var.add_trace(go.Bar(name="Budget 2024", x=variance_df["category"],
                                 y=variance_df["budget_2024"] / 1000,
                                 marker_color="rgba(29,78,216,0.55)",
                                 marker_line=dict(color=C_BLUE, width=1)))
        fig_var.add_trace(go.Bar(name="Actual 2024", x=variance_df["category"],
                                 y=variance_df["actual_2024"] / 1000,
                                 marker_color="rgba(220,38,38,0.55)",
                                 marker_line=dict(color=C_RED, width=1)))
        fig_var.add_trace(go.Bar(name="Actual 2023", x=variance_df["category"],
                                 y=variance_df["actual_2023"] / 1000,
                                 marker_color="rgba(13,148,136,0.45)",
                                 marker_line=dict(color=C_TEAL, width=1)))
        fig_var.update_layout(barmode="group", title="Budget vs Actual vs Prior Year (R billions)",
                              height=300, margin=dict(t=50,b=20), yaxis_title="R billions",
                              legend=dict(orientation="h", y=-0.3), xaxis_tickangle=-20)
        st.plotly_chart(fig_var, use_container_width=True)

    with c2:
        fig_wfall = go.Figure(go.Bar(
            x=variance_df["category"], y=variance_df["variance"] / 1000,
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

    fig_trend_annual = go.Figure()
    fig_trend_annual.add_trace(go.Bar(x=trend_df["year"].astype(str),
                                      y=trend_df["revenue"] / 1000,
                                      name="Revenue", marker_color="rgba(29,78,216,0.55)"))
    fig_trend_annual.update_layout(
        title="Pick n Pay Annual Revenue Trend (2021–2025, R billions)",
        height=260, margin=dict(t=50,b=20), yaxis_title="R billions", showlegend=False)
    st.plotly_chart(fig_trend_annual, use_container_width=True)

    fig_rev = go.Figure()
    fig_rev.add_trace(go.Bar(x=monthly_df["date"], y=monthly_df["revenue"] / 1000,
                             name="Revenue", marker_color="rgba(29,78,216,0.4)"))
    fig_rev.add_trace(go.Scatter(x=monthly_df["date"],
                                 y=monthly_df["operating_expenses"] / 1000,
                                 name="Operating Expenses",
                                 line=dict(color="#dc2626", width=2, dash="dash")))
    fig_rev.update_layout(title="Monthly Revenue vs Operating Expenses (R millions)",
                          height=280, margin=dict(t=50,b=20), yaxis_title="R millions",
                          legend=dict(orientation="h", y=-0.3))
    st.plotly_chart(fig_rev, use_container_width=True)

    unfav = variance_df[variance_df["status"] == "Unfavourable"]
    if not unfav.empty:
        st.markdown("#### Automated Variance Alerts")
        for _, row in variance_df.iterrows():
            if row["status"] == "Unfavourable":
                pct = abs(row["variance_pct"])
                if pct > 15:
                    bg, border, icon = "#FFF1F2","#E11D48","🔴"
                    note = "Immediate executive review required."
                else:
                    bg, border, icon = "#FFFBEB","#D97706","🟡"
                    note = "Monitor closely — within tolerance."
                st.markdown(f"""
                <div style="background:{bg};border-left:4px solid {border};
                     padding:10px 16px;border-radius:0 10px 10px 0;margin:6px 0;
                     box-shadow:0 1px 3px rgba(0,0,0,0.06);">
                  <span style="font-weight:700;color:#0F172A;font-size:13px;">
                    {icon} {row['category']}
                  </span>
                  <span style="color:#475569;font-size:12px;">
                    &nbsp;&mdash; R{abs(row['variance'])/1000:.2f}B unfavourable
                    ({pct:.1f}% vs budget). {note}
                  </span>
                </div>""", unsafe_allow_html=True)


def render_anomaly_detection():
    st.subheader("Financial Anomaly Detection Model")
    st.caption("Z-Score analysis and Isolation Forest identify unusual financial patterns, expense spikes, and potential fraud indicators.")

    features = anomaly_df[["revenue","operating_expenses","net_cashflow","net_profit"]]
    z = np.abs(np.asarray(stats.zscore(features.to_numpy())))
    _adf = anomaly_df.copy()
    _adf["max_zscore"]  = z.max(axis=1).round(2)
    _adf["zscore_flag"] = _adf["max_zscore"] > zscore_threshold
    _adf["anomaly"]     = _adf["zscore_flag"] | _adf["iso_flag"]

    _n = _adf["anomaly"].sum()
    max_z = _adf["max_zscore"].max()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Anomalies Detected", int(_n), "This period")
    c2.metric("Max Z-Score", f"{max_z:.2f}", f"Threshold: {zscore_threshold}")
    c3.metric("Model", "Z-Score + IF")
    c4.metric("Fraud Risk Signal", "Low" if _n < 4 else "Medium")

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        colors = [C_RED if a else C_BLUE for a in _adf["anomaly"]]
        sizes  = [14 if a else 8 for a in _adf["anomaly"]]
        fig_sc = go.Figure()
        fig_sc.add_trace(go.Scatter(
            x=_adf["date"], y=_adf["operating_expenses"] / 1e3,
            mode="markers",
            marker=dict(color=colors, size=sizes, line=dict(width=1, color="white")),
            text=[f"Z={z:.2f}" for z in _adf["max_zscore"]],
            hovertemplate="<b>%{x|%b %Y}</b><br>Expenses: R%{y:.0f}K<br>%{text}<extra></extra>",
        ))
        fig_sc.update_layout(title="Operating Expense Anomalies (Z-Score)",
                             height=300, margin=dict(t=50,b=20), yaxis_title="Expenses (R000)")
        st.plotly_chart(fig_sc, use_container_width=True)

    with c2:
        bar_colors = [C_RED if z > zscore_threshold else "#94a3b8"
                      for z in _adf["max_zscore"]]
        fig_z = go.Figure(go.Bar(
            x=_adf["date"].dt.strftime("%b %y"), y=_adf["max_zscore"],
            marker_color=bar_colors,
            text=[f"{z:.2f}" for z in _adf["max_zscore"]], textposition="outside",
        ))
        fig_z.add_hline(y=zscore_threshold, line_dash="dash", line_color="#dc2626",
                        annotation_text=f"Threshold ({zscore_threshold})")
        fig_z.update_layout(title="Max Z-Score Per Month", height=300,
                            margin=dict(t=50,b=20), yaxis_title="Z-Score", showlegend=False)
        st.plotly_chart(fig_z, use_container_width=True)

    flagged = _adf[_adf["anomaly"]].copy()
    if not flagged.empty:
        st.markdown("#### Flagged Anomalous Records")
        flagged["date"] = flagged["date"].dt.strftime("%b %Y")
        for c in ["revenue","operating_expenses","net_profit","net_cashflow"]:
            flagged[c] = flagged[c].apply(lambda x: f"R{x/1e3:.0f}K")
        flagged["anomaly_type"] = flagged.apply(
            lambda r: "Z-Score + IsoForest" if r["zscore_flag"] and r["iso_flag"]
                      else "Z-Score" if r["zscore_flag"] else "Isolation Forest", axis=1)
        st.dataframe(flagged[["date","revenue","operating_expenses","net_profit",
                               "net_cashflow","max_zscore","anomaly_type"]].rename(columns={
            "date":"Month","revenue":"Revenue","operating_expenses":"Op. Expenses",
            "net_profit":"Net Profit","net_cashflow":"Net Cash Flow",
            "max_zscore":"Max Z-Score","anomaly_type":"Detection Method",
        }), use_container_width=True, hide_index=True)

    fig_multi = make_subplots(rows=2, cols=2, subplot_titles=[
        "Revenue","Operating Expenses","Net Profit","Net Cash Flow"])
    for i, (col, name, color) in enumerate([
        ("revenue",             "Revenue",       C_BLUE),
        ("operating_expenses",  "Op. Expenses",  C_RED),
        ("net_profit",          "Net Profit",     C_TEAL),
        ("net_cashflow",        "Cash Flow",      "#f59e0b"),
    ]):
        r, c = i // 2 + 1, i % 2 + 1
        fig_multi.add_trace(go.Scatter(
            x=_adf["date"], y=_adf[col] / 1e3, name=name,
            line=dict(color=color, width=1.8), mode="lines+markers",
            marker=dict(color=[C_RED if a else color for a in _adf["anomaly"]],
                        size=[10 if a else 4 for a in _adf["anomaly"]])
        ), row=r, col=c)
    fig_multi.update_layout(height=400, showlegend=False,
                             title_text="Financial Features — Anomalies Highlighted in Red",
                             margin=dict(t=60,b=20))
    st.plotly_chart(fig_multi, use_container_width=True)


def render_early_warning():
    st.subheader("Early-Warning Financial Risk Dashboard")
    st.caption("Real-time risk indicators, automated alerts and 6-month cash flow forecast to support proactive managerial decisions.")

    def risk_level(thresholds):
        if thresholds[0]: return "🔴 HIGH"
        if thresholds[1]: return "🟡 MEDIUM"
        return "🟢 LOW"

    liq = latest_r["current_ratio"]
    qr  = latest_r["quick_ratio"]
    de  = latest_r["debt_to_equity"]

    risks = {
        "Liquidity Risk":     risk_level([liq < 1.2, liq < 1.5]),
        "Cash Flow Risk":     risk_level([anomaly_df["net_cashflow"].tail(3).mean() < 0,
                                          anomaly_df["net_cashflow"].tail(3).mean() < 50000]),
        "Solvency Risk":      risk_level([de > 2.5, de > 1.8]),
        "Profitability Risk": risk_level([health["npm"] < 5, health["npm"] < 12]),
        "Anomaly / Fraud":    risk_level([n_anomalies > 5, n_anomalies > 2]),
        "Operational Risk":   risk_level([abs(variance_df["variance_pct"]).mean() > 25,
                                          abs(variance_df["variance_pct"]).mean() > 12]),
    }

    cols = st.columns(3)
    for i, (label, level) in enumerate(risks.items()):
        if "HIGH"   in level: bg,border,tc = "#FFF1F2","#E11D48","#9F1239"
        elif "MEDIUM" in level: bg,border,tc = "#FFFBEB","#D97706","#78350F"
        else: bg,border,tc = "#F0FDF4","#16A34A","#14532D"
        cols[i % 3].markdown(f"""
        <div style="background:{bg};border:1.5px solid {border};border-radius:12px;
             padding:14px 10px;text-align:center;margin-bottom:8px;
             box-shadow:0 2px 6px rgba(0,0,0,0.07);">
          <div style="font-size:1.5rem;line-height:1;">{level.split()[0]}</div>
          <div style="font-size:0.7rem;font-weight:800;color:{tc};margin-top:5px;
                      letter-spacing:0.08em;text-transform:uppercase;">{label}</div>
          <div style="font-size:0.7rem;font-weight:600;color:{border};margin-top:2px;">
            &#9679; {level.split()[-1]}
          </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    c1, c2 = st.columns([1.2, 1])

    with c1:
        st.markdown("#### Active Alerts")
        alerts = []
        if qr < 1.0:
            alerts.append(("critical","Quick Ratio below 1.0",
                           f"At {qr:.2f} — organisation cannot cover current liabilities with liquid assets."))
        if de > 1.8:
            alerts.append(("warning",f"Debt-to-Equity elevated at {de:.2f}",
                           "Approaching high-risk threshold of 2.0. Limit new debt obligations."))
        if liq < 1.5:
            alerts.append(("warning","Current Ratio below benchmark",
                           f"At {liq:.2f} vs 1.5 benchmark. Working capital buffer is shrinking."))
        unfav_count = (variance_df["status"] == "Unfavourable").sum()
        if unfav_count >= 3:
            alerts.append(("critical",f"{unfav_count} unfavourable budget variances",
                           "Operating expenses and payroll are significantly over budget."))
        if n_anomalies > 2:
            alerts.append(("warning",f"{int(n_anomalies)} financial anomalies detected",
                           "Expense spikes and cash flow abnormalities require investigation."))
        alerts.append(("info","Revenue growth remains positive",
                       "YoY revenue growth is a key stabilising factor for the organisation."))

        for level, title, desc in alerts:
            if level == "critical": bg,border,icon = "#FFF1F2","#E11D48","🔴"
            elif level == "warning": bg,border,icon = "#FFFBEB","#D97706","🟡"
            else: bg,border,icon = "#EFF6FF","#2563EB","🔵"
            st.markdown(f"""
            <div style="background:{bg};border-left:4px solid {border};
                 padding:10px 16px;border-radius:0 10px 10px 0;margin:6px 0;
                 box-shadow:0 1px 4px rgba(0,0,0,0.06);">
              <p style="font-weight:700;color:#0F172A;font-size:13px;margin:0;">
                {icon}&nbsp; {title}
              </p>
              <p style="color:#64748B;font-size:11.5px;margin:3px 0 0;">{desc}</p>
            </div>""", unsafe_allow_html=True)

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
            r=risk_vals + [risk_vals[0]], theta=dim_labels + [dim_labels[0]],
            fill="toself", fillcolor="rgba(220,38,38,0.12)",
            line=dict(color=C_RED, width=2), marker=dict(size=6, color=C_RED),
        ))
        fig_rr.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0,100])),
            showlegend=False, height=310,
            title=dict(text="Risk Radar", font=dict(size=13), x=0.5),
            margin=dict(t=50,b=20,l=30,r=30),
        )
        st.plotly_chart(fig_rr, use_container_width=True)

    st.markdown("---")
    last_inflow  = monthly_df["cash_inflow"].tail(3).mean()
    last_outflow = monthly_df["cash_outflow"].tail(3).mean()
    forecast_months = pd.date_range(
        start=monthly_df["date"].max() + pd.DateOffset(months=1), periods=6, freq="MS")
    inflow_f  = [last_inflow  * (1+0.015*i) * np.random.uniform(0.97,1.03) for i in range(1,7)]
    outflow_f = [last_outflow * (1+0.018*i) * np.random.uniform(0.97,1.03) for i in range(1,7)]
    net_f     = [i-o for i,o in zip(inflow_f, outflow_f)]

    fig_fcst = go.Figure()
    fig_fcst.add_trace(go.Scatter(x=forecast_months, y=[v/1e3 for v in inflow_f],
                                  name="Projected Inflow", line=dict(color=C_TEAL, width=2.5),
                                  fill="tozeroy", fillcolor="rgba(13,148,136,0.08)"))
    fig_fcst.add_trace(go.Scatter(x=forecast_months, y=[v/1e3 for v in outflow_f],
                                  name="Projected Outflow",
                                  line=dict(color="#dc2626", width=2.5, dash="dash"),
                                  fill="tozeroy", fillcolor="rgba(220,38,38,0.06)"))
    fig_fcst.add_trace(go.Bar(x=forecast_months, y=[v/1e3 for v in net_f],
                              name="Net Cash Flow",
                              marker_color=["#16a34a" if v > 0 else "#dc2626" for v in net_f],
                              opacity=0.5))
    fig_fcst.update_layout(title="6-Month Cash Flow Forecast", height=300,
                           margin=dict(t=50,b=20), yaxis_title="R thousands",
                           legend=dict(orientation="h", y=-0.3))
    st.plotly_chart(fig_fcst, use_container_width=True)

    st.markdown("#### Strategic Recommendations for Management")
    for icon, title, desc in [
        ("🔴","Improve liquidity immediately",
         f"Quick ratio is {qr:.2f}. Accelerate debtor collections, review stock holding, and renegotiate short-term credit lines."),
        ("🟡","Control operating expenditure",
         "Operating expenses are significantly over budget. Conduct a cost audit and freeze discretionary spending."),
        ("🟡","Monitor leverage levels",
         f"D/E at {de:.2f}. Avoid additional long-term debt until the ratio falls below 1.5."),
        ("🟢","Leverage revenue growth",
         f"Revenue growing at {health['gpm']:.1f}% gross margin. Channel growth into cash reserves to rebuild liquidity."),
    ]:
        bg, border = (("#FFF1F2","#E11D48") if icon=="🔴"
                      else ("#FFFBEB","#D97706") if icon=="🟡"
                      else ("#F0FDF4","#16A34A"))
        st.markdown(f"""
        <div style="background:{bg};border-left:5px solid {border};
             padding:12px 18px;border-radius:0 12px 12px 0;margin:8px 0;
             box-shadow:0 2px 6px rgba(0,0,0,0.07);">
          <p style="font-weight:800;color:#0F172A;font-size:13px;margin:0;">{icon}&nbsp; {title}</p>
          <p style="color:#475569;font-size:12px;margin:4px 0 0;line-height:1.5;">{desc}</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-top:2rem;border-top:1px solid #E2E8F0;padding-top:14px;
         display:flex;justify-content:space-between;align-items:center;">
      <span style="color:#94A3B8;font-size:11px;font-weight:600;">
        Pick n Pay Financial Health System &mdash; AI-Enhanced Financial Diagnostics
      </span>
      <span style="color:#94A3B8;font-size:11px;">
        Managerial Finance PBL &bull; Pathway B &bull; Durban University of Technology
      </span>
    </div>""", unsafe_allow_html=True)


def render_admin_panel():
    st.subheader("Admin Panel — User Management")
    st.caption("Manage system users, roles, and access permissions.")

    users = load_users()

    # Current users table
    st.markdown("#### Current Users")
    users_display = pd.DataFrame([
        {"Name": u["name"], "Email": u["email"], "Role": u["role"]} for u in users
    ])
    st.dataframe(users_display, use_container_width=True, hide_index=True)

    st.markdown("---")
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("#### Add New User")
        with st.form("add_user_form"):
            new_name  = st.text_input("Full Name")
            new_email = st.text_input("Email Address")
            new_role  = st.selectbox("Role", ["Viewer", "Analyst", "Admin"])
            new_pw    = st.text_input("Temporary Password", type="password")
            add_sub   = st.form_submit_button("Add User", type="primary",
                                               use_container_width=True)
        if add_sub:
            if not all([new_name, new_email, new_role, new_pw]):
                st.error("All fields are required.")
            elif any(u["email"].lower() == new_email.lower() for u in users):
                st.error("A user with that email already exists.")
            else:
                users.append({"email": new_email, "name": new_name,
                               "role": new_role, "password": _hash(new_pw)})
                save_users(users)
                log_action(_user["email"], f"Added user: {new_email} ({new_role})")
                st.success(f"User {new_name} added as {new_role}.")
                st.rerun()

    with c2:
        st.markdown("#### Change User Role")
        non_self = [u for u in users if u["email"] != _user["email"]]
        if non_self:
            with st.form("change_role_form"):
                target_email = st.selectbox("Select User",
                                            [u["email"] for u in non_self])
                new_role_val = st.selectbox("New Role", ["Viewer","Analyst","Admin"],
                                            key="new_role_sel")
                role_sub = st.form_submit_button("Update Role", use_container_width=True)
            if role_sub:
                for u in users:
                    if u["email"] == target_email:
                        u["role"] = new_role_val
                        break
                save_users(users)
                log_action(_user["email"], f"Changed role of {target_email} to {new_role_val}")
                st.success(f"Role updated.")
                st.rerun()
        else:
            st.info("No other users to manage.")

        st.markdown("#### Remove User")
        non_self_emails = [u["email"] for u in non_self]
        if non_self_emails:
            del_email = st.selectbox("Select user to remove", non_self_emails,
                                     key="del_email")
            if st.button("Remove User", type="primary", use_container_width=True):
                users = [u for u in users if u["email"] != del_email]
                save_users(users)
                log_action(_user["email"], f"Removed user: {del_email}")
                st.success(f"User {del_email} removed.")
                st.rerun()
        else:
            st.info("No other users to remove.")

    st.markdown("---")
    st.markdown("#### Default Login Credentials")
    st.markdown("""
    | Role | Email | Password |
    |------|-------|----------|
    | Admin | admin@findiag.com | Admin@123 |
    | Analyst | analyst@findiag.com | Analyst@123 |
    | Viewer | viewer@findiag.com | Viewer@123 |
    """)
    st.caption("Change passwords by removing and re-adding the user.")


def render_audit_log():
    st.subheader("System Audit Log")
    st.caption("Complete log of all user actions — logins, uploads, and dashboard views.")

    audit = load_audit()
    if audit.empty:
        st.info("No audit records yet.")
        return

    audit = audit.sort_values("timestamp", ascending=False).reset_index(drop=True)

    c1, c2 = st.columns(2)
    with c1:
        user_filter = st.selectbox("Filter by user", ["All"] + sorted(audit["user"].unique().tolist()))
    with c2:
        action_search = st.text_input("Search action", placeholder="e.g. Login")

    filtered = audit.copy()
    if user_filter != "All":
        filtered = filtered[filtered["user"] == user_filter]
    if action_search:
        filtered = filtered[filtered["action"].str.contains(action_search, case=False, na=False)]

    st.dataframe(filtered, use_container_width=True, hide_index=True)
    st.caption(f"{len(filtered)} records shown (of {len(audit)} total)")

    # Activity chart
    if len(audit) > 1:
        audit["date"] = pd.to_datetime(audit["timestamp"]).dt.date.astype(str)
        daily = audit.groupby("date").size().reset_index(name="actions")
        fig = px.bar(daily, x="date", y="actions", title="Daily Activity",
                     color_discrete_sequence=[C_BLUE])
        fig.update_layout(height=220, margin=dict(t=40,b=20))
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# ROLE-BASED TAB RENDERING
# ══════════════════════════════════════════════════════════════════════════════
if role == "Viewer":
    tabs = st.tabs(["Health Scorecard", "Early Warning"])
    with tabs[0]: render_health_scorecard()
    with tabs[1]: render_early_warning()

elif role == "Analyst":
    tabs = st.tabs(["Health Scorecard", "Ratio Analysis",
                    "Variance Monitor", "Anomaly Detection", "Early Warning"])
    with tabs[0]: render_health_scorecard()
    with tabs[1]: render_ratio_analysis()
    with tabs[2]: render_variance_monitor()
    with tabs[3]: render_anomaly_detection()
    with tabs[4]: render_early_warning()

else:  # Admin
    tabs = st.tabs(["Health Scorecard", "Ratio Analysis", "Variance Monitor",
                    "Anomaly Detection", "Early Warning", "Admin Panel", "Audit Log"])
    with tabs[0]: render_health_scorecard()
    with tabs[1]: render_ratio_analysis()
    with tabs[2]: render_variance_monitor()
    with tabs[3]: render_anomaly_detection()
    with tabs[4]: render_early_warning()
    with tabs[5]: render_admin_panel()
    with tabs[6]: render_audit_log()
