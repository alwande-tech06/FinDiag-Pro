# FinDiag Pro — Pick n Pay Financial Health System
### Pathway B | Managerial Finance PBL | Durban University of Technology

---

## Project Overview

FinDiag Pro is an AI-enhanced financial diagnostic system built for Pick n Pay Stores Ltd. It provides real-time financial monitoring, anomaly detection, and early-warning alerts to support executive decision-making. The system functions as a multi-user web dashboard with role-based access control, automated ratio analysis, and exportable reports.

---

## Features

| Module | Description |
|--------|-------------|
| **Health Scorecard** | Composite 0–100 score across 5 financial dimensions |
| **Ratio Analysis** | Automated calculation of 7 key financial ratios with trend charts |
| **Variance Monitor** | Budget vs Actual comparison with favourable/unfavourable flags |
| **Anomaly Detection** | Z-Score + Isolation Forest ML to flag unusual financial patterns |
| **Early Warning** | Risk radar, 6-month cash flow forecast, and strategic recommendations |
| **PDF Report** | One-click branded diagnostic report download |
| **Excel Export** | Full data export (financials, ratios, anomalies) to `.xlsx` |
| **User Management** | Role-based access (Admin / Analyst / Viewer) with audit logging |

---

## User Roles

| Role | Access |
|------|--------|
| **Admin** | Full access — dashboards, exports, user settings |
| **Analyst** | Full access — dashboards and exports |
| **Viewer** | Read-only — dashboards only, no exports or settings |

### Default credentials

| Email | Password | Role |
|-------|----------|------|
| admin@findiag.com | Admin@123 | Admin |
| analyst@findiag.com | Analyst@123 | Analyst |
| viewer@findiag.com | Viewer@123 | Viewer |

> Change default passwords after first login in a production deployment.

---

## Setup Instructions

### Step 1 — Install Python (3.10 or higher)
Download from https://www.python.org/downloads/

### Step 2 — Install dependencies
```
pip install -r requirements.txt
```

### Step 3 — Generate sample data
```
python generate_data.py
```
Creates three CSV files in the `data/` folder:
- `monthly_financials.csv` — 24 months of revenue, expenses, and cash flow
- `balance_sheet.csv` — 8 quarters of balance sheet data
- `variance.csv` — Budget vs Actual for 6 categories

### Step 4 — Run the dashboard
```
streamlit run app.py
```
The dashboard opens at http://localhost:8501

---

## Using Your Own Data

Upload a custom CSV via the sidebar file uploader.  
Required columns: `date, revenue, cogs, operating_expenses, cash_inflow, cash_outflow`

---

## Project Structure

```
findiag/
├── app.py               ← Main Streamlit dashboard (auth, dashboard, exports)
├── generate_data.py     ← Simulated financial data generator
├── requirements.txt     ← Python library dependencies
├── README.md            ← This file
└── data/
    ├── monthly_financials.csv
    ├── balance_sheet.csv
    ├── variance.csv
    └── users.json       ← User accounts (auto-created on first run)
```

---

## Technical Stack

| Library | Purpose |
|---------|---------|
| **Python 3.10+** | Core language |
| **Streamlit** | Interactive web dashboard |
| **Plotly** | Financial charts and visualisations |
| **Pandas / NumPy** | Data processing and ratio calculations |
| **Scikit-learn** | Isolation Forest anomaly detection |
| **SciPy** | Z-Score statistical analysis |
| **fpdf2** | PDF report generation |
| **openpyxl** | Excel export |

---

## AI Techniques Used

- **Z-Score Analysis** — flags data points more than N standard deviations from the mean
- **Isolation Forest** — unsupervised ML model that isolates anomalies across financial features
- **Composite Scoring** — weighted aggregation of financial dimension scores into a single health index
- **Cash Flow Forecasting** — trend-based 6-month forward projection with confidence bands

---

## Deployment (Streamlit Cloud)

1. Push the repository to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect the repo.
3. Set the main file path to `app.py`.
4. All dependencies in `requirements.txt` are installed automatically.




