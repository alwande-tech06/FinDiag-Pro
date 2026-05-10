# FinDiag Pro — AI-Enhanced Financial Health Diagnostics
### Pathway B | Managerial Finance PBL | Durban University of Technology

---

## Project Overview

FinDiag Pro is a Python-based financial diagnostic system that monitors organisational
financial health, detects risks and inefficiencies, and provides early-warning indicators
for potential financial distress. It functions as an AI-enhanced decision-support tool
for managers and financial analysts.

---

## Features

| Module | Description |
|--------|-------------|
| **Health Scorecard** | Composite 0–100 score across 5 financial dimensions |
| **Ratio Analysis** | Automated calculation of 7 key financial ratios with trends |
| **Variance Monitor** | Budget vs Actual comparison with favourable/unfavourable flags |
| **Anomaly Detection** | Z-Score + Isolation Forest to flag unusual patterns |
| **Early Warning** | Risk radar, cash flow forecast, and strategic recommendations |

---

## Setup Instructions

### Step 1 — Install Python (3.10 or higher)
Download from https://www.python.org/downloads/

### Step 2 — Install dependencies
Open a terminal in the project folder and run:
```
pip install -r requirements.txt
```

### Step 3 — Generate sample data
```
python generate_data.py
```
This creates three CSV files in the `data/` folder:
- `monthly_financials.csv` — 24 months of revenue, expenses, cash flow
- `balance_sheet.csv`      — 8 quarters of balance sheet data
- `variance.csv`           — Budget vs Actual for 6 categories

### Step 4 — Run the dashboard
```
streamlit run app.py
```
The dashboard opens automatically at http://localhost:8501

---

## Using Your Own Data

Upload your own CSV file via the sidebar file uploader.
Required columns: `date, revenue, cogs, operating_expenses, cash_inflow, cash_outflow`

---

## Project Structure

```
findiag/
├── app.py               ← Main Streamlit dashboard
├── generate_data.py     ← Simulated financial data generator
├── requirements.txt     ← Python library dependencies
├── README.md            ← This file
└── data/
    ├── monthly_financials.csv
    ├── balance_sheet.csv
    └── variance.csv
```

---

## Technical Stack

- **Python** — core language
- **Streamlit** — interactive web dashboard
- **Plotly** — financial charts and visualisations
- **Pandas / NumPy** — data processing and ratio calculations
- **Scikit-learn** — Isolation Forest anomaly detection
- **SciPy** — Z-Score statistical analysis

---

## AI Techniques Used

- **Z-Score Analysis** — flags data points more than N standard deviations from the mean
- **Isolation Forest** — unsupervised ML model that isolates anomalies in financial features
- **Composite Scoring** — weighted aggregation of financial dimension scores

---

*Group 31 & 18 | IPRT301 / Managerial Finance | DUT Faculty of Accounting and Informatics*
