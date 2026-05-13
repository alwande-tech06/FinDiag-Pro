import pandas as pd

monthly  = pd.read_csv("data/monthly_financials.csv", parse_dates=["date"])
balance  = pd.read_csv("data/balance_sheet.csv")
variance = pd.read_csv("data/variance.csv")
trend    = pd.read_csv("data/revenue_trend.csv")

print("=== REVENUE TREND (system) vs DOCUMENT ===")
doc = {2021:94.66, 2022:99.63, 2023:108.57, 2024:114.94, 2025:121.56}
for _, row in trend.iterrows():
    yr  = int(row["year"])
    sys = row["revenue"] / 1000
    tgt = doc.get(yr, "—")
    match = "OK" if isinstance(tgt, float) and abs(sys - tgt) < 0.01 else "DIFF"
    print(f"  {yr}: system={sys:.2f}B  doc={tgt}B  [{match}]")

print()
print("=== ANNUAL ACTUALS (from monthly model) ===")
monthly["year"] = monthly["date"].dt.year
ann = monthly.groupby("year").agg(
    rev=("revenue","sum"),
    np=("net_profit","sum"),
    opex=("operating_expenses","sum"),
    gp=("gross_profit","sum"),
).round(0)
ann["npm"] = (ann["np"]  / ann["rev"] * 100).round(2)
ann["gpm"] = (ann["gp"]  / ann["rev"] * 100).round(2)
for yr, row in ann.iterrows():
    print(f"  {yr}: Rev=R{row['rev']/1000:.2f}B  NP=R{row['np']/1000:.2f}B"
          f"  NPM={row['npm']:.2f}%  GPM={row['gpm']:.2f}%  OpEx=R{row['opex']/1000:.2f}B")

print()
print("=== DOCUMENT TARGETS ===")
print("  2023: Rev=R108.57B  NP=R1.17B  NPM=1.08%  GPM~19%  OpEx=R20.2B")
print("  2024: Rev=R114.94B  NP=-R3.19B NPM=-2.87% GPM=18.1% OpEx=R22.6B")

print()
print("=== BALANCE SHEET RATIOS (Q8 = 2024-Q4) ===")
last = balance.iloc[-1]
cr  = last["current_assets"]  / last["current_liabilities"]
qr  = last["quick_assets"]    / last["current_liabilities"]
it  = last["inventory_turnover"]
print(f"  Current Ratio:      {cr:.3f}  (doc target ~1.05)")
print(f"  Quick Ratio:        {qr:.3f}  (doc target ~0.40)")
print(f"  Inventory Turnover: {it:.2f}x  (doc target ~8.8x)")

print()
print("=== VARIANCE TABLE ===")
print(variance[["category","actual_2024","actual_2023"]].to_string(index=False))
