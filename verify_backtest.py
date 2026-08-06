# verify_backtest.py
# Run this script to verify backtesting is working correctly
# Usage: python verify_backtest.py

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import yfinance as yf
import pandas as pd
import numpy as np

print("=" * 60)
print("BACKTESTING VERIFICATION SCRIPT")
print("=" * 60)

# -----------------------------------------------
# STEP 1: Fetch real data
# -----------------------------------------------
print("\n📥 STEP 1: Fetching real AAPL 4H data...")
df = yf.download("AAPL", period="730d", interval="1h", progress=False, auto_adjust=True)

if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)
df.columns = [c.lower() for c in df.columns]
df = df.reset_index()
df = df[df["volume"] > 0]
df = df.rename(columns={df.columns[0]: "datetime"})
df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
df = df.set_index("datetime")
df = df.resample("4h").agg({
    "open": "first", "high": "max",
    "low": "min", "close": "last", "volume": "sum"
}).dropna().reset_index()

delta = df["close"].diff()
gain  = delta.clip(lower=0).rolling(14).mean()
loss  = (-delta.clip(upper=0)).rolling(14).mean()
df["rsi"] = 100 - (100 / (1 + gain / loss))

print(f"✅ Fetched {len(df)} candles of 4H data")
print(f"   Date range: {df['datetime'].iloc[0].strftime('%Y-%m-%d')} → {df['datetime'].iloc[-1].strftime('%Y-%m-%d')}")
print(f"   Price range: ${df['close'].min():.2f} → ${df['close'].max():.2f}")

# -----------------------------------------------
# STEP 2: Run backtest (default strategy)
# -----------------------------------------------
print("\n🔁 STEP 2: Running backtest (advanced_combined)...")
from app.backtesting import run_backtest
trades, metrics, equity = run_backtest(df, strategy="advanced_combined", initial_capital=10000)
print(f"✅ Backtest complete")

# -----------------------------------------------
# STEP 3: Verify trade logic
# -----------------------------------------------
print("\n🔍 STEP 3: Verifying trade logic...")
trade_types = [t["type"] for t in trades]
errors = 0

for i in range(1, len(trade_types)):
    if trade_types[i] == trade_types[i-1]:
        print(f"❌ ERROR: Two consecutive {trade_types[i]} trades at index {i}")
        errors += 1
if errors == 0:
    print(f"✅ Trade alternation correct — BUY/SELL always alternate")

for t in trades:
    if t.get("shares", 0) < 0:
        print(f"❌ ERROR: Negative shares in trade: {t}")
        errors += 1
if errors == 0:
    print(f"✅ All share quantities are positive")

for t in trades:
    price = t["price"]
    if price < 50 or price > 5000:
        print(f"❌ ERROR: Unrealistic price ${price} in trade: {t}")
        errors += 1
if errors == 0:
    print(f"✅ All trade prices are realistic")

# -----------------------------------------------
# STEP 4: Verify metrics math
# -----------------------------------------------
print("\n🧮 STEP 4: Verifying metrics calculations...")
expected_return = ((metrics["final_value"] - 10000) / 10000) * 100
actual_return   = metrics["total_return"]
if abs(expected_return - actual_return) < 0.01:
    print(f"✅ Total return math correct: {actual_return:.2f}%")
else:
    print(f"❌ Return mismatch: expected {expected_return:.2f}% got {actual_return:.2f}%")
    errors += 1

sell_trades     = [t for t in trades if t["type"] == "SELL"]
winning_trades  = [t for t in sell_trades if t.get("pnl", 0) > 0]
expected_wr     = (len(winning_trades) / len(sell_trades) * 100) if sell_trades else 0
if abs(expected_wr - metrics["win_rate"]) < 0.01:
    print(f"✅ Win rate math correct: {metrics['win_rate']:.1f}%")
else:
    print(f"❌ Win rate mismatch: expected {expected_wr:.2f}% got {metrics['win_rate']:.2f}%")
    errors += 1

if 0 <= metrics["max_drawdown"] <= 100:
    print(f"✅ Max drawdown valid: {metrics['max_drawdown']:.2f}%")
else:
    print(f"❌ Invalid max drawdown: {metrics['max_drawdown']}")
    errors += 1

# -----------------------------------------------
# STEP 5: Buy & Hold verification
# -----------------------------------------------
print("\n📊 STEP 5: Verifying Buy & Hold calculation...")
first_price = float(df["close"].iloc[0])
last_price  = float(df["close"].iloc[-1])
expected_bh = ((last_price - first_price) / first_price) * 100
actual_bh   = metrics["buy_hold_return"]

if abs(expected_bh - actual_bh) < 0.01:
    print(f"✅ Buy & Hold correct: {actual_bh:.2f}%")
    print(f"   (bought at ${first_price:.2f}, now ${last_price:.2f})")
else:
    print(f"❌ Buy & Hold mismatch: expected {expected_bh:.2f}% got {actual_bh:.2f}%")
    errors += 1

# -----------------------------------------------
# STEP 6: Final results (default strategy)
# -----------------------------------------------
print("\n" + "=" * 60)
print("FINAL RESULTS — advanced_combined")
print("=" * 60)
print(f"Initial Capital:   ${metrics['initial_capital']:,.2f}")
print(f"Final Value:       ${metrics['final_value']:,.2f}")
print(f"Total Return:      {metrics['total_return']:+.2f}%")
print(f"Buy & Hold Return: {metrics['buy_hold_return']:+.2f}%")
print(f"Outperformance:    {metrics['total_return'] - metrics['buy_hold_return']:+.2f}%")
print(f"Total Trades:      {metrics['total_trades']}")
print(f"Win Rate:          {metrics['win_rate']:.1f}%")
print(f"Profit Factor:     {metrics['profit_factor']:.2f}")
print(f"Max Drawdown:      {metrics['max_drawdown']:.2f}%")
print(f"Sharpe Ratio:      {metrics.get('sharpe_ratio', 0):.2f}")
print("=" * 60)

if errors == 0:
    print("\n✅ ALL CHECKS PASSED — Backtesting is working correctly!")
else:
    print(f"\n❌ {errors} ERRORS FOUND — Check the issues above")

# -----------------------------------------------
# STEP 7: ALL 10 STRATEGIES COMPARISON
# -----------------------------------------------
print("\n📊 FULL STRATEGY COMPARISON (ALL 10 STRATEGIES)")
print("-" * 95)
print(f"{'Strategy':<22} {'Return':>9} {'Win Rate':>10} {'Trades':>8} {'Drawdown':>10} {'Sharpe':>8} {'PF':>6}")
print("-" * 95)

all_strategies = [
    "rsi", "macd", "bollinger", "ema_crossover",
    "rsi_macd_confluence", "volume_confirmed",
    "trend_following", "stochastic_rsi",
    "combined", "advanced_combined"
]

results_summary = []
for strat in all_strategies:
    try:
        _, m, _ = run_backtest(df, strategy=strat, initial_capital=10000)
        results_summary.append((strat, m))
        print(f"{strat:<22} {m['total_return']:>+8.2f}% {m['win_rate']:>9.1f}% {m['total_trades']:>8} {m['max_drawdown']:>9.2f}% {m.get('sharpe_ratio', 0):>7.2f} {m['profit_factor']:>6.2f}")
    except Exception as e:
        print(f"{strat:<22} ERROR: {e}")

print("-" * 95)
print(f"{'Buy & Hold':<22} {metrics['buy_hold_return']:>+8.2f}%  {'N/A':>9}  {'N/A':>8}  {'N/A':>9}  {'N/A':>7}  {'N/A':>6}")
print("=" * 95)

# -----------------------------------------------
# STEP 8: Best & Worst Strategy
# -----------------------------------------------
if results_summary:
    best  = max(results_summary, key=lambda x: x[1]["total_return"])
    worst = min(results_summary, key=lambda x: x[1]["total_return"])
    best_sharpe = max(results_summary, key=lambda x: x[1].get("sharpe_ratio", -999))

    print("\n🏆 BEST PERFORMERS")
    print(f"   Highest Return:  {best[0]} ({best[1]['total_return']:+.2f}%)")
    print(f"   Best Sharpe:     {best_sharpe[0]} ({best_sharpe[1].get('sharpe_ratio', 0):.2f})")
    print(f"\n⚠️  WEAKEST PERFORMER")
    print(f"   Lowest Return:   {worst[0]} ({worst[1]['total_return']:+.2f}%)")

print("\n✅ Verification complete! Copy the results and share them.")