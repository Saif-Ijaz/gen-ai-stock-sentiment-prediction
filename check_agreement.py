# check_agreement.py
# Tests whether the ML model and Trading Signal agree/disagree across all 10 tickers

import sys, os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import yfinance as yf
import pandas as pd
from app.api import predict_api
from app.strategies import get_current_signal

TICKERS = ["AAPL", "TSLA", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "NFLX", "JPM", "V"]


def fetch_strategy_data(ticker):
    """Same logic as dashboard.py's fetch_strategy_data, standalone here"""
    df = yf.download(
        tickers=ticker, period="730d", interval="1h",
        progress=False, threads=False, auto_adjust=True
    )
    if df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).strip().lower() for c in df.columns]
    df = df.reset_index()
    if "volume" in df.columns:
        df = df[df["volume"] > 0]

    date_col = df.columns[0]
    df = df.rename(columns={date_col: "datetime"})
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.set_index("datetime")
    df = df.resample("4h", closed="left", label="left").agg({
        "open": "first", "high": "max",
        "low": "min", "close": "last", "volume": "sum"
    }).dropna().reset_index()

    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))

    return df


print(f"{'Ticker':<8} {'ML Pred':<10} {'ML Conf':<10} {'Strategy Signal':<16} {'Agreement':<12}")
print("-" * 65)

agree_count = 0
disagree_count = 0
neutral_count = 0

for ticker in TICKERS:
    try:
        result = predict_api(ticker, "4hr")
        if "error" in result:
            print(f"{ticker:<8} ERROR: {result['error']}")
            continue

        ml_pred = result["prediction"]
        ml_conf = f"{result['confidence']*100:.1f}%"

        strategy_df = fetch_strategy_data(ticker)
        signal_data = get_current_signal(strategy_df)
        sig = signal_data["signal"]

        strategy_direction = "UP" if "BUY" in sig else "DOWN" if "SELL" in sig else "NEUTRAL"

        if ml_pred == strategy_direction:
            agreement = "✅ AGREE"
            agree_count += 1
        elif ml_pred == "NEUTRAL" or strategy_direction == "NEUTRAL":
            agreement = "➖ NEUTRAL"
            neutral_count += 1
        else:
            agreement = "⚠️ DISAGREE"
            disagree_count += 1

        print(f"{ticker:<8} {ml_pred:<10} {ml_conf:<10} {sig:<16} {agreement:<12}")

    except Exception as e:
        print(f"{ticker:<8} ERROR: {e}")
        import traceback
        traceback.print_exc()

print("-" * 65)
total = agree_count + disagree_count + neutral_count
print(f"\nSummary across {total} tickers:")
if total:
    print(f"  Agree:    {agree_count} ({agree_count/total*100:.0f}%)")
    print(f"  Disagree: {disagree_count} ({disagree_count/total*100:.0f}%)")
    print(f"  Neutral:  {neutral_count} ({neutral_count/total*100:.0f}%)")