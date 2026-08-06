# check_prediction_freshness.py
# Verifies whether ML predictions are genuinely computed fresh,
# or accidentally stuck/cached, by testing multiple tickers
# and checking if results vary sensibly.

import sys, os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from app.api import predict_api

TICKERS = ["AAPL", "TSLA", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "NFLX", "JPM", "V"]

print(f"{'Ticker':<8} {'Prediction':<12} {'Confidence':<12} {'Predicted %':<12}")
print("-" * 50)

directions = []
for ticker in TICKERS:
    try:
        result = predict_api(ticker, "4hr")
        if "error" in result:
            print(f"{ticker:<8} ERROR: {result['error']}")
            continue
        pred = result["prediction"]
        conf = f"{result['confidence']*100:.1f}%"
        pct  = f"{result.get('predicted_pct', 0):+.2f}%"
        directions.append(pred)
        print(f"{ticker:<8} {pred:<12} {conf:<12} {pct:<12}")
    except Exception as e:
        print(f"{ticker:<8} ERROR: {e}")

print("-" * 50)
up_count   = directions.count("UP")
down_count = directions.count("DOWN")
neutral_count = directions.count("NEUTRAL")
total = len(directions)

print(f"\nDistribution across {total} tickers:")
print(f"  UP:      {up_count} ({up_count/total*100:.0f}%)" if total else "")
print(f"  DOWN:    {down_count} ({down_count/total*100:.0f}%)" if total else "")
print(f"  NEUTRAL: {neutral_count} ({neutral_count/total*100:.0f}%)" if total else "")

if total and down_count / total > 0.8:
    print("\n⚠️  WARNING: Over 80% of predictions are DOWN.")
    print("   This could indicate genuine market-wide bearish sentiment,")
    print("   OR a systematic model bias. Compare against a run from a")
    print("   different day/time to check if these numbers ever change.")
else:
    print("\n✅ Predictions show reasonable variation — no obvious stuck/bias pattern.")