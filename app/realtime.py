# app/realtime.py
import sys
import os
import joblib
import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pipelines.stock_pipeline import fetch_stock_data
from pipelines.news_pipeline import fetch_news
from indicators.technicals import add_technical_indicators
from sentiment.sentiment_pipeline import score_news_sentiment
from pipelines.alignment import align_news_with_stock
from features.feature_engineering import build_features
from rag.retriever import hybrid_retrieve
from rag.query_expansion import expand_financial_query

MODEL_PATH = "models/model.pkl"

# ---------------------------------------------------
# SUPPORTED FREQUENCIES
# ---------------------------------------------------
INTERVAL_MAP = {
    "1hr":  "1h",
    "4hr":  "4h",
    "1day": "1d",
}

def resample_to_4h(df):
    """Resample 1H to 4H for model prediction"""
    date_col = df.columns[0]
    df = df.rename(columns={date_col: "datetime"})
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.set_index("datetime")
    df = df.resample("4h", closed="left", label="left").agg({
        "open":   "first",
        "high":   "max",
        "low":    "min",
        "close":  "last",
        "volume": "sum"
    }).dropna()
    df = df[df["volume"] > 0]
    return df.reset_index()

# ---------------------------------------------------
# LIVE PRICE (NO CACHE – FORCED REFRESH)
# ---------------------------------------------------
def fetch_live_price(ticker: str):
    """
    Fetch fresh 1-minute price every run.
    Handles MultiIndex columns from yfinance v0.2+
    """
    try:
        # Method 1: Use Ticker object (most reliable)
        stock = yf.Ticker(ticker)
        info = stock.fast_info

        price = getattr(info, "last_price", None)
        if price and float(price) > 0:
            return float(price)

    except Exception:
        pass

    try:
        # Method 2: Download with 1m interval fallback
        df = yf.download(
            tickers=ticker,
            period="1d",
            interval="1m",
            progress=False,
            threads=False,
            auto_adjust=True
        )

        if df.empty:
            return None

        # FIX: Handle MultiIndex columns (yfinance v0.2+)
        # MultiIndex looks like: ("Close", "AAPL")
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Normalize column names to title case
        df.columns = [str(c).strip().title() for c in df.columns]

        if "Close" in df.columns:
            return float(df["Close"].iloc[-1])

        return None

    except Exception:
        return None


# ---------------------------------------------------
# CACHED RAW DATA (CANDLE DATA ONLY)
# ---------------------------------------------------
@st.cache_data(ttl=300)
def cached_stock_data(ticker, interval):
    return fetch_stock_data(ticker, interval=interval)


@st.cache_data(ttl=900)
def cached_news_data(ticker, date_key):
    return fetch_news(ticker)


# ---------------------------------------------------
# CORE PREDICTION LOGIC
# ---------------------------------------------------
def _compute_prediction(ticker, freq):

    interval = INTERVAL_MAP.get(freq, "1h")

    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]

    # ---- Stock candles — always use 4H for model prediction ----
    stock_df = cached_stock_data(ticker, "1h")  # fetch 1H
    stock_df = resample_to_4h(stock_df)         # resample to 4H
    stock_df = add_technical_indicators(stock_df)

    # ---- News + Sentiment ----
    today_key = datetime.utcnow().strftime("%Y-%m-%d")
    news_df = cached_news_data(ticker, today_key)
    sentiment_df = score_news_sentiment(news_df)

    # ---- Features ----
    aligned_df = align_news_with_stock(stock_df, news_df)
    X, _ = build_features(stock_df, sentiment_df, aligned_df)

    # SAFETY: if no features
    if X.empty:
        return {
            "prediction": "NEUTRAL",
            "confidence": 0.5,
            "latest_price": fetch_live_price(ticker)
            or float(stock_df["close"].iloc[-1]),
            "rsi": None,
            "sentiment": {
                "positive": 0.0,
                "neutral": 1.0,
                "negative": 0.0,
            },
            "rag_news": [],
        }

    X_latest = X.tail(1)

    # --- Align features with model ---
    expected_features = model.feature_names_in_

    for col in expected_features:
        if col not in X_latest.columns:
            X_latest[col] = 0.0

    X_latest = X_latest[expected_features]

    if X_latest.shape[1] != len(expected_features):
        raise ValueError(
            f"Feature mismatch: expected {len(expected_features)}, got {X_latest.shape[1]}"
        )

    # ---- Model Prediction ----
    prob_up = float(model.predict_proba(X_latest)[0][1])

    # FIX: Guard against NaN probability
    import math
    if math.isnan(prob_up):
        prob_up = 0.5

    # Convert probability into signed percentage move
    direction_strength = abs(prob_up - 0.5) * 2  # scale 0–1
    predicted_pct = direction_strength * 2        # max 2% move (tunable)

    if prob_up > 0.5:
        signed_pct = predicted_pct
    else:
        signed_pct = -predicted_pct

    # FIX: Guard signed_pct against NaN
    if math.isnan(signed_pct):
        signed_pct = 0.0

    if abs(prob_up - 0.5) < 0.01:
        prediction = "NEUTRAL"
    else:
        prediction = "UP" if prob_up > 0.5 else "DOWN"

    # ---- Sentiment summary ----
    if sentiment_df.empty:
        sentiment_summary = {
            "positive": 0.0,
            "neutral": 1.0,
            "negative": 0.0,
        }
    else:
        sentiment_summary = {
            "positive": float(sentiment_df["positive"].mean()),
            "neutral": float(sentiment_df["neutral"].mean()),
            "negative": float(sentiment_df["negative"].mean()),
        }

    # ---- RAG ----
    rag_docs = hybrid_retrieve(
        expand_financial_query(ticker, "stock outlook"),
        top_k=5
    )

    return {
        "prediction": prediction,
        "confidence": prob_up,
        "predicted_pct": signed_pct,
        "rsi": float(stock_df["rsi"].iloc[-1]) if "rsi" in stock_df else None,
        "atr_14": float(stock_df["atr_14"].iloc[-1]) if "atr_14" in stock_df else None,
        "sentiment": sentiment_summary,
        "rag_news": rag_docs,
    }


# ---------------------------------------------------
# CACHE PER CANDLE (NOT PER CLOCK)
# ---------------------------------------------------
@st.cache_data
def cached_prediction(ticker, freq, last_ts):
    return _compute_prediction(ticker, freq)


def get_realtime_prediction(ticker, freq):
    interval = INTERVAL_MAP.get(freq, "1h")
    stock_df = cached_stock_data(ticker, interval)

    if stock_df is None or stock_df.empty:
        return {
            "error": f"No price data returned for {ticker}. This is usually a "
                     f"temporary yfinance/Yahoo Finance issue (rate limit or "
                     f"API hiccup), not an actual delisting. Try refreshing "
                     f"in a moment, or run 'pip install yfinance --upgrade'."
        }

    last_ts = str(stock_df.index[-1])

    return cached_prediction(ticker, freq, last_ts)