# app/dashboard.py
from realtime import fetch_live_price
import sys
import os
import math
import streamlit as st
from streamlit_autorefresh import st_autorefresh
from datetime import datetime
import pytz
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.api import predict_api
from app.app_logic import derive_key_drivers, estimate_percentage_move, confidence_label
from rag.ingest import ingest_news
from sentiment.finbert_model import finbert_predict
from app.strategies import get_current_signal, calculate_macd, calculate_bollinger_bands
from app.backtesting import run_backtest
from app.model_performance import render_model_performance_page

st.set_page_config(page_title="AI Stock Predictor", layout="wide")
st_autorefresh(interval=2 * 60 * 1000, key="auto_refresh")

st.markdown("""
<style>
.section-label{font-size:12px;font-weight:600;color:var(--text-color);opacity:0.6;margin-bottom:10px;letter-spacing:0.06em;text-transform:uppercase}
.card{background:var(--secondary-background-color);border:1px solid rgba(128,128,128,0.2);border-radius:12px;padding:16px 18px;margin-bottom:10px}
.signal-card{border-radius:12px;padding:20px 24px;margin-bottom:10px;text-align:center}
.signal-buy{background:linear-gradient(135deg,#1a3a1a,#0d2b1a);border:1px solid #00c896}
.signal-sell{background:linear-gradient(135deg,#3a1a1a,#2b0d0d);border:1px solid #ff4b4b}
.signal-hold{background:linear-gradient(135deg,#1a1a2e,#2a2a3e);border:1px solid #4c9be8}
.signal-text{font-size:32px;font-weight:700;letter-spacing:2px}
.signal-buy .signal-text{color:#00c896}
.signal-sell .signal-text{color:#ff4b4b}
.signal-hold .signal-text{color:#4c9be8}
.target-grid{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:12px}
.target-metric{background:var(--background-color);border:1px solid rgba(128,128,128,0.15);border-radius:8px;padding:12px 14px}
.target-metric-label{font-size:11px;color:var(--text-color);opacity:0.55;margin-bottom:4px;font-weight:600;letter-spacing:0.04em;text-transform:uppercase}
.target-metric-value{font-size:22px;font-weight:600;color:var(--text-color)}
.range-labels{display:flex;justify-content:space-between;font-size:11px;color:var(--text-color);opacity:0.55;margin-top:6px}
.move-pill{display:inline-flex;align-items:center;gap:6px;font-size:13px;font-weight:600;padding:5px 12px;border-radius:20px;margin-bottom:14px}
.pill-bear{background:rgba(255,75,75,0.15);color:#ff4b4b}
.pill-bull{background:rgba(0,200,150,0.15);color:#00c896}
.pill-neut{background:var(--secondary-background-color);color:var(--text-color);opacity:0.7}
.driver-card{background:var(--secondary-background-color);border:1px solid rgba(128,128,128,0.2);border-radius:10px;padding:12px 14px;margin-bottom:8px;display:flex;align-items:center;gap:12px}
.driver-icon{width:34px;height:34px;border-radius:8px;display:flex;align-items:center;justify-content:center;flex-shrink:0}
.driver-text{font-size:13px;color:var(--text-color);line-height:1.4}
.driver-sub{font-size:11px;color:var(--text-color);opacity:0.55;margin-top:2px}
.news-card{background:var(--secondary-background-color);border:1px solid rgba(128,128,128,0.2);border-radius:12px;padding:14px 16px;margin-bottom:10px;display:flex;gap:14px;align-items:flex-start}
.news-index{font-size:12px;font-weight:600;color:var(--text-color);opacity:0.55;min-width:18px;padding-top:2px}
.news-body{flex:1;min-width:0}
.news-title{font-size:14px;font-weight:600;color:var(--text-color);line-height:1.4;margin-bottom:5px}
.news-snippet{font-size:13px;color:var(--text-color);opacity:0.65;line-height:1.5;margin-bottom:8px;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden}
.news-meta{display:flex;align-items:center;gap:8px;flex-wrap:wrap}
.badge{font-size:11px;font-weight:600;padding:2px 8px;border-radius:6px}
.badge-pos{background:rgba(0,200,150,0.15);color:#00c896}
.badge-neg{background:rgba(255,75,75,0.15);color:#f44336}
.badge-neu{background:var(--secondary-background-color);color:var(--text-color);opacity:0.7}
.news-date{font-size:11px;color:var(--text-color);opacity:0.55}
.bt-metric{background:var(--background-color);border:1px solid rgba(128,128,128,0.15);border-radius:8px;padding:12px;text-align:center}
.bt-metric-val{font-size:22px;font-weight:700;color:var(--text-color)}
.bt-metric-label{font-size:11px;color:var(--text-color);opacity:0.6;margin-top:4px}
</style>
""", unsafe_allow_html=True)

def get_market_status():
    tz = pytz.timezone("US/Eastern")
    now = datetime.now(tz)
    if now.weekday() >= 5:
        return "CLOSED 🔴"
    open_time = now.replace(hour=9, minute=30, second=0)
    close_time = now.replace(hour=16, minute=0, second=0)
    return "OPEN 🟢" if open_time <= now <= close_time else "CLOSED 🔴"

@st.cache_data(ttl=300)
def fetch_chart_data(ticker, freq):
    period_map   = {"1hr": "5d",  "4hr": "80d", "1day": "6mo"}
    interval_map = {"1hr": "1h",  "4hr": "1h",  "1day": "1d"}

    df = yf.download(
        tickers=ticker,
        period=period_map.get(freq, "5d"),
        interval=interval_map.get(freq, "1h"),
        progress=False, threads=False, auto_adjust=True
    )
    if df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).strip().title() for c in df.columns]
    df = df.reset_index()

    if "Volume" in df.columns:
        df = df[df["Volume"] > 0].copy()

    # Resample to 4h candles
    if freq == "4hr":
        date_col = df.columns[0]
        df = df.rename(columns={date_col: "Datetime"})
        df["Datetime"] = pd.to_datetime(df["Datetime"], utc=True)
        df = df.set_index("Datetime")
        df = df.resample("4h", closed="left", label="left").agg({
            "Open": "first", "High": "max",
            "Low": "min", "Close": "last", "Volume": "sum"
        }).dropna().reset_index()

    # Convert to US Eastern time
    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col], utc=True).dt.tz_convert("US/Eastern")
    df = df.reset_index(drop=True)
    return df

@st.cache_data(ttl=300)
def fetch_strategy_data(ticker):
    """Fetch 4hr data specifically for strategy analysis — max 2 years"""
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

    # Resample to 4h
    date_col = df.columns[0]
    df = df.rename(columns={date_col: "datetime"})
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.set_index("datetime")
    df = df.resample("4h", closed="left", label="left").agg({
        "open": "first", "high": "max",
        "low": "min", "close": "last", "volume": "sum"
    }).dropna().reset_index()

    # Add RSI
    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))

    return df

# SIDEBAR
st.sidebar.markdown("### 📊 Controls")
ticker = st.sidebar.selectbox("Stock", [
    "AAPL", "TSLA", "MSFT", "GOOGL",
    "AMZN", "NVDA", "META", "NFLX", "JPM", "V"
])
freq = st.sidebar.selectbox("Frequency", ["1hr", "4hr", "1day"])
st.sidebar.markdown("🕒 Live Time")
st.sidebar.code(datetime.now().strftime("%H:%M:%S"))

st.sidebar.markdown("---")
page = st.sidebar.radio("📑 View", ["Dashboard", "Model Performance"])

if page == "Model Performance":
    render_model_performance_page()
    st.stop()

@st.cache_data(ttl=900)
def run_ingestion(ticker):
    return ingest_news(ticker)


@st.cache_data(ttl=900, show_spinner=False)
def score_displayed_news(news_texts):
    """Run FinBERT on the exact news articles shown in the dashboard."""
    item_scores = []
    totals = {"positive": 0.0, "neutral": 0.0, "negative": 0.0}

    for text in news_texts:
        try:
            label, _, scores = finbert_predict(text)
        except Exception:
            label = "neutral"
            scores = {"positive": 0.0, "neutral": 1.0, "negative": 0.0}

        item_scores.append({"label": label, "scores": scores})
        for key in totals:
            totals[key] += float(scores.get(key, 0.0))

    if item_scores:
        average_scores = {key: value / len(item_scores) for key, value in totals.items()}
    else:
        average_scores = {"positive": 0.0, "neutral": 1.0, "negative": 0.0}

    return item_scores, average_scores

with st.spinner("📰 Fetching latest news..."):
    run_ingestion(ticker)

result = predict_api(ticker, freq)
if "error" in result:
    st.error(f"⚠️ {result['error']}")
    st.stop()

# Use FinBERT on the exact RAG news articles displayed below. This keeps the
# pie chart, percentages, and individual card badges consistent.
rag_news = result.get("rag_news", []) or []
news_texts = tuple(str(doc.get("text", "")) for doc in rag_news)
news_item_scores, displayed_news_sentiment = score_displayed_news(news_texts)
dominant_news_label = max(displayed_news_sentiment, key=displayed_news_sentiment.get).upper()

live_price = fetch_live_price(ticker)

# METRICS
c1, c2, c3, c4 = st.columns(4)
c1.metric("PRICE", f"${live_price:.2f}" if live_price else "N/A")
if result["prediction"] == "NEUTRAL":
    sentiment_label, sentiment_icon = "NEUTRAL", "🔵"
elif result["prediction"] == "UP":
    sentiment_label, sentiment_icon = "BULLISH", "🟢"
else:
    sentiment_label, sentiment_icon = "BEARISH", "🔴"
c2.metric("SENTIMENT", f"{sentiment_icon} {sentiment_label}")
conf_pct = round(result["confidence"] * 100)
conf_text = confidence_label(result["confidence"])
c3.metric("CONFIDENCE", f"{conf_pct}%", conf_text)
# Use the backend's predicted move everywhere. Key Drivers and Target already
# use this value, so the top Prediction card must not create another estimate
# from confidence.
pct_move = abs(float(result.get("predicted_pct", 0) or 0))
sign = "+" if result["prediction"] == "UP" else "-" if result["prediction"] == "DOWN" else ""
horizon_label = {"1hr": "1h Outlook", "4hr": "4h Outlook", "1day": "1D Outlook"}[freq]
c4.metric("PREDICTION", f"{sign}{pct_move:.2f}%", horizon_label)
st.caption(f"Market Status: {get_market_status()}")

# AI SUMMARY
st.markdown("### 🧠 AI Analysis & Prediction Summary")
st.success(f"Sentiment: **{sentiment_label}** | Confidence: **{conf_text}** ({conf_pct}%)")

# =============================================
# TRADING SIGNAL (NEW)
# =============================================
st.markdown("### 🎯 Trading Signal (4H Strategy)")

sig_col1, sig_col2 = st.columns([1, 3])
with sig_col1:
    live_strategy = st.selectbox("Live Signal Strategy", [
        "advanced_combined",
        "rsi_macd_confluence",
        "volume_confirmed",
        "trend_following",
        "stochastic_rsi",
        "ema_crossover",
        "combined",
        "macd",
        "rsi",
        "bollinger",
    ], key="live_strategy_select")

# Fetch chart data FIRST so Trading Signal and Price Chart
# can share the exact same Support/Resistance values (avoids
# the two sections showing different numbers for the same concept)
chart_df_preview = fetch_chart_data(ticker, freq)

strategy_df = fetch_strategy_data(ticker)
if not strategy_df.empty:
    signal_data = get_current_signal(strategy_df, strategy=live_strategy)

    # FIX: Override support/resistance with the chart's own min/max
    # so the two sections always agree exactly
    if not chart_df_preview.empty:
        signal_data["support"]    = float(chart_df_preview["Low"].min())
        signal_data["resistance"] = float(chart_df_preview["High"].max())

    sig = signal_data["signal"]

    # -----------------------------------------------
    # FINAL VERDICT — combines ML model + 4H strategy into
    # ONE actionable decision, weighted by how confident
    # each system actually is (not a flat 50/50 average).
    # -----------------------------------------------
    ml_direction   = result["prediction"]                    # UP, DOWN, NEUTRAL
    ml_confidence  = result.get("confidence", 0.5)            # 0..1, distance from 0.5 = conviction
    ml_conviction  = abs(ml_confidence - 0.5) * 2              # 0 (random) .. 1 (max conviction)
    ml_direction_score = ml_conviction if ml_direction == "UP" else (-ml_conviction if ml_direction == "DOWN" else 0)

    # Real, continuous per-strategy conviction (0..1) — NOT a fixed
    # 85%/55%/0% bucket. Each strategy computes its own magnitude-based
    # strength, so genuinely different setups show genuinely different %.
    buy_score  = signal_data.get("buy_score", 0)   # kept for display only
    sell_score = signal_data.get("sell_score", 0)  # kept for display only
    max_score  = signal_data.get("max_score", 14)
    strategy_conviction = signal_data.get("conviction", 0.0)
    strategy_direction_score = strategy_conviction if "BUY" in sig else (-strategy_conviction if "SELL" in sig else 0)

    # Weight: strategy is backtested with real historical performance data,
    # ML model hovers near random (~56-58% accuracy) — so strategy gets more weight.
    ML_WEIGHT = 0.35
    STRATEGY_WEIGHT = 0.65

    combined_score = (ML_WEIGHT * ml_direction_score) + (STRATEGY_WEIGHT * strategy_direction_score)
    # combined_score ranges roughly -1 (strong sell) to +1 (strong buy)
    combined_confidence_pct = int(abs(combined_score) * 100)
    combined_signed_pct = combined_score * 100  # signed: negative = sell side, positive = buy side

    if combined_score >= 0.35:
        verdict, verdict_color, verdict_icon = "BUY", "#00c896", "📈"
    elif combined_score >= 0.05:
        verdict, verdict_color, verdict_icon = "LEAN BUY", "#00c896", "↗️"
    elif combined_score <= -0.35:
        verdict, verdict_color, verdict_icon = "SELL", "#ff4b4b", "📉"
    elif combined_score <= -0.05:
        verdict, verdict_color, verdict_icon = "LEAN SELL", "#ff4b4b", "↘️"
    else:
        verdict, verdict_color, verdict_icon = "HOLD", "#888888", "⏸️"

    # Position on a -100% (full SELL) to +100% (full BUY) gauge, 0% = center
    gauge_pos_pct = 50 + (combined_signed_pct / 2)  # maps -100..100 to 0..100 (left..right)
    gauge_pos_pct = max(0, min(100, gauge_pos_pct))
    sign_str = "+" if combined_signed_pct >= 0 else ""

    st.markdown(f"""
    <div style="background:var(--secondary-background-color);border:1px solid {verdict_color};border-radius:12px;padding:16px 20px;margin-bottom:14px">
        <div>
            <div style="font-size:11px;color:var(--text-color);opacity:0.6;letter-spacing:0.06em;text-transform:uppercase;margin-bottom:4px">Final Verdict</div>
            <div style="font-size:26px;font-weight:800;color:{verdict_color}">{verdict_icon} {verdict} <span style="font-size:15px;font-weight:500;color:var(--text-color);opacity:0.7">({sign_str}{combined_signed_pct:.0f}%)</span></div>
        </div>
        <div style="margin-top:12px">
            <div style="position:relative;height:8px;border-radius:4px;background:linear-gradient(90deg,#ff4b4b 0%,#3a2a2a 45%,#2a2a2a 50%,#1a3a2a 55%,#00c896 100%)">
                <div style="position:absolute;left:{gauge_pos_pct}%;top:-4px;width:3px;height:16px;background:#ffffff;border-radius:2px;transform:translateX(-50%);box-shadow:0 0 4px rgba(0,0,0,0.6)"></div>
            </div>
            <div style="display:flex;justify-content:space-between;font-size:10px;color:var(--text-color);opacity:0.55;margin-top:4px">
                <span>← Strong Sell (-100%)</span>
                <span>Neutral (0%)</span>
                <span>Strong Buy (+100%) →</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # The big card now shows the SAME decision as the Final Verdict above —
    # "verdict" (BUY/LEAN BUY/SELL/LEAN SELL/HOLD), not the raw strategy-only
    # "sig". This guarantees the two boxes can never disagree; the raw
    # per-strategy reading is still shown below as supporting evidence text.
    if "BUY" in verdict:
        card_class, icon = "signal-buy", "📈"
    elif "SELL" in verdict:
        card_class, icon = "signal-sell", "📉"
    else:
        card_class, icon = "signal-hold", "⏸️"

    col_sig, col_reasons = st.columns([1, 2])

    with col_sig:
        st.markdown(f"""
        <div class="signal-card {card_class}">
            <div style="font-size:36px">{icon}</div>
            <div class="signal-text">{verdict}</div>
            <div style="font-size:12px;color:#888;margin-top:8px">{sign_str}{combined_signed_pct:.0f}% score · {live_strategy} + ML combined</div>
        </div>
        """, unsafe_allow_html=True)

        # The card above is the combined ML + strategy verdict. Show the
        # raw selected-strategy signal as well, so HOLD can be understood.
        st.caption(
            f"Raw {live_strategy} signal: {sig} | "
            f"strategy conviction: {strategy_conviction:.0%}"
        )

        # Support & Resistance
        st.markdown(f"""
        <div class="card" style="margin-top:10px">
            <div class="section-label">Support & Resistance</div>
            <div style="display:flex;justify-content:space-between;margin-top:8px">
                <div>
                    <div style="font-size:11px;color:#00c896">SUPPORT</div>
                    <div style="font-size:18px;font-weight:600;color:#00c896">${signal_data['support']:.2f}</div>
                </div>
                <div style="text-align:right">
                    <div style="font-size:11px;color:#ff4b4b">RESISTANCE</div>
                    <div style="font-size:18px;font-weight:600;color:#ff4b4b">${signal_data['resistance']:.2f}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_reasons:
        st.markdown('<div class="section-label">Signal Reasoning</div>', unsafe_allow_html=True)
        for reason in signal_data["reasons"]:
            icon_r = "🟢" if "bull" in reason.lower() or "buy" in reason.lower() or "oversold" in reason.lower() else \
                     "🔴" if "bear" in reason.lower() or "sell" in reason.lower() or "overbought" in reason.lower() else "🔵"
            st.markdown(f"""
            <div class="driver-card">
                <div style="font-size:18px">{icon_r}</div>
                <div class="driver-text">{reason}</div>
            </div>
            """, unsafe_allow_html=True)

        # MACD values
        st.markdown(f"""
        <div class="card" style="margin-top:8px">
            <div class="section-label">Indicator Values</div>
            <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;margin-top:8px">
                <div class="target-metric">
                    <div class="target-metric-label">MACD</div>
                    <div style="font-size:16px;font-weight:600;color:{'#00c896' if signal_data['macd']>0 else '#ff4b4b'}">{signal_data['macd']:.3f}</div>
                </div>
                <div class="target-metric">
                    <div class="target-metric-label">BB Upper</div>
                    <div style="font-size:16px;font-weight:600;color:#ff4b4b">${signal_data['bb_upper']:.2f}</div>
                </div>
                <div class="target-metric">
                    <div class="target-metric-label">BB Lower</div>
                    <div style="font-size:16px;font-weight:600;color:#00c896">${signal_data['bb_lower']:.2f}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# PRICE CHART
st.markdown("### 📈 Price Chart &nbsp; <small style='color:#888;font-size:13px'>🕐 Times in US Eastern (ET)</small>", unsafe_allow_html=True)
chart_df = chart_df_preview  # reuse the already-fetched data (avoids duplicate fetch + ensures consistency)
support_val = resistance_val = None

if not chart_df.empty:
    date_col = chart_df.columns[0]
    date_labels = pd.to_datetime(chart_df[date_col]).dt.strftime("%b %d %I:%M %p").tolist()

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.05, row_heights=[0.7, 0.3])

    fig.add_trace(go.Candlestick(
        x=date_labels,
        open=chart_df["Open"], high=chart_df["High"],
        low=chart_df["Low"], close=chart_df["Close"],
        name="Price",
        increasing_line_color="#00c896", decreasing_line_color="#ff4b4b"
    ), row=1, col=1)

    # Support & Resistance horizontal lines
    try:
        support_val    = float(chart_df["Low"].min())
        resistance_val = float(chart_df["High"].max())

        fig.add_hline(
            y=support_val,
            line=dict(color="#00c896", width=1, dash="dot"),
            opacity=0.6, row=1, col=1
        )
        fig.add_hline(
            y=resistance_val,
            line=dict(color="#ff4b4b", width=1, dash="dot"),
            opacity=0.6, row=1, col=1
        )
    except Exception:
        pass

    # DIAGONAL TREND LINES — 4H ONLY (primary strategy timeframe).
    # 1H and 1D trend lines removed per request; everything else unchanged.
    try:
        if freq != "4hr":
            raise StopIteration  # skip the rest of this block for 1hr/1day

        highs = chart_df["High"].values
        lows  = chart_df["Low"].values
        n     = len(chart_df)

        TIMEFRAME_TREND_CONFIG = {
            "1hr":  {"lookback": 35,  "swing_window": 2},
            "4hr":  {"lookback": 110, "swing_window": 4},
            "1day": {"lookback": None, "swing_window": 8},  # None = use full visible history
        }
        cfg = TIMEFRAME_TREND_CONFIG.get(freq, {"lookback": 60, "swing_window": 3})
        lookback = cfg["lookback"]
        start_idx = max(0, n - lookback) if lookback else 0

        def find_swings(values, n, start_idx, is_high, window):
            """Find anchor swing points within [start_idx, n), shrinking
            the window if too few are found so a line is still available."""
            for w in range(window, 0, -1):
                lo = max(start_idx, w)
                hi = n - w
                if hi <= lo:
                    continue
                pts = [i for i in range(lo, hi)
                       if (values[i] == max(values[i-w:i+w+1]) if is_high
                           else values[i] == min(values[i-w:i+w+1]))]
                if len(pts) >= 2:
                    return pts
            # Fallback: 2 most extreme points within the allowed range
            rng = list(range(start_idx, n))
            if len(rng) >= 2:
                idx_sorted = sorted(rng, key=lambda i: values[i], reverse=is_high)
                p1, p2 = sorted(idx_sorted[:2])
                if p1 != p2:
                    return [p1, p2]
            return []

        swing_highs = find_swings(highs, n, start_idx, is_high=True,  window=cfg["swing_window"])
        swing_lows  = find_swings(lows,  n, start_idx, is_high=False, window=cfg["swing_window"])

        # Resistance trend line — anchor point extended to the ACTUAL
        # highest point since the anchor (naturally includes the last
        # candle if it made a new high, instead of ignoring it)
        if len(swing_highs) >= 2:
            h1 = swing_highs[-2]
            h2 = h1 + 1 + int(highs[h1+1:].argmax()) if h1 + 1 < n else swing_highs[-1]
            if h2 != h1:
                slope = (highs[h2] - highs[h1]) / (h2 - h1)
                h3_y  = highs[h2] + slope * (n - 1 - h2)
                fig.add_trace(go.Scatter(
                    x=[date_labels[h1], date_labels[h2], date_labels[-1]],
                    y=[float(highs[h1]), float(highs[h2]), float(h3_y)],
                    mode="lines", name="Resistance Trend",
                    line=dict(color="#ff6b6b", width=2),
                    opacity=0.85, hoverinfo="skip", showlegend=False
                ), row=1, col=1)

        # Support trend line — anchor point extended to the ACTUAL
        # lowest point since the anchor (naturally includes the last
        # candle if it made a new low)
        if len(swing_lows) >= 2:
            l1 = swing_lows[-2]
            l2 = l1 + 1 + int(lows[l1+1:].argmin()) if l1 + 1 < n else swing_lows[-1]
            if l2 != l1:
                slope = (lows[l2] - lows[l1]) / (l2 - l1)
                l3_y  = lows[l2] + slope * (n - 1 - l2)
                fig.add_trace(go.Scatter(
                    x=[date_labels[l1], date_labels[l2], date_labels[-1]],
                    y=[float(lows[l1]), float(lows[l2]), float(l3_y)],
                    mode="lines", name="Support Trend",
                    line=dict(color="#51cf66", width=2),
                    opacity=0.85, hoverinfo="skip", showlegend=False
                ), row=1, col=1)

    except Exception:
        pass

    # Current price line
    if live_price:
        fig.add_hline(
            y=live_price,
            line=dict(color="#333333", width=1, dash="solid"),
            opacity=0.5, row=1, col=1
        )

    # Volume bars
    if "Volume" in chart_df.columns:
        colors = ["#00c896" if chart_df["Close"].iloc[i] >= chart_df["Open"].iloc[i]
                  else "#ff4b4b" for i in range(len(chart_df))]
        fig.add_trace(go.Bar(
            x=date_labels, y=chart_df["Volume"],
            name="Volume", marker_color=colors, opacity=0.6
        ), row=2, col=1)

    # AI Prediction marker — single clean badge above/below last candle
    pred         = result.get("prediction", "NEUTRAL")
    conf         = int(result.get("confidence", 0) * 100)
    pred_pct_val = result.get("predicted_pct", 0) or 0
    last_x       = date_labels[-1]
    price_span   = float(chart_df["High"].max()) - float(chart_df["Low"].min())
    offset       = price_span * 0.07

    if pred == "UP":
        marker_color, arrow, marker_y = "#00c896", "▲", float(chart_df["High"].iloc[-1]) + offset
    elif pred == "DOWN":
        marker_color, arrow, marker_y = "#ff4b4b", "▼", float(chart_df["Low"].iloc[-1]) - offset
    else:
        marker_color, arrow, marker_y = "#888888", "●", float(chart_df["High"].iloc[-1]) + offset

    fig.add_annotation(
        x=last_x, y=marker_y,
        text=f"<b>{arrow} {pred}</b>",
        showarrow=False,
        font=dict(color=marker_color, size=12),
        bgcolor="rgba(0,0,0,0)",
        bordercolor=marker_color,
        borderwidth=1,
        borderpad=4,
        row=1, col=1
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color=None,
        xaxis_rangeslider_visible=False, height=480,
        margin=dict(l=0, r=0, t=10, b=0),
        yaxis=dict(gridcolor="rgba(128,128,128,0.2)"),
        yaxis2=dict(gridcolor="rgba(128,128,128,0.2)"),
        xaxis=dict(type="category", gridcolor="rgba(128,128,128,0.2)", tickangle=-45, tickfont=dict(size=10), nticks=8),
        xaxis2=dict(type="category", gridcolor="rgba(128,128,128,0.2)", tickangle=-45, tickfont=dict(size=10), nticks=8),
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)

    # Clean info row below chart — replaces cluttered on-chart labels
    pred_icon = arrow  # reuse the same ▲ ▼ ● symbol shown on the chart badge
    pred_word = "BUY" if pred == "UP" else "SELL" if pred == "DOWN" else "HOLD"

    info_cols = st.columns(4)
    if support_val:
        info_cols[0].markdown(f"<div style='font-size:11px;color:#666'>SUPPORT</div><div style='font-size:15px;font-weight:600;color:#00c896'>${support_val:.2f}</div>", unsafe_allow_html=True)
    if resistance_val:
        info_cols[1].markdown(f"<div style='font-size:11px;color:#666'>RESISTANCE</div><div style='font-size:15px;font-weight:600;color:#ff4b4b'>${resistance_val:.2f}</div>", unsafe_allow_html=True)
    if live_price:
        info_cols[2].markdown(f"<div style='font-size:11px;color:var(--text-color);opacity:0.6'>CURRENT PRICE</div><div style='font-size:15px;font-weight:600;color:var(--text-color)'>${live_price:.2f}</div>", unsafe_allow_html=True)
    info_cols[3].markdown(f"<div style='font-size:11px;color:#666'>AI PREDICTION</div><div style='font-size:15px;font-weight:600;color:{marker_color}'>{pred_icon} {pred_word} ({conf}%)</div>", unsafe_allow_html=True)

else:
    st.warning("⚠️ Chart data unavailable.")

# RSI + SENTIMENT
col_rsi, col_sent = st.columns(2)
with col_rsi:
    st.markdown("### 📊 RSI Gauge")
    rsi_val = result.get("rsi")
    if rsi_val is not None:
        fig_rsi = go.Figure(go.Indicator(
            mode="gauge+number", value=rsi_val,
            number={"font": {"size": 36}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#4c9be8"},
                "steps": [
                    {"range": [0, 30], "color": "#ff4b4b"},
                    {"range": [30, 50], "color": "#f0a500"},
                    {"range": [50, 70], "color": "#4c9be8"},
                    {"range": [70, 100], "color": "#00c896"},
                ],
                "bgcolor": "rgba(0,0,0,0)",
            },
            title={"text": "RSI (14)", "font": {"size": 16}},
        ))
        fig_rsi.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color=None,
                               height=250, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_rsi, use_container_width=True)
        if rsi_val >= 70: st.caption("🔴 Overbought — potential pullback")
        elif rsi_val <= 30: st.caption("🟢 Oversold — potential bounce")
        elif rsi_val >= 50: st.caption("🔵 Bullish Zone")
        else: st.caption("🟡 Bearish Zone")
    else:
        st.info("RSI data unavailable.")

with col_sent:
    st.markdown("### 🧭 Sentiment Breakdown")
    sentiment = displayed_news_sentiment
    pos = sentiment.get("positive", 0)
    neu = sentiment.get("neutral", 0)
    neg = sentiment.get("negative", 0)
    fig_sent = go.Figure(go.Pie(
        labels=["Positive", "Neutral", "Negative"],
        values=[pos, neu, neg], hole=0.5,
        marker_colors=["#00c896", "#4c9be8", "#ff4b4b"],
        textinfo="label+percent", textfont=dict(size=13),
    ))
    fig_sent.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", font_color=None, height=250,
        margin=dict(l=20, r=20, t=40, b=20), showlegend=False,
        annotations=[dict(text=dominant_news_label, x=0.5, y=0.5,
                          font_size=14, showarrow=False)]
    )
    st.plotly_chart(fig_sent, use_container_width=True)
    st.progress(float(pos), text=f"Positive: {pos*100:.1f}%")
    st.progress(float(neu), text=f"Neutral:  {neu*100:.1f}%")
    st.progress(float(neg), text=f"Negative: {neg*100:.1f}%")

# TARGET
st.markdown("### 🎯 Target")
pred_pct = result.get("predicted_pct", 0)
if pred_pct is None or (isinstance(pred_pct, float) and math.isnan(pred_pct)):
    pred_pct = 0.0
if not live_price:
    st.warning("⚠️ Live price unavailable.")
else:
    low_pct  = pred_pct * 0.8
    high_pct = pred_pct * 1.2
    atr_value = result.get("atr_14")
    if atr_value and live_price and not math.isnan(float(atr_value)):
        volatility_factor = min(max(float(atr_value) / live_price, 0.5), 2)
        low_pct  *= volatility_factor
        high_pct *= volatility_factor
    # A bearish move produces two negative percentages. Sort the calculated
    # prices so "Low Target" is always the lower price and "High Target" is
    # always the higher price.
    target_a = live_price * (1 + low_pct / 100)
    target_b = live_price * (1 + high_pct / 100)
    low_target, high_target = sorted((target_a, target_b))
    move_low_pct, move_high_pct = sorted((low_pct, high_pct))
    if pred_pct > 0:
        arrow, pill_class, val_color = "↑", "pill-bull", "#00c896"
    elif pred_pct < 0:
        arrow, pill_class, val_color = "↓", "pill-bear", "#ff4b4b"
    else:
        arrow, pill_class, val_color = "→", "pill-neut", "#888"

    st.markdown(f"""
    <div class="card">
        <div class="move-pill {pill_class}">
            <span>{arrow}</span>
            <span>Expected Move: {move_low_pct:.2f}% to {move_high_pct:.2f}%</span>
        </div>
        <div class="target-grid">
            <div class="target-metric">
                <div class="target-metric-label">Low Target</div>
                <div class="target-metric-value" style="color:{val_color}">${low_target:.2f}</div>
            </div>
            <div class="target-metric">
                <div class="target-metric-label">High Target</div>
                <div class="target-metric-value" style="color:{val_color}">${high_target:.2f}</div>
            </div>
        </div>
        <div style="height:6px;border-radius:3px;background:var(--secondary-background-color);margin-top:4px">
            <div style="height:100%;border-radius:3px;background:{val_color};width:55%"></div>
        </div>
        <div class="range-labels">
            <span>${low_target:.2f}</span>
            <span>Current ${live_price:.2f}</span>
            <span>${high_target:.2f}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# KEY DRIVERS
st.markdown("### 🔑 Key Drivers")
# Key Drivers must use the same FinBERT average as the sentiment breakdown.
driver_result = {**result, "sentiment": displayed_news_sentiment}
drivers = derive_key_drivers(driver_result)
driver_icons = [
    ("📈", "#1a2a3a", "Technical indicator"),
    ("📊", "#2a1a3a", "Volatility indicator"),
    ("📰", "#1a3a1a", "News sentiment"),
    ("🛡️", "#2a2a1a", "Model confidence"),
    ("📉" if result["prediction"] == "DOWN" else "🚀",
     "#3a1a1a" if result["prediction"] == "DOWN" else "#1a3a1a", "Prediction"),
]
for i, driver in enumerate(drivers):
    emoji, bg, sublabel = driver_icons[i] if i < len(driver_icons) else ("💡", "#2a2a2a", "Signal")
    st.markdown(f"""
    <div class="driver-card">
        <div class="driver-icon" style="background:{bg};font-size:18px">{emoji}</div>
        <div>
            <div class="driver-text">{driver}</div>
            <div class="driver-sub">{sublabel}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# =============================================
# BACKTESTING (NEW)
# =============================================
st.markdown("### 🔁 Backtesting (4H Strategy)")

if not strategy_df.empty:
    bt_col1, bt_col2 = st.columns([1, 3])

    with bt_col1:
        bt_strategy = st.selectbox("Strategy", [
            "advanced_combined",
            "rsi_macd_confluence",
            "volume_confirmed",
            "trend_following",
            "stochastic_rsi",
            "ema_crossover",
            "combined",
            "macd",
            "rsi",
            "bollinger",
        ])
        bt_capital  = st.number_input("Initial Capital ($)", value=10000, step=1000)
        st.caption("💡 Capital scales dollar amounts only — % metrics (Return, Win Rate) stay the same.")
        run_bt = st.button("▶ Run Backtest", use_container_width=True)

    if "bt_strategy_prev" not in st.session_state:
        st.session_state["bt_strategy_prev"] = bt_strategy
    if "bt_capital_prev" not in st.session_state:
        st.session_state["bt_capital_prev"] = bt_capital
    if "bt_ticker_prev" not in st.session_state:
        st.session_state["bt_ticker_prev"] = ticker

    inputs_changed = (st.session_state["bt_strategy_prev"] != bt_strategy or
                       st.session_state["bt_capital_prev"] != bt_capital or
                       st.session_state["bt_ticker_prev"] != ticker)

    if run_bt or "bt_results" not in st.session_state or inputs_changed:
        st.session_state["bt_strategy_prev"] = bt_strategy
        st.session_state["bt_capital_prev"] = bt_capital
        st.session_state["bt_ticker_prev"] = ticker
        trades, metrics, equity = run_backtest(strategy_df, strategy=bt_strategy, initial_capital=bt_capital)
        st.session_state["bt_results"] = (trades, metrics, equity)

    if "bt_results" in st.session_state:
        trades, metrics, equity = st.session_state["bt_results"]

        # Metrics row
        m1, m2, m3, m4, m5, m6, m7 = st.columns(7)
        ret_color  = "#00c896" if metrics["total_return"] >= 0 else "#ff4b4b"
        bh_color   = "#00c896" if metrics["buy_hold_return"] >= 0 else "#ff4b4b"

        m1.markdown(f"""<div class="bt-metric">
            <div class="bt-metric-val" style="color:{ret_color}">{metrics['total_return']:+.1f}%</div>
            <div class="bt-metric-label">Total Return</div></div>""", unsafe_allow_html=True)
        m2.markdown(f"""<div class="bt-metric">
            <div class="bt-metric-val" style="color:{bh_color}">{metrics['buy_hold_return']:+.1f}%</div>
            <div class="bt-metric-label">Buy & Hold</div></div>""", unsafe_allow_html=True)
        m3.markdown(f"""<div class="bt-metric">
            <div class="bt-metric-val" style="color:#4c9be8">{metrics['win_rate']:.0f}%</div>
            <div class="bt-metric-label">Win Rate</div></div>""", unsafe_allow_html=True)
        m4.markdown(f"""<div class="bt-metric">
            <div class="bt-metric-val" style="color:#888">{metrics['total_trades']}</div>
            <div class="bt-metric-label">Total Trades</div></div>""", unsafe_allow_html=True)
        m5.markdown(f"""<div class="bt-metric">
            <div class="bt-metric-val" style="color:#ff4b4b">-{metrics['max_drawdown']:.1f}%</div>
            <div class="bt-metric-label">Max Drawdown</div></div>""", unsafe_allow_html=True)
        m6.markdown(f"""<div class="bt-metric">
            <div class="bt-metric-val" style="color:#f0a500">{metrics['profit_factor']:.2f}</div>
            <div class="bt-metric-label">Profit Factor</div></div>""", unsafe_allow_html=True)
        sharpe_color = "#00c896" if metrics.get('sharpe_ratio', 0) > 1 else "#f0a500" if metrics.get('sharpe_ratio', 0) > 0 else "#ff4b4b"
        m7.markdown(f"""<div class="bt-metric">
            <div class="bt-metric-val" style="color:{sharpe_color}">{metrics.get('sharpe_ratio', 0):.2f}</div>
            <div class="bt-metric-label">Sharpe Ratio</div></div>""", unsafe_allow_html=True)

        # Equity curve chart
        if equity:
            eq_df = pd.DataFrame(equity)
            fig_eq = go.Figure()
            fig_eq.add_trace(go.Scatter(
                x=eq_df["index"], y=eq_df["value"],
                name="Portfolio", mode="lines",
                line=dict(color="#4c9be8", width=2),
                fill="tozeroy", fillcolor="rgba(76,155,232,0.1)"
            ))
            fig_eq.add_hline(
                y=bt_capital,
                line=dict(color="#888", width=1, dash="dash"),
                annotation_text=f"Initial ${bt_capital:,}"
            )
            fig_eq.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color=None, height=250,
                margin=dict(l=0, r=0, t=20, b=0),
                yaxis=dict(gridcolor="rgba(128,128,128,0.2)", tickprefix="$"),
                xaxis=dict(gridcolor="rgba(128,128,128,0.2)"),
                showlegend=False,
                title=dict(text="Portfolio Equity Curve", font=dict(size=12))
            )
            st.plotly_chart(fig_eq, use_container_width=True)

        # Trades table
        if trades:
            st.markdown('<div class="section-label" style="margin-top:10px">Trade History</div>', unsafe_allow_html=True)
            sell_trades = [t for t in trades if t["type"] == "SELL"][::-1]
            if sell_trades:
                trades_df = pd.DataFrame(sell_trades)

                # Map candle index back to real date for clarity
                if "index" in trades_df.columns and "datetime" in strategy_df.columns:
                    trades_df["date"] = trades_df["index"].apply(
                        lambda i: pd.to_datetime(strategy_df["datetime"].iloc[i]).strftime("%b %d, %Y")
                        if 0 <= i < len(strategy_df) else ""
                    )
                    trades_df = trades_df[["date", "type", "price", "shares", "value", "pnl", "pnl_pct", "signal"]]
                    trades_df.columns = ["Date", "Type", "Price", "Shares", "Value", "P&L", "P&L %", "Signal"]
                else:
                    trades_df = trades_df[["type", "price", "shares", "value", "pnl", "pnl_pct", "signal"]]
                    trades_df.columns = ["Type", "Price", "Shares", "Value", "P&L", "P&L %", "Signal"]
                st.dataframe(
                    trades_df.style.map(
                        lambda v: "color: #00c896" if isinstance(v, (int, float)) and v > 0
                        else ("color: #ff4b4b" if isinstance(v, (int, float)) and v < 0 else ""),
                        subset=["P&L", "P&L %"]
                    ),
                    use_container_width=True, height=200
                )
else:
    st.info("⚠️ Strategy data unavailable for backtesting.")

st.markdown("---")
# NEWS
st.markdown("### 📰 Relevant News Driving Prediction")
if rag_news:
    pos_w = int(pos * 100)
    neu_w = int(neu * 100)
    neg_w = int(neg * 100)
    st.markdown(f"""
    <div style="display:flex;gap:4px;height:5px;border-radius:3px;overflow:hidden;margin-bottom:14px">
        <div style="background:#00c896;flex:{pos_w}"></div>
        <div style="background:#4c9be8;flex:{neu_w}"></div>
        <div style="background:#ff4b4b;flex:{neg_w}"></div>
    </div>
    """, unsafe_allow_html=True)
    for i, (doc, news_sentiment) in enumerate(zip(rag_news, news_item_scores)):
        text = doc.get("text", "")
        published = doc.get("publishedAt", "")
        parts = text.split(". ", 1)
        title = parts[0].strip()
        snippet = parts[1].strip() if len(parts) > 1 else ""
        if len(title) > 100:
            snippet = title[100:] + (". " + snippet if snippet else "")
            title = title[:100] + "..."
        try:
            dt = datetime.fromisoformat(str(published).replace("Z", "+00:00"))
            date_str = dt.strftime("%b %d, %Y")
        except Exception:
            date_str = ""
        label = news_sentiment["label"].lower()
        badge_class = {
            "positive": "badge-pos",
            "negative": "badge-neg",
            "neutral": "badge-neu",
        }.get(label, "badge-neu")
        badge = f"<span class='badge {badge_class}'>{label.title()}</span>"
        st.markdown(f"""
        <div class="news-card">
            <div class="news-index">{i+1}</div>
            <div class="news-body">
                <div class="news-title">{title}</div>
                {"<div class='news-snippet'>" + snippet + "</div>" if snippet else ""}
                <div class="news-meta">{badge}{"<span class='news-date'>· " + date_str + "</span>" if date_str else ""}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
else:
    st.info("ℹ️ No major sentiment-shifting news detected.")
