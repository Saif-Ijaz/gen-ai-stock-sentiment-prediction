# features/feature_engineering.py
import pandas as pd
import numpy as np
from loguru import logger


def build_features(stock_df, sentiment_df, aligned_df):
    """
    Professional feature engineering with:
    - Extended technical features (MACD, BB, ADX, Stochastic, EMA crossover)
    - Lag features for temporal context
    - Fixed sentiment aggregation
    - Feature validation
    """

    df = stock_df.copy()

    # -----------------------------------------------
    # REQUIRED TECHNICAL INDICATORS
    # -----------------------------------------------
    required_cols = ["rsi", "ema_20", "sma_20"]
    for col in required_cols:
        if col not in df.columns:
            logger.warning(f"Missing required column: {col}")
            return pd.DataFrame(), None

    df = df.dropna(subset=required_cols)
    if df.empty:
        logger.warning("DataFrame empty after dropping NaN technical indicators")
        return pd.DataFrame(), None

    # -----------------------------------------------
    # PRICE MOMENTUM FEATURES
    # -----------------------------------------------
    if "close" in df.columns:
        df["price_change_1"]  = df["close"].pct_change(1)
        df["price_change_3"]  = df["close"].pct_change(3)
        df["price_change_5"]  = df["close"].pct_change(5)
        df["price_change_10"] = df["close"].pct_change(10)

    # -----------------------------------------------
    # MOVING AVERAGE CROSSOVER
    # -----------------------------------------------
    if "ema_20" in df.columns and "sma_20" in df.columns:
        df["ma_crossover"] = (df["ema_20"] - df["sma_20"]) / df["sma_20"]

    # EMA 9 / EMA 21 crossover
    if "ema_9" in df.columns and "ema_21" in df.columns:
        df["ema_cross"] = (df["ema_9"] - df["ema_21"]) / df["ema_21"]

    # -----------------------------------------------
    # RSI FEATURES
    # -----------------------------------------------
    if "rsi" in df.columns:
        df["rsi_change"]    = df["rsi"].diff(1)
        df["rsi_lag_1"]     = df["rsi"].shift(1)
        df["rsi_overbought"] = (df["rsi"] > 70).astype(int)
        df["rsi_oversold"]   = (df["rsi"] < 30).astype(int)

    # -----------------------------------------------
    # MACD FEATURES
    # -----------------------------------------------
    if "macd" in df.columns and "macd_signal" in df.columns:
        df["macd_diff"] = df["macd"] - df["macd_signal"]  # positive = bullish

    # -----------------------------------------------
    # BOLLINGER BAND FEATURES
    # -----------------------------------------------
    # bb_pct and bb_width already computed in technicals.py

    # -----------------------------------------------
    # ADX TREND STRENGTH
    # -----------------------------------------------
    if "adx" in df.columns:
        df["adx_strong"] = (df["adx"] > 25).astype(int)  # 1=strong trend

    # -----------------------------------------------
    # ATR VOLATILITY
    # -----------------------------------------------
    if "atr_14" in df.columns:
        df["atr_normalized"] = df["atr_14"] / df["close"] if "close" in df.columns else df["atr_14"]

    # -----------------------------------------------
    # VOLUME FEATURES
    # -----------------------------------------------
    if "volume" in df.columns:
        df["volume_change"]    = df["volume"].pct_change(1)
        df["volume_ma5"]       = df["volume"].rolling(5).mean()
        df["volume_ratio"]     = df["volume"] / df["volume_ma5"]
        df["volume_ratio_lag"] = df["volume_ratio"].shift(1)

    # -----------------------------------------------
    # ROLLING WINDOW FEATURES (memory / temporal context)
    # -----------------------------------------------

    # RSI rolling — direction and speed of RSI over last N candles
    if "rsi" in df.columns:
        df["rsi_ma5"]   = df["rsi"].rolling(5).mean()   # RSI trend (rising/falling)
        df["rsi_std5"]  = df["rsi"].rolling(5).std()    # RSI stability
        df["rsi_slope"] = df["rsi"].diff(3)             # RSI momentum over 3 candles

    # MACD rolling — is MACD signal strengthening or weakening?
    if "macd_diff" in df.columns:
        df["macd_ma3"]   = df["macd_diff"].rolling(3).mean()  # avg MACD gap last 3
    if "macd" in df.columns:
        df["macd_slope"] = df["macd"].diff(3)                 # MACD acceleration

    # Price volatility — how volatile has price been recently?
    if "close" in df.columns:
        pct = df["close"].pct_change()
        df["price_std5"]  = pct.rolling(5).std()    # short-term volatility
        df["price_std10"] = pct.rolling(10).std()   # medium-term volatility

    # Volume rolling — unusual volume activity
    if "volume" in df.columns:
        df["volume_std5"]  = df["volume"].rolling(5).std()
        vol_ma20           = df["volume"].rolling(20).mean()
        df["volume_spike"] = (df["volume"] > vol_ma20 * 2).astype(int)  # 2x average

    # -----------------------------------------------
    # SENTIMENT FEATURES (fixed — proper scalar assignment)
    # -----------------------------------------------
    if not sentiment_df.empty:
        pos = float(sentiment_df["positive"].mean())
        neu = float(sentiment_df["neutral"].mean())
        neg = float(sentiment_df["negative"].mean())
        std = float(sentiment_df["positive"].std()) if len(sentiment_df) > 1 else 0.0
    else:
        pos, neu, neg, std = 0.0, 1.0, 0.0, 0.0

    df["positive"]        = pos
    df["neutral"]         = neu
    df["negative"]        = neg
    df["sentiment_score"] = pos - neg   # net sentiment
    df["sentiment_std"]   = std         # uncertainty of sentiment

    # -----------------------------------------------
    # FULL FEATURE LIST
    # -----------------------------------------------
    base_features = ["rsi", "ema_20", "sma_20"]

    optional_features = [
        # Price momentum
        "price_change_1", "price_change_3", "price_change_5", "price_change_10",
        # Moving average crossovers
        "ma_crossover", "ema_cross",
        # RSI point-in-time
        "rsi_change", "rsi_lag_1", "rsi_overbought", "rsi_oversold",
        # RSI rolling window
        "rsi_ma5", "rsi_std5", "rsi_slope",
        # MACD point-in-time
        "macd", "macd_signal", "macd_hist", "macd_diff",
        # MACD rolling window
        "macd_ma3", "macd_slope",
        # Bollinger Bands
        "bb_pct", "bb_width",
        # ADX trend
        "adx", "plus_di", "minus_di", "adx_strong",
        # Stochastic
        "stoch_k", "stoch_d",
        # EMA fast/slow
        "ema_9", "ema_21",
        # ATR volatility
        "atr_normalized",
        # Volume point-in-time
        "volume_change", "volume_ratio", "volume_ratio_lag",
        # Volume rolling window
        "volume_std5", "volume_spike",
        # Price volatility rolling
        "price_std5", "price_std10",
        # Sentiment
        "positive", "neutral", "negative", "sentiment_score", "sentiment_std",
    ]

    # Only include features that exist in df
    available_optional = [f for f in optional_features if f in df.columns]
    features = base_features + available_optional

    X = df[features].copy()

    # Replace inf values and fill remaining NaN with 0
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)

    logger.info(f"Built feature matrix: {X.shape[0]} rows x {X.shape[1]} features")
    logger.info(f"Features: {list(X.columns)}")

    return X, None
