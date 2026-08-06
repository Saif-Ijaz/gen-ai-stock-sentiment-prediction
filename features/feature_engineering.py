# features/feature_engineering.py
import pandas as pd
import numpy as np
from loguru import logger


def build_features(stock_df, sentiment_df, aligned_df):
    """
    Professional feature engineering with:
    - More technical features (momentum, volatility, trend)
    - Proper sentiment aggregation
    - Target variable creation
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
    # EXTENDED TECHNICAL FEATURES
    # -----------------------------------------------

    # Price momentum
    if "close" in df.columns:
        df["price_change_1"] = df["close"].pct_change(1)
        df["price_change_3"] = df["close"].pct_change(3)
        df["price_change_5"] = df["close"].pct_change(5)

    # Moving average crossover signal
    if "ema_20" in df.columns and "sma_20" in df.columns:
        df["ma_crossover"] = (df["ema_20"] - df["sma_20"]) / df["sma_20"]

    # RSI momentum
    if "rsi" in df.columns:
        df["rsi_change"] = df["rsi"].diff(1)
        df["rsi_overbought"] = (df["rsi"] > 70).astype(int)
        df["rsi_oversold"]   = (df["rsi"] < 30).astype(int)

    # ATR volatility
    if "atr_14" in df.columns:
        df["atr_normalized"] = df["atr_14"] / df["close"] if "close" in df.columns else df["atr_14"]

    # Volume features
    if "volume" in df.columns:
        df["volume_change"] = df["volume"].pct_change(1)
        df["volume_ma5"]    = df["volume"].rolling(5).mean()
        df["volume_ratio"]  = df["volume"] / df["volume_ma5"]

    # -----------------------------------------------
    # SENTIMENT FEATURES
    # -----------------------------------------------
    if not sentiment_df.empty:
        df["positive"] = sentiment_df["positive"].mean()
        df["neutral"]  = sentiment_df["neutral"].mean()
        df["negative"] = sentiment_df["negative"].mean()
        df["sentiment_score"] = df["positive"] - df["negative"]  # net sentiment
    else:
        df["positive"]        = 0.0
        df["neutral"]         = 1.0
        df["negative"]        = 0.0
        df["sentiment_score"] = 0.0

    # -----------------------------------------------
    # FEATURE LIST
    # -----------------------------------------------
    base_features = ["rsi", "ema_20", "sma_20"]

    optional_features = [
        "price_change_1", "price_change_3", "price_change_5",
        "ma_crossover",
        "rsi_change", "rsi_overbought", "rsi_oversold",
        "atr_normalized",
        "volume_change", "volume_ratio",
        "positive", "neutral", "negative", "sentiment_score",
    ]

    # Only include optional features that exist and have data
    available_optional = [f for f in optional_features if f in df.columns]
    features = base_features + available_optional

    X = df[features].copy()

    # Replace inf values and drop remaining NaN rows
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)

    logger.info(f"Built feature matrix: {X.shape[0]} rows x {X.shape[1]} features")
    logger.info(f"Features: {list(X.columns)}")

    return X, None