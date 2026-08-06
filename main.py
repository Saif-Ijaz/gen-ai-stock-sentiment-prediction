# main.py — Training Pipeline focused on 4H candles
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import pandas as pd
import numpy as np
from loguru import logger
from datetime import datetime

from pipelines.stock_pipeline import fetch_stock_data
from pipelines.news_pipeline import fetch_news
from indicators.technicals import add_technical_indicators
from pipelines.alignment import align_news_with_stock
from sentiment.sentiment_pipeline import score_news_sentiment
from features.feature_engineering import build_features
from models.retrain import retrain_model

TICKERS    = ["AAPL", "TSLA", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "NFLX", "JPM", "V"]
INTERVAL   = "1h"    # fetch 1h then resample to 4h
MODEL_PATH = "models/model.pkl"


def resample_to_4h(df):
    """Resample 1H candles to 4H candles for better signal quality"""
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
    df = df.reset_index()
    df = df.rename(columns={"datetime": "datetime"})
    logger.info(f"Resampled to 4H: {len(df)} candles")
    return df


def create_target(stock_df):
    """
    Binary target:
    1 = next 4H candle closes HIGHER than current
    0 = next 4H candle closes LOWER or equal
    """
    if "close" not in stock_df.columns:
        return None
    target = (stock_df["close"].shift(-1) > stock_df["close"]).astype(int)
    return target


def run_pipeline():
    logger.info("=" * 60)
    logger.info(f"4H TRAINING PIPELINE STARTED — {datetime.utcnow().isoformat()}")
    logger.info("=" * 60)

    all_X = []
    all_y = []

    for ticker in TICKERS:
        logger.info(f"Processing {ticker}...")

        try:
            # Fetch 1H data (max 730 days)
            stock_df = fetch_stock_data(ticker, interval=INTERVAL)
            if stock_df.empty:
                logger.warning(f"No stock data for {ticker}, skipping")
                continue

            # Resample to 4H
            stock_df = resample_to_4h(stock_df)
            if len(stock_df) < 100:
                logger.warning(f"Not enough 4H candles for {ticker}, skipping")
                continue

            # Add technical indicators on 4H data
            stock_df = add_technical_indicators(stock_df)

            # Fetch news & sentiment
            news_df = fetch_news(ticker)
            if not news_df.empty:
                sentiment_df = score_news_sentiment(news_df)
            else:
                sentiment_df = pd.DataFrame()

            # Build features
            aligned_df = align_news_with_stock(stock_df, news_df)
            X, _       = build_features(stock_df, sentiment_df, aligned_df)

            if X.empty:
                logger.warning(f"Empty features for {ticker}, skipping")
                continue

            # Create target
            y = create_target(stock_df)
            if y is None:
                continue

            # FIX: Reset indexes before aligning
            X = X.reset_index(drop=True)
            y = y.reset_index(drop=True)

            # Align lengths
            min_len = min(len(X), len(y))
            X = X.iloc[:min_len]
            y = y.iloc[:min_len]

            # Drop last row (no future price)
            X = X.iloc[:-1]
            y = y.iloc[:-1]

            # Drop NaN targets
            mask = y.notna().values
            X = X.iloc[mask]
            y = y.iloc[mask]

            if len(X) < 50:
                logger.warning(f"Not enough samples for {ticker} ({len(X)}), skipping")
                continue

            all_X.append(X)
            all_y.append(y)
            logger.info(f"{ticker}: {len(X)} 4H samples added")

        except Exception as e:
            logger.error(f"Error processing {ticker}: {e}")
            import traceback
            traceback.print_exc()
            continue

    if not all_X:
        logger.error("No data collected — training aborted!")
        return

    X_combined = pd.concat(all_X, ignore_index=True)
    y_combined = pd.concat(all_y, ignore_index=True)

    logger.info(f"Total 4H training samples: {len(X_combined)}")
    logger.info(f"Class distribution: UP={int(y_combined.sum())} | DOWN={int((y_combined==0).sum())}")

    metrics = retrain_model(X_combined, y_combined, save_path=MODEL_PATH)

    logger.info("=" * 60)
    logger.info("4H TRAINING COMPLETE")
    logger.info(f"Accuracy:  {metrics['accuracy']:.4f}")
    logger.info(f"F1 Score:  {metrics['f1']:.4f}")
    logger.info(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
    logger.info("=" * 60)

if __name__ == "__main__":
    run_pipeline()