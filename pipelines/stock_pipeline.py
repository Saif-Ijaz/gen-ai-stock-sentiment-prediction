# pipelines/stock_pipeline.py
import yfinance as yf
import pandas as pd
from loguru import logger


def fetch_stock_data(ticker: str, interval: str = "1h") -> pd.DataFrame:
    """
    Fetch stock data. Uses 1H data which gets resampled to 4H in pipeline.
    730 days = maximum available for 1H interval in yfinance.
    """

    if interval == "1h":
        period = "730d"   # 2 years max for 1H
    elif interval == "4h":
        period = "730d"   # fetch 1H then resample
    elif interval == "1d":
        period = "10y"    # 10 years for daily
    else:
        period = "730d"

    df = yf.download(
        tickers=ticker,
        period=period,
        interval="1h" if interval == "4h" else interval,
        progress=False,
        threads=False,
        auto_adjust=True
    )

    if df.empty:
        logger.warning(f"No stock data returned for {ticker}")
        return pd.DataFrame()

    # Handle MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.rename(columns=str.lower)
    df = df.reset_index()

    # Normalize date column name
    for col in ["datetime", "date", "Datetime", "Date"]:
        if col in df.columns:
            df = df.rename(columns={col: "datetime"})
            break

    # Remove zero volume rows
    if "volume" in df.columns:
        df = df[df["volume"] > 0].copy()

    df = df.reset_index(drop=True)
    logger.info(f"Fetched {len(df)} rows for {ticker} ({interval}, {period})")
    return df