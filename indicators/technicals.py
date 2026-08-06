import ta
import pandas as pd


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add RSI, EMA(20), SMA(20), ATR(14) safely.
    Handles MultiIndex columns from yfinance.
    """

    df = df.copy()

    # --- Flatten MultiIndex columns if needed ---
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Ensure column names are lowercase strings
    df.columns = [str(col).lower() for col in df.columns]

    if "close" not in df.columns:
        return df

    close_series = df["close"].astype(float)

    # --- RSI ---
    try:
        df["rsi"] = ta.momentum.RSIIndicator(
            close=close_series,
            window=14
        ).rsi()
    except Exception:
        df["rsi"] = None

    # --- EMA 20 ---
    try:
        df["ema_20"] = ta.trend.EMAIndicator(
            close=close_series,
            window=20
        ).ema_indicator()
    except Exception:
        df["ema_20"] = None

    # --- SMA 20 ---
    try:
        df["sma_20"] = ta.trend.SMAIndicator(
            close=close_series,
            window=20
        ).sma_indicator()
    except Exception:
        df["sma_20"] = None

    # --- ATR 14 ---
    try:
        df["atr_14"] = ta.volatility.AverageTrueRange(
            high=df["high"],
            low=df["low"],
            close=close_series,
            window=14
        ).average_true_range()
    except Exception:
        df["atr_14"] = None

    return df
