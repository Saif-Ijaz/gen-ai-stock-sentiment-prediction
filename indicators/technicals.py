import ta
import pandas as pd


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators: RSI, EMA, SMA, ATR, MACD,
    Bollinger Bands, ADX, Stochastic, EMA9/21.
    Handles MultiIndex columns from yfinance.
    """

    df = df.copy()

    # --- Flatten MultiIndex columns if needed ---
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Ensure column names are lowercase stringsff
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

    # --- MACD ---
    try:
        macd_indicator    = ta.trend.MACD(close=close_series)
        df["macd"]        = macd_indicator.macd()
        df["macd_signal"] = macd_indicator.macd_signal()
        df["macd_hist"]   = macd_indicator.macd_diff()
    except Exception:
        df["macd"] = df["macd_signal"] = df["macd_hist"] = None

    # --- Bollinger Bands ---
    try:
        bb             = ta.volatility.BollingerBands(
            close=close_series, window=20, window_dev=2
        )
        df["bb_upper"] = bb.bollinger_hband()
        df["bb_lower"] = bb.bollinger_lband()
        df["bb_mid"]   = bb.bollinger_mavg()
        df["bb_pct"]   = bb.bollinger_pband()   # %B — position within bands
        df["bb_width"] = bb.bollinger_wband()   # band width
    except Exception:
        df["bb_upper"] = df["bb_lower"] = df["bb_mid"] = None
        df["bb_pct"]   = df["bb_width"] = None

    # --- ADX (Trend Strength) ---
    try:
        adx_indicator  = ta.trend.ADXIndicator(
            high=df["high"], low=df["low"], close=close_series, window=14
        )
        df["adx"]      = adx_indicator.adx()
        df["plus_di"]  = adx_indicator.adx_pos()
        df["minus_di"] = adx_indicator.adx_neg()
    except Exception:
        df["adx"] = df["plus_di"] = df["minus_di"] = None

    # --- Stochastic Oscillator ---
    try:
        stoch         = ta.momentum.StochasticOscillator(
            high=df["high"], low=df["low"], close=close_series,
            window=14, smooth_window=3
        )
        df["stoch_k"] = stoch.stoch()
        df["stoch_d"] = stoch.stoch_signal()
    except Exception:
        df["stoch_k"] = df["stoch_d"] = None

    # --- EMA 9 and EMA 21 (Fast/Slow crossover) ---
    try:
        df["ema_9"]  = ta.trend.EMAIndicator(
            close=close_series, window=9
        ).ema_indicator()
        df["ema_21"] = ta.trend.EMAIndicator(
            close=close_series, window=21
        ).ema_indicator()
    except Exception:
        df["ema_9"] = df["ema_21"] = None

    return df
