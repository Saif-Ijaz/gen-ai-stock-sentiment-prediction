# app/strategies.py
import pandas as pd
import numpy as np
from loguru import logger


# -----------------------------------------------
# INDICATOR CALCULATIONS
# -----------------------------------------------

def calculate_macd(df, fast=12, slow=26, signal=9):
    df = df.copy()
    df["ema_fast"]    = df["close"].ewm(span=fast, adjust=False).mean()
    df["ema_slow"]    = df["close"].ewm(span=slow, adjust=False).mean()
    df["macd"]        = df["ema_fast"] - df["ema_slow"]
    df["macd_signal"] = df["macd"].ewm(span=signal, adjust=False).mean()
    df["macd_hist"]   = df["macd"] - df["macd_signal"]
    return df


def calculate_bollinger_bands(df, period=20, std=2):
    df = df.copy()
    df["bb_mid"]   = df["close"].rolling(period).mean()
    df["bb_std"]   = df["close"].rolling(period).std()
    df["bb_upper"] = df["bb_mid"] + std * df["bb_std"]
    df["bb_lower"] = df["bb_mid"] - std * df["bb_std"]
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]
    df["bb_pct"]   = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])
    return df


def calculate_ema_crossover(df, fast=9, slow=21):
    df = df.copy()
    df["ema_fast_cross"] = df["close"].ewm(span=fast, adjust=False).mean()
    df["ema_slow_cross"] = df["close"].ewm(span=slow, adjust=False).mean()
    return df


def calculate_atr(df, period=14):
    df = df.copy()
    df["tr"] = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"]  - df["close"].shift()).abs()
    ], axis=1).max(axis=1)
    df["atr"] = df["tr"].rolling(period).mean()
    return df


def calculate_volume_ma(df, period=20):
    df = df.copy()
    df["volume_ma"] = df["volume"].rolling(period).mean()
    df["volume_ratio"] = df["volume"] / df["volume_ma"]
    return df


def calculate_adx(df, period=14):
    """Average Directional Index — measures trend strength"""
    df = df.copy()
    df["up_move"]   = df["high"].diff()
    df["down_move"] = -df["low"].diff()
    df["plus_dm"]   = np.where((df["up_move"] > df["down_move"]) & (df["up_move"] > 0), df["up_move"], 0)
    df["minus_dm"]  = np.where((df["down_move"] > df["up_move"]) & (df["down_move"] > 0), df["down_move"], 0)

    df["tr"] = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"]  - df["close"].shift()).abs()
    ], axis=1).max(axis=1)

    df["atr14"]      = df["tr"].rolling(period).mean()
    df["plus_di"]    = 100 * (df["plus_dm"].rolling(period).mean()  / df["atr14"])
    df["minus_di"]   = 100 * (df["minus_dm"].rolling(period).mean() / df["atr14"])
    df["dx"]         = 100 * (df["plus_di"] - df["minus_di"]).abs() / (df["plus_di"] + df["minus_di"])
    df["adx"]        = df["dx"].rolling(period).mean()
    return df


def calculate_stochastic(df, k_period=14, d_period=3):
    """Stochastic Oscillator"""
    df = df.copy()
    df["lowest_low"]   = df["low"].rolling(k_period).min()
    df["highest_high"] = df["high"].rolling(k_period).max()
    df["stoch_k"] = 100 * (df["close"] - df["lowest_low"]) / (df["highest_high"] - df["lowest_low"])
    df["stoch_d"] = df["stoch_k"].rolling(d_period).mean()
    return df


def calculate_support_resistance(df, window=10, lookback=48):
    """
    Calculate support/resistance using only the most RECENT candles.
    lookback=48 matches ~30 days of 4H candles (1.6 candles/day during
    market hours), aligning with the Price Chart's default 30-day view
    so both display the same Support/Resistance levels.
    """
    recent = df.tail(lookback)
    resistance = float(recent["high"].max())
    support    = float(recent["low"].min())
    return support, resistance


# -----------------------------------------------
# STRATEGY 1: RSI (Original)
# -----------------------------------------------
def rsi_strategy(df):
    signals = pd.Series("HOLD", index=df.index)
    if "rsi" not in df.columns:
        return signals
    rsi = df["rsi"]
    for i in range(1, len(df)):
        if rsi.iloc[i-1] < 30 and rsi.iloc[i] >= 30:
            signals.iloc[i] = "BUY"
        elif rsi.iloc[i-1] < 70 and rsi.iloc[i] >= 70:
            signals.iloc[i] = "SELL"
    return signals


# -----------------------------------------------
# STRATEGY 2: MACD (Original)
# -----------------------------------------------
def macd_strategy(df):
    df      = calculate_macd(df)
    signals = pd.Series("HOLD", index=df.index)
    for i in range(1, len(df)):
        prev_diff = df["macd"].iloc[i-1] - df["macd_signal"].iloc[i-1]
        curr_diff = df["macd"].iloc[i]   - df["macd_signal"].iloc[i]
        if prev_diff < 0 and curr_diff >= 0:
            signals.iloc[i] = "BUY"
        elif prev_diff > 0 and curr_diff <= 0:
            signals.iloc[i] = "SELL"
    return signals


# -----------------------------------------------
# STRATEGY 3: Bollinger Bands (Original)
# -----------------------------------------------
def bollinger_strategy(df):
    df      = calculate_bollinger_bands(df)
    signals = pd.Series("HOLD", index=df.index)
    for i in range(1, len(df)):
        if df["close"].iloc[i] <= df["bb_lower"].iloc[i]:
            signals.iloc[i] = "BUY"
        elif df["close"].iloc[i] >= df["bb_upper"].iloc[i]:
            signals.iloc[i] = "SELL"
    return signals


# -----------------------------------------------
# STRATEGY 4: EMA Crossover (NEW)
# BUY  when fast EMA (9) crosses above slow EMA (21)
# SELL when fast EMA (9) crosses below slow EMA (21)
# -----------------------------------------------
def ema_crossover_strategy(df):
    df      = calculate_ema_crossover(df)
    signals = pd.Series("HOLD", index=df.index)
    for i in range(1, len(df)):
        prev_diff = df["ema_fast_cross"].iloc[i-1] - df["ema_slow_cross"].iloc[i-1]
        curr_diff = df["ema_fast_cross"].iloc[i]   - df["ema_slow_cross"].iloc[i]
        if prev_diff < 0 and curr_diff >= 0:
            signals.iloc[i] = "BUY"
        elif prev_diff > 0 and curr_diff <= 0:
            signals.iloc[i] = "SELL"
    return signals


# -----------------------------------------------
# STRATEGY 5: RSI + MACD Confluence (NEW)
# BUY  only when BOTH RSI oversold AND MACD bullish cross
# SELL only when BOTH RSI overbought AND MACD bearish cross
# Higher quality signals — fewer but more accurate
# -----------------------------------------------
def rsi_macd_confluence_strategy(df):
    df      = calculate_macd(df)
    signals = pd.Series("HOLD", index=df.index)

    if "rsi" not in df.columns:
        return signals

    for i in range(1, len(df)):
        macd_cross_up   = (df["macd"].iloc[i-1] - df["macd_signal"].iloc[i-1]) < 0 and \
                          (df["macd"].iloc[i]   - df["macd_signal"].iloc[i])   >= 0
        macd_cross_down = (df["macd"].iloc[i-1] - df["macd_signal"].iloc[i-1]) > 0 and \
                          (df["macd"].iloc[i]   - df["macd_signal"].iloc[i])   <= 0

        rsi_oversold    = df["rsi"].iloc[i] < 45   # relaxed threshold
        rsi_overbought  = df["rsi"].iloc[i] > 55   # relaxed threshold

        if macd_cross_up and rsi_oversold:
            signals.iloc[i] = "BUY"
        elif macd_cross_down and rsi_overbought:
            signals.iloc[i] = "SELL"

    return signals


# -----------------------------------------------
# STRATEGY 6: Volume Confirmed (NEW)
# Only take signals when volume is above average
# High volume = stronger conviction behind the move
# -----------------------------------------------
def volume_confirmed_strategy(df):
    df      = calculate_macd(df)
    df      = calculate_volume_ma(df)
    signals = pd.Series("HOLD", index=df.index)

    if "rsi" not in df.columns:
        return signals

    for i in range(1, len(df)):
        high_volume     = df["volume_ratio"].iloc[i] > 1.5   # 50% above average
        macd_cross_up   = (df["macd"].iloc[i-1] - df["macd_signal"].iloc[i-1]) < 0 and \
                          (df["macd"].iloc[i]   - df["macd_signal"].iloc[i])   >= 0
        macd_cross_down = (df["macd"].iloc[i-1] - df["macd_signal"].iloc[i-1]) > 0 and \
                          (df["macd"].iloc[i]   - df["macd_signal"].iloc[i])   <= 0
        rsi_ok_buy      = df["rsi"].iloc[i] < 60
        rsi_ok_sell     = df["rsi"].iloc[i] > 40

        if macd_cross_up and high_volume and rsi_ok_buy:
            signals.iloc[i] = "BUY"
        elif macd_cross_down and high_volume and rsi_ok_sell:
            signals.iloc[i] = "SELL"

    return signals


# -----------------------------------------------
# STRATEGY 7: Trend Following (NEW)
# Only trade IN the direction of the trend
# Uses ADX to confirm trend strength
# -----------------------------------------------
def trend_following_strategy(df):
    df      = calculate_macd(df)
    df      = calculate_adx(df)
    signals = pd.Series("HOLD", index=df.index)

    if "rsi" not in df.columns:
        return signals

    for i in range(2, len(df)):
        strong_trend    = df["adx"].iloc[i] > 25           # ADX > 25 = strong trend
        uptrend         = df["plus_di"].iloc[i] > df["minus_di"].iloc[i]
        downtrend       = df["minus_di"].iloc[i] > df["plus_di"].iloc[i]
        macd_cross_up   = (df["macd"].iloc[i-1] - df["macd_signal"].iloc[i-1]) < 0 and \
                          (df["macd"].iloc[i]   - df["macd_signal"].iloc[i])   >= 0
        macd_cross_down = (df["macd"].iloc[i-1] - df["macd_signal"].iloc[i-1]) > 0 and \
                          (df["macd"].iloc[i]   - df["macd_signal"].iloc[i])   <= 0

        # Only buy in uptrend with strong ADX
        if macd_cross_up and strong_trend and uptrend:
            signals.iloc[i] = "BUY"
        # Only sell in downtrend with strong ADX
        elif macd_cross_down and strong_trend and downtrend:
            signals.iloc[i] = "SELL"

    return signals


# -----------------------------------------------
# STRATEGY 8: Stochastic + RSI (NEW)
# Double oscillator confirmation
# Both must be oversold to BUY, both overbought to SELL
# -----------------------------------------------
def stochastic_rsi_strategy(df):
    df      = calculate_stochastic(df)
    signals = pd.Series("HOLD", index=df.index)

    if "rsi" not in df.columns:
        return signals

    for i in range(1, len(df)):
        stoch_oversold   = df["stoch_k"].iloc[i] < 25 and df["stoch_d"].iloc[i] < 25
        stoch_overbought = df["stoch_k"].iloc[i] > 75 and df["stoch_d"].iloc[i] > 75
        rsi_oversold     = df["rsi"].iloc[i] < 40
        rsi_overbought   = df["rsi"].iloc[i] > 60
        stoch_cross_up   = df["stoch_k"].iloc[i] > df["stoch_d"].iloc[i] and \
                           df["stoch_k"].iloc[i-1] <= df["stoch_d"].iloc[i-1]
        stoch_cross_down = df["stoch_k"].iloc[i] < df["stoch_d"].iloc[i] and \
                           df["stoch_k"].iloc[i-1] >= df["stoch_d"].iloc[i-1]

        if stoch_oversold and rsi_oversold and stoch_cross_up:
            signals.iloc[i] = "BUY"
        elif stoch_overbought and rsi_overbought and stoch_cross_down:
            signals.iloc[i] = "SELL"

    return signals


# -----------------------------------------------
# ORIGINAL COMBINED STRATEGY
# -----------------------------------------------
def combined_strategy(df):
    rsi_sig  = rsi_strategy(df)
    macd_sig = macd_strategy(df)
    bb_sig   = bollinger_strategy(df)
    signals  = pd.Series("HOLD", index=df.index)

    for i in range(len(df)):
        buy_count  = sum([rsi_sig.iloc[i] == "BUY",  macd_sig.iloc[i] == "BUY",  bb_sig.iloc[i] == "BUY"])
        sell_count = sum([rsi_sig.iloc[i] == "SELL", macd_sig.iloc[i] == "SELL", bb_sig.iloc[i] == "SELL"])

        # Require a majority. Previously one BUY vote could win even when
        # two indicators were voting SELL because BUY was checked first.
        if buy_count > sell_count:
            if buy_count >= 2:
                signals.iloc[i] = "STRONG BUY"
            elif buy_count == 1:
                signals.iloc[i] = "BUY"
        elif sell_count > buy_count:
            if sell_count >= 2:
                signals.iloc[i] = "STRONG SELL"
            elif sell_count == 1:
                signals.iloc[i] = "SELL"

    return signals


# -----------------------------------------------
# ADVANCED COMBINED STRATEGY (NEW — Best Win Rate)
# Uses ALL 8 strategies with weighted voting
# -----------------------------------------------

def calculate_signal_scores(df, i):
    """
    Compute buy_score and sell_score for a single candle index i.
    Extracted so both advanced_combined_strategy() and
    get_current_signal() can share the same scoring logic
    without duplicating it or recomputing the whole series.
    """
    buy_score  = 0
    sell_score = 0

    # --- RSI (weight 1) ---
    rsi_val = df["rsi"].iloc[i] if "rsi" in df.columns else None
    if rsi_val is not None and not pd.isna(rsi_val):
        if rsi_val > 55:
            buy_score += 1
        elif rsi_val < 45:
            sell_score += 1

    # --- MACD (weight 1) ---
    macd_diff = df["macd"].iloc[i] - df["macd_signal"].iloc[i]
    if not pd.isna(macd_diff):
        if macd_diff > 0:
            buy_score += 1
        else:
            sell_score += 1

    # --- Bollinger %B (weight 1) ---
    bb_pct = df["bb_pct"].iloc[i] if "bb_pct" in df.columns else None
    if bb_pct is not None and not pd.isna(bb_pct):
        if bb_pct < 0.4:
            buy_score += 1
        elif bb_pct > 0.6:
            sell_score += 1

    # --- EMA crossover (weight 1) ---
    if "ema_fast_cross" in df.columns:
        ema_diff = df["ema_fast_cross"].iloc[i] - df["ema_slow_cross"].iloc[i]
        if not pd.isna(ema_diff):
            if ema_diff > 0:
                buy_score += 1
            else:
                sell_score += 1

    # --- RSI+MACD Confluence (weight 3) ---
    if rsi_val is not None and not pd.isna(rsi_val) and not pd.isna(macd_diff):
        if rsi_val > 50 and macd_diff > 0:
            buy_score += 3
        elif rsi_val < 50 and macd_diff < 0:
            sell_score += 3

    # --- Volume confirmed (weight 3) ---
    vol_ratio = df["volume_ratio"].iloc[i] if "volume_ratio" in df.columns else None
    if vol_ratio is not None and not pd.isna(vol_ratio) and vol_ratio > 1.2:
        if macd_diff > 0:
            buy_score += 3
        elif macd_diff < 0:
            sell_score += 3

    # --- Trend following / ADX (weight 2) ---
    adx_val = df["adx"].iloc[i] if "adx" in df.columns else None
    if adx_val is not None and not pd.isna(adx_val) and adx_val > 25:
        if df["plus_di"].iloc[i] > df["minus_di"].iloc[i]:
            buy_score += 2
        else:
            sell_score += 2

    # --- Stochastic (weight 2) ---
    if "stoch_k" in df.columns and "stoch_d" in df.columns:
        k_val, d_val = df["stoch_k"].iloc[i], df["stoch_d"].iloc[i]
        if not pd.isna(k_val) and not pd.isna(d_val):
            if k_val > d_val and k_val < 80:
                buy_score += 2
            elif k_val < d_val and k_val > 20:
                sell_score += 2

    return buy_score, sell_score


def advanced_combined_strategy(df):
    """
    Weighted voting system using CURRENT STATE (not just crossing events).
    Total possible: 14 points
    BUY  threshold: >= 4 points   STRONG BUY: >= 7
    SELL threshold: >= 4 points   STRONG SELL: >= 7
    """
    df = calculate_macd(df)
    df = calculate_bollinger_bands(df)
    df = calculate_ema_crossover(df)
    df = calculate_adx(df)
    df = calculate_stochastic(df)
    df = calculate_volume_ma(df)

    signals = pd.Series("HOLD", index=df.index)

    for i in range(len(df)):
        buy_score, sell_score = calculate_signal_scores(df, i)

        # Never choose BUY merely because its condition is checked first.
        # A signal is valid only when that side has more evidence than the
        # opposing side and reaches its threshold.
        if buy_score > sell_score:
            if buy_score >= 7:
                signals.iloc[i] = "STRONG BUY"
            elif buy_score >= 4:
                signals.iloc[i] = "BUY"
        elif sell_score > buy_score:
            if sell_score >= 7:
                signals.iloc[i] = "STRONG SELL"
            elif sell_score >= 4:
                signals.iloc[i] = "SELL"

    return signals


# -----------------------------------------------
# GET CURRENT SIGNAL FOR DASHBOARD
# -----------------------------------------------
STRATEGY_FUNCTIONS = {
    "rsi":                  rsi_strategy,
    "macd":                 macd_strategy,
    "bollinger":            bollinger_strategy,
    "ema_crossover":        ema_crossover_strategy,
    "rsi_macd_confluence":  rsi_macd_confluence_strategy,
    "volume_confirmed":     volume_confirmed_strategy,
    "trend_following":      trend_following_strategy,
    "stochastic_rsi":       stochastic_rsi_strategy,
    "combined":             combined_strategy,
    "advanced_combined":    advanced_combined_strategy,
}


def get_live_state_signal(df, strategy, i=None):
    """
    Returns a CURRENT-STATE signal (not crossing-event) for the given
    strategy at candle index i (defaults to the last candle).

    Backtesting strategies use crossing-events by design (correct for
    simulating trade entries/exits across history). But for a single
    live candle, "did it cross THIS exact candle" is almost always
    False, making the live signal show HOLD even when conditions are
    clearly bullish/bearish. This function answers the more useful
    live question: "is it bullish or bearish RIGHT NOW".
    """
    if i is None:
        i = len(df) - 1

    df = calculate_macd(df)
    df = calculate_bollinger_bands(df)
    df = calculate_ema_crossover(df)
    df = calculate_adx(df)
    df = calculate_stochastic(df)
    df = calculate_volume_ma(df)

    rsi_val   = df["rsi"].iloc[i] if "rsi" in df.columns else None
    macd_diff = df["macd"].iloc[i] - df["macd_signal"].iloc[i] if "macd" in df.columns else None
    bb_pct    = df["bb_pct"].iloc[i] if "bb_pct" in df.columns else None
    vol_ratio = df["volume_ratio"].iloc[i] if "volume_ratio" in df.columns else None
    adx_val   = df["adx"].iloc[i] if "adx" in df.columns else None

    if strategy == "rsi":
        if rsi_val is None or pd.isna(rsi_val): return "HOLD"
        if rsi_val < 30: return "BUY"
        if rsi_val > 70: return "SELL"
        return "HOLD"

    if strategy == "macd":
        if macd_diff is None or pd.isna(macd_diff): return "HOLD"
        return "BUY" if macd_diff > 0 else "SELL"

    if strategy == "bollinger":
        if bb_pct is None or pd.isna(bb_pct): return "HOLD"
        if bb_pct <= 0.1: return "BUY"
        if bb_pct >= 0.9: return "SELL"
        return "HOLD"

    if strategy == "ema_crossover":
        if "ema_fast_cross" not in df.columns: return "HOLD"
        diff = df["ema_fast_cross"].iloc[i] - df["ema_slow_cross"].iloc[i]
        if pd.isna(diff): return "HOLD"
        return "BUY" if diff > 0 else "SELL"

    if strategy == "rsi_macd_confluence":
        if rsi_val is None or pd.isna(rsi_val) or macd_diff is None or pd.isna(macd_diff): return "HOLD"
        if rsi_val > 50 and macd_diff > 0: return "BUY"
        if rsi_val < 50 and macd_diff < 0: return "SELL"
        return "HOLD"

    if strategy == "volume_confirmed":
        if vol_ratio is None or pd.isna(vol_ratio) or macd_diff is None or pd.isna(macd_diff): return "HOLD"
        if vol_ratio > 1.2:
            return "BUY" if macd_diff > 0 else "SELL"
        return "HOLD"

    if strategy == "trend_following":
        if adx_val is None or pd.isna(adx_val): return "HOLD"
        if adx_val > 25:
            uptrend = df["plus_di"].iloc[i] > df["minus_di"].iloc[i]
            return "BUY" if uptrend else "SELL"
        return "HOLD"

    if strategy == "stochastic_rsi":
        if "stoch_k" not in df.columns: return "HOLD"
        k_val, d_val = df["stoch_k"].iloc[i], df["stoch_d"].iloc[i]
        if pd.isna(k_val) or pd.isna(d_val): return "HOLD"
        if k_val < 25 and rsi_val is not None and rsi_val < 40 and k_val > d_val: return "BUY"
        if k_val > 75 and rsi_val is not None and rsi_val > 60 and k_val < d_val: return "SELL"
        return "HOLD"

    if strategy == "combined":
        # combined = majority vote of rsi/macd/bollinger current states
        buy = 0
        sell = 0
        if rsi_val is not None and not pd.isna(rsi_val):
            if rsi_val > 55: buy += 1
            elif rsi_val < 45: sell += 1
        if macd_diff is not None and not pd.isna(macd_diff):
            if macd_diff > 0: buy += 1
            else: sell += 1
        if bb_pct is not None and not pd.isna(bb_pct):
            if bb_pct < 0.4: buy += 1
            elif bb_pct > 0.6: sell += 1
        if buy > sell:
            if buy >= 2: return "STRONG BUY"
            if buy == 1: return "BUY"
        elif sell > buy:
            if sell >= 2: return "STRONG SELL"
            if sell == 1: return "SELL"
        return "HOLD"

    # advanced_combined uses its own proper state-based scoring already
    signals = advanced_combined_strategy(df)
    return signals.iloc[i]


def get_live_conviction(df, strategy, i=None):
    """
    Returns a CONTINUOUS conviction score (0.0 to 1.0) reflecting how
    strongly the current data supports the signal — NOT a fixed bucket.

    Without this, every strategy that outputs "BUY" would show the same
    generic conviction number regardless of how strong the underlying
    evidence actually is, making genuinely different strategies look
    coincidentally identical.
    """
    if i is None:
        i = len(df) - 1

    rsi_val   = df["rsi"].iloc[i] if "rsi" in df.columns else None
    macd_diff = df["macd"].iloc[i] - df["macd_signal"].iloc[i] if "macd" in df.columns else None
    bb_pct    = df["bb_pct"].iloc[i] if "bb_pct" in df.columns else None
    vol_ratio = df["volume_ratio"].iloc[i] if "volume_ratio" in df.columns else None
    adx_val   = df["adx"].iloc[i] if "adx" in df.columns else None
    price     = df["close"].iloc[i] if "close" in df.columns else None

    def clamp(x):
        return max(0.0, min(1.0, x))

    if strategy == "rsi":
        if rsi_val is None or pd.isna(rsi_val): return 0.0
        if rsi_val <= 30: return clamp((30 - rsi_val) / 30)
        if rsi_val >= 70: return clamp((rsi_val - 70) / 30)
        return 0.0

    if strategy == "macd":
        if macd_diff is None or pd.isna(macd_diff) or not price: return 0.0
        rel = abs(macd_diff) / price
        return clamp(rel / 0.01)  # 1% of price = full conviction

    if strategy == "bollinger":
        if bb_pct is None or pd.isna(bb_pct): return 0.0
        if bb_pct <= 0.1: return clamp((0.1 - bb_pct) / 0.1)
        if bb_pct >= 0.9: return clamp((bb_pct - 0.9) / 0.1)
        return 0.0

    if strategy == "ema_crossover":
        if "ema_fast_cross" not in df.columns or not price: return 0.0
        diff = df["ema_fast_cross"].iloc[i] - df["ema_slow_cross"].iloc[i]
        if pd.isna(diff): return 0.0
        return clamp(abs(diff) / price / 0.01)

    if strategy == "rsi_macd_confluence":
        if rsi_val is None or pd.isna(rsi_val) or macd_diff is None or pd.isna(macd_diff) or not price:
            return 0.0
        agree = (rsi_val > 50 and macd_diff > 0) or (rsi_val < 50 and macd_diff < 0)
        if not agree:
            return 0.0
        rsi_strength  = clamp(abs(rsi_val - 50) / 50)
        macd_strength = clamp(abs(macd_diff) / price / 0.01)
        return (rsi_strength + macd_strength) / 2

    if strategy == "volume_confirmed":
        if vol_ratio is None or pd.isna(vol_ratio) or macd_diff is None or pd.isna(macd_diff):
            return 0.0
        if vol_ratio <= 1.2:
            return 0.0
        vol_strength  = clamp((vol_ratio - 1.2) / 1.0)
        macd_strength = clamp(abs(macd_diff) / price / 0.01) if price else 0.0
        return (vol_strength + macd_strength) / 2

    if strategy == "trend_following":
        if adx_val is None or pd.isna(adx_val):
            return 0.0
        if adx_val <= 25:
            return 0.0
        return clamp((adx_val - 25) / 25)

    if strategy == "stochastic_rsi":
        if "stoch_k" not in df.columns:
            return 0.0
        k_val, d_val = df["stoch_k"].iloc[i], df["stoch_d"].iloc[i]
        if pd.isna(k_val) or pd.isna(d_val):
            return 0.0
        if k_val < 25 and rsi_val is not None and rsi_val < 40 and k_val > d_val:
            return clamp((25 - k_val) / 25)
        if k_val > 75 and rsi_val is not None and rsi_val > 60 and k_val < d_val:
            return clamp((k_val - 75) / 25)
        return 0.0

    if strategy == "combined":
        buy, sell = 0, 0
        if rsi_val is not None and not pd.isna(rsi_val):
            if rsi_val > 55: buy += 1
            elif rsi_val < 45: sell += 1
        if macd_diff is not None and not pd.isna(macd_diff):
            if macd_diff > 0: buy += 1
            else: sell += 1
        if bb_pct is not None and not pd.isna(bb_pct):
            if bb_pct < 0.4: buy += 1
            elif bb_pct > 0.6: sell += 1
        return max(buy, sell) / 3.0

    # advanced_combined — use the real weighted 0-14 point score
    buy_score, sell_score = calculate_signal_scores(df, i)
    return max(buy_score, sell_score) / 14.0


def get_current_signal(df, strategy="advanced_combined"):
    df = calculate_macd(df)
    df = calculate_bollinger_bands(df)
    df = calculate_ema_crossover(df)
    df = calculate_adx(df)
    df = calculate_stochastic(df)
    df = calculate_volume_ma(df)

    latest = get_live_state_signal(df, strategy)

    reasons = []
    rsi_val = df["rsi"].iloc[-1] if "rsi" in df.columns else None

    if rsi_val:
        if rsi_val < 30:
            reasons.append(f"RSI {rsi_val:.1f} — oversold, strong buy signal")
        elif rsi_val > 70:
            reasons.append(f"RSI {rsi_val:.1f} — overbought, strong sell signal")
        elif rsi_val < 45:
            reasons.append(f"RSI {rsi_val:.1f} — bearish zone")
        elif rsi_val > 55:
            reasons.append(f"RSI {rsi_val:.1f} — bullish zone")
        else:
            reasons.append(f"RSI {rsi_val:.1f} — neutral zone")

    macd_diff = df["macd"].iloc[-1] - df["macd_signal"].iloc[-1]
    if macd_diff > 0:
        reasons.append(f"MACD above signal (+{macd_diff:.3f}) — bullish momentum")
    else:
        reasons.append(f"MACD below signal ({macd_diff:.3f}) — bearish momentum")

    bb_pct = df["bb_pct"].iloc[-1]
    if bb_pct < 0.2:
        reasons.append(f"Price near lower Bollinger Band ({bb_pct:.0%}) — oversold")
    elif bb_pct > 0.8:
        reasons.append(f"Price near upper Bollinger Band ({bb_pct:.0%}) — overbought")
    else:
        reasons.append(f"Price mid Bollinger Band ({bb_pct:.0%}) — neutral")

    adx_val = df["adx"].iloc[-1] if "adx" in df.columns else None
    if adx_val:
        if adx_val > 25:
            trend_dir = "uptrend" if df["plus_di"].iloc[-1] > df["minus_di"].iloc[-1] else "downtrend"
            reasons.append(f"ADX {adx_val:.1f} — strong {trend_dir} confirmed")
        else:
            reasons.append(f"ADX {adx_val:.1f} — weak trend, ranging market")

    vol_ratio = df["volume_ratio"].iloc[-1] if "volume_ratio" in df.columns else None
    if vol_ratio:
        if vol_ratio > 1.5:
            reasons.append(f"Volume {vol_ratio:.1f}x above average — high conviction")
        elif vol_ratio < 0.7:
            reasons.append(f"Volume {vol_ratio:.1f}x below average — low conviction")
        else:
            reasons.append(f"Volume {vol_ratio:.1f}x average — normal activity")

    support, resistance = calculate_support_resistance(df)
    buy_score, sell_score = calculate_signal_scores(df, len(df) - 1)
    conviction = get_live_conviction(df, strategy, len(df) - 1)

    return {
        "signal":      latest,
        "conviction":  conviction,  # 0..1, real per-strategy strength (not a fixed bucket)
        "buy_score":   buy_score,
        "sell_score":  sell_score,
        "max_score":   14,
        "reasons":     reasons,
        "support":     support,
        "resistance":  resistance,
        "macd":        float(df["macd"].iloc[-1]),
        "macd_signal": float(df["macd_signal"].iloc[-1]),
        "bb_upper":    float(df["bb_upper"].iloc[-1]),
        "bb_lower":    float(df["bb_lower"].iloc[-1]),
        "bb_mid":      float(df["bb_mid"].iloc[-1]),
        "adx":         float(df["adx"].iloc[-1]) if "adx" in df.columns else 0,
        "volume_ratio": float(df["volume_ratio"].iloc[-1]) if "volume_ratio" in df.columns else 1,
    }
