# app/app_logic.py

def derive_key_drivers(result):
    """
    Derive explainable key drivers consistent with model output
    """
    drivers = []

    rsi = result.get("rsi")
    atr = result.get("atr_14")
    sentiment = result.get("sentiment", {})
    direction = result.get("prediction", "NEUTRAL")
    confidence = result.get("confidence", 0.5)
    predicted_pct = result.get("predicted_pct", 0)
    has_news = bool(result.get("rag_news"))

    # --- RSI Driver ---
    if rsi is not None:
        if rsi > 70:
            drivers.append(f"RSI at {rsi:.1f} — overbought territory, potential pullback")
        elif rsi < 30:
            drivers.append(f"RSI at {rsi:.1f} — oversold territory, potential bounce")
        elif 50 < rsi <= 70:
            drivers.append(f"RSI at {rsi:.1f} — moderately bullish momentum")
        elif 30 <= rsi < 50:
            drivers.append(f"RSI at {rsi:.1f} — moderately bearish momentum")
        else:
            drivers.append(f"RSI at {rsi:.1f} — neutral momentum")

    # --- ATR Volatility Driver ---
    if atr is not None:
        if atr > 5:
            drivers.append(f"ATR at {atr:.2f} — high volatility detected")
        elif atr > 2:
            drivers.append(f"ATR at {atr:.2f} — moderate volatility")
        else:
            drivers.append(f"ATR at {atr:.2f} — low volatility environment")

    # --- Sentiment Drivers ---
    pos = sentiment.get("positive", 0)
    neg = sentiment.get("negative", 0)
    neu = sentiment.get("neutral", 1)

    if has_news:
        if pos > neg and pos > neu and pos > 0.3:
            drivers.append(f"News sentiment positive ({pos*100:.0f}%) — bullish signal")
        elif neg > pos and neg > neu and neg > 0.3:
            drivers.append(f"News sentiment negative ({neg*100:.0f}%) — bearish signal")
        else:
            drivers.append(f"News sentiment neutral ({neu*100:.0f}%) — no clear bias")
    else:
        # Show sentiment even without RAG news
        if pos > 0.3:
            drivers.append(f"Background sentiment skews positive ({pos*100:.0f}%)")
        elif neg > 0.3:
            drivers.append(f"Background sentiment skews negative ({neg*100:.0f}%)")

    # --- Model Confidence Driver ---
    if confidence >= 0.7:
        drivers.append(f"Model confidence HIGH ({confidence*100:.0f}%) — strong signal")
    elif confidence >= 0.55:
        drivers.append(f"Model confidence MEDIUM ({confidence*100:.0f}%) — moderate signal")
    else:
        drivers.append(f"Model confidence LOW ({confidence*100:.0f}%) — weak signal, treat with caution")

    # --- Direction Driver ---
    if direction == "UP":
        drivers.append(f"Model predicts upward move of +{abs(predicted_pct):.2f}%")
    elif direction == "DOWN":
        drivers.append(f"Model predicts downward move of -{abs(predicted_pct):.2f}%")
    else:
        drivers.append("Model predicts neutral — no clear directional bias")

    return drivers


def estimate_percentage_move(confidence):
    """
    Convert model confidence to expected % move (max ~3%)
    """
    return round(confidence * 3, 2)


def confidence_label(confidence):
    """
    Map probability to qualitative confidence
    """
    if confidence >= 0.7:
        return "HIGH"
    elif confidence >= 0.4:
        return "MEDIUM"
    else:
        return "LOW"
