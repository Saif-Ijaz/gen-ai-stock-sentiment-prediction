# app/backtesting.py
import pandas as pd
import numpy as np
from app.strategies import (
    rsi_strategy, macd_strategy, bollinger_strategy,
    combined_strategy, ema_crossover_strategy,
    rsi_macd_confluence_strategy, volume_confirmed_strategy,
    trend_following_strategy, stochastic_rsi_strategy,
    advanced_combined_strategy,
    calculate_macd, calculate_bollinger_bands
)


def run_backtest(df, strategy="advanced_combined", initial_capital=10000):
    df = df.copy().reset_index(drop=True)
    if df.empty:
        return [], {
            "initial_capital": initial_capital, "final_value": initial_capital,
            "total_return": 0, "buy_hold_return": 0, "total_trades": 0,
            "winning_trades": 0, "losing_trades": 0, "win_rate": 0,
            "avg_win": 0, "avg_loss": 0, "profit_factor": 0,
            "max_drawdown": 0, "sharpe_ratio": 0,
        }, []
    df = calculate_macd(df)
    df = calculate_bollinger_bands(df)

    strategy_map = {
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

    signal_func = strategy_map.get(strategy, advanced_combined_strategy)
    signals     = signal_func(df)

    capital     = initial_capital
    position    = 0
    entry_price = 0
    trades      = []
    equity      = []

    for i in range(len(df)):
        price  = float(df["close"].iloc[i])
        signal = signals.iloc[i]

        if signal in ("BUY", "STRONG BUY") and position == 0 and capital > 0:
            shares      = capital / price
            position    = shares
            entry_price = price
            capital     = 0
            trades.append({
                "type":   "BUY",
                "index":  i,
                "price":  round(price, 2),
                "shares": round(shares, 4),
                "value":  round(shares * price, 2),
                "signal": signal
            })

        elif signal in ("SELL", "STRONG SELL") and position > 0:
            value    = position * price
            pnl      = value - (position * entry_price)
            pnl_pct  = (pnl / (position * entry_price)) * 100
            capital  = value
            trades.append({
                "type":    "SELL",
                "index":   i,
                "price":   round(price, 2),
                "shares":  round(position, 4),
                "value":   round(value, 2),
                "pnl":     round(pnl, 2),
                "pnl_pct": round(pnl_pct, 2),
                "signal":  signal
            })
            position = 0

        portfolio_value = capital + (position * price)
        equity.append({
            "index": i,
            "price": price,
            "value": round(portfolio_value, 2),
            "signal": signal
        })

    # Close an open position at the final available price and record it as a
    # normal SELL.  It must be included in the trade table, win rate, and
    # profit-factor calculation; previously only Total Return included it.
    if position > 0:
        final_price = float(df["close"].iloc[-1])
        value       = position * final_price
        pnl         = value - (position * entry_price)
        pnl_pct     = (pnl / (position * entry_price)) * 100
        capital     = value
        trades.append({
            "type": "SELL",
            "index": len(df) - 1,
            "price": round(final_price, 2),
            "shares": round(position, 4),
            "value": round(value, 2),
            "pnl": round(pnl, 2),
            "pnl_pct": round(pnl_pct, 2),
            "signal": "FINAL CLOSE",
        })
        if equity:
            equity[-1]["value"] = round(capital, 2)
            equity[-1]["signal"] = "FINAL CLOSE"
        position    = 0

    final_value  = capital
    sell_trades  = [t for t in trades if t["type"] == "SELL"]
    winning      = [t for t in sell_trades if t.get("pnl", 0) > 0]
    losing       = [t for t in sell_trades if t.get("pnl", 0) <= 0]

    total_return  = ((final_value - initial_capital) / initial_capital) * 100
    win_rate      = (len(winning) / len(sell_trades) * 100) if sell_trades else 0
    avg_win       = np.mean([t["pnl"] for t in winning]) if winning else 0
    avg_loss      = np.mean([t["pnl"] for t in losing])  if losing  else 0
    gross_profit  = sum(t["pnl"] for t in winning)
    gross_loss    = abs(sum(t["pnl"] for t in losing))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
    max_drawdown  = calculate_max_drawdown([e["value"] for e in equity])
    bh_return     = ((float(df["close"].iloc[-1]) - float(df["close"].iloc[0])) / float(df["close"].iloc[0])) * 100
    sharpe        = calculate_sharpe([e["value"] for e in equity])

    metrics = {
        "initial_capital":  initial_capital,
        "final_value":      round(final_value, 2),
        "total_return":     round(total_return, 2),
        "buy_hold_return":  round(bh_return, 2),
        "total_trades":     len(sell_trades),
        "winning_trades":   len(winning),
        "losing_trades":    len(losing),
        "win_rate":         round(win_rate, 2),
        "avg_win":          round(avg_win, 2),
        "avg_loss":         round(avg_loss, 2),
        "profit_factor":    round(profit_factor, 2),
        "max_drawdown":     round(max_drawdown, 2),
        "sharpe_ratio":     round(sharpe, 2),
    }

    return trades, metrics, equity


def calculate_max_drawdown(equity_values):
    if not equity_values:
        return 0
    peak   = equity_values[0]
    max_dd = 0
    for v in equity_values:
        if v > peak:
            peak = v
        dd = (peak - v) / peak * 100
        if dd > max_dd:
            max_dd = dd
    return max_dd


def calculate_sharpe(equity_values, risk_free=0.05):
    """Calculate Sharpe Ratio — higher is better (>1 is good)"""
    if len(equity_values) < 2:
        return 0
    returns = pd.Series(equity_values).pct_change().dropna()
    if returns.std() == 0:
        return 0
    # Annualize for 4H candles (6 candles/day × 252 trading days)
    annualized_return = returns.mean() * 6 * 252
    annualized_std    = returns.std() * np.sqrt(6 * 252)
    return (annualized_return - risk_free) / annualized_std
