#!/usr/bin/env python3
"""
Analyze a single backtest trades CSV and output summary metrics.
"""

import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime, timezone

# ===== CONFIGURATION =====
TRADES_CSV = "FVG_projectX_bot/backtest/BTC_BACKTEST_NEW/backtest_trades.csv"
PRICE_CSV = "FVG_projectX_bot/backtest/data/BTCUSDT_PERP_15m.csv"
OUTPUT_METRICS_CSV = "FVG_projectX_bot/backtest/BTC_BACKTEST_NEW/backtest_trades_metrics.csv"
OUTPUT_MONTHLY_CSV = "FVG_projectX_bot/backtest/BTC_BACKTEST_NEW/backtest_trades_monthly_pnl.csv"
START_EQUITY = 50.0
# =========================

def calculate_avg_max_monthly_drawdown(trades_df, initial_balance):
    """
    Calculate average of maximum monthly drawdowns from trades DataFrame.
    For each month, finds the maximum drawdown (peak to trough), then averages all monthly maximums.
    
    Args:
        trades_df: DataFrame with trades (must have 'entry_time' and 'pnl' columns)
        initial_balance: Starting balance for the backtest
    
    Returns:
        Average of maximum monthly drawdowns (dollar value)
    """
    if trades_df.empty or "entry_time" not in trades_df.columns:
        return 0.0
    
    # Ensure entry_time is datetime
    trades_df = trades_df.copy()
    trades_df["entry_time"] = pd.to_datetime(trades_df["entry_time"], utc=True, errors="coerce")
    
    # Sort by entry time
    trades_df = trades_df.sort_values("entry_time").reset_index(drop=True)
    
    # Calculate cumulative PnL and equity curve
    trades_df["cumulative_pnl"] = trades_df["pnl"].cumsum()
    trades_df["balance"] = initial_balance + trades_df["cumulative_pnl"]
    
    # Group by month
    trades_df["year_month"] = trades_df["entry_time"].dt.to_period("M")
    
    monthly_drawdowns = []
    
    for month, month_trades in trades_df.groupby("year_month"):
        month_trades = month_trades.sort_values("entry_time").reset_index(drop=True)
        
        if len(month_trades) == 0:
            continue
        
        # Calculate maximum drawdown for this month
        # Drawdown = peak - trough (largest decline from a peak within the month)
        balances = month_trades["balance"].values
        peak = balances[0]  # Start with first balance of the month
        max_drawdown = 0.0
        
        for balance in balances:
            if balance > peak:
                peak = balance  # Update peak if we hit a new high
            drawdown = peak - balance  # Calculate drawdown from current peak
            if drawdown > max_drawdown:
                max_drawdown = drawdown  # Track the maximum drawdown in this month
        
        # Store the maximum drawdown for this month (include all months, even if drawdown is 0)
        monthly_drawdowns.append(max_drawdown)
    
    # Return average of maximum monthly drawdowns
    if monthly_drawdowns:
        return sum(monthly_drawdowns) / len(monthly_drawdowns)
    else:
        return 0.0

def calculate_losing_streaks(trades_df):
    """
    Calculate losing streak metrics from trades DataFrame.
    Returns:
        - most_losing_trades_in_row: Maximum consecutive losing trades
        - avg_most_losing_trades_per_month: Average of max consecutive losses per month
        - dollar_value_most_losses_in_row: Sum of PnL for longest losing streak
        - dollar_value_avg_most_losses_per_month: Average dollar value of max consecutive losses per month
    """
    if trades_df.empty:
        return 0, 0.0, 0.0, 0.0
    
    # Ensure entry_time is datetime
    if "entry_time" in trades_df.columns:
        trades_df["entry_time"] = pd.to_datetime(trades_df["entry_time"], utc=True, errors="coerce")
    
    # Sort by entry time to ensure correct order
    trades_df = trades_df.sort_values("entry_time").reset_index(drop=True)
    
    # Identify losing trades (pnl < 0)
    trades_df["is_loss"] = trades_df["pnl"] < 0
    
    # Calculate consecutive losing trades
    most_losing_trades_in_row = 0
    current_streak = 0
    longest_streak_start = None
    longest_streak_end = None
    
    for idx, is_loss in enumerate(trades_df["is_loss"]):
        if is_loss:
            current_streak += 1
            if current_streak > most_losing_trades_in_row:
                most_losing_trades_in_row = current_streak
                longest_streak_end = idx
                longest_streak_start = idx - current_streak + 1
        else:
            current_streak = 0
    
    # Calculate dollar value of most losses in a row
    dollar_value_most_losses_in_row = 0.0
    if longest_streak_start is not None and longest_streak_end is not None:
        longest_streak_trades = trades_df.iloc[longest_streak_start : longest_streak_end + 1]
        dollar_value_most_losses_in_row = longest_streak_trades["pnl"].sum()
    
    # Calculate monthly metrics
    if "entry_time" in trades_df.columns:
        trades_df["year_month"] = trades_df["entry_time"].dt.to_period("M")
        
        monthly_max_streaks = []
        monthly_max_streak_values = []
        
        for month, month_trades in trades_df.groupby("year_month"):
            month_trades = month_trades.sort_values("entry_time").reset_index(drop=True)
            month_trades["is_loss"] = month_trades["pnl"] < 0
            
            # Find all losing streaks in this month
            max_streak = 0
            max_streak_value = 0.0
            current_streak = 0
            current_streak_start = None
            
            for idx, is_loss in enumerate(month_trades["is_loss"]):
                if is_loss:
                    if current_streak == 0:
                        current_streak_start = idx
                    current_streak += 1
                else:
                    # End of streak
                    if current_streak > 0:
                        if current_streak > max_streak:
                            max_streak = current_streak
                            streak_end = idx - 1
                            streak_trades = month_trades.iloc[current_streak_start : streak_end + 1]
                            max_streak_value = streak_trades["pnl"].sum()
                    current_streak = 0
                    current_streak_start = None
            
            # Handle case where streak continues to end of month
            if current_streak > 0:
                if current_streak > max_streak:
                    max_streak = current_streak
                    streak_end = len(month_trades) - 1
                    streak_trades = month_trades.iloc[current_streak_start : streak_end + 1]
                    max_streak_value = streak_trades["pnl"].sum()
            
            if max_streak > 0:
                monthly_max_streaks.append(max_streak)
                monthly_max_streak_values.append(max_streak_value)
        
        avg_most_losing_trades_per_month = sum(monthly_max_streaks) / len(monthly_max_streaks) if monthly_max_streaks else 0.0
        dollar_value_avg_most_losses_per_month = sum(monthly_max_streak_values) / len(monthly_max_streak_values) if monthly_max_streak_values else 0.0
    else:
        avg_most_losing_trades_per_month = 0.0
        dollar_value_avg_most_losses_per_month = 0.0
    
    return (
        most_losing_trades_in_row,
        avg_most_losing_trades_per_month,
        dollar_value_most_losses_in_row,
        dollar_value_avg_most_losses_per_month,
    )


def _compute_equity_before(trades_df: pd.DataFrame) -> pd.Series:
    if "equity" in trades_df.columns and "pnl" in trades_df.columns:
        return trades_df["equity"] - trades_df["pnl"]
    return pd.Series([None] * len(trades_df), index=trades_df.index)


def _compute_sharpe(trades_df: pd.DataFrame) -> float:
    if trades_df.empty:
        return 0.0
    returns = None
    if "margin_per_trade" in trades_df.columns:
        margin = pd.to_numeric(trades_df["margin_per_trade"], errors="coerce")
        returns = trades_df["pnl"] / margin.replace(0, pd.NA)
    if returns is None or returns.isna().all():
        equity_before = _compute_equity_before(trades_df)
        returns = trades_df["pnl"] / equity_before.replace(0, pd.NA)
    returns = returns.dropna()
    if returns.empty:
        return 0.0
    mean = returns.mean()
    std = returns.std(ddof=0)
    if std == 0 or pd.isna(std):
        return 0.0
    return float(mean / std * (len(returns) ** 0.5))


def _load_price_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "timestamp" not in df.columns:
        return pd.DataFrame()
    ts_series = df["timestamp"]
    if pd.api.types.is_numeric_dtype(ts_series):
        max_val = pd.Series(ts_series).max()
        unit = "ms" if max_val > 10**12 else "s"
        df["timestamp"] = pd.to_datetime(ts_series, unit=unit, utc=True)
    else:
        numeric_ts = pd.to_numeric(ts_series, errors="coerce")
        if numeric_ts.notna().any():
            max_val = numeric_ts.max()
            unit = "ms" if max_val > 10**12 else "s"
            df["timestamp"] = pd.to_datetime(numeric_ts, unit=unit, utc=True)
        else:
            df["timestamp"] = pd.to_datetime(ts_series, utc=True, errors="coerce")
    df = df[df["timestamp"].notna()].sort_values("timestamp").reset_index(drop=True)
    return df


def _build_equity_curve_with_price(trades_df: pd.DataFrame, price_df: pd.DataFrame) -> pd.DataFrame:
    if trades_df.empty or price_df.empty or "close" not in price_df.columns:
        return pd.DataFrame()
    trades = trades_df.copy()
    trades["entry_time"] = pd.to_datetime(trades["entry_time"], utc=True, errors="coerce")
    trades["exit_time"] = pd.to_datetime(trades["exit_time"], utc=True, errors="coerce")
    trades = trades[trades["entry_time"].notna()].sort_values("entry_time").reset_index(drop=True)
    if trades.empty:
        return pd.DataFrame()

    equity_before = _compute_equity_before(trades)
    initial_balance = float(equity_before.iloc[0]) if len(equity_before) else 0.0

    open_positions = []
    realized_pnl = 0.0
    trade_idx = 0
    equity_rows = []

    for row in price_df.itertuples(index=False):
        ts = getattr(row, "timestamp", None)
        close = getattr(row, "close", None)
        if ts is None or close is None or pd.isna(close):
            continue

        while trade_idx < len(trades) and trades.loc[trade_idx, "entry_time"] <= ts:
            open_positions.append(trades.loc[trade_idx])
            trade_idx += 1

        still_open = []
        for trade in open_positions:
            exit_time = trade.get("exit_time")
            if pd.notna(exit_time) and exit_time <= ts:
                pnl_val = trade.get("pnl")
                if pd.notna(pnl_val):
                    realized_pnl += float(pnl_val)
            else:
                still_open.append(trade)
        open_positions = still_open

        unrealized = 0.0
        for trade in open_positions:
            entry_price = trade.get("entry_price")
            size = trade.get("order_size")
            side = trade.get("side")
            if pd.isna(entry_price) or pd.isna(size) or not side:
                continue
            direction = 1 if str(side).upper() == "BUY" else -1
            unrealized += (float(close) - float(entry_price)) * direction * float(size)

        equity = initial_balance + realized_pnl + unrealized
        equity_rows.append({"timestamp": ts, "equity": equity})

    return pd.DataFrame(equity_rows)


def _compute_price_sharpe(price_df: pd.DataFrame) -> float:
    if price_df.empty or "close" not in price_df.columns:
        return 0.0
    close = pd.to_numeric(price_df["close"], errors="coerce")
    returns = close.pct_change().dropna()
    if returns.empty:
        return 0.0
    mean = returns.mean()
    std = returns.std(ddof=0)
    if std == 0 or pd.isna(std):
        return 0.0
    # 15m bars -> 96 bars/day, 365 days/year
    annualization = (365 * 24 * 4) ** 0.5
    return float(mean / std * annualization)


def _compute_max_drawdown(trades_df: pd.DataFrame) -> float:
    if trades_df.empty or "equity" not in trades_df.columns:
        return 0.0
    equity = trades_df["equity"].astype(float)
    peak = equity.cummax()
    drawdown = peak - equity
    return float(drawdown.max())


def _compute_max_drawdown_from_equity(equity_df: pd.DataFrame) -> float:
    if equity_df.empty or "equity" not in equity_df.columns:
        return 0.0
    equity = pd.to_numeric(equity_df["equity"], errors="coerce").dropna()
    if equity.empty:
        return 0.0
    peak = equity.cummax()
    drawdown = peak - equity
    return float(drawdown.max())


def _compute_equity_sharpe(equity_df: pd.DataFrame) -> float:
    if equity_df.empty or "equity" not in equity_df.columns:
        return 0.0
    equity = pd.to_numeric(equity_df["equity"], errors="coerce").dropna()
    returns = equity.pct_change().dropna()
    if returns.empty:
        return 0.0
    mean = returns.mean()
    std = returns.std(ddof=0)
    if std == 0 or pd.isna(std):
        return 0.0
    annualization = (365 * 24 * 4) ** 0.5
    return float(mean / std * annualization)


def _compute_equity_losing_streak(equity_df: pd.DataFrame) -> tuple[int, float]:
    if equity_df.empty or "equity" not in equity_df.columns:
        return 0, 0.0
    equity = pd.to_numeric(equity_df["equity"], errors="coerce").dropna()
    deltas = equity.diff().dropna()
    max_streak = 0
    max_streak_value = 0.0
    current_streak = 0
    current_value = 0.0
    for delta in deltas:
        if delta < 0:
            current_streak += 1
            current_value += float(delta)
            if current_streak > max_streak:
                max_streak = current_streak
                max_streak_value = current_value
        else:
            current_streak = 0
            current_value = 0.0
    return max_streak, max_streak_value


def _plot_equity_vs_btc(
    equity_df: pd.DataFrame,
    price_df: pd.DataFrame,
    output_path: str,
) -> None:
    if equity_df.empty or price_df.empty or "close" not in price_df.columns:
        return
    price_df = price_df.copy()
    price_df["close"] = pd.to_numeric(price_df["close"], errors="coerce")
    price_df = price_df[price_df["close"].notna()]
    if price_df.empty:
        return
    initial_price = float(price_df["close"].iloc[0])
    if initial_price <= 0:
        return
    buy_hold = START_EQUITY * (price_df["close"] / initial_price)
    leverage = 50.0
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(
        equity_df["timestamp"],
        equity_df["equity"],
        label="Strategy Equity",
        linewidth=1.5,
    )
    ax.plot(
        price_df["timestamp"],
        buy_hold,
        label="Buy & Hold BTC",
        linewidth=1.2,
    )
    ax.set_title("Equity vs BTC Hold")
    ax.set_xlabel("Time")
    ax.set_ylabel("Equity")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _compute_monthly_stats(trades_df: pd.DataFrame) -> pd.DataFrame:
    df = trades_df.copy()
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True, errors="coerce")
    df = df[df["entry_time"].notna()].sort_values("entry_time").reset_index(drop=True)
    df["year_month"] = df["entry_time"].dt.to_period("M")
    df["equity_before"] = _compute_equity_before(df)
    rows = []
    for month, month_trades in df.groupby("year_month"):
        month_trades = month_trades.reset_index(drop=True)
        if month_trades.empty:
            continue
        start_balance = float(month_trades["equity_before"].iloc[0])
        end_balance = float(month_trades["equity"].iloc[-1])
        pnl = float(month_trades["pnl"].sum())
        pnl_pct = (pnl / START_EQUITY) * 100 if START_EQUITY else 0.0
        if "margin_per_trade" in month_trades.columns:
            margin = pd.to_numeric(month_trades["margin_per_trade"], errors="coerce")
            returns = month_trades["pnl"] / margin.replace(0, pd.NA)
        else:
            returns = month_trades["pnl"] / month_trades["equity_before"].replace(0, pd.NA)
        returns = returns.dropna()
        monthly_vol = float(returns.std(ddof=0)) if not returns.empty else 0.0
        rows.append(
            {
                "month": str(month),
                "num_trades": int(len(month_trades)),
                "start_balance": start_balance,
                "end_balance": end_balance,
                "pnl": pnl,
                "pnl_pct": pnl_pct,
                "volatility": monthly_vol,
            }
        )
    return pd.DataFrame(rows)


def _infer_bar_seconds(price_df: pd.DataFrame) -> float:
    if price_df.empty or "timestamp" not in price_df.columns:
        return 900.0
    ts = price_df["timestamp"].sort_values()
    if len(ts) < 2:
        return 900.0
    deltas = ts.diff().dropna()
    median_delta = deltas.median()
    if pd.isna(median_delta):
        return 900.0
    return float(median_delta.total_seconds())


def process_trades():
    if not os.path.exists(TRADES_CSV):
        print(f"❌ Error: {TRADES_CSV} not found!")
        return

    print(f"📖 Reading {TRADES_CSV}...")
    trades_df = pd.read_csv(TRADES_CSV)
    if trades_df.empty:
        print("⚠️ No trades found.")
        return

    trades_df["entry_time"] = pd.to_datetime(trades_df["entry_time"], utc=True, errors="coerce")
    trades_df = trades_df[trades_df["entry_time"].notna()].sort_values("entry_time").reset_index(drop=True)
    trades_df["exit_time"] = pd.to_datetime(trades_df["exit_time"], utc=True, errors="coerce")

    start_dt = trades_df["entry_time"].iloc[0]
    end_dt = trades_df["exit_time"].dropna().iloc[-1] if trades_df["exit_time"].notna().any() else trades_df["entry_time"].iloc[-1]
    days_of_backtest = (end_dt - start_dt).total_seconds() / 86400
    num_trades = int(len(trades_df))

    avg_trade = float(trades_df["pnl"].mean())
    avg_win = float(trades_df.loc[trades_df["pnl"] > 0, "pnl"].mean()) if (trades_df["pnl"] > 0).any() else 0.0
    avg_loss = float(trades_df.loc[trades_df["pnl"] < 0, "pnl"].mean()) if (trades_df["pnl"] < 0).any() else 0.0
    trade_sharpe = _compute_sharpe(trades_df)
    price_df = _load_price_data(PRICE_CSV)
    equity_curve = pd.DataFrame()
    if not price_df.empty:
        price_df = price_df[(price_df["timestamp"] >= start_dt) & (price_df["timestamp"] <= end_dt)]
        price_sharpe = _compute_price_sharpe(price_df)
        equity_curve = _build_equity_curve_with_price(trades_df, price_df)
    else:
        price_sharpe = 0.0
    equity_sharpe = _compute_equity_sharpe(equity_curve)
    max_drawdown = _compute_max_drawdown_from_equity(equity_curve) if not equity_curve.empty else _compute_max_drawdown(trades_df)
    equity_losing_streak, equity_losing_streak_value = _compute_equity_losing_streak(equity_curve)

    equity_before = _compute_equity_before(trades_df)
    initial_balance = float(equity_before.iloc[0]) if len(equity_before) else 0.0

    (
        most_losing_trades_in_row,
        avg_most_losing_trades_per_month,
        dollar_value_most_losses_in_row,
        dollar_value_avg_most_losses_per_month,
    ) = calculate_losing_streaks(trades_df)
    avg_max_monthly_drawdown = calculate_avg_max_monthly_drawdown(trades_df, initial_balance)

    monthly_df = _compute_monthly_stats(trades_df)
    if not monthly_df.empty:
        best_row = monthly_df.loc[monthly_df["pnl"].idxmax()]
        worst_row = monthly_df.loc[monthly_df["pnl"].idxmin()]
        best_month = best_row["month"]
        best_month_pnl = float(best_row["pnl"])
        worst_month = worst_row["month"]
        worst_month_pnl = float(worst_row["pnl"])
        avg_monthly_pnl_pct = float(monthly_df["pnl_pct"].mean())
        std_monthly_pnl_pct = float(monthly_df["pnl_pct"].std(ddof=0))
        months_below_30 = int((monthly_df["pnl_pct"] < 30).sum())
        pct_months_below_30 = (months_below_30 / len(monthly_df)) * 100 if len(monthly_df) else 0.0
        if monthly_df["volatility"].notna().any():
            pnl_vol_corr = float(monthly_df["pnl_pct"].corr(monthly_df["volatility"]))
        else:
            pnl_vol_corr = 0.0
        monthly_pnl_pct = "; ".join(
            [f"{row['month']}={row['pnl_pct']:.2f}%" for _, row in monthly_df.iterrows()]
        )
    else:
        best_month = ""
        best_month_pnl = 0.0
        worst_month = ""
        worst_month_pnl = 0.0
        avg_monthly_pnl_pct = 0.0
        std_monthly_pnl_pct = 0.0
        months_below_30 = 0
        pct_months_below_30 = 0.0
        pnl_vol_corr = 0.0
        monthly_pnl_pct = ""

    summary = {
        "days_of_backtest": round(days_of_backtest, 2),
        "num_trades": num_trades,
        "average_trade": round(avg_trade, 6),
        "average_winning_trade": round(avg_win, 6),
        "average_losing_trade": round(avg_loss, 6),
        "sharpe": round(equity_sharpe, 6),
        "trade_sharpe": round(trade_sharpe, 6),
        "price_sharpe": round(price_sharpe, 6),
        "max_drawdown": round(max_drawdown, 6),
        "best_month": best_month,
        "best_month_pnl": round(best_month_pnl, 6),
        "worst_month": worst_month,
        "worst_month_pnl": round(worst_month_pnl, 6),
        "avg_monthly_pnl_pct": round(avg_monthly_pnl_pct, 6),
        "std_monthly_pnl_pct": round(std_monthly_pnl_pct, 6),
        "months_below_30_pct": months_below_30,
        "pct_months_below_30_pct": round(pct_months_below_30, 6),
        "monthly_pnl_vol_corr": round(pnl_vol_corr, 6),
        "equity_losing_streak_bars": equity_losing_streak,
        "equity_losing_streak_value": round(equity_losing_streak_value, 6),
        "most_losing_trades_in_row": most_losing_trades_in_row,
        "avg_most_losing_trades_per_month": round(avg_most_losing_trades_per_month, 2),
        "dollar_value_most_losses_in_row": round(dollar_value_most_losses_in_row, 6),
        "dollar_value_avg_most_losses_per_month": round(dollar_value_avg_most_losses_per_month, 6),
        "avg_max_monthly_drawdown": round(avg_max_monthly_drawdown, 6),
        "monthly_pnl_pct": monthly_pnl_pct,
    }

    print("\n📊 Backtest Summary Metrics")
    for key, value in summary.items():
        print(f"{key}: {value}")

    os.makedirs(os.path.dirname(OUTPUT_METRICS_CSV), exist_ok=True)
    pd.DataFrame([summary]).to_csv(OUTPUT_METRICS_CSV, index=False)
    print(f"\n💾 Saved summary metrics to {OUTPUT_METRICS_CSV}")

    if not monthly_df.empty:
        monthly_df.to_csv(OUTPUT_MONTHLY_CSV, index=False)
        print(f"💾 Saved monthly PnL to {OUTPUT_MONTHLY_CSV}")

    if not equity_curve.empty and not price_df.empty:
        output_dir = os.path.dirname(OUTPUT_MONTHLY_CSV)
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, "equity_vs_btc.png")
        _plot_equity_vs_btc(equity_curve, price_df, plot_path)


if __name__ == "__main__":
    process_trades()

