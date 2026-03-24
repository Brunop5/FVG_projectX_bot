#!/usr/bin/env python3
"""
Evaluate backtest trades and export summary analytics.

Merged replacement for:
- add_losing_streak_metrics.py
- compute_trade_drawdown.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_TRADES = SCRIPT_DIR / "NEW_BTC_BACKTEST" / "backtest_trades.csv"
DEFAULT_PRICE = SCRIPT_DIR / "data" / "BTCUSDT_PERP_15m.csv"
DEFAULT_START_EQUITY = 50.0


def _load_trades(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _load_price_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if len(df.columns) == 1 and "<DATE>" in str(df.columns[0]) and "\t" in str(df.columns[0]):
        df = pd.read_csv(path, sep="\t")

    if "<DATE>" in df.columns and "<TIME>" in df.columns:
        df["timestamp"] = pd.to_datetime(
            df["<DATE>"].astype(str) + " " + df["<TIME>"].astype(str),
            errors="coerce",
            utc=True,
        )
        df = df.rename(
            columns={
                "<OPEN>": "open",
                "<HIGH>": "high",
                "<LOW>": "low",
                "<CLOSE>": "close",
            }
        )
    elif "timestamp" in df.columns:
        ts = df["timestamp"]
        if pd.api.types.is_numeric_dtype(ts):
            max_val = pd.Series(ts).max()
            unit = "ms" if max_val > 10**12 else "s"
            df["timestamp"] = pd.to_datetime(ts, unit=unit, utc=True, errors="coerce")
        else:
            numeric_ts = pd.to_numeric(ts, errors="coerce")
            if numeric_ts.notna().any():
                max_val = numeric_ts.max()
                unit = "ms" if max_val > 10**12 else "s"
                df["timestamp"] = pd.to_datetime(
                    numeric_ts,
                    unit=unit,
                    utc=True,
                    errors="coerce",
                )
            else:
                df["timestamp"] = pd.to_datetime(ts, utc=True, errors="coerce")
    else:
        return pd.DataFrame()

    df = df[df["timestamp"].notna()].sort_values("timestamp").reset_index(drop=True)
    return df


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
    rows = []

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

        rows.append({"timestamp": ts, "equity": initial_balance + realized_pnl + unrealized})

    return pd.DataFrame(rows)


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
    annualization = (365 * 24 * 4) ** 0.5  # 15m bars
    return float(mean / std * annualization)


def _compute_max_drawdown_from_equity(equity_df: pd.DataFrame) -> float:
    if equity_df.empty or "equity" not in equity_df.columns:
        return 0.0
    equity = pd.to_numeric(equity_df["equity"], errors="coerce").dropna()
    if equity.empty:
        return 0.0
    peak = equity.cummax()
    return float((peak - equity).max())


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
    deltas = pd.to_numeric(equity_df["equity"], errors="coerce").dropna().diff().dropna()
    max_streak = 0
    max_streak_value = 0.0
    cur_streak = 0
    cur_value = 0.0
    for delta in deltas:
        if delta < 0:
            cur_streak += 1
            cur_value += float(delta)
            if cur_streak > max_streak:
                max_streak = cur_streak
                max_streak_value = cur_value
        else:
            cur_streak = 0
            cur_value = 0.0
    return max_streak, max_streak_value


def _compute_monthly_stats(trades_df: pd.DataFrame, start_equity: float) -> pd.DataFrame:
    df = trades_df.copy()
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True, errors="coerce")
    df = df[df["entry_time"].notna()].sort_values("entry_time").reset_index(drop=True)
    if df.empty:
        return pd.DataFrame()
    df["year_month"] = df["entry_time"].dt.to_period("M")
    df["equity_before"] = _compute_equity_before(df)
    rows = []
    for month, month_trades in df.groupby("year_month"):
        month_trades = month_trades.reset_index(drop=True)
        pnl = float(month_trades["pnl"].sum())
        pnl_pct = (pnl / start_equity) * 100 if start_equity else 0.0
        if "margin_per_trade" in month_trades.columns:
            margin = pd.to_numeric(month_trades["margin_per_trade"], errors="coerce")
            returns = month_trades["pnl"] / margin.replace(0, pd.NA)
        else:
            returns = month_trades["pnl"] / month_trades["equity_before"].replace(0, pd.NA)
        rows.append(
            {
                "month": str(month),
                "num_trades": int(len(month_trades)),
                "start_balance": float(month_trades["equity_before"].iloc[0]),
                "end_balance": float(month_trades["equity"].iloc[-1]),
                "pnl": pnl,
                "pnl_pct": pnl_pct,
                "volatility": float(returns.dropna().std(ddof=0)) if returns.dropna().size else 0.0,
            }
        )
    return pd.DataFrame(rows)


def calculate_losing_streaks(trades_df: pd.DataFrame) -> tuple[int, float, float, float]:
    if trades_df.empty:
        return 0, 0.0, 0.0, 0.0
    df = trades_df.copy()
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True, errors="coerce")
    df = df[df["entry_time"].notna()].sort_values("entry_time").reset_index(drop=True)
    if df.empty:
        return 0, 0.0, 0.0, 0.0
    df["is_loss"] = df["pnl"] < 0

    max_streak = 0
    cur = 0
    s_idx = None
    e_idx = None
    for idx, is_loss in enumerate(df["is_loss"]):
        if is_loss:
            cur += 1
            if cur > max_streak:
                max_streak = cur
                e_idx = idx
                s_idx = idx - cur + 1
        else:
            cur = 0
    streak_value = float(df.iloc[s_idx : e_idx + 1]["pnl"].sum()) if s_idx is not None else 0.0

    df["year_month"] = df["entry_time"].dt.to_period("M")
    monthly_streaks = []
    monthly_values = []
    for _, month_df in df.groupby("year_month"):
        month_df = month_df.reset_index(drop=True)
        m_max = 0
        m_val = 0.0
        m_cur = 0
        m_start = None
        for idx, is_loss in enumerate(month_df["is_loss"]):
            if is_loss:
                if m_cur == 0:
                    m_start = idx
                m_cur += 1
            else:
                if m_cur > m_max and m_start is not None:
                    m_max = m_cur
                    m_val = float(month_df.iloc[m_start:idx]["pnl"].sum())
                m_cur = 0
                m_start = None
        if m_cur > m_max and m_start is not None:
            m_max = m_cur
            m_val = float(month_df.iloc[m_start:]["pnl"].sum())
        if m_max > 0:
            monthly_streaks.append(m_max)
            monthly_values.append(m_val)

    avg_streak = float(sum(monthly_streaks) / len(monthly_streaks)) if monthly_streaks else 0.0
    avg_streak_value = float(sum(monthly_values) / len(monthly_values)) if monthly_values else 0.0
    return max_streak, avg_streak, streak_value, avg_streak_value


def calculate_avg_max_monthly_drawdown(trades_df: pd.DataFrame, initial_balance: float) -> float:
    if trades_df.empty or "entry_time" not in trades_df.columns:
        return 0.0
    df = trades_df.copy()
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True, errors="coerce")
    df = df[df["entry_time"].notna()].sort_values("entry_time").reset_index(drop=True)
    if df.empty:
        return 0.0
    df["balance"] = initial_balance + df["pnl"].cumsum()
    df["year_month"] = df["entry_time"].dt.to_period("M")
    monthly_dd = []
    for _, mdf in df.groupby("year_month"):
        balances = mdf["balance"].to_numpy()
        peak = balances[0]
        max_dd = 0.0
        for bal in balances:
            if bal > peak:
                peak = bal
            dd = peak - bal
            if dd > max_dd:
                max_dd = dd
        monthly_dd.append(max_dd)
    return float(sum(monthly_dd) / len(monthly_dd)) if monthly_dd else 0.0


def compute_trade_drawdowns(trades_df: pd.DataFrame, price_df: pd.DataFrame) -> pd.DataFrame:
    if trades_df.empty or price_df.empty:
        return pd.DataFrame()
    size_col = "size" if "size" in trades_df.columns else "order_size"
    required = {"entry_time", "exit_time", "entry_price", "side", size_col}
    if not required.issubset(trades_df.columns):
        return pd.DataFrame()

    rows = []
    for idx, trade in trades_df.iterrows():
        try:
            entry_time = pd.to_datetime(trade["entry_time"], utc=True, errors="coerce")
            exit_time = pd.to_datetime(trade["exit_time"], utc=True, errors="coerce")
            entry_price = float(trade["entry_price"])
            size = float(trade[size_col])
        except (TypeError, ValueError):
            continue
        if pd.isna(entry_time) or pd.isna(exit_time):
            continue
        if exit_time < entry_time:
            entry_time, exit_time = exit_time, entry_time
        slice_df = price_df[
            (price_df["timestamp"] >= entry_time)
            & (price_df["timestamp"] <= exit_time)
        ]
        if slice_df.empty:
            continue
        side = str(trade["side"]).upper()
        adverse_move = (
            float(slice_df["low"].min()) - entry_price
            if side == "BUY"
            else entry_price - float(slice_df["high"].max())
        )
        rows.append(
            {
                "trade_index": idx,
                "entry_time": entry_time,
                "exit_time": exit_time,
                "min_adverse_move": adverse_move,
                "min_unrealized_pnl": adverse_move * size,
            }
        )
    return pd.DataFrame(rows)


def _plot_equity_vs_btc(equity_df: pd.DataFrame, price_df: pd.DataFrame, output_path: Path, start_equity: float) -> None:
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
    buy_hold = start_equity * (price_df["close"] / initial_price)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(equity_df["timestamp"], equity_df["equity"], label="Strategy Equity", linewidth=1.5)
    ax.plot(price_df["timestamp"], buy_hold, label="Buy & Hold BTC", linewidth=1.2)
    ax.set_title("Equity vs BTC Hold")
    ax.set_xlabel("Time")
    ax.set_ylabel("Equity")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def evaluate_backtest(
    trades_csv: Path,
    price_csv: Path,
    output_metrics_csv: Path | None = None,
    output_monthly_csv: Path | None = None,
    output_drawdowns_csv: Path | None = None,
    output_plot: Path | None = None,
    start_equity: float = DEFAULT_START_EQUITY,
    tick_size: float = 0.1,
    tick_value: float = 1.0,
    night_start_hour: int = 22,
    night_end_hour: int = 0,
) -> dict:
    trades_df = _load_trades(trades_csv)
    if trades_df.empty:
        print("⚠️ No trades found.")
        return {}

    trades_df["entry_time"] = pd.to_datetime(trades_df["entry_time"], utc=True, errors="coerce")
    trades_df["exit_time"] = pd.to_datetime(trades_df["exit_time"], utc=True, errors="coerce")
    trades_df = trades_df[trades_df["entry_time"].notna()].sort_values("entry_time").reset_index(drop=True)
    if trades_df.empty:
        print("⚠️ No valid trade timestamps found.")
        return {}

    start_dt = trades_df["entry_time"].iloc[0]
    end_dt = trades_df["exit_time"].dropna().iloc[-1] if trades_df["exit_time"].notna().any() else trades_df["entry_time"].iloc[-1]
    days_of_backtest = (end_dt - start_dt).total_seconds() / 86400

    price_df = _load_price_data(price_csv)
    if not price_df.empty:
        price_df = price_df[(price_df["timestamp"] >= start_dt) & (price_df["timestamp"] <= end_dt)]

    avg_trade = float(trades_df["pnl"].mean())
    avg_win = float(trades_df.loc[trades_df["pnl"] > 0, "pnl"].mean()) if (trades_df["pnl"] > 0).any() else 0.0
    avg_loss = float(trades_df.loc[trades_df["pnl"] < 0, "pnl"].mean()) if (trades_df["pnl"] < 0).any() else 0.0
    trade_sharpe = _compute_sharpe(trades_df)

    equity_curve = _build_equity_curve_with_price(trades_df, price_df) if not price_df.empty else pd.DataFrame()
    price_sharpe = _compute_price_sharpe(price_df) if not price_df.empty else 0.0
    equity_sharpe = _compute_equity_sharpe(equity_curve)
    max_drawdown = _compute_max_drawdown_from_equity(equity_curve) if not equity_curve.empty else 0.0
    eq_streak_bars, eq_streak_value = _compute_equity_losing_streak(equity_curve)

    equity_before = _compute_equity_before(trades_df)
    initial_balance = float(equity_before.iloc[0]) if len(equity_before) else start_equity

    max_losing_streak, avg_monthly_losing_streak, streak_value, avg_monthly_streak_value = calculate_losing_streaks(trades_df)
    avg_max_monthly_dd = calculate_avg_max_monthly_drawdown(trades_df, initial_balance)
    monthly_df = _compute_monthly_stats(trades_df, start_equity)

    drawdowns_df = compute_trade_drawdowns(trades_df, price_df) if not price_df.empty else pd.DataFrame()
    avg_trade_dd = float(drawdowns_df["min_unrealized_pnl"].mean()) if not drawdowns_df.empty else 0.0
    max_trade_dd = float(drawdowns_df["min_unrealized_pnl"].min()) if not drawdowns_df.empty else 0.0

    # Overnight close scenario (carried from compute_trade_drawdown)
    def _first_night_start(ts: pd.Timestamp) -> pd.Timestamp:
        start = ts.normalize() + pd.Timedelta(hours=night_start_hour)
        if ts.time() >= start.time():
            start = start + pd.Timedelta(days=1)
        return start

    window_hours = (24 + night_end_hour - night_start_hour) % 24
    if window_hours == 0:
        window_hours = 2
    total_old = float(trades_df["pnl"].sum())
    total_new = 0.0
    changed = 0
    size_col = "size" if "size" in trades_df.columns else "order_size"
    if not price_df.empty and size_col in trades_df.columns:
        for _, trade in trades_df.iterrows():
            entry_time = trade["entry_time"]
            exit_time = trade["exit_time"]
            close_time = _first_night_start(entry_time)
            window_end = close_time + pd.Timedelta(hours=window_hours)
            side = str(trade.get("side", "")).upper()
            size = float(trade.get(size_col, 0.0) or 0.0)
            fees = float(trade.get("fees", trade.get("total_fees", 0.0)) or 0.0)
            entry_price = float(trade.get("entry_price", 0.0) or 0.0)
            if size <= 0 or entry_price <= 0 or side not in ("BUY", "SELL"):
                total_new += float(trade.get("pnl", 0.0) or 0.0)
                continue
            if pd.notna(exit_time) and entry_time < window_end and exit_time > close_time:
                price_row = price_df[price_df["timestamp"] >= close_time].head(1)
                if price_row.empty:
                    total_new += float(trade.get("pnl", 0.0) or 0.0)
                    continue
                exit_price = float(price_row["close"].iloc[0])
                ticks = (exit_price - entry_price) / tick_size
                if side == "SELL":
                    ticks = -ticks
                total_new += ticks * tick_value * size - fees
                changed += 1
            else:
                total_new += float(trade.get("pnl", 0.0) or 0.0)
    else:
        total_new = total_old

    summary = {
        "days_of_backtest": round(days_of_backtest, 2),
        "num_trades": int(len(trades_df)),
        "average_trade": round(avg_trade, 6),
        "average_winning_trade": round(avg_win, 6),
        "average_losing_trade": round(avg_loss, 6),
        "sharpe": round(equity_sharpe, 6),
        "trade_sharpe": round(trade_sharpe, 6),
        "price_sharpe": round(price_sharpe, 6),
        "max_drawdown": round(max_drawdown, 6),
        "equity_losing_streak_bars": eq_streak_bars,
        "equity_losing_streak_value": round(eq_streak_value, 6),
        "most_losing_trades_in_row": max_losing_streak,
        "avg_most_losing_trades_per_month": round(avg_monthly_losing_streak, 2),
        "dollar_value_most_losses_in_row": round(streak_value, 6),
        "dollar_value_avg_most_losses_per_month": round(avg_monthly_streak_value, 6),
        "avg_max_monthly_drawdown": round(avg_max_monthly_dd, 6),
        "avg_trade_drawdown": round(avg_trade_dd, 6),
        "max_trade_drawdown": round(max_trade_dd, 6),
        "overnight_trades_adjusted": int(changed),
        "total_pnl_original": round(total_old, 6),
        "total_pnl_night_close": round(total_new, 6),
    }

    if output_metrics_csv is not None:
        output_metrics_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([summary]).to_csv(output_metrics_csv, index=False)
    if output_monthly_csv is not None and not monthly_df.empty:
        output_monthly_csv.parent.mkdir(parents=True, exist_ok=True)
        monthly_df.to_csv(output_monthly_csv, index=False)
    if output_drawdowns_csv is not None and not drawdowns_df.empty:
        output_drawdowns_csv.parent.mkdir(parents=True, exist_ok=True)
        drawdowns_df.to_csv(output_drawdowns_csv, index=False)
    if output_plot is not None and not equity_curve.empty and not price_df.empty:
        _plot_equity_vs_btc(equity_curve, price_df, output_plot, start_equity)

    print("\n📊 Backtest Evaluation")
    for k, v in summary.items():
        print(f"{k}: {v}")
    if output_metrics_csv is not None:
        print(f"\n💾 Saved metrics: {output_metrics_csv}")
    if output_monthly_csv is not None and not monthly_df.empty:
        print(f"💾 Saved monthly: {output_monthly_csv}")
    if output_drawdowns_csv is not None and not drawdowns_df.empty:
        print(f"💾 Saved drawdowns: {output_drawdowns_csv}")
    if output_plot is not None and output_plot.exists():
        print(f"💾 Saved plot: {output_plot}")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate backtest trades and export summary metrics.")
    parser.add_argument("--trades", default=str(DEFAULT_TRADES), help="Path to backtest_trades.csv")
    parser.add_argument("--price", default=str(DEFAULT_PRICE), help="Path to price CSV")
    parser.add_argument("--output-metrics", default=None, help="Path to output metrics CSV")
    parser.add_argument("--output-monthly", default=None, help="Path to output monthly CSV")
    parser.add_argument("--output-drawdowns", default=None, help="Path to output drawdowns CSV")
    parser.add_argument("--output-plot", default=None, help="Path to output equity plot PNG")
    parser.add_argument("--start-equity", type=float, default=DEFAULT_START_EQUITY)
    parser.add_argument("--tick-size", type=float, default=0.1)
    parser.add_argument("--tick-value", type=float, default=1.0)
    parser.add_argument("--night-start-hour", type=int, default=22)
    parser.add_argument("--night-end-hour", type=int, default=0)
    args = parser.parse_args()

    trades_path = Path(args.trades).resolve()
    output_dir = trades_path.parent
    output_metrics = Path(args.output_metrics).resolve() if args.output_metrics else (output_dir / "backtest_trades_metrics.csv")
    output_monthly = Path(args.output_monthly).resolve() if args.output_monthly else (output_dir / "backtest_trades_monthly_pnl.csv")
    output_drawdowns = Path(args.output_drawdowns).resolve() if args.output_drawdowns else (output_dir / "backtest_trade_drawdowns.csv")
    output_plot = Path(args.output_plot).resolve() if args.output_plot else (output_dir / "equity_vs_btc.png")

    evaluate_backtest(
        trades_csv=trades_path,
        price_csv=Path(args.price).resolve(),
        output_metrics_csv=output_metrics,
        output_monthly_csv=output_monthly,
        output_drawdowns_csv=output_drawdowns,
        output_plot=output_plot,
        start_equity=args.start_equity,
        tick_size=args.tick_size,
        tick_value=args.tick_value,
        night_start_hour=args.night_start_hour,
        night_end_hour=args.night_end_hour,
    )


if __name__ == "__main__":
    main()
