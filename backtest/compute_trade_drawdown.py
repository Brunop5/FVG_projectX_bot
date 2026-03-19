#!/usr/bin/env python3
"""
Compute average and maximum intra-trade drawdown using equity curve data.
"""

import argparse
import os
import pandas as pd


def _load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _load_price_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if len(df.columns) == 1 and "<DATE>" in df.columns[0] and "\t" in df.columns[0]:
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
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    else:
        raise ValueError("Price CSV missing timestamp columns.")
    df = df[df["timestamp"].notna()].sort_values("timestamp").reset_index(drop=True)
    return df


def compute_trade_drawdowns(trades_df: pd.DataFrame, price_df: pd.DataFrame) -> pd.DataFrame:
    required_trade_cols = {"entry_time", "exit_time", "entry_price", "side", "size"}
    if not required_trade_cols.issubset(trades_df.columns):
        missing = required_trade_cols - set(trades_df.columns)
        raise ValueError(f"Trades CSV missing columns: {sorted(missing)}")

    results = []
    for idx, trade in trades_df.iterrows():
        try:
            entry_time = pd.to_datetime(trade["entry_time"], utc=True)
            exit_time = pd.to_datetime(trade["exit_time"], utc=True)
            entry_price = float(trade["entry_price"])
            size = float(trade["size"])
        except (TypeError, ValueError):
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
        if side == "BUY":
            adverse_move = float(slice_df["low"].min()) - entry_price
        else:
            adverse_move = entry_price - float(slice_df["high"].max())
        min_unrealized = adverse_move * size
        results.append(
            {
                "trade_index": idx,
                "entry_time": entry_time,
                "exit_time": exit_time,
                "min_adverse_move": adverse_move,
                "min_unrealized_pnl": min_unrealized,
            }
        )
    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(
        description="Compute average and maximum intra-trade drawdown from equity curve."
    )
    parser.add_argument(
        "--trades",
        default="FVG_projectX_bot/backtest/gold_results/1/backtest_trades_CON.F.US.MGC.G26_20260126.csv",
        help="Trades CSV path (with entry_bar/exit_bar)",
    )
    parser.add_argument(
        "--price",
        default="FVG_projectX_bot/backtest/data/MGCG6/IC_markets_15min.csv",
        help="Price CSV path (with date/time or timestamp)",
    )
    parser.add_argument("--tick-size", type=float, default=0.1)
    parser.add_argument("--tick-value", type=float, default=1.0)
    parser.add_argument("--night-start-hour", type=int, default=22)
    parser.add_argument("--night-end-hour", type=int, default=0)
    args = parser.parse_args()

    trades_df = _load_csv(args.trades)
    price_df = _load_price_data(args.price)
    drawdowns_df = compute_trade_drawdowns(trades_df, price_df)
    if drawdowns_df.empty:
        print("⚠️ No valid trades found for drawdown computation.")
        return

    avg_drawdown = float(drawdowns_df["min_unrealized_pnl"].mean())
    max_drawdown = float(drawdowns_df["min_unrealized_pnl"].min())
    avg_move = float(drawdowns_df["min_adverse_move"].mean())
    max_move = float(drawdowns_df["min_adverse_move"].min())
    print(f"Trades analyzed: {len(drawdowns_df)}")
    print(f"Average trade drawdown (min unrealized): {avg_drawdown:.4f}")
    print(f"Maximum trade drawdown (worst min): {max_drawdown:.4f}")
    print(f"Average adverse move (price): {avg_move:.4f}")
    print(f"Maximum adverse move (price): {max_move:.4f}")

    # Overnight close simulation (22:00 -> 00:00 by default).
    trades_df["entry_time"] = pd.to_datetime(trades_df["entry_time"], utc=True, errors="coerce")
    trades_df["exit_time"] = pd.to_datetime(trades_df["exit_time"], utc=True, errors="coerce")
    trades_df = trades_df[trades_df["entry_time"].notna() & trades_df["exit_time"].notna()].copy()
    if trades_df.empty:
        return
    price_df = price_df.copy()
    price_df = price_df[price_df["timestamp"].notna()].sort_values("timestamp").reset_index(drop=True)

    def _first_night_start(ts: pd.Timestamp) -> pd.Timestamp:
        start = ts.normalize() + pd.Timedelta(hours=args.night_start_hour)
        if ts.time() >= start.time():
            start = start + pd.Timedelta(days=1)
        return start

    window_hours = (24 + args.night_end_hour - args.night_start_hour) % 24
    if window_hours == 0:
        window_hours = 2
    total_old = float(trades_df["pnl"].sum())
    total_new = 0.0
    worst_old = float(trades_df["pnl"].min())
    worst_new = None
    pnl_old_adjusted = []
    pnl_new_adjusted = []
    new_pnls_all = []
    changed = 0
    for _, trade in trades_df.iterrows():
        entry_time = trade["entry_time"]
        exit_time = trade["exit_time"]
        close_time = _first_night_start(entry_time)
        window_end = close_time + pd.Timedelta(hours=window_hours)
        side = str(trade.get("side", "")).upper()
        size = float(trade.get("size", 0.0) or 0.0)
        fees = float(trade.get("fees", 0.0) or 0.0)
        entry_price = float(trade.get("entry_price", 0.0) or 0.0)
        if size <= 0 or entry_price <= 0 or side not in ("BUY", "SELL"):
            total_new += float(trade.get("pnl", 0.0) or 0.0)
            continue
        if entry_time < window_end and exit_time > close_time:
            # Close at first bar on/after close_time.
            price_row = price_df[price_df["timestamp"] >= close_time].head(1)
            if price_row.empty:
                pnl_val = float(trade.get("pnl", 0.0) or 0.0)
                total_new += pnl_val
                pnl_old_adjusted.append(pnl_val)
                pnl_new_adjusted.append(pnl_val)
                new_pnls_all.append(pnl_val)
                continue
            exit_price = float(price_row["close"].iloc[0])
            ticks = (exit_price - entry_price) / args.tick_size
            if side == "SELL":
                ticks = -ticks
            pnl = ticks * args.tick_value * size - fees
            total_new += pnl
            changed += 1
            pnl_old_adjusted.append(float(trade.get("pnl", 0.0) or 0.0))
            pnl_new_adjusted.append(pnl)
            new_pnls_all.append(pnl)
        else:
            pnl_val = float(trade.get("pnl", 0.0) or 0.0)
            total_new += pnl_val
            new_pnls_all.append(pnl_val)

    if new_pnls_all:
        worst_new = float(min(new_pnls_all))
        avg_old_adjusted = float(sum(pnl_old_adjusted) / len(pnl_old_adjusted))
        avg_new_adjusted = float(sum(pnl_new_adjusted) / len(pnl_new_adjusted))
    print(f"Overnight-trades adjusted: {changed}")
    print(f"Total PnL original: {total_old:.4f}")
    print(f"Total PnL with night close: {total_new:.4f}")
    print(f"Worst trade PnL original: {worst_old:.4f}")
    if worst_new is not None:
        print(f"Worst trade PnL adjusted: {worst_new:.4f}")
        print(f"Avg PnL adjusted trades (original): {avg_old_adjusted:.4f}")
        print(f"Avg PnL adjusted trades (new): {avg_new_adjusted:.4f}")


if __name__ == "__main__":
    main()

