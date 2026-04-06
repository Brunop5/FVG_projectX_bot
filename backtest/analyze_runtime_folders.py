#!/usr/bin/env python3
"""
Batch-analyze runtime/backtest result folders.

Given a root directory, this script scans nested folders and processes each folder
that contains `backtest_trades.csv`. For each detected runtime folder it:

- Generates equity plot PNG in that same folder.
- Runs base evaluation metrics.
- Computes extended metrics (monthly/weekly PnL stats, win/loss stats, drawdown stats, etc.).
- Writes per-folder metrics CSV.

It also writes one aggregate CSV at the selected root.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from FVG_projectX_bot.backtest.evaluate_backtest import evaluate_backtest


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _find_runtime_dirs(root_dir: Path) -> list[Path]:
    return sorted({p.parent for p in root_dir.rglob("backtest_trades.csv")})


def _extract_trades_time_bounds(trades_csv: Path) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    if not trades_csv.exists():
        return None
    try:
        trades_df = pd.read_csv(trades_csv, usecols=["entry_time", "exit_time"])
    except Exception:
        return None
    if trades_df.empty:
        return None
    entry_ts = pd.to_datetime(trades_df["entry_time"], utc=True, errors="coerce")
    exit_ts = pd.to_datetime(trades_df["exit_time"], utc=True, errors="coerce")
    start_dt = entry_ts.min()
    end_dt = exit_ts.max() if exit_ts.notna().any() else entry_ts.max()
    if pd.isna(start_dt) or pd.isna(end_dt):
        return None
    return start_dt, end_dt


def _extract_price_time_bounds(price_csv: Path) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    if not price_csv.exists():
        return None
    try:
        price_df = pd.read_csv(price_csv)
    except Exception:
        return None
    if price_df.empty:
        return None
    if "timestamp" not in price_df.columns:
        if "<DATE>" in price_df.columns and "<TIME>" in price_df.columns:
            ts = pd.to_datetime(
                price_df["<DATE>"].astype(str) + " " + price_df["<TIME>"].astype(str),
                utc=True,
                errors="coerce",
            )
        else:
            return None
    else:
        ts_raw = price_df["timestamp"]
        if pd.api.types.is_numeric_dtype(ts_raw):
            max_val = pd.to_numeric(ts_raw, errors="coerce").max()
            unit = "ms" if pd.notna(max_val) and float(max_val) > 10**12 else "s"
            ts = pd.to_datetime(ts_raw, unit=unit, utc=True, errors="coerce")
        else:
            ts_num = pd.to_numeric(ts_raw, errors="coerce")
            if ts_num.notna().any():
                max_val = ts_num.max()
                unit = "ms" if pd.notna(max_val) and float(max_val) > 10**12 else "s"
                ts = pd.to_datetime(ts_num, unit=unit, utc=True, errors="coerce")
            else:
                ts = pd.to_datetime(ts_raw, utc=True, errors="coerce")
    ts = ts[ts.notna()]
    if ts.empty:
        return None
    return ts.min(), ts.max()


def _price_covers_trade_window(price_csv: Path, trades_csv: Path) -> bool:
    trade_bounds = _extract_trades_time_bounds(trades_csv)
    if trade_bounds is None:
        return True
    price_bounds = _extract_price_time_bounds(price_csv)
    if price_bounds is None:
        return False
    trade_start, trade_end = trade_bounds
    price_start, price_end = price_bounds
    return bool(price_start <= trade_start and price_end >= trade_end)


def _resolve_price_csv(runtime_dir: Path, trades_csv: Path) -> Path | None:
    local_price = runtime_dir / "backtest_data.csv"
    local_ok = local_price.exists() and _price_covers_trade_window(local_price, trades_csv)
    if local_ok:
        return local_price

    metadata_path = runtime_dir / "backtest_metadata.json"
    if metadata_path.exists():
        try:
            with metadata_path.open("r", encoding="utf-8") as f:
                metadata = json.load(f)
            data_path = metadata.get("data_path")
            if data_path:
                candidate = Path(str(data_path))
                if candidate.exists() and _price_covers_trade_window(candidate, trades_csv):
                    return candidate
        except Exception:
            pass
    # Fallback order when full coverage is unavailable.
    if local_price.exists():
        return local_price
    return None


def _compute_streaks(signs: list[int]) -> tuple[int, int]:
    max_win = 0
    max_loss = 0
    cur_win = 0
    cur_loss = 0
    for s in signs:
        if s > 0:
            cur_win += 1
            cur_loss = 0
        elif s < 0:
            cur_loss += 1
            cur_win = 0
        else:
            cur_win = 0
            cur_loss = 0
        max_win = max(max_win, cur_win)
        max_loss = max(max_loss, cur_loss)
    return max_win, max_loss


def _compute_drawdown_episode_durations(
    exit_ts: pd.Series,
    equity_series: pd.Series,
) -> tuple[float, float]:
    """
    Returns:
      - duration_seconds of the biggest drawdown episode (by depth)
      - duration_seconds of the longest drawdown episode (by time)
    """
    if len(exit_ts) == 0 or len(equity_series) == 0:
        return 0.0, 0.0

    df = pd.DataFrame({"ts": exit_ts, "equity": equity_series})
    df = df[df["ts"].notna() & df["equity"].notna()].sort_values("ts").reset_index(drop=True)
    if df.empty:
        return 0.0, 0.0

    peak_val = float(df["equity"].iloc[0])
    peak_time = df["ts"].iloc[0]

    in_dd = False
    dd_start_time = None
    dd_max_depth = 0.0
    episodes: list[dict[str, float]] = []

    for row in df.itertuples(index=False):
        ts = row.ts
        eq = float(row.equity)

        if eq >= peak_val:
            if in_dd and dd_start_time is not None:
                dur = max(0.0, float((ts - dd_start_time).total_seconds()))
                episodes.append({"depth": dd_max_depth, "duration_sec": dur})
                in_dd = False
                dd_start_time = None
                dd_max_depth = 0.0
            peak_val = eq
            peak_time = ts
            continue

        # eq < peak_val -> drawdown
        depth = peak_val - eq
        if not in_dd:
            in_dd = True
            dd_start_time = peak_time if peak_time is not None else ts
            dd_max_depth = depth
        else:
            dd_max_depth = max(dd_max_depth, depth)

    if in_dd and dd_start_time is not None:
        end_ts = df["ts"].iloc[-1]
        dur = max(0.0, float((end_ts - dd_start_time).total_seconds()))
        episodes.append({"depth": dd_max_depth, "duration_sec": dur})

    if not episodes:
        return 0.0, 0.0

    biggest_by_depth = max(episodes, key=lambda e: e["depth"])
    longest_by_time = max(episodes, key=lambda e: e["duration_sec"])
    return float(biggest_by_depth["duration_sec"]), float(longest_by_time["duration_sec"])


def _extended_metrics(trades_df: pd.DataFrame, base_summary: dict[str, Any]) -> dict[str, Any]:
    if trades_df.empty:
        return {}

    pnl = pd.to_numeric(trades_df["pnl"], errors="coerce").dropna()
    if pnl.empty:
        return {}

    entry_ts = pd.to_datetime(trades_df["entry_time"], utc=True, errors="coerce")
    exit_ts = pd.to_datetime(trades_df["exit_time"], utc=True, errors="coerce")
    valid_mask = entry_ts.notna()
    entry_ts = entry_ts[valid_mask]
    exit_ts = exit_ts[valid_mask]
    pnl = pnl.loc[valid_mask]

    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    num_wins = int((pnl > 0).sum())
    num_losses = int((pnl < 0).sum())
    num_trades = int(len(pnl))

    gross_profit = float(wins.sum()) if not wins.empty else 0.0
    gross_loss = float(losses.sum()) if not losses.empty else 0.0
    net_profit = float(pnl.sum())

    profit_factor = (
        (gross_profit / abs(gross_loss))
        if abs(gross_loss) > 1e-12
        else (math.inf if gross_profit > 0 else 0.0)
    )

    avg_trade = float(pnl.mean())
    median_trade = float(pnl.median())
    std_trade = float(pnl.std(ddof=0)) if len(pnl) > 1 else 0.0
    largest_win = float(pnl.max())
    largest_loss = float(pnl.min())

    win_rate = (num_wins / num_trades) * 100.0 if num_trades > 0 else 0.0
    avg_win = float(wins.mean()) if not wins.empty else 0.0
    avg_loss = float(losses.mean()) if not losses.empty else 0.0
    avg_win_loss_ratio = (avg_win / abs(avg_loss)) if abs(avg_loss) > 1e-12 else math.inf

    expectancy = (win_rate / 100.0) * avg_win + (1.0 - win_rate / 100.0) * avg_loss

    trade_durations = (exit_ts - entry_ts).dt.total_seconds()
    trade_durations = trade_durations[trade_durations.notna() & (trade_durations >= 0)]
    avg_trade_duration_sec = float(trade_durations.mean()) if not trade_durations.empty else 0.0

    start_dt = entry_ts.min()
    end_dt = exit_ts.max() if exit_ts.notna().any() else entry_ts.max()
    backtest_days = (
        float((end_dt - start_dt).total_seconds() / 86400.0)
        if (pd.notna(start_dt) and pd.notna(end_dt))
        else _safe_float(base_summary.get("days_of_backtest"))
    )
    trades_per_day = (num_trades / backtest_days) if backtest_days > 0 else 0.0

    entry_ts_naive = entry_ts.dt.tz_localize(None)
    pnl_by_month = pnl.groupby(entry_ts_naive.dt.to_period("M")).sum()
    avg_month_pnl = float(pnl_by_month.mean()) if not pnl_by_month.empty else 0.0
    best_month_pnl = float(pnl_by_month.max()) if not pnl_by_month.empty else 0.0
    worst_month_pnl = float(pnl_by_month.min()) if not pnl_by_month.empty else 0.0

    pnl_by_week = pnl.groupby(entry_ts_naive.dt.to_period("W")).sum()
    avg_week_pnl = float(pnl_by_week.mean()) if not pnl_by_week.empty else 0.0
    best_week_pnl = float(pnl_by_week.max()) if not pnl_by_week.empty else 0.0
    worst_week_pnl = float(pnl_by_week.min()) if not pnl_by_week.empty else 0.0

    signs = [1 if x > 0 else (-1 if x < 0 else 0) for x in pnl.tolist()]
    max_win_streak, max_loss_streak = _compute_streaks(signs)

    max_dd = _safe_float(base_summary.get("max_drawdown"))
    recovery_factor = (net_profit / max_dd) if max_dd > 1e-12 else math.inf
    equity_series = pd.to_numeric(trades_df.get("equity", pd.Series(dtype=float)), errors="coerce")
    biggest_dd_dur_sec, longest_dd_dur_sec = _compute_drawdown_episode_durations(
        exit_ts=exit_ts,
        equity_series=equity_series,
    )

    total_fees = 0.0
    if "total_fees" in trades_df.columns:
        total_fees = float(pd.to_numeric(trades_df["total_fees"], errors="coerce").fillna(0).sum())
    elif "fees" in trades_df.columns:
        total_fees = float(pd.to_numeric(trades_df["fees"], errors="coerce").fillna(0).sum())

    return {
        "num_trades": num_trades,
        "num_winning_trades": num_wins,
        "num_losing_trades": num_losses,
        "win_rate_pct": win_rate,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "net_profit": net_profit,
        "profit_factor": profit_factor,
        "expectancy_per_trade": expectancy,
        "average_trade": avg_trade,
        "median_trade": median_trade,
        "std_trade": std_trade,
        "average_win": avg_win,
        "average_loss": avg_loss,
        "average_win_loss_ratio": avg_win_loss_ratio,
        "largest_win": largest_win,
        "largest_loss": largest_loss,
        "max_drawdown": max_dd,
        "avg_trade_duration_sec": avg_trade_duration_sec,
        "avg_trade_duration_hours": avg_trade_duration_sec / 3600.0,
        "biggest_drawdown_duration_sec": biggest_dd_dur_sec,
        "biggest_drawdown_duration_hours": biggest_dd_dur_sec / 3600.0,
        "longest_drawdown_duration_sec": longest_dd_dur_sec,
        "longest_drawdown_duration_hours": longest_dd_dur_sec / 3600.0,
        "recovery_factor": recovery_factor,
        "trades_per_day": trades_per_day,
        "backtest_days": backtest_days,
        "avg_month_pnl_usd": avg_month_pnl,
        "best_month_pnl_usd": best_month_pnl,
        "worst_month_pnl_usd": worst_month_pnl,
        "avg_week_pnl_usd": avg_week_pnl,
        "best_week_pnl_usd": best_week_pnl,
        "worst_week_pnl_usd": worst_week_pnl,
        "max_win_streak": max_win_streak,
        "max_loss_streak": max_loss_streak,
        "total_fees": total_fees,
        "equity_sharpe": _safe_float(base_summary.get("sharpe")),
        "trade_sharpe": _safe_float(base_summary.get("trade_sharpe")),
    }


def _analyze_runtime_dir(runtime_dir: Path, start_equity: float) -> dict[str, Any]:
    trades_csv = runtime_dir / "backtest_trades.csv"
    if not trades_csv.exists():
        raise FileNotFoundError(trades_csv)

    price_csv = _resolve_price_csv(runtime_dir, trades_csv)
    if price_csv is None:
        raise FileNotFoundError(f"No price CSV found for {runtime_dir}")

    metrics_csv = runtime_dir / "backtest_trades_metrics.csv"
    monthly_csv = runtime_dir / "backtest_trades_monthly_pnl.csv"
    drawdowns_csv = runtime_dir / "backtest_trade_drawdowns.csv"
    plot_png = runtime_dir / "equity_vs_btc.png"

    base_summary = evaluate_backtest(
        trades_csv=trades_csv,
        price_csv=price_csv,
        output_metrics_csv=metrics_csv,
        output_monthly_csv=monthly_csv,
        output_drawdowns_csv=drawdowns_csv,
        output_plot=plot_png,
        start_equity=start_equity,
    )

    trades_df = pd.read_csv(trades_csv)
    extended = _extended_metrics(trades_df, base_summary)
    row = {
        "runtime_folder": str(runtime_dir),
        "trades_csv": str(trades_csv),
        "price_csv": str(price_csv),
        "equity_plot": str(plot_png),
    }
    row.update(extended)

    per_folder_csv = runtime_dir / "runtime_summary_extended.csv"
    pd.DataFrame([row]).to_csv(per_folder_csv, index=False)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze all runtime folders under a selected root directory."
    )
    parser.add_argument(
        "--root-dir",
        required=True,
        help="Root directory to scan recursively for folders containing backtest_trades.csv",
    )
    parser.add_argument(
        "--start-equity",
        type=float,
        default=50.0,
        help="Default starting equity for evaluate_backtest.",
    )
    args = parser.parse_args()

    root_dir = Path(args.root_dir).resolve()
    if not root_dir.exists():
        raise FileNotFoundError(root_dir)

    runtime_dirs = _find_runtime_dirs(root_dir)
    if not runtime_dirs:
        print(f"No runtime folders found under: {root_dir}")
        return

    aggregate_rows: list[dict[str, Any]] = []
    for runtime_dir in runtime_dirs:
        try:
            print(f"Analyzing: {runtime_dir}")
            row = _analyze_runtime_dir(runtime_dir, start_equity=args.start_equity)
            aggregate_rows.append(row)
        except Exception as exc:
            print(f"⚠️ Skipped {runtime_dir}: {exc}")

    if not aggregate_rows:
        print("No folders were successfully analyzed.")
        return

    aggregate_csv = root_dir / "runtime_folders_summary.csv"
    pd.DataFrame(aggregate_rows).to_csv(aggregate_csv, index=False)
    print(f"\n✅ Aggregate summary saved: {aggregate_csv}")


if __name__ == "__main__":
    main()
