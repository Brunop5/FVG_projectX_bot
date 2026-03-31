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


def _resolve_price_csv(runtime_dir: Path) -> Path | None:
    local_price = runtime_dir / "backtest_data.csv"
    if local_price.exists():
        return local_price

    metadata_path = runtime_dir / "backtest_metadata.json"
    if metadata_path.exists():
        try:
            with metadata_path.open("r", encoding="utf-8") as f:
                metadata = json.load(f)
            data_path = metadata.get("data_path")
            if data_path:
                candidate = Path(str(data_path))
                if candidate.exists():
                    return candidate
        except Exception:
            return None
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

    start_dt = entry_ts.min()
    end_dt = exit_ts.max() if exit_ts.notna().any() else entry_ts.max()
    backtest_days = (
        float((end_dt - start_dt).total_seconds() / 86400.0)
        if (pd.notna(start_dt) and pd.notna(end_dt))
        else _safe_float(base_summary.get("days_of_backtest"))
    )
    trades_per_day = (num_trades / backtest_days) if backtest_days > 0 else 0.0

    pnl_by_month = pnl.groupby(entry_ts.dt.to_period("M")).sum()
    avg_month_pnl = float(pnl_by_month.mean()) if not pnl_by_month.empty else 0.0
    best_month_pnl = float(pnl_by_month.max()) if not pnl_by_month.empty else 0.0
    worst_month_pnl = float(pnl_by_month.min()) if not pnl_by_month.empty else 0.0

    pnl_by_week = pnl.groupby(entry_ts.dt.to_period("W")).sum()
    avg_week_pnl = float(pnl_by_week.mean()) if not pnl_by_week.empty else 0.0
    best_week_pnl = float(pnl_by_week.max()) if not pnl_by_week.empty else 0.0
    worst_week_pnl = float(pnl_by_week.min()) if not pnl_by_week.empty else 0.0

    signs = [1 if x > 0 else (-1 if x < 0 else 0) for x in pnl.tolist()]
    max_win_streak, max_loss_streak = _compute_streaks(signs)

    max_dd = _safe_float(base_summary.get("max_drawdown"))
    recovery_factor = (net_profit / max_dd) if max_dd > 1e-12 else math.inf

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

    price_csv = _resolve_price_csv(runtime_dir)
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
