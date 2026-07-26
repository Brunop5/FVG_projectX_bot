#!/usr/bin/env python3
"""
Run a single gold (Topstep/MGC) backtest.

Examples (from trading_bots/):

  # Use on-disk MGCQ6 data + live inputs
  python -m FVG_projectX_bot.backtest.run_gold_backtest \\
      --inputs FVG_projectX_bot/projectX/inputs.json

  # Refresh 15m/1m through now, then backtest (live comparison mode)
  python -m FVG_projectX_bot.backtest.run_gold_backtest \\
      --inputs FVG_projectX_bot/projectX/inputs.json \\
      --refresh-data

  # Trial folder inputs
  python -m FVG_projectX_bot.backtest.run_gold_backtest \\
      --inputs FVG_projectX_bot/backtest/results/gold_opt_results/test_3_trial_1034/inputs_used.json \\
      --refresh-data \\
      --out-dir FVG_projectX_bot/backtest/results/gold_manual_t1034
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = SCRIPT_DIR.parent
WORKSPACE_DIR = PACKAGE_DIR.parent

DEFAULT_DATA_DIR = SCRIPT_DIR / "data" / "MGCQ6"
DEFAULT_INPUTS = PACKAGE_DIR / "projectX" / "inputs.json"
DEFAULT_OUT_ROOT = SCRIPT_DIR / "results" / "gold_manual"

ASSET = "MGCQ6"
TIMEFRAME = "15m"
INITIAL_BALANCE = 50_000.0
TICK_SIZE = 0.1
TICK_VALUE = 1.0


def _load_inputs(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Inputs must be a JSON object: {path}")
    return data


def _ensure_trades_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "side",
        "entry_price",
        "exit_price",
        "entry_time",
        "exit_time",
        "order_size",
        "pnl",
        "equity",
        "group_id",
        "margin_per_trade",
        "lot_size",
        "total_fees",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a single MGC gold backtest.")
    parser.add_argument(
        "--inputs",
        type=str,
        default=str(DEFAULT_INPUTS),
        help="Strategy inputs JSON (default: projectX/inputs.json).",
    )
    parser.add_argument(
        "--refresh-data",
        action="store_true",
        help="Chunk-fetch latest 15m/1m Topstep data through now before running.",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=70,
        help="History window for --refresh-data (default: 70).",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(DEFAULT_DATA_DIR),
        help="Directory containing topstep_15min.csv / topstep_1min.csv.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output folder (default: backtest/results/gold_manual/<timestamp>).",
    )
    parser.add_argument(
        "--initial-balance",
        type=float,
        default=INITIAL_BALANCE,
        help="Starting equity (default: 50000).",
    )
    parser.add_argument(
        "--no-overlap-restrict",
        action="store_true",
        help="Do not restrict to 15m/1m overlap (not recommended for intracandle).",
    )
    args = parser.parse_args()

    inputs_path = Path(args.inputs).resolve()
    data_dir = Path(args.data_dir).resolve()
    if args.out_dir:
        out_dir = Path(args.out_dir).resolve()
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_dir = (DEFAULT_OUT_ROOT / stamp).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.refresh_data:
        from FVG_projectX_bot.helping_functions.fetch_topstep_data import refresh_mgc_data

        path15, path1 = refresh_mgc_data(lookback_days=args.lookback_days, out_dir=data_dir)
    else:
        path15 = data_dir / "topstep_15min.csv"
        path1 = data_dir / "topstep_1min.csv"
        if not path15.exists() or not path1.exists():
            raise FileNotFoundError(
                f"Missing data under {data_dir}. Re-run with --refresh-data "
                f"or pass --data-dir to existing CSVs."
            )

    from FVG_projectX_bot.helping_functions.gold_dataset_archive import archive_gold_topstep_csvs

    archive_gold_topstep_csvs(path15, path1, contract=ASSET)

    base_inputs = _load_inputs(inputs_path)
    trial_inputs = dict(base_inputs)
    # Drop live-only wiring; keep strategy knobs.
    trial_inputs.pop("APIS", None)
    trial_inputs["RUNTIME_SUBDIR"] = str(out_dir)
    used_inputs_path = out_dir / "inputs_used.json"
    used_inputs_path.write_text(json.dumps(trial_inputs, indent=2), encoding="utf-8")
    shutil.copy2(used_inputs_path, out_dir / "inputs_source_copy.json")

    os.environ["FVG_INPUTS_JSON"] = str(used_inputs_path)
    os.environ.pop("FVG_NO_DISK_IO", None)
    os.environ["FVG_BACKTEST_ASSET"] = ASSET

    from FVG_projectX_bot.backtest import FVG_backtest as bt
    from FVG_projectX_bot.backtest.evaluate_backtest import evaluate_backtest

    bt.configure_futures_backtest(contracts_path=(SCRIPT_DIR / "contracts.csv").resolve())
    bt.ASSET = ASSET
    bt.TIMEFRAME = TIMEFRAME
    bt.INITIAL_BALANCE = float(args.initial_balance)
    bt.DATA_CSV_PATH = str(path15)
    bt.DATA_1M_CSV_PATH = str(path1)
    bt.USE_LAST_QUARTER_DATA = False

    backtest = bt.FVG_Backtest(
        asset=bt.ASSET,
        timeframe=bt.TIMEFRAME,
        initial_balance=bt.INITIAL_BALANCE,
        data_path=bt.DATA_CSV_PATH,
        start_timestamp=None,
        pyramiding_mode=bt.PYRAMIDING_MODE,
        data_path_1m=bt.DATA_1M_CSV_PATH,
    )

    if not args.no_overlap_restrict:
        if not backtest.restrict_to_1m_overlap():
            raise RuntimeError(
                "No 15m/1m overlap — cannot run intracandle-valid backtest. "
                f"Check {path15} and {path1}."
            )
        overlap_meta = getattr(backtest, "_overlap_meta", {}) or {}
        overlap_meta["restricted_to_1m_overlap"] = True
        if overlap_meta.get("eval_start_ms") and overlap_meta.get("eval_end_ms"):
            overlap_meta["eval_start_utc"] = datetime.fromtimestamp(
                overlap_meta["eval_start_ms"] / 1000.0, tz=timezone.utc
            ).isoformat()
            overlap_meta["eval_end_utc"] = datetime.fromtimestamp(
                overlap_meta["eval_end_ms"] / 1000.0, tz=timezone.utc
            ).isoformat()
        (out_dir / "overlap_window.json").write_text(
            json.dumps(overlap_meta, indent=2), encoding="utf-8"
        )
        backtest._full_data.to_csv(out_dir / "backtest_price_15m_overlap.csv", index=False)

    _ensure_trades_csv(Path(backtest.trades_csv_path))
    print(f"Running backtest -> {out_dir}", flush=True)
    trades = backtest.run()
    print(f"✅ Finished. Trades: {len(trades)}", flush=True)

    # Move/copy trades file into out_dir if written elsewhere
    trades_csv = Path(backtest.trades_csv_path)
    if trades_csv.resolve().parent != out_dir:
        dest_trades = out_dir / "backtest_trades.csv"
        if trades_csv.exists():
            shutil.copy2(trades_csv, dest_trades)
        trades_csv = dest_trades

    summary = evaluate_backtest(
        trades_csv=trades_csv,
        price_df=getattr(backtest, "_full_data", None),
        output_metrics_csv=out_dir / "backtest_trades_metrics.csv",
        output_monthly_csv=out_dir / "backtest_trades_monthly_pnl.csv",
        output_drawdowns_csv=out_dir / "backtest_trade_drawdowns.csv",
        output_plot=out_dir / "equity_vs_gold.png",
        start_equity=bt.INITIAL_BALANCE,
        tick_size=TICK_SIZE,
        tick_value=TICK_VALUE,
        benchmark_label="Buy & Hold Gold",
    )
    print(
        f"PnL={summary.get('total_pnl_original')}  "
        f"DD={summary.get('max_drawdown')}  "
        f"trades={summary.get('num_trades')}  "
        f"days={summary.get('days_of_backtest')}"
    )
    print(f"Outputs: {out_dir}")


if __name__ == "__main__":
    main()
