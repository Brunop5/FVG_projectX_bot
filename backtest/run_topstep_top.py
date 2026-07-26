#!/usr/bin/env python3
"""
Run full gold (Topstep/MGC) backtests for top optimization candidates.

- Reads top N rows by objective from optimization_results/topstep/optimization_results.csv
- Re-runs each strategy on the 15m/1m overlap window only (where 1m intrabar data exists)
- Uses fixed lot sizing + per-contract round-turn fees (matches live ProjectX)
- Writes each strategy output into its own folder under backtest/results/gold_opt_results/
- Appends summary metrics to backtest/results/gold_opt_results/final_result.csv after each run
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = SCRIPT_DIR.parent
WORKSPACE_DIR = PACKAGE_DIR.parent

TOPSTEP_OPT_CSV = SCRIPT_DIR / "optimization_results" / "topstep" / "optimization_results.csv"
TOPSTEP_INPUTS_PATH = SCRIPT_DIR / "inputs_topstep.json"
OUT_DIR = SCRIPT_DIR / "results" / "gold_opt_results"
RESULT_MARKER = "__TOPSTEP_TOP_RESULT__"

TOPSTEP_ASSET = "MGCQ6"
TOPSTEP_TIMEFRAME = "15m"
TOPSTEP_INITIAL_BALANCE = 50_000.0
TOPSTEP_DATA_15M = SCRIPT_DIR / "data" / "MGCQ6" / "topstep_15min.csv"
TOPSTEP_DATA_1M = SCRIPT_DIR / "data" / "MGCQ6" / "topstep_1min.csv"
TOPSTEP_TICK_SIZE = 0.1
TOPSTEP_TICK_VALUE = 1.0

BAD_OBJECTIVE = -9_999_999.0


def _to_native(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


def _json_safe_dict(d: dict[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for k, v in d.items():
        safe[str(k)] = _to_native(v)
    return safe


def _load_top_candidates(path: Path, top_n: int, skip_first: int = 0) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if df.empty:
        return []
    if "failed" in df.columns:
        failed_col = df["failed"].astype(str).str.lower()
        df = df[~failed_col.isin(["true", "1", "yes"])]
    df["objective"] = pd.to_numeric(df["objective"], errors="coerce")
    df = df[df["objective"].notna()]
    df = df[df["objective"] > BAD_OBJECTIVE + 1.0]
    df = df.sort_values("objective", ascending=False)
    if skip_first > 0:
        df = df.iloc[skip_first:]
    return df.head(top_n).to_dict(orient="records")


def _build_trial_inputs(row: dict[str, Any]) -> dict[str, Any]:
    with TOPSTEP_INPUTS_PATH.open("r", encoding="utf-8") as f:
        base_inputs = json.load(f)
    merged = dict(base_inputs)
    for key in base_inputs.keys():
        if key in row:
            v = _to_native(row[key])
            if v is not None:
                merged[key] = v
    return merged


def _strategy_folder_name(rank: int, row: dict[str, Any]) -> str:
    trial = row.get("trial")
    if trial is not None and not (isinstance(trial, float) and math.isnan(trial)):
        return f"test_{rank}_trial_{int(float(trial))}"
    return f"test_{rank}"


def _ensure_trades_csv_exists(path: Path) -> None:
    """Create a fresh trades CSV (overwrite any previous run in this folder)."""
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
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()


def _compute_objective(pnl_total: float, max_dd: float) -> float:
    if pnl_total <= 0:
        return BAD_OBJECTIVE
    if max_dd <= 0:
        return pnl_total
    return pnl_total / max_dd


def _run_worker(
    row: dict[str, Any],
    rank: int,
    out_root: Path,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="gold_top_") as tmp_dir:
        strategy_dir = out_root / _strategy_folder_name(rank, row)
        strategy_dir.mkdir(parents=True, exist_ok=True)

        trial_inputs = _build_trial_inputs(row)
        trial_inputs["RUNTIME_SUBDIR"] = str(strategy_dir.resolve())

        inputs_path = Path(tmp_dir) / "inputs_worker.json"
        payload_path = Path(tmp_dir) / "row_payload.json"
        with inputs_path.open("w", encoding="utf-8") as f:
            json.dump(trial_inputs, f, indent=2)
        with payload_path.open("w", encoding="utf-8") as f:
            json.dump(_json_safe_dict(row), f)

        shutil.copy2(inputs_path, strategy_dir / "inputs_used.json")

        cmd = [
            sys.executable,
            "-u",
            "-m",
            "FVG_projectX_bot.backtest.run_topstep_top",
            "--worker",
            "--inputs-path",
            str(inputs_path),
            "--row-payload",
            str(payload_path),
            "--strategy-dir",
            str(strategy_dir),
        ]
        proc = subprocess.Popen(
            cmd,
            cwd=str(WORKSPACE_DIR),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )

        out_lines: list[str] = []
        result_payload: dict[str, Any] | None = None
        assert proc.stdout is not None
        for raw in proc.stdout:
            line = raw.rstrip("\n")
            out_lines.append(line)
            if line.startswith(RESULT_MARKER):
                result_payload = json.loads(line[len(RESULT_MARKER):])
                continue
            print(f"[top{rank:02d}] {line}")

        rc = proc.wait()
        if rc != 0:
            raise RuntimeError(f"Worker failed (code={rc}): {' | '.join(out_lines[-12:])}")
        if result_payload is None:
            raise RuntimeError("Worker returned no result payload.")
        return result_payload


def _worker_mode(
    inputs_path: Path,
    row_payload: Path,
    strategy_dir: Path,
) -> None:
    os.environ.pop("FVG_NO_DISK_IO", None)
    os.environ["FVG_INPUTS_JSON"] = str(inputs_path.resolve())

    from FVG_projectX_bot.backtest import FVG_backtest as bt
    from FVG_projectX_bot.backtest.evaluate_backtest import evaluate_backtest

    with row_payload.open("r", encoding="utf-8") as f:
        row = json.load(f)

    bt.configure_futures_backtest(contracts_path=(SCRIPT_DIR / "contracts.csv").resolve())
    bt.ASSET = TOPSTEP_ASSET
    bt.TIMEFRAME = TOPSTEP_TIMEFRAME
    bt.INITIAL_BALANCE = TOPSTEP_INITIAL_BALANCE
    bt.DATA_CSV_PATH = str(TOPSTEP_DATA_15M)
    bt.DATA_1M_CSV_PATH = str(TOPSTEP_DATA_1M)
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
    if not backtest.restrict_to_1m_overlap():
        raise RuntimeError(
            "No 15m/1m overlap in gold data — cannot run intrabar-valid backtest. "
            f"Check {TOPSTEP_DATA_15M} and {TOPSTEP_DATA_1M}."
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
    with (strategy_dir / "overlap_window.json").open("w", encoding="utf-8") as f:
        json.dump(overlap_meta, f, indent=2)

    overlap_price_path = strategy_dir / "backtest_price_15m_overlap.csv"
    backtest._full_data.to_csv(overlap_price_path, index=False)

    _ensure_trades_csv_exists(Path(backtest.trades_csv_path))
    backtest.run()

    metrics_csv = strategy_dir / "backtest_trades_metrics.csv"
    monthly_csv = strategy_dir / "backtest_trades_monthly_pnl.csv"
    drawdowns_csv = strategy_dir / "backtest_trade_drawdowns.csv"
    plot_png = strategy_dir / "equity_vs_gold.png"
    summary = evaluate_backtest(
        trades_csv=Path(backtest.trades_csv_path),
        price_csv=overlap_price_path,
        price_df=backtest._full_data,
        output_metrics_csv=metrics_csv,
        output_monthly_csv=monthly_csv,
        output_drawdowns_csv=drawdowns_csv,
        output_plot=plot_png,
        start_equity=bt.INITIAL_BALANCE,
        tick_size=TOPSTEP_TICK_SIZE,
        tick_value=TOPSTEP_TICK_VALUE,
        benchmark_label="Buy & Hold Gold",
    )

    trades_df = pd.read_csv(backtest.trades_csv_path)
    pnl = pd.to_numeric(trades_df.get("pnl", pd.Series(dtype=float)), errors="coerce").dropna()
    win_count = int((pnl > 0).sum()) if not pnl.empty else 0
    loss_count = int((pnl < 0).sum()) if not pnl.empty else 0

    avg_win = float(summary.get("average_winning_trade", 0.0))
    avg_loss = float(summary.get("average_losing_trade", 0.0))

    pnl_total = float(summary.get("total_pnl_original", 0.0))
    max_dd = float(summary.get("max_drawdown", 0.0))
    days = float(summary.get("days_of_backtest", 0.0) or 0.0)
    num_trades = int(summary.get("num_trades", 0))
    tpd = (num_trades / days) if days > 0 else 0.0
    objective = _compute_objective(pnl_total, max_dd)
    ratio = (pnl_total / max_dd) if max_dd > 0 else None

    result_row = dict(row)
    result_row.update(
        {
            "failed": False,
            "error": "",
            "ratio": float(ratio) if ratio is not None and math.isfinite(ratio) else None,
            "tpd_penalty_factor": 1.0,
            "pnl": pnl_total,
            "max_dd": max_dd,
            "trades_per_day": tpd,
            "trades_total": num_trades,
            "average_win": avg_win,
            "average_loss": avg_loss,
            "win_rate": float((win_count / max(win_count + loss_count, 1)) * 100.0),
            "objective": float(objective),
            "RUNTIME_SUBDIR": str(strategy_dir),
        }
    )
    print(f"{RESULT_MARKER}{json.dumps(result_row)}")


def _append_result_row(final_csv: Path, row: dict[str, Any], fieldnames: list[str]) -> None:
    final_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not final_csv.exists()
    normalized_row = {k: row.get(k) for k in fieldnames}
    with final_csv.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(normalized_row)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run full-data gold (Topstep/MGC) backtests for top optimized strategies."
    )
    parser.add_argument("--top-n", type=int, default=12, help="How many top rows by objective to run.")
    parser.add_argument(
        "--skip-first",
        type=int,
        default=0,
        help="Skip the first N top-ranked rows before selecting top-n.",
    )
    parser.add_argument("--max-workers", type=int, default=2, help="How many backtests to run concurrently.")
    parser.add_argument(
        "--opt-csv",
        type=str,
        default=str(TOPSTEP_OPT_CSV),
        help="Path to topstep optimization_results.csv.",
    )
    parser.add_argument("--out-dir", type=str, default=str(OUT_DIR), help="Output folder for this runner.")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--inputs-path", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--row-payload", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--strategy-dir", type=str, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker:
        if not args.inputs_path or not args.row_payload or not args.strategy_dir:
            raise ValueError("Worker mode requires --inputs-path, --row-payload, and --strategy-dir.")
        _worker_mode(
            Path(args.inputs_path),
            Path(args.row_payload),
            Path(args.strategy_dir),
        )
        return

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    final_csv = out_dir / "final_result.csv"
    opt_source_df = pd.read_csv(Path(args.opt_csv).resolve())
    fieldnames = list(opt_source_df.columns)
    if "failed" not in fieldnames:
        fieldnames.append("failed")
    if "error" not in fieldnames:
        fieldnames.append("error")

    candidates = _load_top_candidates(
        Path(args.opt_csv).resolve(),
        args.top_n,
        skip_first=max(0, int(args.skip_first)),
    )
    if not candidates:
        raise RuntimeError("No eligible candidates found in optimization CSV.")

    max_workers = max(1, int(args.max_workers))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _run_worker,
                row=row,
                rank=idx,
                out_root=out_dir,
            ): (idx, row)
            for idx, row in enumerate(candidates, start=1)
        }
        for fut in as_completed(futures):
            idx, base_row = futures[fut]
            try:
                result = fut.result()
            except Exception as exc:
                failed_row = dict(base_row)
                failed_row.update(
                    {
                        "failed": True,
                        "error": str(exc),
                    }
                )
                _append_result_row(final_csv, failed_row, fieldnames)
                print(f"❌ Saved failed result for strategy {idx}/{len(candidates)} -> {final_csv}")
                continue
            _append_result_row(final_csv, result, fieldnames)
            print(f"✅ Saved metrics for strategy {idx}/{len(candidates)} -> {final_csv}")

    print(f"\nDone. Results CSV: {final_csv}")


if __name__ == "__main__":
    main()
