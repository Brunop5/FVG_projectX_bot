#!/usr/bin/env python3
"""
Run full Binance-data backtests for top optimization candidates.

- Reads top N rows by objective from optimization_results/binance/optimization_results.csv
- Re-runs each strategy on full Binance 15m/1m data (no last-quarter or fixed-bar truncation)
- Writes each strategy output into its own folder under backtest/btc_opt_results/
- Appends summary metrics to backtest/btc_opt_results/final_result.csv after each run
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = SCRIPT_DIR.parent
WORKSPACE_DIR = PACKAGE_DIR.parent

BINANCE_OPT_CSV = SCRIPT_DIR / "optimization_results" / "binance" / "optimization_results.csv"
BINANCE_INPUTS_PATH = SCRIPT_DIR / "inputs_binance.json"
OUT_DIR = SCRIPT_DIR / "results" / "btc_opt_results"
RESULT_MARKER = "__BINANCE_TOP_RESULT__"


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
    df = df[df["objective"].notna()].sort_values("objective", ascending=True)
    if skip_first > 0:
        df = df.iloc[skip_first:]
    return df.head(top_n).to_dict(orient="records")


def _build_trial_inputs(row: dict[str, Any]) -> dict[str, Any]:
    with BINANCE_INPUTS_PATH.open("r", encoding="utf-8") as f:
        base_inputs = json.load(f)
    merged = dict(base_inputs)
    for key in base_inputs.keys():
        if key in row:
            v = _to_native(row[key])
            if v is not None:
                merged[key] = v
    return merged


def _strategy_folder_name(rank: int, row: dict[str, Any]) -> str:
    return f"test_{rank}"


def _force_backtest_second_half(backtest: Any) -> None:
    """
    Move backtest cursor to second half of loaded 15m data and rebuild warmup window.
    """
    full_data = getattr(backtest, "_full_data", None)
    if full_data is None or len(full_data) == 0:
        return
    total = len(full_data)
    midpoint = total // 2
    existing_cursor = int(getattr(backtest, "_cursor", 0))
    new_cursor = max(midpoint, existing_cursor)
    if new_cursor >= total:
        new_cursor = max(0, total - 1)
    backtest._cursor = new_cursor
    window = backtest._get_window_size()
    start = max(0, new_cursor - window)
    backtest.data = full_data.iloc[start:new_cursor].copy().reset_index(drop=True)


def _ensure_trades_csv_exists(path: Path) -> None:
    """
    Pre-create trades CSV with header so it's visible during long runs
    even before first close event is recorded.
    """
    if path.exists():
        return
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


def _run_worker(row: dict[str, Any], rank: int, out_root: Path) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="btc_top12_") as tmp_dir:
        strategy_dir = out_root / _strategy_folder_name(rank, row)
        strategy_dir.mkdir(parents=True, exist_ok=True)

        trial_inputs = _build_trial_inputs(row)
        # Write outputs directly in strategy folder (easier to monitor live).
        trial_inputs["RUNTIME_SUBDIR"] = str(OUT_DIR / strategy_dir.name)

        inputs_path = Path(tmp_dir) / "inputs_worker.json"
        payload_path = Path(tmp_dir) / "row_payload.json"
        with inputs_path.open("w", encoding="utf-8") as f:
            json.dump(trial_inputs, f, indent=2)
        with payload_path.open("w", encoding="utf-8") as f:
            json.dump(_json_safe_dict(row), f)

        cmd = [
            sys.executable,
            "-u",
            "-m",
            "FVG_projectX_bot.backtest.run_binance_top",
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
        insufficient_margin_detected = False
        assert proc.stdout is not None
        for raw in proc.stdout:
            line = raw.rstrip("\n")
            out_lines.append(line)
            if "insufficient margin" in line.lower():
                insufficient_margin_detected = True
                print(f"[top{rank:02d}] ⛔ Failing strategy early: insufficient margin detected.")
                try:
                    proc.terminate()
                except Exception:
                    pass
                break
            if line.startswith(RESULT_MARKER):
                result_payload = json.loads(line[len(RESULT_MARKER):])
                continue
            print(f"[top{rank:02d}] {line}")

        if insufficient_margin_detected:
            try:
                proc.wait(timeout=5)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
            raise RuntimeError("Worker aborted early: insufficient margin.")

        rc = proc.wait()
        if rc != 0:
            raise RuntimeError(f"Worker failed (code={rc}): {' | '.join(out_lines[-12:])}")
        if result_payload is None:
            raise RuntimeError("Worker returned no result payload.")
        return result_payload


def _worker_mode(inputs_path: Path, row_payload: Path, strategy_dir: Path) -> None:
    os.environ["FVG_INPUTS_JSON"] = str(inputs_path.resolve())

    from FVG_projectX_bot.backtest import FVG_backtest as bt
    from FVG_projectX_bot.backtest.evaluate_backtest import evaluate_backtest

    with row_payload.open("r", encoding="utf-8") as f:
        row = json.load(f)

    bt.ASSET = "BTC"
    bt.TIMEFRAME = "15m"
    bt.INITIAL_BALANCE = 500.0
    bt.LEVERAGE = 50
    bt.USE_MARGIN_PER_TRADE = True
    bt.MARGIN_PER_TRADE_USD = 10
    bt.DATA_CSV_PATH = str(SCRIPT_DIR / "data" / "BTCUSDT_PERP_15m.csv")
    bt.DATA_1M_CSV_PATH = str(SCRIPT_DIR / "data" / "BTCUSDT_PERP_1m.csv")
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
    _force_backtest_second_half(backtest)
    _ensure_trades_csv_exists(Path(backtest.trades_csv_path))
    backtest.run()

    metrics_csv = strategy_dir / "backtest_trades_metrics.csv"
    monthly_csv = strategy_dir / "backtest_trades_monthly_pnl.csv"
    drawdowns_csv = strategy_dir / "backtest_trade_drawdowns.csv"
    plot_png = strategy_dir / "equity_vs_btc.png"
    summary = evaluate_backtest(
        trades_csv=Path(backtest.trades_csv_path),
        price_csv=Path(backtest.data_path),
        output_metrics_csv=metrics_csv,
        output_monthly_csv=monthly_csv,
        output_drawdowns_csv=drawdowns_csv,
        output_plot=plot_png,
        start_equity=bt.INITIAL_BALANCE,
    )

    trades_df = pd.read_csv(backtest.trades_csv_path)
    pnl = pd.to_numeric(trades_df.get("pnl", pd.Series(dtype=float)), errors="coerce").dropna()
    win_count = int((pnl > 0).sum()) if not pnl.empty else 0
    loss_count = int((pnl < 0).sum()) if not pnl.empty else 0
    largest_win = float(pnl.max()) if not pnl.empty else 0.0
    largest_loss = float(pnl.min()) if not pnl.empty else 0.0

    total_fees = 0.0
    if "total_fees" in trades_df.columns:
        total_fees = float(pd.to_numeric(trades_df["total_fees"], errors="coerce").fillna(0).sum())
    elif "fees" in trades_df.columns:
        total_fees = float(pd.to_numeric(trades_df["fees"], errors="coerce").fillna(0).sum())

    avg_win = float(summary.get("average_winning_trade", 0.0))
    avg_loss = float(summary.get("average_losing_trade", 0.0))
    avg_win_vs_loss = (avg_win / abs(avg_loss)) if avg_loss not in (0.0, -0.0) else math.inf

    pnl_total = float(summary.get("total_pnl_original", 0.0))
    max_dd = float(summary.get("max_drawdown", 0.0))
    ratio = (max_dd / pnl_total) if pnl_total > 0 else math.inf
    objective = (
        (1_000_000.0 + abs(pnl_total) + max_dd)
        if pnl_total <= 0
        else ratio
    )
    result_row = dict(row)
    result_row.update(
        {
            "failed": False,
            "error": "",
            "ratio": float(ratio) if math.isfinite(ratio) else None,
            "tpd_penalty_factor": 1.0,
            "pnl": pnl_total,
            "max_dd": max_dd,
            "trades_per_day": float(summary.get("num_trades", 0) / max(float(summary.get("days_of_backtest", 0.0) or 1.0), 1e-9)),
            "trades_total": int(summary.get("num_trades", 0)),
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
    parser = argparse.ArgumentParser(description="Run full-data Binance backtests for top optimized strategies.")
    parser.add_argument("--top-n", type=int, default=12, help="How many top rows by objective to run.")
    parser.add_argument("--skip-first", type=int, default=4, help="Skip the first N top-ranked rows before selecting top-n.")
    parser.add_argument("--max-workers", type=int, default=4, help="How many backtests to run concurrently.")
    parser.add_argument("--opt-csv", type=str, default=str(BINANCE_OPT_CSV), help="Path to binance optimization_results.csv.")
    parser.add_argument("--out-dir", type=str, default=str(OUT_DIR), help="Output folder for this runner.")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--inputs-path", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--row-payload", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--strategy-dir", type=str, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker:
        if not args.inputs_path or not args.row_payload or not args.strategy_dir:
            raise ValueError("Worker mode requires --inputs-path, --row-payload, and --strategy-dir.")
        _worker_mode(Path(args.inputs_path), Path(args.row_payload), Path(args.strategy_dir))
        return

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    final_csv = out_dir / "optimization_results.csv"
    opt_source_df = pd.read_csv(Path(args.opt_csv).resolve())
    fieldnames = list(opt_source_df.columns)

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
            executor.submit(_run_worker, row=row, rank=idx, out_root=out_dir): (idx, row)
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
