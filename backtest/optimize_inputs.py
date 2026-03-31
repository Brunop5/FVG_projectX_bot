#!/usr/bin/env python3
"""
Optimize strategy inputs for backtest using Optuna (TPE sampler).

Objective (minimize):
    max_drawdown / ending_pnl

Notes:
- Keeps money/fee settings from backtest/FVG_backtest.py unchanged.
- Only mutates the requested INPUTS fields.
- Runs each trial on last N months (default: 3) of Binance backtest data.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import optuna
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "optuna is required. Install with: pip install optuna"
    ) from exc


SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = SCRIPT_DIR.parent
WORKSPACE_DIR = PACKAGE_DIR.parent
DEFAULT_BINANCE_INPUTS_PATH = SCRIPT_DIR / "inputs_binance.json"
DEFAULT_TOPSTEP_INPUTS_PATH = SCRIPT_DIR / "inputs_topstep.json"
DEFAULT_OPT_ROOT = SCRIPT_DIR / "optimization_results"

RESULT_MARKER = "__OPT_RESULT__"

TARGET_CONFIGS: dict[str, dict[str, Any]] = {
    "binance": {
        "asset": "BTC",
        "timeframe": "15m",
        "initial_balance": 50.0,
        "data_path": SCRIPT_DIR / "data" / "BTCUSDT_PERP_15m.csv",
        "data_path_1m": SCRIPT_DIR / "data" / "BTCUSDT_PERP_1m.csv",
        "fixed_15m_bars": 3827,
        "default_inputs_path": DEFAULT_BINANCE_INPUTS_PATH,
    },
    "topstep": {
        "asset": "MGCJ6",
        "timeframe": "15m",
        "initial_balance": 50000.0,
        "data_path": SCRIPT_DIR / "data" / "MGCJ6" / "topstep_15min.csv",
        "data_path_1m": SCRIPT_DIR / "data" / "MGCJ6" / "topstep_1min.csv",
        "fixed_15m_bars": None,
        "default_inputs_path": DEFAULT_TOPSTEP_INPUTS_PATH,
    },
}


def _load_base_inputs(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _compute_start_timestamp_ms(months: int) -> int:
    now_utc = datetime.now(timezone.utc)
    start = now_utc - timedelta(days=30 * months)
    return int(start.timestamp() * 1000)


def _resolve_targets(target_arg: str) -> list[str]:
    normalized = (target_arg or "binance").strip().lower()
    if normalized == "gold":
        normalized = "topstep"
    if normalized == "both":
        return ["binance", "topstep"]
    if normalized not in TARGET_CONFIGS:
        raise ValueError(
            f"Unknown target '{target_arg}'. Use one of: "
            f"{', '.join(sorted(list(TARGET_CONFIGS.keys()) + ['both']))}"
        )
    return [normalized]


def _write_results_snapshot(records: list[dict[str, Any]], out_csv: Path) -> None:
    results_df = pd.DataFrame(records)
    if not results_df.empty:
        results_df = results_df.sort_values(
            by=["ratio", "objective"],
            ascending=[True, True],
            na_position="last",
        ).reset_index(drop=True)
    results_df.to_csv(out_csv, index=False)


def _load_existing_records(in_csv: Path) -> list[dict[str, Any]]:
    if not in_csv.exists():
        return []
    try:
        df = pd.read_csv(in_csv)
    except Exception:
        return []
    if df.empty:
        return []
    return df.to_dict(orient="records")


def _suggest_params(trial: "optuna.trial.Trial", base_inputs: dict[str, Any]) -> dict[str, Any]:
    params: dict[str, Any] = {
        "FVG_HISTORY_NBR": trial.suggest_int("FVG_HISTORY_NBR", 1, 15, step=1),
        "MIN_FVG_POWER_PCT": trial.suggest_float("MIN_FVG_POWER_PCT", 0.0, 0.1, step=0.01),
        "HTF_TF": trial.suggest_categorical("HTF_TF", [30, 60, 120, 240]),
        "EMA_PERIOD": trial.suggest_categorical("EMA_PERIOD", [10, 25, 50, 100, 200]),
        "VOLUME_MULTIPLIER": trial.suggest_float("VOLUME_MULTIPLIER", 1.0, 1.3, step=0.05),
        "USE_VOLUME_CHECK": trial.suggest_categorical("USE_VOLUME_CHECK", [True, False]),
        "ATR_PERIOD": trial.suggest_int("ATR_PERIOD", 5, 25, step=1),
        "SL_MULTIPLIER": trial.suggest_int("SL_MULTIPLIER", 1, 20, step=1),
        "TP_MULTIPLIER": trial.suggest_int("TP_MULTIPLIER", 1, 20, step=1),
        "USE_TRAILING": trial.suggest_categorical("USE_TRAILING", [True, False]),
        "HOLD_UNTIL_OPPOSITE": trial.suggest_categorical("HOLD_UNTIL_OPPOSITE", [True, False]),
    }

    if params["USE_TRAILING"]:
        params["TRAIL_OFFSET_MULT"] = trial.suggest_int("TRAIL_OFFSET_MULT", 1, 20, step=1)
    else:
        # Keep baseline value if trailing is disabled.
        params["TRAIL_OFFSET_MULT"] = base_inputs.get("TRAIL_OFFSET_MULT", 1.0)

    return params


def _run_trial_subprocess(
    *,
    trial_inputs: dict[str, Any],
    trial_idx: int,
    months: int,
    target: str,
    min_trades_per_day: float,
    tpd_penalty_power: float,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="fvg_opt_") as tmp_dir:
        tmp_inputs_path = Path(tmp_dir) / f"inputs_trial_{trial_idx}.json"
        with tmp_inputs_path.open("w", encoding="utf-8") as f:
            json.dump(trial_inputs, f, indent=2)

        cmd = [
            sys.executable,
            "-u",
            "-m",
            "FVG_projectX_bot.backtest.optimize_inputs",
            "--worker",
            "--inputs-path",
            str(tmp_inputs_path),
            "--months",
            str(months),
            "--target",
            target,
            "--min-trades-per-day",
            str(min_trades_per_day),
            "--tpd-penalty-power",
            str(tpd_penalty_power),
        ]
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        proc = subprocess.Popen(
            cmd,
            cwd=str(WORKSPACE_DIR),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            env=env,
        )
        output_lines: list[str] = []
        result_payload: dict[str, Any] | None = None

        assert proc.stdout is not None
        for raw_line in proc.stdout:
            line = raw_line.rstrip("\n")
            output_lines.append(line)
            if line.startswith(RESULT_MARKER):
                result_payload = json.loads(line[len(RESULT_MARKER):])
                continue
            # Show only per-backtest progress lines.
            if line.startswith("[backtest]"):
                print(f"{target.upper()} TEST NO {trial_idx}: {line}")

        return_code = proc.wait()
        if return_code != 0:
            stdout_tail = output_lines[-12:]
            raise RuntimeError(
                f"Worker failed (code={return_code}). "
                f"stdout_tail={' | '.join(stdout_tail)}"
            )
        if result_payload is None:
            raise RuntimeError("Worker completed but returned no result payload.")
        return result_payload


def _force_backtest_tail_bars(backtest: Any, bar_count: int) -> None:
    if bar_count <= 0:
        return
    full_data = getattr(backtest, "_full_data", None)
    if full_data is None or len(full_data) == 0:
        return
    total = len(full_data)
    new_cursor = max(0, total - int(bar_count))
    # Keep warmup safety: if current cursor is already later, respect it.
    existing_cursor = int(getattr(backtest, "_cursor", 0))
    new_cursor = max(new_cursor, existing_cursor)
    if new_cursor >= total:
        new_cursor = max(0, total - 1)
    backtest._cursor = new_cursor
    window = backtest._get_window_size()
    start = max(0, new_cursor - window)
    backtest.data = full_data.iloc[start:new_cursor].copy().reset_index(drop=True)


def _worker_mode(
    inputs_path: Path,
    months: int,
    target: str,
    min_trades_per_day: float,
    tpd_penalty_power: float,
) -> None:
    os.environ["FVG_INPUTS_JSON"] = str(inputs_path.resolve())

    from FVG_projectX_bot.backtest import FVG_backtest as bt
    from FVG_projectX_bot.backtest.evaluate_backtest import evaluate_backtest

    target_cfg = TARGET_CONFIGS[target]
    bt.ASSET = str(target_cfg["asset"])
    bt.TIMEFRAME = str(target_cfg["timeframe"])
    bt.INITIAL_BALANCE = float(target_cfg["initial_balance"])
    bt.DATA_CSV_PATH = str(target_cfg["data_path"])
    bt.DATA_1M_CSV_PATH = str(target_cfg["data_path_1m"])

    # Force exact time window control from start_timestamp (not "last quarter of rows").
    bt.USE_LAST_QUARTER_DATA = False
    start_ts_ms = _compute_start_timestamp_ms(months)

    backtest = bt.FVG_Backtest(
        asset=bt.ASSET,
        timeframe=bt.TIMEFRAME,
        initial_balance=bt.INITIAL_BALANCE,
        data_path=bt.DATA_CSV_PATH,
        start_timestamp=str(start_ts_ms),
        pyramiding_mode=bt.PYRAMIDING_MODE,
        data_path_1m=bt.DATA_1M_CSV_PATH,
    )
    fixed_bars = target_cfg.get("fixed_15m_bars")
    if fixed_bars is not None:
        _force_backtest_tail_bars(backtest, int(fixed_bars))
    start_cursor = int(getattr(backtest, "_cursor", 0))
    total_bars = len(getattr(backtest, "_full_data", []))
    tested_bars = max(1, total_bars - start_cursor)
    stop_progress = threading.Event()

    def _progress_reporter() -> None:
        if tested_bars <= 0:
            return
        last_pct = -5.0
        while not stop_progress.is_set():
            cursor = int(getattr(backtest, "_cursor", 0))
            tested_cursor = max(0, min(tested_bars, cursor - start_cursor))
            pct = max(0.0, min(100.0, (tested_cursor / tested_bars) * 100.0))
            if pct - last_pct >= 5.0 or (pct >= 100.0 and last_pct < 100.0):
                print(
                    f"[backtest] {pct:5.1f}% ({tested_cursor}/{tested_bars})",
                    flush=True,
                )
                last_pct = pct
            time.sleep(0.5)

    progress_thread = threading.Thread(target=_progress_reporter, daemon=True)
    progress_thread.start()
    try:
        backtest.run()
    finally:
        stop_progress.set()
        progress_thread.join(timeout=1.0)

    summary = evaluate_backtest(
        trades_csv=Path(backtest.trades_csv_path),
        price_csv=Path(backtest.data_path),
        start_equity=bt.INITIAL_BALANCE,
    )

    ending_pnl = float(summary.get("total_pnl_original", 0.0))
    max_drawdown = float(summary.get("max_drawdown", 0.0))
    num_trades = int(summary.get("num_trades", 0))
    days_of_backtest = float(summary.get("days_of_backtest", 0.0))
    avg_win = float(summary.get("average_winning_trade", 0.0))
    avg_loss = float(summary.get("average_losing_trade", 0.0))
    trades_per_day = (num_trades / days_of_backtest) if days_of_backtest > 0 else 0.0

    trades_df = pd.read_csv(backtest.trades_csv_path)
    if trades_df.empty or "pnl" not in trades_df.columns:
        win_rate = 0.0
    else:
        pnl_series = pd.to_numeric(trades_df["pnl"], errors="coerce").dropna()
        win_rate = float((pnl_series > 0).mean() * 100.0) if not pnl_series.empty else 0.0

    ratio = (max_drawdown / ending_pnl) if ending_pnl > 0 else math.inf
    tpd_penalty_factor = 1.0
    if trades_per_day < min_trades_per_day:
        # Smoothly increase penalty as activity drops below threshold.
        # Example with default power=2:
        #   0.4 -> (0.5/0.4)^2 = 1.56 (mild)
        #   0.04 -> (0.5/0.04)^2 = 156.25 (huge)
        safe_tpd = max(trades_per_day, 1e-9)
        tpd_penalty_factor = (min_trades_per_day / safe_tpd) ** tpd_penalty_power

    # Objective: lower drawdown-to-profit ratio is better.
    # Penalize non-profitable / zero-profit runs heavily.
    if ending_pnl <= 0:
        objective = 1_000_000.0 + abs(ending_pnl) + max_drawdown
    else:
        objective = ratio * tpd_penalty_factor

    payload = {
        "objective": float(objective),
        "ratio": float(ratio) if math.isfinite(ratio) else None,
        "tpd_penalty_factor": float(tpd_penalty_factor),
        "ending_pnl": ending_pnl,
        "max_drawdown": max_drawdown,
        "num_trades": num_trades,
        "trades_per_day": float(trades_per_day),
        "avg_win": float(avg_win),
        "avg_loss": float(avg_loss),
        "win_rate": float(win_rate),
    }
    print(f"{RESULT_MARKER}{json.dumps(payload)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Optimize backtest inputs with Optuna TPE."
    )
    parser.add_argument("--trials", type=int, default=100, help="Number of optimization trials.")
    parser.add_argument("--months", type=int, default=3, help="Use last N months of data.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampler.")
    parser.add_argument("--timeout-sec", type=int, default=None, help="Optional timeout (seconds).")
    parser.add_argument("--n-jobs", type=int, default=1, help="Parallel trials (default: 1).")
    parser.add_argument(
        "--target",
        type=str,
        default="binance",
        help="Optimization target: binance, topstep (or gold alias), or both.",
    )
    parser.add_argument(
        "--inputs-path",
        type=str,
        default=None,
        help="Optional base inputs.json path override for single-target runs.",
    )
    parser.add_argument(
        "--min-trades-per-day",
        type=float,
        default=0.5,
        help="Apply increasing penalty when trades/day is below this threshold.",
    )
    parser.add_argument(
        "--tpd-penalty-power",
        type=float,
        default=2.0,
        help="Penalty curve exponent for low trades/day.",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--child-run", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    targets = _resolve_targets(args.target)

    if args.worker:
        if len(targets) != 1:
            raise ValueError("Worker mode requires a single --target (binance or gold).")
        worker_inputs_path = (
            Path(args.inputs_path).resolve()
            if args.inputs_path
            else Path(TARGET_CONFIGS[targets[0]]["default_inputs_path"]).resolve()
        )
        _worker_mode(
            worker_inputs_path,
            args.months,
            targets[0],
            args.min_trades_per_day,
            args.tpd_penalty_power,
        )
        return

    # Parent launcher: run both targets concurrently as isolated child processes.
    if len(targets) > 1 and not args.child_run:
        children: list[subprocess.Popen] = []
        for target in targets:
            cmd = [
                sys.executable,
                "-u",
                "-m",
                "FVG_projectX_bot.backtest.optimize_inputs",
                "--target",
                target,
                "--trials",
                str(args.trials),
                "--months",
                str(args.months),
                "--seed",
                str(args.seed),
                "--n-jobs",
                str(args.n_jobs),
                "--child-run",
                "--min-trades-per-day",
                str(args.min_trades_per_day),
                "--tpd-penalty-power",
                str(args.tpd_penalty_power),
            ]
            if args.timeout_sec is not None:
                cmd.extend(["--timeout-sec", str(args.timeout_sec)])
            proc = subprocess.Popen(
                cmd,
                cwd=str(WORKSPACE_DIR),
                text=True,
            )
            children.append(proc)
        exit_code = 0
        for proc in children:
            rc = proc.wait()
            if rc != 0:
                exit_code = rc
        if exit_code != 0:
            raise SystemExit(exit_code)
        return

    target = targets[0]
    opt_root_dir = DEFAULT_OPT_ROOT / target
    opt_root_dir.mkdir(parents=True, exist_ok=True)

    default_inputs = Path(TARGET_CONFIGS[target]["default_inputs_path"]).resolve()
    selected_inputs_path = Path(args.inputs_path).resolve() if args.inputs_path else default_inputs
    base_inputs = _load_base_inputs(selected_inputs_path)
    trial_records: list[dict[str, Any]] = []
    records_lock = threading.Lock()
    results_csv = opt_root_dir / "optimization_results.csv"
    trial_records = _load_existing_records(results_csv)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")

    def objective(trial: "optuna.trial.Trial") -> float:
        trial_params = _suggest_params(trial, base_inputs)
        merged_inputs = dict(base_inputs)
        merged_inputs.update(trial_params)
        # Isolate runtime outputs by run_id + trial number so restarts never overwrite.
        merged_inputs["RUNTIME_SUBDIR"] = (
            f"optimization_results/{target}/trial_runs/run_{run_id}/trial_{trial.number}"
        )

        try:
            result = _run_trial_subprocess(
                trial_inputs=merged_inputs,
                trial_idx=trial.number,
                months=args.months,
                target=target,
                min_trades_per_day=args.min_trades_per_day,
                tpd_penalty_power=args.tpd_penalty_power,
            )
            failed = False
            fail_reason = ""
        except Exception as exc:
            # Keep study running even if one parameter set crashes strategy internals.
            failed = True
            fail_reason = str(exc)
            result = {
                "objective": 9_999_999.0,
                "ratio": None,
                "tpd_penalty_factor": 1.0,
                "ending_pnl": 0.0,
                "max_drawdown": 0.0,
                "num_trades": 0,
                "trades_per_day": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "win_rate": 0.0,
            }

        trial.set_user_attr("ending_pnl", float(result["ending_pnl"]))
        trial.set_user_attr("max_drawdown", float(result["max_drawdown"]))
        trial.set_user_attr("num_trades", int(result["num_trades"]))
        trial.set_user_attr("ratio", result.get("ratio"))
        trial.set_user_attr("tpd_penalty_factor", float(result["tpd_penalty_factor"]))
        trial.set_user_attr("trades_per_day", float(result["trades_per_day"]))
        trial.set_user_attr("avg_win", float(result["avg_win"]))
        trial.set_user_attr("avg_loss", float(result["avg_loss"]))
        trial.set_user_attr("win_rate", float(result["win_rate"]))

        record = {
            "run_id": run_id,
            "trial": trial.number,
            "failed": failed,
            "error": fail_reason,
            "ratio": result.get("ratio"),
            "tpd_penalty_factor": float(result["tpd_penalty_factor"]),
            "pnl": float(result["ending_pnl"]),
            "max_dd": float(result["max_drawdown"]),
            "trades_per_day": float(result["trades_per_day"]),
            "trades_total": int(result["num_trades"]),
            "average_win": float(result["avg_win"]),
            "average_loss": float(result["avg_loss"]),
            "win_rate": float(result["win_rate"]),
            "objective": float(result["objective"]),
        }
        # Include all input fields (not only optimized ones) for each contestant.
        record.update(merged_inputs)
        with records_lock:
            trial_records.append(record)
            _write_results_snapshot(trial_records, results_csv)

        return float(result["objective"])

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(
        objective,
        n_trials=args.trials,
        timeout=args.timeout_sec,
        n_jobs=max(1, int(args.n_jobs)),
        show_progress_bar=False,
    )

    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not complete_trials:
        raise RuntimeError("No completed trials.")

    with records_lock:
        _write_results_snapshot(trial_records, results_csv)

    best = study.best_trial
    best_inputs = dict(base_inputs)
    best_inputs.update(best.params)
    if not best_inputs.get("USE_TRAILING", False):
        best_inputs["TRAIL_OFFSET_MULT"] = base_inputs.get("TRAIL_OFFSET_MULT", 1.0)

    best_inputs_path = opt_root_dir / "optimized_inputs_best.json"
    with best_inputs_path.open("w", encoding="utf-8") as f:
        json.dump(best_inputs, f, indent=2)

    print("\n✅ Optimization finished")
    print(f"Target: {target}")
    print(f"Trials: {len(complete_trials)}")
    print(f"Best objective (max_drawdown / ending_pnl): {best.value:.6f}")
    print(f"Best ending_pnl: {best.user_attrs.get('ending_pnl')}")
    print(f"Best max_drawdown: {best.user_attrs.get('max_drawdown')}")
    print(f"Saved trial table: {results_csv}")
    print(f"Saved best inputs: {best_inputs_path}")
    print(
        "\nRun backtest with best inputs:\n"
        f"FVG_INPUTS_JSON=\"{best_inputs_path}\" python -m FVG_projectX_bot.backtest.FVG_backtest"
    )


if __name__ == "__main__":
    main()
