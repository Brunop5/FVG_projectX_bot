#!/usr/bin/env python3
"""
Optimize strategy inputs for backtest using Optuna (TPE sampler).

Objective (maximize):
    ending_pnl / max_drawdown

Hybrid search:
- Run an initial batch of purely random trials (warmup exploration)
- Then switch to Optuna TPE optimization

Notes:
- Binance keeps margin-per-trade sizing + %-of-notional fees from FVG_backtest.py.
- Gold/topstep uses fixed lot + per-contract round-turn fees (see configure_futures_backtest).
- Gold/topstep evaluates the full 15m/1m overlap window (intrabar-valid bars only).
- Runs each trial on fixed Binance regime windows, or the full overlap for gold.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import random
import re
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import optuna
except ImportError:  # pragma: no cover
    optuna = None


SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = SCRIPT_DIR.parent
WORKSPACE_DIR = PACKAGE_DIR.parent
DEFAULT_BINANCE_INPUTS_PATH = SCRIPT_DIR / "inputs_binance.json"
DEFAULT_TOPSTEP_INPUTS_PATH = SCRIPT_DIR / "inputs_topstep.json"
DEFAULT_OPT_ROOT = SCRIPT_DIR / "optimization_results"

RESULT_MARKER = "__OPT_RESULT__"
PHASE_MIN_COMPLETED_TRIALS = 30
PHASE_MEDIAN_IMPROVEMENT_MIN = 0.015
MAXIMIZE_EPS = 1e-9

TARGET_CONFIGS: dict[str, dict[str, Any]] = {
    "binance": {
        "asset": "BTC",
        "timeframe": "15m",
        "initial_balance": 500.0,
        "data_path": SCRIPT_DIR / "data" / "BTCUSDT_PERP_15m.csv",
        "data_path_1m": SCRIPT_DIR / "data" / "BTCUSDT_PERP_1m.csv",
        "fixed_15m_bars": 3827,
        "default_inputs_path": DEFAULT_BINANCE_INPUTS_PATH,
        "tick_size": 0.0,
        "tick_value": 1.0,
        "benchmark_label": "Buy & Hold BTC",
    },
    "topstep": {
        "asset": "MGCQ6",
        "timeframe": "15m",
        "initial_balance": 50000.0,
        "data_path": SCRIPT_DIR / "data" / "MGCQ6" / "topstep_15min.csv",
        "data_path_1m": SCRIPT_DIR / "data" / "MGCQ6" / "topstep_1min.csv",
        "fixed_15m_bars": None,
        "default_inputs_path": DEFAULT_TOPSTEP_INPUTS_PATH,
        "tick_size": 0.1,
        "tick_value": 1.0,
        "benchmark_label": "Buy & Hold Gold",
    },
}

# Handpicked Binance BTC regime windows from explicit data/performance review.
# Dates are UTC, inclusive on both ends (implemented as [start, end+1day)).
BINANCE_REGIME_WINDOWS: list[tuple[str, str, str]] = [
    ("ftx_aftershock_reflex_bounce", "2022-11-08", "2022-12-20"),
    ("q1_2023_short_squeeze_uptrend", "2023-01-08", "2023-03-20"),
    ("q3_2023_low_volatility_chop", "2023-07-01", "2023-09-30"),
    ("q4_2024_breakout_then_pullback", "2024-10-01", "2024-12-20"),
    ("q1_2025_whipsaw_distribution", "2025-01-15", "2025-03-31"),
]


def _load_base_inputs(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


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
        # Sort best-first for maximize objective.
        # Keep the full table (including non-profitable runs) for debugging.
        if "objective" in results_df.columns:
            results_df["objective"] = pd.to_numeric(results_df["objective"], errors="coerce")
        if "ratio" in results_df.columns:
            results_df["ratio"] = pd.to_numeric(results_df["ratio"], errors="coerce")
        results_df = results_df.sort_values(
            by=["objective", "ratio"],
            ascending=[False, False],
            na_position="last",
        ).reset_index(drop=True)
        # Keep these internals for in-memory bookkeeping, but do not persist them.
        for transient_col in ("failed", "error"):
            if transient_col in results_df.columns:
                results_df = results_df.drop(columns=[transient_col])
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


def _discover_max_trial_id(trial_runs_root: Path) -> int:
    if not trial_runs_root.exists():
        return -1
    max_trial = -1
    for path in trial_runs_root.rglob("trial_*"):
        if not path.is_dir():
            continue
        match = re.search(r"trial_(\d+)$", path.name)
        if not match:
            continue
        trial_id = int(match.group(1))
        if trial_id > max_trial:
            max_trial = trial_id
    return max_trial


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


def _rand_int_step(rng: random.Random, start: int, stop: int, step: int = 1) -> int:
    return int(rng.randrange(int(start), int(stop) + int(step), int(step)))


def _rand_float_step(rng: random.Random, start: float, stop: float, step: float) -> float:
    steps = int(round((float(stop) - float(start)) / float(step)))
    idx = rng.randint(0, max(0, steps))
    value = float(start) + float(step) * idx
    return float(round(value, 10))


def _sample_random_params(base_inputs: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    params: dict[str, Any] = {
        "FVG_HISTORY_NBR": _rand_int_step(rng, 1, 15, 1),
        "MIN_FVG_POWER_PCT": _rand_float_step(rng, 0.0, 0.1, 0.01),
        "HTF_TF": int(rng.choice([30, 60, 120, 240])),
        "EMA_PERIOD": int(rng.choice([10, 25, 50, 100, 200])),
        "VOLUME_MULTIPLIER": _rand_float_step(rng, 1.0, 1.3, 0.05),
        "USE_VOLUME_CHECK": bool(rng.choice([True, False])),
        "ATR_PERIOD": _rand_int_step(rng, 5, 25, 1),
        "SL_MULTIPLIER": _rand_int_step(rng, 1, 20, 1),
        "TP_MULTIPLIER": _rand_int_step(rng, 1, 20, 1),
        "USE_TRAILING": bool(rng.choice([True, False])),
        "HOLD_UNTIL_OPPOSITE": bool(rng.choice([True, False])),
    }
    if params["USE_TRAILING"]:
        params["TRAIL_OFFSET_MULT"] = _rand_int_step(rng, 1, 20, 1)
    else:
        params["TRAIL_OFFSET_MULT"] = base_inputs.get("TRAIL_OFFSET_MULT", 1.0)
    return params


def _count_consecutive_non_improvements(
    completed_trials: list["optuna.trial.FrozenTrial"],
    *,
    objective_abs_tol: float,
    objective_rel_tol: float,
) -> tuple[int, float | None]:
    """
    Return (non_improvement_streak, best_objective_seen).

    A trial counts as an improvement only if it beats the best objective by more than
    max(objective_abs_tol, objective_rel_tol * max(1, abs(best))).
    """
    valid_trials = [
        t
        for t in completed_trials
        if t.value is not None and t.state == optuna.trial.TrialState.COMPLETE
    ]
    if not valid_trials:
        return (0, None)

    best_value = math.inf
    non_improvement_streak = 0
    for trial in valid_trials:
        value = float(trial.value)
        if not math.isfinite(best_value):
            best_value = value
            non_improvement_streak = 0
            continue
        improvement_eps = max(
            float(objective_abs_tol),
            float(objective_rel_tol) * max(1.0, abs(float(best_value))),
        )
        if value < (best_value - improvement_eps):
            best_value = value
            non_improvement_streak = 0
        else:
            non_improvement_streak += 1
    return (non_improvement_streak, float(best_value))


def _run_trial_subprocess(
    *,
    trial_inputs: dict[str, Any],
    trial_idx: int,
    target: str,
    min_trades_per_day: float,
    tpd_penalty_power: float,
) -> dict[str, Any]:
    # Run worker as a subprocess but pass inputs inline (no temp files),
    # and hard-disable any disk writes during the run.
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "FVG_projectX_bot.backtest.optimize_inputs",
        "--worker",
        "--target",
        target,
        "--min-trades-per-day",
        str(min_trades_per_day),
        "--tpd-penalty-power",
        str(tpd_penalty_power),
    ]
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["FVG_NO_DISK_IO"] = "1"
    env["FVG_INPUTS_JSON_INLINE"] = json.dumps(trial_inputs)
    env.pop("FVG_INPUTS_JSON", None)

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


def _timeframe_to_minutes(timeframe: str) -> int:
    tf = str(timeframe).strip().lower()
    if tf.isdigit():
        return int(tf)
    if tf.endswith("m") and tf[:-1].isdigit():
        return int(tf[:-1])
    if tf.endswith("h") and tf[:-1].isdigit():
        return int(tf[:-1]) * 60
    if tf.endswith("d") and tf[:-1].isdigit():
        return int(tf[:-1]) * 1440
    raise ValueError(f"Unsupported timeframe format: {timeframe!r}")


def _force_backtest_spaced_month_chunk(
    backtest: Any,
    *,
    chunk_bars: int,
    end_offset_bars: int,
) -> bool:
    """
    Restrict backtest to a one-month chunk ending `end_offset_bars` before latest bar.
    Returns False when there is not enough data for this chunk.
    """
    full_data = getattr(backtest, "_full_data", None)
    if full_data is None or len(full_data) == 0:
        return False
    total = len(full_data)
    end_idx = total - int(end_offset_bars)
    start_idx = end_idx - int(chunk_bars)
    if start_idx < 0 or end_idx <= 0 or start_idx >= end_idx:
        return False

    warmup = max(1, int(backtest._infer_warmup()), int(backtest._get_window_size()))
    slice_start = max(0, start_idx - warmup)
    chunk_data = full_data.iloc[slice_start:end_idx].copy().reset_index(drop=True)
    if chunk_data.empty:
        return False

    backtest._full_data = chunk_data
    backtest._cursor = max(1, start_idx - slice_start)
    window = backtest._get_window_size()
    data_start = max(0, backtest._cursor - window)
    backtest.data = backtest._full_data.iloc[data_start:backtest._cursor].copy().reset_index(drop=True)

    # Keep 1m data aligned to selected 15m chunk when available.
    full_data_1m = getattr(backtest, "_full_data_1m", None)
    if full_data_1m is not None and len(full_data_1m) > 0 and "timestamp" in full_data_1m.columns:
        ts_start = backtest._full_data["timestamp"].iloc[0]
        ts_end = backtest._full_data["timestamp"].iloc[-1]
        try:
            ts_start = int(float(ts_start))
            ts_end = int(float(ts_end))
            if ts_start < 10**12:
                ts_start *= 1000
            if ts_end < 10**12:
                ts_end *= 1000
            ts_end = ts_end + 15 * 60 * 1000
            ts_1m = pd.to_numeric(full_data_1m["timestamp"], errors="coerce")
            mask = ts_1m.notna() & (ts_1m >= ts_start) & (ts_1m < ts_end)
            backtest._full_data_1m = full_data_1m.loc[mask].copy().reset_index(drop=True)
        except (TypeError, ValueError):
            pass

    # Rebuild HTF cache for chunked data.
    backtest._htf_resampled = None
    backtest._htf_source_indexed = None
    backtest._htf_resample_period = None
    backtest._htf_period_delta = None
    try:
        backtest._build_htf_cache()
    except Exception:
        pass
    return True


def _force_backtest_timestamp_chunk(
    backtest: Any,
    *,
    start_ts_ms: int,
    end_ts_ms: int,
) -> bool:
    """
    Restrict backtest to bars in [start_ts_ms, end_ts_ms).
    Returns False when data is missing for this range.
    """
    if start_ts_ms >= end_ts_ms:
        return False
    full_data = getattr(backtest, "_full_data", None)
    if full_data is None or len(full_data) == 0 or "timestamp" not in full_data.columns:
        return False

    ts = pd.to_numeric(full_data["timestamp"], errors="coerce")
    valid_ts = ts[ts.notna()]
    if valid_ts.empty:
        return False
    # Normalize second-based timestamps to milliseconds if needed.
    if float(valid_ts.iloc[0]) < 10**12:
        ts = ts * 1000.0

    eval_mask = ts.notna() & (ts >= float(start_ts_ms)) & (ts < float(end_ts_ms))
    eval_indices = eval_mask[eval_mask].index
    if len(eval_indices) == 0:
        return False
    start_idx = int(eval_indices[0])
    end_idx = int(eval_indices[-1]) + 1
    if start_idx < 0 or end_idx <= 0 or start_idx >= end_idx:
        return False

    warmup = max(1, int(backtest._infer_warmup()), int(backtest._get_window_size()))
    slice_start = max(0, start_idx - warmup)
    chunk_data = full_data.iloc[slice_start:end_idx].copy().reset_index(drop=True)
    if chunk_data.empty:
        return False

    backtest._full_data = chunk_data
    backtest._cursor = max(1, start_idx - slice_start)
    window = backtest._get_window_size()
    data_start = max(0, backtest._cursor - window)
    backtest.data = backtest._full_data.iloc[data_start:backtest._cursor].copy().reset_index(drop=True)

    # Keep 1m data aligned to selected 15m chunk when available.
    full_data_1m = getattr(backtest, "_full_data_1m", None)
    if full_data_1m is not None and len(full_data_1m) > 0 and "timestamp" in full_data_1m.columns:
        ts_start = backtest._full_data["timestamp"].iloc[0]
        ts_end = backtest._full_data["timestamp"].iloc[-1]
        try:
            ts_start = int(float(ts_start))
            ts_end = int(float(ts_end))
            if ts_start < 10**12:
                ts_start *= 1000
            if ts_end < 10**12:
                ts_end *= 1000
            ts_end = ts_end + 15 * 60 * 1000
            ts_1m = pd.to_numeric(full_data_1m["timestamp"], errors="coerce")
            mask = ts_1m.notna() & (ts_1m >= ts_start) & (ts_1m < ts_end)
            backtest._full_data_1m = full_data_1m.loc[mask].copy().reset_index(drop=True)
        except (TypeError, ValueError):
            pass

    # Rebuild HTF cache for chunked data.
    backtest._htf_resampled = None
    backtest._htf_source_indexed = None
    backtest._htf_resample_period = None
    backtest._htf_period_delta = None
    try:
        backtest._build_htf_cache()
    except Exception:
        pass
    return True


def _worker_mode(
    target: str,
    min_trades_per_day: float,
    tpd_penalty_power: float,
) -> None:
    # Worker inputs must come inline (no files) when running under the optimizer.
    # Fallback to defaults only if inline is missing (manual/dev runs).
    inline = os.getenv("FVG_INPUTS_JSON_INLINE", "").strip()
    if inline:
        try:
            worker_inputs = json.loads(inline)
        except Exception:
            worker_inputs = {}
    else:
        worker_inputs = _load_base_inputs(Path(TARGET_CONFIGS[target]["default_inputs_path"]).resolve())

    from FVG_projectX_bot.backtest import FVG_backtest as bt
    from FVG_projectX_bot.backtest.evaluate_backtest import evaluate_backtest

    target_cfg = TARGET_CONFIGS[target]
    bt.ASSET = str(target_cfg["asset"])
    bt.TIMEFRAME = str(target_cfg["timeframe"])
    bt.INITIAL_BALANCE = float(target_cfg["initial_balance"])
    bt.DATA_CSV_PATH = str(target_cfg["data_path"])
    bt.DATA_1M_CSV_PATH = str(target_cfg["data_path_1m"])
    # Gold/topstep: fixed lot sizing + round-turn fees (matches live ProjectX).
    if target != "binance":
        bt.configure_futures_backtest(contracts_path=(SCRIPT_DIR / "contracts.csv").resolve())
    # Disk IO must be disabled under optimization.
    # `FVG_Backtest` respects FVG_NO_DISK_IO for directory creation and trade CSV writes.
    os.environ["FVG_NO_DISK_IO"] = os.getenv("FVG_NO_DISK_IO", "1")
    trial_runtime_dir = None

    class _ChunkInsufficientMargin(Exception):
        pass

    original_place_order = bt.BacktestOrder.place_order

    def _place_order_chunk_guard(order_self, *args, **kwargs):
        result = original_place_order(order_self, *args, **kwargs)
        if isinstance(result, dict) and not bool(result.get("success", False)):
            msg = str(result.get("message", ""))
            if "insufficient margin" in msg.lower():
                raise _ChunkInsufficientMargin(msg or "insufficient margin")
        return result

    bt.BacktestOrder.place_order = _place_order_chunk_guard

    # Binance: fixed regime windows. Gold/topstep: one full 15m/1m overlap run.
    bt.USE_LAST_QUARTER_DATA = False
    chunk_specs: list[dict[str, Any]] = []
    if target == "binance":
        for label, start_day, end_day in BINANCE_REGIME_WINDOWS:
            start_dt = datetime.fromisoformat(start_day).replace(tzinfo=timezone.utc)
            end_dt_exclusive = (
                datetime.fromisoformat(end_day).replace(tzinfo=timezone.utc) + timedelta(days=1)
            )
            chunk_specs.append(
                {
                    "label": label,
                    "start_ts_ms": int(start_dt.timestamp() * 1000),
                    "end_ts_ms": int(end_dt_exclusive.timestamp() * 1000),
                }
            )
    else:
        chunk_specs.append({"label": "full_1m_overlap", "full_overlap": True})

    if len(chunk_specs) < 1:
        raise RuntimeError("Not enough evaluation chunks configured.")

    chunk_metrics: list[dict[str, float]] = []

    try:
        for chunk_idx, chunk_spec in enumerate(chunk_specs, start=1):
            backtest = bt.FVG_Backtest(
                asset=bt.ASSET,
                timeframe=bt.TIMEFRAME,
                initial_balance=bt.INITIAL_BALANCE,
                data_path=bt.DATA_CSV_PATH,
                start_timestamp=None,
                pyramiding_mode=bt.PYRAMIDING_MODE,
                data_path_1m=bt.DATA_1M_CSV_PATH,
            )
            if target != "binance" and not backtest.restrict_to_1m_overlap():
                print(
                    f"[backtest] chunk={chunk_idx}/{len(chunk_specs)} "
                    f"label={chunk_spec.get('label', 'unknown')} skipped (no 15m/1m overlap).",
                    flush=True,
                )
                continue
            # Avoid persisting per-chunk folders: write any temporary artifacts directly
            # into the runtime dir (which should be a temp directory in parent mode).
            # With disk IO disabled, these filenames are unused.
            if chunk_spec.get("full_overlap"):
                ok_chunk = True
            elif "start_ts_ms" in chunk_spec and "end_ts_ms" in chunk_spec:
                ok_chunk = _force_backtest_timestamp_chunk(
                    backtest,
                    start_ts_ms=int(chunk_spec["start_ts_ms"]),
                    end_ts_ms=int(chunk_spec["end_ts_ms"]),
                )
            else:
                ok_chunk = _force_backtest_spaced_month_chunk(
                    backtest,
                    chunk_bars=int(chunk_spec["chunk_bars"]),
                    end_offset_bars=int(chunk_spec["end_offset_bars"]),
                )
            if not ok_chunk:
                print(
                    f"[backtest] chunk={chunk_idx}/{len(chunk_specs)} "
                    f"label={chunk_spec.get('label', 'unknown')} skipped (insufficient data).",
                    flush=True,
                )
                continue

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
                            f"[backtest] chunk={chunk_idx}/{len(chunk_specs)} "
                            f"label={chunk_spec.get('label', 'unknown')} "
                            f"{pct:5.1f}% ({tested_cursor}/{tested_bars})",
                            flush=True,
                        )
                        last_pct = pct
                    time.sleep(0.5)

            progress_thread = threading.Thread(target=_progress_reporter, daemon=True)
            progress_thread.start()
            chunk_margin_failed = False
            try:
                backtest.run()
            except _ChunkInsufficientMargin:
                chunk_margin_failed = True
                # Backtest run exits early on chunk margin guard; force-close and
                # record any still-open orders so chunk trades CSV has final state.
                try:
                    backtest._close_open_order_at_end()
                except Exception:
                    pass
            finally:
                stop_progress.set()
                progress_thread.join(timeout=1.0)

            if chunk_margin_failed:
                print(
                    f"[backtest] chunk={chunk_idx}/{len(chunk_specs)} "
                    f"label={chunk_spec.get('label', 'unknown')} "
                    "aborted early due to insufficient margin; continuing.",
                    flush=True,
                )
                continue

            # Skip chunks that produced no trades (do not count them in objective).
            if not getattr(backtest, "trades", None):
                print(
                    f"[backtest] chunk={chunk_idx}/{len(chunk_specs)} "
                    f"label={chunk_spec.get('label', 'unknown')} "
                    "skipped (no trades).",
                    flush=True,
                )
                # Clean temporary trade file if created.
                try:
                    if os.path.exists(backtest.trades_csv_path):
                        os.remove(backtest.trades_csv_path)
                except Exception:
                    pass
                continue

            # No-disk evaluation: metrics from in-memory trades + chunked price
            # (after restrict_to_1m_overlap / chunk trim), not the full on-disk CSV.
            tick_size = float(target_cfg.get("tick_size", getattr(backtest, "tick_size", 0.0) or 0.0) or 0.0)
            tick_value = float(target_cfg.get("tick_value", getattr(backtest, "tick_value", 1.0) or 1.0) or 1.0)
            summary = evaluate_backtest(
                trades_df=pd.DataFrame(backtest.trades),
                price_df=getattr(backtest, "_full_data", None),
                start_equity=bt.INITIAL_BALANCE,
                tick_size=tick_size,
                tick_value=tick_value,
                benchmark_label=str(target_cfg.get("benchmark_label", "Buy & Hold")),
            )

            chunk_pnl = float(summary.get("total_pnl_original", 0.0))
            chunk_dd = float(summary.get("max_drawdown", 0.0))
            chunk_num_trades = int(summary.get("num_trades", 0))
            chunk_days = float(summary.get("days_of_backtest", 0.0))
            chunk_avg_win = float(summary.get("average_winning_trade", 0.0))
            chunk_avg_loss = float(summary.get("average_losing_trade", 0.0))
            chunk_tpd = (chunk_num_trades / chunk_days) if chunk_days > 0 else 0.0

            trades_df = pd.DataFrame(backtest.trades)
            if trades_df.empty or "pnl" not in trades_df.columns:
                chunk_win_rate = 0.0
            else:
                pnl_series = pd.to_numeric(trades_df["pnl"], errors="coerce").dropna()
                chunk_win_rate = float((pnl_series > 0).mean() * 100.0) if not pnl_series.empty else 0.0

            chunk_metrics.append(
                {
                    "pnl": chunk_pnl,
                    "max_drawdown": chunk_dd,
                    "num_trades": float(chunk_num_trades),
                    "trades_per_day": chunk_tpd,
                    "avg_win": chunk_avg_win,
                    "avg_loss": chunk_avg_loss,
                    "win_rate": chunk_win_rate,
                }
            )
    finally:
        bt.BacktestOrder.place_order = original_place_order

    if not chunk_metrics:
        # No-trade parameter set: return a payload that the parent can score,
        # but parent should skip persisting this row.
        payload = {
            "objective": -9_999_999.0,
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
        print(f"{RESULT_MARKER}{json.dumps(payload)}")
        return

    ending_pnl = float(sum(m["pnl"] for m in chunk_metrics) / len(chunk_metrics))
    max_drawdown = float(sum(m["max_drawdown"] for m in chunk_metrics) / len(chunk_metrics))
    num_trades = int(round(sum(m["num_trades"] for m in chunk_metrics) / len(chunk_metrics)))
    trades_per_day = float(sum(m["trades_per_day"] for m in chunk_metrics) / len(chunk_metrics))
    avg_win = float(sum(m["avg_win"] for m in chunk_metrics) / len(chunk_metrics))
    avg_loss = float(sum(m["avg_loss"] for m in chunk_metrics) / len(chunk_metrics))
    win_rate = float(sum(m["win_rate"] for m in chunk_metrics) / len(chunk_metrics))

    # Ratio is now profit-to-drawdown (higher is better).
    # Use epsilon to avoid division by zero when drawdown is ~0.
    ratio = (ending_pnl / max(max_drawdown, MAXIMIZE_EPS)) if ending_pnl > 0 else -math.inf
    tpd_penalty_factor = 1.0
    if trades_per_day < min_trades_per_day:
        # Smoothly increase penalty as activity drops below threshold.
        # Example with default power=2:
        #   0.4 -> (0.5/0.4)^2 = 1.56 (mild)
        #   0.04 -> (0.5/0.04)^2 = 156.25 (huge)
        safe_tpd = max(trades_per_day, 1e-9)
        tpd_penalty_factor = (min_trades_per_day / safe_tpd) ** tpd_penalty_power

    # Objective: higher profit-to-drawdown ratio is better.
    # Penalize non-profitable / zero-profit runs heavily (very low objective).
    if ending_pnl <= 0:
        objective = -1_000_000.0 - abs(ending_pnl) - max_drawdown
    else:
        # Keep penalty semantics: penalty_factor >= 1 when activity is too low,
        # so divide by it to reduce the score for inactive strategies.
        objective = ratio / max(tpd_penalty_factor, 1e-9)

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
    parser.add_argument(
        "--random-search",
        action="store_true",
        help=(
            "Run plain random parameter search (no Optuna optimizer). "
            "In this mode, phase_idx and trial_in_phase are recorded as -1."
        ),
    )
    parser.add_argument(
        "--random-warmup-trials",
        type=int,
        default=30,
        help=(
            "In optimizer mode (default), run this many purely random trials first, "
            "then switch to Optuna TPE. Set to 0 to disable warmup."
        ),
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--child-run", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--restart-on-stagnation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Restart optimizer phase after N consecutive non-improving completed trials.",
    )
    parser.add_argument(
        "--stagnation-window",
        type=int,
        default=12,
        help=(
            "Consecutive completed non-improvements required before phase restart "
            "(after min phase trial gate)."
        ),
    )
    parser.add_argument(
        "--stagnation-objective-abs-tol",
        type=float,
        default=0.02,
        help="Absolute tolerance for objective similarity across stagnation window.",
    )
    parser.add_argument(
        "--stagnation-objective-rel-tol",
        type=float,
        default=0.02,
        help="Relative tolerance for objective similarity across stagnation window.",
    )
    parser.add_argument(
        "--stagnation-param-distance",
        type=float,
        default=0.12,
        help="Deprecated; retained for CLI compatibility (unused).",
    )
    args = parser.parse_args()

    targets = _resolve_targets(args.target)

    if args.worker:
        if len(targets) != 1:
            raise ValueError("Worker mode requires a single --target (binance or gold).")
        _worker_mode(
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
                "--seed",
                str(args.seed),
                "--n-jobs",
                str(args.n_jobs),
                "--child-run",
                "--min-trades-per-day",
                str(args.min_trades_per_day),
                "--tpd-penalty-power",
                str(args.tpd_penalty_power),
                "--stagnation-window",
                str(args.stagnation_window),
                "--stagnation-objective-abs-tol",
                str(args.stagnation_objective_abs_tol),
                "--stagnation-objective-rel-tol",
                str(args.stagnation_objective_rel_tol),
                "--stagnation-param-distance",
                str(args.stagnation_param_distance),
            ]
            cmd.append("--restart-on-stagnation" if args.restart_on_stagnation else "--no-restart-on-stagnation")
            if args.timeout_sec is not None:
                cmd.extend(["--timeout-sec", str(args.timeout_sec)])
            if args.random_search:
                cmd.append("--random-search")
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
    max_csv_trial = -1
    for record in trial_records:
        trial_val = record.get("trial")
        try:
            parsed = int(float(trial_val))
        except (TypeError, ValueError):
            continue
        if parsed > max_csv_trial:
            max_csv_trial = parsed
    trial_counter = max_csv_trial + 1
    trial_counter_lock = threading.Lock()
    max_existing_phase = -1
    for record in trial_records:
        phase_val = record.get("phase_idx")
        try:
            parsed_phase = int(float(phase_val))
        except (TypeError, ValueError):
            continue
        if parsed_phase > max_existing_phase:
            max_existing_phase = parsed_phase

    # Continue with a fresh phase index after the highest persisted phase,
    # except random-search mode where phase is intentionally marked as -1.
    phase_idx = -1 if args.random_search else (max_existing_phase + 1 if max_existing_phase >= 0 else 0)
    phase_seed = int((int(args.seed) + max(phase_idx, 0) * 104_729) % (2**31 - 1))
    if phase_seed <= 0:
        phase_seed = int(args.seed)
    phase_restarts = 0
    phase_trial_counter = 0
    optimization_start_monotonic = time.monotonic()

    def objective(trial: "optuna.trial.Trial") -> float:
        nonlocal trial_counter, phase_trial_counter
        with trial_counter_lock:
            global_trial_idx = trial_counter
            trial_counter += 1
            if phase_idx >= 0:
                trial_in_phase = phase_trial_counter + 1
                phase_trial_counter += 1
            else:
                trial_in_phase = -1

        trial_params = _suggest_params(trial, base_inputs)
        merged_inputs = dict(base_inputs)
        merged_inputs.update(trial_params)

        try:
            result = _run_trial_subprocess(
                trial_inputs=merged_inputs,
                trial_idx=global_trial_idx,
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
                "objective": -9_999_999.0,
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

        # Don't persist no-trade trials (but still return objective to Optuna).
        try:
            if int(result.get("num_trades", 0)) <= 0:
                return float(result["objective"])
        except Exception:
            return float(result["objective"])

        trial.set_user_attr("ending_pnl", float(result["ending_pnl"]))
        trial.set_user_attr("max_drawdown", float(result["max_drawdown"]))
        trial.set_user_attr("num_trades", int(result["num_trades"]))
        trial.set_user_attr("ratio", result.get("ratio"))
        trial.set_user_attr("tpd_penalty_factor", float(result["tpd_penalty_factor"]))
        trial.set_user_attr("trades_per_day", float(result["trades_per_day"]))
        trial.set_user_attr("avg_win", float(result["avg_win"]))
        trial.set_user_attr("avg_loss", float(result["avg_loss"]))
        trial.set_user_attr("win_rate", float(result["win_rate"]))
        trial.set_user_attr("global_trial", int(global_trial_idx))
        trial.set_user_attr("phase_idx", int(phase_idx))
        trial.set_user_attr("trial_in_phase", int(trial_in_phase))

        record = {
            "run_id": run_id,
            "trial": global_trial_idx,
            "phase_idx": phase_idx,
            "trial_in_phase": int(trial_in_phase),
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

    if args.random_search:
        rng = random.Random(int(args.seed))
        total_trials = int(args.trials)
        n_jobs = max(1, int(args.n_jobs))
        scheduled = 0

        def _run_random_trial(global_trial_idx: int, sampled_params: dict[str, Any]) -> None:
            merged_inputs = dict(base_inputs)
            merged_inputs.update(sampled_params)
            merged_inputs["RUNTIME_SUBDIR"] = (
                f"optimization_results/{target}/trial_runs/trial_{global_trial_idx}"
            )
            try:
                result = _run_trial_subprocess(
                    trial_inputs=merged_inputs,
                    trial_idx=global_trial_idx,
                    target=target,
                    min_trades_per_day=args.min_trades_per_day,
                    tpd_penalty_power=args.tpd_penalty_power,
                )
                failed = False
                fail_reason = ""
            except Exception as exc:
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

            record = {
                "run_id": run_id,
                "trial": global_trial_idx,
                "phase_idx": -1,
                "trial_in_phase": -1,
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
            record.update(merged_inputs)
            with records_lock:
                trial_records.append(record)
                _write_results_snapshot(trial_records, results_csv)

        with concurrent.futures.ThreadPoolExecutor(max_workers=n_jobs) as executor:
            in_flight: set[concurrent.futures.Future] = set()
            while scheduled < total_trials or in_flight:
                while scheduled < total_trials and len(in_flight) < n_jobs:
                    if args.timeout_sec is not None:
                        elapsed = time.monotonic() - optimization_start_monotonic
                        if elapsed >= float(args.timeout_sec):
                            scheduled = total_trials
                            break
                    with trial_counter_lock:
                        global_trial_idx = trial_counter
                        trial_counter += 1
                    sampled_params = _sample_random_params(base_inputs, rng)
                    fut = executor.submit(_run_random_trial, global_trial_idx, sampled_params)
                    in_flight.add(fut)
                    scheduled += 1
                if not in_flight:
                    continue
                done, _ = concurrent.futures.wait(
                    in_flight,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                for fut in done:
                    in_flight.remove(fut)
                    fut.result()
    else:
        if optuna is None:
            raise SystemExit(
                "optuna is required for optimizer mode. Install with: pip install optuna "
                "or run with --random-search."
            )
        # Hybrid: random warmup, then Optuna TPE.
        total_trials = int(args.trials)
        warmup_trials = max(0, int(args.random_warmup_trials))
        warmup_trials = min(warmup_trials, max(0, total_trials - trial_counter))

        # Generate warmup params once per invocation (reproducible).
        rng = random.Random(int(args.seed) + 99_991 + max(0, int(phase_idx)) * 1_000_003)
        warmup_param_sets: list[dict[str, Any]] = []
        for _ in range(warmup_trials):
            warmup_param_sets.append(_sample_random_params(base_inputs, rng))

        while trial_counter < total_trials:
            timeout_remaining: float | None = None
            if args.timeout_sec is not None:
                elapsed = time.monotonic() - optimization_start_monotonic
                timeout_remaining = float(args.timeout_sec) - elapsed
                if timeout_remaining <= 0:
                    break

            sampler = optuna.samplers.TPESampler(seed=phase_seed)
            study = optuna.create_study(direction="maximize", sampler=sampler)
            restart_triggered = False
            restart_reason = ""

            def _stagnation_callback(st: "optuna.study.Study", _: "optuna.trial.FrozenTrial") -> None:
                nonlocal restart_triggered, restart_reason
                if not args.restart_on_stagnation:
                    return
                window = max(3, int(args.stagnation_window))
                completed = [
                    t for t in st.trials if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None
                ]
                if len(completed) < max(PHASE_MIN_COMPLETED_TRIALS, window):
                    return
                non_improve_streak, best_value = _count_consecutive_non_improvements(
                    completed,
                    objective_abs_tol=float(args.stagnation_objective_abs_tol),
                    objective_rel_tol=float(args.stagnation_objective_rel_tol),
                )
                if non_improve_streak < window:
                    return

                if len(completed) < (2 * window):
                    return

                prior_values = [float(t.value) for t in completed[-2 * window : -window]]
                recent_values = [float(t.value) for t in completed[-window:]]
                prior_median = float(statistics.median(prior_values))
                recent_median = float(statistics.median(recent_values))
                if prior_median != 0.0:
                    # For maximize objective, improvement is positive when recent median is higher.
                    median_improvement = (recent_median - prior_median) / abs(prior_median)
                else:
                    median_improvement = 0.0 if recent_median <= prior_median else 1.0

                # Restart only when the recent window no longer shows material progress.
                if median_improvement >= PHASE_MEDIAN_IMPROVEMENT_MIN:
                    return

                restart_triggered = True
                restart_reason = (
                    f"{non_improve_streak} consecutive non-improvements "
                    f"(best objective={best_value:.6f}); "
                    f"median improvement over last {window} vs previous {window}="
                    f"{median_improvement * 100:.2f}%"
                )
                st.stop()

            n_trials_remaining = int(args.trials) - trial_counter

            # Enqueue random warmup trials at the start of the very first phase (per invocation).
            # These will be consumed first by Optuna, but still evaluated through the normal objective().
            if warmup_param_sets:
                for params in warmup_param_sets:
                    study.enqueue_trial(params)
                # Ensure we only enqueue once (even if phases restart later).
                warmup_param_sets = []

            study.optimize(
                objective,
                n_trials=n_trials_remaining,
                timeout=timeout_remaining,
                n_jobs=max(1, int(args.n_jobs)),
                show_progress_bar=False,
                callbacks=[_stagnation_callback],
            )

            if not restart_triggered:
                break

            phase_restarts += 1
            phase_idx += 1
            phase_trial_counter = 0
            # Equivalent of script restart: fresh sampler seed => fresh random startup exploration.
            phase_seed = int((time.time_ns() + phase_idx * 104_729) % (2**31 - 1))
            print(
                f"↻ Restarting optimization phase {phase_idx} with seed={phase_seed} "
                f"because {restart_reason}."
            )

    with records_lock:
        _write_results_snapshot(trial_records, results_csv)

    complete_records = [
        r for r in trial_records
        if not bool(r.get("failed", False)) and r.get("objective") is not None
    ]
    if not complete_records:
        raise RuntimeError("No completed trials.")

    best_record = max(complete_records, key=lambda r: float(r["objective"]))

    print("\n✅ Optimization finished")
    print(f"Target: {target}")
    print(f"Trials: {len(complete_records)}")
    print(f"Phase restarts: {phase_restarts}")
    print(f"Best objective (ending_pnl / max_drawdown, penalized): {float(best_record['objective']):.6f}")
    print(f"Best ending_pnl: {best_record.get('pnl')}")
    print(f"Best max_drawdown: {best_record.get('max_dd')}")
    print(f"Saved trial table: {results_csv}")


if __name__ == "__main__":
    main()
