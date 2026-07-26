"""Append Topstep gold OHLCV into shared backtest datasets (gold only)."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
BACKTEST_DATA = PACKAGE_ROOT / "backtest" / "data"

GOLD_15M_ARCHIVE = BACKTEST_DATA / "manually_fetched_gold_15m.csv"
GOLD_1M_ARCHIVE = BACKTEST_DATA / "manually_fetched_gold_1min.csv"

OHLCV_COLS = ["timestamp", "open", "high", "low", "close", "volume"]
ARCHIVE_COLS = ["contract", *OHLCV_COLS]


def _normalize_bars(df: pd.DataFrame, contract: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=ARCHIVE_COLS)

    out = df.copy()
    if "timestamp" not in out.columns:
        raise ValueError("Expected OHLCV frame with a timestamp column")

    ts = out["timestamp"]
    if pd.api.types.is_datetime64_any_dtype(ts):
        out["timestamp"] = (pd.to_datetime(ts, utc=True).astype("int64") // 10**9).astype(int)
    else:
        numeric = pd.to_numeric(ts, errors="coerce")
        if numeric.notna().any() and float(numeric.max()) > 10**12:
            out["timestamp"] = (numeric // 1000).astype("int64")
        else:
            out["timestamp"] = numeric.astype("int64")

    for col in OHLCV_COLS[1:]:
        if col not in out.columns:
            raise ValueError(f"Missing column {col!r} in gold OHLCV data")
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.dropna(subset=["timestamp", "open", "high", "low", "close"])
    out["contract"] = str(contract)
    out["timestamp"] = out["timestamp"].astype(int)
    out = out[ARCHIVE_COLS].drop_duplicates(subset=["contract", "timestamp"], keep="last")
    return out.sort_values("timestamp").reset_index(drop=True)


def _load_archive(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=ARCHIVE_COLS)
    df = pd.read_csv(path)
    if df.empty:
        return pd.DataFrame(columns=ARCHIVE_COLS)
    if "contract" not in df.columns:
        df["contract"] = "MGCQ6"
    for col in ARCHIVE_COLS:
        if col not in df.columns:
            raise ValueError(f"Archive {path} is missing column {col!r}")
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["timestamp"])
    df["timestamp"] = df["timestamp"].astype(int)
    df["contract"] = df["contract"].astype(str)
    return df[ARCHIVE_COLS].drop_duplicates(subset=["contract", "timestamp"], keep="last")


def _span_label(df: pd.DataFrame) -> str:
    if df.empty:
        return "empty"
    start = datetime.fromtimestamp(int(df["timestamp"].iloc[0]), tz=timezone.utc)
    end = datetime.fromtimestamp(int(df["timestamp"].iloc[-1]), tz=timezone.utc)
    return f"{start.isoformat()} -> {end.isoformat()}"


def merge_gold_bars(
    df: pd.DataFrame,
    *,
    timeframe: str,
    contract: str = "MGCQ6",
    archive_path: Path | None = None,
) -> Path:
    """
    Merge ``df`` into the shared gold archive for ``timeframe`` (``15m`` or ``1m``).
    Creates the file when missing; dedupes on (contract, timestamp).
    """
    tf = timeframe.strip().lower().replace("min", "m")
    if tf in {"15", "15m"}:
        path = archive_path or GOLD_15M_ARCHIVE
    elif tf in {"1", "1m"}:
        path = archive_path or GOLD_1M_ARCHIVE
    else:
        raise ValueError(f"Unsupported gold archive timeframe: {timeframe!r}")

    incoming = _normalize_bars(df, contract)
    if incoming.empty:
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    existing = _load_archive(path)
    before_n = len(existing)
    before_span = _span_label(existing)

    merged = pd.concat([existing, incoming], ignore_index=True)
    merged = merged.drop_duplicates(subset=["contract", "timestamp"], keep="last")
    merged = merged.sort_values(["contract", "timestamp"]).reset_index(drop=True)
    merged.to_csv(path, index=False)

    added = len(merged) - before_n
    print(
        f"Gold archive {path.name}: {before_n} -> {len(merged)} bars "
        f"(+{added} new) span {before_span} | merged {_span_label(merged)}",
        flush=True,
    )
    return path


def archive_gold_topstep_csvs(
    path_15m: Path | str,
    path_1m: Path | str,
    *,
    contract: str = "MGCQ6",
) -> tuple[Path, Path]:
    """Load Topstep gold CSVs and merge into the global 15m / 1m archives."""
    p15 = Path(path_15m)
    p1 = Path(path_1m)
    if not p15.exists() or not p1.exists():
        raise FileNotFoundError(f"Missing gold CSV(s): {p15} / {p1}")

    df15 = pd.read_csv(p15)
    df1 = pd.read_csv(p1)
    out15 = merge_gold_bars(df15, timeframe="15m", contract=contract)
    out1 = merge_gold_bars(df1, timeframe="1m", contract=contract)
    return out15, out1
