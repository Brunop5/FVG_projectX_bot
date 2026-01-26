#!/usr/bin/env python3
"""
Clean futures CSV data:
1) Drop any row that has a negative value in any numeric column.
2) If multiple rows share the same minute, keep only the first.

By default this updates files in-place.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


TARGET_CSV = "data/MGCG6/1mdata_gold.csv"  # Set to a file path string to process only one CSV (F5-friendly)
READ_CSV_KWARGS = {"sep": None, "engine": "python"}  # auto-detect delimiter
TIME_COL_CANDIDATES = {"timestamp", "date", "datetime", "time", "<DATE>", "<TIME>", "ts_event"}
PRICE_COLS = {"open", "high", "low", "close"}
PRICE_MIN_THRESHOLD = 100.0


def _find_minute_key(df: pd.DataFrame) -> pd.Series:
    """Return a Series representing the minute for each row."""
    col_map = {c.lower(): c for c in df.columns}
    if "timestamp" in col_map:
        ts = df[col_map["timestamp"]]
        if pd.api.types.is_numeric_dtype(ts):
            ts = pd.to_datetime(ts, unit="ms", utc=True, errors="coerce")
        else:
            ts = pd.to_datetime(ts, utc=True, errors="coerce")
        return ts.dt.floor("min")

    if "datetime" in col_map:
        ts = pd.to_datetime(df[col_map["datetime"]], utc=True, errors="coerce")
        return ts.dt.floor("min")

    for candidate in ("iso_datetime", "datetime_iso", "date_time", "ts_event"):
        if candidate in col_map:
            ts = pd.to_datetime(df[col_map[candidate]], utc=True, errors="coerce")
            return ts.dt.floor("min")

    if "date" in col_map and "time" in col_map:
        ts = pd.to_datetime(
            df[col_map["date"]].astype(str) + " " + df[col_map["time"]].astype(str),
            utc=True,
            errors="coerce",
        )
        return ts.dt.floor("min")

    if "<date>" in col_map and "<time>" in col_map:
        ts = pd.to_datetime(
            df[col_map["<date>"]].astype(str) + " " + df[col_map["<time>"]].astype(str),
            format="%Y.%m.%d %H:%M:%S",
            utc=True,
            errors="coerce",
        )
        return ts.dt.floor("min")

    raise ValueError("No recognizable timestamp columns found for minute de-duplication.")


def _drop_price_below_threshold(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    price_cols = [c for c in df.columns if c.lower() in PRICE_COLS]
    if not price_cols:
        return df
    price_df = df[price_cols].apply(pd.to_numeric, errors="coerce")
    bad_mask = (price_df < threshold).any(axis=1)
    return df.loc[~bad_mask].copy()


def _get_symbol_col(df: pd.DataFrame) -> str | None:
    col_map = {c.lower(): c for c in df.columns}
    return col_map.get("symbol")


def _dedupe_by_minute_keep_symbol_continuity(df: pd.DataFrame) -> pd.DataFrame:
    minute_key = _find_minute_key(df)
    df = df.copy()
    df["_minute_key"] = minute_key
    df = df[df["_minute_key"].notna()].copy()
    if df.empty:
        return df

    symbol_col = _get_symbol_col(df)
    # Preserve original order within each minute
    df["_orig_index"] = df.index
    df = df.sort_values(["_minute_key", "_orig_index"])

    selected_indices = []
    current_symbol = None

    for _, group in df.groupby("_minute_key", sort=True):
        if symbol_col and current_symbol is not None and current_symbol in set(group[symbol_col]):
            row = group[group[symbol_col] == current_symbol].iloc[0]
        else:
            row = group.iloc[0]
            if symbol_col:
                current_symbol = row[symbol_col]
        selected_indices.append(row["_orig_index"])

    return (
        df.loc[selected_indices]
        .sort_values("_orig_index")
        .drop(columns=["_minute_key", "_orig_index"])
        .copy()
    )


def _filter_every_15_minutes(df: pd.DataFrame) -> pd.DataFrame:
    minute_key = _find_minute_key(df)
    mask = minute_key.notna() & (minute_key.dt.minute % 15 == 0)
    return df.loc[mask].copy()


def clean_file(csv_path: Path) -> None:
    df = pd.read_csv(csv_path, **READ_CSV_KWARGS)
    original_rows = len(df)

    # Remove rows with any price below threshold
    df = _drop_price_below_threshold(df, PRICE_MIN_THRESHOLD)
    after_price = len(df)

    # Remove duplicates so there is only one row per minute, preserving symbol continuity
    df = _dedupe_by_minute_keep_symbol_continuity(df)
    after_dedupe = len(df)

    # Save cleaned data back to original file
    tmp_path = csv_path.with_suffix(csv_path.suffix + ".tmp")
    df.to_csv(tmp_path, index=False)
    tmp_path.replace(csv_path)

    # Create 15-minute sampled CSV (keep rows at :00, :15, :30, :45)
    df_15m = _filter_every_15_minutes(df)
    out_15m = csv_path.with_name(f"{csv_path.stem}_15min{csv_path.suffix}")
    df_15m.to_csv(out_15m, index=False)

    print(
        f"{csv_path.name}: {original_rows} -> {after_price} (prices >= {PRICE_MIN_THRESHOLD}) "
        f"-> {after_dedupe} (deduped by minute) | 15m rows: {len(df_15m)} "
        f"| saved: {out_15m.name}"
    )


def main() -> None:
    # F5-friendly: no CLI args required.
    if TARGET_CSV:
        csv_path = Path(TARGET_CSV).resolve()
        if not csv_path.exists():
            raise SystemExit(f"CSV file not found: {csv_path}")
        clean_file(csv_path)
        return

    input_dir = Path("data").resolve()
    if not input_dir.exists():
        raise SystemExit(f"Input directory not found: {input_dir}")

    pattern = "**/*.csv"
    csv_files = sorted(input_dir.glob(pattern))
    if not csv_files:
        raise SystemExit(f"No CSV files found under: {input_dir}")

    for csv_path in csv_files:
        try:
            clean_file(csv_path)
        except Exception as exc:
            print(f"⚠️  Failed to clean {csv_path}: {exc}")


if __name__ == "__main__":
    main()

