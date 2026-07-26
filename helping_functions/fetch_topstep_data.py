"""Fetch Topstep/ProjectX OHLCV for the configured gold contract into backtest/data.

Supports a single short pull (legacy) or chunked lookback refresh used for
live-vs-backtest comparisons (enough 15m + 1m history through now).
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from FVG_projectX_bot.projectX.projectx_api_functions import (
    DEFAULT_TOPSTEPX_TIMEOUT,
    _map_timeframe_to_unit,
    login_to_api,
    topstepx_post,
)

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
BACKTEST_DATA = PACKAGE_ROOT / "backtest" / "data"


def _contract_out_dir(asset: str) -> Path:
    # CON.F.US.MGC.Q26 -> MGCQ6 (matches optimizer / run_topstep_top paths).
    month = asset.rsplit(".", 1)[-1]  # e.g. Q26
    if month.upper().startswith("Q") and len(month) >= 2:
        folder = f"MGCQ{month[-1]}"
    else:
        folder = f"MGC{month}"
    return BACKTEST_DATA / folder


def _bars_to_df(bars: list[dict]) -> pd.DataFrame:
    if not bars:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    rows = []
    for b in bars:
        t = b.get("t") or b.get("timestamp")
        ts = pd.to_datetime(t, utc=True, errors="coerce")
        if pd.isna(ts):
            continue
        rows.append(
            {
                "timestamp": int(ts.timestamp()),
                "open": float(b.get("o", b.get("open"))),
                "high": float(b.get("h", b.get("high"))),
                "low": float(b.get("l", b.get("low"))),
                "close": float(b.get("c", b.get("close"))),
                "volume": float(b.get("v", b.get("volume", 0)) or 0),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    return pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)


def fetch_window(
    *,
    asset: str,
    timeframe: str,
    token: str,
    start: datetime,
    end: datetime,
    limit: int = 20000,
) -> pd.DataFrame:
    unit, unit_number = _map_timeframe_to_unit(timeframe)
    headers = {
        "accept": "text/plain",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
    }
    payload = {
        "contractId": asset,
        "live": False,
        "startTime": start.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "endTime": end.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "unit": unit,
        "unitNumber": unit_number,
        "limit": limit,
        "includePartialBar": False,
    }
    resp = topstepx_post(
        "https://api.topstepx.com/api/History/retrieveBars",
        headers=headers,
        payload=payload,
        timeout=DEFAULT_TOPSTEPX_TIMEOUT,
    )
    resp.raise_for_status()
    body = resp.json()
    return _bars_to_df(body.get("bars") or [])


def fetch_chunked(
    *,
    asset: str,
    timeframe: str,
    token: str,
    lookback_days: int,
    step_days: int,
    limit: int = 20000,
) -> pd.DataFrame:
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=lookback_days)
    frames: list[pd.DataFrame] = []
    cur_end = end
    empty_streak = 0
    while cur_end > start and empty_streak < 3:
        cur_start = max(start, cur_end - timedelta(days=step_days))
        print(f"fetch {timeframe} {cur_start.isoformat()} -> {cur_end.isoformat()}", flush=True)
        part = fetch_window(
            asset=asset,
            timeframe=timeframe,
            token=token,
            start=cur_start,
            end=cur_end,
            limit=limit,
        )
        print(f"  got {len(part)}", flush=True)
        if part.empty:
            empty_streak += 1
        else:
            empty_streak = 0
            frames.append(part)
        if cur_start <= start:
            break
        cur_end = cur_start - timedelta(seconds=1)
    if not frames:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    df = pd.concat(frames, ignore_index=True)
    return df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)


def refresh_mgc_data(
    *,
    lookback_days: int = 70,
    out_dir: Path | None = None,
    asset: str | None = None,
) -> tuple[Path, Path]:
    """
    Chunk-fetch 15m + 1m gold history through now into backtest/data/MGCQ6/.
    Returns (path_15m, path_1m).
    """
    load_dotenv(PACKAGE_ROOT / ".env")
    key = os.getenv("PROJECTX_API_KEY") or os.getenv("API_KEY")
    login = os.getenv("PROJECTX_USERNAME") or os.getenv("USERNAME")
    asset = asset or os.getenv("PROJECTX_ASSET", "CON.F.US.MGC.Q26")
    if not key or not login:
        raise RuntimeError("PROJECTX_USERNAME / PROJECTX_API_KEY required in .env")

    dest = out_dir or _contract_out_dir(asset)
    # Prefer existing MGCQ6 used by optimizer if present
    preferred = BACKTEST_DATA / "MGCQ6"
    if out_dir is None and preferred.exists():
        dest = preferred
    dest.mkdir(parents=True, exist_ok=True)

    token = login_to_api(login, key)["token"]
    print(f"Refreshing {asset} -> {dest} (lookback={lookback_days}d)")

    df15 = fetch_chunked(
        asset=asset,
        timeframe="15min",
        token=token,
        lookback_days=lookback_days,
        step_days=20,
    )
    df1 = fetch_chunked(
        asset=asset,
        timeframe="1min",
        token=token,
        lookback_days=lookback_days,
        step_days=8,
    )
    if df15.empty or df1.empty:
        raise RuntimeError(f"Refresh failed: 15m={len(df15)} 1m={len(df1)}")

    def _span(df: pd.DataFrame) -> str:
        a = datetime.fromtimestamp(int(df["timestamp"].iloc[0]), tz=timezone.utc)
        b = datetime.fromtimestamp(int(df["timestamp"].iloc[-1]), tz=timezone.utc)
        return f"{a.isoformat()} -> {b.isoformat()}"

    path15 = dest / "topstep_15min.csv"
    path1 = dest / "topstep_1min.csv"
    df15.to_csv(path15, index=False)
    df1.to_csv(path1, index=False)
    print(f"15m: {len(df15)} bars {_span(df15)} -> {path15}")
    print(f"1m:  {len(df1)} bars {_span(df1)} -> {path1}")
    return path15, path1


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fetch Topstep MGC OHLCV into backtest/data.")
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Chunked lookback refresh through now (recommended for live comparisons).",
    )
    parser.add_argument("--lookback-days", type=int, default=70, help="Days of history for --refresh.")
    parser.add_argument("--out-dir", type=str, default=None, help="Override output directory.")
    args = parser.parse_args()

    load_dotenv(PACKAGE_ROOT / ".env")
    out = Path(args.out_dir).resolve() if args.out_dir else None

    if args.refresh:
        refresh_mgc_data(lookback_days=args.lookback_days, out_dir=out)
    else:
        # Legacy single-pull (limited history).
        key = os.getenv("PROJECTX_API_KEY") or os.getenv("API_KEY")
        login = os.getenv("PROJECTX_USERNAME") or os.getenv("USERNAME")
        asset = os.getenv("PROJECTX_ASSET", "CON.F.US.MGC.Q26")
        dest = out or (BACKTEST_DATA / "MGCQ6")
        dest.mkdir(parents=True, exist_ok=True)
        from FVG_projectX_bot.projectX.projectx_api_functions import fetch_data

        token = login_to_api(login, key)["token"]
        df_15 = fetch_data(asset, "15min", 20000, token)
        df_1 = fetch_data(asset, "1min", 20000, token)
        path_15 = dest / "topstep_15min.csv"
        path_1 = dest / "topstep_1min.csv"
        df_15.to_csv(path_15, index=False)
        df_1.to_csv(path_1, index=False)
        print(f"15m: {len(df_15)} -> {path_15}")
        print(f"1m: {len(df_1)} -> {path_1}")
