import argparse
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://fapi.binance.com/fapi/v1/continuousKlines"
INTERVAL_MS = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "2h": 7_200_000,
    "4h": 14_400_000,
    "6h": 21_600_000,
    "8h": 28_800_000,
    "12h": 43_200_000,
    "1d": 86_400_000,
}


def fetch_klines(pair, interval, contract_type, start_time=None, limit=1500, timeout=10):
    params = {
        "pair": pair,
        "contractType": contract_type,
        "interval": interval,
        "limit": limit,
    }
    if start_time is not None:
        params["startTime"] = int(start_time)

    response = requests.get(BASE_URL, params=params, timeout=timeout)
    response.raise_for_status()
    return response.json()


def load_existing_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "timestamp" in df.columns:
        df["timestamp"] = df["timestamp"].astype("int64")
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Download BTCUSDT perpetual futures OHLCV 15m data from Binance."
    )
    parser.add_argument("--pair", default="BTCUSDT")
    parser.add_argument("--interval", default="15m")
    parser.add_argument("--contract-type", default="PERPETUAL")
    parser.add_argument(
        "--output",
        default="data/BTCUSDT_PERP_15m.csv",
        help="Output CSV path",
    )
    parser.add_argument("--sleep", type=float, default=0.25, help="Sleep seconds between requests")
    parser.add_argument("--limit", type=int, default=1500, help="Max bars per request (<=1500)")
    args = parser.parse_args()

    if args.interval not in INTERVAL_MS:
        raise ValueError(f"Unsupported interval: {args.interval}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    existing_df = load_existing_csv(output_path)
    existing_last_ts = None
    if existing_df is not None and not existing_df.empty:
        existing_last_ts = int(existing_df["timestamp"].max())
        print(f"Found existing data. Last timestamp: {existing_last_ts}")

    # Discover earliest available candle
    earliest_batch = fetch_klines(
        pair=args.pair,
        interval=args.interval,
        contract_type=args.contract_type,
        start_time=0,
        limit=1,
    )
    if not earliest_batch:
        print("No data returned from Binance.")
        return

    earliest_ts = int(earliest_batch[0][0])
    interval_ms = INTERVAL_MS[args.interval]
    start_time = earliest_ts if existing_last_ts is None else max(existing_last_ts + interval_ms, earliest_ts)

    all_rows = []
    total = 0
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

    print(f"Starting from: {datetime.utcfromtimestamp(start_time / 1000).isoformat()}Z")
    while start_time < now_ms:
        data = fetch_klines(
            pair=args.pair,
            interval=args.interval,
            contract_type=args.contract_type,
            start_time=start_time,
            limit=args.limit,
        )
        if not data:
            break

        for bar in data:
            all_rows.append(
                {
                    "timestamp": int(bar[0]),
                    "open": float(bar[1]),
                    "high": float(bar[2]),
                    "low": float(bar[3]),
                    "close": float(bar[4]),
                    "volume": float(bar[5]),
                }
            )

        total += len(data)
        start_time = int(data[-1][0]) + interval_ms
        last_dt = datetime.utcfromtimestamp(int(data[-1][0]) / 1000).isoformat()
        print(f"Fetched {len(data)} bars. Total: {total}. Last: {last_dt}Z")

        if len(data) < args.limit:
            break
        time.sleep(args.sleep)

    new_df = pd.DataFrame(all_rows)
    if new_df.empty:
        print("No new data fetched.")
        return

    if existing_df is not None and not existing_df.empty:
        combined = pd.concat([existing_df, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["timestamp"], keep="last")
        combined = combined.sort_values("timestamp").reset_index(drop=True)
    else:
        combined = new_df.sort_values("timestamp").reset_index(drop=True)

    combined.to_csv(output_path, index=False)
    oldest = datetime.utcfromtimestamp(combined["timestamp"].iloc[0] / 1000).isoformat()
    newest = datetime.utcfromtimestamp(combined["timestamp"].iloc[-1] / 1000).isoformat()
    print(f"Saved {len(combined):,} bars to {output_path}")
    print(f"Range: {oldest}Z -> {newest}Z")


if __name__ == "__main__":
    main()

