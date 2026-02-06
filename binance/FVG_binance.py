import json
import os
import threading
import time
from datetime import datetime, timezone
from typing import Any
from dotenv import load_dotenv

import pandas as pd
from binance.um_futures import UMFutures
from binance.websocket.um_futures.websocket_client import UMFuturesWebsocketClient

from ..FVG_strategy import USE_TRAILING, FVG_Order, FVG_Strategy, HTF_TF, EMA_PERIOD
from ..FVG_strategy import USE_FIXED_LOT, FIXED_LOT, MAX_DAILY_TRADES
from ..FVG_strategy import RISK_PERCENT, ORDER_SIZE, DEBUG_STOPS

from ..projectX.projectx_api_functions import sleep_until_next_boundary


ASSETS = [("BTCUSDT", "15min")]  # BTCUSDT perpetual (USDT-margined)
USE_CONTINUOUS_KLINES = False
CONTRACT_TYPE = "PERPETUAL"

BINANCE_BASE_URL = "https://fapi.binance.com"
BINANCE_TESTNET_URL = "https://testnet.binancefuture.com"
# UMFuturesWebsocketClient appends "/ws" internally; keep base URL here.
WEBSOCKET_BASE_URL = "wss://fstream.binance.com"
WEBSOCKET_TESTNET_URL = "wss://stream.binancefuture.com"

load_dotenv()
API_KEY = os.getenv("BINANCE_API_KEY")
API_SECRET = os.getenv("BINANCE_API_SECRET")
USE_TESTNET = os.getenv("BINANCE_TESTNET", "0").lower() in {"1", "true", "yes"}


TIMEFRAME_MAP = {
    "1min": "1m",
    "3min": "3m",
    "5min": "5m",
    "15min": "15m",
    "30min": "30m",
    "1h": "1h",
    "2h": "2h",
    "4h": "4h",
    "6h": "6h",
    "8h": "8h",
    "12h": "12h",
    "1d": "1d",
}


def _binance_interval_from_timeframe(timeframe: str) -> str:
    if timeframe in TIMEFRAME_MAP:
        return TIMEFRAME_MAP[timeframe]
    if timeframe.isdigit():
        minutes = int(timeframe)
        if minutes % 60 == 0:
            return f"{minutes // 60}h"
        return f"{minutes}m"
    if timeframe.endswith("min"):
        return timeframe.replace("min", "m")
    return timeframe


def _binance_base_url() -> str:
    return BINANCE_TESTNET_URL if USE_TESTNET else BINANCE_BASE_URL


def _binance_ws_url() -> str:
    return WEBSOCKET_TESTNET_URL if USE_TESTNET else WEBSOCKET_BASE_URL


def _klines_to_df(klines: list[list[Any]]) -> pd.DataFrame:
    rows = []
    for k in klines:
        rows.append(
            {
                "timestamp": int(k[0]),
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5]),
            }
        )
    return pd.DataFrame(rows)


def fetch_klines(client: UMFutures, symbol: str, interval: str, limit: int = 100) -> pd.DataFrame:
    if USE_CONTINUOUS_KLINES and hasattr(client, "continuous_klines"):
        klines = client.continuous_klines(
            pair=symbol,
            contractType=CONTRACT_TYPE,
            interval=interval,
            limit=limit,
        )
    else:
        klines = client.klines(symbol=symbol, interval=interval, limit=limit)
    return _klines_to_df(klines)


def load_cached_data(path: str) -> pd.DataFrame | None:
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if "timestamp" in df.columns:
        df["timestamp"] = df["timestamp"].astype("int64")
    return df


def _select_latest_closed_kline(klines: list[list[Any]]) -> list[Any] | None:
    if not klines:
        return None
    now_ms = int(time.time() * 1000)
    latest = klines[-1]
    close_time = int(latest[6])
    if close_time > now_ms and len(klines) > 1:
        return klines[-2]
    return latest


class Binance_Order(FVG_Order):
    symbol: str
    _client: UMFutures
    _position_side: str | None

    def __init__(self, symbol: str, client: UMFutures, **kwargs):
        super().__init__(**kwargs)
        self.symbol = symbol
        self._client = client
        self._position_side = None
        self.use_trailing = USE_TRAILING

        if self._client is not None:
            try:
                mode = self._client.get_position_mode()
                if isinstance(mode, dict) and mode.get("dualSidePosition"):
                    self._position_side = "LONG" if self.side.upper() == "BUY" else "SHORT"
            except Exception:
                self._position_side = None

    def place_order(self):
        if self._client is None:
            print("Error: Binance REST client not initialized.")
            return {"success": False, "message": "Missing API keys"}

        params = {
            "symbol": self.symbol,
            "side": self.side.upper(),
            "type": "MARKET",
            "quantity": self.order_size,
        }
        if self._position_side:
            params["positionSide"] = self._position_side
        try:
            result = self._client.new_order(**params)
            print(
                f"✅ Binance order placed: {self.side} {self.symbol} "
                f"qty={self.order_size} entry={self.entry_price}"
            )
            return {"success": True, "order": result}
        except Exception as exc:
            print(f"❌ Binance order failed: {exc}")
            return {"success": False, "message": str(exc)}

    def close_order(self):
        if self._client is None:
            print("Error: Binance REST client not initialized.")
            return {"success": False, "message": "Missing API keys"}

        opposite = "SELL" if self.side.upper() == "BUY" else "BUY"
        params = {
            "symbol": self.symbol,
            "side": opposite,
            "type": "MARKET",
            "quantity": self.order_size,
        }
        if self._position_side:
            params["positionSide"] = self._position_side
        try:
            result = self._client.new_order(**params)
            print(
                f"✅ Binance order closed: {opposite} {self.symbol} "
                f"qty={self.order_size}"
            )
            return {"success": True, "order": result}
        except Exception as exc:
            print(f"❌ Binance close failed: {exc}")
            return {"success": False, "message": str(exc)}


class Binance_Strategy(FVG_Strategy):
    Order = Binance_Order
    api_key: str
    api_secret: str
    asset: str
    _client: UMFutures | None
    _ws_client: UMFuturesWebsocketClient | None

    def __init__(self, asset_tuple: tuple[str, str]):
        self.api_key = ""
        self.api_secret = ""
        self.asset = asset_tuple[0]
        self.timeframe = asset_tuple[1]
        self._client = None
        self._ws_client = None

        suffix = f"-{CONTRACT_TYPE.lower()}" if USE_CONTINUOUS_KLINES else ""
        filename = f"{self.asset}-{self.timeframe}{suffix}"
        self.csv_filename = f"{filename}.csv"
        self.metadata_filename = f"{filename}.json"

    def init_api(self, api_key: str, api_secret: str):
        self.api_key = api_key
        self.api_secret = api_secret
        self._client = UMFutures(key=api_key, secret=api_secret, base_url=_binance_base_url())
        self.account_balance = self.get_account_balance()
        super().__init__()

    def api_order_kwargs(self) -> dict:
        return {"symbol": self.asset, "client": self._client}

    def get_account_balance(self) -> float:
        try:
            if self._client is None:
                return 0.0
            balances = self._client.balance()
            for item in balances:
                if item.get("asset") == "USDT":
                    return float(item.get("availableBalance", 0))
        except Exception as exc:
            print(f"⚠️ Failed to fetch balance: {exc}")
        return 0.0

    def gather_data(self) -> pd.DataFrame:
        cached = load_cached_data(self.csv_filename)
        if cached is not None and not cached.empty:
            return cached
        interval = _binance_interval_from_timeframe(self.timeframe)
        if self._client is None:
            return pd.DataFrame()
        return fetch_klines(self._client, self.asset, interval, limit=200)

    def fetch_new_data(self):
        if self._client is None:
            return
        interval = _binance_interval_from_timeframe(self.timeframe)
        if USE_CONTINUOUS_KLINES and hasattr(self._client, "continuous_klines"):
            klines = self._client.continuous_klines(
                pair=self.asset,
                contractType=CONTRACT_TYPE,
                interval=interval,
                limit=2,
            )
        else:
            klines = self._client.klines(symbol=self.asset, interval=interval, limit=2)
        latest_closed = _select_latest_closed_kline(klines)
        if latest_closed is None:
            return

        new_row = _klines_to_df([latest_closed])
        if new_row["timestamp"].iloc[-1] > self.data["timestamp"].iloc[-1]:
            self.cur_close = new_row["close"].iloc[-1]
            self.cur_volume = new_row["volume"].iloc[-1]
            self.data = pd.concat([self.data, new_row], ignore_index=True).iloc[-100:]
            print(f"\n⏰ New bar - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} close: {self.cur_close}")

    def fetch_htf_data(self) -> pd.DataFrame:
        htf_tf = str(HTF_TF)
        if htf_tf.isdigit():
            if int(htf_tf) % 60 == 0:
                htf_tf = f"{int(htf_tf) // 60}h"
            else:
                htf_tf = f"{htf_tf}m"

        interval = _binance_interval_from_timeframe(htf_tf)
        num_bars = max(EMA_PERIOD + 51, 101)
        if self._client is None:
            return pd.DataFrame()
        data = fetch_klines(self._client, self.asset, interval, limit=num_bars)
        if data is None or len(data) == 0:
            return pd.DataFrame()

        if hasattr(self, "cur_close"):
            data = data.copy()
            data.loc[data.index[-1], "close"] = float(self.cur_close)
        return data

    def check_daily_trade_limit(self):
        today = datetime.now().date()
        if self.last_trade_date != str(today):
            self.daily_trades_count = 0
            self.last_trade_date = str(today)
        return self.daily_trades_count < MAX_DAILY_TRADES

    def calculate_order_size(self, atr, sl_mult):
        if USE_FIXED_LOT:
            return FIXED_LOT

        risk_amount = self.account_balance * (RISK_PERCENT / 100)
        stop_distance = atr * sl_mult

        if stop_distance > 0:
            lot_size = risk_amount / stop_distance
            lot_size = round(lot_size, 3)
            return max(0.001, min(lot_size, 100))
        return ORDER_SIZE

    def subscribe_to_price_updates(self):
        interval = _binance_interval_from_timeframe(self.timeframe)
        symbol = self.asset.lower()

        def on_message(_, message):
            try:
                payload = json.loads(message) if isinstance(message, str) else message
                kline = payload.get("k") or payload.get("data", {}).get("k", {})
                if not kline:
                    return
                close = float(kline.get("c"))
                volume = float(kline.get("v", 0))
                new_row = pd.DataFrame([{"close": close, "volume": volume}])
                self.update_price(new_row)
            except Exception as exc:
                print(f"⚠️ Websocket parse error: {exc}")

        while True:
            try:
                self._ws_client = UMFuturesWebsocketClient(
                    on_message=on_message,
                    stream_url=_binance_ws_url(),
                )
                if USE_CONTINUOUS_KLINES and hasattr(self._ws_client, "continuous_kline"):
                    try:
                        self._ws_client.continuous_kline(
                            pair=symbol,
                            contract_type=CONTRACT_TYPE,
                            interval=interval,
                        )
                    except TypeError:
                        self._ws_client.continuous_kline(
                            pair=symbol,
                            contractType=CONTRACT_TYPE,
                            interval=interval,
                        )
                else:
                    self._ws_client.kline(symbol=symbol, interval=interval)

                while True:
                    time.sleep(1)
            except Exception as exc:
                if self._ws_client is not None:
                    try:
                        self._ws_client.stop()
                    except Exception:
                        pass
                print(f"⚠️ Websocket reconnecting after error: {exc}")
                time.sleep(5)

    def start_bar_iterations(self):
        while True:
            try:
                sleep_until_next_boundary(self.timeframe)
                self.bar_iteration()
            except Exception as exc:
                print(f"❌ Error in bar iteration: {exc}")
                time.sleep(60)

    def run(self):
        print(f"\n{'='*60}")
        print(f"🤖 Trading Bot Started for {self.asset}")
        print(f"{'='*60}")
        print(f"Timeframe: {self.timeframe}")
        print(f"HTF Bias: {HTF_TF} | EMA Period: {EMA_PERIOD}")

        t1 = threading.Thread(target=self.start_bar_iterations)
        t2 = threading.Thread(target=self.subscribe_to_price_updates)
        t1.start()
        t2.start()


def run_strat(strat: Binance_Strategy, api_key: str, api_secret: str):
    strat.init_api(api_key, api_secret)
    strat.run()



if __name__ == "__main__":
    print("main?")
    import threading

    if not API_KEY or not API_SECRET:
        raise RuntimeError("Set BINANCE_API_KEY and BINANCE_API_SECRET in env.")

    threads = []
    strats = [Binance_Strategy(asset) for asset in ASSETS]

    for strat in strats:
        t = threading.Thread(target=run_strat, args=(strat, API_KEY, API_SECRET), daemon=True)
        t.start()
        threads.append(t)

    while True:
        time.sleep(5)

