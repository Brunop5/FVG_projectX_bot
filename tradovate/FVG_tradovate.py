import json
import os
import threading
import time
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv
from websocket import WebSocketApp

from ..FVG_strategy import (
    EMA_PERIOD,
    FVG_Order,
    FVG_Strategy,
    HTF_TF,
    MAX_DAILY_TRADES,
    ORDER_SIZE,
    RISK_PERCENT,
    USE_FIXED_LOT,
    USE_TRAILING,
    FIXED_LOT,
)
from ..helping_functions.pyramiding import MaxOrdersPolicy
from ..projectX.projectx_api_functions import sleep_until_next_boundary


load_dotenv()

# === Runtime configuration ===
TRADOVATE_BASE_URL = "https://live.tradovateapi.com/v1"
TRADOVATE_WS_URL = "wss://live.tradovateapi.com/v1/websocket"

TRADOVATE_USERNAME = os.getenv("TRADOVATE_USERNAME", "")
TRADOVATE_PASSWORD = os.getenv("TRADOVATE_PASSWORD", "")
TRADOVATE_APP_ID = os.getenv("TRADOVATE_APP_ID", "")
TRADOVATE_APP_VERSION = os.getenv("TRADOVATE_APP_VERSION", "")
TRADOVATE_ACCOUNT_ID = os.getenv("TRADOVATE_ACCOUNT_ID")

TRADOVATE_AUTH_PATH = "/auth/accesstoken"
TRADOVATE_ACCOUNT_LIST_PATH = "/account/list"
TRADOVATE_BALANCE_PATH = "/account/balance"
TRADOVATE_ORDER_PATH = "/order/placeorder"
TRADOVATE_BARCHART_PATH = "/marketdata/barchart"
TRADOVATE_BARCHART_ALT_PATH = "/md/getchart"
TRADOVATE_CONTRACT_LIST_PATH = "/contract/list"

TRADOVATE_WS_SUBSCRIBE_TEMPLATE = None

MAX_OPEN_ORDERS = 1
MIN_ORDER_QTY = 1.0
SHOW_FUTURES_CONTRACTS = False
SHOW_ACCOUNT_IDS = False


def _load_assets_from_env() -> list[tuple[str, str, int]]:
    raw = os.getenv("TRADOVATE_ASSETS_JSON", "")
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []
    assets: list[tuple[str, str, int]] = []
    for item in data:
        if not isinstance(item, (list, tuple)) or len(item) < 3:
            continue
        symbol, timeframe, contract_id = item[0], item[1], item[2]
        try:
            assets.append((str(symbol), str(timeframe), int(contract_id)))
        except (TypeError, ValueError):
            continue
    return assets


ASSETS = [["MGC", "15min", 55]]


def _tf_to_minutes(timeframe: str) -> int:
    if timeframe.isdigit():
        return int(timeframe)
    tf = timeframe.lower().strip()
    if tf.endswith("min"):
        return int(tf.replace("min", ""))
    if tf.endswith("m"):
        return int(tf.replace("m", ""))
    if tf.endswith("h"):
        return int(tf.replace("h", "")) * 60
    if tf.endswith("d"):
        return int(tf.replace("d", "")) * 1440
    return 15


def _normalize_timestamp(ts: Any) -> int | None:
    if ts is None:
        return None
    try:
        val = float(ts)
        if val > 10**12:
            return int(val)
        return int(val * 1000)
    except (TypeError, ValueError):
        return None


def _normalize_bars(rows: list[Any]) -> pd.DataFrame:
    normalized = []
    for bar in rows:
        if isinstance(bar, dict):
            ts = bar.get("timestamp") or bar.get("time") or bar.get("t")
            normalized.append(
                {
                    "timestamp": _normalize_timestamp(ts),
                    "open": float(bar.get("open", bar.get("o", 0))),
                    "high": float(bar.get("high", bar.get("h", 0))),
                    "low": float(bar.get("low", bar.get("l", 0))),
                    "close": float(bar.get("close", bar.get("c", 0))),
                    "volume": float(bar.get("volume", bar.get("v", 0))),
                }
            )
        elif isinstance(bar, (list, tuple)) and len(bar) >= 6:
            normalized.append(
                {
                    "timestamp": _normalize_timestamp(bar[0]),
                    "open": float(bar[1]),
                    "high": float(bar[2]),
                    "low": float(bar[3]),
                    "close": float(bar[4]),
                    "volume": float(bar[5]),
                }
            )
    df = pd.DataFrame(normalized)
    if "timestamp" in df.columns:
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return df


def _extract_price_payload(payload: Any) -> dict | None:
    if payload is None:
        return None
    if isinstance(payload, list):
        for item in payload:
            found = _extract_price_payload(item)
            if found:
                return found
        return None
    if not isinstance(payload, dict):
        return None
    if "data" in payload and isinstance(payload["data"], dict):
        nested = _extract_price_payload(payload["data"])
        if nested:
            return nested
    keys = payload.keys()
    price_key = next((k for k in ("close", "last", "price", "lastPrice") if k in keys), None)
    if price_key is None:
        return None
    volume_key = next((k for k in ("volume", "vol", "lastSize") if k in keys), None)
    high_key = next((k for k in ("high", "h") if k in keys), None)
    low_key = next((k for k in ("low", "l") if k in keys), None)
    row = {"close": float(payload[price_key])}
    if volume_key is not None:
        row["volume"] = float(payload[volume_key])
    if high_key is not None:
        row["high"] = float(payload[high_key])
    if low_key is not None:
        row["low"] = float(payload[low_key])
    return row


def _tradovate_side(side: str) -> str:
    return "Buy" if side.upper() == "BUY" else "Sell"


class TradovateClient:
    def __init__(self, base_url: str, ws_url: str):
        self.base_url = base_url.rstrip("/")
        self.ws_url = ws_url
        self._token = None
        self._token_expiry = None
        self._lock = threading.Lock()

    def _auth_headers(self) -> dict:
        token = self._token or ""
        return {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

    def login(self) -> bool:
        if not (TRADOVATE_USERNAME and TRADOVATE_PASSWORD and TRADOVATE_APP_ID and TRADOVATE_APP_VERSION):
            return False
        payload = {
            "name": TRADOVATE_USERNAME,
            "password": TRADOVATE_PASSWORD,
            "appId": TRADOVATE_APP_ID,
            "appVersion": TRADOVATE_APP_VERSION,
        }
        url = f"{self.base_url}{TRADOVATE_AUTH_PATH}"
        try:
            resp = requests.post(url, json=payload, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            print(f"❌ Tradovate auth failed: {exc}")
            return False
        token = data.get("accessToken") or data.get("token")
        if not token:
            print("❌ Tradovate auth failed: missing access token.")
            return False
        self._token = token
        expires_in = data.get("expiresIn") or data.get("expires_in") or 3600
        try:
            self._token_expiry = time.time() + float(expires_in) * 0.9
        except (TypeError, ValueError):
            self._token_expiry = time.time() + 3600
        return True

    def _ensure_token(self) -> bool:
        with self._lock:
            if self._token and self._token_expiry and time.time() < self._token_expiry:
                return True
            return self.login()

    def request(self, method: str, path: str, params: dict | None = None, payload: dict | None = None):
        if not self._ensure_token():
            return None
        url = path if path.startswith("http") else f"{self.base_url}{path}"
        try:
            resp = requests.request(
                method=method,
                url=url,
                params=params,
                json=payload,
                headers=self._auth_headers(),
                timeout=20,
            )
            if resp.status_code == 401:
                if self.login():
                    resp = requests.request(
                        method=method,
                        url=url,
                        params=params,
                        json=payload,
                        headers=self._auth_headers(),
                        timeout=20,
                    )
            resp.raise_for_status()
            if resp.text:
                return resp.json()
            return None
        except Exception as exc:
            print(f"⚠️ Tradovate request failed ({method} {path}): {exc}")
            return None

    def get_accounts(self) -> list[dict]:
        data = self.request("GET", TRADOVATE_ACCOUNT_LIST_PATH)
        if isinstance(data, list):
            return data
        return []

    def get_account_balance(self, account_id: int) -> float:
        params = {"id": account_id}
        data = self.request("GET", TRADOVATE_BALANCE_PATH, params=params)
        if isinstance(data, dict):
            for key in ("cashBalance", "balance", "netLiquidation"):
                if key in data:
                    try:
                        return float(data[key])
                    except (TypeError, ValueError):
                        continue
        return 0.0

    def list_contracts(self, contract_type: str | None = None) -> list[dict]:
        params = {"contractType": contract_type} if contract_type else None
        data = self.request("GET", TRADOVATE_CONTRACT_LIST_PATH, params=params)
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and "contracts" in data:
            return data["contracts"]
        return []

    def fetch_bars(self, contract_id: int, timeframe_minutes: int, bars_back: int) -> pd.DataFrame:
        params = {"contractId": contract_id, "timeframe": timeframe_minutes, "barsBack": bars_back}
        data = self.request("GET", TRADOVATE_BARCHART_PATH, params=params)
        if isinstance(data, dict) and "bars" in data:
            return _normalize_bars(data["bars"])
        if isinstance(data, list):
            return _normalize_bars(data)
        fallback_payload = {
            "contractId": contract_id,
            "timeframe": timeframe_minutes,
            "barsBack": bars_back,
        }
        data = self.request("POST", TRADOVATE_BARCHART_ALT_PATH, payload=fallback_payload)
        if isinstance(data, dict) and "bars" in data:
            return _normalize_bars(data["bars"])
        if isinstance(data, list):
            return _normalize_bars(data)
        return pd.DataFrame()

    def place_order(self, account_id: int, contract_id: int, side: str, qty: float) -> dict:
        payload = {
            "accountId": account_id,
            "contractId": contract_id,
            "action": _tradovate_side(side),
            "orderQty": qty,
            "orderType": "Market",
            "timeInForce": "Day",
            "isAutomated": True,
        }
        result = self.request("POST", TRADOVATE_ORDER_PATH, payload=payload)
        return result if isinstance(result, dict) else {"success": False, "message": "Order failed"}


class Tradovate_Order(FVG_Order):
    MIN_ORDER_SIZE = 1.0
    ORDER_SIZE_STEP = None
    ORDER_SIZE_INTEGER_ONLY = True

    def __init__(self, client: TradovateClient, account_id: int, contract_id: int, **kwargs):
        super().__init__(**kwargs)
        self._client = client
        self._account_id = account_id
        self._contract_id = contract_id
        self.use_trailing = USE_TRAILING

    def place_order(self):
        if self._client is None:
            return {"success": False, "message": "Missing Tradovate client"}
        result = self._client.place_order(
            account_id=self._account_id,
            contract_id=self._contract_id,
            side=self.side,
            qty=self.order_size,
        )
        success = isinstance(result, dict) and result.get("id") is not None
        return {"success": success, "order": result}

    def close_order(self):
        if self._client is None:
            return {"success": False, "message": "Missing Tradovate client"}
        opposite = "SELL" if self.side.upper() == "BUY" else "BUY"
        result = self._client.place_order(
            account_id=self._account_id,
            contract_id=self._contract_id,
            side=opposite,
            qty=self.order_size,
        )
        success = isinstance(result, dict) and result.get("id") is not None
        return {"success": success, "order": result}


class Tradovate_Strategy(FVG_Strategy):
    Order = Tradovate_Order

    def __init__(self, asset_tuple: tuple[str, str, int], client: TradovateClient):
        self.asset = asset_tuple[0]
        self.timeframe = asset_tuple[1]
        self.contract_id = int(asset_tuple[2])
        self._client = client
        self._ws_client = None
        self.account_id = int(TRADOVATE_ACCOUNT_ID) if TRADOVATE_ACCOUNT_ID else None
        self.pyramiding = MaxOrdersPolicy(MAX_OPEN_ORDERS)

        filename = f"{self.asset}-{self.timeframe}-tradovate"
        self.csv_filename = f"{filename}.csv"
        self.metadata_filename = f"{filename}.json"

    def init_api(self):
        if self._client is None:
            raise RuntimeError("Tradovate client not initialized.")
        if not self._client._ensure_token():
            raise RuntimeError("Tradovate authentication failed.")
        if self.account_id is None:
            accounts = self._client.get_accounts()
            if accounts:
                self.account_id = int(accounts[0].get("id"))
        if self.account_id is None:
            raise RuntimeError("Missing Tradovate account ID.")
        self.account_balance = self._client.get_account_balance(self.account_id)
        super().__init__()
        self.require_intrabar_entry = True

    def api_order_kwargs(self) -> dict:
        return {
            "client": self._client,
            "account_id": self.account_id,
            "contract_id": self.contract_id,
        }

    def gather_data(self) -> pd.DataFrame:
        timeframe_minutes = _tf_to_minutes(self.timeframe)
        bars_back = max(200, EMA_PERIOD + 60)
        return self._client.fetch_bars(self.contract_id, timeframe_minutes, bars_back)

    def fetch_new_data(self):
        timeframe_minutes = _tf_to_minutes(self.timeframe)
        bars = self._client.fetch_bars(self.contract_id, timeframe_minutes, 2)
        if bars is None or bars.empty:
            return
        latest = bars.iloc[[-1]]
        if "timestamp" in latest.columns and not self.data.empty:
            if latest["timestamp"].iloc[-1] <= self.data["timestamp"].iloc[-1]:
                return
        self.cur_close = latest["close"].iloc[-1]
        self.cur_volume = latest["volume"].iloc[-1]
        self.data = pd.concat([self.data, latest], ignore_index=True).iloc[-200:]
        print(f"\n⏰ New bar - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} close: {self.cur_close}")

    def fetch_htf_data(self) -> pd.DataFrame:
        htf_minutes = _tf_to_minutes(str(HTF_TF))
        bars_back = max(EMA_PERIOD + 51, 101)
        data = self._client.fetch_bars(self.contract_id, htf_minutes, bars_back)
        if data is None or data.empty:
            return pd.DataFrame()
        if hasattr(self, "cur_close"):
            data = data.copy()
            data.loc[data.index[-1], "close"] = float(self.cur_close)
        return data

    def check_daily_trade_limit(self):
        today = datetime.now(timezone.utc).date()
        if self.last_trade_date != str(today):
            self.daily_trades_count = 0
            self.last_trade_date = str(today)
        return self.daily_trades_count < MAX_DAILY_TRADES

    def calculate_order_size(self, atr, sl_mult):
        if USE_FIXED_LOT:
            return max(MIN_ORDER_QTY, FIXED_LOT)
        risk_amount = self.account_balance * (RISK_PERCENT / 100)
        stop_distance = atr * sl_mult
        if stop_distance > 0:
            lot_size = risk_amount / stop_distance
            lot_size = round(lot_size, 3)
            return max(MIN_ORDER_QTY, lot_size)
        return max(MIN_ORDER_QTY, ORDER_SIZE)

    def subscribe_to_price_updates(self):
        def on_open(ws):
            ws.send(json.dumps({"action": "setAuthToken", "token": self._client._token}))
            subscribe_message = self._build_ws_subscribe_message()
            if subscribe_message is not None:
                ws.send(json.dumps(subscribe_message))

        def on_message(_, message):
            try:
                payload = json.loads(message) if isinstance(message, str) else message
                row = _extract_price_payload(payload)
                if not row:
                    return
                if "volume" not in row:
                    row["volume"] = 0.0
                self.update_price(pd.DataFrame([row]))
            except Exception as exc:
                print(f"⚠️ Tradovate websocket parse error: {exc}")

        def on_close(_, code, reason):
            print(f"⚠️ Tradovate websocket closed: {code} {reason}")

        while True:
            try:
                self._ws_client = WebSocketApp(
                    self._client.ws_url,
                    on_open=on_open,
                    on_message=on_message,
                    on_close=on_close,
                )
                self._ws_client.run_forever(ping_interval=20, ping_timeout=10)
            except Exception as exc:
                print(f"⚠️ Tradovate websocket reconnecting after error: {exc}")
                time.sleep(5)

    def _build_ws_subscribe_message(self) -> dict | None:
        if TRADOVATE_WS_SUBSCRIBE_TEMPLATE:
            try:
                template = TRADOVATE_WS_SUBSCRIBE_TEMPLATE.format(
                    contract_id=self.contract_id,
                    symbol=self.asset,
                    timeframe=_tf_to_minutes(self.timeframe),
                )
                return json.loads(template)
            except Exception:
                return None
        return {"action": "subscribe", "topic": "md/quote", "contractId": self.contract_id}

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


def run_strat(strat: Tradovate_Strategy):
    strat.init_api()
    strat.run()


def list_available_futures_contracts(
    client: TradovateClient,
    show: bool = SHOW_FUTURES_CONTRACTS,
):
    if not show:
        return
    if not client._ensure_token():
        print("⚠️ Unable to authenticate for contracts list.")
        return
    contracts = client.list_contracts(contract_type="Future")
    if not contracts:
        print("⚠️ No futures contracts returned.")
        return
    print("📄 Available futures contracts:")
    for contract in contracts:
        name = contract.get("name")
        contract_id = contract.get("id")
        symbol = contract.get("symbol")
        print(f"- {name} ({symbol}) id={contract_id}")


def list_available_account_ids(
    client: TradovateClient,
    show: bool = SHOW_ACCOUNT_IDS,
):
    if not show:
        return
    if not client._ensure_token():
        print("⚠️ Unable to authenticate for account list.")
        return
    accounts = client.get_accounts()
    if not accounts:
        print("⚠️ No accounts returned.")
        return
    print("📄 Available account IDs:")
    for account in accounts:
        account_id = account.get("id")
        name = account.get("name")
        account_type = account.get("accountType")
        status = account.get("status")
        print(f"- id={account_id} name={name} type={account_type} status={status}")


if __name__ == "__main__":
    if not ASSETS:
        raise RuntimeError(
            "Set TRADOVATE_ASSETS_JSON to a list like "
            '[["ESM6","15min",12345]] before running.'
        )
    if not TRADOVATE_USERNAME or not TRADOVATE_PASSWORD:
        raise RuntimeError("Set TRADOVATE_USERNAME and TRADOVATE_PASSWORD in env.")
    if not TRADOVATE_APP_ID or not TRADOVATE_APP_VERSION:
        raise RuntimeError("Set TRADOVATE_APP_ID and TRADOVATE_APP_VERSION in env.")

    client = TradovateClient(TRADOVATE_BASE_URL, TRADOVATE_WS_URL)
    list_available_account_ids(client)
    list_available_futures_contracts(client)
    threads = []
    strats = [Tradovate_Strategy(asset, client) for asset in ASSETS]
    for strat in strats:
        t = threading.Thread(target=run_strat, args=(strat,), daemon=True)
        t.start()
        threads.append(t)
    while True:
        time.sleep(5)

