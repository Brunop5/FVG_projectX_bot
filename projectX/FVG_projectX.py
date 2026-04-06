import os
import sys
import logging
import warnings
import requests
import pandas as pd
from datetime import datetime, timedelta
from time import sleep
import threading
import time

# Load projectX-local inputs before importing FVG strategy globals.
PROJECTX_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("FVG_INPUTS_JSON", os.path.join(PROJECTX_DIR, "inputs.json"))

from ..FVG_strategy import *
from .projectx_api_functions import get_account_id
from .projectx_api_functions import get_account_balance
from .projectx_api_functions import load_data
from .projectx_api_functions import fetch_data
from .projectx_api_functions import login_to_api
from .projectx_api_functions import validate_token
from .projectx_api_functions import sleep_until_next_boundary
from .projectx_api_functions import topstepx_post
from .projectx_api_functions import search_trades

PROJECTX_RUNTIME_DIR = os.path.join(PROJECTX_DIR, INPUTS.RUNTIME_SUBDIR)
PROJECTX_LOG_PATH = os.path.join(
    PROJECTX_RUNTIME_DIR,
    INPUTS.LOG_FILE_NAMES.get("projectx", "projectx_run.log"),
)


def setup_global_logging(log_path: str) -> None:
    log_dir = os.path.dirname(log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.FileHandler(log_path, mode="a", encoding="utf-8")],
    )

    class _StreamToLogger:
        def __init__(self, logger: logging.Logger, level: int):
            self.logger = logger
            self.level = level

        def write(self, message: str) -> None:
            message = message.rstrip()
            if message:
                self.logger.log(self.level, message)

        def flush(self) -> None:
            pass

    stdout_logger = logging.getLogger("stdout")
    stderr_logger = logging.getLogger("stderr")
    sys.stdout = _StreamToLogger(stdout_logger, logging.INFO)
    sys.stderr = _StreamToLogger(stderr_logger, logging.ERROR)

    def _showwarning(message, category, filename, lineno, file=None, line=None):
        warn_logger = logging.getLogger("warnings")
        warn_logger.warning(
            "%s:%s: %s: %s",
            filename,
            lineno,
            category.__name__,
            message,
        )

    warnings.showwarning = _showwarning

    def _excepthook(exc_type, exc_value, exc_traceback):
        logger = logging.getLogger("exceptions")
        logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

    sys.excepthook = _excepthook


def init_api(username, api_key):
    res = login_to_api(username, api_key)
    if not res["success"]:
        print(username, api_key)
        print(res)
        raise RuntimeError("❌ API login failed")

    global_token = res["token"]
    print(f"✅ API initialized.")
    return global_token


class ProjectX_Order(FVG_Order):
    MIN_ORDER_SIZE = 1.0
    ORDER_SIZE_STEP = None
    ORDER_SIZE_INTEGER_ONLY = True

    account_id: str
    asset_id: str
    auth_token: str

    def __init__(self, account_id, asset_id, auth_token, **kwargs):
        super().__init__(**kwargs)

        self.account_id = account_id
        self.asset_id = asset_id
        self.auth_token = auth_token

    def place_order(self):
        """
        Place an order using ProjectX Gateway API.
        Based on: https://gateway.docs.projectx.com/docs/api-reference/order/order-place
        """
        if not self.auth_token:
            print("Error: auth_token is required to place order")
            return {'success': False, 'message': 'auth_token is required'}
        
        url = "https://api.topstepx.com/api/Order/place"
        
        headers = {
            'accept': 'text/plain',
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.auth_token}'
        }
        
        # Map side: "BUY" -> 0 (Bid), "SELL" -> 1 (Ask)
        side_code = 0 if self.side.upper() == "BUY" else 1
        
        payload = {
            "accountId": self.account_id,
            "contractId": self.asset_id,
            "type": 2,  # 2 = Market order
            "side": side_code,  # 0 = Bid (buy), 1 = Ask (sell)
            "size": self.order_size,
            "limitPrice": None,
            "stopPrice": None,
            "trailPrice": None,
            "customTag": None,
        }

        try:
            response = topstepx_post(url, headers=headers, payload=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                if result.get("success", False):
                    order_id = result.get("orderId")
                    print(f"✅ Order placed successfully. Order ID: {order_id}")
                    print(f"   Side: {self.side}, Size: {self.order_size}, Entry: {self.entry_price:.5f}")
                    print(f"   TP: {self.take_profit:.5f}, SL: {self.trailing_stop_loss:.5f}")
                    return {
                        'success': True,
                        'order_id': order_id,
                        'message': 'Order placed successfully'
                    }
                else:
                    error_msg = result.get("errorMessage", "Unknown error")
                    print(f"❌ Order placement failed: {error_msg}")
                    return {
                        'success': False,
                        'order_id': None,
                        'message': error_msg
                    }
            else:
                error_msg = f"HTTP {response.status_code}: {response.text}"
                print(f"❌ Order placement failed: {error_msg}")
                return {
                    'success': False,
                    'order_id': None,
                    'message': error_msg
                }

        except ImportError:
            return {
                'success': False,
                'order_id': None,
                'message': 'requests library not installed. Install with: pip install requests'
            }
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            print(error_msg)
            return {
                'success': False,
                'order_id': None,
                'message': error_msg
            }


    def close_order(self):
        url = "https://api.topstepx.com/api/Position/closeContract"

        headers = {
            "Authorization": f"Bearer {self.auth_token}",
            "Accept": "application/json",
            "Content-Type": "application/json"
        }

        payload = {
            "accountId": self.account_id,
            "contractId": self.asset_id
        }

        try:
            response = topstepx_post(url, headers=headers, payload=payload, timeout=30)
            response.raise_for_status()
            print(f"🛑 Order closed")
            return response.json()
        except Exception as e:
            print(f"❌ Failed to close position: {e}")
            raise Exception(f"Unexpected response: {e}")


class ProjectX_Strategy(FVG_Strategy):
    Order = ProjectX_Order

    auth_token: str
    account_id: str
    account_name: str
    asset: str

    def __init__(self, asset_tuple):
        print("layer 3 init ran!")
        os.makedirs(PROJECTX_RUNTIME_DIR, exist_ok=True)
        self.auth_token = None
        self.account_id = None
        self._contract_specs = None
        self.username = None
        self.api_key = None
        self._fetch_failures = 0
        self._last_reauth_ts = 0.0

        self.asset = asset_tuple[0]
        self.timeframe = asset_tuple[1]
        self.account_name = asset_tuple[2]

        filename = self._safe_filename(f"{self.asset}-{self.timeframe}-{self.account_name}")
        self.csv_filename = os.path.join(PROJECTX_RUNTIME_DIR, f"{filename}.csv")
        self.metadata_filename = os.path.join(PROJECTX_RUNTIME_DIR, f"{filename}.json")

    def _safe_filename(self, name: str) -> str:
        # Avoid Windows reserved device names like CON, PRN, AUX, NUL, COM1, LPT1
        safe = name.replace("CON.", "")
        safe = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in safe)
        upper = safe.upper()
        reserved = {
            "CON", "PRN", "AUX", "NUL",
            "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9",
            "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9",
        }
        if upper in reserved or upper.split(".")[0] in reserved:
            safe = f"PX_{safe}"
        return safe

    def init_api(self, auth_token):
        self.set_token(auth_token)
        super().__init__()
        self.require_intrabar_entry = True
        # Reload token after metadata load (metadata may override auth_token).
        self.set_token(auth_token)
        self.account_id = get_account_id(self.auth_token, self.account_name)
        self.account_balance = get_account_balance(self.account_id, self.auth_token)
        token_state = "set" if self.auth_token else "missing"
        print(f"strat initializeds (auth_token {token_state})")

    def load_metadata(self) -> bool:
        ok = super().load_metadata()
        if ok:
            # Never trust persisted tokens; always re-authenticate.
            self.auth_token = None
        return ok

    def save_data(self) -> None:
        token = getattr(self, "auth_token", None)
        try:
            self.auth_token = None
            super().save_data()
        finally:
            self.auth_token = token

    
    def set_token(self, token):
        self.auth_token = token

    def api_order_kwargs(self) -> dict:
        return {"account_id": self.account_id, "asset_id": self.asset, "auth_token": self.auth_token}


    def get_assets(self):
        url = "https://api.topstepx.com/api/Contract/available"

        headers = {
            "Authorization": f"Bearer {self.auth_token}",
            "accept": "text/plain",
            "Content-Type": "application/json"
        }

        payload = {
            "live": False
        }

        response = topstepx_post(url, headers=headers, payload=payload, timeout=30)
        response.raise_for_status()
        
        return response.json()["contracts"]

    def gather_data(self) -> pd.DataFrame:
        data = load_data(self.asset, self.timeframe, data_dir=PROJECTX_RUNTIME_DIR)
        if data is not None:
            return data
        data = fetch_data(self.asset, self.timeframe, 100, self.auth_token)
        if data is None:
            print("⚠️  Initial data fetch failed; starting with empty dataset.")
            return pd.DataFrame()
        return data

    def fetch_new_data(self):
        new_row = fetch_data(self.asset, self.timeframe, 1, self.auth_token)
        if new_row is None:
            print("⚠️  No new data available; waiting for next bar.")
            self._fetch_failures += 1
            if self._fetch_failures >= 3 and self.username and self.api_key:
                now = time.time()
                if now - self._last_reauth_ts > 300:
                    try:
                        print("🔄 Re-authenticating after repeated fetch failures...")
                        new_token = init_api(self.username, self.api_key)
                        self.set_token(new_token)
                        self._last_reauth_ts = now
                        self._fetch_failures = 0
                    except Exception as exc:
                        print(f"⚠️ Re-auth failed: {exc}")
            return False
        self._fetch_failures = 0
        if new_row["timestamp"].iloc[-1] > self.data["timestamp"].iloc[-1]:
            self.cur_close = new_row["close"].iloc[-1]
            self.cur_volume = new_row["volume"].iloc[-1]
            self.data = pd.concat([self.data, new_row], ignore_index=True).iloc[-100:] # last 100
            return True
        return False

    def bar_iteration(self):
        with self._lock:
            if not self.fetch_new_data():
                return
            if self.data is None or len(self.data) < 5:
                return
            if self._check_max_drawdown(self._get_current_timestamp(), float(self.cur_close)):
                self.save_data()
                return
            # Enforce session-based time rules (entry cutoff and forced close)
            self._apply_session_time_guards()
            self.update_indicators()
            self.add_fvg_zones()
            self.entry_logic()
            self.update_stops()
            self.save_data()
            print(f"\n⏰ New bar - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} close: {self.cur_close}")


    def fetch_htf_data(self) -> pd.DataFrame:
        htf_tf = str(INPUTS.HTF_TF)
        if htf_tf.isdigit():
            htf_tf = f"{htf_tf}min"

        num_bars = max(INPUTS.EMA_PERIOD + 51, 101)

        data = load_data(self.asset, htf_tf, data_dir=PROJECTX_RUNTIME_DIR)
        if data is None or len(data) == 0:
            data = fetch_data(self.asset, htf_tf, num_bars, self.auth_token)

        if data is None or len(data) == 0:
            return pd.DataFrame()

        if hasattr(self, "cur_close"):
            data = data.copy()
            data.loc[data.index[-1], "close"] = float(self.cur_close)

        return data


    def check_daily_trade_limit(self):
        """Check if maximum daily trades has been reached"""
        today = datetime.now().date()
        
        if self.last_trade_date != str(today):
            # Reset counter for new day
            self.daily_trades_count = 0
            self.last_trade_date = str(today)
        
        return self.daily_trades_count < INPUTS.MAX_DAILY_TRADES

    def calculate_order_size(self, atr, stop_distance_atr_mult):
        """Calculate position size based on risk management"""
        if INPUTS.USE_FIXED_LOT:
            return INPUTS.FIXED_LOT
        
        # Calculate lot size based on risk percentage
        # This is a simplified calculation - adjust based on your broker's requirements
        risk_amount = self.account_balance * (INPUTS.RISK_PERCENT / 100)
        stop_distance = atr * stop_distance_atr_mult
        
        if stop_distance > 0:
            lot_size = risk_amount / stop_distance
            # Round to appropriate precision
            lot_size = round(lot_size, 2)
            return max(0.01, min(lot_size, 100))  # Ensure reasonable bounds
        
        return INPUTS.ORDER_SIZE

    def _contracts_csv_path(self) -> str | None:
        candidates = [
            os.path.join(PROJECTX_RUNTIME_DIR, "contracts.csv"),
            os.path.join(os.getcwd(), "contracts.csv"),
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "contracts.csv")),
        ]
        for path in candidates:
            if os.path.exists(path):
                return path
        return None

    def _load_contract_specs(self) -> dict:
        if self._contract_specs is not None:
            return self._contract_specs
        path = self._contracts_csv_path()
        if path is None:
            self._contract_specs = {}
            return self._contract_specs
        try:
            df = pd.read_csv(path)
        except Exception:
            self._contract_specs = {}
            return self._contract_specs
        if "id" not in df.columns:
            self._contract_specs = {}
            return self._contract_specs
        df = df.set_index("id", drop=False)
        self._contract_specs = df.to_dict(orient="index")
        return self._contract_specs

    def _get_contract_tick_info(self, contract_id: str) -> tuple[float | None, float | None]:
        specs = self._load_contract_specs()
        row = specs.get(contract_id)
        if not row:
            return None, None
        try:
            tick_size = float(row.get("tickSize")) if row.get("tickSize") is not None else None
            tick_value = float(row.get("tickValue")) if row.get("tickValue") is not None else None
        except (TypeError, ValueError):
            return None, None
        if tick_size is None or tick_value is None or tick_size == 0:
            return None, None
        return tick_size, tick_value

    def calculate_trade_pnl(
        self,
        order: Order,
        exit_price: float | None,
        exit_timestamp: datetime | None = None,
    ) -> float:
        entry_price = getattr(order, "avg_entry_price", None) or order.entry_price
        if entry_price is None or exit_price is None:
            return 0.0
        try:
            entry_price = float(entry_price)
            exit_price = float(exit_price)
            size = float(order.order_size or 0.0)
        except (TypeError, ValueError):
            return 0.0
        if size <= 0:
            return 0.0

        contract_id = getattr(order, "asset_id", None) or self.asset
        tick_size, tick_value = self._get_contract_tick_info(contract_id)
        if tick_size is None or tick_value is None:
            if order.side == "BUY":
                return (exit_price - entry_price) * size
            return (entry_price - exit_price) * size

        price_move = exit_price - entry_price
        ticks = price_move / tick_size
        if order.side == "SELL":
            ticks = -ticks
        gross = ticks * tick_value * size
        fee = 3.5 * size
        return gross - fee

    def _get_unrealized_pnl(self, current_price: float | None) -> float:
        if not self.active_orders or current_price is None:
            return 0.0
        try:
            cur_price = float(current_price)
        except (TypeError, ValueError):
            return 0.0
        total = 0.0
        for order in list(self.active_orders):
            entry_price = getattr(order, "avg_entry_price", None) or order.entry_price
            if entry_price is None:
                continue
            try:
                entry_price = float(entry_price)
                size = float(order.order_size or 0.0)
            except (TypeError, ValueError):
                continue
            if size <= 0:
                continue
            contract_id = getattr(order, "asset_id", None) or self.asset
            tick_size, tick_value = self._get_contract_tick_info(contract_id)
            if tick_size is None or tick_value is None:
                if order.side == "BUY":
                    total += (cur_price - entry_price) * size
                else:
                    total += (entry_price - cur_price) * size
                continue
            price_move = cur_price - entry_price
            ticks = price_move / tick_size
            if order.side == "SELL":
                ticks = -ticks
            total += ticks * tick_value * size
        return total

    def subscribe_to_price_updates(self):
        print("subscribed")
        while True:
            sleep(10)
            new_row = fetch_data(self.asset, "1min", 1, self.auth_token, include_partial_bar=True)
            self.update_price(new_row)

    def start_bar_iterations(self):
        sleep_until_next_boundary(self.timeframe)
        while True:
            try:
                self.bar_iteration()
                sleep_until_next_boundary(self.timeframe)
                
            except Exception as e:
                print(f"❌ Error in bar iteration: {e}")
                sleep(60)

    def run(self):
        """Start the trading bot"""
        print(f"\n{'='*60}")
        print(f"🤖 Trading Bot Started for {self.asset}")
        print(f"{'='*60}")
        print(f"Timeframe: {self.timeframe}")
        print(f"HTF Bias: {INPUTS.HTF_TF}min | EMA Period: {INPUTS.EMA_PERIOD}")
        tick_size, tick_value = self._get_contract_tick_info(self.asset)
        if tick_size is not None and tick_value is not None:
            print(
                f"✅ Tick data loaded: size={tick_size} value={tick_value} "
                f"for {self.asset}"
            )
        else:
            print(f"⚠️  Tick data not available for {self.asset}")


        t1 = threading.Thread(target=self.start_bar_iterations)
        t2 = threading.Thread(target=self.subscribe_to_price_updates)
        t1.start()
        t2.start()


class ProjectX_AccountRunner:
    def __init__(
        self,
        api_config: dict,
        check_interval: int = 5,
        enable_limits: bool = INPUTS.ENABLE_DAILY_PNL_LIMITS,
        max_daily_gain: float = INPUTS.MAX_DAILY_GAIN,
        max_daily_loss: float = INPUTS.MAX_DAILY_LOSS,
    ):
        self.api_config = api_config
        self.check_interval = check_interval
        self.enable_limits = enable_limits
        self.max_daily_gain = max_daily_gain
        self.max_daily_loss = max_daily_loss
        self.strategies = [
            ProjectX_Strategy(asset_tuple)
            for asset_tuple in api_config.get("assets_list", [])
        ]
        for strat in self.strategies:
            strat.username = api_config.get("username")
            strat.api_key = api_config.get("api_key")
        self._limit_triggered = False
        self._limit_triggered_date = None

    def start(self):
        token = init_api(self.api_config["username"], self.api_config["api_key"])
        v_thread = threading.Thread(
            target=validation_thread,
            args=(token, self.strategies, self.api_config),
            daemon=True,
        )
        v_thread.start()

        for strat in self.strategies:
            t = threading.Thread(
                target=run_strat,
                args=(strat, token,),
                daemon=True,
            )
            t.start()

        monitor = threading.Thread(target=self._monitor_pnl, daemon=True)
        monitor.start()

    def _monitor_pnl(self):
        while True:
            if not self.enable_limits:
                sleep(self.check_interval)
                continue

            today = datetime.now().date()
            if self._limit_triggered_date != today:
                self._limit_triggered = False
                self._limit_triggered_date = today
                for strat in self.strategies:
                    strat.trading_paused = False
                    strat.daily_realized_pnl = 0.0
                    strat.last_pnl_date = str(today)

            total_pnl = 0.0
            for strat in self.strategies:
                unrealized = self._calculate_strategy_unrealized_pnl(strat)
                total_pnl += float(getattr(strat, "daily_realized_pnl", 0.0)) + unrealized

            hit_gain = self.max_daily_gain and total_pnl >= self.max_daily_gain
            hit_loss = self.max_daily_loss and total_pnl <= -self.max_daily_loss
            if (not self._limit_triggered) and (hit_gain or hit_loss):
                self._limit_triggered = True
                for strat in self.strategies:
                    strat.trading_paused = True
                    cur_price = getattr(strat, "cur_close", None)
                    if cur_price is None:
                        continue
                    try:
                        cur_ts = strat._get_current_timestamp()
                    except Exception:
                        cur_ts = datetime.now()
                    try:
                        strat._close_all_positions(float(cur_price), cur_ts, "daily_pnl_limit")
                    except Exception:
                        pass
                print(
                    "⚠️ Daily account PnL limit reached "
                    f"({total_pnl:.2f}) for {self.api_config.get('username', 'account')}"
                )

            sleep(self.check_interval)

    def _calculate_strategy_unrealized_pnl(self, strat: ProjectX_Strategy) -> float:
        cur_price = getattr(strat, "cur_close", None)
        if cur_price is None or not strat.active_orders:
            return 0.0
        total = 0.0
        for order in list(strat.active_orders):
            entry_price = getattr(order, "avg_entry_price", None) or order.entry_price
            if entry_price is None:
                continue
            try:
                entry_price = float(entry_price)
                cur_price_val = float(cur_price)
                size = float(order.order_size or 0.0)
            except (TypeError, ValueError):
                continue
            if size <= 0:
                continue
            contract_id = getattr(order, "asset_id", None) or strat.asset
            tick_size, tick_value = strat._get_contract_tick_info(contract_id)
            if tick_size is not None and tick_value is not None and tick_size != 0:
                price_move = cur_price_val - entry_price
                ticks = price_move / float(tick_size)
                if order.side == "SELL":
                    ticks = -ticks
                total += ticks * float(tick_value) * size
            else:
                if order.side == "BUY":
                    total += (cur_price_val - entry_price) * size
                else:
                    total += (entry_price - cur_price_val) * size
        return total


def run_strat(strat: ProjectX_Strategy, token):
    strat.init_api(token)
    strat.run()

def validation_thread(
    auth_token: str,
    strategies: list[ProjectX_Strategy],
    api_config: dict | None = None,
    refresh_interval: int = 50000,
):
    print("starting validation thread...")
    while True:
        sleep(refresh_interval)
        res = validate_token(auth_token)
        if not res or res.get("success") is False:
            print("token update failed, API connection might fail soon...")
            if res:
                print(res.get("message"))
            if api_config:
                try:
                    auth_token = init_api(api_config["username"], api_config["api_key"])
                    print("✅ Re-authenticated after validation failure.")
                except Exception as exc:
                    print(f"❌ Re-authentication failed: {exc}")
                    continue
            else:
                continue
        else:
            new_token = (
                res.get("newToken")
                or res.get("new_token")
                or res.get("token")
                or auth_token
            )
            if new_token != auth_token:
                auth_token = new_token
                print("Sucessfully updated connection token")

        for strat in strategies:
            strat.set_token(auth_token)


if __name__ == "__main__":
    setup_global_logging(PROJECTX_LOG_PATH)
    for api in INPUTS.APIS.values():
        global_token = init_api(api["username"], api["api_key"])

        if INPUTS.UPDATE_CONTRACT_LIST:
            strat = ProjectX_Strategy(api["assets_list"][0])
            strat.username = api.get("username")
            strat.init_api(global_token)
            data = strat.get_assets()
            data = pd.DataFrame(data)
            data.to_csv(os.path.join(PROJECTX_RUNTIME_DIR, "contracts.csv"), index=False)
            print("Contract list updated successfully!!")
        elif INPUTS.SHOW_ACCOUNTS:
            strat = ProjectX_Strategy(api["assets_list"][0])
            strat.username = api.get("username")
            strat.set_token(global_token)
            print(get_account_id(strat.auth_token, show=True))
        elif INPUTS.SHOW_TRADES:
            # Fetch and print recent trades for the first account in this API config
            first_asset = api["assets_list"][0]
            account_name = first_asset[2]
            account_id = get_account_id(global_token, account_name=account_name)

            # Default to last 7 days of trades
            now_utc = datetime.utcnow()
            start_ts = now_utc - timedelta(days=7)
            end_ts = now_utc

            print(f"Fetching trades for account {account_name} (ID {account_id}) "
                  f"from {start_ts.isoformat()} to {end_ts.isoformat()}...")
            try:
                res = search_trades(
                    auth_token=global_token,
                    account_id=account_id,
                    start_timestamp=start_ts,
                    end_timestamp=end_ts,
                )
                print("Trade search result:")
                print(res)
            except Exception as exc:
                print(f"❌ Failed to fetch trades: {exc}")

        else:
            runner = ProjectX_AccountRunner(api)
            runner.start()


    while True:
        sleep(5)
            