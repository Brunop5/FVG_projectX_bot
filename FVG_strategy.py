import math
import pandas as pd
import threading

from abc import abstractmethod
from datetime import datetime, timedelta
import os

from .helping_functions.indicators import get_atr
from .helping_functions.indicators import sma
from .helping_functions.indicators import ema
from .helping_functions.indicators import crossover
from .helping_functions.indicators import crossunder
from .helping_functions.pyramiding import (
    AddOnSpec,
    ClientAtrPyramidingPolicy,
    NoPyramidingPolicy,
)

from strategyTemplate import Strategy, Order


# all account names can be whatever ("pc account", "whorehouse account",...)
# every account has to have the same structure:
# "username": "actual_username",
# "api_key": "actual_api_key",
# "assets_list":[the actuall asset list you want, like you see below. the same structure]

# if "username", "api_key", "assets_list" or even the actual ones are spelled wrong, it wont work


# if "username", "api_key", "assets_list" or even the actual ones are spelled wrong, it wont work

APIS = {
    "Personal account": {
        "username": "REDACTED_USERNAME",
        "api_key": "REDACTED_API_KEY=",
        "assets_list": [
            ("CON.F.US.MGC.J26", "15min", "50KTC-V2-252499-69493223"),
            ("CON.F.US.RTY.H26", "15min", "50KTC-V2-252499-69493223"),
            ("CON.F.US.MCLE.H26", "15min", "50KTC-V2-252499-69493223"),
            ("CON.F.US.M2K.H26", "15min", "50KTC-V2-252499-69493223"),
        ]
    },

    "Wifes account": {
        "username": "afribuymoz@gmail.com",
        "api_key": "NEV2P9zTW7+cTNDi/yW8rcjn22EAPH8/t6SgxObrVvM=",
        "assets_list": [
            ("CON.F.US.MNQ.H26", "15min", "50KTC-V2-498538-71564652"),
            ("CON.F.US.SIL.H26", "15min", "50KTC-V2-498538-71564652"),
            ("CON.F.US.MYM.H26", "15min", "50KTC-V2-498538-71564652"),
            ("CON.F.US.MES.H26", "15min", "50KTC-V2-498538-71564652"),
            ("CON.F.US.M2K.H26", "15min", "50KTC-V2-498538-71564652")
        ]
    }
}


# ====================
# if true, updates contracts.csv - this should be done at least monthly
UPDATE_CONTRACT_LIST = False

# if true it will print the list of valid accounts for this api key
SHOW_ACCOUNTS = False
# ======== if any of those two is true, it will run the option, but not the strategy


# ==================== CONFIGURATION PARAMETERS ====================
# Display Settings
FVG_HISTORY_NBR = 1              # Number of FVGs to work with
MIN_FVG_POWER_PCT = 0.01          # Min FVG Power % (formerly MinFVGPowerPct)

# Timeframe and Trend Settings
HTF_TF = "30"                     # HTF Bias (4H) - PERIOD_H4
EMA_PERIOD = 15                    # EMA Period for trend detection
VOLUME_MULTIPLIER = 1.2
USE_VOLUME_CHECK = True            # If False, volume check is skipped in marketOK calculation
VOLUME_DATA_START_TIMESTAMP = 1755464400000  # Timestamp where reliable volume data starts (ms)
START_FROM_VOLUME_TIMESTAMP = False  # None = auto (True if USE_VOLUME_CHECK, False otherwise). Set to True/False to override

# ATR and Risk Management
ATR_PERIOD = 23                    # ATR Period (min 1)
SL_MULTIPLIER = 6.5               # SL ATR Multiplier (formerly SL_ATR_Mult)
TP_MULTIPLIER = 7                # TP ATR Multiplier (formerly TP_ATR_Mult)

# Trailing Stop Settings
USE_TRAILING = True                # use trailing stop (formerly UseTrailing)
TRAIL_OFFSET_MULT = 4            # Trailing Offset ATR Multiplier (formerly TrailATRMult)

# Position Management
HOLD_UNTIL_OPPOSITE = True         # Hold Until Opposite BOS/CHoCH

# Lot Size and Risk Settings
USE_FIXED_LOT = True        # Use fixed lot size (formerly UseFixedLot)
FIXED_LOT = 6                   # Fixed lot size (formerly FixedLot)
RISK_PERCENT = 1.0                 # Risk percentage per trade (formerly RiskPercent)
ORDER_SIZE = 1                     # Default order size (overridden by risk calculation if not USE_FIXED_LOT)

# Partial close sizing and ATR steps
SPLIT_ORDERS_ENABLED = True        # If False, place a single order with FIXED_LOT
EACH_TRADE_SIZE = 1                # Size per child order when splitting FIXED_LOT
PARTIAL_TP_ATR_STEP = 1  # ATR step size for favorable partial closes
PARTIAL_SL_ATR_STEP = 2  # ATR step size for adverse partial closes
PARTIAL_TP_CLOSE_SIZE = 1  # Size to close per favorable step (multiple of EACH_TRADE_SIZE)
PARTIAL_SL_CLOSE_SIZE = 2  # Size to close per adverse step (multiple of EACH_TRADE_SIZE)
ENABLE_PARTIAL_TP = True
ENABLE_PARTIAL_SL = True

# Daily Trading Limits
MAX_DAILY_TRADES = 3
ENABLE_DAILY_PNL_LIMITS = True
MAX_DAILY_GAIN = 1480
MAX_DAILY_LOSS = 1000

ALLOW_INTRACANDLE_ENTRY = True
DEBUG_STOPS = False
DEBUG_PYRAMIDING = False

# Max drawdown protection (per strategy instance)
MAX_DRAWDOWN_ENABLED = True
MAX_DRAWDOWN_PCT = 50.0

# Pyramiding (client mode)
ALLOW_PYRAMIDING = True
PYR_ATR_STEP = 1.0
PYR_ADD_ON_SIZE = 1
PYR_MAX_ADDS = 10

def quiet_log(msg):
    pass


class FVG_Order(Order):
    entry_atr: float
    pyramid_count: int
    next_add_price: float | None
    avg_entry_price: float | None

    def __init__(
        self, entry_atr, **kwargs):
        super().__init__(**kwargs)

        self.entry_atr = entry_atr
        self.pyramid_count = 0
        self.next_add_price = None
        self.avg_entry_price = self.entry_price
        self.group_id = None
        self.group_seq = None
        self.entry_reference_price = self.entry_price

    def add_to_position(self, add_size: float, log=print):
        if add_size is None or add_size <= 0:
            return {"success": False, "message": "Invalid add-on size"}
        original_size = float(self.order_size or 0.0)
        add_size_int = self._normalize_order_size(add_size)
        self.order_size = add_size_int
        result = self.place_order()
        success = isinstance(result, dict) and result.get("success", False)
        if result is None:
            success = True
        if success:
            new_size = original_size + float(add_size_int)
            if new_size > 0 and self.avg_entry_price is not None:
                self.avg_entry_price = (
                    (self.avg_entry_price * original_size)
                    + (self.entry_price * float(add_size_int))
                ) / new_size
            self.order_size = self._normalize_order_size(new_size)
            log(
                f"➕ Add-on placed: size={add_size_int} new_size={self.order_size} "
                f"side={self.side} entry={self.entry_price}"
            )
            return {"success": True, "new_size": self.order_size, "result": result}
        self.order_size = original_size
        return {"success": False, "result": result}


    def check_close_conditions(self, log=print, **kwargs) -> bool:
        current_price = kwargs["current_price"]
        last_long = kwargs["last_long"]
        last_short = kwargs["last_short"] 
        isBOS = kwargs["isBOS"]
        isCHOCH = kwargs["isCHOCH"]
        current_high = kwargs.get("current_high")
        current_low = kwargs.get("current_low")
        if True:
            log(
                f"🧪 check_close_conditions: side={self.side} "
                f"price={current_price} tsl={self.trailing_stop_loss} "
                f"sl={self.stop_loss} tp={self.take_profit} "
                f"last_long={last_long} last_short={last_short} "
                f"isBOS={isBOS} isCHOCH={isCHOCH}"
            )

        if self.side == "BUY":
            stop_check_price = current_low if current_low is not None else current_price
            tp_check_price = current_high if current_high is not None else current_price
            if self.trailing_stop_loss is not None and stop_check_price <= self.trailing_stop_loss:
                log(f"🛑 Trailing Stop Loss hit for LONG position at {current_price:.5f}")
                self.close_order()
                return True

            if self.stop_loss is not None and stop_check_price <= self.stop_loss:
                log(f"🛑 Stop Loss hit for LONG position at {current_price:.5f}")
                self.close_order()
                return True

            if tp_check_price >= self.take_profit:
                log(f"🎯 Take Profit hit for LONG position at {current_price:.5f}")
                self.close_order()
                return True


        elif self.side == "SELL":
            stop_check_price = current_high if current_high is not None else current_price
            tp_check_price = current_low if current_low is not None else current_price
            if self.trailing_stop_loss is not None and stop_check_price >= self.trailing_stop_loss:
                log(f"🛑 Trailing Stop Loss hit for SHORT position at {current_price:.5f}")
                self.close_order()
                return True

            if self.stop_loss is not None and stop_check_price >= self.stop_loss:
                log(f"🛑 Stop Loss hit for SHORT position at {current_price:.5f}")
                self.close_order()
                return True

            if tp_check_price <= self.take_profit:
                log(f"🎯 Take Profit hit for SHORT position at {current_price:.5f}")
                self.close_order()
                return True


        # === CLOSE ON OPPOSITE BOS/CHoCH ===
        if HOLD_UNTIL_OPPOSITE:
            if last_long and isCHOCH:
                log("🔄 CHoCH detected - Closing LONG position")
                self.close_order()
                return True

            if last_short and isBOS:
                log("🔄 BOS detected - Closing SHORT position")
                self.close_order()
                return True
        
        return False


class FVG_Strategy(Strategy):
    Order = FVG_Order
    timeframe: str  #"15min" or "1h" or such
    metadata_filename: str
    csv_filename: str

    cur_close: float
    cur_volume: float

    isBullishHTF: bool
    isBearishHTF: bool
    marketOK: bool
    bullishPowerOK: bool
    bearishPowerOK: bool
    isBOS: bool
    isCHOCH: bool
    lastBullFvg: bool
    lastBearFvg: bool
    lastPositionWasLong: bool
    lastPositionWasShort: bool

    peak_unrealized_pnl: float
    max_dd_triggered_until: datetime | None

    daily_trades_count: int
    last_trade_date: str | None

    fvg_zones: list[dict]
    
    def __init__(self):
        print("layer 2 init")
        self.isBOS = False
        self.isCHOCH = False

        self.lastBullFvg = False
        self.lastBearFvg = False
        self.lastPositionWasLong = False
        self.lastPositionWasShort = False

        self.daily_trades_count = 0
        self.last_trade_date = None
        self._lock = threading.Lock()  # Protects shared state when bar thread and price-update thread run concurrently

        self.fvg_zones = []
        if not hasattr(self, "debug_pyramiding"):
            self.debug_pyramiding = DEBUG_PYRAMIDING
        if not hasattr(self, "pyramiding") or self.pyramiding is None:
            if ALLOW_PYRAMIDING:
                self.pyramiding = ClientAtrPyramidingPolicy(
                    atr_step=PYR_ATR_STEP,
                    add_on_size=PYR_ADD_ON_SIZE,
                    max_adds=PYR_MAX_ADDS,
                )
            else:
                self.pyramiding = NoPyramidingPolicy()

        self._partial_groups = {}
        self._partial_group_counter = 0
        self._split_order_count = self._validate_split_config()
        self._partial_tp_close_count = (
            self._validate_partial_close_size(
                PARTIAL_TP_CLOSE_SIZE,
                "PARTIAL_TP_CLOSE_SIZE",
            )
            if ENABLE_PARTIAL_TP
            else 0
        )
        self._partial_sl_close_count = (
            self._validate_partial_close_size(
                PARTIAL_SL_CLOSE_SIZE,
                "PARTIAL_SL_CLOSE_SIZE",
            )
            if ENABLE_PARTIAL_SL
            else 0
        )
        self.peak_unrealized_pnl = 500
        self.max_dd_triggered_until = None

        super().__init__()
        if not hasattr(self, "daily_realized_pnl"):
            self.daily_realized_pnl = 500
        if not hasattr(self, "last_pnl_date"):
            self.last_pnl_date = None

        self.cur_close = self.data["close"].iloc[-1]
        self.cur_volume = self.data["volume"].iloc[-1]
        

        self.update_indicators()
        self.add_fvg_zones()

    def _validate_split_config(self) -> int:
        if not SPLIT_ORDERS_ENABLED:
            return 1
        if not USE_FIXED_LOT:
            raise ValueError(
                "Partial close logic requires USE_FIXED_LOT=True to split orders."
            )
        if EACH_TRADE_SIZE is None or EACH_TRADE_SIZE <= 0:
            raise ValueError("EACH_TRADE_SIZE must be a positive number.")
        if FIXED_LOT is None or FIXED_LOT <= 0:
            raise ValueError("FIXED_LOT must be a positive number.")
        count = FIXED_LOT / EACH_TRADE_SIZE
        if not math.isclose(count, round(count), rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError(
                f"FIXED_LOT ({FIXED_LOT}) must be an exact multiple of "
                f"EACH_TRADE_SIZE ({EACH_TRADE_SIZE})."
            )
        count_int = int(round(count))
        if count_int < 1:
            raise ValueError("Split order count must be at least 1.")
        return count_int

    def _validate_partial_close_size(self, size_value: float | None, label: str) -> int:
        if size_value is None:
            size_value = EACH_TRADE_SIZE
        if size_value <= 0:
            raise ValueError(f"{label} must be a positive number.")
        count = size_value / EACH_TRADE_SIZE
        if not math.isclose(count, round(count), rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError(
                f"{label} ({size_value}) must be an exact multiple of "
                f"EACH_TRADE_SIZE ({EACH_TRADE_SIZE})."
            )
        count_int = int(round(count))
        if count_int < 1:
            raise ValueError(f"{label} must be at least one child order.")
        return count_int

    def _next_partial_group_id(self) -> int:
        self._partial_group_counter += 1
        return self._partial_group_counter

    def _get_partial_close_targets(self, current_price: float) -> dict:
        if not self.active_orders:
            return {}
        if (not ENABLE_PARTIAL_TP) and (not ENABLE_PARTIAL_SL):
            return {}
        if PARTIAL_TP_ATR_STEP <= 0 and PARTIAL_SL_ATR_STEP <= 0:
            return {}

        groups = {}
        for order in self.active_orders:
            group_id = getattr(order, "group_id", None)
            if group_id is None:
                continue
            groups.setdefault(group_id, []).append(order)

        close_map = {}
        for group_id, orders in groups.items():
            state = self._partial_groups.get(group_id)
            if state is None:
                anchor = orders[0]
                state = {
                    "entry_price": getattr(anchor, "entry_reference_price", anchor.entry_price),
                    "entry_atr": anchor.entry_atr,
                    "side": anchor.side,
                    "tp_steps_closed": 0,
                    "sl_steps_closed": 0,
                }
                self._partial_groups[group_id] = state

            entry_price = state["entry_price"]
            entry_atr = state["entry_atr"]
            if entry_atr is None or entry_atr <= 0:
                continue

            if state["side"] == "BUY":
                favorable_move = current_price - entry_price
                adverse_move = entry_price - current_price
            else:
                favorable_move = entry_price - current_price
                adverse_move = current_price - entry_price

            sorted_orders = sorted(orders, key=lambda o: getattr(o, "group_seq", 0))
            available = [o for o in sorted_orders if o not in close_map]

            if ENABLE_PARTIAL_TP and PARTIAL_TP_ATR_STEP > 0 and favorable_move > 0:
                step_size = entry_atr * PARTIAL_TP_ATR_STEP
                if step_size > 0:
                    steps_reached = int(favorable_move // step_size)
                    to_close = steps_reached - state["tp_steps_closed"]
                    if to_close > 0 and available:
                        for _ in range(to_close):
                            if not available:
                                break
                            close_count = min(self._partial_tp_close_count, len(available))
                            for order in available[:close_count]:
                                close_map[order] = "partial_tp"
                            available = available[close_count:]
                            state["tp_steps_closed"] += 1

            if ENABLE_PARTIAL_SL and PARTIAL_SL_ATR_STEP > 0 and adverse_move > 0 and available:
                step_size = entry_atr * PARTIAL_SL_ATR_STEP
                if step_size > 0:
                    steps_reached = int(adverse_move // step_size)
                    to_close = steps_reached - state["sl_steps_closed"]
                    if to_close > 0:
                        for _ in range(to_close):
                            if not available:
                                break
                            close_count = min(self._partial_sl_close_count, len(available))
                            for order in available[:close_count]:
                                close_map[order] = "partial_sl"
                            available = available[close_count:]
                            state["sl_steps_closed"] += 1

        return close_map

    def _cleanup_partial_groups(self):
        if not self.active_orders:
            self._partial_groups = {}
            return
        active_ids = {getattr(order, "group_id", None) for order in self.active_orders}
        for group_id in list(self._partial_groups.keys()):
            if group_id not in active_ids:
                self._partial_groups.pop(group_id, None)

    def _get_current_timestamp(self) -> datetime:
        if hasattr(self, "_current_dt") and self._current_dt is not None:
            return self._current_dt
        if isinstance(getattr(self, "data", None), pd.DataFrame) and "timestamp" in self.data.columns:
            ts = self.data["timestamp"].iloc[-1]
            if pd.isna(ts):
                return datetime.now()
            try:
                return pd.to_datetime(ts, utc=True).to_pydatetime(warn=False)
            except Exception:
                return datetime.now()
        return datetime.now()

    def _reset_daily_pnl_if_needed(self, current_timestamp: datetime) -> None:
        today = str(current_timestamp.date())
        if self.last_pnl_date != today:
            self.daily_realized_pnl = 0.0
            self.last_pnl_date = today

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
        if order.side == "BUY":
            return (exit_price - entry_price) * size
        return (entry_price - exit_price) * size

    def _record_trade(
        self,
        order: Order,
        exit_price: float | None,
        exit_timestamp: datetime | None = None,
    ) -> float:
        if exit_timestamp is None:
            exit_timestamp = self._get_current_timestamp()
        self._reset_daily_pnl_if_needed(exit_timestamp)
        pnl = self.calculate_trade_pnl(order, exit_price, exit_timestamp)
        self.daily_realized_pnl += pnl
        if hasattr(order, "realized_pnl"):
            order.realized_pnl = pnl
        return pnl

    def _get_unrealized_pnl(self, current_price: float | None) -> float:
        if not self.active_orders:
            return 0.0
        if current_price is None:
            return 0.0
        pnl = 0.0
        for order in list(self.active_orders):
            entry_price = getattr(order, "avg_entry_price", None) or order.entry_price
            if entry_price is None:
                continue
            try:
                entry_price = float(entry_price)
                cur_price = float(current_price)
                size = float(order.order_size or 0.0)
            except (TypeError, ValueError):
                continue
            if order.side == "BUY":
                pnl += (cur_price - entry_price) * size
            else:
                pnl += (entry_price - cur_price) * size
        return pnl

    def _is_drawdown_lockout(self, current_timestamp: datetime) -> bool:
        if self.max_dd_triggered_until is None:
            return False
        if isinstance(self.max_dd_triggered_until, datetime):
            lockout_date = self.max_dd_triggered_until.date()
        elif isinstance(self.max_dd_triggered_until, str):
            try:
                lockout_date = datetime.fromisoformat(self.max_dd_triggered_until).date()
            except ValueError:
                lockout_date = None
        else:
            lockout_date = self.max_dd_triggered_until
        if lockout_date is None:
            self.max_dd_triggered_until = None
            return False
        if current_timestamp.date() >= lockout_date:
            self.max_dd_triggered_until = None
            self.peak_unrealized_pnl = 0.0
            return False
        return True

    def _close_all_positions(self, current_price: float, current_timestamp: datetime, reason: str):
        for order in list(self.active_orders):
            if hasattr(order, "_last_price"):
                order._last_price = float(current_price)
            if hasattr(order, "_last_timestamp"):
                order._last_timestamp = current_timestamp
            if hasattr(order, "exit_reason"):
                order.exit_reason = reason
            order.close_order()
            if hasattr(self, "_record_trade"):
                try:
                    self._record_trade(order, current_price, current_timestamp)
                except Exception:
                    pass
        self.active_orders = []
        self.lastPositionWasLong = False
        self.lastPositionWasShort = False

    def _check_max_drawdown(self, current_timestamp: datetime, current_price: float) -> bool:
        if not MAX_DRAWDOWN_ENABLED:
            return False
        if self._is_drawdown_lockout(current_timestamp):
            return True

        unrealized_pnl = self._get_unrealized_pnl(current_price)
        if unrealized_pnl > self.peak_unrealized_pnl:
            self.peak_unrealized_pnl = unrealized_pnl
            return False
        if self.peak_unrealized_pnl <= 0:
            return False

        drawdown_pct = ((self.peak_unrealized_pnl - unrealized_pnl) / self.peak_unrealized_pnl) * 100.0
        if drawdown_pct >= MAX_DRAWDOWN_PCT:
            self._close_all_positions(current_price, current_timestamp, "max_drawdown")
            self.max_dd_triggered_until = current_timestamp.date() + timedelta(days=1)
            return True
        return False


    @abstractmethod
    def api_order_kwargs(self) -> dict:
        """
        Returns required kwargs for successful order placement with chosen broker.
        Those are set to be attributes of the third layer class
        For backtest class this returns empty dict
        """
        pass

    @abstractmethod
    def fetch_htf_data(self) -> pd.DataFrame:
        """
        Fetches a pd.dataframe of higher timeframe data
        Needs to fetch HTF_TF, and EMA_PERIOD from FVG_strategy.py
        to figure out how much and what data to fetch
        """
        pass

    @abstractmethod
    def check_daily_trade_limit(self):
        pass

    @abstractmethod
    def subscribe_to_price_updates(self):
        """
        Subscribes (or starts with while True cycle) to frequent price updates
        on every update it should run self.update_price()
        """
        pass

    @abstractmethod
    def start_bar_iterations(self):
        """
        Starts a thread with while True for bar_iteration()
        This is separated to API specific class just to allow backtest implementation
        """
        pass

    @abstractmethod
    def run(self):
        """
        run first iteration
        and then start two threads:
         - one for bar iteration
         - one for frequent price updates

        Not implemented in FVG to allow backtest flexibility
        """
        pass



    def update_price(self, new_row: pd.DataFrame):
        """
        A price update as often as possible for the FVG strategy.
        Here it should check stops (and close order if it doesnt set them in order sending)
        and Move trailing stops

        This is just one iteration after data is already fetched.
        Uses _lock so bar-iteration and price-update threads do not race on shared state.
        """
        if new_row is None or len(new_row) == 0:
            return  # Skip this iteration if no data, but keep running
        
        # Check if 'close' column exists
        if 'close' not in new_row.columns:
            print(f"⚠️  Warning: 'close' column not found in data. Available columns: {new_row.columns.tolist()}")
            return

        with self._lock:
            self.cur_close = float(new_row["close"].iloc[-1])
            current_high = (
                float(new_row["high"].iloc[-1])
                if "high" in new_row.columns
                else self.cur_close
            )
            current_low = (
                float(new_row["low"].iloc[-1])
                if "low" in new_row.columns
                else self.cur_close
            )
            ts = new_row["timestamp"].iloc[-1] if "timestamp" in new_row.columns else None
            try:
                self._current_dt = pd.to_datetime(ts, utc=True).to_pydatetime(warn=False) if ts is not None else None
            except Exception:
                self._current_dt = None
            if DEBUG_STOPS:
                print(
                    f"🧪 update_price: close={self.cur_close} "
                    f"orders={len(self.active_orders)} "
                    f"last_long={self.lastPositionWasLong} "
                    f"last_short={self.lastPositionWasShort}"
                )

            current_timestamp = self._get_current_timestamp()
            self._reset_daily_pnl_if_needed(current_timestamp)
            if self._check_max_drawdown(current_timestamp, float(self.cur_close)):
                return

            if len(self.active_orders) > 0:
                high_changed = False
                low_changed = False
                if hasattr(self, "_last_kline_high") and hasattr(self, "_last_kline_low"):
                    try:
                        high_changed = float(current_high) != float(self._last_kline_high)
                        low_changed = float(current_low) != float(self._last_kline_low)
                    except (TypeError, ValueError):
                        high_changed = True
                        low_changed = True
                self._last_kline_high = current_high
                self._last_kline_low = current_low
                if DEBUG_STOPS:
                    print("🧪 update_price: calling update_stops + check_close_conditions")
                if not high_changed and not low_changed:
                    return
                self.update_stops(
                    current_high=current_high,
                    current_low=current_low,
                    high_changed=high_changed,
                    low_changed=low_changed,
                )
                allow_partial_closes = SPLIT_ORDERS_ENABLED and (
                    ENABLE_PARTIAL_TP or ENABLE_PARTIAL_SL
                )
                partial_close_map = (
                    self._get_partial_close_targets(self.cur_close)
                    if allow_partial_closes
                    else {}
                )
                remaining = []
                closed_any = False
                for order in list(self.active_orders):
                    recorded_trade = False
                    closed = order.check_close_conditions(
                        current_price=self.cur_close,
                        current_high=current_high,
                        current_low=current_low,
                        last_long=order.side == "BUY",
                        last_short=order.side == "SELL",
                        isBOS=self.isBOS,
                        isCHOCH=self.isCHOCH,
                    )
                    if (not closed) and allow_partial_closes and order in partial_close_map:
                        if hasattr(order, "exit_reason"):
                            order.exit_reason = partial_close_map[order]
                        if hasattr(order, "_last_price"):
                            order._last_price = float(self.cur_close)
                        if hasattr(order, "_last_timestamp"):
                            order._last_timestamp = current_timestamp
                        order.close_order()
                        try:
                            self._record_trade(order, self.cur_close, current_timestamp)
                            recorded_trade = True
                        except Exception:
                            pass
                        closed = True
                    if closed:
                        if hasattr(order, "_last_price"):
                            order._last_price = float(self.cur_close)
                        if hasattr(order, "_last_timestamp"):
                            order._last_timestamp = current_timestamp
                        if (not recorded_trade) and hasattr(self, "_record_trade"):
                            try:
                                self._record_trade(order, self.cur_close, current_timestamp)
                            except Exception:
                                pass
                        closed_any = True
                        self.pyramiding.on_position_closed(order, self)

                    else:
                        remaining.append(order)
                self.active_orders = remaining
                if self.active_orders:
                    self._apply_pyramiding_add_on(
                        self.cur_close,
                        current_high=current_high,
                        current_low=current_low,
                    )
                if closed_any:
                    self.lastPositionWasLong = any(o.side == "BUY" for o in self.active_orders)
                    self.lastPositionWasShort = any(o.side == "SELL" for o in self.active_orders)
                    self._cleanup_partial_groups()
                    self.save_data()

            elif ALLOW_INTRACANDLE_ENTRY:
                tick_price = float(self.cur_close)
                tick_high = tick_price
                tick_low = tick_price
                if "high" in new_row.columns:
                    tick_high = float(new_row["high"].iloc[-1])
                if "low" in new_row.columns:
                    tick_low = float(new_row["low"].iloc[-1])
                self._intrabar_price = tick_price
                self._intrabar_high = tick_high
                self._intrabar_low = tick_low
                # Update latest bar with intrabar values for live touch checks
                if self.data is not None and len(self.data) > 0:
                    last_index = self.data.index[-1]
                    if "high" in self.data.columns:
                        self.data.at[last_index, "high"] = max(
                            float(self.data.at[last_index, "high"]), tick_high
                        )
                    if "low" in self.data.columns:
                        self.data.at[last_index, "low"] = min(
                            float(self.data.at[last_index, "low"]), tick_low
                        )
                    if "close" in self.data.columns:
                        self.data.at[last_index, "close"] = tick_price
                self.entry_logic()

    def add_fvg_zones(self):
        # === FVG ZONE CREATION (equivalent to the box.new blocks) ===
        if self.bullishPowerOK and self.isBullishHTF and self.marketOK:
            # Bullish FVG uses low[1] as top and high[3] as bottom in Pine
            self.fvg_zones.append(
                {
                    "direction": "bull",
                    "top": self.data["low"].iloc[-1],     # low[1]
                    "bottom": self.data["high"].iloc[-3], # high[3]
                    "mitigated": False,
                }
            )
            print(f"🟢 Bullish FVG detected: {self.data['high'].iloc[-3]:.5f} - {self.data['low'].iloc[-1]:.5f}")


        if self.bearishPowerOK and self.isBearishHTF and self.marketOK:
            # Bearish FVG uses low[3] as top and high[1] as bottom in Pine
            self.fvg_zones.append(
                {
                    "direction": "bear",
                    "top": self.data["low"].iloc[-3],     # low[3]
                    "bottom": self.data["high"].iloc[-1], # high[1]
                    "mitigated": False,
                }
            )
            print(f"🔴 Bearish FVG detected: {self.data['high'].iloc[-1]:.5f} - {self.data['low'].iloc[-3]:.5f}")


        # Limit number of stored FVGs, similar to fvgHistoryNbr trimming in Pine
        if len(self.fvg_zones) > FVG_HISTORY_NBR:
            self.fvg_zones = self.fvg_zones[-FVG_HISTORY_NBR:]


    def _update_trend_indicators(self):
        bars = self.fetch_htf_data()
        htfEMA = ema(bars, EMA_PERIOD)

        if htfEMA is None:
            self.isBullishHTF = False
            self.isBearishHTF = False
        else:
            self.isBullishHTF = self.cur_close > htfEMA
            self.isBearishHTF = self.cur_close < htfEMA

        atrVal = get_atr(self.data, ATR_PERIOD)
        atr_sma = sma(atrVal, 20) if len(atrVal) > 0 else None
        atrOK = atrVal.iloc[-1] > atr_sma if (len(atrVal) > 0 and atr_sma is not None) else False
        
        if USE_VOLUME_CHECK:
            vol_sma = sma(self.data["volume"], 20)
            volOK = self.cur_volume > vol_sma * VOLUME_MULTIPLIER if vol_sma is not None else False
            self.marketOK = volOK and atrOK
        else:
            # Skip volume check, only use ATR
            self.marketOK = atrOK


        self.lastBullFvg = self.data["high"].iloc[-3] < self.data["low"].iloc[-1] and not self.lastBullFvg
        self.lastBearFvg = self.data["low"].iloc[-3] > self.data["high"].iloc[-1] and not self.lastBearFvg

    def _calc_BOS_and_CHOCH(self):
        prevStructureHigh = self.data["high"].iloc[-21:-1].max()
        prevStructureLow = self.data["low"].iloc[-21:-1].min()

        previous_close = self.data["close"].iloc[-2]

        self.isBOS = crossover(
            self.cur_close,
            previous_close,
            prevStructureHigh,
        )

        self.isCHOCH = crossunder(
            self.cur_close,
            previous_close,
            prevStructureLow,
        )
    
    def update_indicators(self):
        self._update_trend_indicators()

        gapClose = self.data["close"].iloc[-3]

        self.bullishPowerOK = (
            self.lastBullFvg
            and (self.data["low"].iloc[-1] - self.data["high"].iloc[-3]) / gapClose * 100 >= MIN_FVG_POWER_PCT
        )

        self.bearishPowerOK = (
            self.lastBearFvg
            and (self.data["low"].iloc[-3] - self.data["high"].iloc[-1]) / gapClose * 100 >= MIN_FVG_POWER_PCT
        )

        self._calc_BOS_and_CHOCH()


    def entry_logic(self):
        if len(self.fvg_zones) == 0:
            return

        if getattr(self, "trading_paused", False):
            return

        if MAX_DRAWDOWN_ENABLED and self._is_drawdown_lockout(self._get_current_timestamp()):
            return


        if not self.check_daily_trade_limit():
            print(f"⚠️ Daily trade limit reached ({MAX_DAILY_TRADES}). No new trades today.")
            return


        allow_intracandle = ALLOW_INTRACANDLE_ENTRY
        current_high = self.data["high"].iloc[-1]
        current_low = self.data["low"].iloc[-1]
        if allow_intracandle and hasattr(self, "_intrabar_high") and hasattr(self, "_intrabar_low"):
            current_high = self._intrabar_high
            current_low = self._intrabar_low

        atr_series = get_atr(self.data, ATR_PERIOD)
        atr = atr_series.iloc[-1] if len(atr_series) > 0 else None

        for zone in self.fvg_zones[-FVG_HISTORY_NBR:]:
            if not self.pyramiding.should_allow_entry(self, zone):
                continue
            if zone["mitigated"]:
                continue

            fvg_bottom = zone["bottom"]
            fvg_top = zone["top"]

            # Full touch: current bar's high/low overlaps the FVG zone
            touchesFVG = current_high >= fvg_bottom and current_low <= fvg_top

            if (
                zone["direction"] == "bull"
                and touchesFVG
                and self.isBullishHTF
                and self.marketOK
            ):
                entry_price = self.cur_close
                if allow_intracandle:
                    entry_price = fvg_top
                stop_loss = entry_price - atr * SL_MULTIPLIER
                if USE_TRAILING:
                    trail_stop = entry_price - atr * TRAIL_OFFSET_MULT
                else:
                    trail_stop = None

                tp = entry_price + atr * TP_MULTIPLIER
                entryAtr = atr
                group_id = self._next_partial_group_id()
                self._partial_groups[group_id] = {
                    "entry_price": entry_price,
                    "entry_atr": entryAtr,
                    "side": "BUY",
                    "tp_steps_closed": 0,
                    "sl_steps_closed": 0,
                }
                any_success = False
                if SPLIT_ORDERS_ENABLED:
                    for idx in range(self._split_order_count):
                        active_order = self.Order(
                            entry_atr=entryAtr,
                            side="BUY",
                            entry_price=entry_price,
                            take_profit=tp,
                            stop_loss=stop_loss,
                            trailing_stop_loss=trail_stop,
                            order_size=EACH_TRADE_SIZE,
                            **self.api_order_kwargs(),
                        )
                        active_order.group_id = group_id
                        active_order.group_seq = idx + 1
                        active_order.entry_reference_price = entry_price
                        result = active_order.place_order()

                        success = isinstance(result, dict) and result.get("success", False)
                        if result is None:
                            # Backtest or non-API mode: treat as success
                            success = True
                        if success:
                            self.active_orders.append(active_order)
                            any_success = True
                            self.pyramiding.on_position_opened(active_order, self)
                else:
                    order_size = FIXED_LOT if USE_FIXED_LOT else self.calculate_order_size(
                        atr=atr, sl_mult=SL_MULTIPLIER
                    )
                    active_order = self.Order(
                        entry_atr=entryAtr,
                        side="BUY",
                        entry_price=entry_price,
                        take_profit=tp,
                        stop_loss=stop_loss,
                        trailing_stop_loss=trail_stop,
                        order_size=order_size,
                        **self.api_order_kwargs(),
                    )
                    active_order.group_id = group_id
                    active_order.group_seq = 1
                    active_order.entry_reference_price = entry_price
                    result = active_order.place_order()
                    success = isinstance(result, dict) and result.get("success", False)
                    if result is None:
                        # Backtest or non-API mode: treat as success
                        success = True
                    if success:
                        self.active_orders.append(active_order)
                        any_success = True
                        self.pyramiding.on_position_opened(active_order, self)

                if any_success:
                    zone["mitigated"] = True
                    self.lastPositionWasLong = True
                    self.lastPositionWasShort = False
                    self.daily_trades_count += 1
                    self.last_trade_date = str(datetime.now().date())
                    print(f"📈 LONG position opened. Daily trades: {self.daily_trades_count}/{MAX_DAILY_TRADES}")
                else:
                    self._partial_groups.pop(group_id, None)
                break



            elif (
                zone["direction"] == "bear"
                and touchesFVG
                and self.isBearishHTF
                and self.marketOK
            ):
                entry_price = self.cur_close
                if allow_intracandle:
                    entry_price = fvg_bottom
                stop_loss = entry_price + atr * SL_MULTIPLIER
                if USE_TRAILING:
                    trail_stop = entry_price + atr * TRAIL_OFFSET_MULT
                else:
                    trail_stop = None

                tp = entry_price - atr * TP_MULTIPLIER
                entryAtr = atr
                group_id = self._next_partial_group_id()
                self._partial_groups[group_id] = {
                    "entry_price": entry_price,
                    "entry_atr": entryAtr,
                    "side": "SELL",
                    "tp_steps_closed": 0,
                    "sl_steps_closed": 0,
                }
                any_success = False
                if SPLIT_ORDERS_ENABLED:
                    for idx in range(self._split_order_count):
                        active_order = self.Order(
                            entry_atr=entryAtr,
                            side="SELL",
                            entry_price=entry_price,
                            take_profit=tp,
                            trailing_stop_loss=trail_stop,
                            stop_loss=stop_loss,
                            order_size=EACH_TRADE_SIZE,
                            **self.api_order_kwargs(),
                        )
                        active_order.group_id = group_id
                        active_order.group_seq = idx + 1
                        active_order.entry_reference_price = entry_price
                        result = active_order.place_order()

                        success = isinstance(result, dict) and result.get("success", False)
                        if result is None:
                            # Backtest or non-API mode: treat as success
                            success = True
                        if success:
                            self.active_orders.append(active_order)
                            any_success = True
                            self.pyramiding.on_position_opened(active_order, self)
                else:
                    order_size = FIXED_LOT if USE_FIXED_LOT else self.calculate_order_size(
                        atr=atr, sl_mult=SL_MULTIPLIER
                    )
                    active_order = self.Order(
                        entry_atr=entryAtr,
                        side="SELL",
                        entry_price=entry_price,
                        take_profit=tp,
                        trailing_stop_loss=trail_stop,
                        stop_loss=stop_loss,
                        order_size=order_size,
                        **self.api_order_kwargs(),
                    )
                    active_order.group_id = group_id
                    active_order.group_seq = 1
                    active_order.entry_reference_price = entry_price
                    result = active_order.place_order()
                    success = isinstance(result, dict) and result.get("success", False)
                    if result is None:
                        # Backtest or non-API mode: treat as success
                        success = True
                    if success:
                        self.active_orders.append(active_order)
                        any_success = True
                        self.pyramiding.on_position_opened(active_order, self)

                if any_success:
                    zone["mitigated"] = True
                    self.lastPositionWasShort = True
                    self.lastPositionWasLong = False
                    self.daily_trades_count += 1
                    self.last_trade_date = str(datetime.now().date())
                    print(f"📉 SHORT position opened. Daily trades: {self.daily_trades_count}/{MAX_DAILY_TRADES}")
                else:
                    self._partial_groups.pop(group_id, None)
                break


    def update_stops(
        self,
        current_high: float | None = None,
        current_low: float | None = None,
        high_changed: bool = True,
        low_changed: bool = True,
    ):
        if len(self.active_orders) == 0:
            return


        if current_high is None:
            current_high = self.data["high"].iloc[-1]
        if current_low is None:
            current_low = self.data["low"].iloc[-1]

        # === UPDATE TRAILING STOPS ===
        for pos in list(self.active_orders):
            if DEBUG_STOPS:
                print(
                    f"🧪 update_stops: side={pos.side} high={current_high} low={current_low} "
                    f"entry_atr={pos.entry_atr} tsl={pos.trailing_stop_loss}"
                )
            if pos.side == "BUY":
                if not high_changed:
                    continue
                if USE_TRAILING and pos.entry_atr is not None:
                    potentialStop = current_high - pos.entry_atr * TRAIL_OFFSET_MULT
                    if pos.trailing_stop_loss is not None:
                        new_stop = max(pos.trailing_stop_loss, potentialStop)
                        if new_stop > pos.trailing_stop_loss:
                            print(f"📊 Trailing stop updated: {pos.trailing_stop_loss:.5f} → {new_stop:.5f}")
                            pos.trailing_stop_loss = new_stop
                    else:
                        pos.trailing_stop_loss = potentialStop
            elif pos.side == "SELL":
                if not low_changed:
                    continue
                if USE_TRAILING and pos.entry_atr is not None:
                    potentialStop = current_low + pos.entry_atr * TRAIL_OFFSET_MULT
                    if pos.trailing_stop_loss is not None:
                        new_stop = min(pos.trailing_stop_loss, potentialStop)
                        if new_stop < pos.trailing_stop_loss:
                            print(f"📊 Trailing stop updated: {pos.trailing_stop_loss:.5f} → {new_stop:.5f}")
                            pos.trailing_stop_loss = new_stop
                    else:
                        pos.trailing_stop_loss = potentialStop

    def _apply_pyramiding_add_on(
        self,
        current_price: float,
        current_high: float | None = None,
        current_low: float | None = None,
    ):
        if not self.active_orders:
            return
        order = self.active_orders[0]
        trigger_price = current_price
        if order.side == "BUY" and current_high is not None:
            trigger_price = max(current_price, current_high)
        elif order.side == "SELL" and current_low is not None:
            trigger_price = min(current_price, current_low)
        add_spec: AddOnSpec | None = self.pyramiding.should_add_on(self, trigger_price)
        if add_spec is None or add_spec.size is None or add_spec.size <= 0:
            return
        result = order.add_to_position(add_spec.size)
        if isinstance(result, dict) and result.get("success"):
            if add_spec.new_pyramid_count is not None:
                order.pyramid_count = add_spec.new_pyramid_count
            if add_spec.next_add_price is not None:
                order.next_add_price = add_spec.next_add_price


    def bar_iteration(self):
        with self._lock:
            self.fetch_new_data()
            if self._check_max_drawdown(self._get_current_timestamp(), float(self.cur_close)):
                self.save_data()
                return
            self.update_indicators()
            self.add_fvg_zones()
            self.entry_logic()
            self.update_stops()
            self.save_data()
