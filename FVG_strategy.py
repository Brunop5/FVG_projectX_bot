import csv
import math
import os
import sys
import threading
from abc import abstractmethod
from datetime import datetime, timedelta, time as dt_time, timezone
from pathlib import Path
from typing import Any, TypedDict

import pandas as pd

from .helping_functions.indicators import get_atr
from .helping_functions.indicators import sma
from .helping_functions.indicators import ema
from .helping_functions.indicators import crossover
from .helping_functions.indicators import crossunder
from .helping_functions.pyramiding import (
    apply_pyramiding_add_on,
    ClientAtrPyramidingPolicy,
    NoPyramidingPolicy,
    place_strategy_entry_orders,
)
from .helping_functions.partial_close import (
    PartialGroupState,
    build_partial_close_map,
    cleanup_partial_groups,
    make_partial_group_state,
    next_partial_group_id,
    try_partial_close_order,
    validate_partial_close_size,
    validate_split_config,
)
from .utils.settings import StrategyInputs, load_strategy_inputs

# Import strategyTemplate from parent workspace when running from child package root.
try:
    from strategyTemplate import Strategy, Order
except ModuleNotFoundError as exc:
    if exc.name != "strategyTemplate":
        raise
    # Allow editor/runtime execution when cwd is `FVG_projectX_bot` by adding
    # its parent (`trading_bots`) where strategyTemplate.py lives.
    parent_dir = Path(__file__).resolve().parent.parent
    parent_dir_str = str(parent_dir)
    if parent_dir_str not in sys.path:
        sys.path.insert(0, parent_dir_str)
    from strategyTemplate import Strategy, Order


class FVGZone(TypedDict):
    direction: str
    top: float
    bottom: float
    mitigated: bool


def _parse_utc_time(raw: str) -> dt_time:
    parts = raw.split(":")
    if len(parts) != 2:
        raise ValueError(f"Invalid UTC time format: {raw!r}. Expected HH:MM")
    return dt_time(int(parts[0]), int(parts[1]))


# Runtime inputs loaded from JSON (if provided), validated by pydantic.
INPUTS: StrategyInputs = load_strategy_inputs()


def _entry_cutoff_utc() -> dt_time:
    return _parse_utc_time(INPUTS.MARKET_ENTRY_CUTOFF_UTC)


def _market_close_utc() -> dt_time:
    return _parse_utc_time(INPUTS.MARKET_CLOSE_UTC)


def _market_reopen_utc() -> dt_time:
    return _parse_utc_time(INPUTS.MARKET_REOPEN_UTC)



def _apply_username_overrides(username: str | None) -> None:
    if not username:
        return
    overrides = INPUTS.USERNAME_OVERRIDES.get(username)
    if not overrides:
        return
    for key, value in overrides.items():
        if hasattr(INPUTS, key):
            setattr(INPUTS, key, value)


class FVG_Order(Order):
    # Defaults for fractional-size venues (e.g., Binance in backtest/live simulation).
    # Venue-specific subclasses can override these.
    MIN_ORDER_SIZE: float = 0.002
    ORDER_SIZE_STEP: float | None = None
    ORDER_SIZE_INTEGER_ONLY: bool = False

    entry_atr: float
    pyramid_count: int
    next_add_price: float | None
    avg_entry_price: float | None
    # Minute-bucket timestamp (epoch ms) used for TP/SL evaluation gating in live mode.
    # This prevents evaluating the same candle bucket that the order was opened in.
    opened_eval_ts_ms: int | None

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
        self.opened_eval_ts_ms = None

    def _normalize_order_size(self, value: float) -> float | int:
        if value is None or isinstance(value, bool):
            mode = "integer" if self.ORDER_SIZE_INTEGER_ONLY else "numeric"
            raise ValueError(f"order_size must be a positive {mode} value.")
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"order_size must be numeric, got {value!r}.") from None
        if not math.isfinite(numeric):
            raise ValueError(f"order_size must be finite, got {value!r}.")

        if self.ORDER_SIZE_INTEGER_ONLY:
            rounded = int(round(numeric))
            if not math.isclose(numeric, rounded, rel_tol=1e-9, abs_tol=1e-9):
                print(
                    "⚠️  order_size was not an integer; "
                    f"rounded {numeric!r} → {rounded}"
                )
            min_qty_int = max(1, int(round(float(self.MIN_ORDER_SIZE or 1.0))))
            if rounded < min_qty_int:
                print(
                    "⚠️  order_size rounded below minimum; "
                    f"clamped {rounded} → {min_qty_int}"
                )
                rounded = min_qty_int
            return rounded

        normalized = numeric
        min_qty = float(self.MIN_ORDER_SIZE or 0.0)
        if normalized < min_qty:
            print(
                "⚠️  order_size below minimum; "
                f"clamped {normalized} → {min_qty}"
            )
            normalized = min_qty

        step = self.ORDER_SIZE_STEP
        if step is not None and float(step) > 0:
            step_val = float(step)
            stepped = round(normalized / step_val) * step_val
            if not math.isclose(normalized, stepped, rel_tol=1e-9, abs_tol=1e-9):
                print(
                    "⚠️  order_size not aligned to step; "
                    f"rounded {normalized!r} → {stepped!r}"
                )
            normalized = max(min_qty, stepped)
        return float(normalized)

    def add_to_position(self, add_size: float, log=print):
        if add_size is None or add_size <= 0:
            return {"success": False, "message": "Invalid add-on size"}
        original_size = float(self.order_size or 0.0)
        add_size_norm = self._normalize_order_size(add_size)
        self.order_size = add_size_norm
        result = self.place_order()
        success = isinstance(result, dict) and result.get("success", False)
        if result is None:
            success = True
        if success:
            new_size = original_size + float(add_size_norm)
            if new_size > 0 and self.avg_entry_price is not None:
                self.avg_entry_price = (
                    (self.avg_entry_price * original_size)
                    + (self.entry_price * float(add_size_norm))
                ) / new_size
            self.order_size = self._normalize_order_size(new_size)
            log(
                f"➕ Add-on placed: size={add_size_norm} new_size={self.order_size} "
                f"side={self.side} entry={self.entry_price}"
            )
            return {"success": True, "new_size": self.order_size, "result": result}
        self.order_size = original_size
        return {"success": False, "result": result}

    def _price_checks(
        self,
        current_price: float,
        current_high: float | None,
        current_low: float | None,
    ) -> tuple[float, float]:
        if self.side == "BUY":
            stop_check = current_low if current_low is not None else current_price
            tp_check = current_high if current_high is not None else current_price
            return stop_check, tp_check
        stop_check = current_high if current_high is not None else current_price
        tp_check = current_low if current_low is not None else current_price
        return stop_check, tp_check

    def _check_stops_and_tp(
        self,
        stop_check_price: float,
        tp_check_price: float,
        current_price: float,
        log=print,
    ) -> bool:
        if self.side == "BUY":
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
            return False

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
        return False


    def check_close_conditions(self, log=print, **kwargs) -> bool:
        current_price = kwargs["current_price"]
        last_long = kwargs["last_long"]
        last_short = kwargs["last_short"]
        isBOS = kwargs["isBOS"]
        isCHOCH = kwargs["isCHOCH"]
        current_high = kwargs.get("current_high")
        current_low = kwargs.get("current_low")

        stop_check_price, tp_check_price = self._price_checks(
            current_price=current_price,
            current_high=current_high,
            current_low=current_low,
        )
        if self._check_stops_and_tp(
            stop_check_price=stop_check_price,
            tp_check_price=tp_check_price,
            current_price=current_price,
            log=log,
        ):
            return True

        # === CLOSE ON OPPOSITE BOS/CHoCH ===
        if INPUTS.HOLD_UNTIL_OPPOSITE:
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

    fvg_zones: list[FVGZone]
    
    def __init__(self):
        _apply_username_overrides(getattr(self, "username", None))
        self.isBOS = False
        self.isCHOCH = False

        self.lastBullFvg = False
        self.lastBearFvg = False
        self.lastPositionWasLong = False
        self.lastPositionWasShort = False

        self.daily_trades_count = 0
        self.last_trade_date = None
        self._lock = threading.Lock()  # Protects shared state when bar thread and price-update thread run concurrently
        self._entry_reopen_notified_date = None

        # Used by live runners: timestamp of the most recently received frequent price update candle.
        # Truncated to minute-bucket (epoch ms) inside `update_price()`.
        self._last_price_update_ts_ms: int | None = None

        self.fvg_zones = []
        if not hasattr(self, "pyramiding") or self.pyramiding is None:
            if INPUTS.ALLOW_PYRAMIDING:
                self.pyramiding = ClientAtrPyramidingPolicy(
                    atr_step=INPUTS.PYR_ATR_STEP,
                    add_on_size=INPUTS.PYR_ADD_ON_SIZE,
                    max_adds=INPUTS.PYR_MAX_ADDS,
                )
            else:
                self.pyramiding = NoPyramidingPolicy()

        self._partial_groups: dict[int, PartialGroupState] = {}
        self._partial_group_counter = 0
        self._split_order_count = validate_split_config(self, INPUTS)
        self.require_intrabar_entry = False
        self._intrabar_mode = False

        self._partial_tp_close_count = (
            validate_partial_close_size(
                self,
                INPUTS,
                INPUTS.PARTIAL_TP_CLOSE_SIZE,
                "PARTIAL_TP_CLOSE_SIZE",
            )
            if INPUTS.ENABLE_PARTIAL_TP
            else 0
        )
        self._partial_sl_close_count = (
            validate_partial_close_size(
                self,
                INPUTS,
                INPUTS.PARTIAL_SL_CLOSE_SIZE,
                "PARTIAL_SL_CLOSE_SIZE",
            )
            if INPUTS.ENABLE_PARTIAL_SL
            else 0
        )

        self.peak_unrealized_pnl = INPUTS.STARTING_PNL
        if not hasattr(self, "peak_total_pnl"):
            try:
                base_realized = float(getattr(self, "daily_realized_pnl", 0.0) or 0.0)
            except (TypeError, ValueError):
                base_realized = 0.0
            base_realized += INPUTS.STARTING_PNL
            self.peak_total_pnl = max(INPUTS.STARTING_PNL, base_realized)
        if not hasattr(self, "lockout_start_pnl"):
            self.lockout_start_pnl = None
        self.max_dd_triggered_until = None

        # Session time control state (per day, in UTC)
        self._entry_cutoff_notified_date = None
        self._session_close_executed_date = None

        if not hasattr(self, "trade_log_filename"):
            base = os.path.splitext(self.csv_filename)[0] if getattr(self, "csv_filename", None) else "trade_log"
            self.trade_log_filename = f"{base}_trades.csv"

        super().__init__()
        if not hasattr(self, "daily_realized_pnl"):
            self.daily_realized_pnl = 0.0
        if not hasattr(self, "last_pnl_date"):
            self.last_pnl_date = None

        if isinstance(self.data, pd.DataFrame) and not self.data.empty:
            self.cur_close = self.data["close"].iloc[-1]
            self.cur_volume = self.data["volume"].iloc[-1]
            self.update_indicators()
            self.add_fvg_zones()
        else:
            self.cur_close = 0.0
            self.cur_volume = 0.0


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

    

    @staticmethod
    def _to_utc_datetime(value: Any) -> datetime | None:
        if value is None or pd.isna(value):
            return None
        try:
            raw = value.item() if hasattr(value, "item") else value
        except Exception:
            raw = value
        try:
            if isinstance(raw, (int, float)) and not isinstance(raw, bool):
                numeric = float(raw)
                if numeric > 1e17:
                    return pd.to_datetime(numeric, unit="ns", utc=True).to_pydatetime(warn=False)
                if numeric > 1e14:
                    return pd.to_datetime(numeric, unit="us", utc=True).to_pydatetime(warn=False)
                if numeric > 1e11:
                    return pd.to_datetime(numeric, unit="ms", utc=True).to_pydatetime(warn=False)
                if numeric > 1e8:
                    return pd.to_datetime(numeric, unit="s", utc=True).to_pydatetime(warn=False)
            return pd.to_datetime(raw, utc=True).to_pydatetime(warn=False)
        except Exception:
            return None

    @staticmethod
    def _to_minute_bucket_ts_ms(dt_value: datetime | None) -> int | None:
        if dt_value is None:
            return None
        epoch_ms = int(dt_value.timestamp() * 1000)
        return (epoch_ms // 60_000) * 60_000

    @staticmethod
    def _set_order_exit_context(order: Order, price: float, timestamp: datetime, reason: str | None = None) -> None:
        if hasattr(order, "_last_price"):
            order._last_price = float(price)
        if hasattr(order, "_last_timestamp"):
            order._last_timestamp = timestamp
        if reason is not None and hasattr(order, "exit_reason"):
            order.exit_reason = reason

    def _resolve_opened_eval_bucket_ts_ms(self) -> int | None:
        if self._last_price_update_ts_ms is not None:
            return self._last_price_update_ts_ms
        return self._to_minute_bucket_ts_ms(self._get_current_timestamp())

    def _record_trade_safe(self, order: Order, exit_price: float, exit_timestamp: datetime) -> bool:
        try:
            self._record_trade(order, exit_price, exit_timestamp)
            return True
        except Exception:
            return False

    def _get_current_timestamp(self) -> datetime:
        if getattr(self, "_current_dt", None) is not None:
            return self._current_dt
        data_obj = getattr(self, "data", None)
        if isinstance(data_obj, pd.DataFrame) and "timestamp" in data_obj.columns:
            parsed = self._to_utc_datetime(data_obj["timestamp"].iloc[-1])
            if parsed is not None:
                return parsed
        return datetime.now(timezone.utc)

    def _reset_daily_pnl_if_needed(self, current_timestamp: datetime) -> None:
        today = str(current_timestamp.date())
        if self.last_pnl_date != today:
            self.daily_realized_pnl = 0.0
            self.last_pnl_date = today

    def calculate_trade_pnl(
        self,
        order: FVG_Order,
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
        order: FVG_Order,
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
        self._append_trade_log(order, exit_price, pnl, exit_timestamp)
        return pnl

    def _append_trade_log(
        self,
        order: FVG_Order,
        exit_price: float | None,
        pnl: float,
        exit_timestamp: datetime | None,
    ) -> None:
        try:
            entry_price = getattr(order, "avg_entry_price", None) or order.entry_price
            if entry_price is None or exit_price is None:
                return
            entry_price = float(entry_price)
            exit_price = float(exit_price)
            size = float(order.order_size or 0.0)
        except (TypeError, ValueError):
            return
        if size <= 0:
            return
        reason = getattr(order, "exit_reason", None) or ""
        side = getattr(order, "side", None) or ""
        asset = getattr(order, "asset_id", None) or getattr(self, "asset", "")
        timestamp = exit_timestamp or self._get_current_timestamp()
        if isinstance(timestamp, datetime):
            timestamp = timestamp.isoformat()

        log_path = getattr(self, "trade_log_filename", None)
        if not log_path:
            return
        write_header = not os.path.exists(log_path)
        try:
            with open(log_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow(
                        [
                            "timestamp",
                            "asset",
                            "side",
                            "size",
                            "entry_price",
                            "exit_price",
                            "reason",
                            "pnl",
                        ]
                    )
                writer.writerow(
                    [
                        timestamp,
                        asset,
                        side,
                        size,
                        entry_price,
                        exit_price,
                        reason,
                        pnl,
                    ]
                )
        except Exception:
            return

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
        lockout_end_dt = datetime.combine(lockout_date, _market_reopen_utc(), tzinfo=timezone.utc)
        if current_timestamp >= lockout_end_dt:
            self.max_dd_triggered_until = None
            self.peak_unrealized_pnl = 0.0
            lockout_pnl = getattr(self, "lockout_start_pnl", None)
            if lockout_pnl is None:
                lockout_pnl = INPUTS.STARTING_PNL
            try:
                lockout_pnl = float(lockout_pnl)
            except (TypeError, ValueError):
                lockout_pnl = INPUTS.STARTING_PNL
            self.peak_total_pnl = max(INPUTS.STARTING_PNL, lockout_pnl)
            self.lockout_start_pnl = None
            return False
        return True

    def _close_all_positions(self, current_price: float, current_timestamp: datetime, reason: str):
        for order in list(self.active_orders):
            self._set_order_exit_context(order, current_price, current_timestamp, reason=reason)
            order.close_order()
            self._record_trade_safe(order, current_price, current_timestamp)
        self.active_orders = []
        self.lastPositionWasLong = False
        self.lastPositionWasShort = False

    def _check_max_drawdown(self, current_timestamp: datetime, current_price: float) -> bool:
        if not INPUTS.MAX_DRAWDOWN_ENABLED:
            return False
        if self._is_drawdown_lockout(current_timestamp):
            return True

        self._reset_daily_pnl_if_needed(current_timestamp)
        unrealized_pnl = self._get_unrealized_pnl(current_price)
        try:
            realized_pnl = float(getattr(self, "daily_realized_pnl", 0.0) or 0.0)
        except (TypeError, ValueError):
            realized_pnl = 0.0
        total_pnl = INPUTS.STARTING_PNL + realized_pnl + unrealized_pnl

        peak_total = getattr(self, "peak_total_pnl", None)
        if peak_total is None:
            peak_total = max(INPUTS.STARTING_PNL, total_pnl)
        if total_pnl > peak_total:
            self.peak_total_pnl = total_pnl
            return False
        if peak_total <= 0:
            return False

        drawdown_pct = ((peak_total - total_pnl) / peak_total) * 100.0
        if drawdown_pct >= INPUTS.MAX_DRAWDOWN_PCT:
            self._close_all_positions(current_price, current_timestamp, "max_drawdown")
            self.max_dd_triggered_until = current_timestamp.date() + timedelta(days=1)
            try:
                realized_after_close = float(getattr(self, "daily_realized_pnl", 0.0) or 0.0)
            except (TypeError, ValueError):
                realized_after_close = 0.0
            lockout_pnl = INPUTS.STARTING_PNL + realized_after_close
            if lockout_pnl < INPUTS.STARTING_PNL:
                lockout_pnl = INPUTS.STARTING_PNL
            self.lockout_start_pnl = lockout_pnl
            print(
                "🛑 Max drawdown hit: "
                f"peak_total={peak_total:.2f} total={total_pnl:.2f} "
                f"realized={realized_pnl:.2f} unrealized={unrealized_pnl:.2f} "
                f"dd={drawdown_pct:.2f}% lockout_until={self.max_dd_triggered_until}"
            )
            return True
        return False

    def _apply_session_time_guards(self) -> None:
        """
        Enforce session time rules in UTC:
        - Block new entries after MARKET_ENTRY_CUTOFF_UTC
        - Close all open positions at MARKET_CLOSE_UTC
        """
        if not INPUTS.ENABLE_SESSION_TIME_GUARDS:
            return
        current_timestamp = self._get_current_timestamp()
        current_time = current_timestamp.time()

        # Force-close all positions at/after MARKET_CLOSE_UTC, once per day
        if current_time >= _market_close_utc():
            current_date = current_timestamp.date()
            if self._session_close_executed_date != current_date:
                try:
                    current_price = float(getattr(self, "cur_close", None) or 0.0)
                except (TypeError, ValueError):
                    current_price = 0.0
                if self.active_orders:
                    print(
                        f"🕒 Session close reached at {current_timestamp.isoformat()} (UTC). "
                        f"Closing all open positions."
                    )
                    self._close_all_positions(current_price, current_timestamp, "session_close")
                else:
                    print(
                        f"🕒 Session close reached at {current_timestamp.isoformat()} (UTC). "
                        "No open positions to close."
                    )
                self._session_close_executed_date = current_date

    
    def _validate_new_row(self, new_row):
        if new_row is None or len(new_row) == 0:
            return False 
        # Skip this iteration if no data, but keep running

        
        # Check if 'close' column exists
        if 'close' not in new_row.columns:
            print(f"⚠️  Warning: 'close' column not found in data. Available columns: {new_row.columns.tolist()}")
            return False

        return True

    def _update_ohlc(self, new_row):
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
        self._current_dt = self._to_utc_datetime(ts)
        self._last_price_update_ts_ms = self._to_minute_bucket_ts_ms(self._current_dt)
        current_timestamp = self._get_current_timestamp()

        return current_timestamp, current_high, current_low

    def _run_entry_logic_intrabar_mode(self):
        self._intrabar_mode = True
        try:
            self.entry_logic()
        finally:
            self._intrabar_mode = False


    def _run_intrabar_entry_logic(
        self,
        *,
        tick_price: float,
        tick_high: float,
        tick_low: float,
    ) -> None:
        # Use intrabar values for live checks without persisting them to history.
        if self.data is not None and len(self.data) > 0:
            last_index = self.data.index[-1]
            orig_high = self.data.at[last_index, "high"] if "high" in self.data.columns else None
            orig_low = self.data.at[last_index, "low"] if "low" in self.data.columns else None
            orig_close = self.data.at[last_index, "close"] if "close" in self.data.columns else None
            try:
                if "high" in self.data.columns:
                    self.data.at[last_index, "high"] = tick_high
                if "low" in self.data.columns:
                    self.data.at[last_index, "low"] = tick_low
                if "close" in self.data.columns:
                    self.data.at[last_index, "close"] = tick_price
                self._run_entry_logic_intrabar_mode()
            finally:
                if orig_high is not None:
                    self.data.at[last_index, "high"] = orig_high
                if orig_low is not None:
                    self.data.at[last_index, "low"] = orig_low
                if orig_close is not None:
                    self.data.at[last_index, "close"] = orig_close
            return
        self._run_entry_logic_intrabar_mode()

    def update_price(self, new_row: pd.DataFrame):
        """
        A price update as often as possible for the FVG strategy.
        Here it should check stops (and close order if it doesnt set them in order sending)
        and Move trailing stops

        This is just one iteration after data is already fetched.
        Uses _lock so bar-iteration and price-update threads do not race on shared state.
        """
        if not self._validate_new_row(new_row):
            return

        with self._lock:
            current_timestamp, current_high, current_low = self._update_ohlc(new_row)
            self._reset_daily_pnl_if_needed(current_timestamp)
            if self._check_max_drawdown(current_timestamp, float(self.cur_close)):
                return

            if len(self.active_orders) > 0:
                eval_bucket_ts_ms = self._last_price_update_ts_ms
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
                if not high_changed and not low_changed:
                    return
                self.update_stops(
                    current_high=current_high,
                    current_low=current_low,
                    high_changed=high_changed,
                    low_changed=low_changed,
                    eval_bucket_ts_ms=eval_bucket_ts_ms,
                )
                partial_close_map = build_partial_close_map(self, INPUTS)

                remaining = []
                closed_any = False
                for order in list(self.active_orders):
                    recorded_trade = False

                    # Skip TP/SL evaluation for the minute-bucket the order was opened in.
                    opened_eval_ts_ms = getattr(order, "opened_eval_ts_ms", None)
                    if (
                        eval_bucket_ts_ms is not None
                        and opened_eval_ts_ms is not None
                        and opened_eval_ts_ms == eval_bucket_ts_ms
                    ):
                        remaining.append(order)
                        continue

                    closed = order.check_close_conditions(
                        current_price=self.cur_close,
                        current_high=current_high,
                        current_low=current_low,
                        last_long=order.side == "BUY",
                        last_short=order.side == "SELL",
                        isBOS=self.isBOS,
                        isCHOCH=self.isCHOCH,
                    )
                    if not closed:
                        closed, recorded_trade = try_partial_close_order(
                            self,
                            INPUTS,
                            order,
                            partial_close_map,
                            current_timestamp,
                        )
                    if closed:
                        self._set_order_exit_context(order, float(self.cur_close), current_timestamp)
                        if not recorded_trade:
                            self._record_trade_safe(order, self.cur_close, current_timestamp)
                        closed_any = True
                        self.pyramiding.on_position_closed(order, self)

                    else:
                        remaining.append(order)
                self.active_orders = remaining
                if self.active_orders:
                    apply_pyramiding_add_on(
                        self,
                        INPUTS,
                        self.cur_close,
                        current_high,
                        current_low,
                    )
                if closed_any:
                    self.lastPositionWasLong = any(o.side == "BUY" for o in self.active_orders)
                    self.lastPositionWasShort = any(o.side == "SELL" for o in self.active_orders)
                    cleanup_partial_groups(self.active_orders, self._partial_groups)
                    self.save_data()


            elif INPUTS.ALLOW_INTRACANDLE_ENTRY:
                is_final = False
                if "is_final" in new_row.columns:
                    try:
                        is_final = bool(new_row["is_final"].iloc[-1])
                    except Exception:
                        is_final = False
                if is_final:
                    return
                tick_price = float(self.cur_close)
                tick_high = (float(new_row["high"].iloc[-1]) if "high" in new_row.columns else tick_price)
                tick_low = (float(new_row["low"].iloc[-1]) if "low" in new_row.columns else tick_price)

                self._intrabar_price = tick_price
                self._intrabar_high = tick_high
                self._intrabar_low = tick_low
                self._run_intrabar_entry_logic(
                    tick_price=tick_price,
                    tick_high=tick_high,
                    tick_low=tick_low,
                )


    def _append_fvg_zone(self, direction: str, top: float, bottom: float) -> None:
        self.fvg_zones.append(
            {
                "direction": direction,
                "top": float(top),
                "bottom": float(bottom),
                "mitigated": False,
            }
        )
        if len(self.fvg_zones) > INPUTS.FVG_HISTORY_NBR:
            self.fvg_zones = self.fvg_zones[-INPUTS.FVG_HISTORY_NBR:]

    def add_fvg_zones(self):
        # === FVG ZONE CREATION (equivalent to the box.new blocks) ===
        gap_close = self.data["close"].iloc[-3]
        bull_power_pct = (
            (self.data["low"].iloc[-1] - self.data["high"].iloc[-3])
            / gap_close
            * 100
        )
        bear_power_pct = (
            (self.data["low"].iloc[-3] - self.data["high"].iloc[-1])
            / gap_close
            * 100
        )

        if self.lastBullFvg and not (
            self.bullishPowerOK and self.isBullishHTF and self.marketOK
        ):
            reasons = []
            if not self.bullishPowerOK:
                reasons.append("power<min")
            if not self.isBullishHTF:
                reasons.append("HTF not bullish")
            if not self.marketOK:
                reasons.append("marketOK false")
            reason_text = ", ".join(reasons) if reasons else "unknown"
            if INPUTS.DEBUG_FVG:
                print("🚨 FVG FOUND BUT ZONE NOT CREATED (BULL)")
                print(f"Reason(s): {reason_text}")
                print(
                    "Criteria: "
                    f"lastBullFvg={self.lastBullFvg} "
                    f"powerPct={bull_power_pct:.5f} "
                    f"minPowerPct={INPUTS.MIN_FVG_POWER_PCT:.5f} "
                    f"bullishPowerOK={self.bullishPowerOK} "
                    f"isBullishHTF={self.isBullishHTF} "
                    f"marketOK={self.marketOK}"
                )

        if self.lastBearFvg and not (
            self.bearishPowerOK and self.isBearishHTF and self.marketOK
        ):
            reasons = []
            if not self.bearishPowerOK:
                reasons.append("power<min")
            if not self.isBearishHTF:
                reasons.append("HTF not bearish")
            if not self.marketOK:
                reasons.append("marketOK false")
            reason_text = ", ".join(reasons) if reasons else "unknown"
            if INPUTS.DEBUG_FVG:
                print("🚨 FVG FOUND BUT ZONE NOT CREATED (BEAR)")
                print(f"Reason(s): {reason_text}")
                print(
                    "Criteria: "
                    f"lastBearFvg={self.lastBearFvg} "
                    f"powerPct={bear_power_pct:.5f} "
                    f"minPowerPct={INPUTS.MIN_FVG_POWER_PCT:.5f} "
                    f"bearishPowerOK={self.bearishPowerOK} "
                    f"isBearishHTF={self.isBearishHTF} "
                    f"marketOK={self.marketOK}"
                )

        if self.bullishPowerOK and self.isBullishHTF and self.marketOK:
            # Bullish FVG uses low[1] as top and high[3] as bottom in Pine.
            self._append_fvg_zone(
                direction="bull",
                top=self.data["low"].iloc[-1],
                bottom=self.data["high"].iloc[-3],
            )
            print(f"🟢 Bullish FVG detected: {self.data['high'].iloc[-3]:.5f} - {self.data['low'].iloc[-1]:.5f}")


        if self.bearishPowerOK and self.isBearishHTF and self.marketOK:
            # Bearish FVG uses low[3] as top and high[1] as bottom in Pine.
            self._append_fvg_zone(
                direction="bear",
                top=self.data["low"].iloc[-3],
                bottom=self.data["high"].iloc[-1],
            )
            print(f"🔴 Bearish FVG detected: {self.data['high'].iloc[-1]:.5f} - {self.data['low'].iloc[-3]:.5f}")


    def _update_trend_indicators(self):
        bars = self.fetch_htf_data()
        htfEMA = ema(bars, INPUTS.EMA_PERIOD)

        if htfEMA is None:
            self.isBullishHTF = False
            self.isBearishHTF = False
        else:
            self.isBullishHTF = self.cur_close > htfEMA
            self.isBearishHTF = self.cur_close < htfEMA

        atrVal = get_atr(self.data, INPUTS.ATR_PERIOD)
        atr_sma = sma(atrVal, min(20, len(atrVal))) if not atrVal.empty else None
        atrOK = atrVal.iloc[-1] > atr_sma if (not atrVal.empty and atr_sma is not None) else False
        atr_last = atrVal.iloc[-1] if not atrVal.empty else None
        vol_sma = None
        volOK = None

        if INPUTS.USE_VOLUME_CHECK:
            vol_sma = sma(self.data["volume"], 20)
            volOK = self.cur_volume > vol_sma * INPUTS.VOLUME_MULTIPLIER if vol_sma is not None else False
            self.marketOK = volOK and atrOK
        else:
            # Skip volume check, only use ATR
            self.marketOK = atrOK

        self._marketok_debug = {
            "use_volume_check": INPUTS.USE_VOLUME_CHECK,
            "atr": atr_last,
            "atr_sma": atr_sma,
            "atr_ok": atrOK,
            "cur_volume": self.cur_volume,
            "vol_sma": vol_sma,
            "vol_mult": INPUTS.VOLUME_MULTIPLIER,
            "vol_ok": volOK,
            "market_ok": self.marketOK,
        }


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
            and (self.data["low"].iloc[-1] - self.data["high"].iloc[-3]) / gapClose * 100 >= INPUTS.MIN_FVG_POWER_PCT
        )

        self.bearishPowerOK = (
            self.lastBearFvg
            and (self.data["low"].iloc[-3] - self.data["high"].iloc[-1]) / gapClose * 100 >= INPUTS.MIN_FVG_POWER_PCT
        )

        self._calc_BOS_and_CHOCH()

    def entry_logic(self):
        if not self.fvg_zones:
            return

        if getattr(self, "trading_paused", False):
            return

        if INPUTS.MAX_DRAWDOWN_ENABLED and self._is_drawdown_lockout(self._get_current_timestamp()):
            return

        # Block new entries after the configured UTC cutoff time
        if INPUTS.ENABLE_SESSION_TIME_GUARDS:
            current_timestamp = self._get_current_timestamp()
            current_time = current_timestamp.time()
            if _entry_cutoff_utc() <= current_time < _market_close_utc():
                current_date = current_timestamp.date()
                if self._entry_cutoff_notified_date != current_date:
                    print(
                        f"🕒 Entry cutoff reached at {current_timestamp.isoformat()} (UTC). "
                        f"No new trades after {_entry_cutoff_utc().strftime('%H:%M')} UTC "
                        f"until {_market_reopen_utc().strftime('%H:%M')} UTC."
                    )
                    self._entry_cutoff_notified_date = current_date
                return
            if _market_close_utc() <= current_time < _market_reopen_utc():
                current_date = current_timestamp.date()
                if self._entry_reopen_notified_date != current_date:
                    print(
                        f"🕒 Market closed at {_market_close_utc().strftime('%H:%M')} UTC. "
                        f"Entries resume at {_market_reopen_utc().strftime('%H:%M')} UTC."
                    )
                    self._entry_reopen_notified_date = current_date
                return


        if not self.check_daily_trade_limit():
            print(f"⚠️ Daily trade limit reached ({INPUTS.MAX_DAILY_TRADES}). No new trades today.")
            return


        allow_intracandle = INPUTS.ALLOW_INTRACANDLE_ENTRY
        if allow_intracandle and getattr(self, "require_intrabar_entry", False):
            if not getattr(self, "_intrabar_mode", False):
                return
        current_high = self.data["high"].iloc[-1]
        current_low = self.data["low"].iloc[-1]
        if allow_intracandle and hasattr(self, "_intrabar_high") and hasattr(self, "_intrabar_low"):
            current_high = self._intrabar_high
            current_low = self._intrabar_low

        atr_series = get_atr(self.data, INPUTS.ATR_PERIOD)
        atr = atr_series.iloc[-1] if not atr_series.empty else None
        if atr is None:
            return

        for zone in self.fvg_zones[-INPUTS.FVG_HISTORY_NBR:]:
            if not self.pyramiding.should_allow_entry(self, zone):
                continue
            if zone["mitigated"]:
                continue

            fvg_bottom = zone["bottom"]
            fvg_top = zone["top"]

            # Full touch: current bar's high/low overlaps the FVG zone
            touchesFVG = current_high > fvg_bottom and current_low < fvg_top

            if (
                zone["direction"] == "bull"
                and touchesFVG
                and self.isBullishHTF
                and self.marketOK
            ):
                entry_price = self.cur_close
                if allow_intracandle:
                    entry_price = fvg_top
                stop_loss = entry_price - atr * INPUTS.SL_MULTIPLIER
                if INPUTS.USE_TRAILING:
                    trail_stop = entry_price - atr * INPUTS.TRAIL_OFFSET_MULT
                else:
                    trail_stop = None

                tp = entry_price + atr * INPUTS.TP_MULTIPLIER
                entryAtr = atr
                group_id = next_partial_group_id(self, INPUTS)
                self._partial_groups[group_id] = make_partial_group_state(
                    entry_price=entry_price,
                    entry_atr=entryAtr,
                    side="BUY",
                )
                any_success = place_strategy_entry_orders(
                    self,
                    INPUTS,
                    side="BUY",
                    entry_atr=entryAtr,
                    entry_price=entry_price,
                    tp=tp,
                    stop_loss=stop_loss,
                    trail_stop=trail_stop,
                    group_id=group_id,
                    split_order_count=self._split_order_count,
                )

                if any_success:
                    zone["mitigated"] = True
                    self.lastPositionWasLong = True
                    self.lastPositionWasShort = False
                    self.daily_trades_count += 1
                    self.last_trade_date = str(datetime.now().date())
                    print(f"📈 LONG position opened. Daily trades: {self.daily_trades_count}/{INPUTS.MAX_DAILY_TRADES}")
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
                stop_loss = entry_price + atr * INPUTS.SL_MULTIPLIER
                if INPUTS.USE_TRAILING:
                    trail_stop = entry_price + atr * INPUTS.TRAIL_OFFSET_MULT
                else:
                    trail_stop = None

                tp = entry_price - atr * INPUTS.TP_MULTIPLIER
                entryAtr = atr
                group_id = next_partial_group_id(self, INPUTS)
                self._partial_groups[group_id] = make_partial_group_state(
                    entry_price=entry_price,
                    entry_atr=entryAtr,
                    side="SELL",
                )
                any_success = place_strategy_entry_orders(
                    self,
                    INPUTS,
                    side="SELL",
                    entry_atr=entryAtr,
                    entry_price=entry_price,
                    tp=tp,
                    stop_loss=stop_loss,
                    trail_stop=trail_stop,
                    group_id=group_id,
                    split_order_count=self._split_order_count,
                )

                if any_success:
                    zone["mitigated"] = True
                    self.lastPositionWasShort = True
                    self.lastPositionWasLong = False
                    self.daily_trades_count += 1
                    self.last_trade_date = str(datetime.now().date())
                    print(f"📉 SHORT position opened. Daily trades: {self.daily_trades_count}/{INPUTS.MAX_DAILY_TRADES}")
                else:
                    self._partial_groups.pop(group_id, None)
                break

    def _update_order_trailing_stop(
        self,
        order: FVG_Order,
        current_high: float,
        current_low: float,
        high_changed: bool,
        low_changed: bool,
    ) -> None:
        if not INPUTS.USE_TRAILING or order.entry_atr is None:
            return
        if order.side == "BUY":
            if not high_changed:
                return
            potential_stop = current_high - order.entry_atr * INPUTS.TRAIL_OFFSET_MULT
            if order.trailing_stop_loss is None:
                order.trailing_stop_loss = potential_stop
                return
            new_stop = max(order.trailing_stop_loss, potential_stop)
            if new_stop > order.trailing_stop_loss:
                print(f"📊 Trailing stop updated: {order.trailing_stop_loss:.5f} → {new_stop:.5f}")
                order.trailing_stop_loss = new_stop
            return

        if not low_changed:
            return
        potential_stop = current_low + order.entry_atr * INPUTS.TRAIL_OFFSET_MULT
        if order.trailing_stop_loss is None:
            order.trailing_stop_loss = potential_stop
            return
        new_stop = min(order.trailing_stop_loss, potential_stop)
        if new_stop < order.trailing_stop_loss:
            print(f"📊 Trailing stop updated: {order.trailing_stop_loss:.5f} → {new_stop:.5f}")
            order.trailing_stop_loss = new_stop


    def update_stops(
        self,
        current_high: float | None = None,
        current_low: float | None = None,
        high_changed: bool = True,
        low_changed: bool = True,
        eval_bucket_ts_ms: int | None = None,
    ):
        if not self.active_orders:
            return


        if current_high is None:
            current_high = self.data["high"].iloc[-1]
        if current_low is None:
            current_low = self.data["low"].iloc[-1]

        # === UPDATE TRAILING STOPS ===
        for pos in list(self.active_orders):
            # Skip trailing updates for the minute-bucket the order was opened in.
            opened_eval_ts_ms = getattr(pos, "opened_eval_ts_ms", None)
            if (
                eval_bucket_ts_ms is not None
                and opened_eval_ts_ms is not None
                and opened_eval_ts_ms == eval_bucket_ts_ms
            ):
                continue


            self._update_order_trailing_stop(
                order=pos,
                current_high=current_high,
                current_low=current_low,
                high_changed=high_changed,
                low_changed=low_changed,
            )

    def bar_iteration(self):
        with self._lock:
            self.fetch_new_data()
            if self._check_max_drawdown(self._get_current_timestamp(), float(self.cur_close)):
                self.save_data()
                return
            # Enforce session-based time rules (entry cutoff and forced close)
            self._apply_session_time_guards()
            self.update_indicators()
            self.add_fvg_zones()
            self.entry_logic()
            self.update_stops(eval_bucket_ts_ms=self._last_price_update_ts_ms)
            self.save_data()
