"""
Backtesting module for the FVG strategy.
NOT DONE
Standalone implementation with all configuration in this file.
"""

from __future__ import annotations

import os
import sys
import random
import itertools
import multiprocessing
from datetime import datetime, timedelta
from io import StringIO
from threading import Lock, Thread
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from .. import FVG_strategy as fvg_module
from ..FVG_strategy import *
from ..helping_functions.indicators import ema, sma, get_atr
from ..FVG_strategy import FVG_Order


# ==================== USER INPUTS / CONSTANTS ====================
BACKTEST_ASSET_NAME = "BTC"
BACKTEST_INITIAL_BALANCE = 1000.0
BACKTEST_MAX_LOSS = 1000
BACKTEST_SELECTED_TIMEFRAME = "15min"
BACKTEST_TIMEFRAMES_TO_TEST = ["5min", "15min", "30min", "1h"]
BACKTEST_START_DATE = None
BACKTEST_END_DATE = None

USE_MARGIN_PER_TRADE = True
MARGIN_PER_TRADE = 100
MIN_MARGIN_LOT_SIZE = 0.002
MAX_MARGIN_LOT_SIZE = 100.0

USE_MULTITHREADING = False
USE_MULTIPROCESSING = True

USE_DIRECT_DATA_FILE = True
DIRECT_DATA_FILE_PATH = os.path.join(CURRENT_DIR, "data", "BTCUSDT_PERP_15m.csv")

USE_CFD_PRICING = True
DEFAULT_CFD_SETTINGS = {"leverage": 20, "fee_pct": 0.001, "spread": 0.1}
CFD_SETTINGS_BY_ASSET = {}

OPTIMIZATION_CONFIG = {
    "FVG_HISTORY_NBR": {"range": list(range(1, 16)), "current": 10},
    "MIN_FVG_POWER_PCT": {"range": [round(0.01 + i * 0.01, 2) for i in range(20)], "current": 0.1},
    "HTF_TF": {"range": ["30", "60", "90", "120", "240", "1440"], "current": "240"},
    "EMA_PERIOD": {"range": [15, 25, 50, 100, 200], "current": 50},
    "VOLUME_MULTIPLIER": {"range": [round(1.0 + i * 0.05, 2) for i in range(11)], "current": 1.2},
    "ATR_PERIOD": {"range": list(range(5, 26)), "current": 14},
    "SL_MULTIPLIER": {"range": [round(1.0 + i * 0.5, 1) for i in range(19)], "current": 4.0},
    "TP_MULTIPLIER": {"range": list(range(1, 21)) + [2000000], "current": 2000000.0},
    "USE_TRAILING": {"range": [True, False], "current": True},
    "TRAIL_OFFSET_MULT": {"range": list(range(1, 21)), "current": 6.0},
    "HOLD_UNTIL_OPPOSITE": {"range": [True, False], "current": True},
}

OUT_PATH = os.path.join(CURRENT_DIR, "btc_results")
FIXED_PARAMS = {
    "USE_VOLUME_CHECK": True,
    "VOLUME_DATA_START_TIMESTAMP": 1755464400000,
    "START_FROM_VOLUME_TIMESTAMP": False,
}

RUN_OPTIMIZATION = False
USE_EXHAUSTIVE_SEARCH = False
RANDOM_SEARCH_SAMPLES = 10000
PROGRESS_STEP_PCT = 5
USE_AUTO_WORKERS = True
MAX_WORKERS = 4
USE_FIRST_TENTH_ONLY = False

USE_CSV_INPUT = False
CSV_INPUT_FILE = "filtered_backtest_results.csv"

SAVE_INDIVIDUAL_RESULTS_IN_OPTIMIZATION = True
ALLOW_INTRACANDLE_CHECKS = True

TIMEFRAME_FILE_MAP = {
    "5min": "GOLD.m_M5.csv",
    "30min": "GOLD.m_M30.csv",
    "1h": "GOLD.m_H1.csv",
    "15min": "1mdata_gold_15min.csv",
}


# Asset list for round-turn fee lookup (4th element is fee per contract)
ASSETS = []

CONTRACTS_DATA = {}
CONTRACTS_BY_NAME = {}
ROUND_TURN_FEES = {}

RESULTS_CSV_LOCK = Lock()
SUMMARY_CSV_LOCK = Lock()
FINAL_RESULT_LOCK = Lock()


# ==================== HELPERS ====================
def _parse_datetime_input(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return pd.to_datetime(value, unit="ms", utc=True, errors="coerce")
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit():
            return pd.to_datetime(int(stripped), unit="ms", utc=True, errors="coerce")
        return pd.to_datetime(stripped, utc=True, errors="coerce")
    return pd.to_datetime(value, utc=True, errors="coerce")


def _safe_asset_name(asset_name: str) -> str:
    safe = "".join(c for c in asset_name if c.isalnum() or c in (" ", "-", "_"))
    return safe.strip().replace(" ", "_")


def _load_contracts_data():
    global CONTRACTS_DATA, CONTRACTS_BY_NAME
    candidates = [
        os.path.join(CURRENT_DIR, "contracts.csv"),
        os.path.join(PROJECT_ROOT, "contracts.csv"),
    ]
    path = next((p for p in candidates if os.path.exists(p)), None)
    if not path:
        return
    df = pd.read_csv(path)
    for _, row in df.iterrows():
        CONTRACTS_DATA[row["id"]] = {
            "tick_size": float(row.get("tickSize", 0.0)),
            "tick_value": float(row.get("tickValue", 0.0)),
        }
        CONTRACTS_BY_NAME[str(row.get("name"))] = row["id"]


def _load_round_turn_fees():
    for asset_tuple in ASSETS:
        if len(asset_tuple) >= 4:
            ROUND_TURN_FEES[asset_tuple[0]] = float(asset_tuple[3])


def initialize_backtest_data():
    _load_contracts_data()
    _load_round_turn_fees()


def get_contract_id_by_name(asset_name: str):
    return CONTRACTS_BY_NAME.get(asset_name)


def get_contract_info(asset_id: str):
    return CONTRACTS_DATA.get(asset_id, {"tick_size": None, "tick_value": None})


def get_round_turn_fee(asset_id: str) -> float:
    return ROUND_TURN_FEES.get(asset_id, 0.0)


def _read_csv_data(data_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(data_path, sep="\t")
    except Exception:
        df = pd.read_csv(data_path)
    else:
        # If tab read produced a single merged column, retry as comma CSV.
        if len(df.columns) == 1 and "," in df.columns[0]:
            df = pd.read_csv(data_path)

    if "<DATE>" in df.columns and "<TIME>" in df.columns:
        ts = df["<DATE>"].astype(str) + " " + df["<TIME>"].astype(str)
        df["timestamp"] = pd.to_datetime(ts, format="%Y.%m.%d %H:%M:%S", utc=True, errors="coerce")
        df = df.rename(columns={
            "<OPEN>": "open",
            "<HIGH>": "high",
            "<LOW>": "low",
            "<CLOSE>": "close",
            "<TICKVOL>": "volume",
            "<VOL>": "vol",
        })
    else:
        cols_lower = {c.lower(): c for c in df.columns}
        if "timestamp" in cols_lower:
            ts_col = cols_lower["timestamp"]
            if pd.api.types.is_numeric_dtype(df[ts_col]):
                df["timestamp"] = pd.to_datetime(df[ts_col], unit="ms", utc=True, errors="coerce")
            else:
                df["timestamp"] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
        elif "date" in cols_lower:
            ts_col = cols_lower["date"]
            df["timestamp"] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
        else:
            raise ValueError("No timestamp/date column found in CSV")

    rename_norm = {}
    for col in df.columns:
        key = col.lower()
        if key in {"open", "high", "low", "close", "volume", "vol", "tickvol", "tick_volume"}:
            rename_norm[col] = "volume" if key in {"vol", "tickvol", "tick_volume"} else key
    df = df.rename(columns=rename_norm)

    if "volume" not in df.columns:
        df["volume"] = 0.0

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["timestamp", "open", "high", "low", "close"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df[["timestamp", "open", "high", "low", "close", "volume"]]


def load_backtest_data(asset_name: str, timeframe: str, direct_file_path: str | None = None):
    if direct_file_path:
        if not os.path.exists(direct_file_path):
            return None, None, None
        contract_id = get_contract_id_by_name(asset_name) or asset_name
        data = _read_csv_data(direct_file_path)
        return data, (contract_id, timeframe, "backtest_account"), contract_id

    contract_id = get_contract_id_by_name(asset_name)
    if not contract_id:
        return None, None, None

    safe_name = _safe_asset_name(asset_name)
    file_name = TIMEFRAME_FILE_MAP.get(timeframe, f"{timeframe}.csv")
    data_path = os.path.join("data", safe_name, file_name)
    if not os.path.exists(data_path):
        return None, None, None

    data = _read_csv_data(data_path)
    return data, (contract_id, timeframe, "backtest_account"), contract_id


def _apply_strategy_params(params: dict):
    for key, value in params.items():
        setattr(fvg_module, key, value)


def _get_current_strategy_params() -> dict:
    params = {}
    for name in OPTIMIZATION_CONFIG.keys():
        params[name] = getattr(fvg_module, name, OPTIMIZATION_CONFIG[name]["current"])
    for name, default in FIXED_PARAMS.items():
        params[name] = getattr(fvg_module, name, default)
    return params


def _sync_defaults_to_fvg():
    defaults = {
        "FVG_HISTORY_NBR": FVG_HISTORY_NBR,
        "MIN_FVG_POWER_PCT": MIN_FVG_POWER_PCT,
        "HTF_TF": HTF_TF,
        "EMA_PERIOD": EMA_PERIOD,
        "VOLUME_MULTIPLIER": VOLUME_MULTIPLIER,
        "USE_VOLUME_CHECK": USE_VOLUME_CHECK,
        "VOLUME_DATA_START_TIMESTAMP": VOLUME_DATA_START_TIMESTAMP,
        "START_FROM_VOLUME_TIMESTAMP": START_FROM_VOLUME_TIMESTAMP,
        "ATR_PERIOD": ATR_PERIOD,
        "SL_MULTIPLIER": SL_MULTIPLIER,
        "TP_MULTIPLIER": TP_MULTIPLIER,
        "USE_TRAILING": USE_TRAILING,
        "TRAIL_OFFSET_MULT": TRAIL_OFFSET_MULT,
        "HOLD_UNTIL_OPPOSITE": HOLD_UNTIL_OPPOSITE,
    }
    _apply_strategy_params(defaults)


def get_optimal_worker_count() -> int:
    if not USE_AUTO_WORKERS:
        return max(1, MAX_WORKERS)
    cpu_count = multiprocessing.cpu_count()
    if USE_MULTIPROCESSING:
        return max(1, (cpu_count // 2) - 1)
    return max(1, int(cpu_count * 1.5))


class BacktestOrder(FVG_Order):
    def __init__(
        self,
        entry_atr: float,
        side: str,
        entry_price: float,
        take_profit: float,
        stop_loss: float | None,
        trailing_stop_loss: float | None,
        order_size: float,
        tick_size: float | None,
        tick_value: float | None,
        round_turn_fee: float | None,
        use_cfd_pricing: bool = False,
        cfd_leverage: float = 1.0,
        cfd_fee_pct: float = 0.0,
        cfd_spread: float = 0.0,
        use_margin_per_trade: bool = False,
    ):
        super().__init__(
            entry_atr=entry_atr,
            side=side,
            entry_price=entry_price,
            order_size=order_size,
            take_profit=take_profit,
            stop_loss=stop_loss,
            trailing_stop_loss=trailing_stop_loss,
            use_trailing=True,
        )
        self.lot_size = order_size

        self.tick_size = tick_size
        self.tick_value = tick_value
        self.round_turn_fee = float(round_turn_fee or 0.0)

        self.use_cfd_pricing = bool(use_cfd_pricing)
        self.cfd_leverage = float(cfd_leverage or 1.0)
        self.cfd_fee_pct = float(cfd_fee_pct or 0.0)
        self.cfd_spread = float(cfd_spread or 0.0)
        self.use_margin_per_trade = bool(use_margin_per_trade)

        self.filled = False
        self.fill_price = None
        self.fill_time = None
        self.exit_price = None
        self.exit_time = None
        self.exit_reason = None
        self.pnl = 0.0
        self.pnl_pct = 0.0
        self.entry_bar = None

    def place_order(self, fill_time=None, entry_bar=None):
        self.filled = True
        self.fill_price = self.entry_price
        self.fill_time = fill_time or datetime.now()
        self.entry_bar = entry_bar
        return {"success": True, "order_id": "backtest_order", "message": "Order filled"}

    def _effective_price(self, price: float, is_entry: bool) -> float:
        if not self.use_cfd_pricing or self.cfd_spread <= 0:
            return price
        half_spread = self.cfd_spread / 2
        if self.side == "BUY":
            return price + half_spread if is_entry else price - half_spread
        return price - half_spread if is_entry else price + half_spread

    def unrealized_pnl(self, current_price: float) -> float:
        if not self.filled:
            return 0.0
        entry_price = self._effective_price(self.fill_price, is_entry=True)
        exit_price = self._effective_price(current_price, is_entry=False)
        price_diff = exit_price - entry_price if self.side == "BUY" else entry_price - exit_price
        leverage = self.cfd_leverage if self.use_cfd_pricing else 1.0
        leverage = 1.0 if self.use_margin_per_trade else leverage
        if self.tick_size and self.tick_value and self.tick_size > 0:
            ticks = price_diff / self.tick_size
            return ticks * self.tick_value * self.lot_size * leverage
        return price_diff * self.lot_size * leverage

    def close_order(self, exit_price: float, exit_time: datetime, reason: str):
        if not self.filled:
            return
        self.exit_price = exit_price
        self.exit_time = exit_time
        self.exit_reason = reason

        entry_price = self._effective_price(self.fill_price, is_entry=True)
        exit_price_eff = self._effective_price(exit_price, is_entry=False)
        price_diff = exit_price_eff - entry_price if self.side == "BUY" else entry_price - exit_price_eff

        leverage = self.cfd_leverage if self.use_cfd_pricing else 1.0
        leverage = 1.0 if self.use_margin_per_trade else leverage

        if self.tick_size and self.tick_value and self.tick_size > 0:
            ticks = price_diff / self.tick_size
            gross = ticks * self.tick_value * self.lot_size * leverage
        else:
            gross = price_diff * self.lot_size * leverage

        if self.use_cfd_pricing and self.cfd_fee_pct > 0:
            entry_value = entry_price * self.lot_size * leverage
            fees = entry_value * self.cfd_fee_pct
        else:
            fees = self.round_turn_fee * self.lot_size

        self.pnl = gross - fees
        entry_value = entry_price * self.lot_size * leverage
        self.pnl_pct = (self.pnl / entry_value * 100) if entry_value > 0 else 0.0

    def check_close_conditions(self, log=print, **kwargs) -> bool:
        return False


class BacktestStrategy(FVG_Strategy):
    Order = BacktestOrder

    def __init__(
        self,
        asset_tuple,
        historical_data: pd.DataFrame,
        initial_balance: float = BACKTEST_INITIAL_BALANCE,
        start_date=None,
        end_date=None,
        max_loss=None,
        asset_name=None,
        strategy_params=None,
    ):
        self.asset = asset_tuple[0]
        self.timeframe = asset_tuple[1]
        self.account_name = asset_tuple[2]

        self.asset_name = asset_name or self.asset
        self.historical_data = self._filter_dates(historical_data, start_date, end_date)
        if USE_FIRST_TENTH_ONLY:
            tenth = max(1, len(self.historical_data) // 10)
            self.historical_data = self.historical_data.iloc[:tenth].reset_index(drop=True)

        start_from_ts = START_FROM_VOLUME_TIMESTAMP or USE_VOLUME_CHECK
        if start_from_ts and "timestamp" in self.historical_data.columns:
            ts = pd.to_datetime(VOLUME_DATA_START_TIMESTAMP, unit="ms", utc=True, errors="coerce")
            self.historical_data = self.historical_data[self.historical_data["timestamp"] >= ts]
            self.historical_data = self.historical_data.reset_index(drop=True)

        if len(self.historical_data) == 0:
            raise ValueError("No historical data available after filtering.")

        self.initial_balance = float(initial_balance)
        self.current_balance = float(initial_balance)
        self.account_balance = float(initial_balance)

        filename = f"backtest-{self.asset}-{self.timeframe}-{self.account_name}"
        self.csv_filename = f"{filename}.csv"
        self.metadata_filename = f"{filename}.json"

        self.trades = []
        self.equity_curve = []

        self.strategy_params = strategy_params or _get_current_strategy_params()

        self.max_loss = max_loss
        if max_loss is not None and 0 < max_loss < 1:
            self.max_loss_amount = self.initial_balance * max_loss
        else:
            self.max_loss_amount = max_loss

        contract_info = get_contract_info(self.asset)
        self.tick_size = contract_info.get("tick_size")
        self.tick_value = contract_info.get("tick_value")
        self.round_turn_fee = get_round_turn_fee(self.asset)

        self.use_cfd_pricing = bool(USE_CFD_PRICING)
        if self.use_cfd_pricing:
            settings = CFD_SETTINGS_BY_ASSET.get(self.asset_name) or DEFAULT_CFD_SETTINGS
            self.cfd_leverage = float(settings.get("leverage", 1.0))
            self.cfd_fee_pct = float(settings.get("fee_pct", 0.0))
            self.cfd_spread = float(settings.get("spread", 0.0))
        else:
            self.cfd_leverage = 1.0
            self.cfd_fee_pct = 0.0
            self.cfd_spread = 0.0

        super().__init__()

    def api_order_kwargs(self) -> dict:
        return {}

    def load_metadata(self) -> bool:
        return False

    def gather_data(self) -> pd.DataFrame:
        # Ensure we start far enough in to compute HTF EMA reliably.
        tf_minutes = int("".join(filter(str.isdigit, str(self.timeframe))) or 1)
        htf_minutes = int(str(getattr(fvg_module, "HTF_TF", "240")))
        bars_per_htf = max(1, htf_minutes // tf_minutes)
        ema_period = int(getattr(fvg_module, "EMA_PERIOD", 100))
        bars_needed_htf = max(101, ema_period + 51)
        window_needed = max(100, bars_needed_htf * bars_per_htf)
        if len(self.historical_data) < window_needed:
            print(
                f"⚠️ Not enough data for HTF EMA warmup. "
                f"Need {window_needed} bars, have {len(self.historical_data)}."
            )
            window = min(len(self.historical_data), 100)
        else:
            window = window_needed
        self.current_bar_index = window
        return self.historical_data.iloc[:window].copy()

    def fetch_new_data(self):
        if self.current_bar_index >= len(self.historical_data):
            return None
        start_idx = max(0, self.current_bar_index - 99)
        end_idx = self.current_bar_index + 1
        self.data = self.historical_data.iloc[start_idx:end_idx].copy()
        self.current_bar_index += 1
        self.cur_close = float(self.data["close"].iloc[-1])
        self.cur_volume = float(self.data["volume"].iloc[-1]) if "volume" in self.data.columns else 0.0
        return self.data.iloc[-1:]

    def fetch_htf_data(self) -> pd.DataFrame:
        if self.current_bar_index <= 0:
            return pd.DataFrame()
        df = self.historical_data.iloc[: self.current_bar_index].copy()
        if len(df) == 0:
            return pd.DataFrame()

        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True, errors="coerce")
        df = df.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()

        htf_rule = self._htf_rule()
        agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
        if "volume" in df.columns:
            agg["volume"] = "sum"
        htf = df.resample(htf_rule, label="right", closed="right").agg(agg).dropna()
        if len(htf) == 0:
            return pd.DataFrame()

        bars_needed = max(101, int(getattr(fvg_module, "EMA_PERIOD", 100)) + 51)
        htf = htf.iloc[-bars_needed:].reset_index()
        htf.loc[htf.index[-1], "close"] = float(self.cur_close)
        return htf

    def _update_trend_indicators(self):
        bars = self.fetch_htf_data()
        ema_period = int(getattr(fvg_module, "EMA_PERIOD", 100))
        htfEMA = ema(bars, ema_period)
        if htfEMA is None and isinstance(bars, pd.DataFrame) and len(bars) > 1:
            fallback_len = min(len(bars), ema_period)
            htfEMA = ema(bars, fallback_len)

        if htfEMA is None:
            self.isBullishHTF = None
            self.isBearishHTF = None
        else:
            self.isBullishHTF = self.cur_close > htfEMA
            self.isBearishHTF = self.cur_close < htfEMA

        atrVal = get_atr(self.data, getattr(fvg_module, "ATR_PERIOD", 14))
        atr_sma = sma(atrVal, 20) if len(atrVal) > 0 else None
        atrOK = atrVal.iloc[-1] > atr_sma if (len(atrVal) > 0 and atr_sma is not None) else False

        if getattr(fvg_module, "USE_VOLUME_CHECK", False) and "volume" in self.data.columns:
            vol_sma = sma(self.data["volume"], 20)
            volOK = self.cur_volume > vol_sma * getattr(fvg_module, "VOLUME_MULTIPLIER", 1.2) if vol_sma is not None else False
            self.marketOK = volOK and atrOK
        else:
            self.marketOK = atrOK

        # Match backtest.py detection window (no look-ahead)
        self.lastBullFvg = self.data["high"].iloc[-3] < self.data["low"].iloc[-1] and not self.lastBullFvg
        self.lastBearFvg = self.data["low"].iloc[-3] > self.data["high"].iloc[-1] and not self.lastBearFvg

    def update_indicators(self):
        self._update_trend_indicators()

        gapClose = self.data["close"].iloc[-3]
        min_power_pct = MIN_FVG_POWER_PCT

        self.bullishPowerOK = (
            self.lastBullFvg
            and (self.data["low"].iloc[-2] - self.data["high"].iloc[-4]) / gapClose * 100 >= min_power_pct
        )

        self.bearishPowerOK = (
            self.lastBearFvg
            and (self.data["low"].iloc[-4] - self.data["high"].iloc[-2]) / gapClose * 100 >= min_power_pct
        )

        self._calc_BOS_and_CHOCH()

    def check_daily_trade_limit(self):
        day = pd.to_datetime(self.data["timestamp"].iloc[-1]).date()
        if self.last_trade_date != str(day):
            self.daily_trades_count = 0
            self.last_trade_date = str(day)
        return self.daily_trades_count < getattr(fvg_module, "MAX_DAILY_TRADES", 3)

    def calculate_order_size(self, **kwargs):
        atr = kwargs.get("atr")
        sl_mult = kwargs.get("sl_mult")
        entry_price = kwargs.get("entry_price")

        if getattr(fvg_module, "USE_FIXED_LOT", False):
            return getattr(fvg_module, "FIXED_LOT", 1)

        if USE_MARGIN_PER_TRADE and entry_price:
            leverage = self.cfd_leverage if self.use_cfd_pricing else 1.0
            notional = MARGIN_PER_TRADE * leverage
            lot = max(MIN_MARGIN_LOT_SIZE, notional / entry_price)
            if MAX_MARGIN_LOT_SIZE is not None:
                lot = min(MAX_MARGIN_LOT_SIZE, lot)
            return lot

        if atr is None or sl_mult is None:
            return getattr(fvg_module, "ORDER_SIZE", 1)

        risk_amount = self.account_balance * (getattr(fvg_module, "RISK_PERCENT", 1.0) / 100)
        stop_distance = atr * sl_mult
        if stop_distance <= 0:
            return getattr(fvg_module, "ORDER_SIZE", 1)
        lot = risk_amount / stop_distance
        return max(0.01, min(round(lot, 2), 100))

    def subscribe_to_price_updates(self):
        return None

    def start_bar_iterations(self):
        while self.current_bar_index < len(self.historical_data):
            self.bar_iteration()
            self._record_equity()
            if self._max_loss_hit():
                break

    def run(self):
        self.start_bar_iterations()

    def entry_logic(self):
        if len(self.fvg_zones) == 0 or len(self.active_orders) > 0:
            return
        if not self.check_daily_trade_limit():
            return

        current_high = self.data["high"].iloc[-1]
        current_low = self.data["low"].iloc[-1]
        atr = get_atr(self.data, getattr(fvg_module, "ATR_PERIOD", 14)).iloc[-1]

        for zone in self.fvg_zones[-getattr(fvg_module, "FVG_HISTORY_NBR", 3):]:
            if zone["mitigated"]:
                continue

            fvg_bottom = zone["bottom"]
            fvg_top = zone["top"]
            touches_fvg = current_high >= fvg_bottom and current_low <= fvg_top

            if zone["direction"] == "bull" and touches_fvg and self.isBullishHTF and self.marketOK:
                entry_price = fvg_top if ALLOW_INTRACANDLE_CHECKS else self.cur_close
                stop_loss = entry_price - atr * getattr(fvg_module, "SL_MULTIPLIER", 4.0)
                tp = entry_price + atr * getattr(fvg_module, "TP_MULTIPLIER", 2000000.0)
                trail = entry_price - atr * getattr(fvg_module, "TRAIL_OFFSET_MULT", 6.0) if getattr(fvg_module, "USE_TRAILING", True) else None
                lot_size = self.calculate_order_size(atr=atr, sl_mult=getattr(fvg_module, "SL_MULTIPLIER", 4.0), entry_price=entry_price)

                order = self.Order(
                    entry_atr=atr,
                    side="BUY",
                    entry_price=entry_price,
                    take_profit=tp,
                    stop_loss=stop_loss,
                    trailing_stop_loss=trail,
                    order_size=lot_size,
                    tick_size=self.tick_size,
                    tick_value=self.tick_value,
                    round_turn_fee=self.round_turn_fee,
                    use_cfd_pricing=self.use_cfd_pricing,
                    cfd_leverage=self.cfd_leverage,
                    cfd_fee_pct=self.cfd_fee_pct,
                    cfd_spread=self.cfd_spread,
                    use_margin_per_trade=USE_MARGIN_PER_TRADE,
                )
                order.place_order(fill_time=self._bar_timestamp(), entry_bar=self.current_bar_index)
                self.active_orders.append(order)
                zone["mitigated"] = True
                self.lastPositionWasLong = True
                self.lastPositionWasShort = False
                self.daily_trades_count += 1
                self.last_trade_date = str(self._bar_timestamp().date())
                break

            if zone["direction"] == "bear" and touches_fvg and self.isBearishHTF and self.marketOK:
                entry_price = fvg_bottom if ALLOW_INTRACANDLE_CHECKS else self.cur_close
                stop_loss = entry_price + atr * getattr(fvg_module, "SL_MULTIPLIER", 4.0)
                tp = entry_price - atr * getattr(fvg_module, "TP_MULTIPLIER", 2000000.0)
                trail = entry_price + atr * getattr(fvg_module, "TRAIL_OFFSET_MULT", 6.0) if getattr(fvg_module, "USE_TRAILING", True) else None
                lot_size = self.calculate_order_size(atr=atr, sl_mult=getattr(fvg_module, "SL_MULTIPLIER", 4.0), entry_price=entry_price)

                order = self.Order(
                    entry_atr=atr,
                    side="SELL",
                    entry_price=entry_price,
                    take_profit=tp,
                    stop_loss=stop_loss,
                    trailing_stop_loss=trail,
                    order_size=lot_size,
                    tick_size=self.tick_size,
                    tick_value=self.tick_value,
                    round_turn_fee=self.round_turn_fee,
                    use_cfd_pricing=self.use_cfd_pricing,
                    cfd_leverage=self.cfd_leverage,
                    cfd_fee_pct=self.cfd_fee_pct,
                    cfd_spread=self.cfd_spread,
                    use_margin_per_trade=USE_MARGIN_PER_TRADE,
                )
                order.place_order(fill_time=self._bar_timestamp(), entry_bar=self.current_bar_index)
                self.active_orders.append(order)
                zone["mitigated"] = True
                self.lastPositionWasShort = True
                self.lastPositionWasLong = False
                self.daily_trades_count += 1
                self.last_trade_date = str(self._bar_timestamp().date())
                break

    def update_stops(self):
        if len(self.active_orders) == 0:
            return

        order = self.active_orders[0]
        bar = self.data.iloc[-1]
        high = bar["high"]
        low = bar["low"]
        close = bar["close"]
        ts = self._bar_timestamp()

        if self.lastPositionWasLong:
            if getattr(fvg_module, "USE_TRAILING", True) and order.entry_atr is not None:
                trail_candidate = high - order.entry_atr * getattr(fvg_module, "TRAIL_OFFSET_MULT", 6.0)
                if order.trailing_stop_loss is None or trail_candidate > order.trailing_stop_loss:
                    order.trailing_stop_loss = trail_candidate

            if order.trailing_stop_loss is not None and low <= order.trailing_stop_loss:
                order.close_order(order.trailing_stop_loss, ts, "Stop Loss")
                self._record_trade(order)
                self._close_position()
                return
            if order.stop_loss is not None and low <= order.stop_loss:
                order.close_order(order.stop_loss, ts, "Stop Loss")
                self._record_trade(order)
                self._close_position()
                return
            if high >= order.take_profit:
                order.close_order(order.take_profit, ts, "Take Profit")
                self._record_trade(order)
                self._close_position()
                return

        if self.lastPositionWasShort:
            if getattr(fvg_module, "USE_TRAILING", True) and order.entry_atr is not None:
                trail_candidate = low + order.entry_atr * getattr(fvg_module, "TRAIL_OFFSET_MULT", 6.0)
                if order.trailing_stop_loss is None or trail_candidate < order.trailing_stop_loss:
                    order.trailing_stop_loss = trail_candidate

            if order.trailing_stop_loss is not None and high >= order.trailing_stop_loss:
                order.close_order(order.trailing_stop_loss, ts, "Stop Loss")
                self._record_trade(order)
                self._close_position()
                return
            if order.stop_loss is not None and high >= order.stop_loss:
                order.close_order(order.stop_loss, ts, "Stop Loss")
                self._record_trade(order)
                self._close_position()
                return
            if low <= order.take_profit:
                order.close_order(order.take_profit, ts, "Take Profit")
                self._record_trade(order)
                self._close_position()
                return

        if getattr(fvg_module, "HOLD_UNTIL_OPPOSITE", True):
            if self.lastPositionWasLong and self.isCHOCH:
                order.close_order(close, ts, "CHoCH")
                self._record_trade(order)
                self._close_position()
                return
            if self.lastPositionWasShort and self.isBOS:
                order.close_order(close, ts, "BOS")
                self._record_trade(order)
                self._close_position()
                return

    def _record_trade(self, order: BacktestOrder):
        trade = {
            "entry_time": order.fill_time,
            "exit_time": order.exit_time,
            "side": order.side,
            "entry_price": order.fill_price,
            "exit_price": order.exit_price,
            "size": order.lot_size,
            "pnl": order.pnl,
            "pnl_pct": order.pnl_pct,
            "fees": order.round_turn_fee * order.lot_size,
            "exit_reason": order.exit_reason,
            "entry_bar": order.entry_bar,
            "exit_bar": self.current_bar_index,
        }
        self.trades.append(trade)
        self.current_balance += order.pnl

    def _record_equity(self):
        unrealized = 0.0
        if self.active_orders and self.active_orders[0].filled:
            unrealized = self.active_orders[0].unrealized_pnl(self.cur_close)
        self.equity_curve.append(
            {
                "bar": self.current_bar_index,
                "timestamp": self._bar_timestamp(),
                "balance": self.current_balance,
                "equity": self.current_balance + unrealized,
            }
        )

    def _close_position(self):
        self.active_orders = []
        self.lastPositionWasLong = False
        self.lastPositionWasShort = False

    def _filter_dates(self, data: pd.DataFrame, start_date, end_date) -> pd.DataFrame:
        if data is None or len(data) == 0:
            return pd.DataFrame()
        df = data.copy()
        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True, errors="coerce")
        if start_date:
            start_ts = _parse_datetime_input(start_date)
            df = df[df["timestamp"] >= start_ts]
        if end_date:
            end_ts = _parse_datetime_input(end_date)
            df = df[df["timestamp"] <= end_ts]
        return df.reset_index(drop=True)

    def _htf_rule(self) -> str:
        htf = str(getattr(fvg_module, "HTF_TF", "240"))
        if htf.isdigit():
            return f"{htf}min"
        return htf

    def _bar_timestamp(self) -> datetime:
        ts = self.data["timestamp"].iloc[-1]
        if isinstance(ts, datetime):
            return ts
        if pd.isna(ts):
            return pd.NaT
        if isinstance(ts, (int, float, np.integer, np.floating)):
            return pd.to_datetime(ts, unit="ms", utc=True, errors="coerce")
        return pd.to_datetime(ts, utc=True, errors="coerce")

    def _max_loss_hit(self) -> bool:
        if self.max_loss_amount is None:
            return False
        unrealized = 0.0
        if self.active_orders and self.active_orders[0].filled:
            unrealized = self.active_orders[0].unrealized_pnl(self.cur_close)
        equity = self.current_balance + unrealized
        return (self.initial_balance - equity) >= self.max_loss_amount


def _compute_results(trades: list, equity_curve: list, initial_balance: float) -> dict:
    if not trades:
        return {
            "total_pnl": 0.0,
            "total_trades": 0,
            "win_rate": 0.0,
            "final_balance": initial_balance,
            "max_drawdown": 0.0,
            "max_drawdown_pct": 0.0,
            "total_fees": 0.0,
            "winning_trades": 0,
            "losing_trades": 0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "profit_factor": 0.0,
            "total_return": 0.0,
            "net_profit": 0.0,
            "largest_win": 0.0,
            "largest_loss": 0.0,
            "trades_per_day": 0.0,
            "backtest_period_days": 0,
        }

    trades_df = pd.DataFrame(trades)
    total_pnl = trades_df["pnl"].sum()
    total_trades = len(trades_df)
    winning = trades_df[trades_df["pnl"] > 0]
    losing = trades_df[trades_df["pnl"] < 0]

    win_rate = (len(winning) / total_trades * 100) if total_trades else 0.0
    avg_win = winning["pnl"].mean() if len(winning) else 0.0
    avg_loss = losing["pnl"].mean() if len(losing) else 0.0
    profit_factor = abs(avg_win * len(winning) / (avg_loss * len(losing))) if len(losing) and avg_loss else 0.0

    final_balance = initial_balance + total_pnl
    total_return = ((final_balance - initial_balance) / initial_balance * 100) if initial_balance else 0.0
    net_profit = final_balance - initial_balance

    max_drawdown = 0.0
    max_drawdown_pct = 0.0
    if equity_curve:
        eq = pd.DataFrame(equity_curve)
        if "equity" in eq.columns:
            values = eq["equity"].values
            running_max = np.maximum.accumulate(values)
            drawdowns = running_max - values
            max_drawdown = float(np.max(drawdowns)) if len(drawdowns) else 0.0
            peak = float(np.max(values)) if len(values) else initial_balance
            max_drawdown_pct = (max_drawdown / peak * 100) if peak else 0.0

    trades_per_day = 0.0
    backtest_days = 0
    if "entry_time" in trades_df.columns and len(trades_df) > 0:
        start = pd.to_datetime(trades_df["entry_time"].iloc[0])
        end = pd.to_datetime(trades_df["exit_time"].iloc[-1])
        backtest_days = (end - start).days + 1
        trades_per_day = total_trades / backtest_days if backtest_days > 0 else 0.0

    return {
        "total_pnl": total_pnl,
        "total_trades": total_trades,
        "win_rate": win_rate,
        "final_balance": final_balance,
        "max_drawdown": max_drawdown,
        "max_drawdown_pct": max_drawdown_pct,
        "total_fees": float(trades_df["fees"].sum()) if "fees" in trades_df.columns else 0.0,
        "winning_trades": len(winning),
        "losing_trades": len(losing),
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
        "total_return": total_return,
        "net_profit": net_profit,
        "largest_win": float(trades_df["pnl"].max()),
        "largest_loss": float(trades_df["pnl"].min()),
        "trades_per_day": trades_per_day,
        "backtest_period_days": backtest_days,
    }


def _save_summary(results: dict, params: dict, asset_name: str, timeframe: str, initial_balance: float, max_loss):
    row = {
        "id": params.get("result_id"),
        "asset_name": asset_name,
        "timeframe": timeframe,
        "initial_balance": initial_balance,
        "max_loss": max_loss,
        **params,
        **results,
        "backtest_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    os.makedirs(OUT_PATH, exist_ok=True)
    path = os.path.join(OUT_PATH, "backtest_summary.csv")
    with SUMMARY_CSV_LOCK:
        df_new = pd.DataFrame([row])
        if os.path.exists(path):
            df_existing = pd.read_csv(path)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            df_combined.to_csv(path, index=False)
        else:
            df_new.to_csv(path, index=False)
    return path


def _next_result_id(out_path: str) -> int:
    path = os.path.join(out_path, "final_result.csv")
    if not os.path.exists(path):
        return 1
    try:
        df = pd.read_csv(path)
        if "id" in df.columns and not df["id"].empty:
            return int(df["id"].max()) + 1
    except Exception:
        pass
    return 1


def _save_final_result(result_id: int, results: dict, params: dict, asset_name: str, timeframe: str):
    row = {
        "id": result_id,
        "timeframe": timeframe,
        **params,
        **results,
        "strategy_failed": False,
        "failed_reason": None,
        "backtest_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    os.makedirs(OUT_PATH, exist_ok=True)
    path = os.path.join(OUT_PATH, "final_result.csv")
    with FINAL_RESULT_LOCK:
        df_new = pd.DataFrame([row])
        if os.path.exists(path):
            df_existing = pd.read_csv(path)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            df_combined.to_csv(path, index=False)
        else:
            df_new.to_csv(path, index=False)
    return path


def _plot_equity_curve(equity_curve: list, filename: str):
    if not equity_curve:
        return
    eq = pd.DataFrame(equity_curve)
    if "timestamp" in eq.columns:
        eq["timestamp"] = pd.to_datetime(eq["timestamp"], utc=True, errors="coerce")
        eq = eq.dropna(subset=["timestamp"])
        if eq.empty:
            return
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    ax1.plot(eq["timestamp"], eq["equity"], label="Equity", linewidth=1.5)
    ax2.plot(eq["timestamp"], eq["balance"], label="Balance", linewidth=1.5, color="#A23B72")
    ax1.set_ylabel("Equity")
    ax2.set_ylabel("Balance")
    ax2.set_xlabel("Date")
    ax1.grid(True, alpha=0.3)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")


def _save_result_row(row: dict, csv_path: str):
    with RESULTS_CSV_LOCK:
        df_new = pd.DataFrame([row])
        if os.path.exists(csv_path):
            df_existing = pd.read_csv(csv_path)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            df_combined.to_csv(csv_path, index=False)
        else:
            df_new.to_csv(csv_path, index=False)


def _run_single_backtest(asset_name, timeframe, initial_balance, max_loss, params, direct_file_path=None):
    data, asset_tuple, _ = load_backtest_data(asset_name, timeframe, direct_file_path=direct_file_path)
    if data is None or len(data) == 0:
        return None
    _apply_strategy_params(params)
    strat = BacktestStrategy(
        asset_tuple=asset_tuple,
        historical_data=data,
        initial_balance=initial_balance,
        start_date=BACKTEST_START_DATE,
        end_date=BACKTEST_END_DATE,
        max_loss=max_loss,
        asset_name=asset_name,
        strategy_params=params,
    )
    strat.run()
    results = _compute_results(strat.trades, strat.equity_curve, initial_balance)
    return results


def _run_task_payload(payload):
    asset_name, timeframe, initial_balance, max_loss, params, direct_file_path = payload
    res = _run_single_backtest(
        asset_name,
        timeframe,
        initial_balance,
        max_loss,
        params,
        direct_file_path=direct_file_path,
    )
    if res:
        return {"timeframe": timeframe, **params, **res}
    return None


def optimize_random(asset_name, timeframes, initial_balance, max_loss, num_samples, direct_file_path=None, csv_input_file=None):
    if csv_input_file and os.path.exists(csv_input_file):
        df = pd.read_csv(csv_input_file)
        combos = df.to_dict(orient="records")
        for combo in combos:
            for key, val in FIXED_PARAMS.items():
                combo.setdefault(key, val)
    else:
        combos = []
        for _ in range(num_samples):
            combo = {k: random.choice(v["range"]) for k, v in OPTIMIZATION_CONFIG.items()}
            combo.update(FIXED_PARAMS)
            combo["timeframe"] = random.choice(timeframes)
            combos.append(combo)

    results = []
    tasks = []
    for combo in combos:
        timeframe = combo.get("timeframe", BACKTEST_SELECTED_TIMEFRAME)
        params = {k: v for k, v in combo.items() if k != "timeframe"}
        tasks.append((asset_name, timeframe, initial_balance, max_loss, params, direct_file_path))

    if USE_MULTIPROCESSING:
        with ProcessPoolExecutor(max_workers=get_optimal_worker_count()) as executor:
            for result in executor.map(_run_task_payload, tasks):
                if result:
                    results.append(result)
    elif USE_MULTITHREADING:
        with ThreadPoolExecutor(max_workers=get_optimal_worker_count()) as executor:
            for result in executor.map(_run_task_payload, tasks):
                if result:
                    results.append(result)
    else:
        for task in tasks:
            result = _run_task_payload(task)
            if result:
                results.append(result)

    return results


def optimize_exhaustive(asset_name, timeframe, initial_balance, max_loss, direct_file_path=None):
    names = list(OPTIMIZATION_CONFIG.keys())
    ranges = [OPTIMIZATION_CONFIG[n]["range"] for n in names]
    results = []
    tasks = []
    for combo_values in itertools.product(*ranges):
        params = dict(zip(names, combo_values))
        params.update(FIXED_PARAMS)
        tasks.append((asset_name, timeframe, initial_balance, max_loss, params, direct_file_path))

    if USE_MULTIPROCESSING:
        with ProcessPoolExecutor(max_workers=get_optimal_worker_count()) as executor:
            for result in executor.map(_run_task_payload, tasks):
                if result:
                    results.append({k: v for k, v in result.items() if k != "timeframe"})
    elif USE_MULTITHREADING:
        with ThreadPoolExecutor(max_workers=get_optimal_worker_count()) as executor:
            for result in executor.map(_run_task_payload, tasks):
                if result:
                    results.append({k: v for k, v in result.items() if k != "timeframe"})
    else:
        for params in tasks:
            result = _run_task_payload(params)
            if result:
                results.append({k: v for k, v in result.items() if k != "timeframe"})

    return results


def run_single_backtest(asset_name, timeframe, initial_balance, max_loss, direct_file_path=None):
    _sync_defaults_to_fvg()
    data, asset_tuple, _ = load_backtest_data(asset_name, timeframe, direct_file_path=direct_file_path)
    if data is None or len(data) == 0:
        raise ValueError("No data loaded for backtest.")
    strat = BacktestStrategy(
        asset_tuple=asset_tuple,
        historical_data=data,
        initial_balance=initial_balance,
        start_date=BACKTEST_START_DATE,
        end_date=BACKTEST_END_DATE,
        max_loss=max_loss,
        asset_name=asset_name,
        strategy_params=_get_current_strategy_params(),
    )
    strat.run()
    results = _compute_results(strat.trades, strat.equity_curve, initial_balance)
    result_id = _next_result_id(OUT_PATH)
    params_with_id = {**strat.strategy_params, "result_id": result_id}
    _save_summary(results, params_with_id, asset_name, timeframe, initial_balance, max_loss)
    _save_final_result(result_id, results, strat.strategy_params, asset_name, timeframe)
    if SAVE_INDIVIDUAL_RESULTS_IN_OPTIMIZATION:
        os.makedirs(OUT_PATH, exist_ok=True)
        result_dir = os.path.join(OUT_PATH, str(result_id))
        os.makedirs(result_dir, exist_ok=True)
        trades_df = pd.DataFrame(strat.trades)
        equity_df = pd.DataFrame(strat.equity_curve)
        trades_df.to_csv(
            os.path.join(result_dir, f"backtest_trades_{asset_name}_{datetime.now().strftime('%Y%m%d')}.csv"),
            index=False,
        )
        equity_df.to_csv(
            os.path.join(result_dir, f"backtest_equity_{asset_name}_{datetime.now().strftime('%Y%m%d')}.csv"),
            index=False,
        )
        plot_path = os.path.join(
            result_dir, f"backtest_equity_curve_{asset_name}_{datetime.now().strftime('%Y%m%d')}.png"
        )
        _plot_equity_curve(strat.equity_curve, plot_path)
    return results


def run_backtest_main():
    initialize_backtest_data()

    direct_file_path = DIRECT_DATA_FILE_PATH if USE_DIRECT_DATA_FILE else None

    if RUN_OPTIMIZATION:
        if USE_EXHAUSTIVE_SEARCH:
            results = optimize_exhaustive(
                BACKTEST_ASSET_NAME,
                BACKTEST_SELECTED_TIMEFRAME,
                BACKTEST_INITIAL_BALANCE,
                BACKTEST_MAX_LOSS,
                direct_file_path=direct_file_path,
            )
        else:
            results = optimize_random(
                BACKTEST_ASSET_NAME,
                BACKTEST_TIMEFRAMES_TO_TEST or [BACKTEST_SELECTED_TIMEFRAME],
                BACKTEST_INITIAL_BALANCE,
                BACKTEST_MAX_LOSS,
                RANDOM_SEARCH_SAMPLES,
                direct_file_path=direct_file_path,
                csv_input_file=CSV_INPUT_FILE if USE_CSV_INPUT else None,
            )

        if results:
            os.makedirs(OUT_PATH, exist_ok=True)
            out_path = os.path.join(OUT_PATH, f"optimization_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
            pd.DataFrame(results).to_csv(out_path, index=False)
    else:
        run_single_backtest(
            BACKTEST_ASSET_NAME,
            BACKTEST_SELECTED_TIMEFRAME,
            BACKTEST_INITIAL_BALANCE,
            BACKTEST_MAX_LOSS,
            direct_file_path=direct_file_path,
        )


if __name__ == "__main__":
    run_backtest_main()

