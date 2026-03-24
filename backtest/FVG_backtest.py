import os
import csv
from datetime import datetime, timezone, timedelta
from pathlib import Path
import pandas as pd

BACKTEST_DIR = Path(__file__).resolve().parent
os.environ.setdefault("FVG_INPUTS_JSON", str(BACKTEST_DIR / "inputs.json"))

from FVG_projectX_bot.FVG_strategy import *
from FVG_projectX_bot.helping_functions.indicators import get_atr, ema
from FVG_projectX_bot.helping_functions.partial_close import (
    cleanup_partial_groups,
    get_partial_close_targets,
    next_partial_group_id,
)
from FVG_projectX_bot.helping_functions.pyramiding import (
    MaxOrdersPolicy,
    apply_pyramiding_add_on,
)


PARENT_DIR = Path(__file__).parents[1]
BACKTEST_RUNTIME_DIR = BACKTEST_DIR / INPUTS.RUNTIME_SUBDIR
BACKTEST_RUNTIME_DIR.mkdir(parents=True, exist_ok=True)

# ==================== USER CONFIG ====================
ASSET = "BTC"
TIMEFRAME = "15m"
INITIAL_BALANCE = 50
DATA_CSV_PATH = str(PARENT_DIR / "backtest" / "data" / "BTCUSDT_PERP_15m.csv")
# Optional 1min data for entry/close TP/SL (matches live behavior). If None, use 15m bar high/low.
DATA_1M_CSV_PATH = str(PARENT_DIR / "backtest" / "data" / "BTCUSDT_PERP_1m.csv")
START_TIMESTAMP = str(
    int((datetime.now(timezone.utc) - timedelta(days=182)).timestamp() * 1000)
)

# Pyramiding mode: "none", "client_atr", or "max_orders"
PYRAMIDING_MODE = "none" # remake to str enum
MAX_PYRAMID_ORDERS = 3

# Backtest data window (bars)
BACKTEST_WINDOW_BARS = None
USE_LAST_QUARTER_DATA = True  # If True, only use the last 25% of rows

# Contract / fee inputs
USE_CONTRACTS_CSV = False
CONTRACTS_CSV_PATH = str(PARENT_DIR.parent / "contracts.csv")
USE_ROUND_TURN_FEE = False
ROUND_TURN_FEE_USD = 3.5

# Margin trading inputs
USE_MARGIN_PRICING = True  # If False, use ROUND_TURN_FEE_USD (if enabled)
FEE_PCT = 0.001  # Round-turn fee as % of notional
LEVERAGE = 50
# Position sizing inputs (backtest only)
USE_MARGIN_PER_TRADE = True
MARGIN_PER_TRADE_USD = 10

# CSV output options
TRADE_CSV_WRITE_MODE = "append"  # "prepend" (newest first) or "append" (faster)

# Local aliases for strategy inputs used throughout this module.
HTF_TF = INPUTS.HTF_TF
EMA_PERIOD = INPUTS.EMA_PERIOD
ATR_PERIOD = INPUTS.ATR_PERIOD
USE_VOLUME_CHECK = INPUTS.USE_VOLUME_CHECK
VOLUME_MULTIPLIER = INPUTS.VOLUME_MULTIPLIER
MAX_DRAWDOWN_ENABLED = INPUTS.MAX_DRAWDOWN_ENABLED
MAX_DAILY_TRADES = INPUTS.MAX_DAILY_TRADES
ALLOW_INTRACANDLE_ENTRY = INPUTS.ALLOW_INTRACANDLE_ENTRY
SL_MULTIPLIER = INPUTS.SL_MULTIPLIER
TP_MULTIPLIER = INPUTS.TP_MULTIPLIER
USE_TRAILING = INPUTS.USE_TRAILING
TRAIL_OFFSET_MULT = INPUTS.TRAIL_OFFSET_MULT
FVG_HISTORY_NBR = INPUTS.FVG_HISTORY_NBR
SPLIT_ORDERS_ENABLED = INPUTS.SPLIT_ORDERS_ENABLED
EACH_TRADE_SIZE = INPUTS.EACH_TRADE_SIZE
USE_FIXED_LOT = INPUTS.USE_FIXED_LOT
FIXED_LOT = INPUTS.FIXED_LOT
RISK_PERCENT = INPUTS.RISK_PERCENT
ORDER_SIZE = INPUTS.ORDER_SIZE
PYR_ATR_STEP = INPUTS.PYR_ATR_STEP
PYR_ADD_ON_SIZE = INPUTS.PYR_ADD_ON_SIZE
PYR_MAX_ADDS = INPUTS.PYR_MAX_ADDS


_ACTIVE_BACKTEST = None


def _get_active_backtest():
    return _ACTIVE_BACKTEST


class BacktestOrder(FVG_Order):
    is_open: bool
    entry_time: datetime | None
    exit_time: datetime | None
    exit_price: float | None
    exit_reason: str | None
    pnl: float | None
    _last_price: float | None
    _last_timestamp: float| None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_open = False
        self.entry_time = None
        self.exit_time = None
        self.exit_price = None
        self.exit_reason = None
        self.pnl = None
        self._last_price = None
        self._last_timestamp = None
        self.margin_required = None
        self.margin_used = None
        self.fee_paid = None

    def _calc_notional(self, price: float | None = None) -> float:
        ref_price = price if price is not None else self.entry_price
        return abs(float(ref_price) * float(self.order_size or 0.0))

    def _calc_margin(self, notional: float) -> float:
        if LEVERAGE <= 0:
            return notional
        return notional / float(LEVERAGE)

    def _calc_fee(self, notional: float) -> float:
        if USE_MARGIN_PRICING:
            return notional * float(FEE_PCT)
        backtest = _get_active_backtest()
        if backtest is not None and backtest.round_turn_fee_usd is not None:
            return float(backtest.round_turn_fee_usd) * float(self.order_size or 0.0)
        # Legacy fallback (0.01% of notional)
        return notional * 0.0001

    def place_order(self):
        was_open = self.is_open
        notional = self._calc_notional()
        if USE_MARGIN_PRICING:
            required_margin = self._calc_margin(notional)
            backtest = _get_active_backtest()
            if backtest is not None:
                if backtest.account_balance < required_margin:
                    print(
                        "❌ OPEN rejected (insufficient margin): "
                        f"needed={required_margin:.4f} balance={backtest.account_balance:.4f}"
                    )
                    return {"success": False, "message": "Insufficient margin"}
                backtest.account_balance -= required_margin
                backtest.used_margin += required_margin
            if was_open and self.margin_required is not None:
                self.margin_required += required_margin
                if self.margin_used is not None:
                    self.margin_used += required_margin
            else:
                self.margin_required = required_margin
                self.margin_used = required_margin
        self.is_open = True
        group_id = getattr(self, "group_id", None)
        group_seq = getattr(self, "group_seq", None)
        print(
            "🧾 OPEN "
            f"side={self.side} size={self.order_size} entry={self.entry_price} "
            f"group_id={group_id} group_seq={group_seq}"
        )
        return {"success": True}

    def close_order(self):
        self.is_open = False
        if self._last_price is not None:
            self.exit_price = float(self._last_price)
        if self._last_timestamp is not None:
            self.exit_time = self._last_timestamp
        if self.exit_price is not None:
            backtest = _get_active_backtest()
            if backtest is not None:
                self.pnl = backtest.calculate_trade_pnl(self, self.exit_price, self.exit_time)
                self.fee_paid = backtest.calculate_trade_fee(self, self.exit_price, self.exit_time)
            else:
                entry_price = self.avg_entry_price if self.avg_entry_price is not None else self.entry_price
                price_delta = self.exit_price - entry_price
                direction = 1 if self.side == "BUY" else -1
                notional = self._calc_notional(price=entry_price)
                fee = self._calc_fee(notional)
                self.fee_paid = fee
                self.pnl = price_delta * direction * float(self.order_size or 0.0) - fee
        if USE_MARGIN_PRICING and self.margin_required:
            backtest = _get_active_backtest()
            if backtest is not None:
                backtest.account_balance += float(self.margin_required)
                backtest.used_margin -= float(self.margin_required)
            self.margin_required = None
        group_id = getattr(self, "group_id", None)
        group_seq = getattr(self, "group_seq", None)
        print(
            "🧾 CLOSE "
            f"side={self.side} size={self.order_size} entry={self.entry_price} "
            f"exit={self.exit_price} pnl={self.pnl} reason={self.exit_reason} "
            f"group_id={group_id} group_seq={group_seq}"
        )
        return {"success": True}

    def check_close_conditions(self, log=print, **kwargs) -> bool:
        self._last_price = kwargs.get("current_price")
        self._last_timestamp = kwargs.get("timestamp")
        closed = super().check_close_conditions(log=log, **kwargs)
        if closed and self.is_open:
            self.close_order()
        return closed


class FVG_Backtest(FVG_Strategy):
    Order = BacktestOrder

    def __init__(
        self,
        data_path: str,
        asset: str = "BTCUSDT",
        timeframe: str = "15m",
        initial_balance: float = 10000.0,
        warmup_bars: int | None = None,
        start_timestamp=None,
        pyramiding_mode: str | None = None,
        data_path_1m: str | None = None,
    ):
        global _ACTIVE_BACKTEST
        _ACTIVE_BACKTEST = self
        self.asset = asset
        self.timeframe = timeframe
        self.account_balance = float(initial_balance)
        self.used_margin = 0.0
        self.data_path = data_path
        self.data_path_1m = data_path_1m
        self.metadata_filename = os.path.join(BACKTEST_RUNTIME_DIR, "backtest_metadata.json")
        self.csv_filename = os.path.join(BACKTEST_RUNTIME_DIR, "backtest_data.csv")
        self._warmup_bars = warmup_bars
        self._full_data = None
        self._full_data_1m = None
        self._htf_resampled = None
        self._current_index = None
        self._cursor = 0
        self._current_dt = None
        self.trades = []
        self._stopped = False
        self.trades_csv_path = os.path.join(BACKTEST_RUNTIME_DIR, "backtest_trades.csv")
        self._start_from_dt = self._parse_start_timestamp(start_timestamp) if start_timestamp is not None else None
        self.tick_size = None
        self.tick_value = None
        self.round_turn_fee_usd = ROUND_TURN_FEE_USD if USE_ROUND_TURN_FEE else None
        if USE_CONTRACTS_CSV:
            self._load_contract_info()
        self._configure_pyramiding(pyramiding_mode)
        super().__init__()
        self.require_intrabar_entry = True

    def _next_partial_group_id(self) -> int:
        return next_partial_group_id(self, INPUTS)

    def _get_partial_close_targets(self, current_price: float) -> dict:
        return get_partial_close_targets(
            active_orders=list(self.active_orders),
            partial_groups=self._partial_groups,
            current_price=current_price,
            enable_partial_tp=INPUTS.ENABLE_PARTIAL_TP,
            enable_partial_sl=INPUTS.ENABLE_PARTIAL_SL,
            partial_tp_atr_step=INPUTS.PARTIAL_TP_ATR_STEP,
            partial_sl_atr_step=INPUTS.PARTIAL_SL_ATR_STEP,
            partial_tp_close_count=self._partial_tp_close_count,
            partial_sl_close_count=self._partial_sl_close_count,
        )

    def _cleanup_partial_groups(self) -> None:
        cleanup_partial_groups(self.active_orders, self._partial_groups)

    def _apply_pyramiding_add_on(
        self,
        current_price: float,
        current_high: float | None = None,
        current_low: float | None = None,
    ) -> None:
        apply_pyramiding_add_on(
            self,
            INPUTS,
            current_price,
            current_high,
            current_low,
        )

    def api_order_kwargs(self) -> dict:
        return {}

    def _configure_pyramiding(self, mode: str | None) -> None:
        selected = (mode or PYRAMIDING_MODE or "none").strip().lower()
        if selected in ("none", "off", "no"):
            self.pyramiding = NoPyramidingPolicy()
        elif selected in ("client_atr", "atr", "client"):
            self.pyramiding = ClientAtrPyramidingPolicy(
                atr_step=PYR_ATR_STEP,
                add_on_size=PYR_ADD_ON_SIZE,
                max_adds=PYR_MAX_ADDS,
            )
        elif selected in ("max_orders", "max", "orders"):
            self.pyramiding = MaxOrdersPolicy(MAX_PYRAMID_ORDERS)
        else:
            raise ValueError(
                f"Unknown pyramiding_mode '{mode}'. "
                "Use 'none', 'client_atr', or 'max_orders'."
            )
        print(f"📐 Backtest pyramiding mode: {selected}")

    def _load_data_from_path(self, path: str) -> pd.DataFrame:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Data file not found: {path}")
        sep = ","
        try:
            with open(path, "r", encoding="utf-8") as f:
                header_line = f.readline()
            if "\t" in header_line and "," not in header_line:
                sep = "\t"
        except Exception:
            sep = ","
        data = pd.read_csv(path, sep=sep)
        normalized = []
        for col in data.columns:
            name = str(col).strip().lower()
            if name.startswith("<") and name.endswith(">"):
                name = name[1:-1]
            normalized.append(name)
        data.columns = normalized
        if "timestamp" not in data.columns and "date" in data.columns and "time" in data.columns:
            combined = data["date"].astype(str) + " " + data["time"].astype(str)
            data["timestamp"] = pd.to_datetime(combined, errors="coerce", utc=True)
            data["timestamp"] = (data["timestamp"].astype("int64") // 10**6).where(
                data["timestamp"].notna()
            )
        required = {"open", "high", "low", "close"}
        missing = required - set(data.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")
        if "volume" not in data.columns:
            if "tickvol" in data.columns:
                data["volume"] = data["tickvol"]
            elif "vol" in data.columns:
                data["volume"] = data["vol"]
            else:
                data["volume"] = 0.0
        elif data["volume"].fillna(0).sum() == 0 and "tickvol" in data.columns:
            data["volume"] = data["tickvol"]
        if "timestamp" in data.columns:
            data = data.sort_values("timestamp").reset_index(drop=True)
        return data

    def _load_data(self) -> pd.DataFrame:
        data = self._load_data_from_path(self.data_path)
        if USE_LAST_QUARTER_DATA and not data.empty:
            start_idx = int(len(data) * 0.75)
            data = data.iloc[start_idx:].reset_index(drop=True)
        return data

    def _load_data_1m(self) -> pd.DataFrame | None:
        if not self.data_path_1m:
            return None
        try:
            return self._load_data_from_path(self.data_path_1m)
        except FileNotFoundError as e:
            print(f"⚠️ 1min data not found: {e}. Using 15m bar high/low for TP/SL.")
            return None

    def _get_1m_bars_for_15m_bar(self, bar_ts_ms: int) -> pd.DataFrame:
        """Return 1m bars whose open time falls in the 15m bar [bar_ts_ms, bar_ts_ms + 15*60*1000)."""
        if self._full_data_1m is None:
            return pd.DataFrame()
        ts_col = "timestamp"
        if ts_col not in self._full_data_1m.columns:
            return pd.DataFrame()
        end_ms = bar_ts_ms + 15 * 60 * 1000
        mask = (self._full_data_1m[ts_col] >= bar_ts_ms) & (self._full_data_1m[ts_col] < end_ms)
        return self._full_data_1m.loc[mask].copy()

    def _get_opened_eval_ts_ms_for_current_bar(self) -> int | None:
        """Last 1m bar timestamp in current 15m bar (for opened_eval_ts_ms). None if no 1m data."""
        if self._full_data_1m is None or self._current_dt is None:
            return None
        bar_ts_ms = int(self._current_dt.timestamp() * 1000)
        one_min = self._get_1m_bars_for_15m_bar(bar_ts_ms)
        if one_min.empty:
            return None
        return int(one_min["timestamp"].iloc[-1])

    def _load_contract_info(self) -> None:
        if not os.path.exists(CONTRACTS_CSV_PATH):
            print(f"⚠️ contracts.csv not found: {CONTRACTS_CSV_PATH}")
            return
        try:
            df = pd.read_csv(CONTRACTS_CSV_PATH)
        except Exception as exc:
            print(f"⚠️ Failed to read contracts.csv: {exc}")
            return
        if "name" not in df.columns:
            print("⚠️ contracts.csv missing 'name' column")
            return
        match = df[df["name"] == self.asset]
        if match.empty:
            print(f"⚠️ No contract row found for asset '{self.asset}' in contracts.csv")
            return
        row = match.iloc[0]
        self.tick_size = row.get("tickSize")
        self.tick_value = row.get("tickValue")
        print(
            f"📄 Contract info loaded for {self.asset}: "
            f"tickSize={self.tick_size} tickValue={self.tick_value}"
        )

    def _resample_htf_data(self, current_timestamp: datetime) -> pd.DataFrame:
        if self._full_data is None or "timestamp" not in self._full_data.columns:
            return pd.DataFrame()
        htf_data = self._full_data.copy()
        # Normalize timestamps to UTC datetimes even if stored as numeric strings.
        ts_series = htf_data["timestamp"]
        if pd.api.types.is_numeric_dtype(ts_series):
            max_val = pd.Series(ts_series).max()
            unit = "ms" if max_val > 10**12 else "s"
            htf_data["timestamp"] = pd.to_datetime(ts_series, unit=unit, utc=True)
        else:
            numeric_ts = pd.to_numeric(ts_series, errors="coerce")
            if numeric_ts.notna().any():
                max_val = numeric_ts.max()
                unit = "ms" if max_val > 10**12 else "s"
                htf_data["timestamp"] = pd.to_datetime(numeric_ts, unit=unit, utc=True)
            else:
                htf_data["timestamp"] = pd.to_datetime(ts_series, utc=True, errors="coerce")
        htf_data = htf_data[htf_data["timestamp"].notna()]
        htf_data = htf_data[htf_data["timestamp"] <= current_timestamp]
        if htf_data.empty:
            return pd.DataFrame()
        htf_data = htf_data.set_index("timestamp")

        htf_minutes = int(HTF_TF)
        if htf_minutes == 240:
            resample_period = "4h"
        elif htf_minutes == 120:
            resample_period = "2h"
        elif htf_minutes == 60:
            resample_period = "1h"
        elif htf_minutes == 1440:
            resample_period = "1d"
        elif htf_minutes >= 1440:
            resample_period = f"{htf_minutes // 1440}d"
        elif htf_minutes >= 60:
            resample_period = f"{htf_minutes // 60}h"
        else:
            resample_period = f"{htf_minutes}min"

        agg_dict = {"open": "first", "high": "max", "low": "min", "close": "last"}
        if "volume" in htf_data.columns:
            agg_dict["volume"] = "sum"
        # Use origin='epoch' to align bars to true timeframe boundaries (e.g. 00:00, 04:00, ...)
        htf_resampled = (
            htf_data.resample(
                resample_period, label="right", closed="right", origin="epoch"
            )
            .agg(agg_dict)
            .dropna()
            .reset_index()
        )
        return htf_resampled

    def _precompute_indicators(self) -> None:
        """
        Precompute heavy, fully-vectorizable indicators once over the full dataset
        so we don't recompute them on every bar in the backtest loop.
        """
        if self._full_data is None:
            return

        df = self._full_data

        # === Precompute HTF EMA on close prices (matches ema(..., EMA_PERIOD)) ===
        if "close" in df.columns:
            df["htf_ema"] = df["close"].ewm(span=EMA_PERIOD, adjust=False).mean()

        # === Precompute ATR (full series) and its SMA ===
        try:
            high = df["high"]
            low = df["low"]
            close = df["close"]
            tr1 = high - low
            tr2 = (high - close.shift(1)).abs()
            tr3 = (low - close.shift(1)).abs()
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            full_atr = pd.Series(index=df.index, dtype=float)
            if len(true_range) >= ATR_PERIOD:
                full_atr.iloc[ATR_PERIOD - 1] = true_range.iloc[:ATR_PERIOD].mean()
                for i in range(ATR_PERIOD, len(true_range)):
                    prev_atr = full_atr.iloc[i - 1]
                    full_atr.iloc[i] = (prev_atr * (ATR_PERIOD - 1) + true_range.iloc[i]) / ATR_PERIOD
            df["atr"] = full_atr
            df["atr_sma"] = full_atr.rolling(20, min_periods=1).mean()
        except Exception:
            pass

        # === Precompute volume SMA used for volume check ===
        if "volume" in df.columns:
            df["vol_sma"] = df["volume"].rolling(20, min_periods=1).mean()
        self._precompute_htf_resample()

    def _precompute_htf_resample(self) -> None:
        if self._full_data is None or "timestamp" not in self._full_data.columns:
            self._htf_resampled = None
            return
        htf_data = self._full_data.copy()
        ts_series = htf_data["timestamp"]
        if pd.api.types.is_numeric_dtype(ts_series):
            max_val = pd.Series(ts_series).max()
            unit = "ms" if max_val > 10**12 else "s"
            htf_data["timestamp"] = pd.to_datetime(ts_series, unit=unit, utc=True)
        else:
            numeric_ts = pd.to_numeric(ts_series, errors="coerce")
            if numeric_ts.notna().any():
                max_val = numeric_ts.max()
                unit = "ms" if max_val > 10**12 else "s"
                htf_data["timestamp"] = pd.to_datetime(numeric_ts, unit=unit, utc=True)
            else:
                htf_data["timestamp"] = pd.to_datetime(ts_series, utc=True, errors="coerce")
        htf_data = htf_data[htf_data["timestamp"].notna()]
        if htf_data.empty:
            self._htf_resampled = None
            return
        htf_data = htf_data.set_index("timestamp")

        htf_minutes = int(HTF_TF)
        if htf_minutes == 240:
            resample_period = "4h"
        elif htf_minutes == 120:
            resample_period = "2h"
        elif htf_minutes == 60:
            resample_period = "1h"
        elif htf_minutes == 1440:
            resample_period = "1d"
        elif htf_minutes >= 1440:
            resample_period = f"{htf_minutes // 1440}d"
        elif htf_minutes >= 60:
            resample_period = f"{htf_minutes // 60}h"
        else:
            resample_period = f"{htf_minutes}min"

        agg_dict = {"open": "first", "high": "max", "low": "min", "close": "last"}
        if "volume" in htf_data.columns:
            agg_dict["volume"] = "sum"

        self._htf_resampled = (
            htf_data.resample(
                resample_period, label="right", closed="right", origin="epoch"
            )
            .agg(agg_dict)
            .dropna()
            .reset_index()
        )

    def _parse_start_timestamp(self, ts) -> datetime | None:
        """
        Accepts a timestamp as:
        - datetime (naive or tz-aware)
        - numeric seconds since epoch
        - numeric milliseconds since epoch (if > 1e12)
        - string parseable by pandas.to_datetime
        Returns a timezone-aware UTC datetime, or None if parsing fails.
        """
        if isinstance(ts, datetime):
            if ts.tzinfo is None:
                return ts.replace(tzinfo=timezone.utc)
            return ts.astimezone(timezone.utc)

        # Try numeric epoch (seconds or ms)
        try:
            val = float(ts)
            if val > 10**12:
                return datetime.fromtimestamp(val / 1000.0, tz=timezone.utc)
            return datetime.fromtimestamp(val, tz=timezone.utc)
        except (TypeError, ValueError):
            pass

        # Fallback to pandas parser
        try:
            return pd.to_datetime(ts, utc=True).to_pydatetime(warn=False)
        except Exception:
            return None

    def _infer_warmup(self) -> int:
        min_bars = max(ATR_PERIOD + 1, 25, EMA_PERIOD + 1)
        return min_bars

    def gather_data(self) -> pd.DataFrame:
        self._full_data = self._load_data()
        self._full_data_1m = self._load_data_1m()
        if self._full_data_1m is not None:
            self._full_data_1m = self._full_data_1m.sort_values("timestamp").reset_index(drop=True)
            print(f"📊 Loaded {len(self._full_data_1m)} 1min bars for TP/SL (matches live)")
        warmup = self._warmup_bars or self._infer_warmup()
        if len(self._full_data) < warmup:
            raise ValueError("Not enough bars for warmup/backtest.")
        # Default cursor after warmup
        self._cursor = warmup

        window = self._get_window_size()
        if window < warmup:
            window = warmup

        # If no explicit start timestamp, just return warmup slice
        if self._start_from_dt is None or "timestamp" not in self._full_data.columns:
            start_idx = warmup
            start = max(0, start_idx - window)
            return self._full_data.iloc[start:start_idx].copy()

        # Find first bar at or after requested start timestamp
        start_idx = None
        for idx in range(warmup, len(self._full_data)):
            row = self._full_data.iloc[[idx]]
            bar_dt = self._extract_bar_time(row)
            if bar_dt is not None and bar_dt >= self._start_from_dt:
                start_idx = idx
                break

        # If a later start index is found, extend initial data to that point
        if start_idx is not None and start_idx > warmup:
            self._cursor = start_idx
            start = max(0, start_idx - window)
            return self._full_data.iloc[start:start_idx].copy()

        # Fallback: start from warmup as usual
        start_idx = warmup
        start = max(0, start_idx - window)
        return self._full_data.iloc[start:start_idx].copy()

    def _get_window_size(self) -> int:
        min_window = max(ATR_PERIOD + 20, EMA_PERIOD + 51, 150)
        min_window = max(min_window, 25, 21 + 4)
        if BACKTEST_WINDOW_BARS is None:
            return min_window
        return max(int(BACKTEST_WINDOW_BARS), min_window)

    def fetch_new_data(self) -> None:
        if self._cursor >= len(self._full_data):
            return
        new_row = self._full_data.iloc[[self._cursor]]
        self.data = pd.concat([self.data, new_row], ignore_index=True)
        window = self._get_window_size()
        if len(self.data) > window:
            self.data = self.data.iloc[-window:].reset_index(drop=True)
        self._cursor += 1

    def fetch_htf_data(self) -> pd.DataFrame:
        if self.data is None or len(self.data) == 0:
            return pd.DataFrame()
        current_timestamp = self._extract_bar_time(self.data.iloc[[-1]])
        if current_timestamp is None:
            return pd.DataFrame()
        bars_needed = max(101, EMA_PERIOD + 51)

        if self._htf_resampled is None:
            htf_resampled = self._resample_htf_data(current_timestamp)
        else:
            htf_resampled = self._htf_resampled
            htf_resampled = htf_resampled[htf_resampled["timestamp"] <= current_timestamp]
        if htf_resampled.empty:
            return pd.DataFrame()
        if len(htf_resampled) < EMA_PERIOD:
            return pd.DataFrame()
        start_idx = max(0, len(htf_resampled) - bars_needed)
        return htf_resampled.iloc[start_idx:].copy()

    def _get_current_atr_value(self):
        if (
            self._full_data is not None
            and "atr" in self._full_data.columns
            and self._current_index is not None
        ):
            val = self._full_data["atr"].iloc[self._current_index]
            if pd.notna(val):
                return float(val)
        atr_series = get_atr(self.data, ATR_PERIOD)
        return atr_series.iloc[-1] if len(atr_series) > 0 else None

    def _calc_order_size_from_margin(self, entry_price: float) -> float:
        if entry_price <= 0:
            return 0.0
        notional = float(MARGIN_PER_TRADE_USD) * float(LEVERAGE)
        return notional / float(entry_price)

    def _update_trend_indicators(self):
        bars = self.fetch_htf_data()
        htfEMA = ema(bars, EMA_PERIOD)

        if htfEMA is None:
            self.isBullishHTF = False
            self.isBearishHTF = False
        else:
            self.isBullishHTF = self.cur_close > htfEMA
            self.isBearishHTF = self.cur_close < htfEMA

        atr_val = get_atr(self.data, ATR_PERIOD)
        atr_val = atr_val.iloc[-1] if (atr_val is not None and len(atr_val) > 0) else None
        atr_sma = None
        if atr_val is not None:
            atr_series = get_atr(self.data, ATR_PERIOD)
            if atr_series is not None and len(atr_series) > 0:
                atr_sma = sma(atr_series, min(20, len(atr_series)))
        atrOK = (
            atr_val > atr_sma
            if (atr_val is not None and atr_sma is not None and pd.notna(atr_val) and pd.notna(atr_sma))
            else False
        )

        if USE_VOLUME_CHECK:
            vol_sma = sma(self.data["volume"], 20)
            volOK = self.cur_volume > vol_sma * VOLUME_MULTIPLIER if vol_sma is not None else False
            self.marketOK = volOK and atrOK
        else:
            self.marketOK = atrOK

        self.lastBullFvg = (
            self.data["high"].iloc[-3] < self.data["low"].iloc[-1] and not self.lastBullFvg
        )
        self.lastBearFvg = (
            self.data["low"].iloc[-3] > self.data["high"].iloc[-1] and not self.lastBearFvg
        )

    def entry_logic(self):
        if len(self.fvg_zones) == 0:
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

        atr = self._get_current_atr_value()

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
                    if USE_MARGIN_PER_TRADE:
                        total_size = self._calc_order_size_from_margin(entry_price)
                        per_order_size = total_size / float(self._split_order_count)
                    else:
                        per_order_size = EACH_TRADE_SIZE
                    for idx in range(self._split_order_count):
                        active_order = self.Order(
                            entry_atr=entryAtr,
                            side="BUY",
                            entry_price=entry_price,
                            take_profit=tp,
                            stop_loss=stop_loss,
                            trailing_stop_loss=trail_stop,
                            order_size=per_order_size,
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
                            active_order.opened_eval_ts_ms = self._get_opened_eval_ts_ms_for_current_bar()
                            self.active_orders.append(active_order)
                            any_success = True
                            self.pyramiding.on_position_opened(active_order, self)
                else:
                    if USE_MARGIN_PER_TRADE:
                        order_size = self._calc_order_size_from_margin(entry_price)
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
                        active_order.opened_eval_ts_ms = self._get_opened_eval_ts_ms_for_current_bar()
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
                    if USE_MARGIN_PER_TRADE:
                        total_size = self._calc_order_size_from_margin(entry_price)
                        per_order_size = total_size / float(self._split_order_count)
                    else:
                        per_order_size = EACH_TRADE_SIZE
                    for idx in range(self._split_order_count):
                        active_order = self.Order(
                            entry_atr=entryAtr,
                            side="SELL",
                            entry_price=entry_price,
                            take_profit=tp,
                            trailing_stop_loss=trail_stop,
                            stop_loss=stop_loss,
                            order_size=per_order_size,
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
                            active_order.opened_eval_ts_ms = self._get_opened_eval_ts_ms_for_current_bar()
                            self.active_orders.append(active_order)
                            any_success = True
                            self.pyramiding.on_position_opened(active_order, self)
                else:
                    if USE_MARGIN_PER_TRADE:
                        order_size = self._calc_order_size_from_margin(entry_price)
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
                        success = True
                    if success:
                        active_order.opened_eval_ts_ms = self._get_opened_eval_ts_ms_for_current_bar()
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

    def check_daily_trade_limit(self):
        if self._current_dt is None:
            return True
        today = self._current_dt.date()
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
            return max(0.001, round(lot_size, 3))
        return ORDER_SIZE

    def calculate_trade_fee(
        self,
        order: BacktestOrder,
        exit_price: float | None = None,
        exit_timestamp: datetime | None = None,
    ) -> float:
        try:
            entry_price = float(
                order.avg_entry_price if order.avg_entry_price is not None else order.entry_price
            )
            size = float(order.order_size or 0.0)
        except (TypeError, ValueError):
            return 0.0
        if size <= 0 or entry_price <= 0:
            return 0.0
        notional = abs(entry_price * size)
        return order._calc_fee(notional)

    def calculate_trade_pnl(
        self,
        order: BacktestOrder,
        exit_price: float | None,
        exit_timestamp: datetime | None = None,
    ) -> float:
        entry_price = order.avg_entry_price if order.avg_entry_price is not None else order.entry_price
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

        if self.tick_size is not None and self.tick_value is not None and float(self.tick_size) != 0:
            tick_size = float(self.tick_size)
            tick_value = float(self.tick_value)
            price_move = exit_price - entry_price
            ticks = price_move / tick_size
            if order.side == "SELL":
                ticks = -ticks
            gross = ticks * tick_value * size
            fee = self.calculate_trade_fee(order, exit_price, exit_timestamp)
            return gross - fee

        price_delta = exit_price - entry_price
        direction = 1 if order.side == "BUY" else -1
        fee = self.calculate_trade_fee(order, exit_price, exit_timestamp)
        return price_delta * direction * size - fee

    def subscribe_to_price_updates(self):
        return

    def start_bar_iterations(self):
        return

    def _extract_bar_time(self, row: pd.DataFrame) -> datetime | None:
        if "timestamp" not in row.columns:
            return None
        ts = row["timestamp"].iloc[-1]
        if pd.isna(ts):
            return None
        try:
            ts = float(ts)
            if ts > 10**12:
                return datetime.fromtimestamp(ts / 1000.0, tz=timezone.utc)
            return datetime.fromtimestamp(ts, tz=timezone.utc)
        except Exception:
            try:
                return pd.to_datetime(ts, utc=True).to_pydatetime(warn=False)
            except Exception:
                return None

    def _record_trade(self, order: BacktestOrder):
        row = {
            "side": order.side,
            "entry_price": order.entry_price,
            "exit_price": order.exit_price,
            "entry_time": order.entry_time,
            "exit_time": order.exit_time,
            "order_size": order.order_size,
            "pnl": order.pnl,
            "equity": self.account_balance,
            "group_id": getattr(order, "group_id", None),
            "margin_per_trade": order.margin_used,
            "lot_size": order.order_size,
            "total_fees": order.fee_paid,
        }
        self.trades.append(row)
        self._append_trade_csv(row)
        # Persist full strategy/backtest state after each completed trade
        try:
            self.save_data()
        except Exception:
            # Saving should not break the backtest loop; ignore persistence errors
            pass

    def _append_trade_csv(self, row: dict):
        # Ensure output directory exists
        os.makedirs(os.path.dirname(self.trades_csv_path), exist_ok=True)
        fieldnames = [
            "side",
            "entry_price",
            "exit_price",
            "entry_time",
            "exit_time",
            "order_size",
            "pnl",
            "equity",
            "group_id",
            "margin_per_trade",
            "lot_size",
            "total_fees",
        ]
        if TRADE_CSV_WRITE_MODE == "prepend":
            temp_path = f"{self.trades_csv_path}.tmp"
            with open(temp_path, "w", newline="", encoding="utf-8") as out_f:
                writer = csv.DictWriter(out_f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow(row)
                if os.path.exists(self.trades_csv_path):
                    with open(self.trades_csv_path, "r", newline="", encoding="utf-8") as in_f:
                        reader = csv.DictReader(in_f)
                        for existing in reader:
                            writer.writerow(existing)
            os.replace(temp_path, self.trades_csv_path)
            return
        write_header = not os.path.exists(self.trades_csv_path)
        with open(self.trades_csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def _close_open_order_at_end(self):
        if not self.active_orders:
            return
        for order in list(self.active_orders):
            self.active_orders.pop(0)
            order._last_price = float(self.cur_close)
            order._last_timestamp = self._current_dt
            order.close_order()
            self.pyramiding.on_position_closed(order, self)
            if order.pnl is not None:
                self.account_balance += order.pnl
            self._record_trade(order)

    def _process_order_closes(
        self,
        current_high: float,
        current_low: float,
        current_close: float,
        current_dt: datetime | None,
        use_1m: bool = False,
        eval_bucket_ts_ms: int | None = None,
    ) -> None:
        """Check TP/SL on each order; close if hit. Respects opened_eval_ts_ms when use_1m."""
        self._closed_any_this_bar = False
        partial_close_map = self._get_partial_close_targets(current_close)
        remaining = []
        for order in list(self.active_orders):
            if use_1m and eval_bucket_ts_ms is not None:
                opened_ms = getattr(order, "opened_eval_ts_ms", None)
                if opened_ms is not None and opened_ms == eval_bucket_ts_ms:
                    remaining.append(order)
                    continue
            closed = False
            if order.side == "BUY":
                if order.trailing_stop_loss is not None and current_low <= order.trailing_stop_loss:
                    order._last_price = float(order.trailing_stop_loss)
                    order._last_timestamp = current_dt
                    order.exit_reason = "trailing_stop"
                    order.close_order()
                    closed = True
                elif order.stop_loss is not None and current_low <= order.stop_loss:
                    order._last_price = float(order.stop_loss)
                    order._last_timestamp = current_dt
                    order.exit_reason = "stop_loss"
                    order.close_order()
                    closed = True
                elif order.take_profit is not None and current_high >= order.take_profit:
                    order._last_price = float(order.take_profit)
                    order._last_timestamp = current_dt
                    order.exit_reason = "take_profit"
                    order.close_order()
                    closed = True
            else:
                if order.trailing_stop_loss is not None and current_high >= order.trailing_stop_loss:
                    order._last_price = float(order.trailing_stop_loss)
                    order._last_timestamp = current_dt
                    order.exit_reason = "trailing_stop"
                    order.close_order()
                    closed = True
                elif order.stop_loss is not None and current_high >= order.stop_loss:
                    order._last_price = float(order.stop_loss)
                    order._last_timestamp = current_dt
                    order.exit_reason = "stop_loss"
                    order.close_order()
                    closed = True
                elif order.take_profit is not None and current_low <= order.take_profit:
                    order._last_price = float(order.take_profit)
                    order._last_timestamp = current_dt
                    order.exit_reason = "take_profit"
                    order.close_order()
                    closed = True
            if not closed and order in partial_close_map:
                order._last_price = current_close
                order._last_timestamp = current_dt
                order.exit_reason = partial_close_map[order]
                order.close_order()
                closed = True
            if not closed:
                closed = order.check_close_conditions(
                    current_price=current_close,
                    current_high=current_high,
                    current_low=current_low,
                    last_long=order.side == "BUY",
                    last_short=order.side == "SELL",
                    isBOS=self.isBOS,
                    isCHOCH=self.isCHOCH,
                    timestamp=current_dt,
                )
            if closed:
                self._closed_any_this_bar = True
                self.pyramiding.on_position_closed(order, self)
                if order.pnl is not None:
                    self.account_balance += order.pnl
                self._record_trade(order)
            else:
                remaining.append(order)
        self.active_orders = remaining

    def run(self):
        total_bars = len(self._full_data)
        while self._cursor < total_bars:
            if self.account_balance <= 0:
                self._stopped = True
                break
            new_row = self._full_data.iloc[[self._cursor]]
            self.data = pd.concat([self.data, new_row], ignore_index=True)
            self.cur_close = float(new_row["close"].iloc[-1])
            self.cur_volume = float(new_row["volume"].iloc[-1]) if "volume" in new_row.columns else 0.0
            self._current_dt = self._extract_bar_time(new_row)
            self._current_index = self._cursor

            if self._check_max_drawdown(self._current_dt, float(self.cur_close)):
                self._cursor += 1
                continue

            self.update_indicators()
            prev_zone_count = len(self.fvg_zones)
            self.add_fvg_zones()
            created_new_zone = len(self.fvg_zones) > prev_zone_count


            if len(self.active_orders) > 0:
                self._closed_any_this_bar = False
                ts_val = new_row["timestamp"].iloc[-1]
                try:
                    bar_ts_ms = int(float(ts_val))
                    if bar_ts_ms < 10**12:
                        bar_ts_ms = int(bar_ts_ms * 1000)
                except (TypeError, ValueError):
                    bar_ts = self._extract_bar_time(new_row)
                    bar_ts_ms = int(bar_ts.timestamp() * 1000) if bar_ts else 0
                one_min_bars = self._get_1m_bars_for_15m_bar(bar_ts_ms)

                if one_min_bars.empty:
                    current_high = float(new_row["high"].iloc[-1])
                    current_low = float(new_row["low"].iloc[-1])
                    self._process_order_closes(
                        current_high, current_low, self.cur_close, self._current_dt,
                        use_1m=False,
                    )
                else:
                    for _, row_1m in one_min_bars.iterrows():
                        if not self.active_orders:
                            break
                        ts_val = row_1m["timestamp"]
                        try:
                            ts_val = int(float(ts_val))
                        except (TypeError, ValueError):
                            ts_val = 0
                        eval_bucket_ms = ts_val if ts_val > 10**12 else ts_val * 1000
                        h_1m = float(row_1m["high"]) if "high" in row_1m else self.cur_close
                        l_1m = float(row_1m["low"]) if "low" in row_1m else self.cur_close
                        c_1m = float(row_1m["close"]) if "close" in row_1m else self.cur_close
                        dt_1m = self._parse_start_timestamp(ts_val) if ts_val else self._current_dt
                        self.update_stops(
                            current_high=h_1m,
                            current_low=l_1m,
                            high_changed=True,
                            low_changed=True,
                            eval_bucket_ts_ms=eval_bucket_ms,
                        )
                        self._process_order_closes(
                            h_1m, l_1m, c_1m, dt_1m,
                            use_1m=True,
                            eval_bucket_ts_ms=eval_bucket_ms,
                        )
                current_high = float(new_row["high"].iloc[-1])
                current_low = float(new_row["low"].iloc[-1])
                if self.active_orders:
                    self._apply_pyramiding_add_on(
                        self.cur_close,
                        current_high=current_high,
                        current_low=current_low,
                    )
                if getattr(self, "_closed_any_this_bar", False):
                    self.lastPositionWasLong = any(o.side == "BUY" for o in self.active_orders)
                    self.lastPositionWasShort = any(o.side == "SELL" for o in self.active_orders)
                    self._cleanup_partial_groups()
                if self.account_balance <= 0:
                    self._stopped = True
                    break
            else:
                if self.account_balance <= 0:
                    self._stopped = True
                    break
                if getattr(self, "require_intrabar_entry", False) and created_new_zone:
                    self._cursor += 1
                    continue
                last_index = self.data.index[-1]
                orig_high = self.data.at[last_index, "high"]
                orig_low = self.data.at[last_index, "low"]
                if not ALLOW_INTRACANDLE_ENTRY:
                    self.data.at[last_index, "high"] = self.cur_close
                    self.data.at[last_index, "low"] = self.cur_close
                try:
                    if getattr(self, "require_intrabar_entry", False):
                        self._intrabar_mode = True
                    self.entry_logic()
                finally:
                    if getattr(self, "require_intrabar_entry", False):
                        self._intrabar_mode = False
                    if not ALLOW_INTRACANDLE_ENTRY:
                        self.data.at[last_index, "high"] = orig_high
                        self.data.at[last_index, "low"] = orig_low
                if self.active_orders:
                    for active_order in self.active_orders:
                        if active_order.entry_time is None:
                            active_order.entry_time = self._current_dt

            self._cursor += 1

        self._close_open_order_at_end()
        return self.trades


if __name__ == "__main__":
    data_path = DATA_CSV_PATH
    backtest = FVG_Backtest(
        asset=ASSET,
        timeframe=TIMEFRAME,
        initial_balance=INITIAL_BALANCE,
        data_path=data_path,
        start_timestamp=START_TIMESTAMP,
        pyramiding_mode=PYRAMIDING_MODE,
        data_path_1m=DATA_1M_CSV_PATH,
    )
    trades = backtest.run()
    print(f"✅ Backtest finished. Trades: {len(trades)}")
    try:
        from FVG_projectX_bot.backtest.evaluate_backtest import evaluate_backtest

        evaluate_backtest(
            trades_csv=Path(backtest.trades_csv_path),
            price_csv=Path(backtest.data_path),
            start_equity=INITIAL_BALANCE,
        )
    except Exception as exc:
        print(f"⚠️ Backtest evaluation failed: {exc}")
