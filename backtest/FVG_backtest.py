import os
import csv
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

from FVG_projectX_bot.FVG_strategy import *
from FVG_projectX_bot.helping_functions.indicators import get_atr, ema
from FVG_projectX_bot.helping_functions.pyramiding import MaxOrdersPolicy


PARENT_DIR = Path(__file__).parents[1]
CURRENT_DIR = Path(__file__).parent / "BTC_BACKTEST"

# ==================== USER CONFIG ====================
ASSET = "BTC"
TIMEFRAME = "15m"
INITIAL_BALANCE = 50.0
DATA_CSV_PATH = str(PARENT_DIR / "backtest" / "data" / "BTCUSDT_PERP_15m.csv")
START_TIMESTAMP = "1755528300000"

# Pyramiding mode: "none", "client_atr", or "max_orders"
PYRAMIDING_MODE = "client_atr" # remake to str enum
MAX_PYRAMID_ORDERS = 3

# Backtest data window (bars)
BACKTEST_WINDOW_BARS = None

# Contract / fee inputs
USE_CONTRACTS_CSV = False
CONTRACTS_CSV_PATH = str(PARENT_DIR.parent / "contracts.csv")
USE_ROUND_TURN_FEE = False
ROUND_TURN_FEE_USD = 3.5


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

    def place_order(self):
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
            entry_price = self.avg_entry_price if self.avg_entry_price is not None else self.entry_price
            price_delta = self.exit_price - entry_price
            direction = 1 if self.side == "BUY" else -1
            self.pnl = price_delta * direction * float(self.order_size or 0.0) - float(self.order_size or 0.0) * self._last_price * 0.0001
        group_id = getattr(self, "group_id", None)
        group_seq = getattr(self, "group_seq", None)
        print(
            "🧾 CLOSE "
            f"side={self.side} size={self.order_size} entry={self.entry_price} "
            f"exit={self.exit_price} pnl={self.pnl} reason={self.exit_reason} "
            f"group_id={group_id} group_seq={group_seq}"
        )
        self._pnl_printed = True
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
    ):
        self.asset = asset
        self.timeframe = timeframe
        self.account_balance = float(initial_balance)
        self.data_path = data_path
        self.metadata_filename = os.path.join(CURRENT_DIR, "backtest_metadata.json")
        self.csv_filename = os.path.join(CURRENT_DIR, "backtest_data.csv")
        self._warmup_bars = warmup_bars
        self._full_data = None
        self._cursor = 0
        self._current_dt = None
        self.trades = []
        self._stopped = False
        self.trades_csv_path = os.path.join(CURRENT_DIR, "backtest_trades.csv")
        self._start_from_dt = self._parse_start_timestamp(start_timestamp) if start_timestamp is not None else None
        self.tick_size = None
        self.tick_value = None
        self.round_turn_fee_usd = ROUND_TURN_FEE_USD if USE_ROUND_TURN_FEE else None
        if USE_CONTRACTS_CSV:
            self._load_contract_info()
        self._configure_pyramiding(pyramiding_mode)
        super().__init__()

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

    def _load_data(self) -> pd.DataFrame:
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        sep = ","
        try:
            with open(self.data_path, "r", encoding="utf-8") as f:
                header_line = f.readline()
            if "\t" in header_line and "," not in header_line:
                sep = "\t"
        except Exception:
            sep = ","
        data = pd.read_csv(self.data_path, sep=sep)
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

        # === Precompute ATR and its SMA ===
        try:
            atr_series = get_atr(df, ATR_PERIOD)
        except Exception:
            atr_series = None

        if atr_series is not None and len(atr_series) > 0:
            # Align ATR series back to full DataFrame index
            full_atr = pd.Series(index=df.index, dtype=float)
            start_idx = ATR_PERIOD - 1
            full_atr.iloc[start_idx:] = atr_series.values
            df["atr"] = full_atr
            df["atr_sma"] = full_atr.rolling(20, min_periods=1).mean()

        # === Precompute volume SMA used for volume check ===
        if "volume" in df.columns:
            df["vol_sma"] = df["volume"].rolling(20, min_periods=1).mean()

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
            return pd.to_datetime(ts, utc=True).to_pydatetime()
        except Exception:
            return None

    def _infer_warmup(self) -> int:
        min_bars = max(ATR_PERIOD + 1, 25, EMA_PERIOD + 1)
        return min_bars

    def gather_data(self) -> pd.DataFrame:
        self._full_data = self._load_data()
        # Precompute heavy indicators once, to speed up per-bar loop
        self._precompute_indicators()
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

        htf_resampled = self._resample_htf_data(current_timestamp)
        if htf_resampled.empty:
            return pd.DataFrame()
        if len(htf_resampled) < EMA_PERIOD:
            return pd.DataFrame()
        start_idx = max(0, len(htf_resampled) - bars_needed)
        return htf_resampled.iloc[start_idx:].copy()

    def check_daily_trade_limit(self):
        if self._current_dt is None:
            return True
        today = self._current_dt.date()
        if self.last_trade_date != str(today):
            self.daily_trades_count = 0
            self.last_trade_date = str(today)
        return self.daily_trades_count < MAX_DAILY_TRADES

    def _update_trend_indicators(self):
        bars = self.fetch_htf_data()
        htfEMA = ema(bars, EMA_PERIOD)
        last_ts_raw = None
        current_ts = None
        full_ts_min = None
        full_ts_max = None
        if self.data is not None and len(self.data) > 0 and "timestamp" in self.data.columns:
            last_ts_raw = self.data["timestamp"].iloc[-1]
            current_ts = self._extract_bar_time(self.data.iloc[[-1]])
        if self._full_data is not None and "timestamp" in self._full_data.columns:
            full_ts_min = self._full_data["timestamp"].iloc[0]
            full_ts_max = self._full_data["timestamp"].iloc[-1]

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

        self.lastBullFvg = (
            self.data["high"].iloc[-3] < self.data["low"].iloc[-1] and not self.lastBullFvg
        )
        self.lastBearFvg = (
            self.data["low"].iloc[-3] > self.data["high"].iloc[-1] and not self.lastBearFvg
        )

    def calculate_order_size(self, atr, sl_mult):
        if USE_FIXED_LOT:
            return FIXED_LOT
        risk_amount = self.account_balance * (RISK_PERCENT / 100)
        stop_distance = atr * sl_mult
        if stop_distance > 0:
            lot_size = risk_amount / stop_distance
            return max(0.001, round(lot_size, 3))
        return ORDER_SIZE

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
                return pd.to_datetime(ts, utc=True).to_pydatetime()
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
        write_header = not os.path.exists(self.trades_csv_path)
        with open(self.trades_csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "side",
                    "entry_price",
                    "exit_price",
                    "entry_time",
                    "exit_time",
                    "order_size",
                    "pnl",
                    "equity",
                    "group_id",
                ],
            )
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

            if self._check_max_drawdown(self._current_dt, float(self.cur_close)):
                self._cursor += 1
                continue

            self.update_indicators()
            self.add_fvg_zones()


            if len(self.active_orders) > 0:
                self.update_stops()
                current_high = float(new_row["high"].iloc[-1])
                current_low = float(new_row["low"].iloc[-1])
                partial_close_map = self._get_partial_close_targets(self.cur_close)
                remaining = []
                closed_any = False
                for order in list(self.active_orders):
                    intrabar_closed = False
                    if order.side == "BUY":
                        if order.trailing_stop_loss is not None and current_low <= order.trailing_stop_loss:
                            order._last_price = float(order.trailing_stop_loss)
                            order._last_timestamp = self._current_dt
                            order.exit_reason = "trailing_stop"
                            order.close_order()
                            intrabar_closed = True
                        elif order.stop_loss is not None and current_low <= order.stop_loss:
                            order._last_price = float(order.stop_loss)
                            order._last_timestamp = self._current_dt
                            order.exit_reason = "stop_loss"
                            order.close_order()
                            intrabar_closed = True
                        elif order.take_profit is not None and current_high >= order.take_profit:
                            order._last_price = float(order.take_profit)
                            order._last_timestamp = self._current_dt
                            order.exit_reason = "take_profit"
                            order.close_order()
                            intrabar_closed = True
                    else:
                        if order.trailing_stop_loss is not None and current_high >= order.trailing_stop_loss:
                            order._last_price = float(order.trailing_stop_loss)
                            order._last_timestamp = self._current_dt
                            order.exit_reason = "trailing_stop"
                            order.close_order()
                            intrabar_closed = True
                        elif order.stop_loss is not None and current_high >= order.stop_loss:
                            order._last_price = float(order.stop_loss)
                            order._last_timestamp = self._current_dt
                            order.exit_reason = "stop_loss"
                            order.close_order()
                            intrabar_closed = True
                        elif order.take_profit is not None and current_low <= order.take_profit:
                            order._last_price = float(order.take_profit)
                            order._last_timestamp = self._current_dt
                            order.exit_reason = "take_profit"
                            order.close_order()
                            intrabar_closed = True

                    if intrabar_closed:
                        closed = True
                    elif order in partial_close_map:
                        if hasattr(order, "_last_price"):
                            order._last_price = self.cur_close
                        if hasattr(order, "_last_timestamp"):
                            order._last_timestamp = self._current_dt
                        if hasattr(order, "exit_reason"):
                            order.exit_reason = partial_close_map[order]
                        order.close_order()
                        closed = True
                    else:
                        closed = order.check_close_conditions(
                            current_price=self.cur_close,
                            last_long=order.side == "BUY",
                            last_short=order.side == "SELL",
                            isBOS=self.isBOS,
                            isCHOCH=self.isCHOCH,
                            timestamp=self._current_dt,
                        )
                    if closed:
                        closed_any = True
                        self.pyramiding.on_position_closed(order, self)
                        if order.pnl is not None:
                            self.account_balance += order.pnl
                        self._record_trade(order)
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
                if self.account_balance <= 0:
                    self._stopped = True
                    break
            else:
                if self.account_balance <= 0:
                    self._stopped = True
                    break
                last_index = self.data.index[-1]
                orig_high = self.data.at[last_index, "high"]
                orig_low = self.data.at[last_index, "low"]
                if not ALLOW_INTRACANDLE_ENTRY:
                    self.data.at[last_index, "high"] = self.cur_close
                    self.data.at[last_index, "low"] = self.cur_close
                try:
                    self.entry_logic()
                finally:
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
        pyramiding_mode=PYRAMIDING_MODE,
    )
    trades = backtest.run()
    print(f"✅ Backtest finished. Trades: {len(trades)}")
