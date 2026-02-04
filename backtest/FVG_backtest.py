import os
import csv
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

CURRENT_DIR = Path(__file__).parent

from FVG_projectX_bot.FVG_strategy import (
    FVG_Strategy,
    FVG_Order,
    ALLOW_INTRACANDLE_ENTRY,
    ATR_PERIOD,
    EMA_PERIOD,
    MAX_DAILY_TRADES,
    USE_FIXED_LOT,
    FIXED_LOT,
    RISK_PERCENT,
    ORDER_SIZE,
)


class BacktestOrder(FVG_Order):
    is_open: bool
    entry_time: datetime | None
    exit_time: datetime | None
    exit_price: float | None
    exit_reason: str | None
    pnl: float | None

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
        return {"success": True}

    def close_order(self):
        self.is_open = False
        if self._last_price is not None:
            self.exit_price = float(self._last_price)
        if self._last_timestamp is not None:
            self.exit_time = self._last_timestamp
        if self.exit_price is not None:
            price_delta = self.exit_price - self.entry_price
            direction = 1 if self.side == "BUY" else -1
            self.pnl = price_delta * direction * float(self.order_size or 0.0)
        return {"success": True}

    def check_close_conditions(self, log=print, **kwargs) -> bool:
        self._last_price = kwargs.get("current_price")
        self._last_timestamp = kwargs.get("timestamp")
        return super().check_close_conditions(log=log, **kwargs)


class FVG_Backtest(FVG_Strategy):
    Order = BacktestOrder

    def __init__(
        self,
        data_path: str,
        asset: str = "BTCUSDT",
        timeframe: str = "15m",
        initial_balance: float = 10000.0,
        warmup_bars: int | None = None,
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
        super().__init__()

    def api_order_kwargs(self) -> dict:
        return {}

    def _load_data(self) -> pd.DataFrame:
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        data = pd.read_csv(self.data_path)
        data.columns = [str(col).lower() for col in data.columns]
        required = {"open", "high", "low", "close"}
        missing = required - set(data.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")
        if "volume" not in data.columns:
            data["volume"] = 0.0
        if "timestamp" in data.columns:
            data = data.sort_values("timestamp").reset_index(drop=True)
        return data

    def _infer_warmup(self) -> int:
        min_bars = max(ATR_PERIOD + 1, 25, EMA_PERIOD + 1)
        return min_bars

    def gather_data(self) -> pd.DataFrame:
        self._full_data = self._load_data()
        warmup = self._warmup_bars or self._infer_warmup()
        if len(self._full_data) < warmup:
            raise ValueError("Not enough bars for warmup/backtest.")
        self._cursor = warmup
        return self._full_data.iloc[:warmup].copy()

    def fetch_new_data(self) -> None:
        if self._cursor >= len(self._full_data):
            return
        new_row = self._full_data.iloc[[self._cursor]]
        self.data = pd.concat([self.data, new_row], ignore_index=True)
        self._cursor += 1

    def fetch_htf_data(self) -> pd.DataFrame:
        return self.data

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
        }
        self.trades.append(row)
        self._append_trade_csv(row)

    def _append_trade_csv(self, row: dict):
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
                ],
            )
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def _close_open_order_at_end(self):
        if not self.active_orders:
            return
        order: BacktestOrder = self.active_orders.pop(0)
        order._last_price = float(self.cur_close)
        order._last_timestamp = self._current_dt
        order.close_order()
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

            self.update_indicators()
            self.add_fvg_zones()

            if len(self.active_orders) > 0:
                self.update_stops()
                closed = self.active_orders[0].check_close_conditions(
                    current_price=self.cur_close,
                    last_long=self.lastPositionWasLong,
                    last_short=self.lastPositionWasShort,
                    isBOS=self.isBOS,
                    isCHOCH=self.isCHOCH,
                    timestamp=self._current_dt,
                )
                if closed:
                    closed_order = self.active_orders.pop(0)
                    self.lastPositionWasLong = False
                    self.lastPositionWasShort = False
                    if closed_order.pnl is not None:
                        self.account_balance += closed_order.pnl
                    self._record_trade(closed_order)
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
                    active_order = self.active_orders[0]
                    active_order.entry_time = self._current_dt

            self._cursor += 1

        self._close_open_order_at_end()
        return self.trades


if __name__ == "__main__":
    data_path = os.path.join(CURRENT_DIR, "data", "BTCUSDT_PERP_15m.csv")
    backtest = FVG_Backtest(data_path=data_path)
    trades = backtest.run()
    print(f"✅ Backtest finished. Trades: {len(trades)}")
