import pandas as pd
import threading

from abc import abstractmethod
from datetime import datetime

from FVG_projectX_bot.helping_functions.indicators import get_atr
from FVG_projectX_bot.helping_functions.indicators import sma
from FVG_projectX_bot.helping_functions.indicators import ema
from FVG_projectX_bot.helping_functions.indicators import crossover
from FVG_projectX_bot.helping_functions.indicators import crossunder

from FVG_projectX_bot.helping_functions.api_functions import sleep_until_next_boundary
from FVG_projectX_bot.helping_functions.api_functions import TIMEFRAME_SECONDS


from ..strategyTemplate import Strategy, Order


# ==================== CONFIGURATION PARAMETERS ====================
# Display Settings
FVG_HISTORY_NBR = 14              # Number of FVGs to work with
MIN_FVG_POWER_PCT = 0.01          # Min FVG Power % (formerly MinFVGPowerPct)

# Timeframe and Trend Settings
HTF_TF = "240"                     # HTF Bias (4H) - PERIOD_H4
EMA_PERIOD = 100                    # EMA Period for trend detection
VOLUME_MULTIPLIER = 1.25
USE_VOLUME_CHECK = True            # If False, volume check is skipped in marketOK calculation
VOLUME_DATA_START_TIMESTAMP = 1755464400000  # Timestamp where reliable volume data starts (ms)
START_FROM_VOLUME_TIMESTAMP = False  # None = auto (True if USE_VOLUME_CHECK, False otherwise). Set to True/False to override

# ATR and Risk Management
ATR_PERIOD = 22                    # ATR Period (min 1)
SL_MULTIPLIER = 6               # SL ATR Multiplier (formerly SL_ATR_Mult)
TP_MULTIPLIER = 1                # TP ATR Multiplier (formerly TP_ATR_Mult)

# Trailing Stop Settings
USE_TRAILING = True                # use trailing stop (formerly UseTrailing)
TRAIL_OFFSET_MULT = 10            # Trailing Offset ATR Multiplier (formerly TrailATRMult)

# Position Management
HOLD_UNTIL_OPPOSITE = True         # Hold Until Opposite BOS/CHoCH

# Lot Size and Risk Settings
USE_FIXED_LOT = True        # Use fixed lot size (formerly UseFixedLot)
FIXED_LOT = 1                   # Fixed lot size (formerly FixedLot)
RISK_PERCENT = 1.0                 # Risk percentage per trade (formerly RiskPercent)
ORDER_SIZE = 1                     # Default order size (overridden by risk calculation if not USE_FIXED_LOT)

# Daily Trading Limits
MAX_DAILY_TRADES = 3

ALLOW_INTRACANDLE_ENTRY = True

def quiet_log(msg):
    pass


class FVG_Order(Order):
    entry_atr: float

    def __init__(
        self, entry_atr, side, entry_price, take_profit, stop_loss, 
        trailing_stop_loss, order_size, use_trailing=USE_TRAILING
        ):
        super().__init__(
            side, entry_price, order_size, take_profit, stop_loss, 
            trailing_stop_loss, use_trailing
        )

        self.entry_atr = entry_atr


    def check_close_conditions(self, log=print, **kwargs) -> bool:
        current_price = kwargs["current_price"]
        last_long = kwargs["last_long"]
        last_short = kwargs["last_short"] 
        isBOS = kwargs["isBOS"]
        isCHOCH = kwargs["isCHOCH"] 

        if self.side == "BUY":
            if self.trailing_stop_loss is not None and current_price <= self.trailing_stop_loss:
                log(f"🛑 Trailing Stop Loss hit for LONG position at {current_price:.5f}")
                self.close_order()
                return True

            if self.stop_loss is not None and current_price <= self.stop_loss:
                log(f"🛑 Stop Loss hit for LONG position at {current_price:.5f}")
                self.close_order()
                return True

            if current_price >= self.take_profit:
                log(f"🎯 Take Profit hit for LONG position at {current_price:.5f}")
                self.close_order()
                return True


        elif self.side == "SELL":
            if self.trailing_stop_loss is not None and current_price >= self.trailing_stop_loss:
                log(f"🛑 Trailing Stop Loss hit for SHORT position at {current_price:.5f}")
                self.close_order()
                return True

            if self.stop_loss is not None and current_price >= self.stop_loss:
                log(f"🛑 Stop Loss hit for SHORT position at {current_price:.5f}")
                self.close_order()
                return True

            if current_price <= self.take_profit:
                log(f"🎯 Take Profit hit for SHORT position at {current_price:.5f}")
                self.close_order()
                return True


        # === CLOSE ON OPPOSITE BOS/CHoCH ===
        if HOLD_UNTIL_OPPOSITE:
            if last_long and isCHOCH:
                log("🔄 CHoCH detected - Closing LONG position")
                return True

            if last_short and isBOS:
                log("🔄 BOS detected - Closing SHORT position")
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

    daily_trades_count: int
    last_trade_date: str | None

    fvg_zones: list[dict]
    
    def __init__(self, timeframe, metadata_filename, csv_filename):
        self.timeframe = timeframe
        self.metadata_filename = metadata_filename
        self.csv_filename = csv_filename
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

        super().__init__()

        self.cur_close = self.data["close"].iloc[-1]
        self.cur_volume = self.data["volume"].iloc[-1]
        

        self.update_indicators()
        self.add_fvg_zones()


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
            self.cur_close = new_row["close"].iloc[-1]
            print(f"updated price: {self.cur_close}")

            if len(self.active_orders) > 0:
                self.update_stops()
                closed = self.active_orders[0].check_close_conditions(
                    current_price=self.cur_close, 
                    last_long=self.lastPositionWasLong, 
                    last_short=self.lastPositionWasShort,
                    isBOS=self.isBOS,
                    isCHOCH=self.isCHOCH
                    )
                if closed:
                    self.active_orders.pop(0)
                    self.lastPositionWasLong = False
                    self.lastPositionWasShort = False
                    self.save_data()

            elif ALLOW_INTRACANDLE_ENTRY:
                self.entry_logic()

    def add_fvg_zones(self):
        # === FVG ZONE CREATION (equivalent to the box.new blocks) ===
        if self.bullishPowerOK and self.isBullishHTF and self.marketOK:
            # Bullish FVG uses low[1] as top and high[3] as bottom in Pine
            self.fvg_zones.append(
                {
                    "direction": "bull",
                    "top": self.data["low"].iloc[-2],     # low[1]
                    "bottom": self.data["high"].iloc[-4], # high[3]
                    "mitigated": False,
                }
            )
            print(f"🟢 Bullish FVG detected: {self.data['high'].iloc[-4]:.5f} - {self.data['low'].iloc[-2]:.5f}")


        if self.bearishPowerOK and self.isBearishHTF and self.marketOK:
            # Bearish FVG uses low[3] as top and high[1] as bottom in Pine
            self.fvg_zones.append(
                {
                    "direction": "bear",
                    "top": self.data["low"].iloc[-4],     # low[3]
                    "bottom": self.data["high"].iloc[-2], # high[1]
                    "mitigated": False,
                }
            )
            print(f"🔴 Bearish FVG detected: {self.data['high'].iloc[-2]:.5f} - {self.data['low'].iloc[-4]:.5f}")


        # Limit number of stored FVGs, similar to fvgHistoryNbr trimming in Pine
        if len(self.fvg_zones) > FVG_HISTORY_NBR:
            self.fvg_zones = self.fvg_zones[-FVG_HISTORY_NBR:]


    def _update_trend_indicators(self):
        bars = self.fetch_htf_data()
        htfEMA = ema(bars, EMA_PERIOD)

        self.isBullishHTF = self.cur_close > htfEMA
        self.isBearishHTF = self.cur_close < htfEMA

        atrVal = get_atr(self.data, ATR_PERIOD)
        atrOK = atrVal.iloc[-1] > sma(atrVal, 20)
        
        if USE_VOLUME_CHECK:
            volOK = self.cur_volume > sma(self.data["volume"], 20) * VOLUME_MULTIPLIER
            self.marketOK = volOK and atrOK
        else:
            # Skip volume check, only use ATR
            self.marketOK = atrOK


        self.lastBullFvg = self.data["high"].iloc[-4] < self.data["low"].iloc[-2] and not self.lastBullFvg
        self.lastBearFvg = self.data["low"].iloc[-4] > self.data["high"].iloc[-2] and not self.lastBearFvg

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
            and (self.data["low"].iloc[-2] - self.data["high"].iloc[-4]) / gapClose * 100 >= MIN_FVG_POWER_PCT
        )

        self.bearishPowerOK = (
            self.lastBearFvg
            and (self.data["low"].iloc[-4] - self.data["high"].iloc[-2]) / gapClose * 100 >= MIN_FVG_POWER_PCT
        )

        self._calc_BOS_and_CHOCH()


    def entry_logic(self):
        if len(self.fvg_zones) == 0 or len(self.active_orders) > 0:
            return


        if not self.check_daily_trade_limit():
            print(f"⚠️ Daily trade limit reached ({MAX_DAILY_TRADES}). No new trades today.")
            return


        current_high = self.data["high"].iloc[-1]
        current_low = self.data["low"].iloc[-1]

        atr = get_atr(self.data, ATR_PERIOD).iloc[-1]

        for zone in self.fvg_zones[-FVG_HISTORY_NBR:]:
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
                stop_loss = self.cur_close - atr * SL_MULTIPLIER
                if USE_TRAILING:
                    trail_stop = self.cur_close - atr * TRAIL_OFFSET_MULT
                else:
                    trail_stop = None

                tp = self.cur_close + atr * TP_MULTIPLIER
                entryAtr = atr
                lot_size = self.calculate_order_size(atr=atr, sl_mult=SL_MULTIPLIER)

                active_order = self.Order(
                    side="BUY", entry_price=self.cur_close, take_profit=tp,
                    stop_loss=stop_loss, trailing_stop_loss=trail_stop, entry_atr=entryAtr, order_size=lot_size,
                    **self.api_order_kwargs()
                )
                result = active_order.place_order()

                self.active_orders.append(active_order)
                
                if result['success']:
                    zone["mitigated"] = True
                    self.lastPositionWasLong = True
                    self.lastPositionWasShort = False
                    self.daily_trades_count += 1
                    self.last_trade_date = str(datetime.now().date())
                    print(f"📈 LONG position opened. Daily trades: {self.daily_trades_count}/{MAX_DAILY_TRADES}")
                break



            elif (
                zone["direction"] == "bear"
                and touchesFVG
                and self.isBearishHTF
                and self.marketOK
            ):
                stop_loss = self.cur_close + atr * SL_MULTIPLIER
                if USE_TRAILING:
                    trail_stop = self.cur_close + atr * TRAIL_OFFSET_MULT
                else:
                    trail_stop = None

                tp = self.cur_close - atr * TP_MULTIPLIER
                entryAtr = atr
                lot_size = self.calculate_order_size(atr=atr, sl_mult=SL_MULTIPLIER)
                active_order = self.Order(
                    side="SELL", entry_price=self.cur_close, take_profit=tp,
                    trailing_stop_loss=trail_stop, stop_loss=stop_loss, entry_atr=entryAtr, order_size=lot_size,
                    **self.api_order_kwargs()
                )
                result = active_order.place_order()

                self.active_orders.append(active_order)
                
                if result['success']:
                    zone["mitigated"] = True
                    self.lastPositionWasShort = True
                    self.lastPositionWasLong = False
                    self.daily_trades_count += 1
                    self.last_trade_date = str(datetime.now().date())
                    print(f"📉 SHORT position opened. Daily trades: {self.daily_trades_count}/{MAX_DAILY_TRADES}")
                break


    def update_stops(self):
        if len(self.active_orders) == 0:
            return

        pos: FVG_Order = self.active_orders[0]

        current_high = self.data["high"].iloc[-1]
        current_low = self.data["low"].iloc[-1]

        # === UPDATE TRAILING STOPS ===
        if self.lastPositionWasLong:
            if USE_TRAILING and pos.entry_atr is not None:
                potentialStop = current_high - pos.entry_atr * TRAIL_OFFSET_MULT
                if pos.trailing_stop_loss is not None:
                    new_stop = max(pos.trailing_stop_loss, potentialStop)
                    if new_stop > pos.trailing_stop_loss:
                        print(f"📊 Trailing stop updated: {pos.trailing_stop_loss:.5f} → {new_stop:.5f}")
                        pos.trailing_stop_loss = new_stop
                else:
                    pos.trailing_stop_loss = potentialStop

        if self.lastPositionWasShort:
            if USE_TRAILING and pos.entry_atr is not None:
                potentialStop = current_low + pos.entry_atr * TRAIL_OFFSET_MULT
                if pos.trailing_stop_loss is not None:
                    new_stop = min(pos.trailing_stop_loss, potentialStop)
                    if new_stop < pos.trailing_stop_loss:
                        print(f"📊 Trailing stop updated: {pos.trailing_stop_loss:.5f} → {new_stop:.5f}")
                        pos.trailing_stop_loss = new_stop
                else:
                    pos.trailing_stop_loss = potentialStop

    def first_iteration(self):
        print("🚀 Starting first iteration...")
        print("Waiting for the first closing bar")
        sleep_until_next_boundary(self.timeframe)

        self.data = self.gather_data()
        self.update_indicators()
        self.add_fvg_zones()
        self.entry_logic()
        self.update_stops()
        self.save_data()
        self.subscribe_to_price_updates()

    def bar_iteration(self):
        with self._lock:
            self.fetch_new_data()
            self.update_indicators()
            self.add_fvg_zones()
            self.entry_logic()
            self.update_stops()
            self.save_data()
        print(f"\n⏰ New bar - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} close: {self.cur_close}")

