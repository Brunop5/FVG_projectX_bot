from api_functions import *
from indicators import *
import pandas as pd

IS_FVG_TO_SHOW = True              # Display FVG
FVG_HISTORY_NBR = 5                # Number of FVGs to show (1-50)
IS_MITIGATED_FVG_TO_REDUCE = True  # Reduce mitigated FVG
MIN_FVG_POWER_PCT = 0.1            # Min FVG Power %
HTF_TF = "240"                     # HTF Bias (4H)
ATR_PERIOD = 14                    # ATR Period (min 1)
SL_MULTIPLIER = 4.0                # SL ATR Multiplier
TP_MULTIPLIER = 25.0               # TP ATR Multiplier (Positional: Wider Targets)
USE_TRAILING = True                # use trailing stop
TRAIL_OFFSET_MULT = 8.0            # Trailing Offset ATR Multiplier (Wide for Positional)
HOLD_UNTIL_OPPOSITE = True         # Hold Until Opposite BOS/CHoCH
ASSETS = {"gold":"30min"}

class Order:
    def __init__(self, side: str, entry_price: float,
                take_profit: float, trailing_stop_loss,
                entry_atr: float):
        self.side = side
        self.entry_price = entry_price
        self.take_profit = take_profit
        self.trailing_stop_loss = trailing_stop_loss
        self.entry_atr = entry_atr

        self.place_order()

    def place_order(self):
        # place order through api
        pass

    def close_order(self):
        pass

    def check_stops(self, current_price):
        if self.side == "BUY":
            if current_price <= self.trailing_stop_loss:
                self.close_order()

            if current_price >= self.take_profit:
                self.close_order()
        
        elif self.side == "SELL":
            if current_price >= self.trailing_stop_loss:
                self.close_order()

            if current_price <= self.take_profit:
                self.close_order()


class Strategy:
    def __init__(self, asset_pair):
        self.active_order = None

        self.timeframe = asset_pair[1]
        self.asset = asset_pair[0]
        self.data = self.gather_data()
        self.cur_close = self.data["close"].iloc[-1]
        self.cur_volume = self.data["volume"].iloc[-1]

        self.isBullishHTF = None
        self.isBearishHTF = None
        self.marketOK = None
        self.bullishPowerOK = None
        self.bearishPowerOK = None
        self.isBOS = None
        self.isCHOCH = None
        self.prevStructureHigh = None
        self.prevStructureLow = None

        self.inPosition = False
        self.lastBullFvg = False
        self.lastBearFvg = False
        self.lastPositionWasLong = False
        self.lastPositionWasShort = False

        self.calculate_indicators()

        self.fvg_zones: list[dict] = []
        self.add_fvg_zones()


    def gather_data(self):
        data = load_data(self.asset)
        if data is not None:
            return data
        
        return fetch_data(self.asset, self.timeframe, 100)

    def fetch_new_data(self):
        new_row = fetch_data(self.asset, self.timeframe, 1)
        self.cur_close = new_row["close"].iloc[0]
        self.cur_volume = new_row["volume"].iloc[0]
        self.data = pd.concat([self.data, new_row], ignore_index=True).iloc[-100:] # last 100

    def update_trend_indicators(self):
        bars = fetch_data("gold", "4h", 50)
        htfEMA = ema(bars, 50)

        self.isBullishHTF = self.cur_close > htfEMA
        self.isBearishHTF = self.cur_close < htfEMA

        volOK = self.cur_volume > sma(self.data["volume"], 20) * 1.2
        atrVal = get_atr(self.data, ATR_PERIOD)
        atrOK = atrVal.iloc[-1] > sma(atrVal, 20)
        self.marketOK = volOK and atrOK


        self.lastBullFvg  = self.data["high"].iloc[-4] < self.data["low"].iloc[-2] and not self.lastBullFvg
        self.lastBearFvg = self.data["low"].iloc[-4] > self.data["high"].iloc[-2] and not self.lastBearFvg

    def calc_BOS_and_CHOCH(self):
        self.prevStructureHigh = self.data["high"].iloc[-21:-1].max()
        self.prevStructureLow = self.data["low"].iloc[-21:-1].min()

        previous_close = self.data["close"].iloc[-2]

        self.isBOS = crossover(
            self.cur_close,
            previous_close,
            self.prevStructureHigh,  # prevStructureHigh[1] in PineScript
        )

        # PineScript: isCHOCH = ta.crossunder(close, prevStructureLow[1])
        self.isCHOCH = crossunder(
            self.cur_close,
            previous_close,
            self.prevStructureLow,   # prevStructureLow[1] in PineScript
        )

    def calculate_indicators(self):
        self.update_trend_indicators()

        gapClose = self.data["close"].iloc[-3]

        self.bullishPowerOK = (
            self.lastBullFvg
            and (self.data["low"].iloc[-2] - self.data["high"].iloc[-4]) / gapClose * 100 >= MIN_FVG_POWER_PCT
        )

        self.bearishPowerOK = (
            self.lastBearFvg
            and (self.data["low"].iloc[-4] - self.data["high"].iloc[-2]) / gapClose * 100 >= MIN_FVG_POWER_PCT
        )

        self.calc_BOS_and_CHOCH()

    def add_fvg_zones(self):
        # === FVG ZONE CREATION (equivalent to the box.new blocks) ===
        if self.bullishPowerOK and self.isBullishHTF and self.marketOK and IS_FVG_TO_SHOW:
            # Bullish FVG uses low[1] as top and high[3] as bottom in Pine
            self.fvg_zones.append(
                {
                    "direction": "bull",
                    "top": self.data["low"].iloc[-2],     # low[1]
                    "bottom": self.data["high"].iloc[-4], # high[3]
                    "mitigated": False,
                }
            )

        if self.bearishPowerOK and self.isBearishHTF and self.marketOK and IS_FVG_TO_SHOW:
            # Bearish FVG uses low[3] as top and high[1] as bottom in Pine
            self.fvg_zones.append(
                {
                    "direction": "bear",
                    "top": self.data["low"].iloc[-4],     # low[3]
                    "bottom": self.data["high"].iloc[-2], # high[1]
                    "mitigated": False,
                }
            )

        # Limit number of stored FVGs, similar to fvgHistoryNbr trimming in Pine
        if len(self.fvg_zones) > FVG_HISTORY_NBR:
            self.fvg_zones = self.fvg_zones[-FVG_HISTORY_NBR:]

    def entry_logic(self):
        if len(self.fvg_zones) == 0 or self.inPosition:
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
                trailStop = self.cur_close - atr * SL_MULTIPLIER
                tp = self.cur_close + atr * TP_MULTIPLIER
                entryAtr = atr
                self.active_order = Order("BUY", self.cur_close, tp, trailStop, entryAtr)
                zone["mitigated"] = True
                self.lastPositionWasLong = True
                self.lastPositionWasShort = False
                self.inPosition = True
                break


            elif (
                zone["direction"] == "bear"
                and touchesFVG
                and self.isBearishHTF
                and self.marketOK
            ):
                trailStop = self.cur_close + atr * SL_MULTIPLIER
                tp = self.cur_close - atr * TP_MULTIPLIER
                entryAtr = atr
                self.active_order = Order("SELL", self.cur_close, tp, trailStop, entryAtr)
                zone["mitigated"] = True
                self.lastPositionWasShort = True
                self.lastPositionWasLong = False
                self.inPosition = True
                break

    def update_stops(self):
        pos = self.active_order
        if pos is None:
            return

        current_high = self.data["high"].iloc[-1]
        current_low = self.data["low"].iloc[-1]

        # === UPDATE TRAILING STOPS ===
        if self.inPosition and self.lastPositionWasLong:
            if USE_TRAILING and pos.entry_atr is not None:
                potentialStop = current_high - pos.entry_atr * TRAIL_OFFSET_MULT
                if pos.trailing_stop_loss is not None:
                    pos.trailing_stop_loss = max(pos.trailing_stop_loss, potentialStop)
                else:
                    pos.trailing_stop_loss = potentialStop

        if self.inPosition and self.lastPositionWasShort:
            if USE_TRAILING and pos.entry_atr is not None:
                potentialStop = current_low + pos.entry_atr * TRAIL_OFFSET_MULT
                if pos.trailing_stop_loss is not None:
                    pos.trailing_stop_loss = min(pos.trailing_stop_loss, potentialStop)
                else:
                    pos.trailing_stop_loss = potentialStop

        # === CLOSE ON OPPOSITE BOS/CHoCH ===
        if HOLD_UNTIL_OPPOSITE and self.inPosition:
            if self.lastPositionWasLong and self.isCHOCH:
                self.active_order.close_order()
                self.active_order = None
                self.inPosition = False
                self.lastPositionWasLong = False

            if self.lastPositionWasShort and self.isBOS:
                self.active_order.close_order()
                self.active_order = None
                self.inPosition = False
                self.lastPositionWasShort = False



    def first_iteration(self):
        self.data = self.gather_data()
        self.calculate_indicators()
        self.add_fvg_zones()
        self.entry_logic()
        self.update_stops()

    def run_bar_iteration(self):
        self.fetch_new_data()
        self.calculate_indicators()
        self.add_fvg_zones()
        self.entry_logic()
        self.update_stops()

    def run_websocket_iteration(self):
        if self.active_order is None:
            return
        self.active_order.check_stops()

if __name__ == "__main__":
    for pair in ASSETS.items():
        strat = Strategy(pair)
        print(strat.marketOK)