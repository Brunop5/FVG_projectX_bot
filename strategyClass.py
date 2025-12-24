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

class Strategy:
    def __init__(self, asset_pair):
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
        self.longTrailStop = None
        self.longTp = None
        self.entryAtrLong = None
        self.shortTrailStop = None
        self.shortTp = None
        self.entryAtrShort = None

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
        atrOK = atrVal > sma(atrVal, 20)
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
        current_high = self.data["high"].iloc[-1]
        current_low = self.data["low"].iloc[-1]

        atr = get_atr(self.data, ATR_PERIOD)

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
                # TODO: buy order here
                self.longTrailStop = self.cur_close - atr * SL_MULTIPLIER
                self.longTp = self.cur_close + atr * TP_MULTIPLIER
                self.entryAtrLong = atr
                zone["mitigated"] = True
                self.lastPositionWasLong = True
                self.lastPositionWasShort = False
                self.inPosition = True

            elif (
                zone["direction"] == "bear"
                and touchesFVG
                and self.isBearishHTF
                and self.marketOK
            ):
                # TODO: place sell order here
                self.shortTrailStop = self.cur_close + atr * SL_MULTIPLIER
                self.shortTp = self.cur_close - atr * TP_MULTIPLIER
                self.entryAtrShort = atr
                zone["mitigated"] = True
                self.lastPositionWasShort = True
                self.lastPositionWasLong = False
                self.inPosition = True


    def first_iteration(self):
        self.data = self.gather_data()
        self.calculate_indicators()
        self.add_fvg_zones()

    def run_iteration(self):
        pass

if __name__ == "__main__":
    for pair in ASSETS.items():
        strat = Strategy(pair)
        print(strat.marketOK)