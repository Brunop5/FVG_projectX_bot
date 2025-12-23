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
        self.lastBullFvg = False
        self.lastBearFvg = False
        self.bullishPowerOK = None
        self.bearishPowerOK = None

        self.calculate_indicators()

    def gather_data(self):
        data = load_data(self.asset)
        if data is not None:
            return data
        
        return fetch_data(self.asset, self.timeframe, 100)

    def fetch_new_data(self):
        new_row = fetch_data(self.asset, self.timeframe, 1)
        self.cur_close = new_row["close"].iloc[0]
        self.cur_volume = new_row["volume"].iloc[0]
        self.data = pd.concat([self.data, new_row], ignore_index=True)

    def calculate_indicators(self):
        bars = fetch_data("gold", "4h", 50)
        htfEMA = ema(bars, 50)

        self.isBullishHTF = self.cur_close > htfEMA
        self.isBearishHTF = self.cur_close < htfEMA

        volOK = self.cur_volume > sma(self.data["volume"], 20) * 1.2
        atrVal = get_atr(ATR_PERIOD)
        atrOK = atrVal > sma(atrVal, 20)
        self.marketOK = volOK and atrOK


        self.lastBullFvg  = self.data["high"].iloc[-4] < self.data["low"].iloc[-2] and not self.lastBullFvg
        self.lastBearFvg = self.data["low"].iloc[-4] > self.data["high"].iloc[-2] and not self.lastBearFvg

        gapClose = self.data["close"].iloc[-3]

        self.bullishPowerOK = (
            self.isBullishFVG
            and (self.data["low"].iloc[-2] - self.data["high"].iloc[-4]) / gapClose * 100 >= MIN_FVG_POWER_PCT
        )

        self.bearishPowerOK = (
            self.isBearishFVG
            and (self.data["low"].iloc[-4] - self.data["high"].iloc[-2]) / gapClose * 100 >= MIN_FVG_POWER_PCT
        )




if __name__ == "__main__":
    for pair in ASSETS.items():
        strat = Strategy(pair)
        print(strat.marketOK)