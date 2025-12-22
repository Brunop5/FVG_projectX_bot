from api_functions import fetch_cur_price, fetch_cur_volume, fetch_data
from indicators import ema

assets = {"gold":"30min"}


isFvgToShow = True              # Display FVG
fvgHistoryNbr = 5               # Number of FVGs to show (1-50)
isMitigatedFvgToReduce = True   # Reduce mitigated FVG
minFvgPowerPct = 0.1            # Min FVG Power %
htfTF = "240"                   # HTF Bias (4H)
atrPeriod = 14                  # ATR Period (min 1)
slMultiplier = 4.0              # SL ATR Multiplier
tpMultiplier = 25.0             # TP ATR Multiplier (Positional: Wider Targets)
useTrailing = True              # use trailing stop
trailOffsetMult = 8.0           # Trailing Offset ATR Multiplier (Wide for Positional)
holdUntilOpposite = True        # Hold Until Opposite BOS/CHoCH

bars = fetch_data("gold", "4h", 50)
htfEMA = ema(bars)

close = fetch_cur_price("gold")
volume = fetch_cur_volume("gold", assets["gold"])

isBullishHTF = close > htfEMA
isBearishHTF = close < htfEMA

volOK = volume > ta.sma(volume, 20) * 1.2
atrVal = ta.atr(atrPeriod)
atrOK = atrVal > ta.sma(atrVal, 20)
marketOK = volOK and atrOK
var bool lastBullFvg = false
var bool lastBearFvg = false
var bool lastPositionWasLong = false
var bool lastPositionWasShort = false
var float longTrailStop = na
var float longTp = na
var float entryAtrLong = na
var float shortTrailStop = na
var float shortTp = na
var float entryAtrShort = na