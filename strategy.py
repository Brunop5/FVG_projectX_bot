from api_functions import fetch_data
from indicators import ema


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

bars = fetch_data(50, "4h")
four_h_ema = ema(bars)
