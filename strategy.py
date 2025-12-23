from api_functions import *
from indicators import *
import pandas as pd


def add_fvg_zones(fvg_zones, bullishPowerOK, isBullishHTF, marketOK, isFvgToShow, bearishPowerOK, isBearishHTF):
    # === FVG ZONE CREATION (equivalent to the box.new blocks) ===
    if bullishPowerOK and isBullishHTF and marketOK and isFvgToShow:
        # Bullish FVG uses low[1] as top and high[3] as bottom in Pine
        fvg_zones.append(
            {
                "direction": "bull",
                "top": data["low"].iloc[-2],     # low[1]
                "bottom": data["high"].iloc[-4], # high[3]
                "mitigated": False,
            }
        )

    if bearishPowerOK and isBearishHTF and marketOK and isFvgToShow:
        # Bearish FVG uses low[3] as top and high[1] as bottom in Pine
        fvg_zones.append(
            {
                "direction": "bear",
                "top": data["low"].iloc[-4],     # low[3]
                "bottom": data["high"].iloc[-2], # high[1]
                "mitigated": False,
            }
        )

    # Limit number of stored FVGs, similar to fvgHistoryNbr trimming in Pine
    if len(fvg_zones) > fvgHistoryNbr:
        fvg_zones = fvg_zones[-fvgHistoryNbr:]

# ===== INPUTS ======
assets = {"gold":"30min"}

# if running for the first time or after a break:
for asset, timeframe in assets.items():
    data = fetch_data(asset, timeframe, 100)
    save_data(data, asset)
    


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
# ===================

close = fetch_cur_price("gold")
add_new_price(close)
volume = fetch_cur_volume("gold", assets["gold"])

# ====== INDICATORS ======
bars = fetch_data("gold", "4h", 50)
htfEMA = ema(bars)

isBullishHTF = close > htfEMA
isBearishHTF = close < htfEMA

volOK = volume > sma(volume, 20) * 1.2
atrVal = get_atr(atrPeriod)
atrOK = atrVal > sma(atrVal, 20)
marketOK = volOK and atrOK
# =========================


lastBullFvg = False
lastBearFvg = False
lastPositionWasLong = False
lastPositionWasShort = False
longTrailStop = None
longTp = None
entryAtrLong = None
shortTrailStop = None
shortTp = None
entryAtrShort = None
prevStructureHigh = data["high"].iloc[-21:-1].max()
prevStructureLow =  data["low"].iloc[-21:-1].min()

isBullishFVG = data["high"].iloc[-4] < data["low"].iloc[-2] and not lastBullFvg
isBearishFVG = data["low"].iloc[-4] > data["high"].iloc[-2] and not lastBearFvg
lastBullFvg = isBullishFVG
lastBearFvg = isBearishFVG

gapClose = data["close"].iloc[-3]

bullishPowerOK = (
    isBullishFVG
    and (data["low"].iloc[-2] - data["high"].iloc[-4]) / gapClose * 100 >= minFvgPowerPct
)

bearishPowerOK = (
    isBearishFVG
    and (data["low"].iloc[-4] - data["high"].iloc[-2]) / gapClose * 100 >= minFvgPowerPct
)


"""
Simple in‑memory representation of FVG zones, equivalent to the PineScript fvgBoxes/fvgTypes/isMitigated logic.
Each zone is a dict: {"direction": "bull"|"bear", "top": float, "bottom": float, "mitigated": bool}
"""
fvg_zones: list[dict] = []
add_fvg_zones


# === ENTRY LOGIC (translated from the Pine loop over fvgBoxes) ===
atr = get_atr(atrPeriod)
inPosition = False  # TODO: wire this up to your actual position state

current_high = data["high"].iloc[-1]
current_low = data["low"].iloc[-1]

if fvg_zones and not inPosition:
    # Only check the most recent fvgHistoryNbr zones, like in Pine
    for zone in fvg_zones[-fvgHistoryNbr:]:
        if zone["mitigated"]:
            continue

        fvg_bottom = zone["bottom"]
        fvg_top = zone["top"]

        # Full touch: current bar's high/low overlaps the FVG zone
        touchesFVG = current_high >= fvg_bottom and current_low <= fvg_top

        if (
            zone["direction"] == "bull"
            and touchesFVG
            and isBullishHTF
            and marketOK
        ):
            # TODO: place buy order here
            longTrailStop = close - atr * slMultiplier
            longTp = close + atr * tpMultiplier
            entryAtrLong = atr
            zone["mitigated"] = True
            lastPositionWasLong = True
            lastPositionWasShort = False
            inPosition = True

        elif (
            zone["direction"] == "bear"
            and touchesFVG
            and isBearishHTF
            and marketOK
        ):
            # TODO: place sell order here
            shortTrailStop = close + atr * slMultiplier
            shortTp = close - atr * tpMultiplier
            entryAtrShort = atr
            zone["mitigated"] = True
            lastPositionWasShort = True
            lastPositionWasLong = False
            inPosition = True