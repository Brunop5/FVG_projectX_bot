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
