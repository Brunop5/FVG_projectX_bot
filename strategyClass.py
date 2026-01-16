from api_functions import *
from indicators import *
import pandas as pd
import json
from time import sleep
import threading
import os
import requests
metadata_lock = threading.Lock()


# ==================== CONFIGURATION PARAMETERS ====================
# Display Settings
FVG_HISTORY_NBR = 15              # Number of FVGs to work with
MIN_FVG_POWER_PCT = 0.01          # Min FVG Power % (formerly MinFVGPowerPct)

# Timeframe and Trend Settings
HTF_TF = "120"                     # HTF Bias (4H) - PERIOD_H4
EMA_PERIOD = 25                    # EMA Period for trend detection
VOLUME_MULTIPLIER = 1.25
USE_VOLUME_CHECK = True            # If False, volume check is skipped in marketOK calculation
VOLUME_DATA_START_TIMESTAMP = 1755464400000  # Timestamp where reliable volume data starts (ms)
START_FROM_VOLUME_TIMESTAMP = False  # None = auto (True if USE_VOLUME_CHECK, False otherwise). Set to True/False to override

# ATR and Risk Management
ATR_PERIOD = 18                    # ATR Period (min 1)
SL_MULTIPLIER = 6                # SL ATR Multiplier (formerly SL_ATR_Mult)
TP_MULTIPLIER = 19                # TP ATR Multiplier (formerly TP_ATR_Mult)

# Trailing Stop Settings
USE_TRAILING = True                # use trailing stop (formerly UseTrailing)
TRAIL_OFFSET_MULT = 1            # Trailing Offset ATR Multiplier (formerly TrailATRMult)

# Position Management
HOLD_UNTIL_OPPOSITE = False         # Hold Until Opposite BOS/CHoCH

# Lot Size and Risk Settings
USE_FIXED_LOT = True        # Use fixed lot size (formerly UseFixedLot)
FIXED_LOT = 5                   # Fixed lot size (formerly FixedLot)
RISK_PERCENT = 1.0                 # Risk percentage per trade (formerly RiskPercent)
ORDER_SIZE = 1                     # Default order size (overridden by risk calculation if not USE_FIXED_LOT)

# Daily Trading Limits
MAX_DAILY_TRADES = 3               # Maximum trades per day (formerly MaxDailyTrades)

# Assets and API Settings
# list of asset, timeframe and account name combinations;
# format: [(asset1, timeframe1, account_name1), (asset2, timeframe2, account1), (..., ..., account2), ...]
ASSETS = [("CON.F.US.GCE.G26","1min", "50KTC-V2-252499-38617147"), ("CON.F.US.MNQ.H26", "5min", "50KTC-V2-252499-66765377")]

USERNAME = os.getenv("USERNAME")
API_KEY = os.getenv("API_KEY")
LIVE = False  # or False


# ====================
# if true, updates contracts.csv - this should be done at least monthly
UPDATE_CONTRACT_LIST = False

# if true it will print the list of valid accounts for this api key
SHOW_ACCOUNTS = False
# ======== if any of those two is true, it will run the option, but not the strategy


def init_api():
    res = login_to_api(USERNAME, API_KEY)
    if not res["success"]:
        raise RuntimeError("❌ API login failed")

    global_token = res["token"]
    print(f"✅ API initialized.")
    return global_token


class Order:
    def __init__(self, side: str, entry_price: float,
                take_profit: float, trailing_stop_loss, entry_atr,
                account_id, asset_id, auth_token, lot_size=None):
        self.side = side
        self.entry_price = entry_price
        self.take_profit = take_profit
        self.trailing_stop_loss = trailing_stop_loss
        self.entry_atr = entry_atr
        self.lot_size = lot_size if lot_size else ORDER_SIZE


        self.account_id = account_id
        self.asset_id = asset_id
        self.auth_token = auth_token

    def place_order(self):
        """
        Place an order using ProjectX Gateway API.
        Based on: https://gateway.docs.projectx.com/docs/api-reference/order/order-place
        """        
        if not self.auth_token:
            print("Error: auth_token is required to place order")
            return {'success': False, 'message': 'auth_token is required'}
        
        url = "https://api.topstepx.com/api/Order/place"
        
        headers = {
            'accept': 'text/plain',
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.auth_token}'
        }
        
        # Map side: "BUY" -> 0 (Bid), "SELL" -> 1 (Ask)
        side_code = 0 if self.side.upper() == "BUY" else 1
        
        payload = {
            "accountId": self.account_id,
            "contractId": self.asset_id,
            "type": 2,  # 2 = Market order
            "side": side_code,  # 0 = Bid (buy), 1 = Ask (sell)
            "size": self.lot_size,
            "limitPrice": None,
            "stopPrice": None,
            "trailPrice": None,
            "customTag": None,
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                if result.get("success", False):
                    order_id = result.get("orderId")
                    print(f"✅ Order placed successfully. Order ID: {order_id}")
                    print(f"   Side: {self.side}, Size: {self.lot_size}, Entry: {self.entry_price:.5f}")
                    print(f"   TP: {self.take_profit:.5f}, SL: {self.trailing_stop_loss:.5f}")
                    return {
                        'success': True,
                        'order_id': order_id,
                        'message': 'Order placed successfully'
                    }
                else:
                    error_msg = result.get("errorMessage", "Unknown error")
                    print(f"❌ Order placement failed: {error_msg}")
                    return {
                        'success': False,
                        'order_id': None,
                        'message': error_msg
                    }
            else:
                error_msg = f"HTTP {response.status_code}: {response.text}"
                print(f"❌ Order placement failed: {error_msg}")
                return {
                    'success': False,
                    'order_id': None,
                    'message': error_msg
                }
        
        except ImportError:
            return {
                'success': False,
                'order_id': None,
                'message': 'requests library not installed. Install with: pip install requests'
            }
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            print(error_msg)
            return {
                'success': False,
                'order_id': None,
                'message': error_msg
            }

    def close_order(self):
        url = "https://api.topstepx.com/api/Position/closeContract"

        headers = {
            "Authorization": f"Bearer {self.auth_token}",
            "Accept": "application/json",
            "Content-Type": "application/json"
        }

        payload = {
            "accountId": self.account_id,
            "contractId": self.asset_id
        }

        try:
            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"❌ Failed to close position: {e}")
            raise Exception(f"Unexpected response: {e}")

    def check_stops(self, current_price):
        """Check if stop loss or take profit has been hit"""
        if self.side == "BUY":
            if current_price <= self.trailing_stop_loss:
                print(f"🛑 Stop Loss hit for LONG position at {current_price:.5f}")
                self.close_order()
                return True

            if current_price >= self.take_profit:
                print(f"🎯 Take Profit hit for LONG position at {current_price:.5f}")
                self.close_order()
                return True
        
        elif self.side == "SELL":
            if current_price >= self.trailing_stop_loss:
                print(f"🛑 Stop Loss hit for SHORT position at {current_price:.5f}")
                self.close_order()
                return True

            if current_price <= self.take_profit:
                print(f"🎯 Take Profit hit for SHORT position at {current_price:.5f}")
                self.close_order()
                return True
        
        return False


class Strategy:
    def __init__(self, asset_tuple):
        self.auth_token = None
        self.account_id = None

        self.timeframe = asset_tuple[1]
        self.asset = asset_tuple[0]
        self.account_name = asset_tuple[2]

    def init_rest(self):
        self.account_id = get_account_id(self.auth_token, account_name=self.account_name)
        self.account_balance = get_account_balance(self.account_id, self.auth_token)
        self.active_order = None

        self.data = self.gather_data()
        print(f"📊 Loaded {len(self.data)} bars for {self.asset}")

        self.cur_close = self.data["close"].iloc[-1]
        self.cur_volume = self.data["volume"].iloc[-1]

        self.isBullishHTF = None
        self.isBearishHTF = None
        self.marketOK = None
        self.bullishPowerOK = None
        self.bearishPowerOK = None
        self.isBOS = False
        self.isCHOCH = False
        self.prevStructureHigh = None
        self.prevStructureLow = None

        self.inPosition = False
        self.lastBullFvg = False
        self.lastBearFvg = False
        self.lastPositionWasLong = False
        self.lastPositionWasShort = False

        self.daily_trades_count = 0
        self.last_trade_date = None

        self.load_metadata()
        self.calculate_indicators()

        self.fvg_zones: list[dict] = []
        self.add_fvg_zones()

    def set_token(self, token):
        self.auth_token = token

    def load_metadata(self):
        with metadata_lock: 
            path = f"metadata.json"
            if not os.path.exists(path):
                return

            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            f.close()

        data = data.get(f"{self.account_id}-{self.asset}-{self.timeframe}", None)
        if data is None:
            return

        if data["active_order"] is not None:
            self.active_order = Order(**data["active_order"])

        self.inPosition = bool(data.get("inPosition", False))
        self.lastPositionWasLong = bool(data.get("lastPositionWasLong", False))
        self.lastPositionWasShort = bool(data.get("lastPositionWasShort", False))

        self.daily_trades_count = int(data.get("daily_trades_count", 0))
        self.last_trade_date = data.get("last_trade_date")

    def get_assets(self):
        url = "https://api.topstepx.com/api/Contract/available"

        headers = {
            "Authorization": f"Bearer {self.auth_token}",
            "accept": "text/plain",
            "Content-Type": "application/json"
        }

        payload = {
            "live": LIVE
        }

        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        
        return response.json()["contracts"]

    def gather_data(self):
        data = load_data(self.asset, self.timeframe)
        if data is not None:
            return data
        
        return fetch_data(self.asset, self.timeframe, 100, self.auth_token, LIVE)

    def fetch_new_data(self):
        new_row = fetch_data(self.asset, self.timeframe, 1, self.auth_token, LIVE)
        if new_row is None:
            return
        if new_row["timestamp"].iloc[-1] > self.data["timestamp"].iloc[-1]:
            self.cur_close = new_row["close"].iloc[-1]
            self.cur_volume = new_row["volume"].iloc[-1]
            self.data = pd.concat([self.data, new_row], ignore_index=True).iloc[-100:] # last 100

    def update_trend_indicators(self):
        bars = fetch_data(self.asset, f"{HTF_TF}min", max(101, EMA_PERIOD+51), self.auth_token, LIVE)
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

    def check_daily_trade_limit(self):
        """Check if maximum daily trades has been reached"""
        today = datetime.now().date()
        
        if self.last_trade_date != str(today):
            # Reset counter for new day
            self.daily_trades_count = 0
            self.last_trade_date = str(today)
        
        return self.daily_trades_count < MAX_DAILY_TRADES

    def calculate_lot_size(self, atr, stop_distance_atr_mult):
        """Calculate position size based on risk management"""
        if USE_FIXED_LOT:
            return FIXED_LOT
        
        # Calculate lot size based on risk percentage
        # This is a simplified calculation - adjust based on your broker's requirements
        risk_amount = self.account_balance * (RISK_PERCENT / 100)
        stop_distance = atr * stop_distance_atr_mult
        
        if stop_distance > 0:
            lot_size = risk_amount / stop_distance
            # Round to appropriate precision
            lot_size = round(lot_size, 2)
            return max(0.01, min(lot_size, 100))  # Ensure reasonable bounds
        
        return ORDER_SIZE

    def entry_logic(self):
        if len(self.fvg_zones) == 0 or self.inPosition:
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
                trailStop = self.cur_close - atr * SL_MULTIPLIER
                tp = self.cur_close + atr * TP_MULTIPLIER
                entryAtr = atr
                lot_size = self.calculate_lot_size(atr, SL_MULTIPLIER)

                self.active_order = Order("BUY", self.cur_close, tp, trailStop, entryAtr,
                                          self.account_id, self.asset, self.auth_token, lot_size)
                result = self.active_order.place_order()
                
                if result['success']:
                    zone["mitigated"] = True
                    self.lastPositionWasLong = True
                    self.lastPositionWasShort = False
                    self.inPosition = True
                    self.daily_trades_count += 1
                    self.last_trade_date = str(datetime.now().date())
                    self.pending_signal = None
                    print(f"📈 LONG position opened. Daily trades: {self.daily_trades_count}/{MAX_DAILY_TRADES}")
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
                self.active_order = Order("SELL", self.cur_close, tp, trailStop, entryAtr,
                                          self.account_id, self.asset, self.auth_token)
                result = self.active_order.place_order()
                
                if result['success']:
                    zone["mitigated"] = True
                    self.lastPositionWasShort = True
                    self.lastPositionWasLong = False
                    self.inPosition = True
                    self.daily_trades_count += 1
                    self.last_trade_date = str(datetime.now().date())
                    self.pending_signal = None
                    print(f"📉 SHORT position opened. Daily trades: {self.daily_trades_count}/{MAX_DAILY_TRADES}")
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
                    new_stop = max(pos.trailing_stop_loss, potentialStop)
                    if new_stop > pos.trailing_stop_loss:
                        print(f"📊 Trailing stop updated: {pos.trailing_stop_loss:.5f} → {new_stop:.5f}")
                        pos.trailing_stop_loss = new_stop
                else:
                    pos.trailing_stop_loss = potentialStop

        if self.inPosition and self.lastPositionWasShort:
            if USE_TRAILING and pos.entry_atr is not None:
                potentialStop = current_low + pos.entry_atr * TRAIL_OFFSET_MULT
                if pos.trailing_stop_loss is not None:
                    new_stop = min(pos.trailing_stop_loss, potentialStop)
                    if new_stop < pos.trailing_stop_loss:
                        print(f"📊 Trailing stop updated: {pos.trailing_stop_loss:.5f} → {new_stop:.5f}")
                        pos.trailing_stop_loss = new_stop
                else:
                    pos.trailing_stop_loss = potentialStop

        # === CLOSE ON OPPOSITE BOS/CHoCH ===
        if HOLD_UNTIL_OPPOSITE and self.inPosition:
            if self.lastPositionWasLong and self.isCHOCH:
                print("🔄 CHoCH detected - Closing LONG position")
                self.active_order.close_order()
                self.active_order = None
                self.inPosition = False
                self.lastPositionWasLong = False

            if self.lastPositionWasShort and self.isBOS:
                print("🔄 BOS detected - Closing SHORT position")
                self.active_order.close_order()
                self.active_order = None
                self.inPosition = False
                self.lastPositionWasShort = False

    def save_data(self):
        order_dict = None
        if self.active_order is not None:
            order_dict = self.active_order.__dict__

        res_dict = {
            "active_order": order_dict,
            "inPosition": self.inPosition,
            "lastPositionWasShort": self.lastPositionWasShort,
            "lastPositionWasLong": self.lastPositionWasLong,
            "daily_trades_count": self.daily_trades_count,
            "last_trade_date": self.last_trade_date
        }

        key = f"{self.account_id}-{self.asset}-{self.timeframe}"
        path = "metadata.json"

        with metadata_lock:
            # Load existing metadata (if any)
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    try:
                        metadata = json.load(f)
                    except json.JSONDecodeError:
                        metadata = {}
            else:
                metadata = {}

            # Update only this strategy's entry
            metadata[key] = res_dict

            # Write back atomically
            with open(path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)

        # CSV saving is independent
        self.data.to_csv(f"{self.asset}-{self.timeframe}.csv")


    def first_iteration(self):
        print("🚀 Starting first iteration...")
        print("Waiting for the first closing bar")
        sleep_until_next_boundary(self.timeframe)

        self.data = self.gather_data()
        self.calculate_indicators()
        self.add_fvg_zones()
        self.entry_logic()
        self.update_stops()
        self.save_data()

    def run_bar_iterations(self):
        """Main loop for bar-based updates, runs aligned with timeframe"""
        timeframe_sec = TIMEFRAME_SECONDS[self.timeframe]
        
        # next bar start time
        next_bar = datetime.now() + timedelta(seconds=timeframe_sec)
        
        while True:
            try:
                self.fetch_new_data()
                self.calculate_indicators()
                self.add_fvg_zones()
                self.entry_logic()
                self.update_stops()
                self.save_data()
                print(f"\n⏰ New bar - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} close: {self.cur_close}")
                
                # calculate exact sleep until next bar
                now = datetime.now()
                sleep_seconds = (next_bar - now).total_seconds()
                if sleep_seconds < 0:
                    # we are behind schedule, skip sleep
                    sleep_seconds = 0
                sleep(sleep_seconds)
                
                # schedule next bar
                next_bar += timedelta(seconds=timeframe_sec)
                
            except Exception as e:
                print(f"❌ Error in bar iteration: {e}")
                sleep(60)

    def update_price(self):
        while True:
            sleep(10)
            new_row = fetch_data(self.asset, "10s", 1, self.auth_token, LIVE)
            if new_row is None:
                return

            self.cur_price = new_row["close"].iloc[-1]
            print(f"updated price: {self.cur_price}")

            if self.active_order is not None:
                closed = self.active_order.check_stops(self.cur_close)
                if closed:
                    self.active_order = None
                    self.inPosition = False
                    self.lastPositionWasLong = False
                    self.lastPositionWasShort = False
                    self.save_data()

    def run(self):
        """Start the trading bot"""
        print(f"\n{'='*60}")
        print(f"🤖 Trading Bot Started for {self.asset}")
        print(f"{'='*60}")
        print(f"Timeframe: {self.timeframe}")
        print(f"HTF Bias: {HTF_TF}min | EMA Period: {EMA_PERIOD}")
        self.first_iteration()


        t1 = threading.Thread(target=self.run_bar_iterations)
        t2 = threading.Thread(target=self.update_price)
        t1.start()
        t2.start()


def run_strat(strat: Strategy, token):
    strat.set_token(token)
    strat.init_rest()
    strat.run()

def validation_thread(auth_token, strategies: list[Strategy]):
    print("starting validation thread...")
    while True:
        sleep(72000)
        res = validate_token(auth_token)
        if res["success"] == False:
            print("token update failed, API connection might fail soon...")
            print(res["message"])
            return

        new_token = res["newToken"]
        print("Sucessfully updated connection token")
        for strat in strategies:
            strat.set_token(new_token)


if __name__ == "__main__":
    #global_token = init_api()
    # ("CON.F.US.CLE.G26", 3), 
    assets = [("CON.F.US.MGC.G26", 0.5), ("CON.F.US.GCE.G26", 3.1)
              ("CON.F.US.YM.H26", 0.5), ("CON.F.US.SIL.H26", 0.5), ("CON.F.US.MNQ.H26", 0.74)]

    gather_historical_data(assets)
    
    # if UPDATE_CONTRACT_LIST:
    #     strat = Strategy(ASSETS[0])
    #     strat.set_token(global_token)
    #     strat.init_rest()
    #     data = strat.get_assets()
    #     data = pd.DataFrame(data)
    #     data.to_csv("contracts.csv")
    #     print("Contract list updated successfully!!")
    # elif SHOW_ACCOUNTS:
    #     strat = Strategy(ASSETS[0])
    #     strat.set_token(global_token)
    #     print(get_account_id(strat.auth_token, show=True))

    # else:
    #     threads = []
    #     strats = []
    #     for asset_pair in ASSETS:
    #         strats.append(Strategy(asset_pair))
        
    #     v_thread = threading.Thread(
    #         target = validation_thread,
    #         args = (global_token, strats,),
    #         daemon=True
    #     )
    #     v_thread.start()

    #     for strat in strats:
    #         t = threading.Thread(
    #             target=run_strat,
    #             args=(strat, global_token,),
    #             daemon=True
    #         )
    #         t.start()
    #         threads.append(t)



    # while True:
    #     time.sleep(5)
            