import requests
import pandas as pd
from datetime import datetime, timedelta
import os
from time import sleep
import threading
from dotenv import load_dotenv

from ..FVG_strategy import FVG_Order, FVG_Strategy, HTF_TF, EMA_PERIOD
from ..FVG_strategy import USE_FIXED_LOT, FIXED_LOT, MAX_DAILY_TRADES
from ..FVG_strategy import RISK_PERCENT, ORDER_SIZE
from .projectx_api_functions import get_account_id
from .projectx_api_functions import get_account_balance
from .projectx_api_functions import load_data
from .projectx_api_functions import fetch_data
from .projectx_api_functions import login_to_api
from .projectx_api_functions import validate_token
from .projectx_api_functions import sleep_until_next_boundary



ASSETS = [("CON.F.US.GCE.G26","1min", "PRAC-V2-252499-51361945")]

load_dotenv()
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


class ProjectX_Order(FVG_Order):
    account_id: str
    asset_id: str
    auth_token: str

    def __init__(self, account_id, asset_id, auth_token, **kwargs):
        super().__init__(**kwargs)

        self.account_id = account_id
        self.asset_id = asset_id
        self.auth_token = auth_token


    def place_order(self):
        """
        Place an order using ProjectX Gateway API.
        Based on: https://gateway.docs.projectx.com/docs/api-reference/order/order-place
        """        
        print(f"ORDER_PLACED:{self.__dict__}")
        return
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
        print(f"ORDER_CLOSED:{self.__dict__}")
        return
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


class ProjectX_Strategy(FVG_Strategy):
    auth_token: str
    account_id: str
    account_name: str
    asset: str

    def __init__(self, asset_tuple):
        print("layer 3 init ran!")
        self.auth_token = None
        self.account_id = None

        self.asset = asset_tuple[0]
        self.timeframe = asset_tuple[1]
        self.account_name = asset_tuple[2]

        filename = f"{self.asset}-{self.timeframe}-{self.account_name}"
        self.csv_filename = f"{filename}.csv"
        self.metadata_filename = f"{filename}.json"

    def init_api(self, auth_token):
        self.set_token(auth_token)
        self.account_id = get_account_id(self.auth_token, self.account_name)
        self.account_balance = get_account_balance(self.account_id, self.auth_token)

        super().__init__()

    
    def set_token(self, token):
        self.auth_token = token

    def api_order_kwargs(self) -> dict:
        return {"account_id": self.account_id, "asset_id": self.asset, "auth_token": self.auth_token}


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

    def gather_data(self) -> pd.DataFrame:
        data = load_data(self.asset, self.timeframe)
        if data is not None:
            return data
        
        return fetch_data(self.asset, self.timeframe, 100, self.auth_token, LIVE)

    def fetch_new_data(self):
        new_row = fetch_data(self.asset, self.timeframe, 1, self.auth_token, LIVE)
        if new_row is None:
            print("now new data")
            return
        if new_row["timestamp"].iloc[-1] > self.data["timestamp"].iloc[-1]:
            self.cur_close = new_row["close"].iloc[-1]
            self.cur_volume = new_row["volume"].iloc[-1]
            self.data = pd.concat([self.data, new_row], ignore_index=True).iloc[-100:] # last 100
            print(f"\n⏰ New bar - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} close: {self.cur_close}")


    def fetch_htf_data(self) -> pd.DataFrame:
        htf_tf = str(HTF_TF)
        if htf_tf.isdigit():
            htf_tf = f"{htf_tf}min"

        num_bars = max(EMA_PERIOD + 51, 101)

        data = load_data(self.asset, htf_tf)
        if data is None or len(data) == 0:
            data = fetch_data(self.asset, htf_tf, num_bars, self.auth_token, LIVE)

        if data is None or len(data) == 0:
            return pd.DataFrame()

        if hasattr(self, "cur_close"):
            data = data.copy()
            data.loc[data.index[-1], "close"] = float(self.cur_close)

        return data


    def check_daily_trade_limit(self):
        """Check if maximum daily trades has been reached"""
        today = datetime.now().date()
        
        if self.last_trade_date != str(today):
            # Reset counter for new day
            self.daily_trades_count = 0
            self.last_trade_date = str(today)
        
        return self.daily_trades_count < MAX_DAILY_TRADES

    def calculate_order_size(self, atr, stop_distance_atr_mult):
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

    def subscribe_to_price_updates(self):
        while True:
            sleep(10)
            new_row = fetch_data(self.asset, self.timeframe, 1, self.auth_token, LIVE, include_partial_bar=True)
            self.update_price(new_row)

    def start_bar_iterations(self):
        sleep_until_next_boundary(self.timeframe)
        while True:
            try:
                self.bar_iteration()
                sleep_until_next_boundary(self.timeframe)
                
            except Exception as e:
                print(f"❌ Error in bar iteration: {e}")
                sleep(60)

    def run(self):
        """Start the trading bot"""
        print(f"\n{'='*60}")
        print(f"🤖 Trading Bot Started for {self.asset}")
        print(f"{'='*60}")
        print(f"Timeframe: {self.timeframe}")
        print(f"HTF Bias: {HTF_TF}min | EMA Period: {EMA_PERIOD}")
        self.first_iteration()


        t1 = threading.Thread(target=self.start_bar_iterations)
        t2 = threading.Thread(target=self.subscribe_to_price_updates)
        t1.start()
        t2.start()



def run_strat(strat: ProjectX_Strategy, token):
    strat.init_api(token)
    strat.run()

def validation_thread(auth_token, strategies: list[ProjectX_Strategy]):
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
    global_token = init_api()

    if UPDATE_CONTRACT_LIST:
        strat = ProjectX_Strategy(ASSETS[0])
        strat.init_api(global_token)
        data = strat.get_assets()
        data = pd.DataFrame(data)
        data.to_csv("contracts.csv")
        print("Contract list updated successfully!!")
    elif SHOW_ACCOUNTS:
        strat = ProjectX_Strategy(ASSETS[0])
        strat.set_token(global_token)
        print(get_account_id(strat.auth_token, show=True))

    else:
        threads = []
        strats = []
        for asset_pair in ASSETS:
            strats.append(ProjectX_Strategy(asset_pair))
        
        v_thread = threading.Thread(
            target = validation_thread,
            args = (global_token, strats,),
            daemon=True
        )
        v_thread.start()

        for strat in strats:
            t = threading.Thread(
                target=run_strat,
                args=(strat, global_token,),
                daemon=True
            )
            t.start()
            threads.append(t)



    while True:
        sleep(5)
            