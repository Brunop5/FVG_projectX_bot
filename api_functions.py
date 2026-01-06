import pandas as pd
import os
import time
import requests
from datetime import datetime, timedelta, timezone
from signalrcore.hub_connection_builder import HubConnectionBuilder
import logging

TIMEFRAME_SECONDS = {
    "1s": 1,
    "5s": 5,
    "30s": 30,
    "1min": 60,
    "5min": 300,
    "15min": 900,
    "30min": 1800,
    "1h": 3600,
    "4h": 14400,
    "1d": 86400,
}

def login_to_api(user_name, api_key):
    url = "https://api.topstepx.com/api/Auth/loginKey"
    
    headers = {
        'accept': 'text/plain',
        'Content-Type': 'application/json'
    }
    
    payload = {
        "userName": user_name,
        "apiKey": api_key
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        #print(response.text, response.status_code)
        
        if response.status_code == 200:
            # Success - token is typically returned in response
            token = response.text if response.text else None
            return response.json()
        else:
            # Authentication failed
            return {
                'success': False,
                'token': None,
                'message': f'Authentication failed: {response.text}',
                'status_code': response.status_code
            }
    
    except requests.exceptions.RequestException as e:
        return {
            'success': False,
            'token': None,
            'message': f'Connection error: {str(e)}',
            'status_code': None
        }

def get_account_id(token, account_name=None, show=False):
    url = "https://api.topstepx.com/api/Account/search"

    headers = {
        "Authorization": f"Bearer {token}",
        "accept": "text/plain",
        "Content-Type": "application/json"
    }

    payload = {
        "onlyActiveAccounts": True
    }

    response = requests.post(url, json=payload, headers=headers)
    response.raise_for_status()

    data = response.json()
    # The API returns a list of accounts under "accounts"
    accounts = data.get("accounts")
    if not accounts:
        raise Exception("No active accounts found")

    if show:
        return accounts
    
    return [acc for acc in accounts if acc["name"] == account_name][0]["id"]

def _map_timeframe_to_unit(timeframe: str):
    """
    Map timeframe string to TopStepX API unit and unitNumber.

    Examples:
        "30s"    -> (1, 30)
        "5min"   -> (2, 5)
        "240min" -> (3, 4)
        "1h"     -> (3, 1)
        "4h"     -> (3, 4)
        "1d"     -> (4, 1)
        "1w"     -> (5, 1)
        "1M"     -> (6, 1)
    """

    tf = timeframe.strip()
    tf_lower = tf.lower()

    # Extract numeric part (default = 1)
    number_str = ''.join(filter(str.isdigit, tf))
    number = int(number_str) if number_str else 1

    # Seconds
    if tf_lower.endswith('s'):
        return 1, number

    # Minutes (explicit)
    if 'min' in tf_lower or (tf_lower.endswith('m') and not tf.endswith('M')):
        # Normalize minutes to hours if divisible by 60
        if number % 60 == 0:
            return 3, number // 60  # Hours
        return 2, number  # Minutes

    # Hours
    if 'h' in tf_lower:
        return 3, number

    # Days
    if 'd' in tf_lower:
        return 4, number

    # Weeks
    if 'w' in tf_lower:
        return 5, number

    # Months (capital M or 'mo')
    if tf.endswith('M') or 'mo' in tf_lower:
        return 6, number

    # Fallback (minutes)
    return 2, number

def fetch_data(asset, timeframe, num_bars, auth_token=None, live=False):

    """
    Fetch historical bar data from TopStepX API.
    
    Args:
        asset: Contract ID (e.g., "CON.F.US.RTY.Z24" or asset name that maps to contract)
        timeframe: Timeframe string (e.g., "30min", "1h", "4h", "1d")
        num_bars: Number of bars to retrieve
        auth_token: Authentication token from test_api_connection (optional if not required)
        live: Whether to fetch live data (default: False)
    
    Returns:
        pandas.DataFrame: DataFrame with OHLCV data, or None if error
    """
    url = "https://api.topstepx.com/api/History/retrieveBars"
    
    headers = {
        'accept': 'text/plain',
        'Content-Type': 'application/json'
    }
    
    # Add authorization token if provided
    if auth_token:
        headers['Authorization'] = f'Bearer {auth_token}'
    
    # Map timeframe to unit and unitNumber
    unit, unit_number = _map_timeframe_to_unit(timeframe)    

    end_time = datetime.utcnow()

    if unit == 1:  # Seconds
        delta = timedelta(seconds=unit_number * num_bars*200)

    elif unit == 2:  # Minutes
        delta = timedelta(minutes=unit_number * max([num_bars, 4320]))

    elif unit == 3:  # Hours
        delta = timedelta(hours=unit_number * max([num_bars, 72]))

    elif unit == 4:  # Days
        delta = timedelta(days=unit_number * max([num_bars, 3]))

    else:
        raise ValueError("Unsupported unit")

    start_time = end_time - delta

    
    # Format times in ISO 8601 format
    start_time_str = start_time.strftime('%Y-%m-%dT%H:%M:%SZ')
    end_time_str = end_time.strftime('%Y-%m-%dT%H:%M:%SZ')

    payload = {
        "contractId": asset,
        "live": live,
        "startTime": start_time_str,
        "endTime": end_time_str,
        "unit": unit,
        "unitNumber": unit_number,
        "limit": num_bars,
        "includePartialBar": False
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        if response.status_code == 200:
            # Parse response - assuming it returns JSON array of bars
            data = response.json()["bars"]            
            # Convert to DataFrame
            # Adjust column names based on actual API response structure
            if isinstance(data, list) and len(data) > 0:
                df = pd.DataFrame(data)
    
                # Rename columns
                df = df.rename(columns={
                    't': 'timestamp',
                    'o': 'open',
                    'h': 'high',
                    'l': 'low',
                    'c': 'close',
                    'v': 'volume'
                })
                
                # Convert timestamp to datetime if needed
                df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
                df['timestamp'] = df['timestamp'].astype('int64') // 1_000_000
                
                # Optional: reorder columns
                df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
                df = df.iloc[::-1].reset_index(drop=True)
                return df
            else:
                return pd.DataFrame()  # Empty DataFrame if no data
        else:
            print(f"Error fetching data: {response.status_code} - {response.text}")
            return None
    
    except requests.exceptions.RequestException as e:
        print(f"Request error: {str(e)}")
        return None
    except Exception as e:
        print(f"Error parsing response: {str(e)}")
        return None

def fetch_data_(ugh, eeehm, lol, tok=None, l=False):
    """Fetch 100 bars of 30m BTC-USDT perpetual futures data from Binance Futures."""
    url = "https://fapi.binance.com/fapi/v1/continuousKlines"
    params = {
        "pair": "BTCUSDT",
        "contractType": "PERPETUAL",
        "interval": "1m",
        "limit": 100  # number of candles (max ~1500 allowed)
    }
    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    data = response.json()
    
    # Extract only first 6 items from each bar
    rows = [
        {
            "timestamp": int(bar[0]),
            "open": float(bar[1]),
            "high": float(bar[2]),
            "low": float(bar[3]),
            "close": float(bar[4]),
            "volume": float(bar[5])
        }
        for bar in data
    ]
    
    df = pd.DataFrame(rows)
    
    # Optional: convert timestamp to datetime
    #df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df
    
def load_data(asset, timeframe):
    """
    If asset data exist and are not older than 35 mins, it returns them as pandas df.
    otherwise returns None
    """
    path = f"{asset}-{timeframe}.csv"
    if os.path.exists(path):
        df = pd.read_csv(path)
        if int(time.time() * 1000) - df["timestamp"].iloc[-1] > 35 * 1000: # if "timestamp" column is in ms
            return None
        else:
            return df

    else:
        return None

def stream_market_data(token: str, contract_id: str, callback, user_name, api_key):
    response = login_to_api(user_name, api_key)
    if response["success"]:
        print("new_token")
        token = response["token"]
    market_hub_url = f"https://rtc.topstepx.com/hubs/market?access_token={token}"

    hub_connection = HubConnectionBuilder()\
        .with_url(
            market_hub_url,
            options={
                "access_token_factory": lambda: token
            }
        )\
        .configure_logging(logging.INFO)\
        .with_automatic_reconnect({
            "type": "raw",
            "keep_alive_interval": 10,
            "reconnect_interval": 5
        })\
        .build()

    # Event handler
    def handle_event(args):
        if isinstance(args, list) and len(args) >= 2:
            _, data = args
            callback(data)
        elif isinstance(args, dict):
            callback(args)
        else:
            print("Warning: unexpected payload format:", args)

    hub_connection.on("GatewayQuote", handle_event)

    # Subscribe when connection is open
    def on_open():
        print(f"Connected. Subscribing to {contract_id}")
        hub_connection.invoke("SubscribeContractQuotes", [contract_id])

    hub_connection.on_open(on_open)
    hub_connection.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        hub_connection.invoke("UnsubscribeContractQuotes", [contract_id])
        hub_connection.stop()
        print("Connection closed.")


def sleep_until_next_boundary(timeframe: str):

    tf_seconds = TIMEFRAME_SECONDS[timeframe]

    now = datetime.now(timezone.utc)
    now_ts = now.timestamp()

    # Next exact multiple of timeframe seconds
    next_boundary = ((now_ts // tf_seconds) + 1) * tf_seconds

    sleep_seconds = next_boundary - now_ts
    time.sleep(sleep_seconds)

def get_account_balance(account_id, auth_token):
    url = "https://api.topstepx.com/api/Account/search"

    headers = {
        "Authorization": f"Bearer {auth_token}",
        "accept": "text/plain",
        "Content-Type": "application/json"
    }

    payload = {
        "onlyActiveAccounts": True
    }

    response = requests.post(url, json=payload, headers=headers)
    response.raise_for_status()

    data = response.json()
    # The API returns a list of accounts under "accounts"
    accounts = data.get("accounts")
    if not accounts:
        raise Exception("No active accounts found")
    
    return float([acc for acc in accounts if acc["id"] == account_id][0]["balance"])

def validate_token(auth_token: str):
    url = "https://api.topstepx.com/api/Auth/validate"

    headers = {
        "accept": "text/plain",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {auth_token}"
    }

    try:
        response = requests.post(url, headers=headers, timeout=10)

        if response.status_code != 200:
            return {
                "success": False,
                "new_token": None,
                "status_code": response.status_code,
                "message": response.text
            }

        data = response.json()

        return data

    except requests.exceptions.RequestException as e:
        return {
            "success": False,
            "new_token": None,
            "status_code": None,
            "message": str(e)
        }
