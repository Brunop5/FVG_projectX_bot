import pandas as pd
import os
import time
import requests
from datetime import datetime, timedelta
from signalrcore.hub_connection_builder import HubConnectionBuilder
import logging



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

def get_account_id(token):
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

    return accounts[0]["id"]



def _map_timeframe_to_unit(timeframe):
    """
    Map timeframe string to TopStepX API unit and unitNumber.
    Examples: "30min" -> (3, 30), "1h" -> (2, 1), "4h" -> (2, 4), "1d" -> (1, 1)
    
    Unit codes (based on common API patterns):
    1 = Days
    2 = Hours  
    3 = Minutes
    """
    timeframe = timeframe.lower().strip()
    
    if 'd' in timeframe or 'day' in timeframe:
        # Days
        number = int(''.join(filter(str.isdigit, timeframe)) or 1)
        return (1, number)
    elif 'h' in timeframe or 'hour' in timeframe:
        # Hours
        number = int(''.join(filter(str.isdigit, timeframe)) or 1)
        return (2, number)
    elif 'min' in timeframe or 'm' in timeframe:
        # Minutes
        number = int(''.join(filter(str.isdigit, timeframe)) or 1)
        return (3, number)
    else:
        # Default to minutes
        number = int(''.join(filter(str.isdigit, timeframe)) or 30)
        return (3, number)

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
    
    # Calculate endTime (current time) and startTime (based on num_bars)
    end_time = datetime.utcnow()
    
    # Calculate start time based on timeframe and num_bars
    if unit == 1:  # Days
        start_time = end_time - timedelta(days=num_bars * unit_number)
    elif unit == 2:  # Hours
        start_time = end_time - timedelta(hours=num_bars * unit_number)
    else:  # Minutes
        start_time = end_time - timedelta(minutes=num_bars * unit_number)
    
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
            print(data)
            
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
                
                # Optional: reorder columns
                df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
                
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
    

def load_data(asset):
    """
    If asset data exist and are not older than 35 mins, it returns them as pandas df.
    otherwise returns None
    """
    path = f"{asset}.csv"
    if os.path.exists(path):
        df = pd.read_csv(f"{asset}.csv")
        if int(time.time() * 1000) - df["timestamp"].iloc[-1] > 35 * 1000: # if "timestamp" column is in ms
            return None
        else:
            return df

    else:
        return None


def stream_market_data(token: str, contract_id: str, callback):
    """
    Connects to ProjectX market hub and streams last price updates for a specific contract.
    
    :param token: Bearer JWT token
    :param contract_id: Contract ID string, e.g., 'CON.F.US.RTY.H25'
    :param callback: Function to call on every payload. Signature: callback(payload)
    """

    market_hub_url = f"https://rtc.topstepx.com/hubs/market?access_token={token}"

    # Build the hub connection
    hub_connection = HubConnectionBuilder()\
        .with_url(market_hub_url, options={"access_token_factory": lambda: token})\
        .configure_logging(logging.INFO)\
        .with_automatic_reconnect({
            "type": "raw",
            "keep_alive_interval": 10,
            "reconnect_interval": 5
        })\
        .build()

    # Handler for incoming GatewayQuote events
    def handle_event(payload):
        """
        SignalR sends the payload as a dict or a list [contract_id, dict].
        """
        data = None
        if isinstance(payload, list) and len(payload) >= 2 and isinstance(payload[1], dict):
            data = payload[1]
        elif isinstance(payload, dict):
            data = payload

        if data is not None:
            callback(data)
        else:
            print("Warning: unexpected payload format:", payload)

    hub_connection.on("GatewayQuote", handle_event)

    # Start the connection
    hub_connection.start()
    time.sleep(1)  # Give it a moment to establish

    # Subscribe to contract quotes
    hub_connection.send("SubscribeContractQuotes", [contract_id])
    print(f"Subscribed to {contract_id} market quotes.")

    try:
        # Keep running indefinitely
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        # Unsubscribe and close
        hub_connection.send("UnsubscribeContractQuotes", [contract_id])
        hub_connection.stop()
        print("Connection closed.")
