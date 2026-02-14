import pandas as pd
import os
import time
import requests
from datetime import datetime, timedelta, timezone
import logging
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
# Massive API is accessed via direct HTTP requests

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

def fetch_data(asset, timeframe, num_bars, auth_token=None, live=False, include_partial_bar=False):

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
        "includePartialBar": include_partial_bar
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
                return None  # Empty DataFrame if no data
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
    print(data)
    
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
    If asset data exist and are not older than 35 seconds, it returns them as pandas df.
    otherwise returns None
    """
    path = f"{asset[3:]}-{timeframe}.csv"
    if os.path.exists(path):
        df = pd.read_csv(path)
        if int(time.time() * 1000) - df["timestamp"].iloc[-1] > 35 * 1000: # if "timestamp" column is in ms
            return None
        else:
            return df

    else:
        return None


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

# Mapping from futures contracts (base IDs from contracts.csv) to Polygon.io tickers
# Format: 
#   - Forex: C:XXXYYY (e.g., C:EURUSD, C:GBPUSD)
#   - Commodities: C:XXXUSD (e.g., C:XAUUSD for gold, C:XAGUSD for silver)
#   - Indices: I:XXX (e.g., I:SPX for S&P 500, I:NDX for NASDAQ)
CONTRACTS_TO_POLYGON = {
    # ========== FOREX ==========
    # Euro
    "CON.F.US.EEU": "C:EURUSD",   # E-mini Euro FX -> EUR/USD
    "CON.F.US.EU6": "C:EURUSD",    # Euro FX (Globex) -> EUR/USD
    "CON.F.US.M6E": "C:EURUSD",   # E-Micro EUR/USD -> EUR/USD
    
    # British Pound
    "CON.F.US.BP6": "C:GBPUSD",   # British Pound -> GBP/USD
    "CON.F.US.M6B": "C:GBPUSD",   # E-Micro GBP/USD -> GBP/USD
    
    # Japanese Yen
    "CON.F.US.JY6": "C:USDJPY",   # Japanese Yen -> USD/JPY
    
    # Canadian Dollar
    "CON.F.US.CA6": "C:USDCAD",   # Canadian Dollar -> USD/CAD
    
    # Australian Dollar
    "CON.F.US.DA6": "C:AUDUSD",   # Australian Dollar -> AUD/USD
    "CON.F.US.M6A": "C:AUDUSD",   # E-Micro AUD/USD -> AUD/USD
    
    # New Zealand Dollar
    "CON.F.US.NE6": "C:NZDUSD",   # New Zealand Dollar -> NZD/USD
    
    # Swiss Franc
    "CON.F.US.SF6": "C:USDCHF",   # Swiss Franc -> USD/CHF
    
    # Mexican Peso
    "CON.F.US.MX6": "C:USDMXN",   # Mexican Peso -> USD/MXN
    
    # ========== COMMODITIES ==========
    # Gold
    "CON.F.US.MGC": "C:XAUUSD",   # Micro Gold -> Gold (XAU/USD)
    "CON.F.US.GCE": "C:XAUUSD",   # Gold (Globex) -> Gold (XAU/USD)
    
    # Silver
    "CON.F.US.SIL": "C:XAGUSD",   # Micro Silver -> Silver (XAG/USD)
    "CON.F.US.SIE": "C:XAGUSD",   # Silver (Globex) -> Silver (XAG/USD)
    
    # Platinum
    "CON.F.US.PLE": "C:XPTUSD",   # Platinum -> Platinum (XPT/USD)
    
    # ========== INDICES ==========
    # S&P 500
    "CON.F.US.MES": "I:SPX",      # Micro E-mini S&P 500 -> S&P 500 Index
    "CON.F.US.EP": "I:SPX",       # E-Mini S&P 500 -> S&P 500 Index
    
    # NASDAQ-100
    "CON.F.US.MNQ": "I:NDX",      # Micro E-mini Nasdaq-100 -> NASDAQ-100 Index
    "CON.F.US.ENQ": "I:NDX",      # E-mini NASDAQ-100 -> NASDAQ-100 Index
    
    # Dow Jones
    "CON.F.US.YM": "I:DJI",       # E-mini Dow -> Dow Jones Industrial Average
    "CON.F.US.MYM": "I:DJI",      # Micro E-mini Dow -> Dow Jones Industrial Average
    
    # Russell 2000
    "CON.F.US.M2K": "I:RUT",      # Micro E-mini Russell 2000 -> Russell 2000 Index
    "CON.F.US.RTY": "I:RUT",      # E-mini Russell 2000 -> Russell 2000 Index
    
    # Nikkei 225
    "CON.F.US.NKD": "I:N225",     # Nikkei 225 -> Nikkei 225 Index
}

def get_polygon_ticker(futures_contract_id):
    """
    Get Polygon.io ticker for a futures contract (forex, commodity, or index).
    Returns None if no mapping exists.
    
    Returns:
        tuple: (polygon_ticker, asset_type) where asset_type is "forex", "commodity", or "index"
               Returns (None, None) if no mapping exists
    """
    # Extract base contract ID (remove month/year suffix like .H26, .G26)
    # Format: CON.F.US.EU6.H26 -> CON.F.US.EU6
    parts = futures_contract_id.split('.')
    if len(parts) >= 4:
        base_id = '.'.join(parts[:4])  # CON.F.US.EU6
        polygon_ticker = CONTRACTS_TO_POLYGON.get(base_id)
        if polygon_ticker:
            # Determine asset type based on ticker prefix
            if polygon_ticker.startswith("C:X"):
                asset_type = "commodity"
            elif polygon_ticker.startswith("C:"):
                asset_type = "forex"
            elif polygon_ticker.startswith("I:"):
                asset_type = "index"
            else:
                asset_type = "unknown"
            return (polygon_ticker, asset_type)
    return (None, None)

def get_polygon_forex_ticker(futures_contract_id):
    """
    Legacy helper kept for compatibility – simply forwards to get_polygon_ticker.
    """
    ticker, _ = get_polygon_ticker(futures_contract_id)
    return ticker


def _get_massive_api_key(explicit_key: str | None = None) -> str | None:
    """
    Resolve the Massive API key.
    Prefer an explicit key, then MASSIVE_API_KEY env, then POLYGON_API_KEY env (for compatibility).
    """
    if explicit_key:
        return explicit_key
    key = os.getenv("MASSIVE_API_KEY")
    if key:
        return key
    # Backwards compatibility – allow existing POLYGON_API_KEY env var
    return os.getenv("POLYGON_API_KEY")


def fetch_polygon_data(ticker, multiplier, timespan, from_date, to_date, api_key, sort="asc", limit=50000):
    """
    Fetch historical data from Massive API using direct HTTP requests (forex, commodities, or indices).
    
    Based on: https://massive.com/docs/rest/forex/aggregates/custom-bars
    
    Args:
        ticker: Massive ticker (e.g., "C:EURUSD", "C:XAUUSD", "I:SPX")
        multiplier: Size of the timespan multiplier (e.g., 15 for 15-minute bars)
        timespan: Timespan (minute, hour, day, week, month, quarter, year)
        from_date: Start date (YYYY-MM-DD)
        to_date: End date (YYYY-MM-DD)
        api_key: API key (if None, will try to get from env vars)
        sort: Sort order ("asc" or "desc", default "asc")
        limit: Maximum number of base aggregates per request (default 50000)
    
    Returns:
        pandas.DataFrame with columns: timestamp, open, high, low, close, volume
        Returns None if error
    """
    # Resolve API key
    resolved_key = _get_massive_api_key(api_key)
    if not resolved_key:
        print("⚠️  Error: Massive API key required. Set MASSIVE_API_KEY or POLYGON_API_KEY environment variable.")
        return None

    base_url = "https://api.massive.com"
    endpoint = f"/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from_date}/{to_date}"
    url = f"{base_url}{endpoint}"
    
    params = {
        "adjusted": "true",
        "sort": sort,
        "limit": limit,
        "apiKey": resolved_key
    }
    
    all_results = []
    max_pagination_requests = 1000  # Safety limit to prevent infinite loops
    pagination_count = 0
    max_retries = 4  # 4 retries = 5 total attempts
    # Retry delays for 429 errors: 30s, 1min (60s), 2min (120s), 4min (240s)
    retry_delays_429 = [30, 60, 120, 240]
    request_delay = 10  # 10 seconds between each request
    
    def make_request_with_retry(request_url, request_params, is_first_request=False):
        """Make a request with retry logic for rate limiting"""
        # Wait 10 seconds before each request (except the very first one if specified)
        if not is_first_request:
            time.sleep(request_delay)
        
        for attempt in range(max_retries):
            try:
                response = requests.get(request_url, params=request_params, timeout=30)
                
                # Handle rate limiting (429)
                if response.status_code == 429:
                    if attempt < max_retries - 1:
                        wait_time = retry_delays_429[attempt]
                        wait_minutes = wait_time // 60
                        wait_seconds = wait_time % 60
                        if wait_minutes > 0:
                            wait_str = f"{wait_minutes}m {wait_seconds}s"
                        else:
                            wait_str = f"{wait_seconds}s"
                        print(f"\n   ⚠️  Rate limited (429). Waiting {wait_str} before retry {attempt + 1}/{max_retries}...", end=" ", flush=True)
                        time.sleep(wait_time)
                        continue
                    else:
                        response.raise_for_status()
                
                # Handle forbidden (403) - likely permission issue
                if response.status_code == 403:
                    error_msg = f"403 Forbidden - Your API key may not have access to {ticker}. Check your subscription plan."
                    print(f"\n   ⚠️  {error_msg}")
                    return None, None
                
                response.raise_for_status()
                return response, None
                
            except requests.exceptions.RequestException as e:
                # Check if it's a 429 error (rate limiting)
                is_429 = "429" in str(e) or (hasattr(e, 'response') and e.response is not None and e.response.status_code == 429)
                if attempt < max_retries - 1 and is_429:
                    wait_time = retry_delays_429[attempt]
                    wait_minutes = wait_time // 60
                    wait_seconds = wait_time % 60
                    if wait_minutes > 0:
                        wait_str = f"{wait_minutes}m {wait_seconds}s"
                    else:
                        wait_str = f"{wait_seconds}s"
                    print(f"\n   ⚠️  Rate limited. Waiting {wait_str} before retry {attempt + 1}/{max_retries}...", end=" ", flush=True)
                    time.sleep(wait_time)
                    continue
                else:
                    raise
        
        return None, "Max retries exceeded"
    
    try:
        # Make initial request
        current_url = url
        current_params = params
        is_first_request = True
        
        while pagination_count < max_pagination_requests:
            response, error = make_request_with_retry(current_url, current_params, is_first_request=is_first_request)
            is_first_request = False  # After first request, always wait 10s
            
            if error:
                print(f"⚠️  {error}")
                return None
            if response is None:
                return None
            
            data = response.json()
            
            status = data.get("status")
            
            # Check for errors
            if status not in ["OK", "DELAYED"]:
                error_msg = data.get("error", f"Unknown status: {status}")
                print(f"⚠️  Massive API error: {error_msg}")
                return None
            
            # Collect results from this page
            results = data.get("results", [])
            if results:
                all_results.extend(results)
                print(f"   Got {len(results):,} bars (total: {len(all_results):,})...", end=" ", flush=True)
            
            # Check for next_url - continue pagination as long as there's a next_url
            # Status "OK" just means this page is ready, not that all data is complete
            next_url = data.get("next_url")
            if not next_url:
                # No more pages available - we're done
                if status == "OK":
                    print("Complete!")
                break
            
            # Continue with next_url regardless of status (OK or DELAYED)
            # Status "OK" with a next_url means there's more data to fetch
            
            # Ensure API key is included in next_url
            # Parse the URL and add apiKey if not present
            parsed = urlparse(next_url)
            query_params = parse_qs(parsed.query)
            
            # Add API key if not already present
            if 'apiKey' not in query_params:
                query_params['apiKey'] = [resolved_key]
            
            # Reconstruct URL with API key
            new_query = urlencode(query_params, doseq=True)
            current_url = urlunparse((
                parsed.scheme,
                parsed.netloc,
                parsed.path,
                parsed.params,
                new_query,
                parsed.fragment
            ))
            current_params = {}  # All params are now in the URL
            pagination_count += 1
            
            # Note: 10 second delay is handled in make_request_with_retry before each request
        
        if pagination_count >= max_pagination_requests:
            print(f"\n⚠️  Warning: Reached maximum pagination limit ({max_pagination_requests}). Data may be incomplete.")
        
        if not all_results:
            print(f"⚠️  No data returned from Massive for {ticker}")
            return None
        
        # Check if we got all the data by verifying date range
        if all_results:
            first_timestamp = all_results[0]['t']
            last_timestamp = all_results[-1]['t']
            first_date = pd.to_datetime(first_timestamp, unit='ms', utc=True)
            last_date = pd.to_datetime(last_timestamp, unit='ms', utc=True)
            requested_start = pd.to_datetime(from_date, utc=True)
            requested_end = pd.to_datetime(to_date, utc=True)
            
            # Check if we got data from the full requested range
            if first_date > requested_start or last_date < requested_end:
                print(f"\n⚠️  Warning: Data range incomplete!")
                print(f"   Requested: {requested_start.strftime('%Y-%m-%d')} to {requested_end.strftime('%Y-%m-%d')}")
                print(f"   Received: {first_date.strftime('%Y-%m-%d')} to {last_date.strftime('%Y-%m-%d')}")
                print(f"   Missing: {requested_start.strftime('%Y-%m-%d')} to {first_date.strftime('%Y-%m-%d')} and/or {last_date.strftime('%Y-%m-%d')} to {requested_end.strftime('%Y-%m-%d')}")
        
        # Convert to DataFrame
        df_data = []
        volume_values = []  # Track volume values for validation
        for bar in all_results:
            vol = bar.get('v', 0)
            volume_values.append(vol)
            df_data.append({
                'timestamp': bar['t'],  # Already in milliseconds
                'open': bar['o'],
                'high': bar['h'],
                'low': bar['l'],
                'close': bar['c'],
                'volume': vol
            })
        
        df = pd.DataFrame(df_data)
        
        # Validate volume data quality
        if volume_values:
            import numpy as np
            volumes = np.array(volume_values)
            unique_volumes = np.unique(volumes)
            max_vol = np.max(volumes)
            min_vol = np.min(volumes)
            mean_vol = np.mean(volumes)
            
            # Check for suspicious volume patterns (very low, low variety)
            if max_vol < 100 and len(unique_volumes) < 20:
                print(f"\n   ⚠️  WARNING: Suspicious volume data detected!")
                print(f"      Volume range: {min_vol:.2f} - {max_vol:.2f}")
                print(f"      Unique values: {len(unique_volumes)}")
                print(f"      Mean volume: {mean_vol:.2f}")
                print(f"      This may indicate synthetic/estimated volume data from the API.")
                print(f"      For forex/commodities, volume data quality can vary, especially for older periods.")
        
        # Ensure columns are in correct order
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        
        # Sort by timestamp: oldest first
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        return df
        
    except requests.exceptions.RequestException as e:
        print(f"⚠️  Error fetching data from Massive for {ticker}: {e}")
        return None
    except Exception as e:
        print(f"⚠️  Error processing Massive data for {ticker}: {e}")
        return None


def fetch_polygon_forex_data(forex_ticker, multiplier, timespan, from_date, to_date, api_key, sort="asc", limit=50000):
    """
    Legacy function for backward compatibility.
    Fetch historical data from Massive API (works for forex, commodities, and indices).
    """
    return fetch_polygon_data(forex_ticker, multiplier, timespan, from_date, to_date, api_key, sort, limit)


def _parse_timeframe_to_polygon(timeframe):
    """
    Parse timeframe string (e.g., "5min", "1h", "1d") to Polygon multiplier and timespan.
    
    Returns:
        tuple: (multiplier, timespan) or (None, None) if invalid
    """
    timeframe_lower = timeframe.lower()
    
    if 'min' in timeframe_lower:
        multiplier = int(''.join(filter(str.isdigit, timeframe)) or '1')
        return (multiplier, "minute")
    elif 'h' in timeframe_lower:
        multiplier = int(''.join(filter(str.isdigit, timeframe)) or '1')
        return (multiplier, "hour")
    elif 'd' in timeframe_lower:
        multiplier = int(''.join(filter(str.isdigit, timeframe)) or '1')
        return (multiplier, "day")
    elif 'w' in timeframe_lower or 'week' in timeframe_lower:
        multiplier = int(''.join(filter(str.isdigit, timeframe)) or '1')
        return (multiplier, "week")
    elif 'mo' in timeframe_lower or 'month' in timeframe_lower:
        multiplier = int(''.join(filter(str.isdigit, timeframe)) or '1')
        return (multiplier, "month")
    else:
        return (None, None)


def gather_historical_data(contracts_list, timeframes=None, years=5, auth_token=None):
    """
    Gather historical data for multiple contracts and timeframes using Massive API.
    This function uses Massive API to fetch forex, commodity, and index data.
    
    Args:
        contracts_list: List of tuples (contract_id, price) or list of contract_id strings
                       e.g., [("CON.F.US.EU6.H26", 0.74), ...] or ["CON.F.US.EU6.H26", ...]
        timeframes: List of timeframes to fetch (default: ["5min", "15min", "30min", "1h", "2h", "4h", "1d"])
        years: Number of years of historical data to fetch (default: 5)
        auth_token: Not used (kept for compatibility)
    
    Returns:
        Dictionary with contract_id as key and nested dict of timeframes as values
    
    NOTE:
        - Supported asset types: forex, commodities (gold, silver, platinum), and indices (S&P 500, NASDAQ, Dow, etc.)
        - Unsupported futures contracts (e.g., agricultural, energy, bonds) are skipped
        - Massive API key must be set in MASSIVE_API_KEY or POLYGON_API_KEY environment variable
        - Massive API has pagination support via next_url, so large date ranges are handled automatically
        - Data is fetched in chunks to handle large date ranges efficiently
    """
    # Get Massive API key from environment
    massive_api_key = _get_massive_api_key()
    if not massive_api_key:
        raise ValueError("MASSIVE_API_KEY or POLYGON_API_KEY environment variable is required. Set it with: export MASSIVE_API_KEY=your_key")
    
    if timeframes is None:
        timeframes = ["5min", "15min", "30min", "1h", "2h", "4h", "1d"]
    
    # Create data directory if it doesn't exist (don't clear existing data)
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    
    # Load contracts.csv to get asset names
    contracts_df = pd.read_csv("contracts.csv")
    contract_name_map = {}
    for _, row in contracts_df.iterrows():
        contract_name_map[row['id']] = row['name']
    
    # Extract contract IDs from tuples if needed
    contract_ids = []
    for item in contracts_list:
        if isinstance(item, tuple):
            contract_ids.append(item[0])
        else:
            contract_ids.append(item)
    
    results = {}
    
    for contract_id in contract_ids:
        print(f"\n{'='*60}")
        print(f"📊 Gathering data for {contract_id}")
        print(f"{'='*60}")
        
        # Check if this contract is supported by Massive API
        polygon_ticker, asset_type = get_polygon_ticker(contract_id)
        if not polygon_ticker:
            print(f"   ⚠️  Skipping {contract_id} - no Massive API mapping found")
            print(f"   ⚠️  Supported: forex, commodities (gold/silver/platinum), and indices (S&P 500, NASDAQ, Dow, etc.)")
            continue
        
        results[contract_id] = {}
        
        asset_type_display = {
            "forex": "forex pair",
            "commodity": "commodity",
            "index": "index"
        }.get(asset_type, "asset")
        
        print(f"   📈 Using Massive API: {polygon_ticker} ({asset_type_display})")
        
        # Get asset name from contracts.csv, fallback to contract_id if not found
        asset_name = contract_name_map.get(contract_id, contract_id)
        # Clean asset name for folder name (remove special characters, spaces)
        safe_asset_name = "".join(c for c in asset_name if c.isalnum() or c in (' ', '-', '_')).strip()
        safe_asset_name = safe_asset_name.replace(' ', '_')
        
        # Create directory structure: data/asset_name/
        contract_dir = os.path.join("data", safe_asset_name)
        os.makedirs(contract_dir, exist_ok=True)
        print(f"   📁 Saving to: {contract_dir}/ ({asset_name})")
        
        # Calculate date range
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=years * 365)
        
        for timeframe in timeframes:
            print(f"\n⏱️  Fetching {timeframe} data...")
            
            # Parse timeframe to Polygon format
            multiplier, timespan = _parse_timeframe_to_polygon(timeframe)
            if multiplier is None or timespan is None:
                print(f"   ⚠️  Invalid timeframe: {timeframe}. Skipping.")
                continue
            
            # Format dates for Massive API (YYYY-MM-DD)
            from_date_str = start_date.strftime("%Y-%m-%d")
            to_date_str = end_date.strftime("%Y-%m-%d")
            
            print(f"   Date range: {from_date_str} to {to_date_str} ({years} years)")
            print(f"   Fetching data...", end=" ", flush=True)
            
            # Make single request - API handles pagination automatically via next_url
            combined_df = fetch_polygon_data(
                polygon_ticker, 
                multiplier, 
                timespan, 
                from_date_str, 
                to_date_str,
                massive_api_key
            )
            
            if combined_df is None or combined_df.empty:
                print(f"\n   ⚠️  No data fetched for {timeframe}")
                continue
            
            # Remove duplicates (in case of any overlap from pagination)
            combined_df = combined_df.drop_duplicates(subset=['timestamp'], keep='first')
            combined_df = combined_df.sort_values('timestamp').reset_index(drop=True)
            
            print(f"   ✅ Got {len(combined_df):,} bars total from Massive API")
            
            # Ensure timestamp is int64
            combined_df['timestamp'] = combined_df['timestamp'].astype('int64')
            
            # Ensure columns are in correct order
            combined_df = combined_df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            
            # Save to CSV - merge with existing data if file exists
            filename = f"{timeframe}.csv"
            filepath = os.path.join(contract_dir, filename)
            
            if os.path.exists(filepath):
                # Load existing data and merge
                print(f"   📂 Found existing data file, merging...", end=" ", flush=True)
                existing_df = pd.read_csv(filepath)
                
                # Ensure timestamp is int64 for comparison
                if 'timestamp' in existing_df.columns:
                    existing_df['timestamp'] = existing_df['timestamp'].astype('int64')
                
                # Combine new and existing data
                merged_df = pd.concat([existing_df, combined_df], ignore_index=True)
                
                # Remove duplicates based on timestamp, keeping the last occurrence (newer data takes precedence)
                merged_df = merged_df.drop_duplicates(subset=['timestamp'], keep='last')
                
                # Sort by timestamp
                merged_df = merged_df.sort_values('timestamp').reset_index(drop=True)
                
                # Update combined_df to the merged version
                combined_df = merged_df
                print(f"Merged! Total bars: {len(combined_df):,}")
            
            # Save to CSV
            combined_df.to_csv(filepath, index=False)
            print(f"   ✅ Saved {len(combined_df):,} bars to {filepath}")
            oldest_dt = pd.to_datetime(combined_df['timestamp'].iloc[0], unit='ms', utc=True)
            newest_dt = pd.to_datetime(combined_df['timestamp'].iloc[-1], unit='ms', utc=True)
            days_span = (newest_dt - oldest_dt).days
            print(f"   📅 Date range: {oldest_dt.strftime('%Y-%m-%d %H:%M:%S UTC')} to {newest_dt.strftime('%Y-%m-%d %H:%M:%S UTC')} ({days_span} days)")
            
            results[contract_id][timeframe] = combined_df
            
            # Delay between timeframes to avoid rate limiting
            if timeframe != timeframes[-1]:  # Don't delay after the last timeframe
                time.sleep(1.0)
        
        print(f"\n✅ Completed gathering data for {contract_id}")
        
        # Delay between contracts to avoid rate limiting
        if contract_id != contract_ids[-1]:  # Don't delay after the last contract
            time.sleep(2.0)
    
    return results
