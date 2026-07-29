import pandas as pd
import os
import time
import requests
import random
import threading
import socket
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

TOPSTEPX_MAX_CONCURRENCY = int(os.getenv("TOPSTEPX_MAX_CONCURRENCY", "4"))
_TOPSTEPX_SEMAPHORE = threading.BoundedSemaphore(TOPSTEPX_MAX_CONCURRENCY)
DEFAULT_TOPSTEPX_TIMEOUT = (5, 30)
TOPSTEPX_MAX_RETRY_SECONDS = int(os.getenv("TOPSTEPX_MAX_RETRY_SECONDS", "600"))
_TOPSTEPX_SESSION = requests.Session()
_TOPSTEPX_SESSION.mount(
    "https://",
    requests.adapters.HTTPAdapter(
        pool_connections=TOPSTEPX_MAX_CONCURRENCY,
        pool_maxsize=TOPSTEPX_MAX_CONCURRENCY,
        max_retries=0,
    ),
)

# Force IPv4 if IPv6 DNS/resolution is unreliable (common on some VMs).
try:
    requests.packages.urllib3.util.connection.allowed_gai_family = lambda: socket.AF_INET
except Exception:
    pass


def _seconds_until_sunday_18_et(now_et: datetime | None = None) -> float:
    tz_et = timezone(timedelta(hours=-5))
    now_et = now_et or datetime.now(tz_et)
    weekday = now_et.weekday()  # Mon=0 ... Sun=6
    if weekday == 6 and now_et.hour >= 18:
        return 0.0
    if weekday == 6:
        target = now_et.replace(hour=18, minute=0, second=0, microsecond=0)
    elif weekday == 5:
        target = (now_et + timedelta(days=1)).replace(
            hour=18, minute=0, second=0, microsecond=0
        )
    else:
        return 0.0
    return max(0.0, (target - now_et).total_seconds())


def _wait_before_retry_for_weekend(
    fallback_seconds: float = 10.0,
    max_wait_seconds: float | None = None,
) -> float:
    wait_seconds = _seconds_until_sunday_18_et()
    if wait_seconds > 0:
        wait_minutes = int(wait_seconds // 60)
        wait_rem = int(wait_seconds % 60)
        print(
            f"⏳ Weekend detected. Waiting {wait_minutes}m {wait_rem}s "
            "until Sunday 18:00 (UTC-5) before retrying."
        )
    else:
        wait_seconds = fallback_seconds
    if max_wait_seconds is not None:
        wait_seconds = min(wait_seconds, max_wait_seconds)
    if wait_seconds <= 0:
        return 0.0
    time.sleep(wait_seconds)
    return wait_seconds


def _compute_retry_delay(attempt: int, base_seconds: float = 5.0, max_seconds: float = 60.0) -> float:
    exponent = max(0, attempt - 1)
    delay = min(max_seconds, base_seconds * (2 ** exponent))
    jitter = random.uniform(0.0, delay * 0.2)
    return delay + jitter


def topstepx_post(url: str, headers: dict, payload: dict, timeout=DEFAULT_TOPSTEPX_TIMEOUT):
    with _TOPSTEPX_SEMAPHORE:
        return _TOPSTEPX_SESSION.post(url, headers=headers, json=payload, timeout=timeout)

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
        response = topstepx_post(url, headers=headers, payload=payload, timeout=10)
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

    response = topstepx_post(url, headers=headers, payload=payload, timeout=DEFAULT_TOPSTEPX_TIMEOUT)
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

def fetch_data(
    asset,
    timeframe,
    num_bars,
    auth_token=None,
    live=False,
    include_partial_bar=False,
    max_retry_seconds: float | None = TOPSTEPX_MAX_RETRY_SECONDS,
    _extra_lookback_days: int = 0,
):

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

    end_time = datetime.now(timezone.utc)

    if unit == 1:  # Seconds
        delta = timedelta(seconds=unit_number * num_bars*200)

    elif unit == 2:  # Minutes
        delta = timedelta(minutes=unit_number * num_bars)

    elif unit == 3:  # Hours
        delta = timedelta(hours=unit_number * num_bars)

    elif unit == 4:  # Days
        delta = timedelta(days=unit_number * num_bars)

    else:
        raise ValueError("Unsupported unit")

    # Extend window over weekend gaps so Sat/Sun/Mon still reach last session bars.
    # (Previously only Sun/Mon were covered — Sunday lookbacks often missed Friday.)
    weekday = end_time.weekday()  # Mon=0 ... Sun=6
    if weekday == 5:  # Saturday
        delta = delta + timedelta(days=2)
    elif weekday == 6:  # Sunday
        delta = delta + timedelta(days=3)
    elif weekday == 0:  # Monday
        delta = delta + timedelta(days=2)
    if _extra_lookback_days > 0:
        delta = delta + timedelta(days=_extra_lookback_days)
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
    
    retryable_statuses = {429, 500, 502, 503, 504, 520, 522, 524}
    attempt = 0
    retry_start = time.monotonic()
    if max_retry_seconds is not None and max_retry_seconds <= 0:
        max_retry_seconds = None

    def _retry_budget_exhausted() -> bool:
        if max_retry_seconds is None:
            return False
        return (time.monotonic() - retry_start) >= max_retry_seconds

    def _remaining_retry_seconds() -> float | None:
        if max_retry_seconds is None:
            return None
        remaining = max_retry_seconds - (time.monotonic() - retry_start)
        return max(0.0, remaining)

    while True:
        attempt += 1
        try:
            response = topstepx_post(url, headers=headers, payload=payload, timeout=DEFAULT_TOPSTEPX_TIMEOUT)
            if response.status_code == 200:
                try:
                    payload_json = response.json()
                except ValueError:
                    body_preview = (response.text or "").strip()
                    if len(body_preview) > 500:
                        body_preview = f"{body_preview[:500]}..."
                    delay = _compute_retry_delay(attempt)
                    print(
                        f"⚠️  Invalid JSON from TopStepX. Retrying in {int(delay)}s "
                        f"(attempt {attempt}). Body: {body_preview}"
                    )
                    if _retry_budget_exhausted():
                        print(
                            "⚠️  TopStepX retry budget exceeded; returning None to avoid hang."
                        )
                        return None
                    _wait_before_retry_for_weekend(
                        delay, max_wait_seconds=_remaining_retry_seconds()
                    )
                    continue

                # Parse response - assuming it returns JSON array of bars
                data = payload_json.get("bars")
                if data is None:
                    body_preview = (response.text or "").strip()
                    if len(body_preview) > 500:
                        body_preview = f"{body_preview[:500]}..."
                    delay = _compute_retry_delay(attempt)
                    print(
                        f"⚠️  Missing 'bars' in TopStepX response. Retrying in {int(delay)}s "
                        f"(attempt {attempt}). Body: {body_preview}"
                    )
                    if _retry_budget_exhausted():
                        print(
                            "⚠️  TopStepX retry budget exceeded; returning None to avoid hang."
                        )
                        return None
                    _wait_before_retry_for_weekend(
                        delay, max_wait_seconds=_remaining_retry_seconds()
                    )
                    continue

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
                    
                    # Convert timestamp to milliseconds since epoch (robust to s/ms inputs)
                    ts_series = df["timestamp"]
                    if pd.api.types.is_numeric_dtype(ts_series):
                        max_val = pd.Series(ts_series).max()
                        unit = "ms" if max_val > 1e12 else "s"
                        df["timestamp"] = pd.to_datetime(ts_series, unit=unit, utc=True)
                    else:
                        numeric_ts = pd.to_numeric(ts_series, errors="coerce")
                        if numeric_ts.notna().any():
                            max_val = numeric_ts.max()
                            unit = "ms" if max_val > 1e12 else "s"
                            df["timestamp"] = pd.to_datetime(numeric_ts, unit=unit, utc=True)
                        else:
                            df["timestamp"] = pd.to_datetime(ts_series, utc=True, errors="coerce")
                    df["timestamp"] = (df["timestamp"].astype("int64") // 1_000_000).where(
                        df["timestamp"].notna()
                    )
                    
                    # Optional: reorder columns
                    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
                    df = df.iloc[::-1].reset_index(drop=True)
                    return df
                else:
                    # Tiny/misaligned windows often miss the last session over weekends.
                    # Widen once (historical fetches only — skip live 1-bar price polls).
                    if _extra_lookback_days <= 0 and num_bars >= 5:
                        print(
                            f"⚠️  TopStepX returned empty bars "
                            f"(asset={asset}, tf={timeframe}, limit={num_bars}). "
                            "Retrying with +5 day lookback."
                        )
                        return fetch_data(
                            asset,
                            timeframe,
                            num_bars,
                            auth_token=auth_token,
                            live=live,
                            include_partial_bar=include_partial_bar,
                            max_retry_seconds=_remaining_retry_seconds(),
                            _extra_lookback_days=5,
                        )
                    # Live 1m price polls often get empty responses from TopStepX;
                    # keep that quiet. Still log for larger historical fetches.
                    if num_bars > 2:
                        print(
                            f"⚠️  TopStepX returned empty bars "
                            f"(asset={asset}, tf={timeframe}, limit={num_bars}, "
                            f"includePartialBar={include_partial_bar})."
                        )
                    return None  # Empty DataFrame if no data
            if response.status_code in retryable_statuses:
                body_preview = (response.text or "").strip()
                if len(body_preview) > 500:
                    body_preview = f"{body_preview[:500]}..."
                delay = _compute_retry_delay(attempt)
                if response.status_code == 429:
                    retry_after = response.headers.get("Retry-After")
                    if retry_after:
                        try:
                            delay = max(delay, float(retry_after))
                        except ValueError:
                            pass
                print(
                    f"⚠️  {response.status_code} from TopStepX. Retrying in {int(delay)}s "
                    f"(attempt {attempt}). Body: {body_preview}"
                )
                if _retry_budget_exhausted():
                    print(
                        "⚠️  TopStepX retry budget exceeded; returning None to avoid hang."
                    )
                    return None
                _wait_before_retry_for_weekend(
                    delay, max_wait_seconds=_remaining_retry_seconds()
                )
                continue
            print(f"Error fetching data: {response.status_code} - {response.text}")
            return None
    
        except requests.exceptions.RequestException as e:
            status_code = getattr(getattr(e, "response", None), "status_code", None)
            is_timeout = isinstance(e, requests.exceptions.Timeout)
            is_conn_error = isinstance(e, requests.exceptions.ConnectionError)
            if is_timeout or is_conn_error or status_code in retryable_statuses or "502" in str(e):
                delay = _compute_retry_delay(attempt)
                reason = "timeout" if is_timeout else "connection error" if is_conn_error else "request error"
                print(
                    f"⚠️  TopStepX {reason}: {e}. Retrying in {int(delay)}s "
                    f"(attempt {attempt})."
                )
                if _retry_budget_exhausted():
                    print(
                        "⚠️  TopStepX retry budget exceeded; returning None to avoid hang."
                    )
                    return None
                _wait_before_retry_for_weekend(
                    delay, max_wait_seconds=_remaining_retry_seconds()
                )
                continue
            print(f"Request error: {str(e)}")
            return None
        except Exception as e:
            print(f"Error parsing response: {str(e)}")
            return None

def load_data(asset, timeframe, data_dir=None):
    """
    If asset data exist and are not older than 35 seconds, it returns them as pandas df.
    otherwise returns None
    """
    path = f"{asset[3:]}-{timeframe}.csv"
    if data_dir:
        path = os.path.join(data_dir, path)
    if os.path.exists(path):
        df = pd.read_csv(path)
        if int(time.time() * 1000) - df["timestamp"].iloc[-1] > 35 * 1000: # if "timestamp" column is in ms
            return None
        else:
            return df

    else:
        return None


def _is_likely_futures_session_closed(now_utc: datetime | None = None) -> bool:
    """
    Rough CME equity/metal futures break: Fri after ~20:00 UTC through Sun ~22:00 UTC.
    Used to quiet price-poll spam when the API returns empty bars.
    """
    now = now_utc or datetime.now(timezone.utc)
    weekday = now.weekday()  # Mon=0 ... Sun=6
    if weekday == 5:
        return True
    if weekday == 6 and now.hour < 22:
        return True
    if weekday == 4 and (now.hour, now.minute) >= (20, 0):
        return True
    return False


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

    response = topstepx_post(url, headers=headers, payload=payload, timeout=DEFAULT_TOPSTEPX_TIMEOUT)
    response.raise_for_status()

    data = response.json()
    # The API returns a list of accounts under "accounts"
    accounts = data.get("accounts")
    if not accounts:
        raise Exception("No active accounts found")
    
    return float([acc for acc in accounts if acc["id"] == account_id][0]["balance"])


def search_trades(
    auth_token: str,
    account_id: int | str,
    start_timestamp,
    end_timestamp=None,
):
    """
    Call ProjectX Gateway trade search endpoint.

    Docs: https://gateway.docs.projectx.com/docs/api-reference/trade/trade-search

    Args:
        auth_token: Bearer token from login_to_api / init_api.
        account_id: Account ID (int or string convertible to int).
        start_timestamp: Start of timestamp filter (datetime or ISO 8601 string).
        end_timestamp: Optional end of timestamp filter (datetime or ISO 8601 string).

    Returns:
        Parsed JSON response. On success this will contain a "trades" list:
            {
                "trades": [...],
                "success": true,
                "errorCode": 0,
                "errorMessage": null
            }
    """
    url = "https://api.topstepx.com/api/Trade/search"

    headers = {
        "Authorization": f"Bearer {auth_token}",
        "accept": "text/plain",
        "Content-Type": "application/json",
    }

    def _to_iso(ts):
        if ts is None:
            return None
        if isinstance(ts, datetime):
            if ts.tzinfo is None:
                # Assume UTC if naive
                return ts.replace(tzinfo=timezone.utc).isoformat()
            return ts.isoformat()
        return str(ts)

    payload = {
        "accountId": int(account_id),
        "startTimestamp": _to_iso(start_timestamp),
    }
    et = _to_iso(end_timestamp)
    if et is not None:
        payload["endTimestamp"] = et

    response = topstepx_post(url, headers=headers, payload=payload, timeout=DEFAULT_TOPSTEPX_TIMEOUT)
    response.raise_for_status()
    return response.json()

def validate_token(auth_token: str):
    url = "https://api.topstepx.com/api/Auth/validate"

    headers = {
        "accept": "text/plain",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {auth_token}"
    }

    try:
        response = topstepx_post(url, headers=headers, payload={}, timeout=10)

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
