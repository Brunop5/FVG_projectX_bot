import pandas as pd

def ema(data, length):
    """
    Exponential Moving Average - matches PineScript ta.ema()
    Returns the latest EMA value (single number) for the last bar.
    """
    if isinstance(data, pd.DataFrame):
        # If DataFrame, assume we want the 'close' column
        series = data['close'] if 'close' in data.columns else data.iloc[:, 0]
    else:
        series = data
    
    if len(series) < length:
        return None
    
    # Calculate EMA using pandas ewm (exponential weighted moving average)
    ema_values = series.ewm(span=length, adjust=False).mean()
    return float(ema_values.iloc[-1])


def sma(bars, length):
    """
    Simple Moving Average - matches PineScript ta.sma()
    Returns the latest SMA value (single number) for the last bar.
    """
    if isinstance(bars, pd.DataFrame):
        # If DataFrame, assume we want the first column or 'close'
        series = bars['close'] if 'close' in bars.columns else bars.iloc[:, 0]
    else:
        series = bars
    
    if len(series) < length:
        return None
    
    # Calculate SMA - simple mean of last 'length' values
    sma_value = series.iloc[-length:].mean()
    return float(sma_value)


def get_atr(data: pd.DataFrame, length: int) -> pd.Series:
    """
    Average True Range - matches PineScript ta.atr()
    Uses RMA (Wilder's smoothing) of True Range, not SMA.
    Returns a Series of ATR values starting from the first valid ATR.
    """
    # Validate input
    if not isinstance(data, pd.DataFrame):
        raise ValueError("get_atr requires a DataFrame with 'high', 'low', 'close' columns")
    
    required_cols = ['high', 'low', 'close']
    if not all(col in data.columns for col in required_cols):
        raise ValueError(f"DataFrame must have columns: {required_cols}")
    
    if len(data) < length + 1:
        return pd.Series(dtype=float)  # Not enough data for ATR
    
    # Calculate True Range
    high = data['high']
    low = data['low']
    close = data['close']
    
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Prepare ATR Series
    atr = pd.Series(index=true_range.index, dtype=float)
    
    # First ATR value is simple average of first `length` TR values
    atr.iloc[length - 1] = true_range.iloc[:length].mean()
    
    # Compute subsequent ATR values using Wilder's smoothing
    for i in range(length, len(true_range)):
        atr.iloc[i] = (atr.iloc[i - 1] * (length - 1) + true_range.iloc[i]) / length
    
    # Return ATR starting from the first computed ATR
    return atr.iloc[length - 1:].reset_index(drop=True)

def crossover(current_value, previous_value, threshold):
    return current_value > threshold and previous_value <= threshold


def crossunder(current_value, previous_value, threshold):
    return current_value < threshold and previous_value >= threshold