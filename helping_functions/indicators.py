import pandas as pd

def ema(data, length):
    """
    Exponential Moving Average - matches PineScript ta.ema()
    Returns the latest EMA value (single number) for the last bar.
    """
    if isinstance(data, pd.DataFrame):
        if data.empty:
            return None
        # If DataFrame, assume we want the 'close' column
        series = data['close'] if 'close' in data.columns else data.iloc[:, 0]
    else:
        series = data
    
    if series is None or len(series) < length:
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
        if bars.empty:
            return None
        # If DataFrame, assume we want the first column or 'close'
        series = bars['close'] if 'close' in bars.columns else bars.iloc[:, 0]
    else:
        series = bars
    
    if series is None or len(series) < length:
        return None
    
    # Calculate SMA - simple mean of last 'length' values
    sma_value = series.iloc[-length:].mean()
    return float(sma_value)


def get_atr(data, length):
    """
    Average True Range - matches PineScript ta.atr()
    Uses RMA (Wilder's smoothing) of True Range, not SMA.
    Returns a Series with the last 'length' ATR values.
    """
    if not isinstance(data, pd.DataFrame):
        raise ValueError("get_atr requires a DataFrame with 'high', 'low', 'close' columns")
    
    required_cols = ['high', 'low', 'close']
    if not all(col in data.columns for col in required_cols):
        raise ValueError(f"DataFrame must have columns: {required_cols}")
    
    if len(data) < length + 1:  # Need at least length+1 bars for TR calculation
        return None
    
    # Calculate True Range
    high = data['high']
    low = data['low']
    close = data['close']
    
    # TR = max(high - low, abs(high - prev_close), abs(low - prev_close))
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # ATR uses RMA (Wilder's smoothing), not SMA
    # RMA formula: RMA = (prev_RMA * (length - 1) + current_value) / length
    # For first value, use SMA
    atr_values = pd.Series(index=true_range.index, dtype=float)
    
    # Initialize first ATR value as SMA of first 'length' TR values
    if len(true_range) >= length:
        atr_values.iloc[length - 1] = true_range.iloc[:length].mean()
        
        # Calculate subsequent ATR values using RMA
        for i in range(length, len(true_range)):
            prev_atr = atr_values.iloc[i - 1]
            current_tr = true_range.iloc[i]
            atr_values.iloc[i] = (prev_atr * (length - 1) + current_tr) / length
    
    # Return the last 'length' ATR values as a Series
    return atr_values.iloc[-length:].reset_index(drop=True)


def crossover(current_value, previous_value, threshold):
    return current_value > threshold and previous_value <= threshold


def crossunder(current_value, previous_value, threshold):
    return current_value < threshold and previous_value >= threshold