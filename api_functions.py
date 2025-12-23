import pandas as pd
import os
import time

def fetch_data(asset, timeframe, num_bars):
    return f"{asset}, {timeframe}, {num_bars}"

def fetch_cur_price(asset):
    pass

def fetch_cur_volume(asset, timeframe):
    pass

def save_data(data, asset_name):
    pass

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

def add_new_price(close):
    pass