"""
Backtesting module for the FVG trading strategy.

This module allows you to backtest the Strategy class without modifying it.
It works by:
1. Creating a BacktestStrategy wrapper that inherits from Strategy
2. Overriding API-dependent methods to use historical data instead
3. Running the strategy logic sequentially on historical bars
4. Tracking trades, P&L, and performance metrics
"""

# ==================== IMPORTS ====================
import os
import sys
from datetime import datetime, timedelta
from io import StringIO
from threading import Lock, Thread
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import itertools
import random
import multiprocessing
import inspect
import re

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import strategyClass
from strategyClass import Strategy, Order, ASSETS
from api_functions import fetch_data, load_data
from indicators import ema, sma, get_atr


# ==================== USER INPUTS / CONSTANTS ====================
# Backtest run configuration
BACKTEST_ASSET_NAME = "MGCG6"
BACKTEST_INITIAL_BALANCE = 50000.0
BACKTEST_MAX_LOSS = 2000
BACKTEST_SELECTED_TIMEFRAME = "15min"
BACKTEST_TIMEFRAMES_TO_TEST = ["5min", "15min", "30min", "1h"]
BACKTEST_START_DATE = None
BACKTEST_END_DATE = None
# Margin-based position sizing (optional)
USE_MARGIN_PER_TRADE = False
MARGIN_PER_TRADE = 100 # margin you want to commit per trade
MIN_MARGIN_LOT_SIZE = 0.002
MAX_MARGIN_LOT_SIZE = 100.0
# Optimization execution
USE_MULTITHREADING = False
USE_MULTIPROCESSING = True  # if True, uses process pool for random optimization
# Optional direct data file override (CSV)
USE_DIRECT_DATA_FILE = True
DIRECT_DATA_FILE_PATH = "data/MGCG6/IC_markets_15min.csv"

# CFD pricing configuration (used when USE_CFD_PRICING is True)
USE_CFD_PRICING = False
# fee_pct is a decimal (e.g., 0.0002 = 0.02%) applied once per round-turn trade.
DEFAULT_CFD_SETTINGS = {
    "leverage": 20,
    "fee_pct": 0.001,
    "spread": 0.1,  # absolute price units
}
CFD_SETTINGS_BY_ASSET = {
    # Example:
    # "MGCG6": {"leverage": 20.0, "fee_pct": 0.0002, "spread": 0.2},
}

# ==================== OPTIMIZATION CONFIGURATION ====================
# Edit these ranges to customize the optimization search space

OPTIMIZATION_CONFIG = {
    'FVG_HISTORY_NBR': {
        'range': list(range(1, 16)),  # 1 to 15 (int)
        'current': 10
    },
    'MIN_FVG_POWER_PCT': {
        'range': [round(0.01 + i * 0.01, 2) for i in range(20)],  # 0.01 to 0.2 by 0.01
        'current': 0.1
    },
    'HTF_TF': {
        'range': ["30", "60", "90", "120", "240", "1440"],
        'current': "240"
    },
    'EMA_PERIOD': {
        'range': [15, 25, 50, 100, 200],
        'current': 50
    },
    'VOLUME_MULTIPLIER': {
        'range': [round(1.0 + i * 0.05, 2) for i in range(11)],  # 1.0 to 1.5 by 0.05
        'current': 1.2
    },
    'ATR_PERIOD': {
        'range': list(range(5, 26)),  # 5 to 25 (int)
        'current': 14
    },
    'SL_MULTIPLIER': {
        'range': [round(1.0 + i * 0.5, 1) for i in range(19)],  # 1.0 to 10.0 by 0.5
        'current': 4.0
    },
    'TP_MULTIPLIER': {
        'range': list(range(1, 21)) + [2000000],  # 1 to 20, plus 2000000 (no TP)
        'current': 2000000.0
    },
    'USE_TRAILING': {
        'range': [True, False],
        'current': True
    },
    'TRAIL_OFFSET_MULT': {
        'range': list(range(1, 21)),  # 1 to 20 (int)
        'current': 6.0
    },
    'HOLD_UNTIL_OPPOSITE': {
        'range': [True, False],
        'current': True
    }
}

OUT_PATH = "gold_results"

# Parameters that should NOT be optimized (keep current values)
FIXED_PARAMS = {
    'USE_VOLUME_CHECK': True,
    'VOLUME_DATA_START_TIMESTAMP': 1755464400000,
    'START_FROM_VOLUME_TIMESTAMP': False  # None = auto (True if USE_VOLUME_CHECK, False otherwise). Can be set to True/False to override
}

# ==================== OPTIMIZATION SETTINGS ====================
RUN_OPTIMIZATION = True  # Set to True to run optimization, False for single backtest
USE_EXHAUSTIVE_SEARCH = False  # If True: test ALL parameter combinations in parallel (exhaustive). If False: random search
RANDOM_SEARCH_SAMPLES = 10000  
PROGRESS_STEP_PCT = 5  # Progress update granularity for multiprocessing (per sample)

USE_AUTO_WORKERS = True  # If True: auto-detect CPU count and use that many workers
MAX_WORKERS = 4  # Manual override (ignored if USE_AUTO_WORKERS is True). Set to number like 8, 16, etc.

USE_FIRST_TENTH_ONLY = False

# CSV Input Settings (alternative to random/exhaustive search)
USE_CSV_INPUT = True  # If True: read parameter combinations from CSV file instead of generating them
CSV_INPUT_FILE = "filtered_backtest_results.csv"  # Path to CSV file with strategy parameters 

# Result Saving Settings
SAVE_INDIVIDUAL_RESULTS_IN_OPTIMIZATION = True  # If True: create gold_results/{id}/

# Intra-candle entry settings
ALLOW_INTRACANDLE_CHECKS = True  # If True: enter at FVG boundary when first touched during bar, not at bar close

# ==================== GLOBAL DATA STRUCTURES ====================
# These are loaded once and reused across all backtests
CONTRACTS_DATA = {}
CONTRACTS_BY_NAME = {}
ROUND_TURN_FEES = {}
RESULTS_CSV_LOCK = Lock()  # Lock for CSV writing
SUMMARY_CSV_LOCK = Lock()  # Lock for summary CSV writing
FINAL_RESULT_LOCK = Lock()  # Lock for final_result.csv writing

# ==================== TIMEFRAME MAPPING ====================
# Map timeframes to data file names in data/MGCG6/
TIMEFRAME_FILE_MAP = {
    "5min": "GOLD.m_M5.csv",
    "30min": "GOLD.m_M30.csv",
    "1h": "GOLD.m_H1.csv",
    "15min": "1mdata_gold_15min.csv"  # Added for date range calculation
}

# Global date range for USE_FIRST_TENTH_ONLY (calculated from 5min data - first 205 days)
DATE_RANGE_START = None
DATE_RANGE_END = None
DAYS_FROM_5MIN_FIRST_205 = None  # Number of days to use from 5min data (first 205 days)

def _parse_datetime_input(value):
    """Parse date inputs from ms timestamps or ISO strings into UTC timestamps."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return pd.to_datetime(value, unit='ms', utc=True, errors='coerce')
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit():
            return pd.to_datetime(int(stripped), unit='ms', utc=True, errors='coerce')
        return pd.to_datetime(stripped, utc=True, errors='coerce')
    return pd.to_datetime(value, utc=True, errors='coerce')


# ==================== INITIALIZATION FUNCTIONS ====================

def _load_contracts_data():
    """Load contracts.csv data into global structures"""
    global CONTRACTS_DATA, CONTRACTS_BY_NAME
    
    backtest_dir = os.path.dirname(__file__)
    contracts_candidates = [
        os.path.join(backtest_dir, "contracts.csv"),
        os.path.join(os.path.dirname(backtest_dir), "contracts.csv"),
    ]
    contracts_path = next((p for p in contracts_candidates if os.path.exists(p)), None)
    if not contracts_path:
        print(f"⚠️  Warning: contracts.csv not found at {contracts_candidates}")
        return
    
    contracts_df = pd.read_csv(contracts_path)
    for _, row in contracts_df.iterrows():
        CONTRACTS_DATA[row['id']] = {
            'tick_size': float(row['tickSize']),
            'tick_value': float(row['tickValue'])
        }
        asset_name = row['name']
        CONTRACTS_BY_NAME[asset_name] = row['id']
    print(f"📋 Loaded {len(CONTRACTS_DATA)} contracts from contracts.csv")


def _load_round_turn_fees():
    """Load round turn fees from ASSETS in strategyClass and assets list"""
    global ROUND_TURN_FEES
    
    # Load from ASSETS (4-element tuples with fee as 4th element)
    assets_with_fees = 0
    for asset_tuple in ASSETS:
        if len(asset_tuple) >= 4:
            asset_id = asset_tuple[0]
            round_turn_fee = float(asset_tuple[3])
            ROUND_TURN_FEES[asset_id] = round_turn_fee
            assets_with_fees += 1
    if assets_with_fees > 0:
        print(f"💰 Loaded round turn fees for {assets_with_fees} assets from ASSETS")
    
    # Load from 'assets' list in strategyClass __main__ section (fallback)
    source_file = inspect.getfile(strategyClass)
    with open(source_file, 'r') as f:
        source_code = f.read()
    
    assets_pattern = r'assets\s*=\s*\[(.*?)\]'
    match = re.search(assets_pattern, source_code, re.DOTALL | re.MULTILINE)
    
    if match:
        assets_str = match.group(1)
        tuple_pattern = r'\("([^"]+)",\s*([\d.]+)\)'
        tuples = re.findall(tuple_pattern, assets_str)
        
        fees_loaded = 0
        for asset_id, fee_str in tuples:
            fee = float(fee_str)
            if asset_id not in ROUND_TURN_FEES:
                ROUND_TURN_FEES[asset_id] = fee
                fees_loaded += 1
        
        if fees_loaded > 0:
            print(f"💰 Loaded {fees_loaded} additional round turn fees from 'assets' list")


def initialize_backtest_data():
    """Initialize all backtest data structures (call once at startup)"""
    _load_contracts_data()
    _load_round_turn_fees()


# ==================== UTILITY FUNCTIONS ====================

def get_optimal_worker_count():
    """
    Get optimal number of workers based on system configuration.
    
    Returns:
        int: Optimal number of workers
    """
    if USE_AUTO_WORKERS:
        cpu_count = multiprocessing.cpu_count()
        # Multiprocessing is CPU-heavy; avoid over-subscribing cores.
        if USE_MULTIPROCESSING:
            # Use half the worker budget minus one to reduce contention.
            optimal_workers = max(1, (cpu_count // 2) - 1)
            print(f"💻 Detected {cpu_count} CPU cores. Using {optimal_workers} workers (multiprocessing)")
            return optimal_workers
        # Threading can benefit from modest oversubscription for mixed I/O/CPU work.
        optimal_workers = int(cpu_count * 1.5)
        print(f"💻 Detected {cpu_count} CPU cores. Using {optimal_workers} workers (threading)")
        return optimal_workers
    else:
        if MAX_WORKERS is None:
            # Fallback to CPU count if manual override is None
            cpu_count = multiprocessing.cpu_count()
            print(f"⚠️  MAX_WORKERS is None but USE_AUTO_WORKERS is False. Using CPU count: {cpu_count}")
            return cpu_count
        if USE_MULTIPROCESSING:
            return max(1, (MAX_WORKERS // 2) - 1)
        return MAX_WORKERS


def get_round_turn_fee(asset_id):
    """Get round turn fee per contract for a given asset ID"""
    return ROUND_TURN_FEES.get(asset_id, 0.0)


def get_contract_info(asset_id):
    """Get contract tick size and value for a given asset ID"""
    return CONTRACTS_DATA.get(asset_id, {'tick_size': None, 'tick_value': None})


def get_cfd_settings(asset_name, asset_id=None):
    """Get CFD settings for an asset name or ID."""
    settings = CFD_SETTINGS_BY_ASSET.get(asset_name)
    if settings is None and asset_id is not None:
        settings = CFD_SETTINGS_BY_ASSET.get(asset_id)
    if settings is None:
        settings = DEFAULT_CFD_SETTINGS
    return {
        "leverage": float(settings.get("leverage", DEFAULT_CFD_SETTINGS["leverage"])),
        "fee_pct": float(settings.get("fee_pct", DEFAULT_CFD_SETTINGS["fee_pct"])),
        "spread": float(settings.get("spread", DEFAULT_CFD_SETTINGS["spread"])),
    }


def get_contract_id_by_name(asset_name):
    """Get contract ID from asset name (e.g., 'MESH6' -> 'CON.F.US.MES.H26')"""
    return CONTRACTS_BY_NAME.get(asset_name)


def _calculate_date_range_from_5min(asset_name=BACKTEST_ASSET_NAME):
    """
    Calculate the date range from the first 205 days of 5min data.
    This date range will be used for all timeframes when USE_FIRST_TENTH_ONLY is True.
    """
    global DATE_RANGE_START, DATE_RANGE_END, DAYS_FROM_5MIN_FIRST_205
    
    if DATE_RANGE_START is not None and DATE_RANGE_END is not None:
        # Already calculated, return cached values
        return DATE_RANGE_START, DATE_RANGE_END, DAYS_FROM_5MIN_FIRST_205

    if DAYS_FROM_5MIN_FIRST_205 is None:
        return (None, None, None)
    
    # Load 5min data
    timeframe_5min = "5min"
    direct_file_path_5min = os.path.join("data", asset_name, TIMEFRAME_FILE_MAP[timeframe_5min])
    
    if not os.path.exists(direct_file_path_5min):
        print(f"⚠️  Warning: 5min data file not found: {direct_file_path_5min}")
        print(f"   Cannot calculate date range. USE_FIRST_TENTH_ONLY will use first 10% of bars instead.")
        return None, None, None
    
    try:
        # Read CSV - check if it's tab-separated (MT5 format)
        historical_data = pd.read_csv(direct_file_path_5min, sep='\t')
        
        # Check if this is MT5 format
        if '<DATE>' in historical_data.columns and '<TIME>' in historical_data.columns:
            historical_data['timestamp'] = pd.to_datetime(
                historical_data['<DATE>'].astype(str) + ' ' + historical_data['<TIME>'].astype(str),
                format='%Y.%m.%d %H:%M:%S',
                utc=True
            )
        elif 'timestamp' in historical_data.columns:
            if pd.api.types.is_numeric_dtype(historical_data['timestamp']):
                historical_data['timestamp'] = pd.to_datetime(historical_data['timestamp'], unit='ms', utc=True)
            else:
                historical_data['timestamp'] = pd.to_datetime(historical_data['timestamp'], utc=True)
        else:
            print(f"⚠️  Warning: No timestamp column found in 5min data")
            return None, None, None
        
        # Sort by timestamp
        historical_data = historical_data.sort_values('timestamp').reset_index(drop=True)
        
        # Get start date (first bar)
        start_date = historical_data['timestamp'].iloc[0]
        
        # Calculate end date as start_date + 205 days
        end_date = start_date + pd.Timedelta(days=DAYS_FROM_5MIN_FIRST_205 - 1)  # -1 because we include the start day
        
        # Filter to first 205 days
        mask = (historical_data['timestamp'] >= start_date) & (historical_data['timestamp'] <= end_date)
        first_205_days_data = historical_data[mask].reset_index(drop=True)
        
        if len(first_205_days_data) == 0:
            print(f"⚠️  Warning: No data found in first {DAYS_FROM_5MIN_FIRST_205} days of 5min data")
            return None, None, None
        
        # Get actual date range from filtered data
        DATE_RANGE_START = first_205_days_data['timestamp'].iloc[0]
        DATE_RANGE_END = first_205_days_data['timestamp'].iloc[-1]
        
        # Calculate actual number of days
        actual_days = (DATE_RANGE_END - DATE_RANGE_START).days + 1
        
        print(f"📅 Date range calculated from 5min data (first {DAYS_FROM_5MIN_FIRST_205} days):")
        print(f"   Start: {DATE_RANGE_START.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   End: {DATE_RANGE_END.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   Days: {actual_days} days")
        print(f"   Bars in 5min: {len(first_205_days_data):,} out of {len(historical_data):,} total")
        
        return DATE_RANGE_START, DATE_RANGE_END, DAYS_FROM_5MIN_FIRST_205
    except Exception as e:
        print(f"⚠️  Error calculating date range from 5min data: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


def load_backtest_data(asset_name, timeframe, direct_file_path=None):
    """
    Load historical data for backtesting using asset name and timeframe.
    
    Args:
        asset_name: Asset name from contracts.csv (e.g., "MESH6", "MNQH6")
        timeframe: Timeframe string (e.g., "5min", "15min", "30min", "1h")
        direct_file_path: Optional direct path to CSV file (e.g., "GOLD.m_M15.csv")
    
    Returns:
        tuple: (historical_data DataFrame, asset_tuple, contract_id)
               Returns (None, None, None) if asset not found or data file missing
    """
    # If direct file path is provided, use it
    if direct_file_path:
        data_path = direct_file_path
        if not os.path.exists(data_path):
            print(f"⚠️  Error: Data file not found: {data_path}")
            return None, None, None
        contract_id = get_contract_id_by_name(asset_name)
        if not contract_id:
            contract_id = asset_name
            print(f"⚠️  Warning: Asset name '{asset_name}' not found in contracts.csv. Using asset name as ID for direct file.")
    else:
        contract_id = get_contract_id_by_name(asset_name)
        if not contract_id:
            print(f"⚠️  Error: Asset name '{asset_name}' not found in contracts.csv")
            print(f"   Available assets: {', '.join(sorted(CONTRACTS_BY_NAME.keys()))}")
            return None, None, None
        # Clean asset name for folder name
        safe_asset_name = "".join(c for c in asset_name if c.isalnum() or c in (' ', '-', '_')).strip()
        safe_asset_name = safe_asset_name.replace(' ', '_')
        
        # Load data from CSV
        data_path = os.path.join("data", safe_asset_name, f"{timeframe}.csv")
        if not os.path.exists(data_path):
            print(f"⚠️  Error: Data file not found: {data_path}")
            return None, None, None
    
    # Read CSV - check if it's tab-separated (MT5 format)
    try:
        # Try reading with tab separator first (for MT5 format like GOLD.m_M15.csv)
        historical_data = pd.read_csv(data_path, sep='\t')
        
        # Check if this is MT5 format (has <DATE> and <TIME> columns)
        if '<DATE>' in historical_data.columns and '<TIME>' in historical_data.columns:
            # Combine DATE and TIME into timestamp
            historical_data['timestamp'] = pd.to_datetime(
                historical_data['<DATE>'].astype(str) + ' ' + historical_data['<TIME>'].astype(str),
                format='%Y.%m.%d %H:%M:%S',
                utc=True
            )
            
            # Rename columns to lowercase standard names
            column_mapping = {
                '<OPEN>': 'open',
                '<HIGH>': 'high',
                '<LOW>': 'low',
                '<CLOSE>': 'close',
                '<TICKVOL>': 'volume',  # Use tick volume as volume
                '<VOL>': 'vol',  # Keep original volume column as 'vol' if needed
                '<SPREAD>': 'spread'
            }
            historical_data = historical_data.rename(columns=column_mapping)
            
            # Select only the columns we need
            required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            historical_data = historical_data[required_columns]
            
            print(f"✅ Loaded MT5 format data from {data_path}")
            print(f"   Using TICKVOL as volume column")
        else:
            # Not MT5 format - check if it's already in standard format
            if 'timestamp' not in historical_data.columns:
                # Try reading with comma separator
                historical_data = pd.read_csv(data_path, sep=',')
            
            # Ensure timestamp column exists and is properly formatted
            cols_lower = {c.lower(): c for c in historical_data.columns}
            if 'timestamp' not in cols_lower and 'ts_event' in cols_lower:
                ts_col = cols_lower['ts_event']
                historical_data['timestamp'] = pd.to_datetime(historical_data[ts_col], utc=True, errors='coerce')
            elif 'timestamp' not in cols_lower and 'date' in cols_lower:
                date_col = cols_lower['date']
                historical_data['timestamp'] = pd.to_datetime(historical_data[date_col], utc=True, errors='coerce')
            elif 'timestamp' in cols_lower:
                ts_col = cols_lower['timestamp']
                if pd.api.types.is_numeric_dtype(historical_data[ts_col]):
                    historical_data['timestamp'] = pd.to_datetime(historical_data[ts_col], unit='ms', utc=True, errors='coerce')
                else:
                    historical_data['timestamp'] = pd.to_datetime(historical_data[ts_col], utc=True, errors='coerce')

            # Normalize column names and keep only required OHLCV columns
            standard_map = {
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'volume': 'volume',
                'vol': 'volume',
                'tickvol': 'volume',
                'tick_volume': 'volume',
            }
            rename_map = {}
            for col in historical_data.columns:
                key = col.lower()
                if key in standard_map:
                    rename_map[col] = standard_map[key]
            if rename_map:
                historical_data = historical_data.rename(columns=rename_map)

            required_columns = ['timestamp', 'open', 'high', 'low', 'close']
            for col in required_columns:
                if col not in historical_data.columns:
                    raise ValueError(f"Missing required column '{col}' in {data_path}")

            if 'volume' not in historical_data.columns:
                historical_data['volume'] = 0.0

            # Coerce numeric columns
            for col in ['open', 'high', 'low', 'close', 'volume']:
                historical_data[col] = pd.to_numeric(historical_data[col], errors='coerce')

            # Drop rows with missing OHLC
            historical_data = historical_data.dropna(subset=['timestamp', 'open', 'high', 'low', 'close'])
    except Exception as e:
        print(f"⚠️  Error reading CSV file {data_path}: {e}")
        return None, None, None
    
    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(historical_data['timestamp']):
        historical_data['timestamp'] = pd.to_datetime(historical_data['timestamp'], utc=True)
    
    # Sort by timestamp
    historical_data = historical_data.sort_values('timestamp').reset_index(drop=True)
    
    asset_tuple = (contract_id, timeframe, "backtest_account")
    return historical_data, asset_tuple, contract_id


# ==================== BACKTEST ORDER CLASS ====================

class BacktestOrder(Order):
    """Mock Order class for backtesting - tracks fills and P&L instead of placing real orders"""
    
    def __init__(self, side: str, entry_price: float, take_profit: float, 
                 trailing_stop_loss, entry_atr: float, account_id, asset_id, 
                 auth_token, lot_size=None, tick_size=None, tick_value=None, round_turn_fee=None,
                 use_cfd_pricing=False, cfd_leverage=1.0, cfd_fee_pct=0.0, cfd_spread=0.0,
                 use_margin_per_trade=False):
        super().__init__(side, entry_price, take_profit, trailing_stop_loss, 
                        entry_atr, account_id, asset_id, auth_token, lot_size)
        self.filled = False
        self.fill_price = None
        self.fill_time = None
        self.exit_price = None
        self.exit_time = None
        self.exit_reason = None
        self.pnl = 0.0
        self.pnl_pct = 0.0
        self.entry_bar = None
        self.tick_size = tick_size
        self.tick_value = tick_value
        self.round_turn_fee = round_turn_fee if round_turn_fee is not None else 0.0
        self.use_cfd_pricing = bool(use_cfd_pricing)
        self.cfd_leverage = cfd_leverage if cfd_leverage and cfd_leverage > 0 else 1.0
        self.cfd_fee_pct = cfd_fee_pct if cfd_fee_pct and cfd_fee_pct > 0 else 0.0
        self.cfd_spread = cfd_spread if cfd_spread and cfd_spread > 0 else 0.0
        self.use_margin_per_trade = bool(use_margin_per_trade)
        self.fees = 0.0
    
    def place_order(self, fill_time=None, entry_bar=None):
        """Mock order placement - just marks as filled immediately at entry price"""
        self.filled = True
        self.fill_price = self.entry_price
        self.fill_time = fill_time or datetime.now()
        self.entry_bar = entry_bar
        return {'success': True, 'order_id': 'backtest_order', 'message': 'Order filled'}

    def _get_effective_price(self, price, is_entry):
        if not self.use_cfd_pricing or self.cfd_spread <= 0:
            return price
        half_spread = self.cfd_spread / 2
        if self.side == "BUY":
            return price + half_spread if is_entry else price - half_spread
        return price - half_spread if is_entry else price + half_spread

    def get_unrealized_pnl(self, current_price):
        """Calculate unrealized P&L at a given price (no fees applied)."""
        if not self.filled:
            return 0.0

        if self.use_cfd_pricing:
            entry_price = self._get_effective_price(self.fill_price, is_entry=True)
            exit_price = self._get_effective_price(current_price, is_entry=False)
            if self.side == "BUY":
                price_diff = exit_price - entry_price
            else:
                price_diff = entry_price - exit_price
            leverage_mult = 1.0 if self.use_margin_per_trade else self.cfd_leverage
            return price_diff * self.lot_size * leverage_mult

        if self.side == "BUY":
            price_diff = current_price - self.fill_price
        else:
            price_diff = self.fill_price - current_price
        if self.tick_size is not None and self.tick_value is not None and self.tick_size > 0:
            ticks = price_diff / self.tick_size
            return ticks * self.tick_value * self.lot_size
        return price_diff * self.lot_size
    
    def close_order(self):
        """Mock order closing - calculates P&L using tick value if available"""
        if not self.filled or self.exit_price is None or self.pnl != 0.0:
            return

        if self.use_cfd_pricing:
            entry_price = self._get_effective_price(self.fill_price, is_entry=True)
            exit_price = self._get_effective_price(self.exit_price, is_entry=False)
            if self.side == "BUY":
                price_diff = exit_price - entry_price
            else:
                price_diff = entry_price - exit_price
            leverage_mult = 1.0 if self.use_margin_per_trade else self.cfd_leverage
            gross_pnl = price_diff * self.lot_size * leverage_mult
            entry_value = entry_price * self.lot_size * leverage_mult
            self.fees = entry_value * self.cfd_fee_pct
        else:
            # Calculate price difference
            if self.side == "BUY":
                price_diff = self.exit_price - self.fill_price
            else:  # SELL
                price_diff = self.fill_price - self.exit_price
            # Calculate P&L using tick value if available
            if self.tick_size is not None and self.tick_value is not None and self.tick_size > 0:
                ticks = price_diff / self.tick_size
                gross_pnl = ticks * self.tick_value * self.lot_size
            else:
                gross_pnl = price_diff * self.lot_size
            # Apply round turn fees
            self.fees = self.round_turn_fee * self.lot_size
            entry_value = self.fill_price * self.lot_size

        self.pnl = gross_pnl - self.fees

        # Calculate P&L percentage
        if entry_value > 0:
            self.pnl_pct = (self.pnl / entry_value) * 100
        else:
            self.pnl_pct = 0.0
        
        return {'success': True}


# ==================== BACKTEST STRATEGY CLASS ====================

class BacktestStrategy(Strategy):
    """
    Backtesting wrapper for Strategy class.
    Overrides API-dependent methods to use historical data.
    """
    
    def __init__(self, asset_tuple, historical_data: pd.DataFrame, 
                 initial_balance: float = 10000.0, start_date=None, end_date=None,
                 max_loss=None, asset_name=None, strategy_params=None):
        """
        Initialize backtest strategy.
        
        Args:
            asset_tuple: (asset, timeframe, account_name) - same as live strategy
            historical_data: DataFrame with OHLCV data
            initial_balance: Starting account balance for backtest
            start_date: Start date for backtest (if None, uses all data)
            end_date: End date for backtest (if None, uses all data)
            max_loss: Maximum loss threshold. If between 0 and 1, treated as percentage.
                     If >= 1, treated as absolute dollar amount. If None, no limit.
            asset_name: Asset name from contracts.csv - used to load HTF data
            strategy_params: Dictionary of strategy parameters (captured at backtest start)
        """
        super().__init__(asset_tuple)
        self.asset_name = asset_name
        
        # Store strategy parameters at initialization (before they can be changed by other threads)
        if strategy_params is None:
            # If not provided, capture current parameters
            strategy_params = {}
            for param_name in OPTIMIZATION_CONFIG.keys():
                if hasattr(strategyClass, param_name):
                    strategy_params[param_name] = getattr(strategyClass, param_name)
            for param_name in FIXED_PARAMS.keys():
                if hasattr(strategyClass, param_name):
                    strategy_params[param_name] = getattr(strategyClass, param_name)
        self.strategy_params = strategy_params
        
        # Helper method to get parameter value (from stored params or strategyClass as fallback)
        def get_param(param_name, default=None):
            if self.strategy_params and param_name in self.strategy_params:
                return self.strategy_params[param_name]
            return getattr(strategyClass, param_name, default)
        self.get_param = get_param
        
        # Filter data by date range if provided
        if 'timestamp' in historical_data.columns:
            if pd.api.types.is_numeric_dtype(historical_data['timestamp']):
                historical_data['timestamp'] = pd.to_datetime(historical_data['timestamp'], unit='ms', utc=True)
            else:
                historical_data['timestamp'] = pd.to_datetime(historical_data['timestamp'], utc=True)
            
            if start_date:
                start_ts = _parse_datetime_input(start_date)
                historical_data = historical_data[historical_data['timestamp'] >= start_ts]
            if end_date:
                end_ts = _parse_datetime_input(end_date)
                historical_data = historical_data[historical_data['timestamp'] <= end_ts]
            
            # Find start index for volume data timestamp
            # Determine if we should start from timestamp:
            # - If START_FROM_VOLUME_TIMESTAMP is explicitly set (True/False), use that
            # - If START_FROM_VOLUME_TIMESTAMP is None, auto: use USE_VOLUME_CHECK value (True = start from timestamp, False = start from beginning)
            start_from_timestamp_param = self.get_param('START_FROM_VOLUME_TIMESTAMP')
            if start_from_timestamp_param is None:
                # Auto: use USE_VOLUME_CHECK value
                start_from_timestamp = self.get_param('USE_VOLUME_CHECK', False)
            else:
                # Use explicit value (could be True or False)
                start_from_timestamp = bool(start_from_timestamp_param)
            
            self.volume_start_index = 0
            if start_from_timestamp:
                volume_start_datetime = pd.to_datetime(self.get_param('VOLUME_DATA_START_TIMESTAMP', 1755464400000), unit='ms', utc=True)
                # Find the first index where timestamp >= volume_start_datetime
                mask = historical_data['timestamp'] >= volume_start_datetime
                if mask.any():
                    self.volume_start_index = historical_data[mask].index[0]
                    before_count = self.volume_start_index
                    after_count = len(historical_data) - self.volume_start_index
                    if before_count > 0:
                        reason = "volume check enabled" if self.get_param('USE_VOLUME_CHECK', False) else "explicitly requested"
                        print(f"📊 Volume data starts at: {volume_start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
                        print(f"   Will start backtest from bar {self.volume_start_index} ({before_count:,} bars before, {after_count:,} bars after) - {reason}")
                else:
                    # If no data after timestamp, start from beginning but warn
                    print(f"⚠️  Warning: No data found after volume start timestamp {volume_start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
                    print(f"   Starting from beginning of dataset")
            else:
                self.volume_start_index = 0
        
        self.historical_data = historical_data.sort_values('timestamp').reset_index(drop=True)
        
        # Limit to date range from 5min first 205 days if USE_FIRST_TENTH_ONLY is enabled
        if USE_FIRST_TENTH_ONLY:
            # Calculate date range from 5min data if not already calculated
            date_start, date_end, num_days = _calculate_date_range_from_5min(asset_name)
            date_start = _parse_datetime_input(date_start)
            date_end = _parse_datetime_input(date_end)
            
            if date_start is not None and date_end is not None:
                original_length = len(self.historical_data)
                
                # For all timeframes (including 5min): filter by the same date range
                mask = (self.historical_data['timestamp'] >= date_start) & (self.historical_data['timestamp'] <= date_end)
                self.historical_data = self.historical_data[mask].reset_index(drop=True)
                print(f"📊 Limited {self.timeframe} data to date range from 5min first {num_days} days: {len(self.historical_data):,} bars (from {original_length:,} total)")
                print(f"   Date range: {date_start.strftime('%Y-%m-%d %H:%M:%S')} to {date_end.strftime('%Y-%m-%d %H:%M:%S')}")
                
                # Check if filtering removed all data
                if len(self.historical_data) == 0:
                    print(f"⚠️  ERROR: Filtering removed ALL data for {self.asset} {self.timeframe}!")
                    print(f"   This timeframe has no bars in the date range from 5min first {num_days} days.")
                    print(f"   Date range: {date_start.strftime('%Y-%m-%d %H:%M:%S')} to {date_end.strftime('%Y-%m-%d %H:%M:%S')}")
                    if original_length > 0:
                        # Get original data range before filtering
                        original_data = historical_data.sort_values('timestamp').reset_index(drop=True)
                        print(f"   Original data range: {original_data['timestamp'].iloc[0]} to {original_data['timestamp'].iloc[-1]}")
                    else:
                        print(f"   Original data was already empty")
            else:
                # Fallback: use first 10% of bars if date range calculation failed
                original_length = len(self.historical_data)
                tenth_length = max(1, original_length // 10)  # At least 1 bar
                self.historical_data = self.historical_data.iloc[:tenth_length].reset_index(drop=True)
                print(f"📊 Limited data to first 10% (fallback): {len(self.historical_data):,} bars (from {original_length:,} total)")
        
        # Final check: ensure we have data after all filtering
        if len(self.historical_data) == 0:
            error_msg = f"ERROR: No data available for {self.asset} {self.timeframe} after filtering. This will cause zero trades and zero days."
            print(f"⚠️  {error_msg}")
            raise ValueError(error_msg)
        
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.current_bar_index = 0
        self.trades = []
        self.equity_curve = []
        
        # Check if volume column exists and has valid data
        self.has_volume_data = 'volume' in self.historical_data.columns
        if self.has_volume_data:
            # Check if volume column has any non-null, non-zero values
            volume_valid = self.historical_data['volume'].notna().any()
            if volume_valid:
                # Check if there are meaningful volume values (not all zeros or very low)
                volume_mean = self.historical_data['volume'].mean()
                self.has_volume_data = volume_mean > 0 and not pd.isna(volume_mean)
            else:
                self.has_volume_data = False
        
        # Get contract or CFD information
        asset_id = asset_tuple[0]
        self.use_cfd_pricing = bool(USE_CFD_PRICING)
        self.cfd_leverage = 1.0
        self.cfd_fee_pct = 0.0
        self.cfd_spread = 0.0

        if self.use_cfd_pricing:
            raw_settings = CFD_SETTINGS_BY_ASSET.get(asset_name) or CFD_SETTINGS_BY_ASSET.get(asset_id)
            cfd_settings = get_cfd_settings(asset_name, asset_id)
            self.cfd_leverage = cfd_settings["leverage"]
            self.cfd_fee_pct = cfd_settings["fee_pct"]
            self.cfd_spread = cfd_settings["spread"]
            self.tick_size = None
            self.tick_value = None
            self.round_turn_fee = 0.0
            if raw_settings is None:
                print(f"⚠️  Warning: No CFD settings found for {asset_name or asset_id}. Using DEFAULT_CFD_SETTINGS.")
            print(
                f"📋 Using CFD pricing: {asset_name or asset_id} | "
                f"Leverage: {self.cfd_leverage}x | "
                f"Fee: {self.cfd_fee_pct * 100:.4f}% | "
                f"Spread: {self.cfd_spread}"
            )
        else:
            contract_info = get_contract_info(asset_id)
            self.tick_size = contract_info['tick_size']
            self.tick_value = contract_info['tick_value']
            self.round_turn_fee = get_round_turn_fee(asset_id)

            if self.tick_size is not None and self.tick_value is not None:
                fee_info = f" | Round Turn Fee: ${self.round_turn_fee:.2f}/contract" if self.round_turn_fee > 0 else " | ⚠️  No fee found in ASSETS"
                print(f"📋 Using contract info: {asset_id} | Tick Size: {self.tick_size} | Tick Value: ${self.tick_value}{fee_info}")
            else:
                print(f"⚠️  Warning: Contract {asset_id} not found in contracts.csv. Using simple P&L calculation.")
        
        # Max loss configuration
        self.max_loss = max_loss
        if max_loss is not None:
            if 0 < max_loss < 1:
                self.max_loss_amount = initial_balance * max_loss
                self.max_loss_type = "percentage"
            else:
                self.max_loss_amount = max_loss
                self.max_loss_type = "absolute"
        else:
            self.max_loss_amount = None
            self.max_loss_type = None
        
        self.strategy_failed = False
        self.failed_reason = None
        
        # Override account_id for backtest
        self.account_id = "backtest_account"
        self.auth_token = "backtest_token"
    
    def _get_order_lot_size(self, atr, sl_multiplier, entry_price):
        """Calculate lot size; optionally size by fixed margin per trade."""
        if USE_MARGIN_PER_TRADE:
            if entry_price is None or entry_price <= 0:
                return self.calculate_lot_size(atr, sl_multiplier)
            leverage = self.cfd_leverage if self.use_cfd_pricing else 1.0
            notional = MARGIN_PER_TRADE * leverage
            lot_size = notional / entry_price
            lot_size = max(MIN_MARGIN_LOT_SIZE, lot_size)
            if MAX_MARGIN_LOT_SIZE is not None:
                lot_size = min(MAX_MARGIN_LOT_SIZE, lot_size)
            return lot_size
        return self.calculate_lot_size(atr, sl_multiplier)

    def init_rest(self):
        """Override init_rest to use historical data instead of API"""
        self.account_balance = self.initial_balance
        self.current_balance = self.initial_balance
        self.active_order = None

        min_bars_needed = max(100, 50)
        
        # Start from volume_start_index if USE_VOLUME_CHECK is enabled
        start_index = max(0, self.volume_start_index) if hasattr(self, 'volume_start_index') else 0
        
        # Check if we have any data at all
        if len(self.historical_data) == 0:
            raise ValueError(f"ERROR: No historical data available for {self.asset} {self.timeframe} after filtering! This will result in zero trades and zero days.")
        
        # Ensure we have enough bars after the start index
        available_bars = len(self.historical_data) - start_index
        if available_bars < min_bars_needed:
            if start_index > 0:
                print(f"⚠️  Warning: Only {available_bars} bars available after volume start timestamp (need {min_bars_needed})")
                print(f"   Starting from beginning of dataset instead")
                start_index = 0
                # Re-check after resetting start_index
                available_bars = len(self.historical_data)
                if available_bars < min_bars_needed:
                    raise ValueError(f"Not enough historical data. Need at least {min_bars_needed} bars, got {len(self.historical_data)}")
            else:
                raise ValueError(f"Not enough historical data. Need at least {min_bars_needed} bars, got {len(self.historical_data)}")
        
        # Initialize data starting from start_index
        end_index = start_index + min_bars_needed
        self.data = self.historical_data.iloc[start_index:end_index].copy()
        self.current_bar_index = end_index
        
        print(f"📊 Backtest initialized with {len(self.data)} initial bars")
        print(f"📅 Date range: {self.data['timestamp'].iloc[0]} to {self.historical_data['timestamp'].iloc[-1]}")
        if start_index > 0:
            print(f"   Started from bar {start_index} (after volume data start timestamp)")
        else:
            print(f"   Started from beginning of dataset (bar 0)")
        
        self.cur_close = self.data["close"].iloc[-1]
        # Handle missing volume column gracefully
        if 'volume' in self.data.columns:
            self.cur_volume = self.data["volume"].iloc[-1] if pd.notna(self.data["volume"].iloc[-1]) else 0
        else:
            self.cur_volume = 0
        
        # Initialize indicators (same as live)
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
        
        # Load and cache HTF data once for performance
        self._load_htf_data()
        
        # Skip load_metadata for backtest
        self.calculate_indicators()
        self.fvg_zones: list[dict] = []
        self.add_fvg_zones()
    
    def _load_htf_data(self):
        """Load HTF data once and cache it for performance"""
        self.htf_data = None
        self.htf_data_timestamps = None
        
        if not self.asset_name:
            return
        
        htf_minutes = int(self.get_param('HTF_TF', '240'))
        
        # Map HTF minutes to timeframe string for CSV file
        if htf_minutes == 240:
            htf_timeframe = "4h"
        elif htf_minutes == 120:
            htf_timeframe = "2h"
        elif htf_minutes == 60:
            htf_timeframe = "1h"
        elif htf_minutes == 1440:
            htf_timeframe = "1d"
        else:
            if htf_minutes >= 1440:
                htf_timeframe = f"{htf_minutes // 1440}d"
            elif htf_minutes >= 60:
                htf_timeframe = f"{htf_minutes // 60}h"
            else:
                htf_timeframe = f"{htf_minutes}min"
        
        safe_asset_name = "".join(c for c in self.asset_name if c.isalnum() or c in (' ', '-', '_')).strip()
        safe_asset_name = safe_asset_name.replace(' ', '_')
        
        htf_data_path = os.path.join("data", safe_asset_name, f"{htf_timeframe}.csv")
        
        if not os.path.exists(htf_data_path):
            return
        
        self.htf_data = pd.read_csv(htf_data_path)
        
        if 'timestamp' in self.htf_data.columns:
            if pd.api.types.is_numeric_dtype(self.htf_data['timestamp']):
                self.htf_data['timestamp'] = pd.to_datetime(self.htf_data['timestamp'], unit='ms', utc=True)
            else:
                self.htf_data['timestamp'] = pd.to_datetime(self.htf_data['timestamp'], utc=True)
        
        self.htf_data = self.htf_data.sort_values('timestamp').reset_index(drop=True)
        self.htf_data_timestamps = self.htf_data['timestamp'].values
    
    def calculate_indicators(self):
        """Override to use self.get_param for MIN_FVG_POWER_PCT"""
        self.update_trend_indicators()

        gapClose = self.data["close"].iloc[-3]
        min_power_pct = self.get_param('MIN_FVG_POWER_PCT', 0.08)

        self.bullishPowerOK = (
            self.lastBullFvg
            and (self.data["low"].iloc[-2] - self.data["high"].iloc[-4]) / gapClose * 100 >= min_power_pct
        )

        self.bearishPowerOK = (
            self.lastBearFvg
            and (self.data["low"].iloc[-4] - self.data["high"].iloc[-2]) / gapClose * 100 >= min_power_pct
        )

        self.calc_BOS_and_CHOCH()
    
    def add_fvg_zones(self):
        """Override to remove print statements for performance"""
        if self.bullishPowerOK and self.isBullishHTF and self.marketOK:
            self.fvg_zones.append(
                {
                    "direction": "bull",
                    "top": self.data["low"].iloc[-2],
                    "bottom": self.data["high"].iloc[-4],
                    "mitigated": False,
                }
            )

        if self.bearishPowerOK and self.isBearishHTF and self.marketOK:
            self.fvg_zones.append(
                {
                    "direction": "bear",
                    "top": self.data["low"].iloc[-4],
                    "bottom": self.data["high"].iloc[-2],
                    "mitigated": False,
                }
            )

        if len(self.fvg_zones) > self.get_param('FVG_HISTORY_NBR', 3):
            self.fvg_zones = self.fvg_zones[-self.get_param('FVG_HISTORY_NBR', 3):]
    
    def gather_data(self):
        """Override to return initial data window"""
        return self.data
    
    def fetch_new_data(self):
        """Override to get next bar from historical data"""
        if self.current_bar_index >= len(self.historical_data):
            return None
        
        start_idx = max(0, self.current_bar_index - 99)
        end_idx = self.current_bar_index + 1
        self.data = self.historical_data.iloc[start_idx:end_idx]
        
        self.current_bar_index += 1
        
        self.cur_close = self.data["close"].iloc[-1]
        # Handle missing volume column gracefully
        if 'volume' in self.data.columns:
            self.cur_volume = self.data["volume"].iloc[-1] if pd.notna(self.data["volume"].iloc[-1]) else 0
        else:
            self.cur_volume = 0
        
        return self.data.iloc[-1:]
    
    def update_trend_indicators(self):
        """Override to match live strategy: use last max(101, EMA_PERIOD+51) HTF bars up to current timestamp"""
        if self.htf_data is None:
            self._update_trend_indicators_resample()
            return
        
        current_timestamp = self.data['timestamp'].iloc[-1]
        # Find all HTF bars that closed <= current bar's timestamp (no look-ahead)
        idx = self.htf_data['timestamp'].searchsorted(current_timestamp, side='right')
        
        # Get EMA_PERIOD parameter
        ema_period = self.get_param('EMA_PERIOD', 50)
        
        # Match live strategy: use last max(101, EMA_PERIOD+51) bars (same as live fetch_data call)
        bars_needed = max(101, ema_period + 51)
        
        if idx < ema_period:
            self.isBullishHTF = None
            self.isBearishHTF = None
        else:
            # Take the last bars_needed bars from available HTF bars (up to current timestamp)
            # This matches live strategy behavior: fetch_data gets most recent bars_needed bars
            start_idx = max(0, idx - bars_needed)
            htf_close = self.htf_data['close'].iloc[start_idx:idx]
            htfEMA = ema(htf_close, ema_period)
            
            if htfEMA is not None:
                self.isBullishHTF = self.cur_close > htfEMA
                self.isBearishHTF = self.cur_close < htfEMA
            else:
                self.isBullishHTF = None
                self.isBearishHTF = None
        
        # Calculate marketOK, lastBullFvg, and lastBearFvg (same as live strategy)
        atrVal = get_atr(self.data, self.get_param('ATR_PERIOD', 14))
        atr_sma = sma(atrVal, 20) if len(atrVal) > 0 else None
        atrOK = atrVal.iloc[-1] > atr_sma if (len(atrVal) > 0 and atr_sma is not None) else False
        
        # Volume check: only if USE_VOLUME_CHECK is True AND volume data is available
        if self.get_param('USE_VOLUME_CHECK', False) and self.has_volume_data and 'volume' in self.data.columns:
            vol_sma = sma(self.data["volume"], 20)
            volOK = self.cur_volume > vol_sma * self.get_param('VOLUME_MULTIPLIER', 1.2) if vol_sma is not None else False
            self.marketOK = volOK and atrOK
        else:
            # Skip volume check if volume data is not available or USE_VOLUME_CHECK is False
            self.marketOK = atrOK
        
        # Update FVG detection flags
        self.lastBullFvg = self.data["high"].iloc[-4] < self.data["low"].iloc[-2] and not self.lastBullFvg
        self.lastBearFvg = self.data["low"].iloc[-4] > self.data["high"].iloc[-2] and not self.lastBearFvg
    
    def _update_trend_indicators_resample(self):
        """Fallback method: resample current timeframe data to HTF"""
        htf_data = self.historical_data.iloc[:self.current_bar_index].copy()
        
        if htf_data['timestamp'].dtype in ['int64', 'int32']:
            htf_data['timestamp'] = pd.to_datetime(htf_data['timestamp'], unit='ms', utc=True)
        else:
            htf_data['timestamp'] = pd.to_datetime(htf_data['timestamp'], utc=True)
        
        htf_data = htf_data.set_index('timestamp')
        
        htf_minutes = int(self.get_param('HTF_TF', '240'))
        current_tf_minutes = self._get_timeframe_minutes(self.timeframe)
        bars_per_htf = htf_minutes // current_tf_minutes
        
        if len(htf_data) < self.get_param('EMA_PERIOD', 50) * bars_per_htf:
            self.isBullishHTF = None
            self.isBearishHTF = None
        else:
            if htf_minutes == 240:
                resample_period = '4h'
            elif htf_minutes == 120:
                resample_period = '2h'
            elif htf_minutes == 60:
                resample_period = '1h'
            elif htf_minutes == 1440:
                resample_period = '1d'
            else:
                resample_period = f'{bars_per_htf * current_tf_minutes}min'
            
            # Build aggregation dict conditionally based on available columns
            agg_dict = {
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last'
            }
            if 'volume' in htf_data.columns:
                agg_dict['volume'] = 'sum'
            
            htf_resampled = htf_data.resample(resample_period, label='right', closed='right').agg(agg_dict).dropna()
            
            ema_period = self.get_param('EMA_PERIOD', 50)
            bars_needed = max(101, ema_period + 51)  # Match live strategy: max(101, EMA_PERIOD+51)
            
            if len(htf_resampled) >= ema_period:
                # Take the last bars_needed resampled bars (match live strategy behavior)
                start_idx = max(0, len(htf_resampled) - bars_needed)
                htf_close = htf_resampled['close'].iloc[start_idx:]
                htfEMA = ema(htf_close, ema_period)
                
                if htfEMA is not None:
                    self.isBullishHTF = self.cur_close > htfEMA
                    self.isBearishHTF = self.cur_close < htfEMA
                else:
                    self.isBullishHTF = None
                    self.isBearishHTF = None
            else:
                self.isBullishHTF = None
                self.isBearishHTF = None
        
        # Calculate marketOK
        atrVal = get_atr(self.data, self.get_param('ATR_PERIOD', 14))
        atr_sma = sma(atrVal, 20) if len(atrVal) > 0 else None
        atrOK = atrVal.iloc[-1] > atr_sma if (len(atrVal) > 0 and atr_sma is not None) else False
        
        # Volume check: only if USE_VOLUME_CHECK is True AND volume data is available
        if self.get_param('USE_VOLUME_CHECK', False) and self.has_volume_data and 'volume' in self.data.columns:
            vol_sma = sma(self.data["volume"], 20)
            volOK = self.cur_volume > vol_sma * self.get_param('VOLUME_MULTIPLIER', 1.2) if vol_sma is not None else False
            self.marketOK = volOK and atrOK
        else:
            # Skip volume check if volume data is not available or USE_VOLUME_CHECK is False
            self.marketOK = atrOK
        
        # Update FVG detection flags
        self.lastBullFvg = self.data["high"].iloc[-4] < self.data["low"].iloc[-2] and not self.lastBullFvg
        self.lastBearFvg = self.data["low"].iloc[-4] > self.data["high"].iloc[-2] and not self.lastBearFvg
    
    def _get_timeframe_minutes(self, timeframe):
        """Convert timeframe string to minutes"""
        if 'min' in timeframe.lower():
            return int(''.join(filter(str.isdigit, timeframe)))
        elif 'h' in timeframe.lower():
            hours = int(''.join(filter(str.isdigit, timeframe)) or '1')
            return hours * 60
        elif 'd' in timeframe.lower():
            days = int(''.join(filter(str.isdigit, timeframe)) or '1')
            return days * 24 * 60
        else:
            return 1
    
    def check_daily_trade_limit(self):
        """Check if maximum daily trades has been reached (same as live strategy)"""
        current_date = self.data['timestamp'].iloc[-1].date() if 'timestamp' in self.data.columns else datetime.now().date()
        
        if self.last_trade_date != str(current_date):
            self.daily_trades_count = 0
            self.last_trade_date = str(current_date)
        
        return self.daily_trades_count < self.get_param('MAX_DAILY_TRADES', 3)
    
    def entry_logic(self):
        """Override to use BacktestOrder instead of Order"""
        if len(self.fvg_zones) == 0 or self.inPosition:
            return
        
        if not self.check_daily_trade_limit():
            return
        
        current_high = self.data["high"].iloc[-1]
        current_low = self.data["low"].iloc[-1]
        
        atr = get_atr(self.data, self.get_param('ATR_PERIOD', 14)).iloc[-1]
        
        for zone in self.fvg_zones[-self.get_param('FVG_HISTORY_NBR', 3):]:
            if zone["mitigated"]:
                continue
            
            fvg_bottom = zone["bottom"]
            fvg_top = zone["top"]
            touchesFVG = current_high >= fvg_bottom and current_low <= fvg_top
            
            if (zone["direction"] == "bull" and touchesFVG and 
                self.isBullishHTF and self.marketOK):
                # Determine entry price based on ALLOW_INTRACANDLE_CHECKS
                if ALLOW_INTRACANDLE_CHECKS:
                    # Enter at FVG top (lower boundary of bullish gap) when first touched
                    entry_price = fvg_top
                else:
                    # Enter at bar close (original behavior)
                    entry_price = self.cur_close
                
                print(atr)
                trailStop = entry_price - atr * self.get_param('SL_MULTIPLIER', 4.0)
                tp = entry_price + atr * self.get_param('TP_MULTIPLIER', 2000000.0)
                entryAtr = atr
                lot_size = self._get_order_lot_size(atr, self.get_param('SL_MULTIPLIER', 4.0), entry_price)
                
                current_time = self.data.iloc[-1].get('timestamp', datetime.now())
                self.active_order = BacktestOrder("BUY", entry_price, tp, trailStop, 
                                                  entryAtr, self.account_id, self.asset, 
                                                  self.auth_token, lot_size, 
                                                  tick_size=self.tick_size, tick_value=self.tick_value,
                                                  round_turn_fee=self.round_turn_fee,
                                                  use_cfd_pricing=self.use_cfd_pricing,
                                                  cfd_leverage=self.cfd_leverage,
                                                  cfd_fee_pct=self.cfd_fee_pct,
                                                  cfd_spread=self.cfd_spread,
                                                  use_margin_per_trade=USE_MARGIN_PER_TRADE)
                result = self.active_order.place_order(fill_time=current_time, entry_bar=self.current_bar_index)
                
                if result['success']:
                    zone["mitigated"] = True
                    self.lastPositionWasLong = True
                    self.lastPositionWasShort = False
                    self.inPosition = True
                    self.daily_trades_count += 1
                    self.last_trade_date = str(datetime.now().date())
                break
            
            elif (zone["direction"] == "bear" and touchesFVG and 
                  self.isBearishHTF and self.marketOK):
                # Determine entry price based on ALLOW_INTRACANDLE_CHECKS
                if ALLOW_INTRACANDLE_CHECKS:
                    # Enter at FVG bottom (upper boundary of bearish gap) when first touched
                    entry_price = fvg_bottom
                else:
                    # Enter at bar close (original behavior)
                    entry_price = self.cur_close
                
                trailStop = entry_price + atr * self.get_param('SL_MULTIPLIER', 4.0)
                tp = entry_price - atr * self.get_param('TP_MULTIPLIER', 2000000.0)
                entryAtr = atr
                lot_size = self._get_order_lot_size(atr, self.get_param('SL_MULTIPLIER', 4.0), entry_price)
                
                current_time = self.data.iloc[-1].get('timestamp', datetime.now())
                self.active_order = BacktestOrder("SELL", entry_price, tp, trailStop, 
                                                  entryAtr, self.account_id, self.asset, 
                                                  self.auth_token, lot_size,
                                                  tick_size=self.tick_size, tick_value=self.tick_value,
                                                  round_turn_fee=self.round_turn_fee,
                                                  use_cfd_pricing=self.use_cfd_pricing,
                                                  cfd_leverage=self.cfd_leverage,
                                                  cfd_fee_pct=self.cfd_fee_pct,
                                                  cfd_spread=self.cfd_spread,
                                                  use_margin_per_trade=USE_MARGIN_PER_TRADE)
                result = self.active_order.place_order(fill_time=current_time, entry_bar=self.current_bar_index)
                
                if result['success']:
                    zone["mitigated"] = True
                    self.lastPositionWasShort = True
                    self.lastPositionWasLong = False
                    self.inPosition = True
                    self.daily_trades_count += 1
                    self.last_trade_date = str(datetime.now().date())
                break
    
    def check_exits(self):
        """Check if stop loss, take profit, or trailing stop was hit"""
        if not self.active_order or not self.active_order.filled:
            return
        
        current_bar = self.data.iloc[-1]
        current_high = current_bar['high']
        current_low = current_bar['low']
        current_close = current_bar['close']
        current_time = current_bar.get('timestamp', datetime.now())
        
        order = self.active_order
        
        if order.side == "BUY":
            if current_low <= order.trailing_stop_loss:
                order.exit_price = order.trailing_stop_loss
                order.exit_time = current_time
                order.exit_reason = "Stop Loss"
                order.close_order()
                self._record_trade(order)
                self._close_position()
                return True
            
            if current_high >= order.take_profit:
                order.exit_price = order.take_profit
                order.exit_time = current_time
                order.exit_reason = "Take Profit"
                order.close_order()
                self._record_trade(order)
                self._close_position()
                return True
        
        else:  # SELL
            if current_high >= order.trailing_stop_loss:
                order.exit_price = order.trailing_stop_loss
                order.exit_time = current_time
                order.exit_reason = "Stop Loss"
                order.close_order()
                self._record_trade(order)
                self._close_position()
                return True
            
            if current_low <= order.take_profit:
                order.exit_price = order.take_profit
                order.exit_time = current_time
                order.exit_reason = "Take Profit"
                order.close_order()
                self._record_trade(order)
                self._close_position()
                return True
        
        return False
    
    def _check_max_loss(self):
        """Check if maximum loss threshold has been reached"""
        if self.max_loss_amount is None:
            return False
        
        total_loss = self.initial_balance - self.current_balance
        if total_loss >= self.max_loss_amount:
            self.strategy_failed = True
            if self.max_loss_type == "percentage":
                loss_pct = (self.max_loss_amount / self.initial_balance) * 100
                self.failed_reason = f"Maximum loss threshold reached: {loss_pct:.2f}% (${self.max_loss_amount:,.2f})"
            else:
                self.failed_reason = f"Maximum loss threshold reached: ${self.max_loss_amount:,.2f}"
            return True
        return False
    
    def _record_trade(self, order):
        """Record completed trade"""
        trade = {
            'entry_time': order.fill_time,
            'exit_time': order.exit_time,
            'side': order.side,
            'entry_price': order.fill_price,
            'exit_price': order.exit_price,
            'size': order.lot_size,
            'pnl': order.pnl,
            'pnl_pct': order.pnl_pct,
            'fees': order.fees if hasattr(order, 'fees') else 0.0,
            'exit_reason': order.exit_reason,
            'entry_bar': order.entry_bar,
            'exit_bar': self.current_bar_index,
            'bars_held': self.current_bar_index - order.entry_bar if order.entry_bar else 0,
            'entry_atr': getattr(order, 'entry_atr', None),
            'sl_price': getattr(order, 'trailing_stop_loss', None),
            'tp_price': getattr(order, 'take_profit', None),
            'sl_mult': self.get_param('SL_MULTIPLIER', None),
            'tp_mult': self.get_param('TP_MULTIPLIER', None),
        }
        self.trades.append(trade)
        self.current_balance += order.pnl
        
        if self._check_max_loss():
            print(f"\n⚠️  STRATEGY FAILED: {self.failed_reason}")
            print(f"   Current Balance: ${self.current_balance:,.2f}")
            print(f"   Total Loss: ${self.initial_balance - self.current_balance:,.2f}")
            print(f"   Stopping backtest...\n")
    
    def _close_position(self):
        """Close current position"""
        self.active_order = None
        self.inPosition = False
        self.lastPositionWasLong = False
        self.lastPositionWasShort = False
    
    def update_stops(self):
        """Override to also check exits"""
        if self.check_exits():
            return
        
        pos = self.active_order
        if pos is None:
            return
        
        current_high = self.data["high"].iloc[-1]
        current_low = self.data["low"].iloc[-1]
        
        if self.inPosition and self.lastPositionWasLong:
            if self.get_param('USE_TRAILING', True) and pos.entry_atr is not None:
                potentialStop = current_high - pos.entry_atr * self.get_param('TRAIL_OFFSET_MULT', 6.0)
                if pos.trailing_stop_loss is not None:
                    new_stop = max(pos.trailing_stop_loss, potentialStop)
                    if new_stop > pos.trailing_stop_loss:
                        pos.trailing_stop_loss = new_stop
                else:
                    pos.trailing_stop_loss = potentialStop
        
        if self.inPosition and self.lastPositionWasShort:
            if self.get_param('USE_TRAILING', True) and pos.entry_atr is not None:
                potentialStop = current_low + pos.entry_atr * self.get_param('TRAIL_OFFSET_MULT', 6.0)
                if pos.trailing_stop_loss is not None:
                    new_stop = min(pos.trailing_stop_loss, potentialStop)
                    if new_stop < pos.trailing_stop_loss:
                        pos.trailing_stop_loss = new_stop
                else:
                    pos.trailing_stop_loss = potentialStop
        
        # Check BOS/CHoCH exits (same as live strategy)
        if self.get_param('HOLD_UNTIL_OPPOSITE', True) and self.inPosition:
            if self.lastPositionWasLong and self.isCHOCH:
                current_bar = self.data.iloc[-1]
                pos.exit_price = current_bar['close']
                pos.exit_time = current_bar.get('timestamp', datetime.now())
                pos.exit_reason = "CHoCH"
                pos.close_order()
                self._record_trade(pos)
                self._close_position()
            
            if self.lastPositionWasShort and self.isBOS:
                current_bar = self.data.iloc[-1]
                pos.exit_price = current_bar['close']
                pos.exit_time = current_bar.get('timestamp', datetime.now())
                pos.exit_reason = "BOS"
                pos.close_order()
                self._record_trade(pos)
                self._close_position()
        
        # Record equity curve (calculate unrealized P&L if position is open)
        unrealized_pnl = 0.0
        if pos and pos.filled:
            current_bar = self.data.iloc[-1]
            current_price = current_bar['close']
            unrealized_pnl = pos.get_unrealized_pnl(current_price)
        
        self.equity_curve.append({
            'bar': self.current_bar_index,
            'timestamp': self.data.iloc[-1].get('timestamp', datetime.now()),
            'balance': self.current_balance,
            'equity': self.current_balance + unrealized_pnl
        })
    
    def run_backtest(self, show_progress=True, progress_prefix="", progress_line=None, suppress_header=False, progress_callback=None, progress_step_pct=10):
        """Run the backtest on historical data"""
        if not suppress_header:
            print(f"\n{'='*60}")
            print(f"🧪 Starting Backtest for {self.asset}")
            print(f"{'='*60}")
            print(f"Initial Balance: ${self.initial_balance:,.2f}")
            if self.max_loss_amount is not None:
                if self.max_loss_type == "percentage":
                    loss_pct = (self.max_loss_amount / self.initial_balance) * 100
                    print(f"Max Loss Limit:     {loss_pct:.2f}% (${self.max_loss_amount:,.2f})")
                else:
                    print(f"Max Loss Limit:     ${self.max_loss_amount:,.2f}")
            print(f"Date Range: {self.historical_data['timestamp'].iloc[0]} to {self.historical_data['timestamp'].iloc[-1]}")
            print(f"Total Bars: {len(self.historical_data)}\n")
        
        self.init_rest()
        
        total_bars = len(self.historical_data)
        progress_interval = max(1, total_bars // 100)  # Update more frequently for smoother progress
        next_callback_pct = float(progress_step_pct) if progress_callback else None
        
        while self.current_bar_index < len(self.historical_data) and not self.strategy_failed:
            self.fetch_new_data()
            self.calculate_indicators()
            self.add_fvg_zones()
            self.entry_logic()
            self.update_stops()
            
            if show_progress:
                progress = (self.current_bar_index / total_bars) * 100
                progress_msg = f"{progress_prefix}Progress: {progress:.1f}% ({self.current_bar_index}/{total_bars} bars)"
                
                # For multithreaded runs, use ANSI positioning on dedicated lines
                if progress_line is not None:
                    # Update more frequently - every 0.1% of progress or every 10 bars, whichever is smaller
                    update_interval = max(1, min(total_bars // 1000, 10))
                    if self.current_bar_index % update_interval == 0 or self.current_bar_index == total_bars - 1:
                        # Use ANSI escape codes to position on specific line
                        # \033[n;0H moves to line n, column 0
                        # \033[K clears the rest of the line
                        print(f"\033[{progress_line + 1};0H{progress_msg}\033[K", end='', file=sys.stderr, flush=True)
                else:
                    # Default behavior: overwrite same line (single backtest mode)
                    print(progress_msg, end='\r', file=sys.stderr)
                    sys.stderr.flush()
            elif progress_callback is not None:
                progress = (self.current_bar_index / total_bars) * 100
                if next_callback_pct is not None and progress >= next_callback_pct:
                    progress_callback(progress, self.current_bar_index, total_bars)
                    next_callback_pct += progress_step_pct
            
            # Check max loss after each bar (in case of open position drawdown)
            if self.active_order and self.active_order.filled:
                current_bar = self.data.iloc[-1]
                current_price = current_bar['close']
                
                unrealized_pnl = self.active_order.get_unrealized_pnl(current_price)
                
                current_equity = self.current_balance + unrealized_pnl
                total_loss = self.initial_balance - current_equity
                if self.max_loss_amount is not None and total_loss >= self.max_loss_amount:
                    self.strategy_failed = True
                    self.active_order.exit_price = current_price
                    self.active_order.exit_time = current_bar.get('timestamp', datetime.now())
                    self.active_order.exit_reason = "Max Loss Reached"
                    self.active_order.close_order()
                    self._record_trade(self.active_order)
                    self._close_position()
                    break
        
        # Close any open position at end
        if self.active_order and self.active_order.filled and not self.strategy_failed:
            current_bar = self.data.iloc[-1]
            self.active_order.exit_price = current_bar['close']
            self.active_order.exit_time = current_bar.get('timestamp', datetime.now())
            self.active_order.exit_reason = "End of Data"
            self.active_order.close_order()
            self._record_trade(self.active_order)
            self._close_position()
        
        # Show 100% when done
        if show_progress:
            progress_msg = f"{progress_prefix}Progress: 100.0% ({total_bars}/{total_bars} bars)"
            if progress_line is not None:
                # For multithreaded runs, print final progress on new line
                print(progress_msg, file=sys.stderr, flush=True)
            else:
                print(progress_msg, end='\r', file=sys.stderr)
                sys.stderr.flush()
        elif progress_callback is not None:
            progress_callback(100.0, total_bars, total_bars)
        
        return self._get_backtest_results()
    
    def _get_backtest_results(self):
        """Calculate and return backtest results as a dictionary"""
        if not self.trades:
            return {
                'total_pnl': 0.0,
                'total_trades': 0,
                'win_rate': 0.0,
                'final_balance': self.current_balance,
                'max_drawdown': 0.0,
                'max_drawdown_pct': 0.0,
                'total_fees': 0.0,
                'winning_trades': 0,
                'losing_trades': 0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'profit_factor': 0.0,
                'total_return': 0.0,
                'net_profit': 0.0,
                'largest_win': 0.0,
                'largest_loss': 0.0,
                'trades_per_day': 0.0,
                'backtest_period_days': 0
            }
        
        trades_df = pd.DataFrame(self.trades)
        
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        losing_trades = len(trades_df[trades_df['pnl'] < 0])
        total_pnl = trades_df['pnl'].sum()
        
        if 'fees' in trades_df.columns:
            total_fees = trades_df['fees'].sum()
        else:
            total_fees = self.round_turn_fee * trades_df['size'].sum() if self.round_turn_fee > 0 else 0.0
        
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0
        profit_factor = abs(avg_win * winning_trades / (avg_loss * losing_trades)) if losing_trades > 0 and avg_loss != 0 else float('inf')
        
        final_balance = self.current_balance
        total_return = ((final_balance - self.initial_balance) / self.initial_balance) * 100
        net_profit = final_balance - self.initial_balance
        
        largest_win = trades_df['pnl'].max()
        largest_loss = trades_df['pnl'].min()
        
        # Calculate max drawdown
        if len(self.equity_curve) > 0:
            equity_df = pd.DataFrame(self.equity_curve)
            if 'equity' in equity_df.columns:
                equity_values = equity_df['equity'].values
                running_max = np.maximum.accumulate(equity_values)
                drawdowns = running_max - equity_values
                max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0.0
                peak_equity = np.max(equity_values) if len(equity_values) > 0 else self.initial_balance
                max_drawdown_pct = (max_drawdown / peak_equity * 100) if peak_equity > 0 else 0.0
            else:
                max_drawdown = 0.0
                max_drawdown_pct = 0.0
        else:
            max_drawdown = 0.0
            max_drawdown_pct = 0.0
        
        # Calculate day count
        if len(self.historical_data) > 0 and self.current_bar_index > 0:
            start_date = pd.to_datetime(self.historical_data['timestamp'].iloc[0])
            actual_end_index = min(self.current_bar_index - 1, len(self.historical_data) - 1)
            end_date = pd.to_datetime(self.historical_data['timestamp'].iloc[actual_end_index])
            day_count = (end_date - start_date).days + 1
            trades_per_day = total_trades / day_count if day_count > 0 else 0
        else:
            day_count = 0
            trades_per_day = 0
        
        return {
            'total_pnl': total_pnl,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'final_balance': final_balance,
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown_pct,
            'total_fees': total_fees,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'total_return': total_return,
            'net_profit': net_profit,
            'largest_win': largest_win,
            'largest_loss': largest_loss,
            'trades_per_day': trades_per_day,
            'backtest_period_days': day_count
        }
    
    def generate_report(self, results=None):
        """Generate backtest performance report"""
        if results is None:
            results = self._get_backtest_results()
        
        if results['total_trades'] == 0:
            print("\n❌ No trades executed during backtest period")
            if self.strategy_failed:
                print(f"⚠️  STRATEGY STATUS: FAILED")
                print(f"   Reason: {self.failed_reason}")
            return
        
        trades_df = pd.DataFrame(self.trades)
        
        total_trades = results['total_trades']
        winning_trades = results['winning_trades']
        losing_trades = results['losing_trades']
        win_rate = results['win_rate']
        total_pnl = results['total_pnl']
        total_fees = results['total_fees']
        avg_win = results['avg_win']
        avg_loss = results['avg_loss']
        profit_factor = results['profit_factor']
        final_balance = results['final_balance']
        total_return = results['total_return']
        net_loss = self.initial_balance - final_balance
        max_drawdown = results['max_drawdown']
        max_drawdown_pct = results['max_drawdown_pct']
        day_count = results['backtest_period_days']
        trades_per_day = results['trades_per_day']
        
        print(f"\n{'='*60}")
        print(f"📊 BACKTEST RESULTS")
        print(f"{'='*60}")
        if self.strategy_failed:
            print(f"⚠️  STRATEGY STATUS: FAILED")
            print(f"   {self.failed_reason}")
            print(f"{'='*60}")
        print(f"Initial Balance:     ${self.initial_balance:,.2f}")
        print(f"Final Balance:       ${final_balance:,.2f}")
        print(f"Total Return:        {total_return:.2f}%")
        print(f"Total P&L:           ${total_pnl:,.2f}")
        print(f"Total Fees Paid:     ${total_fees:,.2f}")
        print(f"Max Drawdown:        ${max_drawdown:,.2f} ({max_drawdown_pct:.2f}%)")
        if net_loss > 0:
            print(f"Net Loss:            ${net_loss:,.2f}")
        else:
            print(f"Net Profit:          ${abs(net_loss):,.2f}")
        if self.max_loss_amount is not None:
            if self.max_loss_type == "percentage":
                loss_pct = (self.max_loss_amount / self.initial_balance) * 100
                print(f"Max Loss Limit:      {loss_pct:.2f}% (${self.max_loss_amount:,.2f})")
            else:
                print(f"Max Loss Limit:      ${self.max_loss_amount:,.2f}")
        print(f"\nBacktest Period:     {day_count} days")
        print(f"Total Trades:        {total_trades}")
        print(f"Trades per Day:      {trades_per_day:.2f}")
        print(f"Winning Trades:      {winning_trades}")
        print(f"Losing Trades:       {losing_trades}")
        print(f"Win Rate:            {win_rate:.2f}%")
        print(f"\nAverage Win:         ${avg_win:.2f}")
        print(f"Average Loss:        ${avg_loss:.2f}")
        print(f"Profit Factor:       {profit_factor:.2f}")
        print(f"\nLargest Win:         ${results['largest_win']:.2f}")
        print(f"Largest Loss:        ${results['largest_loss']:.2f}")
        print(f"{'='*60}\n")
        
        # Save results - check if we're in single backtest mode (not optimization)
        # If so, save to gold_results/{id}/ directory
        result_id = getattr(self, 'result_id', None)
        if result_id is not None:
            # Single backtest mode - save to gold_results/{id}/
            result_dir = os.path.join(OUT_PATH, str(result_id))
            os.makedirs(result_dir, exist_ok=True)
            
            trades_df.to_csv(os.path.join(result_dir, f"backtest_trades_{self.asset}_{datetime.now().strftime('%Y%m%d')}.csv"), index=False)
            equity_df = pd.DataFrame(self.equity_curve)
            equity_df.to_csv(os.path.join(result_dir, f"backtest_equity_{self.asset}_{datetime.now().strftime('%Y%m%d')}.csv"), index=False)
            print(f"💾 Results saved to {result_dir}/")
            
            # Save summary to final_result.csv
            self.save_to_final_result_csv(results, result_id, asset_name=getattr(self, 'asset_name', None))
            print(f"💾 Added to final_result.csv with ID {result_id}")
            
            # Plot equity curve - save to result directory
            self._plot_equity_curve(equity_df, result_dir=result_dir)
        else:
            # Optimization mode or old behavior - save to current directory
            trades_df.to_csv(f"backtest_trades_{self.asset}_{datetime.now().strftime('%Y%m%d')}.csv", index=False)
            equity_df = pd.DataFrame(self.equity_curve)
            equity_df.to_csv(f"backtest_equity_{self.asset}_{datetime.now().strftime('%Y%m%d')}.csv", index=False)
            print(f"💾 Results saved to CSV files")
            
            # Save summary to CSV (inputs + metrics)
            summary_csv = self.save_summary_to_csv(results, asset_name=getattr(self, 'asset_name', None))
            print(f"💾 Summary saved to {summary_csv}")
            
            # Plot equity curve
            self._plot_equity_curve(equity_df)
    
    def _plot_equity_curve(self, equity_df, result_dir=None):
        """Plot the equity curve"""
        if len(equity_df) == 0:
            return
        
        if 'timestamp' in equity_df.columns:
            if pd.api.types.is_numeric_dtype(equity_df['timestamp']):
                equity_df['timestamp'] = pd.to_datetime(equity_df['timestamp'], unit='ms', utc=True)
            else:
                equity_df['timestamp'] = pd.to_datetime(equity_df['timestamp'], utc=True)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        ax1.plot(equity_df['timestamp'], equity_df['equity'], label='Equity', linewidth=1.5, color='#2E86AB')
        ax1.axhline(y=self.initial_balance, color='gray', linestyle='--', linewidth=1, label='Initial Balance', alpha=0.7)
        ax1.set_ylabel('Equity ($)', fontsize=11, fontweight='bold')
        ax1.set_title(f'Equity Curve - {self.asset}', fontsize=13, fontweight='bold', pad=10)
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.legend(loc='best', fontsize=9)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        ax2.plot(equity_df['timestamp'], equity_df['balance'], label='Balance', linewidth=1.5, color='#A23B72')
        ax2.axhline(y=self.initial_balance, color='gray', linestyle='--', linewidth=1, label='Initial Balance', alpha=0.7)
        ax2.set_xlabel('Date', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Balance ($)', fontsize=11, fontweight='bold')
        ax2.set_title('Account Balance', fontsize=12, fontweight='bold', pad=10)
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.legend(loc='best', fontsize=9)
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        if self.max_loss_amount is not None:
            max_loss_line = self.initial_balance - self.max_loss_amount
            ax1.axhline(y=max_loss_line, color='red', linestyle=':', linewidth=1.5, label='Max Loss Threshold', alpha=0.7)
            ax2.axhline(y=max_loss_line, color='red', linestyle=':', linewidth=1.5, label='Max Loss Threshold', alpha=0.7)
            ax1.legend(loc='best', fontsize=9)
            ax2.legend(loc='best', fontsize=9)
        
        plt.tight_layout()
        
        if result_dir:
            plot_filename = os.path.join(result_dir, f"backtest_equity_curve_{self.asset}_{datetime.now().strftime('%Y%m%d')}.png")
        else:
            plot_filename = f"backtest_equity_curve_{self.asset}_{datetime.now().strftime('%Y%m%d')}.png"
        plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
        print(f"📈 Equity curve plot saved to {plot_filename}")
    
    def save_summary_to_csv(self, results=None, asset_name=None):
        """Save backtest summary (inputs + metrics) to CSV file (thread-safe)"""
        if results is None:
            results = self._get_backtest_results()
        
        # Use stored strategy parameters (captured at initialization) instead of reading from strategyClass
        # This prevents parameter state pollution in multithreaded environments
        strategy_params = getattr(self, 'strategy_params', {})
        
        # Create result row with all inputs and metrics
        result_row = {
            # Input parameters
            'asset_name': asset_name if asset_name is not None else getattr(self, 'asset_name', self.asset),
            'asset_id': self.asset,
            'timeframe': self.timeframe,
            'initial_balance': self.initial_balance,
            'max_loss': self.max_loss,
            'max_loss_amount': self.max_loss_amount if hasattr(self, 'max_loss_amount') else None,
            'max_loss_type': self.max_loss_type if hasattr(self, 'max_loss_type') else None,
            'strategy_failed': self.strategy_failed if hasattr(self, 'strategy_failed') else False,
            'failed_reason': self.failed_reason if hasattr(self, 'failed_reason') else None,
            # Strategy parameters (from stored params, not current strategyClass state)
            **strategy_params,
            # Metrics
            'total_pnl': results['total_pnl'],
            'total_trades': results['total_trades'],
            'win_rate': results['win_rate'],
            'final_balance': results['final_balance'],
            'max_drawdown': results['max_drawdown'],
            'max_drawdown_pct': results['max_drawdown_pct'],
            'total_fees': results['total_fees'],
            'winning_trades': results['winning_trades'],
            'losing_trades': results['losing_trades'],
            'avg_win': results['avg_win'],
            'avg_loss': results['avg_loss'],
            'profit_factor': results['profit_factor'],
            'total_return': results['total_return'],
            'net_profit': results['net_profit'],
            'largest_win': results['largest_win'],
            'largest_loss': results['largest_loss'],
            'trades_per_day': results['trades_per_day'],
            'backtest_period_days': results['backtest_period_days'],
            # Timestamp
            'backtest_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # CSV filename
        csv_filename = f"backtest_summary.csv"
        
        # Thread-safe CSV writing
        with SUMMARY_CSV_LOCK:
            # Check if file exists
            file_exists = os.path.exists(csv_filename)
            
            # Create DataFrame from single row
            df_new = pd.DataFrame([result_row])
            
            if file_exists:
                # Read existing CSV and append
                df_existing = pd.read_csv(csv_filename)
                # Combine
                df_combined = pd.concat([df_existing, df_new], ignore_index=True)
                df_combined.to_csv(csv_filename, index=False)
            else:
                # Create new file
                df_new.to_csv(csv_filename, index=False)
        
        return csv_filename
    
    def save_to_final_result_csv(self, results=None, result_id=None, asset_name=None):
        """Save backtest summary to final_result.csv with ID and without certain columns"""
        if results is None:
            results = self._get_backtest_results()
        
        if result_id is None:
            return
        
        # Use stored strategy parameters
        strategy_params = getattr(self, 'strategy_params', {})
        
        # Create result row - EXCLUDE: asset_name, asset_id, initial_balance, max_loss, max_loss_amount, max_loss_type
        result_row = {
            # ID at the beginning
            'id': result_id,
            # Timeframe
            'timeframe': self.timeframe,
            # Strategy parameters
            **strategy_params,
            # Metrics
            'total_pnl': results['total_pnl'],
            'total_trades': results['total_trades'],
            'win_rate': results['win_rate'],
            'final_balance': results['final_balance'],
            'max_drawdown': results['max_drawdown'],
            'max_drawdown_pct': results['max_drawdown_pct'],
            'total_fees': results['total_fees'],
            'winning_trades': results['winning_trades'],
            'losing_trades': results['losing_trades'],
            'avg_win': results['avg_win'],
            'avg_loss': results['avg_loss'],
            'profit_factor': results['profit_factor'],
            'total_return': results['total_return'],
            'net_profit': results['net_profit'],
            'largest_win': results['largest_win'],
            'largest_loss': results['largest_loss'],
            'trades_per_day': results['trades_per_day'],
            'backtest_period_days': results['backtest_period_days'],
            'strategy_failed': self.strategy_failed if hasattr(self, 'strategy_failed') else False,
            'failed_reason': self.failed_reason if hasattr(self, 'failed_reason') else None,
            # Timestamp
            'backtest_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # CSV filename in gold_results directory
        os.makedirs(OUT_PATH, exist_ok=True)
        csv_filename = os.path.join(OUT_PATH, "final_result.csv")
        
        # Thread-safe CSV writing
        with FINAL_RESULT_LOCK:
            # Check if file exists
            file_exists = os.path.exists(csv_filename)
            
            # Create DataFrame from single row
            df_new = pd.DataFrame([result_row])
            
            if file_exists:
                # Read existing CSV and append
                df_existing = pd.read_csv(csv_filename)
                # Combine
                df_combined = pd.concat([df_existing, df_new], ignore_index=True)
                df_combined.to_csv(csv_filename, index=False)
            else:
                # Create new file
                df_new.to_csv(csv_filename, index=False)
        
        return csv_filename


def _get_next_result_id():
    """Get the next available result ID by reading final_result.csv (thread-safe)"""
    csv_filename = os.path.join(OUT_PATH, "final_result.csv")
    
    with FINAL_RESULT_LOCK:
        if not os.path.exists(csv_filename):
            return 1
        
        try:
            df = pd.read_csv(csv_filename)
            if 'id' in df.columns and len(df) > 0:
                return int(df['id'].max()) + 1
            else:
                return 1
        except Exception:
            return 1


# ==================== OPTIMIZATION FUNCTIONS ====================

def _apply_strategy_params(params_dict):
    """Temporarily modify strategyClass constants"""
    for param_name, param_value in params_dict.items():
        if hasattr(strategyClass, param_name):
            setattr(strategyClass, param_name, param_value)


def _get_current_strategy_params():
    """Get current parameter values from strategyClass"""
    current_params = {}
    for param_name in OPTIMIZATION_CONFIG.keys():
        if hasattr(strategyClass, param_name):
            current_params[param_name] = getattr(strategyClass, param_name)
    
    # Also get fixed params
    for param_name in FIXED_PARAMS.keys():
        if hasattr(strategyClass, param_name):
            current_params[param_name] = getattr(strategyClass, param_name)
    
    return current_params


def _run_single_backtest(asset_name, timeframe, initial_balance, max_loss, params_dict, verbose=False, progress_prefix="", progress_line=None, direct_file_path=None, save_individual_results=False, show_progress=True, progress_callback=None, progress_step_pct=PROGRESS_STEP_PCT, return_error=False):
    """
    Run a single backtest with given parameters and return results
    
    Args:
        save_individual_results: If True, create gold_results/{id}/ folder with trades, equity, and plot (like single backtest mode)
    """
    # Suppress output if not verbose (but allow progress through stderr)
    if not verbose:
        old_stdout = sys.stdout
        sys.stdout = StringIO()
    
    try:
        # Apply parameters to strategyClass
        _apply_strategy_params(params_dict)
        
        # Immediately capture a copy of all parameters (before they can be changed by other threads)
        # This ensures we save the correct parameters that were used for this backtest
        captured_params = {}
        for param_name in OPTIMIZATION_CONFIG.keys():
            if hasattr(strategyClass, param_name):
                captured_params[param_name] = getattr(strategyClass, param_name)
        for param_name in FIXED_PARAMS.keys():
            if hasattr(strategyClass, param_name):
                captured_params[param_name] = getattr(strategyClass, param_name)
        
        # Load data
        historical_data, asset_tuple, contract_id = load_backtest_data(asset_name, timeframe, direct_file_path=direct_file_path)
        if historical_data is None:
            if verbose:
                print(f"❌ Failed to load data for {asset_name} {timeframe}")
            if return_error:
                return None, f"Failed to load data for {asset_name} {timeframe}"
            return None
        
        # Check if data is empty after loading
        if len(historical_data) == 0:
            if verbose:
                print(f"❌ No data available for {asset_name} {timeframe} after loading")
            if return_error:
                return None, f"No data available for {asset_name} {timeframe} after loading"
            return None
        
        # Create and run backtest with captured parameters
        try:
            backtest = BacktestStrategy(
                asset_tuple=asset_tuple,
                historical_data=historical_data,
                initial_balance=initial_balance,
                max_loss=max_loss,
                asset_name=asset_name,
                strategy_params=captured_params,  # Pass captured parameters
            )
        except ValueError as e:
            # This catches the "Not enough historical data" error from init_rest
            if verbose:
                print(f"❌ Error initializing backtest: {e}")
            if return_error:
                return None, f"Error initializing backtest: {e}"
            return None
        
        # Run with progress shown even when not verbose
        results = backtest.run_backtest(show_progress=show_progress, progress_prefix=progress_prefix, progress_line=progress_line, suppress_header=not verbose, progress_callback=progress_callback, progress_step_pct=progress_step_pct)
        
        # Save individual results if requested (create gold_results/{id}/ folder)
        if save_individual_results and results is not None:
            try:
                # Get next available result ID
                result_id = _get_next_result_id()
                backtest.result_id = result_id
                
                # Generate report which will save to gold_results/{id}/ and final_result.csv
                backtest.generate_report(results)
            except Exception as e:
                # Don't fail the backtest if individual result save fails
                if verbose:
                    print(f"⚠️  Warning: Failed to save individual results: {e}")
        
        # Save summary to CSV after each backtest (always save summary)
        if results is not None:
            try:
                backtest.save_summary_to_csv(results, asset_name=asset_name)
            except Exception as e:
                # Don't fail the backtest if CSV save fails
                if verbose:
                    print(f"⚠️  Warning: Failed to save summary CSV: {e}")
        
        # Clear progress line when done
        if not verbose and progress_prefix:
            print("", file=sys.stderr)  # New line after progress
        
        if return_error:
            return results, None
        return results
    except Exception as e:
        # Always surface errors so failures aren't silent in multithreaded runs.
        print(f"❌ Error in backtest: {e}", file=sys.stderr)
        if verbose:
            import traceback
            traceback.print_exc()
        if return_error:
            return None, f"{type(e).__name__}: {e}"
        return None
    finally:
        if not verbose:
            sys.stdout = old_stdout


def _save_result_to_csv(result_row, csv_filename):
    """Save a single result row to CSV (thread-safe)"""
    with RESULTS_CSV_LOCK:
        # Check if file exists
        file_exists = os.path.exists(csv_filename)
        
        # Create DataFrame from single row
        df_new = pd.DataFrame([result_row])
        
        if file_exists:
            # Read existing CSV and append
            df_existing = pd.read_csv(csv_filename)
            # Combine and remove duplicates (keep latest)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            # Remove duplicates based on parameter columns (keep last)
            param_cols = [col for col in df_combined.columns if col in OPTIMIZATION_CONFIG.keys()]
            if param_cols:
                df_combined = df_combined.drop_duplicates(subset=param_cols, keep='last')
            df_combined.to_csv(csv_filename, index=False)
        else:
            # Create new file
            df_new.to_csv(csv_filename, index=False)


def _run_backtest_worker(args):
    """Worker for multiprocessing: run a single backtest and return results."""
    combo_index, asset_name, combo_timeframe, initial_balance, max_loss, combo, combo_direct_file_path, progress_queue = args
    def progress_callback(pct, current_bar, total_bars):
        if progress_queue is not None:
            progress_queue.put((combo_index, combo_timeframe, pct, current_bar, total_bars))
    results, error_msg = _run_single_backtest(
        asset_name,
        combo_timeframe,
        initial_balance,
        max_loss,
        combo,
        verbose=False,
        progress_prefix="",
        progress_line=None,
        direct_file_path=combo_direct_file_path,
        save_individual_results=SAVE_INDIVIDUAL_RESULTS_IN_OPTIMIZATION,
        show_progress=False,
        progress_callback=progress_callback,
        progress_step_pct=PROGRESS_STEP_PCT,
        return_error=True,
    )
    return combo, combo_timeframe, results, error_msg


def _generate_all_combinations():
    """Generate all parameter combinations to test"""
    # Get all parameter ranges
    param_ranges = {}
    for param_name, config in OPTIMIZATION_CONFIG.items():
        param_ranges[param_name] = config['range']
    
    # Generate all combinations
    param_names = list(param_ranges.keys())
    param_values = [param_ranges[name] for name in param_names]
    
    combinations = []
    for combo in itertools.product(*param_values):
        params_dict = dict(zip(param_names, combo))
        # Add fixed params
        params_dict.update(FIXED_PARAMS)
        combinations.append(params_dict)
    
    return combinations


def _load_already_tested_strategies():
    """
    Load already-tested strategies from final_result.csv.
    
    Returns:
        List of parameter dictionaries representing already-tested strategies.
        Each dict contains all input parameters (strategy params + timeframe).
    """
    final_result_csv = os.path.join(OUT_PATH, "final_result.csv")
    
    if not os.path.exists(final_result_csv):
        return []
    
    try:
        df = pd.read_csv(final_result_csv)
        
        # Get all parameter column names (from OPTIMIZATION_CONFIG and FIXED_PARAMS)
        param_columns = list(OPTIMIZATION_CONFIG.keys()) + list(FIXED_PARAMS.keys())
        
        tested_strategies = []
        for idx, row in df.iterrows():
            params_dict = {}
            
            # Extract strategy parameters
            for param_name in param_columns:
                if param_name in df.columns:
                    value = row[param_name]
                    if pd.notna(value):
                        # Convert to appropriate type
                        if param_name in OPTIMIZATION_CONFIG:
                            if OPTIMIZATION_CONFIG[param_name]['range']:
                                example_value = OPTIMIZATION_CONFIG[param_name]['range'][0]
                                if isinstance(example_value, bool):
                                    params_dict[param_name] = bool(value)
                                elif isinstance(example_value, int):
                                    params_dict[param_name] = int(value)
                                elif isinstance(example_value, float):
                                    params_dict[param_name] = float(value)
                                elif isinstance(example_value, str):
                                    params_dict[param_name] = str(value)
                                else:
                                    params_dict[param_name] = value
                            else:
                                params_dict[param_name] = value
                        else:
                            params_dict[param_name] = value
            
            # Add fixed params that might not be in CSV
            for param_name, default_value in FIXED_PARAMS.items():
                if param_name not in params_dict:
                    params_dict[param_name] = default_value
            
            # Add timeframe if present
            if 'timeframe' in df.columns and pd.notna(row['timeframe']):
                params_dict['timeframe'] = str(row['timeframe'])
            
            tested_strategies.append(params_dict)
        
        return tested_strategies
    
    except Exception as e:
        print(f"⚠️  Warning: Could not load already-tested strategies: {e}")
        return []


def _strategies_match(params1, params2):
    """
    Check if two parameter dictionaries represent the same strategy.
    
    Args:
        params1: First parameter dictionary
        params2: Second parameter dictionary
    
    Returns:
        True if strategies match (all input parameters are the same), False otherwise
    """
    # Get all parameter names to compare (strategy params + timeframe)
    param_names = list(OPTIMIZATION_CONFIG.keys()) + list(FIXED_PARAMS.keys()) + ['timeframe']
    
    for param_name in param_names:
        val1 = params1.get(param_name)
        val2 = params2.get(param_name)
        
        # Handle None/NaN comparisons
        val1_is_nan = val1 is None or (isinstance(val1, float) and pd.isna(val1))
        val2_is_nan = val2 is None or (isinstance(val2, float) and pd.isna(val2))
        
        if val1_is_nan and val2_is_nan:
            continue
        if val1_is_nan or val2_is_nan:
            return False
        
        # Compare values (handle float precision for numeric comparisons)
        # Also handle int vs float comparisons (e.g., 1.0 == 1)
        if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
            if abs(float(val1) - float(val2)) > 1e-10:
                return False
        elif isinstance(val1, bool) and isinstance(val2, bool):
            if val1 != val2:
                return False
        elif isinstance(val1, str) and isinstance(val2, str):
            if val1 != val2:
                return False
        else:
            # Try direct comparison
            if val1 != val2:
                return False
    
    return True


def _is_strategy_already_tested(params_dict, tested_strategies):
    """
    Check if a parameter combination has already been tested.
    
    Args:
        params_dict: Parameter dictionary to check
        tested_strategies: List of already-tested parameter dictionaries
    
    Returns:
        True if already tested, False otherwise
    """
    for tested in tested_strategies:
        if _strategies_match(params_dict, tested):
            return True
    return False


def _load_parameter_combinations_from_csv(csv_file):
    """
    Load parameter combinations from a CSV file (e.g., filtered_backtest_results.csv).
    Filters out strategies that have already been tested (exist in final_result.csv).
    
    Args:
        csv_file: Path to CSV file containing strategy parameters
    
    Returns:
        List of parameter dictionaries, each containing strategy parameters from one row.
        If 'timeframe' column exists, it will be included in each dict.
        Only returns strategies that haven't been tested yet.
    """
    if not os.path.exists(csv_file):
        print(f"❌ Error: CSV file not found: {csv_file}")
        return []
    
    try:
        # Load already-tested strategies
        print("🔍 Checking for already-tested strategies...")
        tested_strategies = _load_already_tested_strategies()
        if tested_strategies:
            print(f"   Found {len(tested_strategies)} already-tested strategies in final_result.csv")
        else:
            print(f"   No existing results found (will test all strategies)")
        
        df = pd.read_csv(csv_file)
        print(f"📊 Loaded {len(df)} parameter combinations from {csv_file}")
        
        # Get all parameter column names (from OPTIMIZATION_CONFIG and FIXED_PARAMS)
        param_columns = list(OPTIMIZATION_CONFIG.keys()) + list(FIXED_PARAMS.keys())
        
        # Also check for timeframe column
        has_timeframe = 'timeframe' in df.columns
        
        combinations = []
        skipped_count = 0
        
        for idx, row in df.iterrows():
            params_dict = {}
            
            # Extract strategy parameters
            for param_name in param_columns:
                if param_name in df.columns:
                    value = row[param_name]
                    # Handle NaN values - skip them (use default from strategyClass)
                    if pd.notna(value):
                        # Convert to appropriate type based on parameter
                        if param_name in OPTIMIZATION_CONFIG:
                            # Get the type from the first value in range
                            if OPTIMIZATION_CONFIG[param_name]['range']:
                                example_value = OPTIMIZATION_CONFIG[param_name]['range'][0]
                                if isinstance(example_value, bool):
                                    params_dict[param_name] = bool(value)
                                elif isinstance(example_value, int):
                                    params_dict[param_name] = int(value)
                                elif isinstance(example_value, float):
                                    params_dict[param_name] = float(value)
                                elif isinstance(example_value, str):
                                    params_dict[param_name] = str(value)
                                else:
                                    params_dict[param_name] = value
                            else:
                                params_dict[param_name] = value
                        else:
                            # Fixed param - keep as is
                            params_dict[param_name] = value
            
            # Add fixed params that might not be in CSV
            for param_name, default_value in FIXED_PARAMS.items():
                if param_name not in params_dict:
                    params_dict[param_name] = default_value
            
            # Add timeframe if present in CSV
            if has_timeframe and 'timeframe' in df.columns and pd.notna(row['timeframe']):
                params_dict['timeframe'] = str(row['timeframe'])
            
            # Check if this strategy has already been tested
            if _is_strategy_already_tested(params_dict, tested_strategies):
                skipped_count += 1
                continue
            
            combinations.append(params_dict)
        
        print(f"✅ Extracted {len(combinations)} untested parameter combinations")
        if skipped_count > 0:
            print(f"⏭️  Skipped {skipped_count} already-tested strategies")
        return combinations
    
    except Exception as e:
        print(f"❌ Error reading CSV file {csv_file}: {e}")
        import traceback
        traceback.print_exc()
        return []


def _generate_random_combinations(num_samples, include_timeframe=False, available_timeframes=None):
    """
    Generate random parameter combinations to test.
    
    Args:
        num_samples: Number of random combinations to generate
        include_timeframe: If True, randomly select timeframe for each combination
        available_timeframes: List of timeframes to choose from (e.g., ["5min", "30min", "1h"])
    
    Returns:
        List of parameter dictionaries. If include_timeframe=True, each dict includes 'timeframe' key.
    """
    # Get all parameter ranges
    param_ranges = {}
    for param_name, config in OPTIMIZATION_CONFIG.items():
        param_ranges[param_name] = config['range']
    
    param_names = list(param_ranges.keys())
    combinations = []
    
    # Generate random combinations
    for _ in range(num_samples):
        params_dict = {}
        for param_name in param_names:
            # Randomly select a value from the parameter's range
            params_dict[param_name] = random.choice(param_ranges[param_name])
        
        # Add fixed params
        params_dict.update(FIXED_PARAMS)
        
        # Randomly select timeframe if requested
        if include_timeframe and available_timeframes:
            params_dict['timeframe'] = random.choice(available_timeframes)
        
        combinations.append(params_dict)
    
    return combinations


def optimize_strategy_multithreaded(asset_name, timeframe, initial_balance=50000.0, max_loss=2000, max_workers=4, direct_file_path=None):
    """
    Multithreaded optimization: test all parameter combinations in parallel.
    Saves results to CSV after each combination completes.
    """
    print(f"\n{'='*60}")
    print(f"🔍 Starting Multithreaded Optimization")
    print(f"{'='*60}")
    print(f"Asset: {asset_name} | Timeframe: {timeframe}")
    print(f"Initial Balance: ${initial_balance:,.2f} | Max Loss: ${max_loss:,.2f}")
    print(f"Max Workers: {max_workers}\n")
    
    # Generate all combinations
    print("📊 Generating parameter combinations...")
    combinations = _generate_all_combinations()
    total_combinations = len(combinations)
    print(f"   Total combinations to test: {total_combinations:,}\n")
    
    # CSV filename - save to gold_results directory
    os.makedirs(OUT_PATH, exist_ok=True)
    csv_filename = os.path.join(OUT_PATH, f"optimization_results_{timeframe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    
    # Progress tracking
    completed_count = 0
    progress_lock = Lock()
    
    def run_and_save(combo, combo_index):
        """Run backtest and save result"""
        nonlocal completed_count
        
        # Create progress prefix showing which combination
        progress_prefix = f"   [Combo {combo_index + 1}/{total_combinations}] "
        # Run backtest (with individual results saving if enabled)
        results = _run_single_backtest(asset_name, timeframe, initial_balance, max_loss, combo, verbose=False, progress_prefix=progress_prefix, direct_file_path=direct_file_path, save_individual_results=SAVE_INDIVIDUAL_RESULTS_IN_OPTIMIZATION)
        
        if results is None:
            return None
        
        # Create result row with all parameters and results
        result_row = combo.copy()
        result_row.update({
            'total_pnl': results['total_pnl'],
            'total_trades': results['total_trades'],
            'win_rate': results['win_rate'],
            'final_balance': results['final_balance'],
            'max_drawdown': results['max_drawdown'],
            'max_drawdown_pct': results['max_drawdown_pct'],
            'total_fees': results['total_fees'],
            'winning_trades': results['winning_trades'],
            'losing_trades': results['losing_trades'],
            'avg_win': results['avg_win'],
            'avg_loss': results['avg_loss'],
            'profit_factor': results['profit_factor'],
            'total_return': results['total_return'],
            'net_profit': results['net_profit'],
            'largest_win': results['largest_win'],
            'largest_loss': results['largest_loss'],
            'trades_per_day': results['trades_per_day'],
            'backtest_period_days': results['backtest_period_days']
        })
        
        # Save to CSV
        _save_result_to_csv(result_row, csv_filename)
        
        # Update progress
        with progress_lock:
            completed_count += 1
            progress_pct = (completed_count / total_combinations) * 100
            print(f"   [{completed_count}/{total_combinations}] ({progress_pct:.1f}%) Completed | P&L: ${results['total_pnl']:,.2f} | Trades: {results['total_trades']}")
        
        return result_row
    
    # Run with thread pool
    print("🚀 Starting multithreaded backtests...\n")
    best_result = None
    best_pnl = float('-inf')
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_combo = {
            executor.submit(run_and_save, combo, i): (combo, i) 
            for i, combo in enumerate(combinations)
        }
        
        # Process completed tasks
        for future in as_completed(future_to_combo):
            result = future.result()
            if result and result['total_pnl'] > best_pnl:
                best_pnl = result['total_pnl']
                best_result = result
    
    # Final summary
    print(f"\n{'='*60}")
    print(f"🎯 OPTIMIZATION COMPLETE")
    print(f"{'='*60}")
    print(f"Total Combinations Tested: {completed_count:,}")
    print(f"Results saved to: {csv_filename}")
    
    if best_result:
        print(f"\nBest Result:")
        print(f"   Total P&L: ${best_result['total_pnl']:,.2f}")
        print(f"   Total Return: {best_result['total_return']:.2f}%")
        print(f"   Total Trades: {best_result['total_trades']}")
        print(f"   Win Rate: {best_result['win_rate']:.2f}%")
        print(f"   Max Drawdown: ${best_result['max_drawdown']:,.2f} ({best_result['max_drawdown_pct']:.2f}%)")
        print(f"\nOptimal Parameters:")
        for param_name in OPTIMIZATION_CONFIG.keys():
            if param_name in best_result:
                print(f"   {param_name}: {best_result[param_name]}")
    
    print(f"{'='*60}\n")
    
    return best_result, csv_filename


def optimize_strategy_random(asset_name, timeframe=None, initial_balance=50000.0, max_loss=2000, max_workers=4, num_samples=100, direct_file_path=None, timeframes_list=None, csv_input_file=None):
    """
    Random search optimization: test random parameter combinations in parallel.
    Can also read parameter combinations from CSV file if csv_input_file is provided.
    Saves results to CSV after each combination completes.
    
    Args:
        asset_name: Asset name
        timeframe: Optional specific timeframe (if None and timeframes_list provided, timeframe will be random)
        initial_balance: Starting balance
        max_loss: Maximum loss threshold
        max_workers: Number of parallel workers
        num_samples: Number of random samples to test (ignored if csv_input_file is provided)
        direct_file_path: Optional direct file path (will be determined from timeframe if None)
        timeframes_list: List of timeframes to randomly choose from (e.g., ["5min", "30min", "1h"])
        csv_input_file: Optional path to CSV file with parameter combinations (if provided, uses CSV instead of random)
    """
    # Check if we should use CSV input
    use_csv_input = csv_input_file is not None and os.path.exists(csv_input_file)
    
    # Initialize timeframe flags
    use_random_timeframes = False
    use_csv_timeframes = False
    
    if use_csv_input:
        print(f"\n{'='*60}")
        print(f"🔍 Starting Optimization from CSV Input")
        print(f"{'='*60}")
        print(f"Asset: {asset_name}")
        print(f"CSV File: {csv_input_file}")
        print(f"Initial Balance: ${initial_balance:,.2f} | Max Loss: ${max_loss:,.2f}")
        print(f"Max Workers: {max_workers}\n")
        
        # Load combinations from CSV
        combinations = _load_parameter_combinations_from_csv(csv_input_file)
        if not combinations:
            print("❌ No valid parameter combinations found in CSV. Exiting.")
            return None, None
        
        total_combinations = len(combinations)
        print(f"   Loaded {total_combinations} parameter combinations from CSV\n")
        
        # Check if CSV has timeframe column
        use_csv_timeframes = any('timeframe' in combo for combo in combinations)
        if not use_csv_timeframes and timeframe is None:
            timeframe = BACKTEST_SELECTED_TIMEFRAME
            print(f"ℹ️  CSV has no timeframe column. Using default timeframe: {timeframe}")
    else:
        # Determine if we're using random timeframes
        use_random_timeframes = (timeframe is None and timeframes_list is not None)
        
        if use_random_timeframes:
            print(f"\n{'='*60}")
            print(f"🔍 Starting Random Search Optimization (with Random Timeframes)")
            print(f"{'='*60}")
            print(f"Asset: {asset_name} | Timeframes: {', '.join(timeframes_list)} (random)")
            print(f"Initial Balance: ${initial_balance:,.2f} | Max Loss: ${max_loss:,.2f}")
            print(f"Max Workers: {max_workers} | Random Samples: {num_samples}\n")
        else:
            print(f"\n{'='*60}")
            print(f"🔍 Starting Random Search Optimization")
            print(f"{'='*60}")
            print(f"Asset: {asset_name} | Timeframe: {timeframe}")
            print(f"Initial Balance: ${initial_balance:,.2f} | Max Loss: ${max_loss:,.2f}")
            print(f"Max Workers: {max_workers} | Random Samples: {num_samples}\n")
        
        # Generate random combinations (with timeframe if using random timeframes)
        print("📊 Generating random parameter combinations...")
        combinations = _generate_random_combinations(num_samples, include_timeframe=use_random_timeframes, available_timeframes=timeframes_list)
        total_combinations = len(combinations)
        print(f"   Generated {total_combinations} random combinations to test\n")
    
    # CSV filename - use single file if random timeframes or CSV input, otherwise per timeframe
    os.makedirs(OUT_PATH, exist_ok=True)
    if use_csv_input or use_csv_timeframes:
        csv_filename = os.path.join(OUT_PATH, f"optimization_results_from_csv_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    elif use_random_timeframes:
        csv_filename = os.path.join(OUT_PATH, f"optimization_results_all_timeframes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    else:
        csv_filename = os.path.join(OUT_PATH, f"optimization_results_{timeframe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    
    # Progress tracking
    completed_count = 0
    progress_lock = Lock()
    # Track which line each sample is using for progress display
    # Use a queue-based system to assign lines in order of execution start
    available_lines = list(range(max_workers))  # Available line slots (0 to max_workers-1)
    sample_to_line = {}  # {combo_index: line_number} - tracks which line each sample uses
    line_to_sample = {}  # {line_number: combo_index} - reverse mapping to track which sample uses each line
    
    def run_and_save(combo, combo_index):
        """Run backtest and save result"""
        nonlocal completed_count
        
        # Extract timeframe from combo if present (from CSV or random), otherwise use provided timeframe
        if use_csv_timeframes and 'timeframe' in combo:
            combo_timeframe = combo.pop('timeframe')
        elif use_random_timeframes:
            combo_timeframe = combo.pop('timeframe', timeframe)
        else:
            combo_timeframe = timeframe
        
        combo_direct_file_path = direct_file_path
        # Determine file path based on timeframe
        if (use_csv_timeframes or use_random_timeframes) and combo_timeframe in TIMEFRAME_FILE_MAP:
            # Get file path for the timeframe from CSV or random selection
            combo_direct_file_path = os.path.join("data", asset_name, TIMEFRAME_FILE_MAP[combo_timeframe])
        
        # Assign a unique line number for this backtest's progress
        # Use a queue-based system: assign next available line, reuse when sample completes
        with progress_lock:
            if combo_index not in sample_to_line:
                # Get next available line (or reuse one if all are taken)
                if available_lines:
                    line_number = available_lines.pop(0)
                else:
                    # All lines in use, reuse the first one (oldest sample)
                    # Find the oldest sample using a line
                    oldest_line = min(line_to_sample.keys())
                    old_sample = line_to_sample.pop(oldest_line)
                    sample_to_line.pop(old_sample, None)
                    line_number = oldest_line
                
                sample_to_line[combo_index] = line_number
                line_to_sample[line_number] = combo_index
            else:
                line_number = sample_to_line[combo_index]
        
        # Calculate absolute line number: base offset + relative line number
        # This ensures progress starts after "Starting multithreaded backtests..." message
        absolute_line_number = progress_base_line_offset + line_number
        
        # Create progress prefix with line positioning
        if use_csv_timeframes or use_random_timeframes:
            tf_display = f" [{combo_timeframe}]"
        else:
            tf_display = ""
        progress_prefix = f"   [Sample {combo_index + 1}/{total_combinations}{tf_display}] "
        # Run backtest with absolute line number for progress display (with individual results saving if enabled)
        results = _run_single_backtest(asset_name, combo_timeframe, initial_balance, max_loss, combo, verbose=False, progress_prefix=progress_prefix, progress_line=absolute_line_number, direct_file_path=combo_direct_file_path, save_individual_results=SAVE_INDIVIDUAL_RESULTS_IN_OPTIMIZATION)
        
        # Release the line when done
        with progress_lock:
            if combo_index in sample_to_line:
                released_line = sample_to_line.pop(combo_index)
                line_to_sample.pop(released_line, None)
                # Add line back to available pool
                if released_line not in available_lines:
                    available_lines.append(released_line)
                    available_lines.sort()  # Keep sorted for consistent assignment
        
        if results is None:
            print(
                f"⚠️  Backtest failed for sample {combo_index + 1}/{total_combinations} "
                f"(timeframe: {combo_timeframe}).",
                file=sys.stderr
            )
            return None
        
        # Create result row with all parameters and results
        result_row = combo.copy()
        # Add timeframe to result if using CSV or random timeframes
        if use_csv_timeframes or use_random_timeframes:
            result_row['timeframe'] = combo_timeframe
        result_row.update({
            'total_pnl': results['total_pnl'],
            'total_trades': results['total_trades'],
            'win_rate': results['win_rate'],
            'final_balance': results['final_balance'],
            'max_drawdown': results['max_drawdown'],
            'max_drawdown_pct': results['max_drawdown_pct'],
            'total_fees': results['total_fees'],
            'winning_trades': results['winning_trades'],
            'losing_trades': results['losing_trades'],
            'avg_win': results['avg_win'],
            'avg_loss': results['avg_loss'],
            'profit_factor': results['profit_factor'],
            'total_return': results['total_return'],
            'net_profit': results['net_profit'],
            'largest_win': results['largest_win'],
            'largest_loss': results['largest_loss'],
            'trades_per_day': results['trades_per_day'],
            'backtest_period_days': results['backtest_period_days']
        })
        
        # Save to CSV
        _save_result_to_csv(result_row, csv_filename)
        
        # Update progress
        with progress_lock:
            completed_count += 1
            progress_pct = (completed_count / total_combinations) * 100
            tf_info = f" | TF: {combo_timeframe}" if (use_csv_timeframes or use_random_timeframes) else ""
            print(f"   [{completed_count}/{total_combinations}] ({progress_pct:.1f}%) Completed | P&L: ${results['total_pnl']:,.2f} | Trades: {results['total_trades']}{tf_info}")
        
        return result_row
    
    # Run with thread pool
    print("🚀 Starting multithreaded backtests...")
    # Reserve lines for progress display - each sample gets its own line
    # Reserve enough lines for active workers (each worker can run one sample at a time)
    # Add 10 extra lines to offset as requested
    max_reserved_lines = max_workers
    for _ in range(max_reserved_lines + 10):  # Reserve worker lines + 10 extra for spacing
        print("", file=sys.stderr)  # Reserve lines for progress
    sys.stderr.flush()
    
    # Update base line offset: lines start after "Starting..." message + 10 extra lines
    progress_base_line_offset = 11  # Start after "Starting..." message (line 1) + 10 extra lines
    
    best_result = None
    best_pnl = float('-inf')
    
    if USE_MULTIPROCESSING:
        manager = multiprocessing.Manager()
        progress_queue = manager.Queue()
        def progress_listener():
            while True:
                msg = progress_queue.get()
                if msg is None:
                    break
                combo_index, combo_timeframe, pct, current_bar, total_bars = msg
                tf_info = f" | TF: {combo_timeframe}" if (use_csv_timeframes or use_random_timeframes) else ""
                print(f"   [Sample {combo_index + 1}/{total_combinations}] {pct:.0f}% ({current_bar}/{total_bars} bars){tf_info}")
        listener_thread = Thread(target=progress_listener, daemon=True)
        listener_thread.start()
        # Multiprocessing: run backtests in separate processes, write CSV in parent
        tasks = []
        for idx, combo in enumerate(combinations):
            combo_copy = combo.copy()
            if use_csv_timeframes and 'timeframe' in combo_copy:
                combo_timeframe = combo_copy.pop('timeframe')
            elif use_random_timeframes:
                combo_timeframe = combo_copy.pop('timeframe', timeframe)
            else:
                combo_timeframe = timeframe

            combo_direct_file_path = direct_file_path
            if (use_csv_timeframes or use_random_timeframes) and combo_timeframe in TIMEFRAME_FILE_MAP:
                combo_direct_file_path = os.path.join("data", asset_name, TIMEFRAME_FILE_MAP[combo_timeframe])
            tasks.append((idx, asset_name, combo_timeframe, initial_balance, max_loss, combo_copy, combo_direct_file_path, progress_queue))

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {executor.submit(_run_backtest_worker, task): i for i, task in enumerate(tasks)}
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    combo_used, combo_timeframe, results, error_msg = future.result()
                    if results is None:
                        if error_msg:
                            print(f"   ⚠️  Backtest failed for combo {idx + 1}: {error_msg}")
                        else:
                            print(f"   ⚠️  Backtest failed for combo {idx + 1}: no results returned")
                        continue

                    result_row = combo_used.copy()
                    if use_csv_timeframes or use_random_timeframes:
                        result_row['timeframe'] = combo_timeframe
                    result_row.update({
                        'total_pnl': results['total_pnl'],
                        'total_trades': results['total_trades'],
                        'win_rate': results['win_rate'],
                        'final_balance': results['final_balance'],
                        'max_drawdown': results['max_drawdown'],
                        'max_drawdown_pct': results['max_drawdown_pct'],
                        'total_fees': results['total_fees'],
                        'winning_trades': results['winning_trades'],
                        'losing_trades': results['losing_trades'],
                        'avg_win': results['avg_win'],
                        'avg_loss': results['avg_loss'],
                        'profit_factor': results['profit_factor'],
                        'total_return': results['total_return'],
                        'net_profit': results['net_profit'],
                        'largest_win': results['largest_win'],
                        'largest_loss': results['largest_loss'],
                        'trades_per_day': results['trades_per_day'],
                        'backtest_period_days': results['backtest_period_days'],
                    })
                    _save_result_to_csv(result_row, csv_filename)

                    completed_count += 1
                    progress_pct = (completed_count / total_combinations) * 100
                    tf_info = f" | TF: {combo_timeframe}" if (use_csv_timeframes or use_random_timeframes) else ""
                    print(
                        f"   [{completed_count}/{total_combinations}] ({progress_pct:.1f}%) Completed | "
                        f"P&L: ${results['total_pnl']:,.2f} | Trades: {results['total_trades']}{tf_info}"
                    )

                    if results['total_pnl'] > best_pnl:
                        best_pnl = results['total_pnl']
                        best_result = result_row
                except Exception as e:
                    print(f"   ⚠️  Error processing combination {idx + 1}: {e}")
        progress_queue.put(None)
        listener_thread.join()
        manager.shutdown()
    elif USE_MULTITHREADING:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_combo = {
                executor.submit(run_and_save, combo, idx): (combo, idx)
                for idx, combo in enumerate(combinations)
            }
            
            # Process completed tasks
            for future in as_completed(future_to_combo):
                combo, idx = future_to_combo[future]
                try:
                    result = future.result()
                    if result is not None and result['total_pnl'] > best_pnl:
                        best_pnl = result['total_pnl']
                        best_result = result
                except Exception as e:
                    print(f"   ⚠️  Error processing combination {idx + 1}: {e}")
    else:
        for idx, combo in enumerate(combinations):
            try:
                result = run_and_save(combo, idx)
                if result is not None and result['total_pnl'] > best_pnl:
                    best_pnl = result['total_pnl']
                    best_result = result
            except Exception as e:
                print(f"   ⚠️  Error processing combination {idx + 1}: {e}")
    
    # Print summary
    print(f"\n{'='*60}")
    if use_csv_input:
        print(f"✅ CSV Input Optimization Complete")
    else:
        print(f"✅ Random Search Optimization Complete")
    print(f"{'='*60}")
    if best_result:
        print(f"🏆 Best Result:")
        if (use_csv_timeframes or use_random_timeframes) and 'timeframe' in best_result:
            print(f"   Timeframe: {best_result['timeframe']}")
        print(f"   Total P&L: ${best_result['total_pnl']:,.2f}")
        print(f"   Total Trades: {best_result['total_trades']}")
        print(f"   Win Rate: {best_result['win_rate']:.2f}%")
        print(f"   Final Balance: ${best_result['final_balance']:,.2f}")
        print(f"   Max Drawdown: ${best_result['max_drawdown']:,.2f} ({best_result['max_drawdown_pct']:.2f}%)")
        print(f"\n   Best Parameters:")
        for param_name in OPTIMIZATION_CONFIG.keys():
            if param_name in best_result:
                print(f"      {param_name}: {best_result[param_name]}")
        print(f"\n   Results saved to: {csv_filename}")
    else:
        print("   ⚠️  No valid results found")
    print(f"{'='*60}\n")
    
    return best_result, csv_filename


def optimize_strategy(asset_name, timeframe, initial_balance=50000.0, max_loss=2000, max_workers=4, direct_file_path=None):
    """
    Greedy optimization: optimize one parameter at a time using multithreading.
    For each parameter, tests all values in parallel, then moves to next parameter.
    Starts with current values, finds best for each parameter sequentially.
    """
    print(f"\n{'='*60}")
    print(f"🔍 Starting Greedy Optimization (Multithreaded)")
    print(f"{'='*60}")
    print(f"Asset: {asset_name} | Timeframe: {timeframe}")
    print(f"Initial Balance: ${initial_balance:,.2f} | Max Loss: ${max_loss:,.2f}")
    print(f"Max Workers: {max_workers}\n")
    
    # Get current parameters from strategyClass
    best_params = _get_current_strategy_params()
    
    # Optimization order (can be customized)
    param_order = [
        'FVG_HISTORY_NBR',
        'MIN_FVG_POWER_PCT',
        'HTF_TF',
        'EMA_PERIOD',
        'VOLUME_MULTIPLIER',
        'ATR_PERIOD',
        'SL_MULTIPLIER',
        'TP_MULTIPLIER',
        'USE_TRAILING',
        'TRAIL_OFFSET_MULT',
        'HOLD_UNTIL_OPPOSITE'
    ]
    
    best_pnl = None
    
    # Start with current configuration
    print("📊 Testing current configuration...")
    current_results = _run_single_backtest(asset_name, timeframe, initial_balance, max_loss, best_params, verbose=True, direct_file_path=direct_file_path, save_individual_results=SAVE_INDIVIDUAL_RESULTS_IN_OPTIMIZATION)
    if current_results is not None:
        best_pnl = current_results['total_pnl']
        print(f"   Current P&L: ${best_pnl:,.2f}\n")
    else:
        print("   ⚠️  Failed to run backtest with current configuration\n")
        return None
    
    # Optimize each parameter
    for param_name in param_order:
        if param_name not in OPTIMIZATION_CONFIG:
            continue
        
        print(f"🔧 Optimizing {param_name}...")
        print(f"   Current value: {best_params.get(param_name, 'N/A')}")
        
        # Get all test values for this parameter
        test_values = OPTIMIZATION_CONFIG[param_name]['range']
        param_best_value = best_params.get(param_name)
        param_best_pnl = best_pnl
        
        # Filter out current best value (we already tested it)
        test_values_to_check = [v for v in test_values if v != param_best_value]
        
        if not test_values_to_check:
            print(f"   ✓ No other values to test (keeping current value)\n")
            continue
        
        print(f"   Testing {len(test_values_to_check)} values in parallel...")
        
        # Progress tracking for this parameter
        completed_count = 0
        progress_lock = Lock()
        results_dict = {}  # Store results: {test_value: results}
        
        def test_single_value(test_value):
            """Test a single parameter value"""
            nonlocal completed_count
            
            # Create test params
            test_params = best_params.copy()
            test_params[param_name] = test_value
            
            # Run backtest with progress prefix (with individual results saving if enabled)
            progress_prefix = f"   [{param_name}={test_value}] "
            test_results = _run_single_backtest(asset_name, timeframe, initial_balance, max_loss, test_params, verbose=False, progress_prefix=progress_prefix, direct_file_path=direct_file_path, save_individual_results=SAVE_INDIVIDUAL_RESULTS_IN_OPTIMIZATION)
            
            # Store result
            results_dict[test_value] = test_results
            
            # Update progress
            with progress_lock:
                completed_count += 1
                progress_pct = (completed_count / len(test_values_to_check)) * 100
                if test_results is not None:
                    pnl_str = f"P&L: ${test_results['total_pnl']:,.2f}"
                    print(f"   [{completed_count}/{len(test_values_to_check)}] ({progress_pct:.1f}%) {param_name}={test_value} → {pnl_str}")
                else:
                    print(f"   [{completed_count}/{len(test_values_to_check)}] ({progress_pct:.1f}%) {param_name}={test_value} → Failed")
            
            return test_value, test_results
        
        # Test all values in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(test_single_value, test_value): test_value 
                for test_value in test_values_to_check
            }
            
            # Wait for all to complete
            for future in as_completed(futures):
                future.result()  # Wait for completion
        
        # Find best result from all tested values
        for test_value, test_results in results_dict.items():
            if test_results is not None and test_results['total_pnl'] > param_best_pnl:
                param_best_value = test_value
                param_best_pnl = test_results['total_pnl']
        
        # Update best params
        if param_best_value != best_params.get(param_name):
            best_params[param_name] = param_best_value
            best_pnl = param_best_pnl
            print(f"   ✅ Updated {param_name} to {param_best_value} (P&L: ${best_pnl:,.2f})\n")
        else:
            print(f"   ✓ Keeping current value (P&L: ${best_pnl:,.2f})\n")
    
    # Final summary
    print(f"\n{'='*60}")
    print(f"🎯 OPTIMIZATION COMPLETE")
    print(f"{'='*60}")
    print(f"Best Total P&L: ${best_pnl:,.2f}")
    print(f"\nOptimal Parameters:")
    for param_name in param_order:
        if param_name in best_params:
            print(f"   {param_name}: {best_params[param_name]}")
    print(f"{'='*60}\n")
    
    # Run final backtest with best parameters and generate full report
    print("📊 Running final backtest with optimal parameters...")
    _apply_strategy_params(best_params)
    
    # Capture parameters after applying them
    captured_params = {}
    for param_name in OPTIMIZATION_CONFIG.keys():
        if hasattr(strategyClass, param_name):
            captured_params[param_name] = getattr(strategyClass, param_name)
    for param_name in FIXED_PARAMS.keys():
        if hasattr(strategyClass, param_name):
            captured_params[param_name] = getattr(strategyClass, param_name)
    
    historical_data, asset_tuple, contract_id = load_backtest_data(asset_name, timeframe, direct_file_path=direct_file_path)
    if historical_data is not None:
        backtest = BacktestStrategy(
            asset_tuple=asset_tuple,
            historical_data=historical_data,
            initial_balance=initial_balance,
            max_loss=max_loss,
            asset_name=asset_name,
            strategy_params=captured_params,
        )
        results = backtest.run_backtest()
        backtest.generate_report(results)
    
    return best_params, best_pnl


# ==================== MAIN EXECUTION ====================

def run_backtest_example(asset_name=BACKTEST_ASSET_NAME, timeframe=BACKTEST_SELECTED_TIMEFRAME, 
                         initial_balance=BACKTEST_INITIAL_BALANCE, max_loss=BACKTEST_MAX_LOSS, direct_file_path=None,
                         start_date=BACKTEST_START_DATE, end_date=BACKTEST_END_DATE):
    """Example of how to run a backtest"""
    historical_data, asset_tuple, contract_id = load_backtest_data(asset_name, timeframe, direct_file_path=direct_file_path)
    
    if historical_data is None:
        print("❌ Failed to load data. Check asset name and timeframe.")
        return
    
    print(f"✅ Loaded data for {asset_name} ({contract_id})")
    print(f"   Timeframe: {timeframe}")
    print(f"   Bars: {len(historical_data):,}")

    
    # Capture current strategy parameters
    captured_params = {}
    for param_name in OPTIMIZATION_CONFIG.keys():
        if hasattr(strategyClass, param_name):
            captured_params[param_name] = getattr(strategyClass, param_name)
    for param_name in FIXED_PARAMS.keys():
        if hasattr(strategyClass, param_name):
            captured_params[param_name] = getattr(strategyClass, param_name)
    
    backtest = BacktestStrategy(
        asset_tuple=asset_tuple,
        historical_data=historical_data,
        initial_balance=initial_balance,
        start_date=start_date,
        end_date=end_date,
        max_loss=max_loss,
        asset_name=asset_name,
        strategy_params=captured_params,
    )
    
    # Run the backtest first
    results = backtest.run_backtest()
    
    # Get next available result ID AFTER backtest completes (to avoid duplicate IDs)
    result_id = _get_next_result_id()
    print(f"📁 Result ID: {result_id}")
    backtest.result_id = result_id
    
    # Generate report (which will save files using the result_id)
    backtest.generate_report(results)


if __name__ == "__main__":
    # Initialize data structures once
    initialize_backtest_data()
    # Configuration for backtest
    asset_name = BACKTEST_ASSET_NAME
    initial_balance = BACKTEST_INITIAL_BALANCE
    max_loss = BACKTEST_MAX_LOSS
    
    # Get optimal worker count (auto-detect or use manual setting)
    optimal_workers = get_optimal_worker_count()
    
    # Calculate date range from 5min data if USE_FIRST_TENTH_ONLY is enabled
    # This ensures all timeframes use the same date period (first 205 days from 5min data)
    if USE_FIRST_TENTH_ONLY:
        print("📅 Calculating date range from 5min data (first 205 days)...")
        _calculate_date_range_from_5min(asset_name)
        print()
    

    
    # Timeframe selection
    # For single backtest: choose one timeframe ("5min", "30min", or "1h")
    # For optimization: will iterate through all timeframes
    SELECTED_TIMEFRAME = BACKTEST_SELECTED_TIMEFRAME  # Change this to "5min", "30min", or "1h" for single backtest
    
    # Timeframes to test for optimization (skip 15min as already done)
    timeframes_to_test = BACKTEST_TIMEFRAMES_TO_TEST
    
    # =================================================
    direct_file_path = None
    if USE_DIRECT_DATA_FILE and DIRECT_DATA_FILE_PATH:
        direct_file_path = DIRECT_DATA_FILE_PATH
        print(f"📂 Using direct data file: {direct_file_path}")
    
    if RUN_OPTIMIZATION:
        if USE_CSV_INPUT:
            # CSV Input: read parameter combinations from CSV file
            print(f"\n{'='*60}")
            print(f"🔄 Optimization from CSV Input")
            print(f"📂 CSV File: {CSV_INPUT_FILE}")
            print(f"{'='*60}\n")
            
            optimize_strategy_random(asset_name, timeframe=None, initial_balance=initial_balance, max_loss=max_loss, 
                                    max_workers=optimal_workers, num_samples=RANDOM_SEARCH_SAMPLES, 
                                    direct_file_path=direct_file_path, timeframes_list=timeframes_to_test, 
                                    csv_input_file=CSV_INPUT_FILE)
        elif USE_EXHAUSTIVE_SEARCH:
            if direct_file_path:
                print(f"\n{'='*60}")
                print(f"🔄 Exhaustive Search (Direct File)")
                print(f"📂 Data file: {direct_file_path}")
                print(f"{'='*60}\n")
                optimize_strategy_multithreaded(asset_name, SELECTED_TIMEFRAME, initial_balance, max_loss, optimal_workers, direct_file_path=direct_file_path)
            else:
                # Exhaustive search: run for each timeframe separately
                for timeframe in timeframes_to_test:
                    # Get file path from mapping
                    if timeframe in TIMEFRAME_FILE_MAP:
                        mapped_file_path = os.path.join("data", asset_name, TIMEFRAME_FILE_MAP[timeframe])
                        print(f"\n{'='*60}")
                        print(f"🔄 Processing timeframe: {timeframe}")
                        print(f"📂 Data file: {mapped_file_path}")
                        print(f"{'='*60}\n")
                        
                        optimize_strategy_multithreaded(asset_name, timeframe, initial_balance, max_loss, optimal_workers, direct_file_path=mapped_file_path)
                    else:
                        print(f"⚠️  Warning: No file mapping for timeframe {timeframe}, skipping...")
        else:
            # Random search: timeframe is randomly selected for each sample
            if direct_file_path:
                print(f"\n{'='*60}")
                print(f"🔄 Random Search (Direct File)")
                print(f"📂 Data file: {direct_file_path}")
                print(f"{'='*60}\n")
                
                optimize_strategy_random(asset_name, timeframe=SELECTED_TIMEFRAME, initial_balance=initial_balance, max_loss=max_loss, 
                                        max_workers=optimal_workers, num_samples=RANDOM_SEARCH_SAMPLES, 
                                        direct_file_path=direct_file_path, timeframes_list=None)
            else:
                print(f"\n{'='*60}")
                print(f"🔄 Random Search with Random Timeframes")
                print(f"📂 Timeframes: {', '.join(timeframes_to_test)}")
                print(f"{'='*60}\n")
                
                optimize_strategy_random(asset_name, timeframe=None, initial_balance=initial_balance, max_loss=max_loss, 
                                        max_workers=optimal_workers, num_samples=RANDOM_SEARCH_SAMPLES, 
                                        direct_file_path=None, timeframes_list=timeframes_to_test)
    else:
        # Run single backtest for selected timeframe
        if direct_file_path:
            print(f"\n{'='*60}")
            print(f"🔄 Processing timeframe: {SELECTED_TIMEFRAME}")
            print(f"📂 Data file: {direct_file_path}")
            print(f"{'='*60}\n")
            
            run_backtest_example(
                asset_name,
                SELECTED_TIMEFRAME,
                initial_balance,
                max_loss,
                direct_file_path=direct_file_path,
                start_date=BACKTEST_START_DATE,
                end_date=BACKTEST_END_DATE
            )
        elif SELECTED_TIMEFRAME in TIMEFRAME_FILE_MAP:
            mapped_file_path = os.path.join("data", asset_name, TIMEFRAME_FILE_MAP[SELECTED_TIMEFRAME])
            print(f"\n{'='*60}")
            print(f"🔄 Processing timeframe: {SELECTED_TIMEFRAME}")
            print(f"📂 Data file: {mapped_file_path}")
            print(f"{'='*60}\n")
            
            run_backtest_example(
                asset_name,
                SELECTED_TIMEFRAME,
                initial_balance,
                max_loss,
                direct_file_path=mapped_file_path,
                start_date=BACKTEST_START_DATE,
                end_date=BACKTEST_END_DATE
            )
        else:
            print(f"❌ Error: Invalid timeframe '{SELECTED_TIMEFRAME}'")
            print(f"   Available timeframes: {', '.join(TIMEFRAME_FILE_MAP.keys())}")
