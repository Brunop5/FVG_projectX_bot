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
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import strategyClass
from strategyClass import Strategy, Order, ASSETS
from api_functions import fetch_data, load_data
from indicators import ema, sma, get_atr
import json
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import inspect
import re
import sys
from io import StringIO
import threading
from threading import Lock
from concurrent.futures import ThreadPoolExecutor, as_completed
import itertools


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

# Parameters that should NOT be optimized (keep current values)
FIXED_PARAMS = {
    'USE_VOLUME_CHECK': True,
    'VOLUME_DATA_START_TIMESTAMP': 1755464400000
}

# ==================== OPTIMIZATION SETTINGS ====================
RUN_OPTIMIZATION = True  # Set to True to run optimization, False for single backtest
USE_EXHAUSTIVE_SEARCH = False  # If True: test ALL parameter combinations in parallel (exhaustive). If False: greedy optimization
MAX_WORKERS = 4  # Number of parallel threads for multithreaded optimization

# ==================== GLOBAL DATA STRUCTURES ====================
# These are loaded once and reused across all backtests
CONTRACTS_DATA = {}
CONTRACTS_BY_NAME = {}
ROUND_TURN_FEES = {}
RESULTS_CSV_LOCK = Lock()  # Lock for CSV writing


# ==================== INITIALIZATION FUNCTIONS ====================

def _load_contracts_data():
    """Load contracts.csv data into global structures"""
    global CONTRACTS_DATA, CONTRACTS_BY_NAME
    
    contracts_path = os.path.join(os.path.dirname(__file__), "contracts.csv")
    if not os.path.exists(contracts_path):
        print(f"⚠️  Warning: contracts.csv not found at {contracts_path}")
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

def get_round_turn_fee(asset_id):
    """Get round turn fee per contract for a given asset ID"""
    return ROUND_TURN_FEES.get(asset_id, 0.0)


def get_contract_info(asset_id):
    """Get contract tick size and value for a given asset ID"""
    return CONTRACTS_DATA.get(asset_id, {'tick_size': None, 'tick_value': None})


def get_contract_id_by_name(asset_name):
    """Get contract ID from asset name (e.g., 'MESH6' -> 'CON.F.US.MES.H26')"""
    return CONTRACTS_BY_NAME.get(asset_name)


def load_backtest_data(asset_name, timeframe):
    """
    Load historical data for backtesting using asset name and timeframe.
    
    Args:
        asset_name: Asset name from contracts.csv (e.g., "MESH6", "MNQH6")
        timeframe: Timeframe string (e.g., "5min", "15min", "30min", "1h")
    
    Returns:
        tuple: (historical_data DataFrame, asset_tuple, contract_id)
               Returns (None, None, None) if asset not found or data file missing
    """
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
    
    historical_data = pd.read_csv(data_path)
    
    # Ensure timestamp column exists and is properly formatted
    if 'timestamp' not in historical_data.columns and 'date' in historical_data.columns:
        historical_data['timestamp'] = pd.to_datetime(historical_data['date'])
    elif 'timestamp' in historical_data.columns:
        if pd.api.types.is_numeric_dtype(historical_data['timestamp']):
            historical_data['timestamp'] = pd.to_datetime(historical_data['timestamp'], unit='ms', utc=True)
        else:
            historical_data['timestamp'] = pd.to_datetime(historical_data['timestamp'], utc=True)
    
    asset_tuple = (contract_id, timeframe, "backtest_account")
    return historical_data, asset_tuple, contract_id


# ==================== BACKTEST ORDER CLASS ====================

class BacktestOrder(Order):
    """Mock Order class for backtesting - tracks fills and P&L instead of placing real orders"""
    
    def __init__(self, side: str, entry_price: float, take_profit: float, 
                 trailing_stop_loss, entry_atr: float, account_id, asset_id, 
                 auth_token, lot_size=None, tick_size=None, tick_value=None, round_turn_fee=None):
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
        self.fees = 0.0
    
    def place_order(self, fill_time=None, entry_bar=None):
        """Mock order placement - just marks as filled immediately at entry price"""
        self.filled = True
        self.fill_price = self.entry_price
        self.fill_time = fill_time or datetime.now()
        self.entry_bar = entry_bar
        return {'success': True, 'order_id': 'backtest_order', 'message': 'Order filled'}
    
    def close_order(self):
        """Mock order closing - calculates P&L using tick value if available"""
        if not self.filled or self.exit_price is None or self.pnl != 0.0:
            return
        
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
        self.pnl = gross_pnl - self.fees
        
        # Calculate P&L percentage
        entry_value = self.fill_price * self.lot_size
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
                 max_loss=None, asset_name=None):
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
        """
        super().__init__(asset_tuple)
        self.asset_name = asset_name
        
        # Filter data by date range if provided
        if 'timestamp' in historical_data.columns:
            if pd.api.types.is_numeric_dtype(historical_data['timestamp']):
                historical_data['timestamp'] = pd.to_datetime(historical_data['timestamp'], unit='ms', utc=True)
            else:
                historical_data['timestamp'] = pd.to_datetime(historical_data['timestamp'], utc=True)
            
            if start_date:
                historical_data = historical_data[historical_data['timestamp'] >= pd.to_datetime(start_date)]
            if end_date:
                historical_data = historical_data[historical_data['timestamp'] <= pd.to_datetime(end_date)]
            
            # Filter by volume data start timestamp if volume check is enabled
            if strategyClass.USE_VOLUME_CHECK:
                volume_start_datetime = pd.to_datetime(strategyClass.VOLUME_DATA_START_TIMESTAMP, unit='ms', utc=True)
                before_count = len(historical_data)
                historical_data = historical_data[historical_data['timestamp'] >= volume_start_datetime]
                after_count = len(historical_data)
                if before_count > after_count:
                    print(f"📊 Filtered data: Removed {before_count - after_count:,} bars before {volume_start_datetime.strftime('%Y-%m-%d %H:%M:%S')} (unreliable volume data)")
                    print(f"   Remaining bars: {after_count:,}")
        
        self.historical_data = historical_data.sort_values('timestamp').reset_index(drop=True)
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.current_bar_index = 0
        self.trades = []
        self.equity_curve = []
        
        # Get contract information
        asset_id = asset_tuple[0]
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
    
    def init_rest(self):
        """Override init_rest to use historical data instead of API"""
        self.account_balance = self.initial_balance
        self.current_balance = self.initial_balance
        self.active_order = None
        
        min_bars_needed = max(100, 50)
        if len(self.historical_data) < min_bars_needed:
            raise ValueError(f"Not enough historical data. Need at least {min_bars_needed} bars, got {len(self.historical_data)}")
        
        self.data = self.historical_data.iloc[:min_bars_needed].copy()
        self.current_bar_index = min_bars_needed
        
        print(f"📊 Backtest initialized with {len(self.data)} initial bars")
        print(f"📅 Date range: {self.data['timestamp'].iloc[0]} to {self.historical_data['timestamp'].iloc[-1]}")
        
        self.cur_close = self.data["close"].iloc[-1]
        self.cur_volume = self.data["volume"].iloc[-1]
        
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
        
        htf_minutes = int(strategyClass.HTF_TF)
        
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

        if len(self.fvg_zones) > strategyClass.FVG_HISTORY_NBR:
            self.fvg_zones = self.fvg_zones[-strategyClass.FVG_HISTORY_NBR:]
    
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
        self.cur_volume = self.data["volume"].iloc[-1]
        
        return self.data.iloc[-1:]
    
    def update_trend_indicators(self):
        """Override to use cached HTF data for performance"""
        if self.htf_data is None:
            self._update_trend_indicators_resample()
            return
        
        current_timestamp = self.data['timestamp'].iloc[-1]
        idx = self.htf_data['timestamp'].searchsorted(current_timestamp, side='right')
        
        if idx < strategyClass.EMA_PERIOD:
            self.isBullishHTF = None
            self.isBearishHTF = None
        else:
            htf_close = self.htf_data['close'].iloc[:idx]
            htfEMA = ema(htf_close, strategyClass.EMA_PERIOD)
            
            if htfEMA is not None:
                self.isBullishHTF = self.cur_close > htfEMA
                self.isBearishHTF = self.cur_close < htfEMA
            else:
                self.isBullishHTF = None
                self.isBearishHTF = None
        
        # Calculate marketOK, lastBullFvg, and lastBearFvg (same as live strategy)
        vol_sma = sma(self.data["volume"], 20)
        volOK = self.cur_volume > vol_sma * strategyClass.VOLUME_MULTIPLIER if vol_sma is not None else False
        atrVal = get_atr(self.data, strategyClass.ATR_PERIOD)
        atr_sma = sma(atrVal, 20) if len(atrVal) > 0 else None
        atrOK = atrVal.iloc[-1] > atr_sma if (len(atrVal) > 0 and atr_sma is not None) else False
        
        if strategyClass.USE_VOLUME_CHECK:
            self.marketOK = volOK and atrOK
        else:
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
        
        htf_minutes = int(strategyClass.HTF_TF)
        current_tf_minutes = self._get_timeframe_minutes(self.timeframe)
        bars_per_htf = htf_minutes // current_tf_minutes
        
        if len(htf_data) < strategyClass.EMA_PERIOD * bars_per_htf:
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
            
            htf_resampled = htf_data.resample(resample_period, label='right', closed='right').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
            
            if len(htf_resampled) >= strategyClass.EMA_PERIOD:
                htf_close = htf_resampled['close']
                htfEMA = ema(htf_close, strategyClass.EMA_PERIOD)
                
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
        atrVal = get_atr(self.data, strategyClass.ATR_PERIOD)
        atr_sma = sma(atrVal, 20) if len(atrVal) > 0 else None
        atrOK = atrVal.iloc[-1] > atr_sma if (len(atrVal) > 0 and atr_sma is not None) else False
        
        if strategyClass.USE_VOLUME_CHECK:
            vol_sma = sma(self.data["volume"], 20)
            volOK = self.cur_volume > vol_sma * strategyClass.VOLUME_MULTIPLIER if vol_sma is not None else False
            self.marketOK = volOK and atrOK
        else:
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
        
        return self.daily_trades_count < strategyClass.MAX_DAILY_TRADES
    
    def entry_logic(self):
        """Override to use BacktestOrder instead of Order"""
        if len(self.fvg_zones) == 0 or self.inPosition:
            return
        
        if not self.check_daily_trade_limit():
            return
        
        current_high = self.data["high"].iloc[-1]
        current_low = self.data["low"].iloc[-1]
        
        atr = get_atr(self.data, strategyClass.ATR_PERIOD).iloc[-1]
        
        for zone in self.fvg_zones[-strategyClass.FVG_HISTORY_NBR:]:
            if zone["mitigated"]:
                continue
            
            fvg_bottom = zone["bottom"]
            fvg_top = zone["top"]
            touchesFVG = current_high >= fvg_bottom and current_low <= fvg_top
            
            if (zone["direction"] == "bull" and touchesFVG and 
                self.isBullishHTF and self.marketOK):
                trailStop = self.cur_close - atr * strategyClass.SL_MULTIPLIER
                tp = self.cur_close + atr * strategyClass.TP_MULTIPLIER
                entryAtr = atr
                lot_size = self.calculate_lot_size(atr, strategyClass.SL_MULTIPLIER)
                
                current_time = self.data.iloc[-1].get('timestamp', datetime.now())
                self.active_order = BacktestOrder("BUY", self.cur_close, tp, trailStop, 
                                                  entryAtr, self.account_id, self.asset, 
                                                  self.auth_token, lot_size, 
                                                  tick_size=self.tick_size, tick_value=self.tick_value,
                                                  round_turn_fee=self.round_turn_fee)
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
                trailStop = self.cur_close + atr * strategyClass.SL_MULTIPLIER
                tp = self.cur_close - atr * strategyClass.TP_MULTIPLIER
                entryAtr = atr
                lot_size = self.calculate_lot_size(atr, strategyClass.SL_MULTIPLIER)
                
                current_time = self.data.iloc[-1].get('timestamp', datetime.now())
                self.active_order = BacktestOrder("SELL", self.cur_close, tp, trailStop, 
                                                  entryAtr, self.account_id, self.asset, 
                                                  self.auth_token, lot_size,
                                                  tick_size=self.tick_size, tick_value=self.tick_value,
                                                  round_turn_fee=self.round_turn_fee)
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
            'bars_held': self.current_bar_index - order.entry_bar if order.entry_bar else 0
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
            if strategyClass.USE_TRAILING and pos.entry_atr is not None:
                potentialStop = current_high - pos.entry_atr * strategyClass.TRAIL_OFFSET_MULT
                if pos.trailing_stop_loss is not None:
                    new_stop = max(pos.trailing_stop_loss, potentialStop)
                    if new_stop > pos.trailing_stop_loss:
                        pos.trailing_stop_loss = new_stop
                else:
                    pos.trailing_stop_loss = potentialStop
        
        if self.inPosition and self.lastPositionWasShort:
            if strategyClass.USE_TRAILING and pos.entry_atr is not None:
                potentialStop = current_low + pos.entry_atr * strategyClass.TRAIL_OFFSET_MULT
                if pos.trailing_stop_loss is not None:
                    new_stop = min(pos.trailing_stop_loss, potentialStop)
                    if new_stop < pos.trailing_stop_loss:
                        pos.trailing_stop_loss = new_stop
                else:
                    pos.trailing_stop_loss = potentialStop
        
        # Check BOS/CHoCH exits (same as live strategy)
        if strategyClass.HOLD_UNTIL_OPPOSITE and self.inPosition:
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
            if pos.side == "BUY":
                price_diff = current_price - pos.fill_price
            else:  # SELL
                price_diff = pos.fill_price - current_price
            
            if pos.tick_size is not None and pos.tick_value is not None and pos.tick_size > 0:
                ticks = price_diff / pos.tick_size
                unrealized_pnl = ticks * pos.tick_value * pos.lot_size
            else:
                unrealized_pnl = price_diff * pos.lot_size
        
        self.equity_curve.append({
            'bar': self.current_bar_index,
            'timestamp': self.data.iloc[-1].get('timestamp', datetime.now()),
            'balance': self.current_balance,
            'equity': self.current_balance + unrealized_pnl
        })
    
    def run_backtest(self, show_progress=True, progress_prefix="", suppress_header=False):
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
        progress_interval = max(1, total_bars // 20)
        
        while self.current_bar_index < len(self.historical_data) and not self.strategy_failed:
            self.fetch_new_data()
            self.calculate_indicators()
            self.add_fvg_zones()
            self.entry_logic()
            self.update_stops()
            
            if show_progress and self.current_bar_index % progress_interval == 0:
                progress = (self.current_bar_index / total_bars) * 100
                progress_msg = f"{progress_prefix}Progress: {progress:.1f}% ({self.current_bar_index}/{total_bars} bars)"
                print(progress_msg, end='\r', file=sys.stderr)
                sys.stderr.flush()
            
            # Check max loss after each bar (in case of open position drawdown)
            if self.active_order and self.active_order.filled:
                current_bar = self.data.iloc[-1]
                current_price = current_bar['close']
                
                if self.active_order.side == "BUY":
                    price_diff = current_price - self.active_order.fill_price
                else:  # SELL
                    price_diff = self.active_order.fill_price - current_price
                
                if (self.active_order.tick_size is not None and 
                    self.active_order.tick_value is not None and 
                    self.active_order.tick_size > 0):
                    ticks = price_diff / self.active_order.tick_size
                    unrealized_pnl = ticks * self.active_order.tick_value * self.active_order.lot_size
                else:
                    unrealized_pnl = price_diff * self.active_order.lot_size
                
                current_equity = self.current_balance + unrealized_pnl
                total_loss = self.initial_balance - current_equity
                if self.max_loss_amount is not None and total_loss >= self.max_loss_amount:
                    self.active_order.exit_price = current_price
                    self.active_order.exit_time = current_bar.get('timestamp', datetime.now())
                    self.active_order.exit_reason = "Max Loss Reached"
                    self.active_order.close_order()
                    self._record_trade(self.active_order)
                    self._close_position()
        
        # Close any open position at end
        if self.active_order and self.active_order.filled and not self.strategy_failed:
            current_bar = self.data.iloc[-1]
            self.active_order.exit_price = current_bar['close']
            self.active_order.exit_time = current_bar.get('timestamp', datetime.now())
            self.active_order.exit_reason = "End of Data"
            self.active_order.close_order()
            self._record_trade(self.active_order)
            self._close_position()
        
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
        
        # Save results
        trades_df.to_csv(f"backtest_trades_{self.asset}_{datetime.now().strftime('%Y%m%d')}.csv", index=False)
        equity_df = pd.DataFrame(self.equity_curve)
        equity_df.to_csv(f"backtest_equity_{self.asset}_{datetime.now().strftime('%Y%m%d')}.csv", index=False)
        print(f"💾 Results saved to CSV files")
        
        # Plot equity curve
        self._plot_equity_curve(equity_df)
    
    def _plot_equity_curve(self, equity_df):
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
        
        plot_filename = f"backtest_equity_curve_{self.asset}_{datetime.now().strftime('%Y%m%d')}.png"
        plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
        print(f"📈 Equity curve plot saved to {plot_filename}")


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


def _run_single_backtest(asset_name, timeframe, initial_balance, max_loss, params_dict, verbose=False, progress_prefix=""):
    """Run a single backtest with given parameters and return results"""
    # Suppress output if not verbose (but allow progress through stderr)
    if not verbose:
        old_stdout = sys.stdout
        sys.stdout = StringIO()
    
    try:
        # Apply parameters
        _apply_strategy_params(params_dict)
        
        # Load data
        historical_data, asset_tuple, contract_id = load_backtest_data(asset_name, timeframe)
        if historical_data is None:
            return None
        
        # Create and run backtest
        backtest = BacktestStrategy(
            asset_tuple=asset_tuple,
            historical_data=historical_data,
            initial_balance=initial_balance,
            max_loss=max_loss,
            asset_name=asset_name,
        )
        
        # Run with progress shown even when not verbose
        results = backtest.run_backtest(show_progress=True, progress_prefix=progress_prefix, suppress_header=not verbose)
        
        # Clear progress line when done
        if not verbose and progress_prefix:
            print("", file=sys.stderr)  # New line after progress
        
        return results
    except Exception as e:
        if verbose:
            print(f"❌ Error in backtest: {e}")
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


def optimize_strategy_multithreaded(asset_name, timeframe, initial_balance=50000.0, max_loss=2000, max_workers=4):
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
    
    # CSV filename
    csv_filename = f"optimization_results_{asset_name}_{timeframe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    # Progress tracking
    completed_count = 0
    progress_lock = Lock()
    
    def run_and_save(combo, combo_index):
        """Run backtest and save result"""
        nonlocal completed_count
        
        # Create progress prefix showing which combination
        progress_prefix = f"   [Combo {combo_index + 1}/{total_combinations}] "
        # Run backtest
        results = _run_single_backtest(asset_name, timeframe, initial_balance, max_loss, combo, verbose=False, progress_prefix=progress_prefix)
        
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


def optimize_strategy(asset_name, timeframe, initial_balance=50000.0, max_loss=2000, max_workers=4):
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
    current_results = _run_single_backtest(asset_name, timeframe, initial_balance, max_loss, best_params, verbose=True)
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
            
            # Run backtest with progress prefix
            progress_prefix = f"   [{param_name}={test_value}] "
            test_results = _run_single_backtest(asset_name, timeframe, initial_balance, max_loss, test_params, verbose=False, progress_prefix=progress_prefix)
            
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
    historical_data, asset_tuple, contract_id = load_backtest_data(asset_name, timeframe)
    if historical_data is not None:
        backtest = BacktestStrategy(
            asset_tuple=asset_tuple,
            historical_data=historical_data,
            initial_balance=initial_balance,
            max_loss=max_loss,
            asset_name=asset_name,
        )
        results = backtest.run_backtest()
        backtest.generate_report(results)
    
    return best_params, best_pnl


# ==================== MAIN EXECUTION ====================

def run_backtest_example(asset_name="MGCG6", timeframe="15min", 
                         initial_balance=50000.0, max_loss=2000):
    """Example of how to run a backtest"""
    historical_data, asset_tuple, contract_id = load_backtest_data(asset_name, timeframe)
    
    if historical_data is None:
        print("❌ Failed to load data. Check asset name and timeframe.")
        return
    
    print(f"✅ Loaded data for {asset_name} ({contract_id})")
    print(f"   Timeframe: {timeframe}")
    print(f"   Bars: {len(historical_data):,}")
    
    backtest = BacktestStrategy(
        asset_tuple=asset_tuple,
        historical_data=historical_data,
        initial_balance=initial_balance,
        max_loss=max_loss,
        asset_name=asset_name,
    )
    
    results = backtest.run_backtest()
    backtest.generate_report(results)


if __name__ == "__main__":
    # Initialize data structures once
    initialize_backtest_data()
    
    asset_name = "MGCG6"
    timeframe = "15min"
    initial_balance = 50000.0
    max_loss = 2000
    # =================================================
    
    if RUN_OPTIMIZATION:
        if USE_EXHAUSTIVE_SEARCH:
            optimize_strategy_multithreaded(asset_name, timeframe, initial_balance, max_loss, MAX_WORKERS)
        else:
            optimize_strategy(asset_name, timeframe, initial_balance, max_loss, MAX_WORKERS)
    else:
        run_backtest_example(asset_name, timeframe, initial_balance, max_loss)
