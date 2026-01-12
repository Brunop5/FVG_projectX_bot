"""
Backtesting module for the FVG trading strategy.

This module allows you to backtest the Strategy class without modifying it.
It works by:
1. Creating a BacktestStrategy wrapper that inherits from Strategy
2. Overriding API-dependent methods to use historical data instead
3. Running the strategy logic sequentially on historical bars
4. Tracking trades, P&L, and performance metrics
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from strategyClass import Strategy, Order, SL_MULTIPLIER, TP_MULTIPLIER, USE_TRAILING, TRAIL_OFFSET_MULT, HOLD_UNTIL_OPPOSITE, ASSETS
from api_functions import fetch_data, load_data
from indicators import ema, sma, get_atr
import json
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ==================== MODULE-LEVEL INITIALIZATION ====================
# Load contracts.csv once at module import time for efficiency in parallel execution
CONTRACTS_DATA = {}
CONTRACTS_BY_NAME = {}  # Map asset name to contract ID
ROUND_TURN_FEES = {}  # Map asset ID to round turn fee per contract

try:
    contracts_path = os.path.join(os.path.dirname(__file__), "contracts.csv")
    if os.path.exists(contracts_path):
        contracts_df = pd.read_csv(contracts_path)
        for _, row in contracts_df.iterrows():
            CONTRACTS_DATA[row['id']] = {
                'tick_size': float(row['tickSize']),
                'tick_value': float(row['tickValue'])
            }
            # Map asset name to contract ID
            asset_name = row['name']
            CONTRACTS_BY_NAME[asset_name] = row['id']
        print(f"📋 Loaded {len(CONTRACTS_DATA)} contracts from contracts.csv")
    else:
        print(f"⚠️  Warning: contracts.csv not found at {contracts_path}")
except Exception as e:
    print(f"⚠️  Warning: Could not load contracts.csv: {e}")

# Load round turn fees from ASSETS in strategyClass
# ASSETS format can be:
# - 4 elements: [(asset_id, timeframe, account_name, round_turn_fee), ...]
# - 3 elements: [(asset_id, timeframe, account_name), ...] (no fee, will use 'assets' list as fallback)
try:
    assets_with_fees = 0
    for asset_tuple in ASSETS:
        if len(asset_tuple) >= 4:
            asset_id = asset_tuple[0]
            round_turn_fee = float(asset_tuple[3])  # 4th element is round turn fee
            ROUND_TURN_FEES[asset_id] = round_turn_fee
            assets_with_fees += 1
    if assets_with_fees > 0:
        print(f"💰 Loaded round turn fees for {assets_with_fees} assets from ASSETS")
except Exception as e:
    print(f"⚠️  Warning: Could not load round turn fees from ASSETS: {e}")

# Also load fees from the 'assets' list in strategyClass (if __main__ section)
# This list has format: [(asset_id, fee), ...] - the fee is the second element
# Only used as fallback if asset not found in ASSETS
try:
    import inspect
    import strategyClass
    import re
    
    # Read the source file to find the assets list
    source_file = inspect.getfile(strategyClass)
    with open(source_file, 'r') as f:
        source_code = f.read()
    
    # Find the assets list definition (look for "assets = [" pattern, can span multiple lines)
    # Match patterns like: assets = [("CON.F.US.MNQ.H26", 0.74), ("CON.F.US.MES.H26", 0.74), ...]
    assets_pattern = r'assets\s*=\s*\[(.*?)\]'
    match = re.search(assets_pattern, source_code, re.DOTALL | re.MULTILINE)
    
    if match:
        assets_str = match.group(1)
        # Parse the tuples: ("CON.F.US.MNQ.H26", 0.74) - handle both single and multi-line
        # Pattern matches: ("string", number) where number can be float
        tuple_pattern = r'\("([^"]+)",\s*([\d.]+)\)'
        tuples = re.findall(tuple_pattern, assets_str)
        
        fees_loaded = 0
        for asset_id, fee_str in tuples:
            try:
                fee = float(fee_str)
                # Only add if not already in ROUND_TURN_FEES (ASSETS takes precedence)
                if asset_id not in ROUND_TURN_FEES:
                    ROUND_TURN_FEES[asset_id] = fee
                    fees_loaded += 1
            except ValueError:
                continue
        
        if fees_loaded > 0:
            print(f"💰 Loaded {fees_loaded} additional round turn fees from 'assets' list in strategyClass.py")
except Exception as e:
    # Silently fail - this is optional
    pass


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
    Automatically finds contract ID and loads data from data/{asset_name}/{timeframe}.csv
    
    Args:
        asset_name: Asset name from contracts.csv (e.g., "MESH6", "MNQH6")
        timeframe: Timeframe string (e.g., "5min", "15min", "30min", "1h")
    
    Returns:
        tuple: (historical_data DataFrame, asset_tuple, contract_id)
               Returns (None, None, None) if asset not found or data file missing
    """
    # Get contract ID from asset name
    contract_id = get_contract_id_by_name(asset_name)
    if not contract_id:
        print(f"⚠️  Error: Asset name '{asset_name}' not found in contracts.csv")
        print(f"   Available assets: {', '.join(sorted(CONTRACTS_BY_NAME.keys()))}")
        return None, None, None
    
    # Clean asset name for folder name (same logic as in gather_historical_data)
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
        # Check if timestamp is numeric (milliseconds) or already datetime
        if pd.api.types.is_numeric_dtype(historical_data['timestamp']):
            # Convert from milliseconds (int64) to datetime
            historical_data['timestamp'] = pd.to_datetime(historical_data['timestamp'], unit='ms', utc=True)
        else:
            # Already datetime, just ensure it's timezone-aware
            historical_data['timestamp'] = pd.to_datetime(historical_data['timestamp'], utc=True)
    
    # Create asset_tuple
    asset_tuple = (contract_id, timeframe, "backtest_account")
    
    return historical_data, asset_tuple, contract_id


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
        self.fees = 0.0  # Track fees separately
    
    def place_order(self, fill_time=None, entry_bar=None):
        """Mock order placement - just marks as filled immediately at entry price"""
        self.filled = True
        self.fill_price = self.entry_price
        self.fill_time = fill_time or datetime.now()
        self.entry_bar = entry_bar
        return {'success': True, 'order_id': 'backtest_order', 'message': 'Order filled'}
    
    def close_order(self):
        """Mock order closing - calculates P&L using tick value if available"""
        # Return early if order not filled, exit_price not set, or P&L already calculated
        if not self.filled or self.exit_price is None or self.pnl != 0.0:
            return
        
        # Calculate price difference
        if self.side == "BUY":
            price_diff = self.exit_price - self.fill_price
        else:  # SELL
            price_diff = self.fill_price - self.exit_price
        
        # Calculate P&L using tick value if available, otherwise use simple calculation
        if self.tick_size is not None and self.tick_value is not None and self.tick_size > 0:
            # P&L = (price_diff / tick_size) * tick_value * lot_size
            ticks = price_diff / self.tick_size
            gross_pnl = ticks * self.tick_value * self.lot_size
        else:
            # Fallback to simple calculation if tick info not available
            gross_pnl = price_diff * self.lot_size
        
        # Apply round turn fees (charged once per complete trade)
        # Round turn fee is per contract, so multiply by lot_size
        self.fees = self.round_turn_fee * self.lot_size
        
        # Net P&L = Gross P&L - Fees
        self.pnl = gross_pnl - self.fees
        
        # Calculate P&L percentage based on entry value
        entry_value = self.fill_price * self.lot_size
        if entry_value > 0:
            self.pnl_pct = (self.pnl / entry_value) * 100
        else:
            self.pnl_pct = 0.0
        
        return {'success': True}


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
            historical_data: DataFrame with OHLCV data (must have columns: timestamp, open, high, low, close, volume)
            initial_balance: Starting account balance for backtest
            start_date: Start date for backtest (if None, uses all data)
            end_date: End date for backtest (if None, uses all data)
            max_loss: Maximum loss threshold. If between 0 and 1, treated as percentage (e.g., 0.2 = 20%).
                     If >= 1, treated as absolute dollar amount. If None, no limit.
            asset_name: Asset name from contracts.csv (e.g., "MESH6", "MNQH6") - used to load HTF data
        """
        super().__init__(asset_tuple)
        self.asset_name = asset_name  # Store asset name for loading HTF data
        
        # Filter data by date range if provided
        if 'timestamp' in historical_data.columns:
            # Check if timestamp is numeric (milliseconds) or already datetime
            if pd.api.types.is_numeric_dtype(historical_data['timestamp']):
                # Convert from milliseconds (int64) to datetime
                historical_data['timestamp'] = pd.to_datetime(historical_data['timestamp'], unit='ms', utc=True)
            else:
                # Already datetime, just ensure it's timezone-aware
                historical_data['timestamp'] = pd.to_datetime(historical_data['timestamp'], utc=True)
            
            if start_date:
                historical_data = historical_data[historical_data['timestamp'] >= pd.to_datetime(start_date)]
            if end_date:
                historical_data = historical_data[historical_data['timestamp'] <= pd.to_datetime(end_date)]
        
        self.historical_data = historical_data.sort_values('timestamp').reset_index(drop=True)
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.current_bar_index = 0
        self.trades = []
        self.equity_curve = []
        
        # Get contract information from pre-loaded CONTRACTS_DATA
        asset_id = asset_tuple[0]  # First element is the asset ID
        contract_info = get_contract_info(asset_id)
        self.tick_size = contract_info['tick_size']
        self.tick_value = contract_info['tick_value']
        self.round_turn_fee = get_round_turn_fee(asset_id)  # Get round turn fee per contract
        
        if self.tick_size is not None and self.tick_value is not None:
            fee_info = f" | Round Turn Fee: ${self.round_turn_fee:.2f}/contract" if self.round_turn_fee > 0 else " | ⚠️  No fee found in ASSETS"
            print(f"📋 Using contract info: {asset_id} | Tick Size: {self.tick_size} | Tick Value: ${self.tick_value}{fee_info}")
        else:
            print(f"⚠️  Warning: Contract {asset_id} not found in contracts.csv. Using simple P&L calculation.")
        if self.round_turn_fee > 0:
            print(f"💰 Round turn fees will be applied: ${self.round_turn_fee:.2f} per contract per trade")
        else:
            print(f"⚠️  Warning: No round turn fee found for {asset_id} in ASSETS. Add fee to ASSETS list in strategyClass.py")
        
        # Max loss configuration
        self.max_loss = max_loss
        if max_loss is not None:
            if 0 < max_loss < 1:
                # Percentage: convert to absolute dollar amount
                self.max_loss_amount = initial_balance * max_loss
                self.max_loss_type = "percentage"
            else:
                # Absolute dollar amount
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
        
        # Load initial data window (need enough for indicators)
        min_bars_needed = max(100, 50)  # Enough for indicators
        if len(self.historical_data) < min_bars_needed:
            raise ValueError(f"Not enough historical data. Need at least {min_bars_needed} bars, got {len(self.historical_data)}")
        
        # Set initial data window
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
        from strategyClass import HTF_TF
        
        self.htf_data = None
        self.htf_data_timestamps = None
        
        if not self.asset_name:
            return
        
        # Calculate HTF timeframe in minutes
        htf_minutes = int(HTF_TF)
        
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
        
        # Clean asset name for folder name
        safe_asset_name = "".join(c for c in self.asset_name if c.isalnum() or c in (' ', '-', '_')).strip()
        safe_asset_name = safe_asset_name.replace(' ', '_')
        
        # Load HTF CSV file
        htf_data_path = os.path.join("data", safe_asset_name, f"{htf_timeframe}.csv")
        
        if not os.path.exists(htf_data_path):
            return
        
        try:
            # Load HTF data once
            self.htf_data = pd.read_csv(htf_data_path)
            
            # Convert timestamp once
            if 'timestamp' in self.htf_data.columns:
                if pd.api.types.is_numeric_dtype(self.htf_data['timestamp']):
                    self.htf_data['timestamp'] = pd.to_datetime(self.htf_data['timestamp'], unit='ms', utc=True)
                else:
                    self.htf_data['timestamp'] = pd.to_datetime(self.htf_data['timestamp'], utc=True)
            
            # Pre-sort and create timestamp index for fast lookup
            self.htf_data = self.htf_data.sort_values('timestamp').reset_index(drop=True)
            self.htf_data_timestamps = self.htf_data['timestamp'].values
            
        except Exception as e:
            print(f"⚠️  Error loading HTF data from {htf_data_path}: {e}")
            self.htf_data = None
    
    def add_fvg_zones(self):
        """Override to remove print statements for performance"""
        # === FVG ZONE CREATION (equivalent to the box.new blocks) ===
        if self.bullishPowerOK and self.isBullishHTF and self.marketOK:
            # Bullish FVG uses low[1] as top and high[3] as bottom in Pine
            self.fvg_zones.append(
                {
                    "direction": "bull",
                    "top": self.data["low"].iloc[-2],     # low[1]
                    "bottom": self.data["high"].iloc[-4], # high[3]
                    "mitigated": False,
                }
            )

        if self.bearishPowerOK and self.isBearishHTF and self.marketOK:
            # Bearish FVG uses low[3] as top and high[1] as bottom in Pine
            self.fvg_zones.append(
                {
                    "direction": "bear",
                    "top": self.data["low"].iloc[-4],     # low[3]
                    "bottom": self.data["high"].iloc[-2], # high[1]
                    "mitigated": False,
                }
            )

        # Limit number of stored FVGs, similar to fvgHistoryNbr trimming in Pine
        from strategyClass import FVG_HISTORY_NBR
        if len(self.fvg_zones) > FVG_HISTORY_NBR:
            self.fvg_zones = self.fvg_zones[-FVG_HISTORY_NBR:]
    
    def gather_data(self):
        """Override to return initial data window"""
        return self.data
    
    def fetch_new_data(self):
        """Override to get next bar from historical data - optimized version"""
        if self.current_bar_index >= len(self.historical_data):
            return None
        
        # Update current data window efficiently (keep last 100 bars)
        # Use view instead of copy for better performance (pandas will handle it safely)
        start_idx = max(0, self.current_bar_index - 99)  # Keep last 100 bars
        end_idx = self.current_bar_index + 1
        # Only copy if we need to modify, otherwise use view
        self.data = self.historical_data.iloc[start_idx:end_idx]
        
        self.current_bar_index += 1
        
        self.cur_close = self.data["close"].iloc[-1]
        self.cur_volume = self.data["volume"].iloc[-1]
        
        return self.data.iloc[-1:]
    
    def update_trend_indicators(self):
        """Override to use cached HTF data for performance"""
        from strategyClass import EMA_PERIOD
        
        # Use cached HTF data if available
        if self.htf_data is None:
            self._update_trend_indicators_resample()
            return
        
        # Get current bar timestamp (already datetime from init)
        current_timestamp = self.data['timestamp'].iloc[-1]
        
        # Use binary search for efficient filtering (much faster than boolean indexing)
        # Find the index where timestamp <= current_timestamp
        # Use pandas Index.searchsorted for datetime compatibility
        idx = self.htf_data['timestamp'].searchsorted(current_timestamp, side='right')
        
        if idx < EMA_PERIOD:
            # Not enough HTF data yet
            self.isBullishHTF = None
            self.isBearishHTF = None
        else:
            # Use iloc for faster slicing (no copy needed)
            htf_close = self.htf_data['close'].iloc[:idx]
            htfEMA = ema(htf_close, EMA_PERIOD)
            
            if htfEMA is not None:
                self.isBullishHTF = self.cur_close > htfEMA
                self.isBearishHTF = self.cur_close < htfEMA
            else:
                self.isBullishHTF = None
                self.isBearishHTF = None
        
        # Calculate marketOK, lastBullFvg, and lastBearFvg (same as live strategy)
        from strategyClass import ATR_PERIOD
        volOK = self.cur_volume > sma(self.data["volume"], 20) * 1.2
        atrVal = get_atr(self.data, ATR_PERIOD)
        atrOK = atrVal.iloc[-1] > sma(atrVal, 20) if len(atrVal) > 0 else False
        self.marketOK = volOK and atrOK
        
        # Update FVG detection flags
        self.lastBullFvg = self.data["high"].iloc[-4] < self.data["low"].iloc[-2] and not self.lastBullFvg
        self.lastBearFvg = self.data["low"].iloc[-4] > self.data["high"].iloc[-2] and not self.lastBearFvg
    
    def _update_trend_indicators_resample(self):
        """Fallback method: resample current timeframe data to HTF (old method)"""
        from strategyClass import HTF_TF, EMA_PERIOD
        
        # Get all historical data up to current bar
        htf_data = self.historical_data.iloc[:self.current_bar_index].copy()
        
        # Convert timestamp to datetime for resampling
        if htf_data['timestamp'].dtype in ['int64', 'int32']:
            htf_data['timestamp'] = pd.to_datetime(htf_data['timestamp'], unit='ms', utc=True)
        else:
            htf_data['timestamp'] = pd.to_datetime(htf_data['timestamp'], utc=True)
        
        # Set timestamp as index for resampling
        htf_data = htf_data.set_index('timestamp')
        
        # Calculate HTF timeframe in minutes
        htf_minutes = int(HTF_TF)
        current_tf_minutes = self._get_timeframe_minutes(self.timeframe)
        
        # Calculate how many current bars = 1 HTF bar
        bars_per_htf = htf_minutes // current_tf_minutes
        
        if len(htf_data) < EMA_PERIOD * bars_per_htf:
            # Not enough data yet for HTF EMA
            self.isBullishHTF = None
            self.isBearishHTF = None
        else:
            # Resample to HTF timeframe
            if htf_minutes == 240:
                resample_period = '4h'
            elif htf_minutes == 120:
                resample_period = '2h'
            elif htf_minutes == 60:
                resample_period = '1h'
            elif htf_minutes == 1440:
                resample_period = '1d'
            else:
                # Fallback: use number of bars
                resample_period = f'{bars_per_htf * current_tf_minutes}min'
            
            # Resample OHLCV data properly
            htf_resampled = htf_data.resample(resample_period, label='right', closed='right').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
            
            if len(htf_resampled) >= EMA_PERIOD:
                # Calculate EMA on HTF resampled close prices
                htf_close = htf_resampled['close']
                htfEMA = ema(htf_close, EMA_PERIOD)
                
                if htfEMA is not None:
                    self.isBullishHTF = self.cur_close > htfEMA
                    self.isBearishHTF = self.cur_close < htfEMA
                else:
                    self.isBullishHTF = None
                    self.isBearishHTF = None
            else:
                self.isBullishHTF = None
                self.isBearishHTF = None
        
        # Calculate marketOK, lastBullFvg, and lastBearFvg (same as live strategy)
        from strategyClass import ATR_PERIOD
        volOK = self.cur_volume > sma(self.data["volume"], 20) * 1.2
        atrVal = get_atr(self.data, ATR_PERIOD)
        atrOK = atrVal.iloc[-1] > sma(atrVal, 20) if len(atrVal) > 0 else False
        self.marketOK = volOK and atrOK
        
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
            # Default to 1 minute if can't parse
            return 1
    
    def check_daily_trade_limit(self):
        """Check if maximum daily trades has been reached (same as live strategy)"""
        from strategyClass import MAX_DAILY_TRADES
        current_date = self.data['timestamp'].iloc[-1].date() if 'timestamp' in self.data.columns else datetime.now().date()
        
        if self.last_trade_date != str(current_date):
            # Reset counter for new day
            self.daily_trades_count = 0
            self.last_trade_date = str(current_date)
        
        return self.daily_trades_count < MAX_DAILY_TRADES
    
    def entry_logic(self):
        """Override to use BacktestOrder instead of Order"""
        if len(self.fvg_zones) == 0 or self.inPosition:
            return
        
        # Check daily trade limit (same as live strategy)
        from strategyClass import MAX_DAILY_TRADES, FVG_HISTORY_NBR, ATR_PERIOD
        if not self.check_daily_trade_limit():
            return
        
        current_high = self.data["high"].iloc[-1]
        current_low = self.data["low"].iloc[-1]
        
        atr = get_atr(self.data, ATR_PERIOD).iloc[-1]
        
        for zone in self.fvg_zones[-FVG_HISTORY_NBR:]:
            if zone["mitigated"]:
                continue
            
            fvg_bottom = zone["bottom"]
            fvg_top = zone["top"]
            touchesFVG = current_high >= fvg_bottom and current_low <= fvg_top
            
            if (zone["direction"] == "bull" and touchesFVG and 
                self.isBullishHTF and self.marketOK):
                trailStop = self.cur_close - atr * SL_MULTIPLIER
                tp = self.cur_close + atr * TP_MULTIPLIER
                entryAtr = atr
                lot_size = self.calculate_lot_size(atr, SL_MULTIPLIER)
                
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
                    # print(f"📈 LONG entry at {self.cur_close:.5f} | Bar: {self.current_bar_index}")
                break
            
            elif (zone["direction"] == "bear" and touchesFVG and 
                  self.isBearishHTF and self.marketOK):
                trailStop = self.cur_close + atr * SL_MULTIPLIER
                tp = self.cur_close - atr * TP_MULTIPLIER
                entryAtr = atr
                lot_size = self.calculate_lot_size(atr, SL_MULTIPLIER)
                
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
                    # print(f"📉 SHORT entry at {self.cur_close:.5f} | Bar: {self.current_bar_index}")
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
        
        # Check stop loss and take profit
        if order.side == "BUY":
            # Check if stop loss hit (price went below trailing_stop_loss)
            if current_low <= order.trailing_stop_loss:
                order.exit_price = order.trailing_stop_loss
                order.exit_time = current_time
                order.exit_reason = "Stop Loss"
                order.close_order()
                self._record_trade(order)
                self._close_position()
                return True
            
            # Check if take profit hit
            if current_high >= order.take_profit:
                order.exit_price = order.take_profit
                order.exit_time = current_time
                order.exit_reason = "Take Profit"
                order.close_order()
                self._record_trade(order)
                self._close_position()
                return True
        
        else:  # SELL
            # Check if stop loss hit (price went above trailing_stop_loss)
            if current_high >= order.trailing_stop_loss:
                order.exit_price = order.trailing_stop_loss
                order.exit_time = current_time
                order.exit_reason = "Stop Loss"
                order.close_order()
                self._record_trade(order)
                self._close_position()
                return True
            
            # Check if take profit hit
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
        
        # print(f"{'✅' if order.pnl > 0 else '❌'} Trade closed: {order.side} | "
        #       f"Entry: {order.fill_price:.5f} | Exit: {order.exit_price:.5f} | "
        #       f"P&L: ${order.pnl:.2f} ({order.pnl_pct:.2f}%) | Reason: {order.exit_reason}")
        
        # Check if max loss reached after recording trade
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
        # First check if we should exit
        if self.check_exits():
            return
        
        # Then update trailing stops (same as live)
        pos = self.active_order
        if pos is None:
            return
        
        current_high = self.data["high"].iloc[-1]
        current_low = self.data["low"].iloc[-1]
        
        if self.inPosition and self.lastPositionWasLong:
            if USE_TRAILING and pos.entry_atr is not None:
                potentialStop = current_high - pos.entry_atr * TRAIL_OFFSET_MULT
                if pos.trailing_stop_loss is not None:
                    new_stop = max(pos.trailing_stop_loss, potentialStop)
                    # Only update if the new stop is better (moves up)
                    if new_stop > pos.trailing_stop_loss:
                        pos.trailing_stop_loss = new_stop
                else:
                    pos.trailing_stop_loss = potentialStop
        
        if self.inPosition and self.lastPositionWasShort:
            if USE_TRAILING and pos.entry_atr is not None:
                potentialStop = current_low + pos.entry_atr * TRAIL_OFFSET_MULT
                if pos.trailing_stop_loss is not None:
                    new_stop = min(pos.trailing_stop_loss, potentialStop)
                    # Only update if the new stop is better (moves down)
                    if new_stop < pos.trailing_stop_loss:
                        pos.trailing_stop_loss = new_stop
                else:
                    pos.trailing_stop_loss = potentialStop
        
        # Check BOS/CHoCH exits (same as live strategy)
        if HOLD_UNTIL_OPPOSITE and self.inPosition:
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
            
            # Use tick value calculation if available
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
    
    def run_backtest(self):
        """Run the backtest on historical data"""
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
        
        # Initialize
        self.init_rest()
        
        # Process each bar
        total_bars = len(self.historical_data)
        progress_interval = max(1, total_bars // 20)  # Show progress every 5%
        
        # Track statistics for debugging
        stats = {
            'fvgs_created': 0,
            'market_ok_count': 0,
            'htf_bullish_count': 0,
            'htf_bearish_count': 0,
            'entry_attempts': 0,
            'entry_blocks': {
                'no_fvg_zones': 0,
                'in_position': 0,
                'daily_limit': 0,
                'conditions_not_met': 0
            }
        }
        
        while self.current_bar_index < len(self.historical_data) and not self.strategy_failed:
            self.fetch_new_data()
            self.calculate_indicators()
            
            # Track statistics
            if self.marketOK:
                stats['market_ok_count'] += 1
            if self.isBullishHTF:
                stats['htf_bullish_count'] += 1
            if self.isBearishHTF:
                stats['htf_bearish_count'] += 1
            
            fvgs_before = len(self.fvg_zones)
            self.add_fvg_zones()
            if len(self.fvg_zones) > fvgs_before:
                stats['fvgs_created'] += 1
            
            # Track entry logic
            if len(self.fvg_zones) == 0:
                stats['entry_blocks']['no_fvg_zones'] += 1
            elif self.inPosition:
                stats['entry_blocks']['in_position'] += 1
            else:
                stats['entry_attempts'] += 1
            
            self.entry_logic()
            self.update_stops()
            
            # Show progress periodically (not every bar to avoid slowdown)
            if self.current_bar_index % progress_interval == 0:
                progress = (self.current_bar_index / total_bars) * 100
                print(f"   Progress: {progress:.1f}% ({self.current_bar_index}/{total_bars} bars)", end='\r')
            
            # Check max loss after each bar (in case of open position drawdown)
            if self.active_order and self.active_order.filled:
                # Calculate current unrealized P&L using tick value if available
                current_bar = self.data.iloc[-1]
                current_price = current_bar['close']
                
                if self.active_order.side == "BUY":
                    price_diff = current_price - self.active_order.fill_price
                else:  # SELL
                    price_diff = self.active_order.fill_price - current_price
                
                # Use tick value calculation if available
                if (self.active_order.tick_size is not None and 
                    self.active_order.tick_value is not None and 
                    self.active_order.tick_size > 0):
                    ticks = price_diff / self.active_order.tick_size
                    unrealized_pnl = ticks * self.active_order.tick_value * self.active_order.lot_size
                else:
                    # Fallback to simple calculation
                    unrealized_pnl = price_diff * self.active_order.lot_size
                
                # Check if current balance + unrealized P&L would breach max loss
                current_equity = self.current_balance + unrealized_pnl
                total_loss = self.initial_balance - current_equity
                if self.max_loss_amount is not None and total_loss >= self.max_loss_amount:
                    # Close position immediately due to max loss
                    self.active_order.exit_price = current_price
                    self.active_order.exit_time = current_bar.get('timestamp', datetime.now())
                    self.active_order.exit_reason = "Max Loss Reached"
                    self.active_order.close_order()
                    self._record_trade(self.active_order)
                    self._close_position()
        
        # Close any open position at end (if not already closed due to max loss)
        if self.active_order and self.active_order.filled and not self.strategy_failed:
            current_bar = self.data.iloc[-1]
            self.active_order.exit_price = current_bar['close']
            self.active_order.exit_time = current_bar.get('timestamp', datetime.now())
            self.active_order.exit_reason = "End of Data"
            self.active_order.close_order()
            self._record_trade(self.active_order)
            self._close_position()
        
        # Generate report
        self.generate_report()
    
    def generate_report(self):
        """Generate backtest performance report"""
        if not self.trades:
            print("\n❌ No trades executed during backtest period")
            if self.strategy_failed:
                print(f"⚠️  STRATEGY STATUS: FAILED")
                print(f"   Reason: {self.failed_reason}")
            
            # Debug: Print diagnostic information
            print("\n🔍 DIAGNOSTIC INFORMATION:")
            print(f"   Total FVG zones created: {len(self.fvg_zones)}")
            print(f"   HTF Bullish: {self.isBullishHTF}")
            print(f"   HTF Bearish: {self.isBearishHTF}")
            print(f"   Market OK: {self.marketOK}")
            print(f"   Bullish Power OK: {self.bullishPowerOK}")
            print(f"   Bearish Power OK: {self.bearishPowerOK}")
            print(f"   In Position: {self.inPosition}")
            print(f"   Daily trades count: {self.daily_trades_count}")
            
            # Check if HTF data was loaded
            if self.asset_name:
                from strategyClass import HTF_TF
                htf_minutes = int(HTF_TF)
                if htf_minutes == 240:
                    htf_timeframe = "4h"
                elif htf_minutes == 120:
                    htf_timeframe = "2h"
                elif htf_minutes == 60:
                    htf_timeframe = "1h"
                elif htf_minutes == 1440:
                    htf_timeframe = "1d"
                else:
                    htf_timeframe = f"{htf_minutes}min"
                
                safe_asset_name = "".join(c for c in self.asset_name if c.isalnum() or c in (' ', '-', '_')).strip()
                safe_asset_name = safe_asset_name.replace(' ', '_')
                htf_data_path = os.path.join("data", safe_asset_name, f"{htf_timeframe}.csv")
                print(f"   HTF data path: {htf_data_path}")
                print(f"   HTF file exists: {os.path.exists(htf_data_path)}")
            
            # Print statistics
            if hasattr(self, 'debug_stats'):
                stats = self.debug_stats
                print(f"\n📊 BACKTEST STATISTICS:")
                print(f"   FVG zones created: {stats['fvgs_created']}")
                print(f"   Bars with marketOK=True: {stats['market_ok_count']} ({stats['market_ok_count']/self.current_bar_index*100:.1f}%)")
                print(f"   Bars with HTF Bullish: {stats['htf_bullish_count']} ({stats['htf_bullish_count']/self.current_bar_index*100:.1f}%)")
                print(f"   Bars with HTF Bearish: {stats['htf_bearish_count']} ({stats['htf_bearish_count']/self.current_bar_index*100:.1f}%)")
                print(f"   Entry attempts: {stats['entry_attempts']}")
                print(f"   Entry blocks:")
                print(f"      - No FVG zones: {stats['entry_blocks']['no_fvg_zones']}")
                print(f"      - Already in position: {stats['entry_blocks']['in_position']}")
                print(f"      - Daily limit reached: {stats['entry_blocks']['daily_limit']}")
            
            return
        
        trades_df = pd.DataFrame(self.trades)
        
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        losing_trades = len(trades_df[trades_df['pnl'] < 0])
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        total_pnl = trades_df['pnl'].sum()
        # Calculate total fees - ensure fees column exists and sum it
        if 'fees' in trades_df.columns:
            total_fees = trades_df['fees'].sum()
        else:
            # If fees column doesn't exist, calculate from round_turn_fee
            total_fees = self.round_turn_fee * trades_df['size'].sum() if self.round_turn_fee > 0 else 0.0
        avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0
        profit_factor = abs(avg_win * winning_trades / (avg_loss * losing_trades)) if losing_trades > 0 and avg_loss != 0 else float('inf')
        
        # Calculate total loss from losing trades only (sum of all negative P&L values)
        # Filter losing trades (pnl < 0) and sum their P&L values (which are negative)
        losing_trades_df = trades_df[trades_df['pnl'] < 0]
        if len(losing_trades_df) > 0:
            # Sum of negative values will be negative, so we take absolute value for display
            total_loss_from_trades = abs(losing_trades_df['pnl'].sum())
        else:
            total_loss_from_trades = 0.0
        
        final_balance = self.current_balance
        total_return = ((final_balance - self.initial_balance) / self.initial_balance) * 100
        net_loss = self.initial_balance - final_balance  # Net change (can be negative if profit)
        
        # Calculate day count based on actual backtest period (not full data range)
        if len(self.historical_data) > 0 and self.current_bar_index > 0:
            start_date = pd.to_datetime(self.historical_data['timestamp'].iloc[0])
            # Use the last bar that was actually processed, not the last bar in the data
            actual_end_index = min(self.current_bar_index - 1, len(self.historical_data) - 1)
            end_date = pd.to_datetime(self.historical_data['timestamp'].iloc[actual_end_index])
            day_count = (end_date - start_date).days + 1  # +1 to include both start and end days
            trades_per_day = total_trades / day_count if day_count > 0 else 0
        else:
            day_count = 0
            trades_per_day = 0
        
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
        print(f"Total Loss (from losing trades): ${total_loss_from_trades:,.2f}")
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
        print(f"\nLargest Win:         ${trades_df['pnl'].max():.2f}")
        print(f"Largest Loss:        ${trades_df['pnl'].min():.2f}")
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
        
        # Convert timestamp to datetime if needed
        if 'timestamp' in equity_df.columns:
            if pd.api.types.is_numeric_dtype(equity_df['timestamp']):
                equity_df['timestamp'] = pd.to_datetime(equity_df['timestamp'], unit='ms', utc=True)
            else:
                equity_df['timestamp'] = pd.to_datetime(equity_df['timestamp'], utc=True)
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # Plot 1: Equity curve
        ax1.plot(equity_df['timestamp'], equity_df['equity'], label='Equity', linewidth=1.5, color='#2E86AB')
        ax1.axhline(y=self.initial_balance, color='gray', linestyle='--', linewidth=1, label='Initial Balance', alpha=0.7)
        ax1.set_ylabel('Equity ($)', fontsize=11, fontweight='bold')
        ax1.set_title(f'Equity Curve - {self.asset}', fontsize=13, fontweight='bold', pad=10)
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.legend(loc='best', fontsize=9)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        # Plot 2: Balance (without unrealized P&L)
        ax2.plot(equity_df['timestamp'], equity_df['balance'], label='Balance', linewidth=1.5, color='#A23B72')
        ax2.axhline(y=self.initial_balance, color='gray', linestyle='--', linewidth=1, label='Initial Balance', alpha=0.7)
        ax2.set_xlabel('Date', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Balance ($)', fontsize=11, fontweight='bold')
        ax2.set_title('Account Balance', fontsize=12, fontweight='bold', pad=10)
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.legend(loc='best', fontsize=9)
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        # Format x-axis dates
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Add max loss line if applicable
        if self.max_loss_amount is not None:
            max_loss_line = self.initial_balance - self.max_loss_amount
            ax1.axhline(y=max_loss_line, color='red', linestyle=':', linewidth=1.5, label='Max Loss Threshold', alpha=0.7)
            ax2.axhline(y=max_loss_line, color='red', linestyle=':', linewidth=1.5, label='Max Loss Threshold', alpha=0.7)
            ax1.legend(loc='best', fontsize=9)
            ax2.legend(loc='best', fontsize=9)
        
        plt.tight_layout()
        
        # Save plot
        plot_filename = f"backtest_equity_curve_{self.asset}_{datetime.now().strftime('%Y%m%d')}.png"
        plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
        print(f"📈 Equity curve plot saved to {plot_filename}")
        
        # Show plot
        plt.show()


# ==================== USAGE EXAMPLE ====================

def run_backtest_example():
    """
    Example of how to run a backtest.
    
    Simply specify the asset name and timeframe - everything else is automatic!
    """
    
    # ========== CONFIGURATION - Only change these ==========
    asset_name = "MGCG6"      # Asset name from contracts.csv (e.g., "MESH6", "MNQH6", "MGCG6")
    timeframe = "15min"        # Timeframe (e.g., "5min", "15min", "30min", "1h")
    initial_balance = 50000.0 # Starting balance
    max_loss = 2000           # Max loss: 2000 = $2000, or 0.2 = 20% of balance
    # =======================================================
    
    # Load data automatically (finds contract ID and loads CSV)
    historical_data, asset_tuple, contract_id = load_backtest_data(asset_name, timeframe)
    
    if historical_data is None:
        print("❌ Failed to load data. Check asset name and timeframe.")
        return
    
    print(f"✅ Loaded data for {asset_name} ({contract_id})")
    print(f"   Timeframe: {timeframe}")
    print(f"   Bars: {len(historical_data):,}")
    
    # Create and run backtest
    backtest = BacktestStrategy(
        asset_tuple=asset_tuple,
        historical_data=historical_data,
        initial_balance=initial_balance,
        max_loss=max_loss,
        asset_name=asset_name,  # Pass asset name for HTF data loading
    )
    
    # Run backtest
    backtest.run_backtest()


if __name__ == "__main__":
    run_backtest_example()

