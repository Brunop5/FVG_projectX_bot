#!/usr/bin/env python3
"""
MT5 Strategy - Inherits from strategyClass but uses MetaTrader 5 instead of ProjectX API.
Only overrides API-dependent methods, keeping all strategy logic the same.
"""

import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta
import time
import threading
from strategyClass import (
    # Import all constants
    FVG_HISTORY_NBR, MIN_FVG_POWER_PCT, HTF_TF, EMA_PERIOD, VOLUME_MULTIPLIER,
    USE_VOLUME_CHECK, VOLUME_DATA_START_TIMESTAMP, START_FROM_VOLUME_TIMESTAMP,
    ATR_PERIOD, SL_MULTIPLIER, TP_MULTIPLIER, USE_TRAILING, TRAIL_OFFSET_MULT,
    HOLD_UNTIL_OPPOSITE, USE_FIXED_LOT, FIXED_LOT, RISK_PERCENT, ORDER_SIZE,
    MAX_DAILY_TRADES, ASSETS,
    # Import classes
    Order, Strategy
)
from api_functions import sleep_until_next_boundary, TIMEFRAME_SECONDS
from indicators import ema, sma, get_atr, crossover, crossunder
import json
import os

metadata_lock = threading.Lock()

# ==================== MT5 CONFIGURATION ====================
# MT5 Account Settings
MT5_LOGIN = os.getenv("LOGIN")
MT5_PASSWORD = os.getenv("PASSWORD")
MT5_SERVER = os.getenv("SERVER")
MT5_PATH = r"C:\Program Files\InstaForex MT5 Terminal\terminal64.exe"

# Asset mapping: Map strategyClass asset IDs to MT5 symbols
# Format: {"CON.F.US.GCE.G26": "GOLD", "CON.F.US.MNQ.H26": "MNQ", ...}
MT5_SYMBOL_MAP = {
    "CON.F.US.GCE.G26": "GOLD.m",
    "CON.F.US.MNQ.H26": "MNQ",
    # Add more mappings as needed
}

# Timeframe mapping: Map strategyClass timeframes to MT5 timeframes
MT5_TIMEFRAME_MAP = {
    "1min": mt5.TIMEFRAME_M1,
    "5min": mt5.TIMEFRAME_M5,
    "15min": mt5.TIMEFRAME_M15,
    "30min": mt5.TIMEFRAME_M30,
    "1h": mt5.TIMEFRAME_H1,
    "4h": mt5.TIMEFRAME_H4,
    "1d": mt5.TIMEFRAME_D1,
}
# ===========================================================


def init_mt5():
    """Initialize MT5 connection"""
    if not mt5.initialize(path=MT5_PATH):
        error = mt5.last_error()
        raise RuntimeError(f"❌ MT5 initialization failed: {error}")
    
    if MT5_LOGIN and MT5_PASSWORD and MT5_SERVER:
        authorized = mt5.login(MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER)
        if not authorized:
            error = mt5.last_error()
            raise RuntimeError(f"❌ MT5 login failed: {error}")
        print(f"✅ MT5 connected. Account: {MT5_LOGIN}, Server: {MT5_SERVER}")
    else:
        print("⚠️  MT5 login credentials not set. Using current terminal connection.")
    
    account_info = mt5.account_info()
    if account_info:
        print(f"   Balance: ${account_info.balance:.2f}, Equity: ${account_info.equity:.2f}")
    
    return True


def get_mt5_symbol(asset_id):
    """Convert strategyClass asset ID to MT5 symbol"""
    return MT5_SYMBOL_MAP.get(asset_id, asset_id)


def get_mt5_timeframe(timeframe_str):
    """Convert strategyClass timeframe string to MT5 timeframe"""
    return MT5_TIMEFRAME_MAP.get(timeframe_str, mt5.TIMEFRAME_M1)


def fetch_mt5_data(symbol, timeframe, count=100):
    """Fetch historical data from MT5"""
    mt5_symbol = get_mt5_symbol(symbol)
    mt5_tf = get_mt5_timeframe(timeframe)
    
    rates = mt5.copy_rates_from_pos(mt5_symbol, mt5_tf, 0, count)
    
    if rates is None or len(rates) == 0:
        print(f"⚠️  No data received for {symbol} ({mt5_symbol}) on {timeframe}")
        return None
    
    # Convert to DataFrame
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df = df.rename(columns={
        'time': 'timestamp',
        'open': 'open',
        'high': 'high',
        'low': 'low',
        'close': 'close',
        'tick_volume': 'volume'  # MT5 uses tick_volume
    })
    
    # Select only needed columns
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    df = df.reset_index(drop=True)
    
    return df


def get_mt5_account_balance():
    """Get account balance from MT5"""
    account_info = mt5.account_info()
    if account_info:
        return account_info.balance
    return None


class MT5Order(Order):
    """MT5 version of Order class - overrides only API-dependent methods"""
    
    def __init__(self, side: str, entry_price: float, take_profit: float, 
                 trailing_stop_loss, entry_atr, account_id, asset_id, auth_token, lot_size=None):
        super().__init__(side, entry_price, take_profit, trailing_stop_loss, 
                        entry_atr, account_id, asset_id, auth_token, lot_size)
        self.mt5_ticket = None  # Store MT5 order ticket
    
    def place_order(self):
        """Place order using MT5"""
        symbol = get_mt5_symbol(self.asset_id)
        
        # Get current price
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            return {'success': False, 'message': f'Symbol {symbol} not found'}
        
        if not symbol_info.visible:
            if not mt5.symbol_select(symbol, True):
                return {'success': False, 'message': f'Failed to select symbol {symbol}'}
        
        # Get current ask/bid prices
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return {'success': False, 'message': f'Failed to get tick for {symbol}'}
        
        # Determine entry price (use current market price)
        if self.side == "BUY":
            price = tick.ask
            order_type = mt5.ORDER_TYPE_BUY
            sl_price = self.trailing_stop_loss
            tp_price = self.take_profit
        else:  # SELL
            price = tick.bid
            order_type = mt5.ORDER_TYPE_SELL
            sl_price = self.trailing_stop_loss
            tp_price = self.take_profit
        
        # Normalize prices
        sl_price = mt5.symbol_info(symbol).normalize_price(sl_price)
        tp_price = mt5.symbol_info(symbol).normalize_price(tp_price)
        
        # Prepare request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": self.lot_size,
            "type": order_type,
            "price": price,
            "sl": sl_price,
            "tp": tp_price,
            "deviation": 20,  # Slippage in points
            "magic": 234000,  # Magic number for identification
            "comment": "FVG Strategy",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,  # Immediate or Cancel
        }
        
        # Send order
        result = mt5.order_send(request)
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            error_msg = f"Order failed: {result.comment} (retcode: {result.retcode})"
            print(f"❌ {error_msg}")
            return {
                'success': False,
                'order_id': None,
                'message': error_msg
            }
        
        self.mt5_ticket = result.order
        print(f"✅ Order placed successfully. Ticket: {self.mt5_ticket}")
        print(f"   Side: {self.side}, Size: {self.lot_size}, Entry: {result.price:.5f}")
        print(f"   TP: {tp_price:.5f}, SL: {sl_price:.5f}")
        
        return {
            'success': True,
            'order_id': self.mt5_ticket,
            'message': 'Order placed successfully'
        }
    
    def close_order(self):
        """Close position using MT5"""
        symbol = get_mt5_symbol(self.asset_id)
        
        # Get open positions
        positions = mt5.positions_get(symbol=symbol)
        if positions is None or len(positions) == 0:
            print(f"⚠️  No open position found for {symbol}")
            return {'success': False, 'message': 'No open position'}
        
        # Close all positions for this symbol
        for position in positions:
            if position.ticket == self.mt5_ticket or self.mt5_ticket is None:
                # Determine opposite order type
                if position.type == mt5.ORDER_TYPE_BUY:
                    order_type = mt5.ORDER_TYPE_SELL
                    price = mt5.symbol_info_tick(symbol).bid
                else:
                    order_type = mt5.ORDER_TYPE_BUY
                    price = mt5.symbol_info_tick(symbol).ask
                
                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": symbol,
                    "volume": position.volume,
                    "type": order_type,
                    "position": position.ticket,
                    "price": price,
                    "deviation": 20,
                    "magic": 234000,
                    "comment": "Close FVG Position",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_IOC,
                }
                
                result = mt5.order_send(request)
                if result.retcode == mt5.TRADE_RETCODE_DONE:
                    print(f"✅ Position closed. Ticket: {position.ticket}")
                    return {'success': True}
                else:
                    print(f"❌ Failed to close position: {result.comment}")
                    return {'success': False, 'message': result.comment}
        
        return {'success': False, 'message': 'Position not found'}


class MT5Strategy(Strategy):
    """MT5 version of Strategy class - overrides only API-dependent methods"""
    
    def __init__(self, asset_tuple):
        super().__init__(asset_tuple)
        self.mt5_symbol = get_mt5_symbol(self.asset)
        self.stop_flag = threading.Event()  # Flag to stop the strategy
    
    def init_rest(self):
        """Override to use MT5 for account info"""
        # MT5 doesn't use account_id/account_name like ProjectX
        # We'll use the account info directly from MT5
        account_info = mt5.account_info()
        if account_info:
            self.account_id = account_info.login
            self.account_balance = account_info.balance
        else:
            self.account_id = None
            self.account_balance = None
        
        self.active_order = None
        
        self.data = self.gather_data()
        if self.data is None or len(self.data) == 0:
            raise ValueError(f"Failed to load data for {self.asset} ({self.mt5_symbol})")
        
        print(f"📊 Loaded {len(self.data)} bars for {self.asset} ({self.mt5_symbol})")
        
        self.cur_close = self.data["close"].iloc[-1]
        self.cur_volume = self.data["volume"].iloc[-1]
        
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
        
        self.load_metadata()
        self.calculate_indicators()
        
        self.fvg_zones: list[dict] = []
        self.add_fvg_zones()
    
    def gather_data(self):
        """Override to fetch data from MT5"""
        return fetch_mt5_data(self.asset, self.timeframe, 100)
    
    def fetch_new_data(self):
        """Override to fetch new bar from MT5"""
        new_data = fetch_mt5_data(self.asset, self.timeframe, 1)
        if new_data is None or len(new_data) == 0:
            return None
        
        # Append to existing data
        new_row = new_data.iloc[-1:]
        self.data = pd.concat([self.data, new_row], ignore_index=True)
        
        # Keep only last 100 bars
        if len(self.data) > 100:
            self.data = self.data.tail(100).reset_index(drop=True)
        
        self.cur_close = self.data["close"].iloc[-1]
        self.cur_volume = self.data["volume"].iloc[-1]
        
        return new_row
    
    def update_trend_indicators(self):
        """Override to fetch HTF data from MT5"""
        htf_timeframe_str = f"{HTF_TF}min"
        bars = fetch_mt5_data(self.asset, htf_timeframe_str, max(101, EMA_PERIOD+51))
        
        if bars is None or len(bars) == 0:
            self.isBullishHTF = None
            self.isBearishHTF = None
        else:
            htfEMA = ema(bars, EMA_PERIOD)
            if htfEMA is not None:
                self.isBullishHTF = self.cur_close > htfEMA
                self.isBearishHTF = self.cur_close < htfEMA
            else:
                self.isBullishHTF = None
                self.isBearishHTF = None
        
        # Rest of the logic is the same as parent class
        atrVal = get_atr(self.data, ATR_PERIOD)
        atrOK = atrVal.iloc[-1] > sma(atrVal, 20)
        
        if USE_VOLUME_CHECK:
            volOK = self.cur_volume > sma(self.data["volume"], 20) * VOLUME_MULTIPLIER
            self.marketOK = volOK and atrOK
        else:
            self.marketOK = atrOK
        
        self.lastBullFvg = self.data["high"].iloc[-4] < self.data["low"].iloc[-2] and not self.lastBullFvg
        self.lastBearFvg = self.data["low"].iloc[-4] > self.data["high"].iloc[-2] and not self.lastBearFvg
    
    def entry_logic(self):
        """Override to use MT5Order instead of Order"""
        if len(self.fvg_zones) == 0 or self.inPosition:
            return
        
        if not self.check_daily_trade_limit():
            print(f"⚠️ Daily trade limit reached ({MAX_DAILY_TRADES}). No new trades today.")
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
                
                self.active_order = MT5Order("BUY", self.cur_close, tp, trailStop, entryAtr,
                                             self.account_id, self.asset, self.auth_token, lot_size)
                result = self.active_order.place_order()
                
                if result['success']:
                    zone["mitigated"] = True
                    self.lastPositionWasLong = True
                    self.lastPositionWasShort = False
                    self.inPosition = True
                    self.daily_trades_count += 1
                    self.last_trade_date = str(datetime.now().date())
                    self.pending_signal = None
                    print(f"📈 LONG position opened. Daily trades: {self.daily_trades_count}/{MAX_DAILY_TRADES}")
                break
            
            elif (zone["direction"] == "bear" and touchesFVG and 
                  self.isBearishHTF and self.marketOK):
                trailStop = self.cur_close + atr * SL_MULTIPLIER
                tp = self.cur_close - atr * TP_MULTIPLIER
                entryAtr = atr
                lot_size = self.calculate_lot_size(atr, SL_MULTIPLIER)
                
                self.active_order = MT5Order("SELL", self.cur_close, tp, trailStop, entryAtr,
                                             self.account_id, self.asset, self.auth_token, lot_size)
                result = self.active_order.place_order()
                
                if result['success']:
                    zone["mitigated"] = True
                    self.lastPositionWasShort = True
                    self.lastPositionWasLong = False
                    self.inPosition = True
                    self.daily_trades_count += 1
                    self.last_trade_date = str(datetime.now().date())
                    self.pending_signal = None
                    print(f"📉 SHORT position opened. Daily trades: {self.daily_trades_count}/{MAX_DAILY_TRADES}")
                break
    
    def run_bar_iterations(self):
        """Override to add stop flag check"""
        timeframe_sec = TIMEFRAME_SECONDS[self.timeframe]
        next_bar = datetime.now() + timedelta(seconds=timeframe_sec)
        
        while not self.stop_flag.is_set():
            try:
                self.fetch_new_data()
                self.calculate_indicators()
                self.add_fvg_zones()
                self.entry_logic()
                self.update_stops()
                self.save_data()
                print(f"\n⏰ New bar - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} close: {self.cur_close}")
                
                now = datetime.now()
                sleep_seconds = (next_bar - now).total_seconds()
                if sleep_seconds < 0:
                    sleep_seconds = 0
                
                # Sleep in small chunks to check stop flag
                while sleep_seconds > 0 and not self.stop_flag.is_set():
                    time.sleep(min(1, sleep_seconds))
                    sleep_seconds -= 1
                
                if not self.stop_flag.is_set():
                    next_bar += timedelta(seconds=timeframe_sec)
                
            except Exception as e:
                print(f"❌ Error in bar iteration: {e}")
                if not self.stop_flag.is_set():
                    time.sleep(60)
        
        print(f"🛑 Bar iterations stopped for {self.asset}")
    
    def update_price(self):
        """Override to get price from MT5 and check stop flag - checks on every tick (frequent polling)"""
        while not self.stop_flag.is_set():
            # Check stop flag frequently
            if self.stop_flag.is_set():
                break
            
            tick = mt5.symbol_info_tick(self.mt5_symbol)
            if tick is None:
                time.sleep(0.1)  # Small delay if tick unavailable
                continue
            
            # Use mid price (average of bid/ask)
            self.cur_price = (tick.bid + tick.ask) / 2
            self.cur_close = self.cur_price
            
            # Check exits on every tick (MT5 doesn't have OnTick callback in Python, so we poll frequently)
            if self.active_order is not None:
                closed = self.active_order.check_stops(self.cur_close)
                if closed:
                    self.active_order = None
                    self.inPosition = False
                    self.lastPositionWasLong = False
                    self.lastPositionWasShort = False
                    self.save_data()
            
            # Small sleep to avoid excessive CPU usage, but check frequently (every ~100ms)
            time.sleep(0.1)
        
        print(f"🛑 Price update stopped for {self.asset}")
    
    def stop(self):
        """Stop the strategy gracefully"""
        print(f"🛑 Stopping strategy for {self.asset}...")
        self.stop_flag.set()


def run_mt5_strat(strat: MT5Strategy):
    """Run MT5 strategy (no token needed)"""
    try:
        strat.init_rest()
        strat.run()
        
        # Wait for threads to finish (they will run until stop_flag is set)
        # The run() method starts threads but doesn't wait, so we wait here
        while not strat.stop_flag.is_set():
            time.sleep(1)
    except Exception as e:
        print(f"❌ Error in strategy: {e}")
        import traceback
        traceback.print_exc()
        strat.stop_flag.set()


if __name__ == "__main__":
    # Initialize MT5
    init_mt5()
    
    # Run strategies for each asset
    for asset_tuple in ASSETS:
        asset_id, timeframe, account_name = asset_tuple
        strategy = MT5Strategy(asset_tuple)
        run_mt5_strat(strategy)
    
    # Shutdown MT5 when done
    mt5.shutdown()

