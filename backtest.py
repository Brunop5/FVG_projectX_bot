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
from strategyClass import Strategy, Order
from api_functions import fetch_data, load_data
import json


class BacktestOrder(Order):
    """Mock Order class for backtesting - tracks fills and P&L instead of placing real orders"""
    
    def __init__(self, side: str, entry_price: float, take_profit: float, 
                 trailing_stop_loss, entry_atr: float, account_id, asset_id, 
                 auth_token, lot_size=None):
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
    
    def place_order(self, fill_time=None, entry_bar=None):
        """Mock order placement - just marks as filled immediately at entry price"""
        self.filled = True
        self.fill_price = self.entry_price
        self.fill_time = fill_time or datetime.now()
        self.entry_bar = entry_bar
        return {'success': True, 'order_id': 'backtest_order', 'message': 'Order filled'}
    
    def close_order(self):
        """Mock order closing - calculates P&L"""
        if not self.filled or self.exit_price is not None:
            return
        
        # Calculate P&L based on side
        if self.side == "BUY":
            self.pnl = (self.exit_price - self.fill_price) * self.lot_size
        else:  # SELL
            self.pnl = (self.fill_price - self.exit_price) * self.lot_size
        
        self.pnl_pct = (self.pnl / (self.fill_price * self.lot_size)) * 100
        return {'success': True}


class BacktestStrategy(Strategy):
    """
    Backtesting wrapper for Strategy class.
    Overrides API-dependent methods to use historical data.
    """
    
    def __init__(self, asset_tuple, historical_data: pd.DataFrame, 
                 initial_balance: float = 10000.0, start_date=None, end_date=None):
        """
        Initialize backtest strategy.
        
        Args:
            asset_tuple: (asset, timeframe, account_name) - same as live strategy
            historical_data: DataFrame with OHLCV data (must have columns: timestamp, open, high, low, close, volume)
            initial_balance: Starting account balance for backtest
            start_date: Start date for backtest (if None, uses all data)
            end_date: End date for backtest (if None, uses all data)
        """
        super().__init__(asset_tuple)
        
        # Filter data by date range if provided
        if 'timestamp' in historical_data.columns:
            historical_data['timestamp'] = pd.to_datetime(historical_data['timestamp'])
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
        
        # Skip load_metadata for backtest
        self.calculate_indicators()
        self.fvg_zones: list[dict] = []
        self.add_fvg_zones()
    
    def gather_data(self):
        """Override to return initial data window"""
        return self.data
    
    def fetch_new_data(self):
        """Override to get next bar from historical data"""
        if self.current_bar_index >= len(self.historical_data):
            return None
        
        # Get next bar
        next_bar = self.historical_data.iloc[self.current_bar_index:self.current_bar_index+1]
        self.current_bar_index += 1
        
        # Update current data window (keep last 100 bars)
        self.data = pd.concat([self.data, next_bar], ignore_index=True).iloc[-100:]
        
        self.cur_close = self.data["close"].iloc[-1]
        self.cur_volume = self.data["volume"].iloc[-1]
        
        return next_bar
    
    def update_trend_indicators(self):
        """Override to use historical data for HTF"""
        # For HTF, we need to fetch from historical data
        # This is a simplified version - you may need to adjust based on your HTF logic
        htf_timeframe = f"{240}min"  # 4H
        htf_bars_needed = 101
        
        # Get HTF data from historical data
        # This is simplified - you'd need to resample or fetch HTF bars properly
        htf_data = self.historical_data.iloc[:self.current_bar_index].copy()
        if len(htf_data) < htf_bars_needed:
            # Not enough data yet
            self.isBullishHTF = None
            self.isBearishHTF = None
        else:
            # Resample to HTF (simplified - you may need proper resampling)
            from indicators import ema
            htf_close = htf_data['close'].iloc[-50:]  # Simplified
            htfEMA = ema(htf_close, 50)
            if htfEMA:
                self.isBullishHTF = self.cur_close > htfEMA
                self.isBearishHTF = self.cur_close < htfEMA
        
        # Rest of the logic same as live
        from indicators import sma, get_atr
        volOK = self.cur_volume > sma(self.data["volume"], 20) * 1.2
        atrVal = get_atr(self.data, 14)
        atrOK = atrVal.iloc[-1] > sma(atrVal, 20) if len(atrVal) > 0 else False
        self.marketOK = volOK and atrOK
        
        self.lastBullFvg = self.data["high"].iloc[-4] < self.data["low"].iloc[-2] and not self.lastBullFvg
        self.lastBearFvg = self.data["low"].iloc[-4] > self.data["high"].iloc[-2] and not self.lastBearFvg
    
    def entry_logic(self):
        """Override to use BacktestOrder instead of Order"""
        if len(self.fvg_zones) == 0 or self.inPosition:
            return
        
        # Check daily trade limit
        current_date = self.data['timestamp'].iloc[-1].date() if 'timestamp' in self.data.columns else datetime.now().date()
        if self.last_trade_date != str(current_date):
            self.daily_trades_count = 0
            self.last_trade_date = str(current_date)
        
        if self.daily_trades_count >= 5:  # MAX_DAILY_TRADES
            return
        
        current_high = self.data["high"].iloc[-1]
        current_low = self.data["low"].iloc[-1]
        
        from indicators import get_atr
        from strategyClass import SL_MULTIPLIER, TP_MULTIPLIER
        
        atr = get_atr(self.data, 14).iloc[-1]
        
        for zone in self.fvg_zones[-15:]:  # FVG_HISTORY_NBR
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
                                                  self.auth_token, lot_size)
                result = self.active_order.place_order(fill_time=current_time, entry_bar=self.current_bar_index)
                
                if result['success']:
                    zone["mitigated"] = True
                    self.lastPositionWasLong = True
                    self.lastPositionWasShort = False
                    self.inPosition = True
                    self.daily_trades_count += 1
                    self.last_trade_date = str(current_date)
                    print(f"📈 LONG entry at {self.cur_close:.5f} | Bar: {self.current_bar_index}")
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
                                                  self.auth_token, lot_size)
                result = self.active_order.place_order(fill_time=current_time, entry_bar=self.current_bar_index)
                
                if result['success']:
                    zone["mitigated"] = True
                    self.lastPositionWasShort = True
                    self.lastPositionWasLong = False
                    self.inPosition = True
                    self.daily_trades_count += 1
                    self.last_trade_date = str(current_date)
                    print(f"📉 SHORT entry at {self.cur_close:.5f} | Bar: {self.current_bar_index}")
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
            'exit_reason': order.exit_reason,
            'entry_bar': order.entry_bar,
            'exit_bar': self.current_bar_index,
            'bars_held': self.current_bar_index - order.entry_bar if order.entry_bar else 0
        }
        self.trades.append(trade)
        self.current_balance += order.pnl
        
        print(f"{'✅' if order.pnl > 0 else '❌'} Trade closed: {order.side} | "
              f"Entry: {order.fill_price:.5f} | Exit: {order.exit_price:.5f} | "
              f"P&L: ${order.pnl:.2f} ({order.pnl_pct:.2f}%) | Reason: {order.exit_reason}")
    
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
        
        from strategyClass import USE_TRAILING, TRAIL_OFFSET_MULT
        
        if self.inPosition and self.lastPositionWasLong:
            if USE_TRAILING and pos.entry_atr is not None:
                potentialStop = current_high - pos.entry_atr * TRAIL_OFFSET_MULT
                if pos.trailing_stop_loss is not None:
                    pos.trailing_stop_loss = max(pos.trailing_stop_loss, potentialStop)
                else:
                    pos.trailing_stop_loss = potentialStop
        
        if self.inPosition and self.lastPositionWasShort:
            if USE_TRAILING and pos.entry_atr is not None:
                potentialStop = current_low + pos.entry_atr * TRAIL_OFFSET_MULT
                if pos.trailing_stop_loss is not None:
                    pos.trailing_stop_loss = min(pos.trailing_stop_loss, potentialStop)
                else:
                    pos.trailing_stop_loss = potentialStop
        
        # Check BOS/CHoCH exits
        from strategyClass import HOLD_UNTIL_OPPOSITE
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
        
        # Record equity curve
        self.equity_curve.append({
            'bar': self.current_bar_index,
            'timestamp': self.data.iloc[-1].get('timestamp', datetime.now()),
            'balance': self.current_balance,
            'equity': self.current_balance + (pos.pnl if pos and pos.filled else 0)
        })
    
    def run_backtest(self):
        """Run the backtest on historical data"""
        print(f"\n{'='*60}")
        print(f"🧪 Starting Backtest for {self.asset}")
        print(f"{'='*60}")
        print(f"Initial Balance: ${self.initial_balance:,.2f}")
        print(f"Date Range: {self.historical_data['timestamp'].iloc[0]} to {self.historical_data['timestamp'].iloc[-1]}")
        print(f"Total Bars: {len(self.historical_data)}\n")
        
        # Initialize
        self.init_rest()
        
        # Process each bar
        while self.current_bar_index < len(self.historical_data):
            self.fetch_new_data()
            self.calculate_indicators()
            self.add_fvg_zones()
            self.entry_logic()
            self.update_stops()
        
        # Close any open position at end
        if self.active_order and self.active_order.filled:
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
            return
        
        trades_df = pd.DataFrame(self.trades)
        
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        losing_trades = len(trades_df[trades_df['pnl'] < 0])
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        total_pnl = trades_df['pnl'].sum()
        avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0
        profit_factor = abs(avg_win * winning_trades / (avg_loss * losing_trades)) if losing_trades > 0 and avg_loss != 0 else float('inf')
        
        final_balance = self.current_balance
        total_return = ((final_balance - self.initial_balance) / self.initial_balance) * 100
        
        print(f"\n{'='*60}")
        print(f"📊 BACKTEST RESULTS")
        print(f"{'='*60}")
        print(f"Initial Balance:     ${self.initial_balance:,.2f}")
        print(f"Final Balance:       ${final_balance:,.2f}")
        print(f"Total Return:        {total_return:.2f}%")
        print(f"Total P&L:           ${total_pnl:,.2f}")
        print(f"\nTotal Trades:        {total_trades}")
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


# ==================== USAGE EXAMPLE ====================

def run_backtest_example():
    """
    Example of how to run a backtest.
    
    You need historical data in CSV format with columns:
    - timestamp (or date)
    - open
    - high
    - low
    - close
    - volume
    """
    
    # Load historical data
    # Option 1: From CSV file
    historical_data = pd.read_csv("your_historical_data.csv")
    
    # Option 2: From API (if you have historical data endpoint)
    # historical_data = fetch_data("CON.F.US.GCE.G26", "1min", 10000, token, False)
    
    # Ensure timestamp column exists
    if 'timestamp' not in historical_data.columns and 'date' in historical_data.columns:
        historical_data['timestamp'] = pd.to_datetime(historical_data['date'])
    
    # Create backtest strategy
    asset_tuple = ("CON.F.US.GCE.G26", "1min", "backtest_account")
    backtest = BacktestStrategy(
        asset_tuple=asset_tuple,
        historical_data=historical_data,
        initial_balance=10000.0,
        start_date="2024-01-01",  # Optional
        end_date="2024-12-31"     # Optional
    )
    
    # Run backtest
    backtest.run_backtest()


if __name__ == "__main__":
    run_backtest_example()

