#!/usr/bin/env python3
"""
MT5 Strategy Runner - Runs the MT5 strategy only on trading days (Monday-Friday)
and manages daily start/stop cycles (stops at 23:00, restarts at 0:00).
"""

import sys
import time
import signal
import threading
from datetime import datetime, timedelta
import MetaTrader5 as mt5
from mt5_strategy import init_mt5, MT5Strategy, run_mt5_strat, ASSETS

# Global flag for graceful shutdown
running = True
strategy_thread = None
strategy_instance = None


def is_trading_day():
    """Check if today is a trading day (Monday-Friday)"""
    today = datetime.now()
    # Monday = 0, Friday = 4
    return today.weekday() < 5


def wait_until_next_midnight():
    """Wait until 0:00 (midnight) of the next day"""
    now = datetime.now()
    # Calculate next midnight
    next_midnight = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    wait_seconds = (next_midnight - now).total_seconds()
    
    print(f"⏰ Waiting until {next_midnight.strftime('%Y-%m-%d %H:%M:%S')} ({wait_seconds/3600:.1f} hours)")
    return wait_seconds


def wait_until_23_00():
    """Wait until 23:00 today"""
    now = datetime.now()
    target_time = now.replace(hour=23, minute=0, second=0, microsecond=0)
    
    # If it's already past 23:00, wait until 23:00 tomorrow
    if now >= target_time:
        target_time += timedelta(days=1)
    
    wait_seconds = (target_time - now).total_seconds()
    return wait_seconds, target_time


def stop_strategy():
    """Stop the running strategy gracefully"""
    global running
    
    print("\n🛑 Stopping strategy...")
    running = False
    
    # Shutdown MT5 connection
    if mt5.initialized():
        mt5.shutdown()
        print("✅ MT5 connection closed")


def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    print("\n⚠️  Interrupt received. Shutting down...")
    stop_strategy()
    sys.exit(0)


def run_strategy_loop():
    """Main loop that runs strategy on trading days"""
    global running, strategy_thread, strategy_instance
    
    # Register signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    while running:
        # Check if it's a trading day
        if not is_trading_day():
            today = datetime.now()
            day_name = today.strftime('%A')
            print(f"📅 Today is {day_name} - Not a trading day. Waiting until Monday...")
            
            # Wait until next Monday
            days_until_monday = (7 - today.weekday()) % 7
            if days_until_monday == 0:
                days_until_monday = 7  # If it's already Monday, wait until next Monday
            
            next_monday = (today + timedelta(days=days_until_monday)).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            wait_seconds = (next_monday - today).total_seconds()
            
            print(f"⏰ Next trading day: {next_monday.strftime('%Y-%m-%d %H:%M:%S')} ({wait_seconds/3600:.1f} hours)")
            
            # Sleep in chunks to allow for interruption
            sleep_interval = 60  # Check every minute
            while wait_seconds > 0 and running:
                time.sleep(min(sleep_interval, wait_seconds))
                wait_seconds -= sleep_interval
            
            continue
        
        # It's a trading day - start the strategy
        today = datetime.now()
        print(f"\n{'='*60}")
        print(f"📈 Starting MT5 Strategy - {today.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")
        
        try:
            # Initialize MT5
            init_mt5()
            
            # Calculate how long until 23:00
            wait_seconds, stop_time = wait_until_23_00()
            print(f"⏰ Strategy will run until {stop_time.strftime('%H:%M:%S')} ({wait_seconds/3600:.1f} hours)")
            
            # Start strategy in a separate thread for each asset
            strategy_threads = []
            strategy_instances = []
            
            for asset_tuple in ASSETS:
                asset_id, timeframe, account_name = asset_tuple
                print(f"\n🚀 Starting strategy for {asset_id} ({timeframe})")
                
                strategy = MT5Strategy(asset_tuple)
                strategy_instances.append(strategy)
                
                # Run strategy in a thread
                thread = threading.Thread(
                    target=run_mt5_strat,
                    args=(strategy,),
                    daemon=False  # Not daemon so we can wait for them
                )
                thread.start()
                strategy_threads.append(thread)
                print(f"✅ Strategy thread started for {asset_id}")
            
            # Monitor time and stop at 23:00
            start_time = datetime.now()
            check_interval = 60  # Check every minute
            
            while running:
                now = datetime.now()
                
                # Check if we've reached 23:00
                if now.hour >= 23:
                    print(f"\n🕐 Reached 23:00. Stopping strategies...")
                    break
                
                # Check if it's still a trading day (in case we cross midnight)
                if not is_trading_day():
                    print(f"\n📅 Day changed to non-trading day. Stopping strategies...")
                    break
                
                # Sleep and check again
                time.sleep(check_interval)
            
            # Stop all strategies gracefully
            print("\n🛑 Stopping all strategies...")
            for strategy in strategy_instances:
                strategy.stop()
            
            # Wait for threads to finish (with timeout)
            for i, thread in enumerate(strategy_threads):
                if thread.is_alive():
                    print(f"⏳ Waiting for strategy {i+1} to stop...")
                    thread.join(timeout=10)
                    if thread.is_alive():
                        print(f"⚠️  Strategy {i+1} did not stop within timeout")
            
            # Shutdown MT5
            if mt5.initialized():
                mt5.shutdown()
                print("✅ MT5 connection closed")
            
            print("✅ All strategies stopped")
            
        except Exception as e:
            print(f"❌ Error running strategy: {e}")
            import traceback
            traceback.print_exc()
            
            # Cleanup on error
            if mt5.initialized():
                mt5.shutdown()
        
        # Wait until next midnight (0:00)
        if running:
            wait_seconds = wait_until_next_midnight()
            sleep_interval = 60  # Check every minute
            
            print(f"💤 Sleeping until next trading day...")
            while wait_seconds > 0 and running:
                time.sleep(min(sleep_interval, wait_seconds))
                wait_seconds -= sleep_interval
    
    print("\n👋 Runner stopped. Goodbye!")


if __name__ == "__main__":
    print("="*60)
    print("🤖 MT5 Strategy Runner")
    print("="*60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Trading days: Monday - Friday")
    print(f"Trading hours: 00:00 - 23:00")
    print("="*60)
    
    try:
        run_strategy_loop()
    except KeyboardInterrupt:
        print("\n⚠️  Keyboard interrupt received")
        stop_strategy()
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        stop_strategy()
        sys.exit(1)

