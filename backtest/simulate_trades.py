#!/usr/bin/env python3
"""
Trade simulator that processes all backtest trades from gold_results/{id}/ folders.
Reads parameters from final_result.csv and runs simulation for each trades CSV.
Saves all results to a single simulation_summary.csv.
"""

import csv
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime

# ===== CONFIGURATION =====
GOLD_RESULTS_DIR = "gold_results"
FINAL_RESULT_CSV = os.path.join(GOLD_RESULTS_DIR, "final_result.csv")
OUTPUT_DIR = "simulation_results"  # Directory to save output CSVs

# Simulation parameters (can be overridden per backtest if needed)
INITIAL_CAPITAL = 10000  # Starting capital in dollars
TRADE_SIZE = 2000  # Dollar value of each trade order (used if USE_CUMULATIVE is False)
USE_CUMULATIVE = False  # If True: trade size is percentage of current equity. If False: fixed dollar amount
TRADE_SIZE_PCT = 100.0  # Percentage of current equity to use for each trade (only if USE_CUMULATIVE is True)
LEVERAGE = 20  # Leverage multiplier (1 = no leverage)
SPREAD = 0.2 # Spread in price units (always exact)
FEE_PERCENT = 0.02  # Fee percentage (0.02% = 0.0002 as decimal)

# Single file test mode
SINGLE_FILE_MODE = True  # If True: test only one file. If False: process all backtests
SINGLE_FILE_PATH = "gold_results/27/backtest_trades_CON.F.US.MGC.G26_20260119.csv"  # Path to trades CSV file for single file mode
# =========================

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

def simulate_trades_from_csv(trades_csv_file, initial_capital=INITIAL_CAPITAL, trade_size=TRADE_SIZE, 
                              leverage=LEVERAGE, spread=SPREAD, fee_percent=FEE_PERCENT,
                              use_cumulative=USE_CUMULATIVE, trade_size_pct=TRADE_SIZE_PCT):
    """
    Simulate trades from a single trades CSV file.
    
    Args:
        trades_csv_file: Path to trades CSV file
        initial_capital: Starting capital
        trade_size: Fixed dollar value per trade (used if use_cumulative is False)
        leverage: Leverage multiplier
        spread: Spread in price units
        fee_percent: Fee percentage
        use_cumulative: If True, trade size is percentage of current equity
        trade_size_pct: Percentage of current equity to use (only if use_cumulative is True)
    
    Returns:
        Dictionary with simulation results and metrics
    """
    capital = initial_capital
    trades_executed = 0
    winning_trades = 0
    losing_trades = 0
    total_pnl = 0.0
    total_fees_paid = 0.0
    equity_curve = []  # List to store equity curve data
    simulated_trades = []  # List to store all trade details
    winning_pnls = []
    losing_pnls = []
    
    # Read and process trades
    try:
        with open(trades_csv_file, 'r') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                # Parse trade data
                side = row['side'].strip()
                entry_price = float(row['entry_price'])
                exit_price = float(row['exit_price'])
                entry_time = row.get('entry_time', '')
                exit_time = row.get('exit_time', '')
                exit_reason = row.get('exit_reason', '')
                original_pnl = float(row.get('pnl', 0))
                original_fees = float(row.get('fees', 0))
                
                # Apply spread: when buying, pay ask price (entry + spread/2), when selling, get bid price (exit - spread/2)
                if side == 'BUY':  # Long position
                    actual_entry_price = entry_price + (spread / 2)  # Pay ask price
                    actual_exit_price = exit_price - (spread / 2)  # Get bid price
                else:  # SELL = Short position
                    actual_entry_price = entry_price - (spread / 2)  # Get bid price (short entry)
                    actual_exit_price = exit_price + (spread / 2)  # Pay ask price (short exit)
                
                # Calculate position size based on actual entry price
                # If cumulative mode: use percentage of current equity. Otherwise: use fixed dollar amount
                if use_cumulative:
                    # Trade size is percentage of current equity
                    position_value = (capital * trade_size_pct / 100) * leverage
                else:
                    # Fixed dollar amount
                    position_value = trade_size * leverage
                
                num_units = position_value / actual_entry_price
                
                # Calculate fees: fee_percent% of the position value (both entry and exit)
                entry_fee = position_value * (fee_percent / 100)
                exit_fee = (num_units * actual_exit_price) * (fee_percent / 100)
                total_fees = entry_fee + exit_fee
                total_fees_paid += total_fees
                
                # Calculate P&L based on side
                if side == 'BUY':  # Long position
                    price_change = actual_exit_price - actual_entry_price
                else:  # SELL = Short position
                    price_change = actual_entry_price - actual_exit_price
                
                # Calculate trade P&L (including spread and fees)
                trade_pnl = price_change * num_units - total_fees
                
                # Update capital
                capital += trade_pnl
                total_pnl += trade_pnl
                trades_executed += 1
                
                if trade_pnl > 0:
                    winning_trades += 1
                    winning_pnls.append(trade_pnl)
                elif trade_pnl < 0:
                    losing_trades += 1
                    losing_pnls.append(trade_pnl)
                
                # Store trade details
                simulated_trades.append({
                    'entry_time': entry_time,
                    'exit_time': exit_time,
                    'side': side,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'actual_entry_price': actual_entry_price,
                    'actual_exit_price': actual_exit_price,
                    'size': num_units,
                    'pnl': trade_pnl,
                    'pnl_pct': (trade_pnl / initial_capital * 100) if initial_capital > 0 else 0,
                    'fees': total_fees,
                    'exit_reason': exit_reason,
                    'balance_after': capital,
                    'cumulative_pnl': total_pnl
                })
                
                # Record equity curve point
                equity_curve.append({
                    'trade_number': trades_executed,
                    'balance': capital,
                    'cumulative_pnl': total_pnl
                })
    except Exception as e:
        print(f"❌ Error reading {trades_csv_file}: {e}")
        return None

    # Calculate additional metrics
    win_rate = (winning_trades / trades_executed * 100) if trades_executed > 0 else 0
    avg_win = np.mean(winning_pnls) if winning_pnls else 0
    avg_loss = np.mean(losing_pnls) if losing_pnls else 0
    largest_win = max(winning_pnls) if winning_pnls else 0
    largest_loss = min(losing_pnls) if losing_pnls else 0
    profit_factor = abs(sum(winning_pnls) / sum(losing_pnls)) if losing_pnls and sum(losing_pnls) != 0 else 0
    total_return = (capital / initial_capital - 1) * 100 if initial_capital > 0 else 0
    
    # Calculate max drawdown from equity curve
    if equity_curve:
        equity_values = np.array([point['balance'] for point in equity_curve])
        running_max = np.maximum.accumulate(equity_values)
        drawdowns = running_max - equity_values
        max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0.0
        peak_equity = np.max(equity_values) if len(equity_values) > 0 else initial_capital
        max_drawdown_pct = (max_drawdown / peak_equity * 100) if peak_equity > 0 else 0.0
    else:
        max_drawdown = 0.0
        max_drawdown_pct = 0.0

    # Create and display equity curve graph (don't save)
    if equity_curve:
        trade_numbers = [point['trade_number'] for point in equity_curve]
        balances = [point['balance'] for point in equity_curve]
        
        plt.figure(figsize=(12, 6))
        plt.plot(trade_numbers, balances, linewidth=2, color='#2E86AB')
        plt.axhline(y=initial_capital, color='gray', linestyle='--', linewidth=1, label='Initial Capital')
        plt.xlabel('Trade Number', fontsize=12)
        plt.ylabel('Account Balance ($)', fontsize=12)
        plt.title('Equity Curve', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

    return {
        'trades_executed': trades_executed,
        'winning_trades': winning_trades,
        'losing_trades': losing_trades,
        'total_pnl': total_pnl,
        'total_fees_paid': total_fees_paid,
        'capital': capital,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'largest_win': largest_win,
        'largest_loss': largest_loss,
        'profit_factor': profit_factor,
        'total_return': total_return,
        'max_drawdown': max_drawdown,
        'max_drawdown_pct': max_drawdown_pct,
        'equity_curve': equity_curve,
        'simulated_trades': simulated_trades,
        'winning_pnls': winning_pnls,
        'losing_pnls': losing_pnls
    }


def calculate_losing_streaks(trades_list):
    """Calculate losing streak metrics"""
    if not trades_list:
        return 0, 0.0, 0.0, 0.0
    
    most_losing_trades_in_row = 0
    current_streak = 0
    longest_streak_start = None
    longest_streak_end = None
    
    for idx, trade in enumerate(trades_list):
        if trade['pnl'] < 0:
            current_streak += 1
            if current_streak > most_losing_trades_in_row:
                most_losing_trades_in_row = current_streak
                longest_streak_end = idx
                longest_streak_start = idx - current_streak + 1
        else:
            current_streak = 0
    
    # Calculate dollar value of most losses in a row
    dollar_value_most_losses_in_row = 0.0
    if longest_streak_start is not None and longest_streak_end is not None:
        longest_streak_trades = trades_list[longest_streak_start:longest_streak_end + 1]
        dollar_value_most_losses_in_row = sum(t['pnl'] for t in longest_streak_trades)
    
    # Calculate monthly metrics (simplified - group by approximate months)
    # For simplicity, we'll calculate average max losing streak per 30 trades
    monthly_max_streaks = []
    monthly_max_streak_values = []
    trades_per_month = 30  # Approximate
    
    for i in range(0, len(trades_list), trades_per_month):
        month_trades = trades_list[i:i + trades_per_month]
        if not month_trades:
            continue
        
        max_streak = 0
        max_streak_value = 0.0
        current_streak = 0
        current_streak_start = None
        
        for idx, trade in enumerate(month_trades):
            if trade['pnl'] < 0:
                if current_streak == 0:
                    current_streak_start = idx
                current_streak += 1
            else:
                if current_streak > 0:
                    if current_streak > max_streak:
                        max_streak = current_streak
                        streak_trades = month_trades[current_streak_start:idx]
                        max_streak_value = sum(t['pnl'] for t in streak_trades)
                current_streak = 0
                current_streak_start = None
        
        # Handle streak at end of month
        if current_streak > 0:
            if current_streak > max_streak:
                max_streak = current_streak
                streak_trades = month_trades[current_streak_start:]
                max_streak_value = sum(t['pnl'] for t in streak_trades)
        
        monthly_max_streaks.append(max_streak)
        monthly_max_streak_values.append(max_streak_value)
    
    avg_most_losing_trades_per_month = np.mean(monthly_max_streaks) if monthly_max_streaks else 0.0
    dollar_value_avg_most_losses_per_month = np.mean(monthly_max_streak_values) if monthly_max_streak_values else 0.0
    
    return most_losing_trades_in_row, avg_most_losing_trades_per_month, dollar_value_most_losses_in_row, dollar_value_avg_most_losses_per_month

# Calculate average max monthly drawdown
def calculate_avg_max_monthly_drawdown(trades_list, initial_balance):
    """Calculate average of maximum monthly drawdowns"""
    if not trades_list:
        return 0.0
    
    # Reconstruct equity curve
    equity_series = [initial_balance]
    for trade in trades_list:
        equity_series.append(equity_series[-1] + trade['pnl'])
    
    # Group by approximate months (30 trades per month)
    trades_per_month = 30
    monthly_drawdowns = []
    
    for i in range(0, len(equity_series) - 1, trades_per_month):
        month_equity = equity_series[i:min(i + trades_per_month + 1, len(equity_series))]
        if len(month_equity) < 2:
            continue
        
        # Calculate max drawdown for this month
        month_equity_array = np.array(month_equity)
        running_max = np.maximum.accumulate(month_equity_array)
        drawdowns = running_max - month_equity_array
        max_monthly_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0.0
        monthly_drawdowns.append(max_monthly_drawdown)
    
    return np.mean(monthly_drawdowns) if monthly_drawdowns else 0.0

def process_all_backtests():
    """Process all backtest trades from gold_results/{id}/ folders"""
    
    # Read final_result.csv to get parameters for each ID
    if not os.path.exists(FINAL_RESULT_CSV):
        print(f"❌ Error: {FINAL_RESULT_CSV} not found")
        return
    
    final_result_df = pd.read_csv(FINAL_RESULT_CSV)
    print(f"📊 Loaded {len(final_result_df)} backtest results from {FINAL_RESULT_CSV}")
    
    # Find all ID folders in gold_results
    id_folders = []
    if os.path.exists(GOLD_RESULTS_DIR):
        for item in os.listdir(GOLD_RESULTS_DIR):
            item_path = os.path.join(GOLD_RESULTS_DIR, item)
            if os.path.isdir(item_path) and item.isdigit():
                id_folders.append((int(item), item_path))
    
    id_folders.sort(key=lambda x: x[0])  # Sort by ID
    print(f"📁 Found {len(id_folders)} ID folders")
    
    all_summary_results = []
    
    # Process each ID folder
    for result_id, folder_path in id_folders:
        print(f"\n{'='*60}")
        print(f"Processing ID {result_id}")
        print(f"{'='*60}")
        
        # Find trades CSV file in this folder
        trades_pattern = os.path.join(folder_path, "backtest_trades*.csv")
        trades_files = glob.glob(trades_pattern)
        
        if not trades_files:
            print(f"⚠️  No trades CSV found in {folder_path}")
            continue
        
        trades_csv_file = trades_files[0]  # Use first match
        print(f"📄 Trades file: {trades_csv_file}")
        
        # Get parameters from final_result.csv for this ID
        result_row = final_result_df[final_result_df['id'] == result_id]
        if result_row.empty:
            print(f"⚠️  No parameters found in final_result.csv for ID {result_id}")
            continue
        
        result_row = result_row.iloc[0]
        
        # Run simulation
        print(f"🔄 Running simulation...")
        sim_results = simulate_trades_from_csv(
            trades_csv_file,
            initial_capital=INITIAL_CAPITAL,
            trade_size=TRADE_SIZE,
            leverage=LEVERAGE,
            spread=SPREAD,
            fee_percent=FEE_PERCENT,
            use_cumulative=USE_CUMULATIVE,
            trade_size_pct=TRADE_SIZE_PCT
        )
        
        if sim_results is None:
            print(f"❌ Simulation failed for ID {result_id}")
            continue
        
        # Calculate additional metrics
        most_losing_trades_in_row, avg_most_losing_trades_per_month, dollar_value_most_losses_in_row, dollar_value_avg_most_losses_per_month = calculate_losing_streaks(sim_results['simulated_trades'])
        avg_max_monthly_drawdown = calculate_avg_max_monthly_drawdown(sim_results['simulated_trades'], INITIAL_CAPITAL)
        
        # Calculate period days
        if sim_results['simulated_trades'] and sim_results['simulated_trades'][0].get('entry_time'):
            try:
                first_date = pd.to_datetime(sim_results['simulated_trades'][0]['entry_time'])
                last_date = pd.to_datetime(sim_results['simulated_trades'][-1]['exit_time'])
                backtest_period_days = (last_date - first_date).days + 1
                trades_per_day = sim_results['trades_executed'] / backtest_period_days if backtest_period_days > 0 else 0
            except:
                backtest_period_days = 0
                trades_per_day = 0
        else:
            backtest_period_days = 0
            trades_per_day = 0
        
        # Calculate yearly return
        yearly_return = (sim_results['total_return'] / backtest_period_days * 365) if backtest_period_days > 0 else 0
        
        # Print results for this backtest
        print(f"✅ Simulation complete:")
        print(f"   Final Capital: ${sim_results['capital']:,.2f}")
        print(f"   Total P&L: ${sim_results['total_pnl']:,.2f}")
        print(f"   Return: {sim_results['total_return']:.2f}%")
        print(f"   Trades: {sim_results['trades_executed']}")
        print(f"   Win Rate: {sim_results['win_rate']:.2f}%")
        print(f"   Max Drawdown: ${sim_results['max_drawdown']:,.2f} ({sim_results['max_drawdown_pct']:.2f}%)")
        
        # Save individual CSV files for this simulation
        try:
            # Create ID-specific directory
            sim_id_dir = os.path.join(OUTPUT_DIR, str(result_id))
            os.makedirs(sim_id_dir, exist_ok=True)
            
            # Get asset name from trades file path or use default
            asset_name = os.path.basename(trades_csv_file).split('_')[2] if '_' in os.path.basename(trades_csv_file) else 'UNKNOWN'
            date_str = datetime.now().strftime('%Y%m%d')
            
            # Save trades CSV
            if sim_results['simulated_trades']:
                trades_df = pd.DataFrame(sim_results['simulated_trades'])
                trades_file = os.path.join(sim_id_dir, f"simulation_trades_{asset_name}_{date_str}.csv")
                trades_df.to_csv(trades_file, index=False)
                print(f"💾 Saved trades CSV: {trades_file}")
            
            # Save equity curve CSV
            if sim_results['equity_curve']:
                equity_df = pd.DataFrame(sim_results['equity_curve'])
                equity_file = os.path.join(sim_id_dir, f"simulation_equity_{asset_name}_{date_str}.csv")
                equity_df.to_csv(equity_file, index=False)
                print(f"💾 Saved equity curve CSV: {equity_file}")
            
            # Save individual summary CSV
            summary_file = os.path.join(sim_id_dir, f"simulation_summary_{asset_name}_{date_str}.csv")
            summary_df_single = pd.DataFrame([summary_row])
            summary_df_single.to_csv(summary_file, index=False)
            print(f"💾 Saved summary CSV: {summary_file}")
            
        except Exception as e:
            print(f"⚠️  Warning: Failed to save individual CSV files for ID {result_id}: {e}")
        
        # Create summary row with parameters from final_result.csv and simulation results
        summary_row = {
            # ID and parameters from final_result.csv
            'id': result_id,
            'timeframe': result_row.get('timeframe', ''),
            'FVG_HISTORY_NBR': result_row.get('FVG_HISTORY_NBR', None),
            'MIN_FVG_POWER_PCT': result_row.get('MIN_FVG_POWER_PCT', None),
            'HTF_TF': result_row.get('HTF_TF', None),
            'EMA_PERIOD': result_row.get('EMA_PERIOD', None),
            'VOLUME_MULTIPLIER': result_row.get('VOLUME_MULTIPLIER', None),
            'ATR_PERIOD': result_row.get('ATR_PERIOD', None),
            'SL_MULTIPLIER': result_row.get('SL_MULTIPLIER', None),
            'TP_MULTIPLIER': result_row.get('TP_MULTIPLIER', None),
            'USE_TRAILING': result_row.get('USE_TRAILING', None),
            'TRAIL_OFFSET_MULT': result_row.get('TRAIL_OFFSET_MULT', None),
            'HOLD_UNTIL_OPPOSITE': result_row.get('HOLD_UNTIL_OPPOSITE', None),
            'USE_VOLUME_CHECK': result_row.get('USE_VOLUME_CHECK', None),
            'VOLUME_DATA_START_TIMESTAMP': result_row.get('VOLUME_DATA_START_TIMESTAMP', None),
            'START_FROM_VOLUME_TIMESTAMP': result_row.get('START_FROM_VOLUME_TIMESTAMP', None),
            # Simulation parameters
            'sim_initial_capital': INITIAL_CAPITAL,
            'sim_trade_size': TRADE_SIZE,
            'sim_use_cumulative': USE_CUMULATIVE,
            'sim_trade_size_pct': TRADE_SIZE_PCT if USE_CUMULATIVE else None,
            'sim_leverage': LEVERAGE,
            'sim_spread': SPREAD,
            'sim_fee_percent': FEE_PERCENT,
            # Simulation results
            'total_pnl': sim_results['total_pnl'],
            'total_trades': sim_results['trades_executed'],
            'win_rate': sim_results['win_rate'],
            'final_balance': sim_results['capital'],
            'max_drawdown': sim_results['max_drawdown'],
            'max_drawdown_pct': sim_results['max_drawdown_pct'],
            'total_fees': sim_results['total_fees_paid'],
            'winning_trades': sim_results['winning_trades'],
            'losing_trades': sim_results['losing_trades'],
            'avg_win': sim_results['avg_win'],
            'avg_loss': sim_results['avg_loss'],
            'profit_factor': sim_results['profit_factor'],
            'total_return': sim_results['total_return'],
            'net_profit': sim_results['total_pnl'],
            'largest_win': sim_results['largest_win'],
            'largest_loss': sim_results['largest_loss'],
            'trades_per_day': trades_per_day,
            'backtest_period_days': backtest_period_days,
            'strategy_failed': False,
            'failed_reason': None,
            'backtest_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'yearly_return': yearly_return,
            'most_losing_trades_in_row': most_losing_trades_in_row,
            'avg_most_losing_trades_per_month': round(avg_most_losing_trades_per_month, 2),
            'dollar_value_most_losses_in_row': round(dollar_value_most_losses_in_row, 2),
            'dollar_value_avg_most_losses_per_month': round(dollar_value_avg_most_losses_per_month, 2),
            'avg_max_monthly_drawdown': round(avg_max_monthly_drawdown, 2)
        }
        
        all_summary_results.append(summary_row)
    
    # Save all results to simulation_summary.csv
    if all_summary_results:
        summary_df = pd.DataFrame(all_summary_results)
        summary_output_file = os.path.join(OUTPUT_DIR, "simulation_summary.csv")
        summary_df.to_csv(summary_output_file, index=False)
        print(f"\n{'='*60}")
        print(f"✅ Saved {len(all_summary_results)} simulation results to {summary_output_file}")
        print(f"{'='*60}")
    else:
        print("\n⚠️  No simulation results to save")


def process_single_file(trades_csv_file):
    """Process a single trades CSV file"""
    
    if not os.path.exists(trades_csv_file):
        print(f"❌ Error: Trades CSV file not found: {trades_csv_file}")
        return
    
    print(f"\n{'='*60}")
    print(f"TRADE SIMULATION - Single File Mode")
    print(f"{'='*60}")
    print(f"Trades File: {trades_csv_file}")
    print(f"\nSimulation Parameters:")
    print(f"  Initial Capital: ${INITIAL_CAPITAL:,.2f}")
    if USE_CUMULATIVE:
        print(f"  Trade Size: {TRADE_SIZE_PCT}% of current equity (CUMULATIVE)")
    else:
        print(f"  Trade Size: ${TRADE_SIZE:,.2f} (FIXED)")
    print(f"  Leverage: {LEVERAGE}x")
    print(f"  Spread: {SPREAD}")
    print(f"  Fee Rate: {FEE_PERCENT}%")
    print(f"{'='*60}\n")
    
    # Run simulation
    print(f"🔄 Running simulation...")
    sim_results = simulate_trades_from_csv(
        trades_csv_file,
        initial_capital=INITIAL_CAPITAL,
        trade_size=TRADE_SIZE,
        leverage=LEVERAGE,
        spread=SPREAD,
        fee_percent=FEE_PERCENT,
        use_cumulative=USE_CUMULATIVE,
        trade_size_pct=TRADE_SIZE_PCT
    )
    
    if sim_results is None:
        print(f"❌ Simulation failed")
        return
    
    # Calculate additional metrics
    most_losing_trades_in_row, avg_most_losing_trades_per_month, dollar_value_most_losses_in_row, dollar_value_avg_most_losses_per_month = calculate_losing_streaks(sim_results['simulated_trades'])
    avg_max_monthly_drawdown = calculate_avg_max_monthly_drawdown(sim_results['simulated_trades'], INITIAL_CAPITAL)
    
    # Calculate period days
    if sim_results['simulated_trades'] and sim_results['simulated_trades'][0].get('entry_time'):
        try:
            first_date = pd.to_datetime(sim_results['simulated_trades'][0]['entry_time'])
            last_date = pd.to_datetime(sim_results['simulated_trades'][-1]['exit_time'])
            backtest_period_days = (last_date - first_date).days + 1
            trades_per_day = sim_results['trades_executed'] / backtest_period_days if backtest_period_days > 0 else 0
        except:
            backtest_period_days = 0
            trades_per_day = 0
    else:
        backtest_period_days = 0
        trades_per_day = 0
    
    # Calculate yearly return
    yearly_return = (sim_results['total_return'] / backtest_period_days * 365) if backtest_period_days > 0 else 0
    
    # Save CSV files for single file mode
    try:
        # Extract ID from path if possible, otherwise use timestamp
        try:
            result_id = int(os.path.basename(os.path.dirname(trades_csv_file)))
        except (ValueError, AttributeError):
            result_id = int(datetime.now().timestamp())
        
        # Create ID-specific directory
        sim_id_dir = os.path.join(OUTPUT_DIR, str(result_id))
        os.makedirs(sim_id_dir, exist_ok=True)
        
        # Get asset name from trades file path or use default
        asset_name = os.path.basename(trades_csv_file).split('_')[2] if '_' in os.path.basename(trades_csv_file) else 'UNKNOWN'
        date_str = datetime.now().strftime('%Y%m%d')
        
        # Save trades CSV
        if sim_results['simulated_trades']:
            trades_df = pd.DataFrame(sim_results['simulated_trades'])
            trades_file = os.path.join(sim_id_dir, f"simulation_trades_{asset_name}_{date_str}.csv")
            trades_df.to_csv(trades_file, index=False)
            print(f"💾 Saved trades CSV: {trades_file}")
        
        # Save equity curve CSV
        if sim_results['equity_curve']:
            equity_df = pd.DataFrame(sim_results['equity_curve'])
            equity_file = os.path.join(sim_id_dir, f"simulation_equity_{asset_name}_{date_str}.csv")
            equity_df.to_csv(equity_file, index=False)
            print(f"💾 Saved equity curve CSV: {equity_file}")
        
        # Create and save summary row
        summary_row = {
            'id': result_id,
            'timeframe': '',
            'sim_initial_capital': INITIAL_CAPITAL,
            'sim_trade_size': TRADE_SIZE,
            'sim_use_cumulative': USE_CUMULATIVE,
            'sim_trade_size_pct': TRADE_SIZE_PCT if USE_CUMULATIVE else None,
            'sim_leverage': LEVERAGE,
            'sim_spread': SPREAD,
            'sim_fee_percent': FEE_PERCENT,
            'total_pnl': sim_results['total_pnl'],
            'total_trades': sim_results['trades_executed'],
            'win_rate': sim_results['win_rate'],
            'final_balance': sim_results['capital'],
            'max_drawdown': sim_results['max_drawdown'],
            'max_drawdown_pct': sim_results['max_drawdown_pct'],
            'total_fees': sim_results['total_fees_paid'],
            'winning_trades': sim_results['winning_trades'],
            'losing_trades': sim_results['losing_trades'],
            'avg_win': sim_results['avg_win'],
            'avg_loss': sim_results['avg_loss'],
            'profit_factor': sim_results['profit_factor'],
            'total_return': sim_results['total_return'],
            'net_profit': sim_results['total_pnl'],
            'largest_win': sim_results['largest_win'],
            'largest_loss': sim_results['largest_loss'],
            'trades_per_day': trades_per_day,
            'backtest_period_days': backtest_period_days,
            'yearly_return': yearly_return,
            'most_losing_trades_in_row': most_losing_trades_in_row,
            'avg_most_losing_trades_per_month': round(avg_most_losing_trades_per_month, 2),
            'dollar_value_most_losses_in_row': round(dollar_value_most_losses_in_row, 2),
            'dollar_value_avg_most_losses_per_month': round(dollar_value_avg_most_losses_per_month, 2),
            'avg_max_monthly_drawdown': round(avg_max_monthly_drawdown, 2),
            'simulation_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        summary_file = os.path.join(sim_id_dir, f"simulation_summary_{asset_name}_{date_str}.csv")
        summary_df_single = pd.DataFrame([summary_row])
        summary_df_single.to_csv(summary_file, index=False)
        print(f"💾 Saved summary CSV: {summary_file}")
        
    except Exception as e:
        print(f"⚠️  Warning: Failed to save CSV files: {e}")
    
    # Print detailed results
    print(f"\n{'='*60}")
    print(f"📊 SIMULATION RESULTS")
    print(f"{'='*60}")
    print(f"Initial Capital:     ${INITIAL_CAPITAL:,.2f}")
    print(f"Final Capital:       ${sim_results['capital']:,.2f}")
    print(f"Total P&L:           ${sim_results['total_pnl']:,.2f}")
    print(f"Total Return:        {sim_results['total_return']:.2f}%")
    print(f"Yearly Return:       {yearly_return:.2f}%")
    print(f"Total Fees Paid:     ${sim_results['total_fees_paid']:,.2f}")
    print(f"Max Drawdown:        ${sim_results['max_drawdown']:,.2f} ({sim_results['max_drawdown_pct']:.2f}%)")
    print(f"\nTotal Trades:        {sim_results['trades_executed']}")
    print(f"Winning Trades:      {sim_results['winning_trades']}")
    print(f"Losing Trades:       {sim_results['losing_trades']}")
    print(f"Win Rate:            {sim_results['win_rate']:.2f}%")
    print(f"\nAverage Win:         ${sim_results['avg_win']:.2f}")
    print(f"Average Loss:         ${sim_results['avg_loss']:.2f}")
    print(f"Profit Factor:       {sim_results['profit_factor']:.2f}")
    print(f"\nLargest Win:         ${sim_results['largest_win']:.2f}")
    print(f"Largest Loss:        ${sim_results['largest_loss']:.2f}")
    print(f"\nTrades per Day:      {trades_per_day:.2f}")
    print(f"Backtest Period:     {backtest_period_days} days")
    print(f"\nMost Losing Trades in Row: {most_losing_trades_in_row}")
    print(f"Avg Most Losing Trades/Month: {avg_most_losing_trades_per_month:.2f}")
    print(f"Dollar Value Most Losses in Row: ${dollar_value_most_losses_in_row:,.2f}")
    print(f"Dollar Value Avg Most Losses/Month: ${dollar_value_avg_most_losses_per_month:,.2f}")
    print(f"Avg Max Monthly Drawdown: ${avg_max_monthly_drawdown:,.2f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    if SINGLE_FILE_MODE:
        # Single file test mode
        process_single_file(SINGLE_FILE_PATH)
    else:
        # Process all backtests mode
        print("=" * 60)
        print("TRADE SIMULATION - Processing All Backtests")
        print("=" * 60)
        print(f"Gold Results Directory: {GOLD_RESULTS_DIR}")
        print(f"Final Result CSV: {FINAL_RESULT_CSV}")
        print(f"Output Directory: {OUTPUT_DIR}")
        print(f"\nSimulation Parameters:")
        print(f"  Initial Capital: ${INITIAL_CAPITAL:,.2f}")
        if USE_CUMULATIVE:
            print(f"  Trade Size: {TRADE_SIZE_PCT}% of current equity (CUMULATIVE)")
        else:
            print(f"  Trade Size: ${TRADE_SIZE:,.2f} (FIXED)")
        print(f"  Leverage: {LEVERAGE}x")
        print(f"  Spread: {SPREAD}")
        print(f"  Fee Rate: {FEE_PERCENT}%")
        print("=" * 60)
        
        process_all_backtests()


