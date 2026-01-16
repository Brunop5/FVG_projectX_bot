#!/usr/bin/env python3
"""
Script to add losing streak metrics to final_result.csv:
- Most losing trades in a row
- Average most losing trades in a row each month
- Dollar value of most losses in a row
- Dollar value of average most losses in a month
- Average of maximum monthly drawdowns
"""

import pandas as pd
import os
import glob
from datetime import datetime

# ===== CONFIGURATION =====
FINAL_RESULT_CSV = "gold_results/final_result.csv"
GOLD_RESULTS_DIR = "gold_results"
# =========================

def find_trades_file(result_id):
    """Find the trades CSV file for a given result ID"""
    result_dir = os.path.join(GOLD_RESULTS_DIR, str(result_id))
    if not os.path.exists(result_dir):
        return None
    
    # Look for trades CSV files
    pattern = os.path.join(result_dir, "backtest_trades_*.csv")
    files = glob.glob(pattern)
    
    if not files:
        return None
    
    # Return the first match (should only be one)
    return files[0]

def calculate_avg_max_monthly_drawdown(trades_df, initial_balance):
    """
    Calculate average of maximum monthly drawdowns from trades DataFrame.
    For each month, finds the maximum drawdown (peak to trough), then averages all monthly maximums.
    
    Args:
        trades_df: DataFrame with trades (must have 'entry_time' and 'pnl' columns)
        initial_balance: Starting balance for the backtest
    
    Returns:
        Average of maximum monthly drawdowns (dollar value)
    """
    if trades_df.empty or 'entry_time' not in trades_df.columns:
        return 0.0
    
    # Ensure entry_time is datetime
    trades_df = trades_df.copy()
    trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
    
    # Sort by entry time
    trades_df = trades_df.sort_values('entry_time').reset_index(drop=True)
    
    # Calculate cumulative PnL and equity curve
    trades_df['cumulative_pnl'] = trades_df['pnl'].cumsum()
    trades_df['balance'] = initial_balance + trades_df['cumulative_pnl']
    
    # Group by month
    trades_df['year_month'] = trades_df['entry_time'].dt.to_period('M')
    
    monthly_drawdowns = []
    
    for month, month_trades in trades_df.groupby('year_month'):
        month_trades = month_trades.sort_values('entry_time').reset_index(drop=True)
        
        if len(month_trades) == 0:
            continue
        
        # Calculate maximum drawdown for this month
        # Drawdown = peak - trough (largest decline from a peak within the month)
        balances = month_trades['balance'].values
        peak = balances[0]  # Start with first balance of the month
        max_drawdown = 0.0
        
        for balance in balances:
            if balance > peak:
                peak = balance  # Update peak if we hit a new high
            drawdown = peak - balance  # Calculate drawdown from current peak
            if drawdown > max_drawdown:
                max_drawdown = drawdown  # Track the maximum drawdown in this month
        
        # Store the maximum drawdown for this month (include all months, even if drawdown is 0)
        monthly_drawdowns.append(max_drawdown)
    
    # Return average of maximum monthly drawdowns
    if monthly_drawdowns:
        return sum(monthly_drawdowns) / len(monthly_drawdowns)
    else:
        return 0.0

def calculate_losing_streaks(trades_df):
    """
    Calculate losing streak metrics from trades DataFrame.
    Returns:
        - most_losing_trades_in_row: Maximum consecutive losing trades
        - avg_most_losing_trades_per_month: Average of max consecutive losses per month
        - dollar_value_most_losses_in_row: Sum of PnL for longest losing streak
        - dollar_value_avg_most_losses_per_month: Average dollar value of max consecutive losses per month
    """
    if trades_df.empty:
        return 0, 0.0, 0.0, 0.0
    
    # Ensure entry_time is datetime
    if 'entry_time' in trades_df.columns:
        trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
    
    # Sort by entry time to ensure correct order
    trades_df = trades_df.sort_values('entry_time').reset_index(drop=True)
    
    # Identify losing trades (pnl < 0)
    trades_df['is_loss'] = trades_df['pnl'] < 0
    
    # Calculate consecutive losing trades
    most_losing_trades_in_row = 0
    current_streak = 0
    longest_streak_start = None
    longest_streak_end = None
    
    for idx, is_loss in enumerate(trades_df['is_loss']):
        if is_loss:
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
        longest_streak_trades = trades_df.iloc[longest_streak_start:longest_streak_end + 1]
        dollar_value_most_losses_in_row = longest_streak_trades['pnl'].sum()
    
    # Calculate monthly metrics
    if 'entry_time' in trades_df.columns:
        trades_df['year_month'] = trades_df['entry_time'].dt.to_period('M')
        
        monthly_max_streaks = []
        monthly_max_streak_values = []
        
        for month, month_trades in trades_df.groupby('year_month'):
            month_trades = month_trades.sort_values('entry_time').reset_index(drop=True)
            month_trades['is_loss'] = month_trades['pnl'] < 0
            
            # Find all losing streaks in this month
            max_streak = 0
            max_streak_value = 0.0
            current_streak = 0
            current_streak_start = None
            
            for idx, is_loss in enumerate(month_trades['is_loss']):
                if is_loss:
                    if current_streak == 0:
                        current_streak_start = idx
                    current_streak += 1
                else:
                    # End of streak
                    if current_streak > 0:
                        if current_streak > max_streak:
                            max_streak = current_streak
                            streak_end = idx - 1
                            streak_trades = month_trades.iloc[current_streak_start:streak_end + 1]
                            max_streak_value = streak_trades['pnl'].sum()
                    current_streak = 0
                    current_streak_start = None
            
            # Handle case where streak continues to end of month
            if current_streak > 0:
                if current_streak > max_streak:
                    max_streak = current_streak
                    streak_end = len(month_trades) - 1
                    streak_trades = month_trades.iloc[current_streak_start:streak_end + 1]
                    max_streak_value = streak_trades['pnl'].sum()
            
            if max_streak > 0:
                monthly_max_streaks.append(max_streak)
                monthly_max_streak_values.append(max_streak_value)
        
        avg_most_losing_trades_per_month = sum(monthly_max_streaks) / len(monthly_max_streaks) if monthly_max_streaks else 0.0
        dollar_value_avg_most_losses_per_month = sum(monthly_max_streak_values) / len(monthly_max_streak_values) if monthly_max_streak_values else 0.0
    else:
        avg_most_losing_trades_per_month = 0.0
        dollar_value_avg_most_losses_per_month = 0.0
    
    return (
        most_losing_trades_in_row,
        avg_most_losing_trades_per_month,
        dollar_value_most_losses_in_row,
        dollar_value_avg_most_losses_per_month
    )

def process_final_result():
    """Process final_result.csv and add losing streak metrics"""
    
    if not os.path.exists(FINAL_RESULT_CSV):
        print(f"❌ Error: {FINAL_RESULT_CSV} not found!")
        return
    
    # Read final_result.csv
    print(f"📖 Reading {FINAL_RESULT_CSV}...")
    df = pd.read_csv(FINAL_RESULT_CSV)
    
    # Check if columns already exist
    new_columns = [
        'most_losing_trades_in_row',
        'avg_most_losing_trades_per_month',
        'dollar_value_most_losses_in_row',
        'dollar_value_avg_most_losses_per_month',
        'avg_max_monthly_drawdown'
    ]
    
    # Remove existing columns if they exist (to recalculate)
    for col in new_columns:
        if col in df.columns:
            df = df.drop(columns=[col])
            print(f"   Removed existing column: {col}")
    
    # Initialize new columns
    for col in new_columns:
        df[col] = None
    
    print(f"📊 Processing {len(df)} backtest results...")
    
    # Process each row
    for idx, row in df.iterrows():
        result_id = row['id']
        print(f"   Processing ID {result_id} ({idx + 1}/{len(df)})...", end=' ')
        
        # Find trades file
        trades_file = find_trades_file(result_id)
        
        if trades_file is None:
            print(f"⚠️  No trades file found")
            continue
        
        try:
            # Read trades CSV
            trades_df = pd.read_csv(trades_file)
            
            # Get initial balance from final_result.csv
            # If not available, calculate from final_balance and total_pnl
            if 'initial_balance' in row and pd.notna(row['initial_balance']):
                initial_balance = float(row['initial_balance'])
            elif 'final_balance' in row and 'total_pnl' in row and pd.notna(row['final_balance']) and pd.notna(row['total_pnl']):
                initial_balance = float(row['final_balance']) - float(row['total_pnl'])
            else:
                # Fallback: calculate from trades (first balance = initial + cumulative PnL at first trade)
                # Or use default
                initial_balance = 50000.0
            
            # Calculate losing streak metrics
            (
                most_losing_trades_in_row,
                avg_most_losing_trades_per_month,
                dollar_value_most_losses_in_row,
                dollar_value_avg_most_losses_per_month
            ) = calculate_losing_streaks(trades_df)
            
            # Calculate average of maximum monthly drawdowns
            avg_max_monthly_drawdown = calculate_avg_max_monthly_drawdown(trades_df, initial_balance)
            
            # Update DataFrame
            df.at[idx, 'most_losing_trades_in_row'] = most_losing_trades_in_row
            df.at[idx, 'avg_most_losing_trades_per_month'] = round(avg_most_losing_trades_per_month, 2)
            df.at[idx, 'dollar_value_most_losses_in_row'] = round(dollar_value_most_losses_in_row, 2)
            df.at[idx, 'dollar_value_avg_most_losses_per_month'] = round(dollar_value_avg_most_losses_per_month, 2)
            df.at[idx, 'avg_max_monthly_drawdown'] = round(avg_max_monthly_drawdown, 2)
            
            print(f"✅ Most losses in row: {most_losing_trades_in_row}, Avg per month: {avg_most_losing_trades_per_month:.2f}, Avg max monthly DD: ${avg_max_monthly_drawdown:.2f}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save updated CSV
    print(f"\n💾 Saving updated {FINAL_RESULT_CSV}...")
    df.to_csv(FINAL_RESULT_CSV, index=False)
    print(f"✅ Done! Added {len(new_columns)} new columns to {FINAL_RESULT_CSV}")

if __name__ == "__main__":
    process_final_result()

