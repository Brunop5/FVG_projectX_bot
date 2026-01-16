#!/usr/bin/env python3
"""
Simple trade simulator that reads trades from CSV and simulates execution.
"""

import csv
import matplotlib.pyplot as plt
from datetime import datetime

# ===== CONFIGURATION =====
CSV_FILE = "gold_results/3/backtest_trades_CON.F.US.MGC.G26_20260116.csv"
INITIAL_CAPITAL = 10000  # Starting capital in dollars
TRADE_SIZE = 10000  # Dollar value of each trade order
LEVERAGE = 20  # Leverage multiplier (1 = no leverage)
SPREAD = 0.2 # Spread in price units (always exact)
FEE_PERCENT = 0.018  # Fee percentage (0.02% = 0.0002 as decimal)
# =========================

capital = INITIAL_CAPITAL
trades_executed = 0
winning_trades = 0
losing_trades = 0
total_pnl = 0.0
total_fees_paid = 0.0
equity_curve = []  # List to store equity curve data

# Read and process trades
with open(CSV_FILE, 'r') as f:
    reader = csv.DictReader(f)
    
    for row in reader:
        # Parse trade data
        side = row['side'].strip()
        entry_price = float(row['entry_price'])
        exit_price = float(row['exit_price'])
        
        # Apply spread: when buying, pay ask price (entry + spread/2), when selling, get bid price (exit - spread/2)
        if side == 'BUY':  # Long position
            actual_entry_price = entry_price + (SPREAD / 2)  # Pay ask price
            actual_exit_price = exit_price - (SPREAD / 2)  # Get bid price
        else:  # SELL = Short position
            actual_entry_price = entry_price - (SPREAD / 2)  # Get bid price (short entry)
            actual_exit_price = exit_price + (SPREAD / 2)  # Pay ask price (short exit)
        
        # Calculate position size based on actual entry price
        position_value = TRADE_SIZE * LEVERAGE
        num_units = position_value / actual_entry_price
        
        # Calculate fees: 0.02% of the position value (both entry and exit)
        entry_fee = position_value * (FEE_PERCENT / 100)
        exit_fee = (num_units * actual_exit_price) * (FEE_PERCENT / 100)
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
        elif trade_pnl < 0:
            losing_trades += 1
        
        # Record equity curve point
        equity_curve.append({
            'trade_number': trades_executed,
            'balance': capital,
            'cumulative_pnl': total_pnl
        })

# Print results
print("=" * 60)
print("TRADE SIMULATION RESULTS")
print("=" * 60)
print(f"Initial Capital:     ${INITIAL_CAPITAL:,.2f}")
print(f"Trade Size:          ${TRADE_SIZE:,.2f}")
print(f"Leverage:            {LEVERAGE}x")
print(f"Spread:              {SPREAD} (applied to each trade)")
print(f"Fee Rate:            {FEE_PERCENT}% (on entry and exit)")
print(f"Final Capital:       ${capital:,.2f}")
print(f"Total P&L:           ${total_pnl:,.2f}")
print(f"Return:              {(capital / INITIAL_CAPITAL - 1) * 100:.2f}%")
print("-" * 60)
print(f"Total Trades:        {trades_executed}")
print(f"Winning Trades:      {winning_trades}")
print(f"Losing Trades:       {losing_trades}")
if trades_executed > 0:
    print(f"Win Rate:            {(winning_trades / trades_executed) * 100:.2f}%")
print(f"Total Fees Paid:     ${total_fees_paid:,.2f}")
print("=" * 60)


# Create and display equity curve graph
if equity_curve:
    trade_numbers = [point['trade_number'] for point in equity_curve]
    balances = [point['balance'] for point in equity_curve]
    
    plt.figure(figsize=(12, 6))
    plt.plot(trade_numbers, balances, linewidth=2, color='#2E86AB')
    plt.axhline(y=INITIAL_CAPITAL, color='gray', linestyle='--', linewidth=1, label='Initial Capital')
    plt.xlabel('Trade Number', fontsize=12)
    plt.ylabel('Account Balance ($)', fontsize=12)
    plt.title('Equity Curve', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


