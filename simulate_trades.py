#!/usr/bin/env python3
"""
Simple trade simulator that reads trades from CSV and simulates execution.
"""

import csv

# ===== CONFIGURATION =====
CSV_FILE = "backtest_trades_CON.F.US.MGC.G26_20260113.csv"
INITIAL_CAPITAL = 10000  # Starting capital in dollars
TRADE_SIZE = 1000  # Dollar value of each trade order
LEVERAGE = 20  # Leverage multiplier (1 = no leverage)
# =========================

capital = INITIAL_CAPITAL
trades_executed = 0
winning_trades = 0
losing_trades = 0
total_pnl = 0.0

# Read and process trades
with open(CSV_FILE, 'r') as f:
    reader = csv.DictReader(f)
    
    for row in reader:
        # Parse trade data
        side = row['side'].strip()
        entry_price = float(row['entry_price'])
        exit_price = float(row['exit_price'])
        fees = float(row['fees'])
        
        # Calculate position size
        position_value = TRADE_SIZE * LEVERAGE
        num_units = position_value / entry_price
        
        # Calculate P&L based on side
        if side == 'BUY':  # Long position
            price_change = exit_price - entry_price
        else:  # SELL = Short position
            price_change = entry_price - exit_price
        
        # Calculate trade P&L
        trade_pnl = price_change * num_units - fees
        
        # Update capital
        capital += trade_pnl
        total_pnl += trade_pnl
        trades_executed += 1
        
        if trade_pnl > 0:
            winning_trades += 1
        elif trade_pnl < 0:
            losing_trades += 1

# Print results
print("=" * 60)
print("TRADE SIMULATION RESULTS")
print("=" * 60)
print(f"Initial Capital:     ${INITIAL_CAPITAL:,.2f}")
print(f"Trade Size:          ${TRADE_SIZE:,.2f}")
print(f"Leverage:            {LEVERAGE}x")
print(f"Final Capital:       ${capital:,.2f}")
print(f"Total P&L:           ${total_pnl:,.2f}")
print(f"Return:              {(capital / INITIAL_CAPITAL - 1) * 100:.2f}%")
print("-" * 60)
print(f"Total Trades:        {trades_executed}")
print(f"Winning Trades:      {winning_trades}")
print(f"Losing Trades:       {losing_trades}")
if trades_executed > 0:
    print(f"Win Rate:            {(winning_trades / trades_executed) * 100:.2f}%")
print("=" * 60)

