#!/usr/bin/env python3
"""
Filter backtest results based on specific criteria.
"""

import pandas as pd

# ===== CONFIGURATION =====
INPUT_CSV = "backtest_summary.csv"
OUTPUT_CSV = "filtered_backtest_results.csv"

# Filter criteria
MIN_TOTAL_PNL = 5000
MAX_DRAWDOWN = 3000
MIN_TRADES_PER_DAY = 0.42
MAX_BACKTEST_PERIOD_DAYS = 204
# =========================

# Read the CSV file
print(f"Reading {INPUT_CSV}...")
df = pd.read_csv(INPUT_CSV)

print(f"Total rows in original CSV: {len(df)}")

# Apply filters
# strategy_failed is False
filtered_df = df[df['strategy_failed'] == False].copy()

# total_pnl is at least 5000
filtered_df = filtered_df[filtered_df['total_pnl'] >= MIN_TOTAL_PNL]

# max_drawdown is not more than 3000 (<= 3000)
filtered_df = filtered_df[filtered_df['max_drawdown'] <= MAX_DRAWDOWN]

# trades_per_day is more than 0.42 (> 0.42)
filtered_df = filtered_df[filtered_df['trades_per_day'] > MIN_TRADES_PER_DAY]

# backtest_period_days is not more than 204 (<= 204)
filtered_df = filtered_df[filtered_df['backtest_period_days'] <= MAX_BACKTEST_PERIOD_DAYS]

print(f"Rows after filtering: {len(filtered_df)}")

# Save to new CSV
if len(filtered_df) > 0:
    filtered_df.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Filtered results saved to {OUTPUT_CSV}")
    print(f"\nFilter criteria applied:")
    print(f"  - strategy_failed = False")
    print(f"  - total_pnl >= {MIN_TOTAL_PNL}")
    print(f"  - max_drawdown <= {MAX_DRAWDOWN}")
    print(f"  - trades_per_day > {MIN_TRADES_PER_DAY}")
    print(f"  - backtest_period_days <= {MAX_BACKTEST_PERIOD_DAYS}")
else:
    print("⚠️  No rows matched the filter criteria. No output file created.")

