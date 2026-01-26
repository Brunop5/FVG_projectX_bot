#!/usr/bin/env python3
"""
Filter backtest results based on specific criteria.
"""

import pandas as pd

# ===== CONFIGURATION =====
INPUT_CSV = "gold_results/optimization_results_15min_20260125_135551.csv"
OUTPUT_CSV = "filtered_backtest_results.csv"

# Filter criteria
MIN_TOTAL_PNL = 10
MAX_DRAWDOWN = 3500
MIN_TRADES_PER_DAY = 0.1
MAX_BACKTEST_PERIOD_DAYS = 240
# =========================

# Read the CSV file
print(f"Reading {INPUT_CSV}...")
df = pd.read_csv(INPUT_CSV)

print(f"Total rows in original CSV: {len(df)}")


# total_pnl is at least 5000
filtered_df = df[df['total_pnl'] >= MIN_TOTAL_PNL]

# max_drawdown is not more than 3000 (<= 3000)
#filtered_df = filtered_df[filtered_df['max_drawdown'] <= MAX_DRAWDOWN]

# trades_per_day is more than 0.42 (> 0.42)
#filtered_df = filtered_df[filtered_df['trades_per_day'] > MIN_TRADES_PER_DAY]

# backtest_period_days is not more than 204 (<= 204)
filtered_df = filtered_df[filtered_df['backtest_period_days'] <= MAX_BACKTEST_PERIOD_DAYS]

print(f"Rows after filtering: {len(filtered_df)}")

# Calculate total_pnl / max_drawdown ratio
# Handle division by zero: if max_drawdown is 0 or very small, set ratio to a high value or NaN
filtered_df = filtered_df.copy()
filtered_df['pnl_drawdown_ratio'] = filtered_df.apply(
    lambda row: row['total_pnl'] / row['max_drawdown'] if row['max_drawdown'] > 0 else float('inf'),
    axis=1
)

# Sort by ratio (descending) and get top 20
top_20 = filtered_df.nlargest(20, 'pnl_drawdown_ratio')

# Print top 20 results
print(f"\n{'='*80}")
print(f"🏆 TOP 20 RESULTS BY TOTAL_PNL / MAX_DRAWDOWN RATIO")
print(f"{'='*80}")
print(f"{'Rank':<6} {'ID':<6} {'Total P&L':<12} {'Max DD':<12} {'Ratio':<12} {'Win Rate':<10} {'Trades':<8} {'Timeframe':<10}")
print(f"{'-'*80}")

for idx, (_, row) in enumerate(top_20.iterrows(), 1):
    ratio = row['pnl_drawdown_ratio']
    ratio_str = f"{ratio:.2f}" if ratio != float('inf') else "∞"
    print(f"{idx:<6} {int(row.get('id', 0)):<6} ${row['total_pnl']:>10,.2f} ${row['max_drawdown']:>10,.2f} {ratio_str:>12} {row.get('win_rate', 0):>8.2f}% {int(row.get('total_trades', 0)):<8} {str(row.get('timeframe', '')):<10}")

print(f"{'='*80}\n")

# Save top 20 to new CSV
if len(top_20) > 0:
    top_20.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Top 20 results saved to {OUTPUT_CSV}")
    print(f"\nFilter criteria applied:")
    print(f"  - strategy_failed = False")
    print(f"  - total_pnl >= {MIN_TOTAL_PNL}")
    print(f"  - max_drawdown <= {MAX_DRAWDOWN}")
    print(f"  - trades_per_day > {MIN_TRADES_PER_DAY}")
    print(f"  - backtest_period_days <= {MAX_BACKTEST_PERIOD_DAYS}")
    print(f"  - Sorted by: total_pnl / max_drawdown ratio (descending)")
    print(f"  - Saved: Top 20 results")
else:
    print("⚠️  No rows matched the filter criteria. No output file created.")

