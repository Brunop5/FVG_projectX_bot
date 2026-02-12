import pandas as pd

df = pd.read_csv("FVG_projectX_bot/backtest/GOLD_BACKTEST/backtest_trades.csv")

print(len(df[df["pnl"] > 0]) / len(df))
print(df["order_size"].mean() * 6)
print(df[df["pnl"] > 0]["pnl"].mean() * 6)
print(df[df["pnl"] < 0]["pnl"].mean() * 6)

print(df)