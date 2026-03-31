from FVG_projectX_bot.projectX.projectx_api_functions import fetch_data
from FVG_projectX_bot.projectX.projectx_api_functions import login_to_api
from dotenv import load_dotenv
import os

if __name__ == "__main__":
    load_dotenv()
    key = os.getenv("API_KEY")
    login = os.getenv("USERNAME")
    token = login_to_api(login, key)["token"]
    df_15 = fetch_data("CON.F.US.MGC.J26", "15min", 20000, token)
    df_15.to_csv("FVG_projectX_bot/backtest/data/MGCJ6/topstep_15min.csv")
    print(len(df_15))

    df_1 = fetch_data("CON.F.US.MGC.J26", "1min", 300000, token)
    df_1.to_csv("FVG_projectX_bot/backtest/data/MGCJ6/topstep_1min.csv")
    print(len(df_1))
