import akshare as ak
import datetime as dt
from dateutil.relativedelta import relativedelta
from tqdm import tqdm
stock_df = ak.stock_info_a_code_name()

with open("../know/a_code.txt", "w", encoding="utf-8") as f:
    for idx,row in stock_df.iterrows():
        f.write(f"{row["code"]}\n")
with open("../konw/a_code_name.txt", "w", encoding="utf-8") as f:
    for idx,row in stock_df.iterrows():
        f.write(f"{row["code"]} {row["name"]}\n")

# print(stock_szse_area_summary_df)
# print(stock_szse_summary_df)