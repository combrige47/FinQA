import datetime
import os

import akshare as ak
import pandas as pd
from mysqlClient import MySQLClient
import pymysql

# os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
# os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"
# # 内网Milvus不走代理
# os.environ["NO_PROXY"] = "127.0.0.1,124.70.51.221"

lastDay = (datetime.datetime.now()-datetime.timedelta(days=1)).strftime("%Y%m%d")


def save_stock(db, df):
    sql = """
    INSERT INTO stock_daily
    (symbol,date,open,high,low,close,volume)
    VALUES (%s,%s,%s,%s,%s,%s,%s)
    ON DUPLICATE KEY UPDATE
        open=VALUES(open),
        high=VALUES(high),
        low=VALUES(low),
        close=VALUES(close),
        volume=VALUES(volume)
    """

    data = [
        (
            row.symbol,
            row.date.date(),
            row.open,
            row.high,
            row.low,
            row.close,
            int(row.volume)
        )
        for row in df.itertuples(index=False)
    ]

    db.executemany(sql, data)

def get_stock_code():
    stock_list = []
    with open("../know/a_code.txt", "r") as f:
        for line in f.readlines():
            stock_list.append(line.strip())
    return stock_list

def get_stock_data(stock_code):
    df = ak.stock_zh_a_hist(symbol="zs"+stock_code, period="daily", start_date="20250601", end_date=lastDay, adjust="qfq")
    df = df.rename(columns={
        "日期": "date",
        "开盘": "open",
        "最高": "high",
        "最低": "low",
        "收盘": "close",
        "成交量": "volume"
    })

    # 格式转换：确保 date 列是 datetime 类型
    df['date'] = pd.to_datetime(df['date'])
    df['symbol'] = stock_code  # 添加代码列

    # 筛选需要的列（可选，AKQuant 会自动忽略多余列）
    df = df[["date", "open", "high", "low", "close", "volume", "symbol"]]
    return df

def exist(db,stock_code):
    check_sql = """
            SELECT 1 FROM stock_daily
            WHERE symbol = %s
            LIMIT 1
        """
    db.execute(check_sql, stock_code)
    row = db.fetchone()
    return row is not None and row[0] == 1


if __name__ == '__main__':
    df = ak.stock_zh_a_hist(symbol="000001", period="daily", start_date="20250601", end_date=lastDay, adjust="qfq")
    print(df)
    # db = MySQLClient()
    # for code in get_stock_code():
    #     try:
    #         if exist(db,code):
    #             #print(f"已插入{code}，跳过")
    #             continue
    #         df = get_stock_data(code)
    #         save_stock(db, df)
    #         print(f"成功插入{code}")
    #         db.commit()
    #     except Exception as e:
    #         print(code,e)
    # db.close()


