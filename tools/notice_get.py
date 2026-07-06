import time

import requests
import re
import os
import akshare as ak
from tqdm import tqdm


def get_a_code():
    list = []
    with open('../know/a_code.txt', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            list.append(line.strip())
    return list

def get_notice(a_code):
    notice_df = ak.stock_individual_notice_report(security=a_code,symbol="重大事项",begin_date="20260101")
    return notice_df

def write_notice(notice_df):
    with open("df.txt","w",encoding="utf-8") as f:
        for idx,row in notice_df.iterrows():
            f.write(f"{row["代码"]} {row["网址"]}\n")


def download_pdf(url, save_path="file.pdf"):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    try:
        resp = requests.get(url, headers=headers, stream=True, timeout=30)
        resp.raise_for_status()
        with open(save_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"下载完成：{save_path}")
    except Exception as e:
        print("下载失败：", e)

def get_donwload_list(notice_df):
    ann=[]
    for idx,row in notice_df.iterrows():
        ann.append((row["代码"],row["网址"]))
    return ann

def get_donwload_list_by_txt():
    ann = []
    with open("df.txt","r") as f:
        lines = f.readlines()
        for line in lines:
            ann.append(line.split())
    return ann
def re_url(stock,url):
    pattern = r'/(AN\d+)\.html'
    result = re.search(pattern, url)
    real_url = f"https://pdf.dfcfw.com/pdf/H2_{result.group(1)}_1.pdf"
    return real_url,result.group(1)

if __name__ == '__main__':
    a_code = get_a_code()[4001:]
    for code in a_code:
        try:
            df = get_notice(code)
            if len(df) == 0:
                continue
            download_lists = get_donwload_list(df)
            for stock,url in download_lists:
                real_url,code = re_url(stock,url)
                save_dir = f"../know/notice/{stock}"
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                save_path = f"{save_dir}/{code}.pdf"
                if os.path.exists(save_path):
                    continue
                download_pdf(real_url,save_path=save_path)
                time.sleep(0.3)
        except Exception as e:
            print(code)

