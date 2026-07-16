"""
=============================================================================
  全能金融资讯获取模块 — 宏观新闻 / 个股资讯 / 专业研报
  Giant Tide Announcement Download — Enhanced News & Research Fetcher
=============================================================================
  数据源 (via akshare):
    宏观: 东方财富、新浪财经、同花顺、百度经济日历
    个股: 东方财富个股新闻、个股公告
    研报: 东方财富个股研报 (含PDF链接)
=============================================================================
"""

import json
import math
import pprint
import re
import time
from typing import Optional

import akshare as ak
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# ==================== 1. 网络会话 ====================
def create_session():
    retry = Retry(
        total=2,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


session = create_session()
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "zh-CN,zh;q=0.9",
}

# ==================== 2. 工具函数 ====================

def _safe_fetch(func, source_name: str, **kwargs):
    """通用安全调用：捕获异常并统一返回格式"""
    try:
        df = func(**kwargs)
        if df is None or df.empty:
            return {
                "status": "empty",
                "source": source_name,
                "count": 0,
                "data": [],
            }
        # 清洗：替换 NaN / NaT 为 None，方便 JSON 序列化
        df = df.where(pd.notnull(df), None)
        records = df.to_dict(orient="records")
        # 二次保险：递归清理所有 NaN / Infinity，确保 JSON 可序列化
        records = _sanitize_for_json(records)
        return {
            "status": "success",
            "source": source_name,
            "count": len(records),
            "data": records,
        }
    except Exception as e:
        return {
            "status": "error",
            "source": source_name,
            "message": f"{type(e).__name__}: {str(e)[:300]}",
        }


def _normalize_stock_code(code: str) -> str:
    """标准化股票代码为 6 位数字字符串"""
    code = str(code).strip().zfill(6)
    return code


def _sanitize_for_json(obj):
    """递归地将对象中的 NaN / Infinity 浮点数替换为 None，确保 JSON 可序列化"""
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    return obj


# ==================== 3. 宏观大方向新闻 ====================

def get_macro_news_eastmoney(top: int = 20):
    """
    【宏观新闻】东方财富全球财经资讯 — 覆盖面广，实时性强
    返回 200 条以内，按时间降序
    """
    df = ak.stock_info_global_em()
    result = _safe_fetch(lambda: df, "eastmoney_global")
    if result["status"] == "success" and len(result["data"]) > top:
        result["data"] = result["data"][:top]
        result["count"] = len(result["data"])
    return result


def get_macro_news_sina(top: int = 20):
    """
    【宏观新闻】新浪财经全球财经 — 快讯风格，简洁明了
    返回最近 20 条
    """
    df = ak.stock_info_global_sina()
    result = _safe_fetch(lambda: df, "sina_global")
    if result["status"] == "success" and len(result["data"]) > top:
        result["data"] = result["data"][:top]
        result["count"] = len(result["data"])
    return result


def get_macro_news_ths(top: int = 20):
    """
    【宏观新闻】同花顺财经资讯 — 深度报道，政策解读
    返回最近 20 条
    """
    df = ak.stock_info_global_ths()
    result = _safe_fetch(lambda: df, "ths_global")
    if result["status"] == "success" and len(result["data"]) > top:
        result["data"] = result["data"][:top]
        result["count"] = len(result["data"])
    return result


def get_macro_news_digest(top: int = 10):
    """
    【宏观新闻】东方财富财经早餐 — 每日盘前重要资讯汇总，含隔夜外盘
    返回 400 条以内（多日累积），取最新的 top 条
    """
    df = ak.stock_info_cjzc_em()
    result = _safe_fetch(lambda: df, "eastmoney_digest")
    if result["status"] == "success" and len(result["data"]) > top:
        result["data"] = result["data"][:top]
        result["count"] = len(result["data"])
    return result


def get_economic_calendar():
    """
    【宏观参考】百度经济数据日历 — 重要经济指标发布时间表
    含：事件、预期值、前值、重要程度
    """
    df = ak.news_economic_baidu()
    return _safe_fetch(lambda: df, "baidu_economic_calendar")


def get_macro_news_all(top_per_source: int = 10):
    """
    【宏观新闻·综合】同时从多个数据源获取宏观新闻，去重合并
    数据源：东方财富、新浪财经、同花顺、东方财富早餐
    """
    sources = [
        ("eastmoney_global", get_macro_news_eastmoney),
        ("sina_global", get_macro_news_sina),
        ("ths_global", get_macro_news_ths),
        ("eastmoney_digest", get_macro_news_digest),
    ]

    all_news = []
    seen_titles = set()

    for src_name, src_func in sources:
        res = src_func(top=top_per_source)
        if res["status"] == "success":
            for item in res["data"]:
                title = item.get("标题", item.get("title", ""))
                if title and title not in seen_titles:
                    seen_titles.add(title)
                    item["_source"] = src_name
                    all_news.append(item)

    # 按时间降序排序
    all_news.sort(
        key=lambda x: x.get("发布时间", x.get("时间", "")) or "",
        reverse=True,
    )

    return {
        "status": "success",
        "source": "macro_combined",
        "count": len(all_news),
        "data": all_news,
    }


# ==================== 4. 个股资讯 ====================

def get_stock_news(code: str, top: int = 20):
    """
    【个股新闻】东方财富个股新闻
    code: 股票代码，如 '600519' (贵州茅台), '000001' (平安银行)
    返回字段：关键词、新闻标题、新闻内容、发布时间、文章来源、新闻链接
    """
    code = _normalize_stock_code(code)
    df = ak.stock_news_em(symbol=code)
    result = _safe_fetch(lambda: df, f"stock_news_{code}")
    if result["status"] == "success" and len(result["data"]) > top:
        result["data"] = result["data"][:top]
        result["count"] = len(result["data"])
    # 附加股票代码到每条记录
    for item in result.get("data", []):
        item["_stock_code"] = code
        # 统一添加原始链接字段，兼容不同列名
        if "_original_url" not in item:
            item["_original_url"] = item.get("新闻链接", item.get("url", item.get("链接", "")))
    return result


def get_stock_notices(code: str, top: int = 20):
    """
    【个股公告】上市公司全部公告 — 含定期报告、重大事项、政策影响等
    code: 股票代码，如 '600519'
    返回字段：代码、公告标题、公告时间、公告类型、网址（原文链接）
    """
    code = _normalize_stock_code(code)
    df = ak.stock_individual_notice_report(security=code)
    result = _safe_fetch(lambda: df, f"stock_notice_{code}")
    if result["status"] == "success" and len(result["data"]) > top:
        result["data"] = result["data"][:top]
        result["count"] = len(result["data"])
    for item in result.get("data", []):
        item["_stock_code"] = code
        # 统一添加原始链接字段，兼容不同列名
        if "_original_url" not in item:
            item["_original_url"] = item.get("网址", item.get("url", item.get("链接", "")))
    return result


# ==================== 5. 专业研报 ====================

def get_stock_research_report(code: str, top: int = 20):
    """
    【个股研报】东方财富个股研究报告 — 含评级、盈利预测、PDF 下载链接
    code: 股票代码，如 '600519'
    返回字段：
      - 报告名称、东财评级、机构名称
      - 近一月个股研报数
      - 2026/2027/2028 盈利预测（每股收益 + 市盈率）
      - 行业、日期
      - 报告PDF链接（可直接下载）
    """
    code = _normalize_stock_code(code)
    df = ak.stock_research_report_em(symbol=code)
    result = _safe_fetch(lambda: df, f"research_report_{code}")
    if result["status"] == "success" and len(result["data"]) > top:
        result["data"] = result["data"][:top]
        result["count"] = len(result["data"])
    for item in result.get("data", []):
        item["_stock_code"] = code
    return result


# ==================== 6. 即时行情 & 板块排行 ====================

def _market_prefix(code: str) -> str:
    """根据股票代码确定沪深/北京市场前缀"""
    if code.startswith(("600", "601", "603", "605", "688", "900")):
        return "sh"
    elif code.startswith(("8", "4", "920")):
        return "bj"
    else:
        return "sz"


def get_stock_spot(code: str):
    """
    【即时行情】腾讯行情接口 — 个股实时价格、买卖盘
    code: 6位股票代码，如 '000001'、'600519'
    返回：当前价、涨跌、成交量、内外盘、买一/卖一
    """
    code = str(code).strip().zfill(6)
    prefix = _market_prefix(code)
    url = f"http://qt.gtimg.cn/q={prefix}{code}"

    try:
        response = session.get(url, headers=HEADERS, timeout=5.0)
        response.raise_for_status()
        response.encoding = "gbk"
        text = response.text
    except requests.RequestException as exc:
        return {
            "status": "error",
            "error_type": "upstream",
            "message": f"Failed to fetch data from upstream: {str(exc)}",
        }

    if "pv_none_" in text or len(text) < 50:
        return {
            "status": "error",
            "error_type": "not_found",
            "message": f"Stock code '{code}' not found or invalid.",
        }

    try:
        data_str = text.split("=")[1].replace('"', "").replace(";\n", "")
        parts = data_str.split("~")

        return _sanitize_for_json({
            "status": "success",
            "metadata": {
                "code": code,
                "name": parts[1].strip(),
                "market": prefix.upper(),
                "timestamp": parts[30],
            },
            "quotes": {
                "current_price": float(parts[3]),
                "prev_close": float(parts[4]),
                "open": float(parts[5]),
                "volume_hand": int(parts[6]),
                "outer_volume": int(parts[7]),
                "inner_volume": int(parts[8]),
                "bid_1": {"price": float(parts[9]), "volume": int(parts[10])},
                "ask_1": {"price": float(parts[19]), "volume": int(parts[20])},
            },
        })
    except (IndexError, ValueError) as exc:
        return {
            "status": "error",
            "error_type": "parse",
            "message": f"Data parsing error: {str(exc)}",
        }


def get_sector_ranking(
    top: int = 10,
    direction: str = "up",
    sector_type: str = "industry",
):
    """
    【板块排行】新浪行业/概念板块涨跌排行
    direction: "up"=涨幅榜, "down"=跌幅榜
    sector_type: "industry"=新浪行业, "concept"=新浪概念
    """
    if sector_type == "industry":
        url = "http://vip.stock.finance.sina.com.cn/q/view/newSinaHy.php"
    else:
        url = "http://vip.stock.finance.sina.com.cn/q/view/newSinaConcept.php"

    raw_text = ""
    max_retries = 3
    last_error = None
    for attempt in range(max_retries):
        try:
            response = session.get(url, headers=HEADERS, timeout=6.0)
            response.raise_for_status()
            response.encoding = "gbk"
            raw_text = response.text
            if raw_text:
                break
        except requests.RequestException as exc:
            last_error = exc
            if attempt == max_retries - 1:
                return {
                    "status": "error",
                    "error_type": "upstream",
                    "message": f"Failed after {max_retries} attempts: {str(exc)}",
                }

    try:
        json_match = re.search(r"\{.*\}", raw_text)
        if not json_match:
            raise ValueError("No valid JSON structure found in upstream response.")
        raw_data = json.loads(json_match.group())
    except Exception as exc:
        return {
            "status": "error",
            "error_type": "parse",
            "message": f"Error parsing raw text to JSON: {str(exc)}",
        }

    ranking_list = []
    for s_code, info_str in raw_data.items():
        normalized_str = info_str.replace("，", ",")
        parts = [p.strip() for p in normalized_str.split(",")]
        if len(parts) < 13:
            continue

        try:
            raw_name = parts[1].strip()
            sector_name = raw_name.replace(s_code, "").strip()
            ranking_list.append({
                "sector_code": s_code.strip(),
                "sector_name": sector_name if sector_name else raw_name,
                "stock_count": int(parts[2]),
                "avg_price": round(float(parts[3]), 2),
                "avg_change_amount": round(float(parts[4]), 2),
                "change_percent": round(float(parts[5]), 4),
                "total_volume_share": int(parts[6]),
                "total_turnover_yuan": int(parts[7]),
                "top_gainer_code": parts[8].strip(),
                "top_gainer_change_amount": round(float(parts[9]), 3),
                "top_gainer_change_percent": round(float(parts[10]), 3),
                "top_gainer_name": parts[12].strip(),
            })
        except (ValueError, IndexError):
            continue

    reverse_sort = (direction == "up")
    sorted_list = sorted(ranking_list, key=lambda x: x["change_percent"], reverse=reverse_sort)
    final_list = sorted_list[:top]

    return _sanitize_for_json({
        "status": "success",
        "direction": direction,
        "sector_type": sector_type,
        "count": len(final_list),
        "data": final_list,
    })


def get_stock_sse_summary():
    """【市场概况】上交所市场总貌"""
    try:
        df = ak.stock_sse_summary()
        df = df.where(pd.notnull(df), None)
        records = _sanitize_for_json(df.to_dict(orient="records"))
        return {"status": "success", "data": records}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def get_stock_szse_summary(date: str):
    """【市场概况】深交所市场总貌，date 格式 YYYYMMDD"""
    try:
        df = ak.stock_szse_summary(date=date)
        df = df.where(pd.notnull(df), None)
        records = _sanitize_for_json(df.to_dict(orient="records"))
        return {"status": "success", "data": records}
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ==================== 7. 综合查询 ====================

def get_stock_full_report(code: str, news_top: int = 10, report_top: int = 10):
    """
    【一站式查询】获取指定个股的：新闻 + 研报 + 公告
    返回一份综合报告，方便 AI 分析或直接阅读
    """
    code = _normalize_stock_code(code)

    results = {
        "code": code,
        "fetch_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "news": get_stock_news(code, top=news_top),
        "research_reports": get_stock_research_report(code, top=report_top),
    }

    # 公告接口较慢，单独捕获
    try:
        results["notices"] = get_stock_notices(code, top=news_top)
    except Exception as e:
        results["notices"] = {
            "status": "error",
            "message": f"Notices fetch failed: {str(e)[:200]}",
        }

    return results
