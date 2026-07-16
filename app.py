import os
import sys
import traceback
from pathlib import Path
from typing import List, Optional
import uvicorn
from openai import OpenAI
from fastapi import FastAPI, Query, File, UploadFile, HTTPException, APIRouter, status
from fastapi.responses import FileResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
from schedule import start_scheduler, scheduler

from p import (
    get_macro_news_eastmoney,
    get_macro_news_sina,
    get_macro_news_ths,
    get_macro_news_digest,
    get_macro_news_all,
    get_economic_calendar,
    get_stock_news,
    get_stock_notices,
    get_stock_research_report,
    get_stock_full_report,
    get_stock_spot,
    get_sector_ranking,
    get_stock_sse_summary,
    get_stock_szse_summary,
)

# 确保项目根目录在 sys.path 中，以便导入 sibling package
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
from vector.milvus_stroe import MilvusImporter
from data import kb_service

API_KEY = os.environ.get("DEEPSEEK_API_KEY")
DASHSCOPE_API_KEY = os.environ.get("BAILIAN_API_KEY")
DEFAULT_COLLECTION = "financial_chunk"
TOP_K = 5
CANDIDATE_K = 100


class QueryRequest(BaseModel):
    query: str
    top_k: int = TOP_K
    kb_id: int | None = None


class KBCreateRequest(BaseModel):
    creator_id: int
    name: str
    description: str = ""


importer: Optional[MilvusImporter] = None


class BatchQueryReq(BaseModel):
    queries: List[str]


@asynccontextmanager
async def lifespan(app: FastAPI):
    global importer
    print("加载资源")
    print("正在连接Milvus并初始化千问Embedding客户端")
    importer = MilvusImporter(
        collection_name=DEFAULT_COLLECTION,
    )
    # 验证 collection 存在（默认 collection 不存在时仅警告，不影响 KB 功能）
    colls = importer.client.list_collections()
    if DEFAULT_COLLECTION not in colls:
        print(f"警告: 默认集合 {DEFAULT_COLLECTION} 不存在，/search 和 /query 需指定 kb_id")
    # 预热客户端（验证 API Key 可用）
    _ = importer.model
    start_scheduler()
    print("=== 资源加载完成，服务就绪 ===")

    yield

    print("=== 服务关闭，释放资源 ===")
    importer.close()
    scheduler.shutdown()


app = FastAPI(title="FinQA Dense Retriever Service", lifespan=lifespan)


# ═══════════════════════════════════════════════════════════════════════
# 知识库管理 API
# ═══════════════════════════════════════════════════════════════════════

@app.post("/kb/create")
async def create_kb(req: KBCreateRequest):
    """创建知识库"""
    try:
        return kb_service.create_kb(req.creator_id, req.name, req.description, importer)
    except kb_service.DuplicateError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"创建知识库失败: {e}")


@app.get("/kb/list")
async def list_kbs(user_id: int = Query(..., description="用户 ID（对应 fp_user.id）")):
    """列出用户的所有知识库"""
    try:
        kb_list = kb_service.list_kbs(user_id)
        return {"kb_list": kb_list}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"查询知识库列表失败: {e}")


@app.get("/kb/{kb_id}")
async def get_kb(kb_id: int, user_id: int = Query(..., description="用户 ID")):
    """获取单个知识库详情"""
    result = kb_service.get_kb(kb_id, user_id)
    if not result:
        raise HTTPException(status_code=404, detail=f"知识库 {kb_id} 不存在或无权访问")
    return result


@app.delete("/kb/{kb_id}")
async def delete_kb(kb_id: int, user_id: int = Query(..., description="用户 ID")):
    """删除知识库"""
    try:
        return kb_service.delete_kb(kb_id, user_id, importer)
    except kb_service.NotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除知识库失败: {e}")


@app.post("/kb/{kb_id}/upload")
async def upload_file(
        kb_id: int,
        user_id: int = Query(..., description="上传者 ID"),
        file: UploadFile = File(...),
):
    """上传文件到知识库"""
    try:
        contents = await file.read()
        return kb_service.upload_to_kb(
            kb_id, user_id, file.filename, contents, importer
        )
    except kb_service.NotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"上传失败: {e}")


@app.get("/kb/{kb_id}/documents")
async def list_documents(
        kb_id: int,
        user_id: int = Query(..., description="用户 ID"),
):
    """列出知识库中的所有文档"""
    try:
        docs = kb_service.list_documents(kb_id, user_id)
        return {"documents": docs}
    except kb_service.NotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"查询文档列表失败: {e}")


@app.delete("/kb/{kb_id}/documents/{doc_id}")
async def delete_document(
        kb_id: int,
        doc_id: int,
        user_id: int = Query(..., description="用户 ID"),
):
    """从知识库中删除指定文档（doc_id 为 fp_document.id）"""
    try:
        return kb_service.delete_document(kb_id, doc_id, user_id, importer)
    except kb_service.NotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除文档失败: {e}")


@app.get("/kb/{kb_id}/doc-by-milvus-id")
async def get_doc_by_milvus_id(
        kb_id: int,
        milvus_doc_id: str = Query(..., description="Milvus 中的 doc_id"),
        user_id: int = Query(..., description="用户 ID"),
):
    """通过 Milvus doc_id 召回文档原文（JSON 格式，含完整 content）"""
    result = kb_service.get_document_by_milvus_doc_id(kb_id, milvus_doc_id, user_id)
    if not result:
        raise HTTPException(status_code=404, detail="文档不存在或无权访问")
    return result


@app.get("/kb/{kb_id}/doc-by-milvus-id/download")
async def download_doc_by_milvus_id(
        kb_id: int,
        milvus_doc_id: str = Query(..., description="Milvus 中的 doc_id"),
        user_id: int = Query(..., description="用户 ID"),
):
    """通过 Milvus doc_id 直接下载 md 文件"""
    result = kb_service.get_document_by_milvus_doc_id(kb_id, milvus_doc_id, user_id)
    if not result:
        raise HTTPException(status_code=404, detail="文档不存在或无权访问")

    file_path = result.get("file_path", "")
    if not file_path or not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="文件不存在")

    return FileResponse(
        path=file_path,
        filename=result.get("original_file_name", "document.md"),
        media_type="text/markdown",
    )


@app.get("/stock_sse_summary")
async def stock_sse_summary():
    return get_stock_sse_summary()


@app.get("/stock_szse_summary")
async def stock_szse_summary(
        date: str = Query(..., description="日期，格式为 YYYMMDD")
):
    return get_stock_szse_summary(date=date)


@app.get("/stock_spot", summary="获取个股即时行情")
async def stock_spot(
        code: str = Query(..., description="6位股票代码，如 000001、600519")
):
    result = get_stock_spot(code=code)
    if result["status"] == "error":
        etype = result.get("error_type", "")
        if etype == "not_found":
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=result["message"])
        else:
            raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=result["message"])
    return result


# ==================== 接口 2：获取行业/概念板块排行 ====================
@app.get("/sector_ranking", summary="获取行业或概念板块排行榜")
async def sector_ranking(
        top: int = Query(10, description="返回前 N 个板块", ge=1, le=100),
        direction: str = Query(..., description="排序方向：up=涨幅榜, down=跌幅榜", regex="^(up|down)$"),
        sector_type: str = Query("industry", description="板块类型：industry=新浪行业, concept=新浪概念",
                                 regex="^(industry|concept)$")
):
    result = get_sector_ranking(top=top, direction=direction, sector_type=sector_type)
    if result["status"] == "error":
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=result["message"])
    return result



# ═══════════════════════════════════════════════════════════════════════
# 金融资讯 API — 宏观新闻 / 个股资讯 / 专业研报
# ═══════════════════════════════════════════════════════════════════════

# --- 宏观新闻 ---

@app.get("/macro_news/eastmoney", summary="宏观新闻-东方财富全球财经资讯")
async def macro_news_eastmoney(
        top: int = Query(20, description="返回条数", ge=1, le=200),
):
    """覆盖面广，实时性强，最多 200 条"""
    return get_macro_news_eastmoney(top=top)


@app.get("/macro_news/sina", summary="宏观新闻-新浪财经全球快讯")
async def macro_news_sina(
        top: int = Query(20, description="返回条数", ge=1, le=20),
):
    """快讯风格，简洁明了"""
    return get_macro_news_sina(top=top)


@app.get("/macro_news/ths", summary="宏观新闻-同花顺财经资讯")
async def macro_news_ths(
        top: int = Query(20, description="返回条数", ge=1, le=20),
):
    """深度报道，政策解读"""
    return get_macro_news_ths(top=top)


@app.get("/macro_news/digest", summary="宏观新闻-东方财富财经早餐")
async def macro_news_digest(
        top: int = Query(10, description="返回条数", ge=1, le=400),
):
    """每日盘前重要资讯汇总，含隔夜外盘"""
    return get_macro_news_digest(top=top)


@app.get("/macro_news/all", summary="宏观新闻-多源综合去重")
async def macro_news_all(
        top_per_source: int = Query(10, description="每个数据源取前N条", ge=1, le=50),
):
    """同时从东方财富/新浪/同花顺/盘前早餐获取，去重合并按时间降序"""
    return get_macro_news_all(top_per_source=top_per_source)


# --- 经济日历 ---

@app.get("/economic_calendar", summary="经济数据日历")
async def economic_calendar():
    """百度经济数据日历 — 含事件、预期值、前值、重要程度"""
    return get_economic_calendar()


# --- 个股资讯 ---

@app.get("/stock/news", summary="个股新闻")
async def stock_news(
        code: str = Query(..., description="6位股票代码，如 600519、000001"),
        top: int = Query(20, description="返回条数", ge=1, le=50),
):
    """东方财富个股新闻 — 含完整内容、来源、原文链接"""
    return get_stock_news(code=code, top=top)


@app.get("/stock/notices", summary="个股公告")
async def stock_notices(
        code: str = Query(..., description="6位股票代码，如 600519、000001"),
        top: int = Query(20, description="返回条数", ge=1, le=50),
):
    """上市公司公告 — 含定期报告、重大事项、分红等"""
    return get_stock_notices(code=code, top=top)


# --- 专业研报 ---

@app.get("/stock/research_report", summary="个股研报")
async def stock_research_report(
        code: str = Query(..., description="6位股票代码，如 600519、000001"),
        top: int = Query(20, description="返回篇数", ge=1, le=50),
):
    """
    东方财富个股研究报告 — 含评级、3年盈利预测(EPS+PE)、PDF下载链接
    返回字段示例:
      - 报告名称 / 东财评级 / 机构
      - 2026-盈利预测-每股收益 / 2026-盈利预测-市盈率
      - 2027-盈利预测-每股收益 / 2027-盈利预测-市盈率
      - 报告PDF链接（可直接下载）
    """
    return get_stock_research_report(code=code, top=top)


# --- 一站式综合查询 ---

@app.get("/stock/full_report", summary="个股一站式综合查询")
async def stock_full_report(
        code: str = Query(..., description="6位股票代码，如 600519、000001"),
        news_top: int = Query(10, description="新闻条数", ge=1, le=50),
        report_top: int = Query(10, description="研报篇数", ge=1, le=50),
):
    """
    一次调用获取：个股新闻 + 研报 + 公告
    返回结构:
      {
        "code": "股票代码",
        "fetch_time": "查询时间",
        "news": {...},
        "research_reports": {...},
        "notices": {...}
      }
    """
    return get_stock_full_report(code=code, news_top=news_top, report_top=report_top)


def main(host, port):
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main("0.0.0.0", 23456)