import os
import sys
import traceback
from pathlib import Path
from typing import List, Optional
import re
import json
import uvicorn
from openai import OpenAI
import requests
from fastapi import FastAPI, Query, File, UploadFile, HTTPException,APIRouter, status
from fastapi.responses import FileResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
from schedule import start_scheduler, scheduler

import akshare as ak
import numpy as np

# 确保项目根目录在 sys.path 中，以便导入 sibling package
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Connection": "close"
}

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
async def stock_sse_summary(
):
    stock_sse_summary_df = ak.stock_sse_summary()
    return stock_sse_summary_df.to_markdown(index=False)

@app.get("/stock_szse_summary")
async def stock_szse_summary(
    date : str = Query(..., description="日期，格式为 YYYMMDD")
):
    stock_szse_summary_df = ak.stock_szse_summary(date=date)
    return stock_szse_summary_df.to_markdown(index=False)
@app.get("/stock_spot", summary="获取个股即时行情")
async def stock_spot(
    code: str = Query(..., description="6位股票代码，如 000001、600519")
):
    # 确定沪深、北京市场前缀
    if code.startswith(("600", "601", "603", "605", "688", "900")):
        prefix = "sh"
    elif code.startswith(("8", "4", "920")):
        prefix = "bj"
    else:
        prefix = "sz"
        
    url = f"http://qt.gtimg.cn/q={prefix}{code}"

    try:
        response = requests.get(url, headers=HEADERS, timeout=5.0)
        response.raise_for_status()
        response.encoding = "gbk"  # 腾讯接口固定使用 GBK 编码
        text = response.text
    except requests.RequestException as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to fetch data from upstream server: {str(exc)}"
        )

    # 校验返回数据是否有效
    if "pv_none_" in text or len(text) < 50:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Stock code '{code}' not found or invalid."
        )

    try:
        # 解析腾讯返回的特有格式：v_sz000001="51~平安银行~000001~12.34~..."
        data_str = text.split("=")[1].replace('"', '').replace(';\n', '')
        parts = data_str.split("~")
        
        return {
            "status": "success",
            "metadata": {
                "code": code,
                "name": parts[1].strip(),
                "market": prefix.upper(),
                "timestamp": parts[30]  # 格式如: "20260716150000"
            },
            "quotes": {
                "current_price": float(parts[3]),
                "prev_close": float(parts[4]),
                "open": float(parts[5]),
                "volume_hand": int(parts[6]),     # 成交量（手）
                "outer_volume": int(parts[7]),    # 外盘
                "inner_volume": int(parts[8]),    # 内盘
                "bid_1": {
                    "price": float(parts[9]),
                    "volume": int(parts[10])
                },
                "ask_1": {
                    "price": float(parts[19]),
                    "volume": int(parts[20])
                }
            }
        }

    except (IndexError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Data parsing error. The upstream data structure might have changed: {str(exc)}"
        )


# ==================== 接口 2：获取行业/概念板块排行 ====================
@app.get("/sector_ranking", summary="获取行业或概念板块排行榜")
async def sector_ranking(
    top: int = Query(10, description="返回前 N 个板块", ge=1, le=100),
    direction: str = Query(..., description="排序方向：up=涨幅榜, down=跌幅榜", regex="^(up|down)$"),
    sector_type: str = Query("industry", description="板块类型：industry=新浪行业, concept=新浪概念", regex="^(industry|concept)$")
):
    if sector_type == "industry":
        url = "http://vip.stock.finance.sina.com.cn/q/view/newSinaHy.php"
    else:
        url = "http://vip.stock.finance.sina.com.cn/q/view/newSinaConcept.php"

    raw_text = ""
    # 自动重试机制：最多重试 3 次以对抗 'Remote end closed connection' 
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=HEADERS, timeout=6.0)
            response.raise_for_status()
            response.encoding = "gbk"  # 新浪接口固定使用 GBK 编码
            raw_text = response.text
            if raw_text:
                break
        except requests.RequestException as exc:
            if attempt == max_retries - 1:
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail=f"Failed to fetch sector data after {max_retries} attempts: {str(exc)}"
                )

    try:
        # 使用正则表达式精准剥离出 JSON 字符串
        json_match = re.search(r"\{.*\}", raw_text)
        if not json_match:
            raise ValueError("No valid JSON structure found in upstream response.")
            
        raw_data = json.loads(json_match.group())
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Error parsing raw text to JSON from upstream: {str(exc)}"
        )

    try:
        ranking_list = []
        for code, info_str in raw_data.items():
            # 统一中英文逗号进行安全分割
            normalized_str = info_str.replace("，", ",")
            parts = [p.strip() for p in normalized_str.split(",")]
            
            # 标准行拆分出来应不小于 13 个元素
            if len(parts) < 13:
                continue

            try:
                # 提取清洗后的纯净中英文数据
                raw_name = parts[1].strip()
                # 剔除板块名中可能残留的代码前缀
                sector_name = raw_name.replace(code, "").strip()

                ranking_list.append({
                    "sector_code": code.strip(),
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
                    "top_gainer_name": parts[12].strip()
                })
            except (ValueError, IndexError):
                continue

        # 根据涨跌幅排序（up=降序, down=升序）
        reverse_sort = (direction == "up")
        sorted_list = sorted(ranking_list, key=lambda x: x["change_percent"], reverse=reverse_sort)
        final_list = sorted_list[:top]

        return {
            "status": "success",
            "direction": direction,
            "sector_type": sector_type,
            "count": len(final_list),
            "data": final_list
        }

    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error occurred while sorting or filtering sector data: {str(exc)}"
        )
def main(host,port):
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    main("0.0.0.0", 23456)