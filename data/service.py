import os
import sys
import traceback
from pathlib import Path
from typing import List, Optional

import uvicorn
from openai import OpenAI
from fastapi import FastAPI, Query, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager

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

def get_answer(prompt):
    client = OpenAI(
        base_url="https://api.deepseek.com",
        api_key=API_KEY,
    )

    response = client.chat.completions.create(
        model="deepseek-v4-flash",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        stream=False,
        extra_body={
            "enable_thinking": False
        }
    )

    return response.choices[0].message.content

def search(query: str, top_k: int = TOP_K):
    return importer.search(
        query=query,
        top_k=top_k,
        candidate_k=CANDIDATE_K,
    )


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
    print("=== 资源加载完成，服务就绪 ===")

    yield

    print("=== 服务关闭，释放资源 ===")
    importer.close()


app = FastAPI(title="FinQA Dense Retriever Service", lifespan=lifespan)

@app.get("/search")
async def search_(
        query: str = Query(...),
        top_k: int = TOP_K,
        kb_id: int | None = Query(default=None, description="知识库 ID，不传则使用默认知识库"),
):
    collection_name = kb_service.get_collection_name_by_kb_id(kb_id=kb_id)
    if kb_id is not None:
        return importer.search(
            query=query,
            top_k=top_k,
            candidate_k=CANDIDATE_K,
            collection_name=collection_name
        )
    else:
        return search(query, top_k)


@app.post("/query")
async def query(
        req: QueryRequest
):
    if req.kb_id is not None:
        res = importer.search(
            query=req.query,
            top_k=req.top_k,
            candidate_k=CANDIDATE_K,
            collection_name=f"kb_{req.kb_id}",
        )
    else:
        res = search(req.query, req.top_k)

    context = "\n\n".join(
        f"""【公司】{item['company_name']}
    【年份】{item['report_year']}
    【标题】{item['title']}
    【内容】
    {item['text_snippet']}"""
        for item in res
    )

    prompt = f"""请根据下面资料回答问题。

    {context}

    问题：
    {req.query}

    如果资料中没有答案，请明确说明没有找到，不要编造。
    """

    answer = get_answer(prompt)

    return {
        "question": req.query,
        "answer": answer,
        "references": res
    }


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



def main(host,port):
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    main("0.0.0.0", 23456)