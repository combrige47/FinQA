import sys
import argparse
from pathlib import Path
from typing import List, Optional

import uvicorn
from pymilvus import MilvusClient
from FlagEmbedding import BGEM3FlagModel
from fastapi import FastAPI, Query
from pydantic import BaseModel
from contextlib import asynccontextmanager

sys.path.insert(0, str(Path(__file__).parent.parent))

MILVUS_URI = "http://124.70.51.221:19530"
MODEL_PATH = str(Path(__file__).parent.parent / "model" / "bge-m3")
DEFAULT_COLLECTION = "financial_chunk"
TOP_K = 5
CANDIDATE_K = 100
DENSE_WEIGHT = 0.3

client: Optional[MilvusClient] = None
model: Optional[BGEM3FlagModel] = None
active_collection: str = DEFAULT_COLLECTION


class BatchQueryReq(BaseModel):
    queries: List[str]


@asynccontextmanager
async def lifespan(app: FastAPI):
    global client, model, qp, retriever, active_collection
    print("加载资源")
    print("正在连接Milvus")
    client = MilvusClient(uri=MILVUS_URI)
    colls = client.list_collections()
    if active_collection not in colls:
        raise RuntimeError(f"集合 {active_collection} 不存在")
    print("正在加载BGE-M3模型")
    model = BGEM3FlagModel(MODEL_PATH, use_fp16=False, device="cpu")
    print("=== 资源加载完成，服务就绪 ===")

    yield

    print("=== 服务关闭，释放资源 ===")
    if client:
        client.close()


app = FastAPI(title="FinQA Hybrid Retriever Service", lifespan=lifespan)


@app.get("/search")
async def search(
        query: str = Query(...),
        top_k: int = TOP_K
):
    res = []

    query_vector = model.encode(
        [query],
        batch_size=1
    )["dense_vecs"][0].tolist()

    results = client.search(
        collection_name=DEFAULT_COLLECTION,
        data=[query_vector],
        limit=top_k,
        output_fields=["text", "chunk_id", "company_name", "doc_id", "report_year", "stock_code", "title"],
        search_params={
            "metric_type": "COSINE",
            "params": {
                "ef": CANDIDATE_K
            }
        }
    )

    if results:
        for rank, hit in enumerate(results[0], 1):
            e = hit.get("entity", {})
            res.append({
                "rank": rank,
                "score": hit["distance"],
                "stock_code": e.get("stock_code"),
                "company_name": e.get("company_name"),
                "doc_id": e.get("doc_id"),
                "report_year": e.get("report_year"),
                "title": e.get("title"),
                "text_snippet": e.get("text", "")
            })

    return {
        "query": query,
        "results": res
    }

def main(host,port):
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    main("127.0.0.1", 8000)