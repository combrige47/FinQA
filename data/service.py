import os
import sys
import argparse
from pathlib import Path
from typing import List, Optional

import uvicorn
from openai import OpenAI
from pymilvus import MilvusClient, AnnSearchRequest, WeightedRanker
from FlagEmbedding import BGEM3FlagModel
from fastapi import FastAPI, Query
from pydantic import BaseModel
from contextlib import asynccontextmanager

MILVUS_URI = "http://124.70.51.221:19530"
MODEL_PATH = "../model/bge-m3"
API_KEY = os.environ.get("DEEPSEEK_API_KEY")
DEFAULT_COLLECTION = "financial_chunk"
TOP_K = 5
CANDIDATE_K = 100
DENSE_WEIGHT = 0.3
class QueryRequest(BaseModel):
    query: str

client: Optional[MilvusClient] = None
model: Optional[BGEM3FlagModel] = None
active_collection: str = DEFAULT_COLLECTION

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

def search(query:str,top_k:int=TOP_K,dense_weight:float=DENSE_WEIGHT):
    res = []

    # 同时生成密集向量和稀疏权重
    output = model.encode(
        [query],
        batch_size=1,
        return_dense=True,
        return_sparse=True
    )
    dense_vec = output["dense_vecs"][0].tolist()
    sparse_vec = output["lexical_weights"][0]

    # 混合检索: dense + sparse
    dense_req = AnnSearchRequest(
        data=[dense_vec],
        anns_field="embedding",
        param={"metric_type": "COSINE", "params": {"ef": CANDIDATE_K}},
        limit=top_k
    )
    sparse_req = AnnSearchRequest(
        data=[sparse_vec],
        anns_field="sparse_embedding",
        param={"metric_type": "IP"},
        limit=top_k
    )
    ranker = WeightedRanker(dense_weight, 1 - dense_weight)

    results = client.hybrid_search(
        collection_name=DEFAULT_COLLECTION,
        reqs=[dense_req, sparse_req],
        ranker=ranker,
        limit=top_k,
        output_fields=["text", "chunk_id", "company_name", "doc_id", "report_year", "stock_code", "title"]
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

    return res


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
        top_k: int = TOP_K,
        dense_weight: float = Query(default=DENSE_WEIGHT,
                                    description="密集检索权重 (0~1)")
):
   return search(query,top_k, dense_weight)

@app.post("/query")
async def query(
        req: QueryRequest
):
    res = search(req.query, TOP_K, dense_weight=DENSE_WEIGHT)

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


def main(host,port):
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    main("127.0.0.1", 8000)