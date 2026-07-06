from datetime import date
from pathlib import Path
from FlagEmbedding import BGEM3FlagModel
from pymilvus import MilvusClient, DataType
from transformers import AutoTokenizer

from parse.parse_md import MarkdownChunker

MILVUS_URI = "http://124.70.51.221:19530"
MODEL_PATH = str(Path(__file__).parent.parent / "model" / "bge-m3")
model = BGEM3FlagModel(MODEL_PATH, use_fp16=False)
COLLECTION_NAME = "financial_chunk"  # 统一使用的集合名称
TOP_K = 5
CANDIDATE_K = 100
DENSE_WEIGHT = 0.3

# 1. 初始化 MilvusClient
client = MilvusClient(uri=MILVUS_URI)

# 2. 检查集合是否存在，若不存在则创建
if not client.has_collection(collection_name=COLLECTION_NAME):
    # 创建 Schema
    schema = client.create_schema(
        description="Financial RAG",
        auto_id=False  # chunk_id 作为主键，不自动生成
    )

    # 添加字段定义
    schema.add_field(field_name="chunk_id", datatype=DataType.VARCHAR, max_length=64, is_primary=True)
    schema.add_field(field_name="doc_id", datatype=DataType.VARCHAR, max_length=64)
    schema.add_field(field_name="company_name", datatype=DataType.VARCHAR, max_length=128)
    schema.add_field(field_name="stock_code", datatype=DataType.VARCHAR, max_length=32)
    schema.add_field(field_name="report_type", datatype=DataType.VARCHAR, max_length=64)
    schema.add_field(field_name="report_year", datatype=DataType.INT32)
    schema.add_field(field_name="report_date", datatype=DataType.VARCHAR, max_length=20)
    schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=256)
    schema.add_field(field_name="title_path", datatype=DataType.VARCHAR, max_length=1000)
    schema.add_field(field_name="page_start", datatype=DataType.INT32)
    schema.add_field(field_name="page_end", datatype=DataType.INT32)
    schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)
    schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=1024)

    # 准备索引参数
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="embedding",
        metric_type="COSINE",
        index_type="HNSW",
        params={
            "M": 32,
            "efConstruction": 200
        }
    )

    # 一键创建集合（包含 Schema 和 索引）
    client.create_collection(
        collection_name=COLLECTION_NAME,
        schema=schema,
        index_params=index_params
    )

# 3. 处理文本数据
chunker = MarkdownChunker(model.tokenizer)
with open("../vector/2026-03-31_000488_ST晨鸣_2025年年度报告.md", encoding="utf-8") as f:
    md = f.read()

chunks = chunker.chunk(
    markdown=md,
    company_name="ST晨鸣",
    stock_code="000488",
    report_type="年度报告",
    report_year=2025,
    report_date=date(2026, 3, 31),
    doc_id="000001_2025_annual"
)

texts = [c.text for c in chunks]

# 4. 生成向量
vectors = model.encode(
    texts,
    batch_size=16
)["dense_vecs"]

# 5. 组装实体数据
entities = []
for chunk, vector in zip(chunks, vectors):
    entities.append({
        "chunk_id": chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "company_name": chunk.company_name,
        "stock_code": chunk.stock_code,
        "report_type": chunk.report_type,
        "report_year": chunk.report_year,
        "report_date": str(chunk.report_date),
        "title": chunk.title,
        "title_path": " > ".join(chunk.title_path),
        "page_start": chunk.page_start or -1,
        "page_end": chunk.page_end or -1,
        "text": chunk.text,
        "embedding": vector.tolist()
    })

# 6. 通过 client 插入数据（MilvusClient 内部会自动处理持久化，无需手动写 flush）
insert_result = client.insert(
    collection_name=COLLECTION_NAME,
    data=entities
)

print(f"数据插入成功！影响行数: {insert_result.get('insert_count')}")