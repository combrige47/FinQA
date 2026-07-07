from datetime import date
from pathlib import Path
from FlagEmbedding import BGEM3FlagModel
from pymilvus import MilvusClient, DataType
from transformers import AutoTokenizer

from parse.parse_md import MarkdownChunker

MILVUS_URI = "http://124.70.51.221:19530"
MODEL_PATH = str(Path(__file__).parent.parent / "model" / "bge-m3")
model = BGEM3FlagModel(MODEL_PATH, use_fp16=False)
COLLECTION_NAME = "financial_chunk"
TOP_K = 5
CANDIDATE_K = 100
DENSE_WEIGHT = 0.3


client = MilvusClient(uri=MILVUS_URI)


# 如果旧 collection 存在（无sparse字段），先删除再重建
if client.has_collection(collection_name=COLLECTION_NAME):
    print(f"删除旧 collection: {COLLECTION_NAME} (schema 变更，需要重建)")
    client.drop_collection(collection_name=COLLECTION_NAME)

if not client.has_collection(collection_name=COLLECTION_NAME):
    schema = client.create_schema(
        description="Financial RAG with hybrid search (dense + sparse)",
        auto_id=False
    )

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
    schema.add_field(field_name="sparse_embedding", datatype=DataType.SPARSE_FLOAT_VECTOR)

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

    # 稀疏向量索引 — BGE-M3 lexical weights
    index_params.add_index(
        field_name="sparse_embedding",
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="IP"
    )

    # 一键创建集合（包含 Schema 和 索引）
    client.create_collection(
        collection_name=COLLECTION_NAME,
        schema=schema,
        index_params=index_params
    )
chunker = MarkdownChunker(model.tokenizer)
def insert(file_path:str):
    with open(f"{file_path}", encoding="utf-8") as f:
        md = f.read()
    path_obj=Path(file_path)
    file_name = path_obj.stem
    list = file_name.split("_")
    chunks = chunker.chunk(
        markdown=md,
        company_name=list[2],
        stock_code=list[1],
        report_type=list[3],
        report_year=2025,
        report_date=date.fromisoformat(list[0]),
        doc_id=file_name,
    )

    texts = [c.text for c in chunks]
    output = model.encode(
        texts,
        batch_size=16,
        return_dense=True,
        return_sparse=True
    )
    vectors = output["dense_vecs"]
    sparse_weights_list = output["lexical_weights"]  # List[Dict[int, float]]
    entities = []
    for chunk, vector, sparse_weights in zip(chunks, vectors, sparse_weights_list):
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
            "embedding": vector.tolist(),
            "sparse_embedding": sparse_weights
        })
    insert_result = client.insert(
        collection_name=COLLECTION_NAME,
        data=entities
    )
    print(f"数据插入成功！影响行数: {insert_result.get('insert_count')}")
def batch_insert(input_dir:str):
    input_dir_obj = Path(input_dir)
    md_files = list(input_dir_obj.glob("*.md"))
    for idx,md_file in enumerate(md_files,1):
        print(f"当前正在处理第{idx}个文档")
        insert(md_file)

batch_insert(input_dir="../parse/test_output")