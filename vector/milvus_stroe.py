"""Milvus 知识库导入器 — 模块化、可复用的金融报告批量导入工具。

用法:
    # CLI
    python FinQA/vector/milvus_stroe.py [input_dir] [--retry-from LOG] [--sync-checkpoint] [--no-skip]

    # 外部调用
    from FinQA.vector.milvus_stroe import MilvusImporter
    importer = MilvusImporter()
    importer.ensure_collection()
    result = importer.batch_insert("open_output")
"""

import json
import logging
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

from pymilvus import MilvusClient, DataType

from parse.parse_md import MarkdownChunker
from parse.parse_name import extract_report_meta
from vector.embedding_client import EmbeddingClient

# ── 默认配置 ──
DEFAULT_MILVUS_URI = "http://124.70.51.221:19530"
DEFAULT_COLLECTION_NAME = "financial_chunk"
DEFAULT_INPUT_DIR = "open_output"
DEFAULT_EMBEDDING_MODEL = "text-embedding-v4"


@dataclass
class InsertResult:
    """单文件插入结果"""
    doc_id: str
    chunk_count: int
    company_name: str
    stock_code: str


@dataclass
class BatchResult:
    """批量导入结果"""
    total_files: int = 0
    success: int = 0
    skipped: int = 0
    failed: int = 0
    total_chunks: int = 0
    failed_entries: list[dict] = field(default_factory=list)
    failed_log_path: str = ""
    checkpoint_path: str = ""

    @property
    def processed(self) -> int:
        return self.success + self.failed


# ═══════════════════════════════════════════════════════════════════════
# MilvusImporter
# ═══════════════════════════════════════════════════════════════════════

class MilvusImporter:
    """Milvus 导入器：管理连接、模型、分块、嵌入、批量导入的完整生命周期。

    使用示例:
        importer = MilvusImporter()
        importer.ensure_collection()
        result = importer.batch_insert("open_output")
        print(f"成功 {result.success}, 失败 {result.failed}")
        importer.close()
    """

    def __init__(
        self,
        milvus_uri: str = DEFAULT_MILVUS_URI,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        api_key: str | None = None,
        **chunk_kwargs,
    ):
        """
        Parameters
        ----------
        milvus_uri : str
            Milvus 服务地址。
        collection_name : str
            Milvus collection 名称。
        embedding_model : str
            千问 embedding 模型名称。
        api_key : str | None
            DashScope API Key。默认从环境变量 DASHSCOPE_API_KEY 读取。
        **chunk_kwargs :
            传递给 MarkdownChunker 的参数 (max_tokens, overlap, min_tokens 等)。
        """
        self.milvus_uri = milvus_uri
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.api_key = api_key
        self._chunk_kwargs = chunk_kwargs

        # 延迟初始化（构造时不加载模型）
        self._client: MilvusClient | None = None
        self._model: EmbeddingClient | None = None
        self._chunker: MarkdownChunker | None = None

    # ── 属性（懒加载） ──

    @property
    def client(self) -> MilvusClient:
        if self._client is None:
            self._client = MilvusClient(uri=self.milvus_uri)
        return self._client

    @property
    def model(self) -> EmbeddingClient:
        if self._model is None:
            print(f"初始化千问 Embedding 客户端: {self.embedding_model} ...")
            self._model = EmbeddingClient(
                api_key=self.api_key,
                model=self.embedding_model,
            )
        return self._model

    @property
    def chunker(self) -> MarkdownChunker:
        if self._chunker is None:
            self._chunker = MarkdownChunker(self.model.tokenizer, **self._chunk_kwargs)
        return self._chunker

    # ── Collection 管理 ──

    def ensure_collection(self, force_recreate: bool = False):
        """确保 collection 存在，必要时创建或重建。

        Parameters
        ----------
        force_recreate : bool
            为 True 时先删除旧 collection 再重建（schema 迁移时使用）。
        """
        if force_recreate and self.client.has_collection(collection_name=self.collection_name):
            print(f"删除旧 collection: {self.collection_name} (force_recreate=True)")
            self.client.drop_collection(collection_name=self.collection_name)

        if self.client.has_collection(collection_name=self.collection_name):
            print(f"Collection 已存在: {self.collection_name}")
            return

        # 创建新 collection（纯 dense 检索，无稀疏向量）
        schema = self.client.create_schema(
            description="Financial RAG with dense-only search (text-embedding-v4)",
            auto_id=False,
        )
        schema.add_field(field_name="chunk_id", datatype=DataType.VARCHAR, max_length=64, is_primary=True)
        schema.add_field(field_name="doc_id", datatype=DataType.VARCHAR, max_length=256)
        schema.add_field(field_name="company_name", datatype=DataType.VARCHAR, max_length=128)
        schema.add_field(field_name="stock_code", datatype=DataType.VARCHAR, max_length=32)
        schema.add_field(field_name="report_type", datatype=DataType.VARCHAR, max_length=64)
        schema.add_field(field_name="report_year", datatype=DataType.INT32)
        schema.add_field(field_name="report_date", datatype=DataType.VARCHAR, max_length=20)
        schema.add_field(field_name="created_date", datatype=DataType.VARCHAR, max_length=20)
        schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=1024)
        schema.add_field(field_name="title_path", datatype=DataType.VARCHAR, max_length=2048)
        schema.add_field(field_name="page_start", datatype=DataType.INT32)
        schema.add_field(field_name="page_end", datatype=DataType.INT32)
        schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=1024)

        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            metric_type="COSINE",
            index_type="HNSW",
            params={"M": 32, "efConstruction": 200},
        )

        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params,
        )
        print(f"Collection 已创建: {self.collection_name}")

    def drop_collection(self):
        """删除 collection（危险操作）。"""
        if self.client.has_collection(collection_name=self.collection_name):
            self.client.drop_collection(collection_name=self.collection_name)
            print(f"Collection 已删除: {self.collection_name}")

    # ── 文件预处理（CPU：读 + 元数据 + 分块，不含嵌入） ──

    def _prepare_file(self, file_path: str) -> tuple[str, dict, list]:
        """读取文件并分块（纯 CPU 工作，不含 GPU 嵌入）。

        Returns:
            (doc_id, meta_dict, chunks_list)
        """
        path_obj = Path(file_path)
        if not path_obj.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        file_name = path_obj.stem

        with open(file_path, encoding="utf-8") as f:
            md = f.read()

        # 元数据提取
        meta = extract_report_meta(file_name)
        if meta is None:
            parts = file_name.split("_")
            company_name = parts[2] if len(parts) > 2 else "未知公司"
            stock_code = parts[1] if len(parts) > 1 else "000000"
            report_type = parts[3] if len(parts) > 3 else "其他公告"
            report_year = 2024
            report_date = date.fromisoformat(parts[0]) if len(parts) > 0 else date.today()
        else:
            company_name = meta.get("company_name", "未知公司")
            stock_code = meta.get("stock_code", "000000")
            report_type = meta.get("report_type", "其他公告")
            report_year = int(meta.get("report_year", 2024) or 2024)
            report_date_str = meta.get("report_date", str(date.today()))
            try:
                report_date = date.fromisoformat(report_date_str)
            except (ValueError, TypeError):
                report_date = date.today()

        doc_id = file_name

        # 分块（CPU）
        chunks = self.chunker.chunk(
            markdown=md,
            company_name=company_name,
            stock_code=stock_code,
            report_type=report_type,
            report_year=report_year,
            report_date=report_date,
            doc_id=doc_id,
        )

        # token 安全截断：chunk.text 已含【章节】前缀（chunker 已 enrich）
        for chunk in chunks:
            tl = len(self.model.tokenizer.encode(chunk.text))
            if tl > self.TOKEN_SAFE_LIMIT:
                chunk.text = self.model.safe_truncate_text(
                    chunk.text, self.TOKEN_SAFE_LIMIT)

        meta_dict = {
            "company_name": company_name,
            "stock_code": stock_code,
            "report_type": report_type,
            "report_year": report_year,
            "report_date": str(report_date),
        }
        return doc_id, meta_dict, chunks

    # ── 截断工具 ──

    TOKEN_SAFE_LIMIT = 7500

    # ── 批量编码 + 插入（GPU） ──

    def _encode_and_insert_batch(
        self,
        pending: list[tuple[str, dict, list]],
        *,
        encode_batch_size: int = 10,
    ) -> list[InsertResult]:
        """将多个文件的 chunks 合并，一次性 API 编码，然后按文件分别插入 Milvus。

        Parameters
        ----------
        pending : list of (doc_id, meta_dict, chunks)
        encode_batch_size : int
            API 编码的内部 batch size（text-embedding-v4 上限为 10）。

        Returns
        -------
        list of InsertResult — 每个文件一个结果。
        """
        # 合并所有文件的 chunks
        all_chunks = []
        split_points = []  # 记录每个文件的 chunks 范围
        offset = 0
        for doc_id, meta, chunks in pending:
            all_chunks.extend(chunks)
            offset += len(chunks)
            split_points.append((doc_id, meta, offset))

        if not all_chunks:
            return []

        # 一次性 API 编码（token 截断已在 _prepare_file 中完成）
        texts = [c.text for c in all_chunks]
        output = self.model.encode(
            texts,
            batch_size=encode_batch_size,
            return_dense=True,
        )
        vectors = output["dense_vecs"]

        # 按文件拆分，构建实体并插入
        created_date_str = date.today().strftime("%Y-%m-%d")
        results = []
        start = 0
        for doc_id, meta, end in split_points:
            file_chunks = all_chunks[start:end]
            file_vectors = vectors[start:end]
            start = end

            entities = []
            for chunk, vec in zip(file_chunks, file_vectors):
                entities.append({
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "company_name": chunk.company_name,
                    "stock_code": chunk.stock_code,
                    "report_type": chunk.report_type,
                    "report_year": chunk.report_year,
                    "report_date": str(chunk.report_date),
                    "created_date": created_date_str,
                    "title": (chunk.title or " > ".join(chunk.title_path))[:1020],
                    "title_path": (" > ".join(chunk.title_path))[:2040],
                    "page_start": chunk.page_start or -1,
                    "page_end": chunk.page_end or -1,
                    "text": chunk.text,
                    "embedding": vec.tolist(),
                })

            # 幂等插入：先删旧数据（失败不阻塞），再插入（带重试）
            try:
                self._retry_milvus(
                    self.client.delete,
                    collection_name=self.collection_name,
                    filter=f'doc_id == "{doc_id}"',
                    max_retries=1,
                )
            except Exception:
                pass  # delete 失败不阻塞（首次插入、collection未加载等）

            try:
                insert_result = self._retry_milvus(
                    self.client.insert,
                    collection_name=self.collection_name,
                    data=entities,
                )
                chunk_count = insert_result.get("insert_count", len(entities))
                results.append(InsertResult(
                    doc_id=doc_id,
                    chunk_count=chunk_count,
                    company_name=meta["company_name"],
                    stock_code=meta["stock_code"],
                ))
            except Exception as e:
                # 单文件失败不影响同批其他文件
                results.append((doc_id, meta, e))

        return results

    # ── 批量 flush ──

    def _flush_pending(
        self,
        pending: list,
        pending_files: list,
        uploaded: set,
        cp_path: str,
        skip_uploaded: bool,
        encode_batch_size: int,
        failed_entries: list,
        pbar,
    ) -> tuple[int, int]:
        """将累积的文件批量编码并插入，返回 (total_chunks, success_count)。"""
        # GPU 编码阶段（整体操作，失败则整批丢弃，由生产者重新准备）
        try:
            results = self._encode_and_insert_batch(
                pending, encode_batch_size=encode_batch_size,
            )
        except Exception as e:
            # GPU 批量编码失败 → 逐文件回退（不丢任何文件）
            print(f"[回退] 批量GPU失败，逐文件处理 {len(pending_files)} 个文件: {type(e).__name__}")
            batch_chunks = 0
            batch_success = 0
            for pf, (doc_id, meta, chunks) in zip(pending_files, pending):
                try:
                    r2 = self._encode_and_insert_batch(
                        [(doc_id, meta, chunks)], encode_batch_size=16)
                    if r2 and isinstance(r2[0], InsertResult):
                        batch_chunks += r2[0].chunk_count
                        batch_success += 1
                        if skip_uploaded:
                            uploaded.add(r2[0].doc_id)
                    else:
                        raise r2[0][2] if isinstance(r2[0], tuple) else RuntimeError("unknown")
                except Exception as e2:
                    entry = {
                        "file": pf.name, "file_path": str(pf),
                        "error_type": type(e2).__name__,
                        "error_message": str(e2)[:500],
                        "traceback": traceback.format_exc()[:2000],
                        "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
                        "file_size": pf.stat().st_size,
                        "stock_code": meta.get("stock_code", "?"),
                        "company_name": meta.get("company_name", "?"),
                    }
                    failed_entries.append(entry)
            return batch_chunks, batch_success

        # 逐文件处理结果（每个文件独立成功/失败）
        batch_chunks = 0
        batch_success = 0
        idx_map = {doc_id: i for i, (doc_id, _, _) in enumerate(pending)}

        for item in results:
            if isinstance(item, InsertResult):
                batch_chunks += item.chunk_count
                batch_success += 1
                if skip_uploaded:
                    uploaded.add(item.doc_id)
            else:
                # (doc_id, meta, exception) — 单文件插入失败
                doc_id, meta, exc = item
                pf = pending_files[idx_map.get(doc_id, 0)] if doc_id in idx_map else pending_files[0]
                entry = {
                    "file": pf.name, "file_path": str(pf),
                    "error_type": type(exc).__name__,
                    "error_message": str(exc)[:500],
                    "traceback": traceback.format_exc()[:2000],
                    "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
                    "file_size": pf.stat().st_size,
                    "stock_code": meta.get("stock_code", "?"),
                    "company_name": meta.get("company_name", "?"),
                }
                failed_entries.append(entry)
                print(f"  [失败-插入] {pf.name}: {type(exc).__name__}: {str(exc)[:100]}")

        pbar.set_postfix_str(
            f"OK {pending_files[-1].name[:30]} [+{batch_success-1}]"
        )
        return batch_chunks, batch_success

    # ── 单文件插入（兼容接口） ──

    def insert_one(self, file_path: str) -> InsertResult:
        """处理单个 md 文件：读取 → 元数据提取 → 分块 → 嵌入 → 插入。

        对于批量导入，推荐使用 batch_insert() 以获得更好的 API 利用率。
        """
        doc_id, meta, chunks = self._prepare_file(file_path)
        results = self._encode_and_insert_batch(
            [(doc_id, meta, chunks)],
            encode_batch_size=10,
        )
        return results[0]

    # ── 从 Milvus 同步 ──

    def get_uploaded_doc_ids(self) -> set[str]:
        """从 Milvus 查询所有已上传的 doc_id。对大 collection 可能较慢。"""
        uploaded = set()
        offset = 0
        batch_size = 10000

        while True:
            try:
                results = self.client.query(
                    collection_name=self.collection_name,
                    filter="chunk_id != ''",
                    output_fields=["doc_id"],
                    limit=batch_size,
                    offset=offset,
                )
            except Exception as e:
                print(f"[sync] Milvus 查询失败: {e}")
                break

            if not results:
                break

            for r in results:
                uploaded.add(r["doc_id"])

            if len(results) < batch_size:
                break
            offset += batch_size

        print(f"[sync] Milvus 中现有 {len(uploaded)} 个唯一 doc_id")
        return uploaded

    # ── 混合检索 ──

    def search(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: int = 100,
        collection_name: str | None = None,
        output_fields: list[str] | None = None,
    ) -> list[dict]:
        """Dense 检索（千问 text-embedding-v4），用于 RAG 查询。

        Parameters
        ----------
        query : str
            查询文本。
        top_k : int
            返回结果数量。
        candidate_k : int
            HNSW 检索的 ef 参数（候选池大小）。
        collection_name : str | None
            目标 collection，默认使用 self.collection_name。
        output_fields : list[str] | None
            需要返回的标量字段列表。

        Returns
        -------
        list[dict]
            每个 dict 包含 rank, score, doc_id, title, text_snippet。
        """
        if collection_name is None:
            collection_name = self.collection_name
        if output_fields is None:
            output_fields = [
                "text", "chunk_id", "doc_id", "title",
            ]

        # 编码查询
        output = self.model.encode([query])
        dense_vec = output["dense_vecs"][0].tolist()

        results = self._retry_milvus(
            self.client.search,
            collection_name=collection_name,
            data=[dense_vec],
            anns_field="embedding",
            search_params={"metric_type": "COSINE", "params": {"ef": candidate_k}},
            limit=top_k,
            output_fields=output_fields,
        )

        res = []
        if results:
            for rank, hit in enumerate(results[0], 1):
                e = hit.get("entity", {})
                res.append({
                    "rank": rank,
                    "score": hit["distance"],
                    "doc_id": e.get("doc_id"),
                    "title": e.get("title"),
                    "text_snippet": e.get("text", ""),
                })

        return res

    # ── 批量导入 ──

    def batch_insert(
        self,
        input_dir: str = DEFAULT_INPUT_DIR,
        *,
        skip_uploaded: bool = True,
        checkpoint_file: str | None = None,
        retry_from_log: str | None = None,
        sync_checkpoint: bool = False,
        accumulate_files: int = 5,
        encode_batch_size: int = 10,
    ) -> BatchResult:
        """批量导入 md 文件到 Milvus。

        采用批量累积策略：先对多个文件做 CPU 分块，然后一次性 API 编码，
        大幅提升 API 利用率和整体吞吐量。

        Parameters
        ----------
        input_dir : str
            包含 .md 文件的目录路径。
        skip_uploaded : bool
            是否跳过 checkpoint 中已记录的文件。
        checkpoint_file : str | None
            checkpoint 文件路径。None 时自动使用 <input_dir>/uploaded_docs.json。
        retry_from_log : str | None
            失败日志路径 (JSONL)。指定时仅重试该日志中记录的文件。
        sync_checkpoint : bool
            是否从 Milvus 查询来校准本地 checkpoint。
        accumulate_files : int
            累积多少个文件后做一次 API 编码（默认 5）。
        encode_batch_size : int
            API 编码的内部 batch size（默认 10，text-embedding-v4 上限）。

        Returns
        -------
        BatchResult
        """
        from tqdm import tqdm

        input_dir_obj = Path(input_dir)
        md_files = sorted(input_dir_obj.glob("*.md"))
        total = len(md_files)

        if total == 0:
            print(f"[警告] {input_dir} 中没有 .md 文件")
            return BatchResult()

        # ── checkpoint 路径 ──
        cp_path = checkpoint_file or str(input_dir_obj / "uploaded_docs.json")

        # ── 重试模式 ──
        if retry_from_log:
            retry_path = Path(retry_from_log)
            if not retry_path.exists():
                print(f"[错误] 失败日志不存在: {retry_from_log}")
                return BatchResult()
            with open(retry_path, encoding="utf-8") as f:
                failed_entries = [json.loads(line) for line in f if line.strip()]
            retry_names = {entry["file"].replace(".md", "") for entry in failed_entries}
            md_files = [f for f in md_files if f.stem in retry_names]
            print(f"[重试] 从 {retry_path.name} 加载 {len(failed_entries)} 条记录, "
                  f"匹配到 {len(md_files)} 个文件")

        # ── 顺序批量累积：CPU 分块攒够一批 → GPU 编码+插入 → 下一批 ──
        failed_entries: list[dict] = []
        total_chunks = 0
        success = 0
        pending: list[tuple[str, dict, list]] = []
        pending_files: list[Path] = []

        pbar = tqdm(md_files, desc="导入进度", unit="file",
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} "
                               "[{elapsed}<{remaining}, {rate_fmt}]")

        for md_file in pbar:
            fname = md_file.name

            # CPU 分块
            try:
                doc_id, meta, chunks = self._prepare_file(str(md_file))
                pending.append((doc_id, meta, chunks))
                pending_files.append(md_file)
            except Exception as e:
                err_type = type(e).__name__
                err_msg = str(e)
                try:
                    file_meta = extract_report_meta(md_file.stem)
                    stock = file_meta.get("stock_code", "?") if file_meta else "?"
                    company = file_meta.get("company_name", "?") if file_meta else "?"
                except Exception:
                    stock, company = "?", "?"
                entry = {
                    "file": fname, "file_path": str(md_file),
                    "error_type": err_type, "error_message": err_msg[:500],
                    "traceback": traceback.format_exc()[:2000],
                    "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
                    "file_size": md_file.stat().st_size,
                    "stock_code": stock, "company_name": company,
                }
                failed_entries.append(entry)
                print(f"  [失败-分块] {fname}: {err_type}: {err_msg[:100]}")
                continue

            # 攒够一批：GPU 编码 + 插入
            if len(pending) >= accumulate_files:
                b_chunks, b_success = self._flush_pending(
                    pending, pending_files, cp_path,
                    skip_uploaded, encode_batch_size, failed_entries, pbar,
                )
                total_chunks += b_chunks
                success += b_success
                pending.clear()
                pending_files.clear()
                time.sleep(1)  # 给 Milvus 时间 flush，减轻内存压力

        # 尾批
        if pending:
            b_chunks, b_success = self._flush_pending(
                pending, pending_files, cp_path,
                skip_uploaded, encode_batch_size, failed_entries, pbar,
            )
            total_chunks += b_chunks
            success += b_success
            time.sleep(1)

        pbar.close()

        return BatchResult(
            total_files=total,
            success=success,
            failed=len(failed_entries),
            total_chunks=total_chunks,
            failed_entries=failed_entries,
            checkpoint_path=cp_path,
        )

    def _ensure_client(self):
        """确保 Milvus 连接有效，断开时自动重连。"""
        if self._client is None:
            self._client = MilvusClient(uri=self.milvus_uri)
            return
        try:
            # 轻量探测
            self._client.has_collection(collection_name=self.collection_name)
        except Exception:
            print("[重连] Milvus 连接断开，重建中...")
            try:
                self._client.close()
            except Exception:
                pass
            self._client = MilvusClient(uri=self.milvus_uri)
            time.sleep(1)

    def _retry_milvus(self, op, *args, max_retries=3, **kwargs):
        """带指数退避的 Milvus 操作重试。"""
        last_err = None
        for attempt in range(max_retries):
            try:
                return op(*args, **kwargs)
            except Exception as e:
                last_err = e
                msg = str(e).lower()
                if "closed" in msg or "channel" in msg or "unavailable" in msg:
                    self._ensure_client()
                if attempt < max_retries - 1:
                    wait = 2 ** attempt
                    time.sleep(wait)
        raise last_err

    def close(self):
        """释放资源。"""
        if self._client:
            try:
                self._client.close()
            except Exception:
                pass
        self._client = None
        self._model = None
        self._chunker = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def main(argv: list[str] | None = None):
    import argparse

    parser = argparse.ArgumentParser(
        description="批量导入 md 文件到 Milvus 知识库",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""示例:
  python FinQA/vector/milvus_stroe.py                          # 默认导入 open_output/
  python FinQA/vector/milvus_stroe.py my_dir                   # 指定目录
  python FinQA/vector/milvus_stroe.py --retry-from failed.jsonl # 重试失败文件
  python FinQA/vector/milvus_stroe.py --sync-checkpoint         # 从 Milvus 校准
  python FinQA/vector/milvus_stroe.py --no-skip                 # 强制全量重新导入
  python FinQA/vector/milvus_stroe.py --force-recreate          # 重建 collection""",
    )
    parser.add_argument(
        "input_dir", nargs="?", default=DEFAULT_INPUT_DIR,
        help=f"md 文件目录 (默认: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--retry-from", dest="retry_from_log", default=None,
        help="从失败日志 JSONL 重试",
    )
    parser.add_argument(
        "--sync-checkpoint", action="store_true",
        help="从 Milvus 查询校准本地 checkpoint",
    )
    parser.add_argument(
        "--no-skip", action="store_true",
        help="不跳过已上传文件（强制重新处理全部）",
    )
    parser.add_argument(
        "--force-recreate", action="store_true",
        help="删除并重建 collection（schema 迁移时使用）",
    )
    parser.add_argument(
        "--uri", default=DEFAULT_MILVUS_URI,
        help=f"Milvus 服务地址 (默认: {DEFAULT_MILVUS_URI})",
    )
    parser.add_argument(
        "--collection", default=DEFAULT_COLLECTION_NAME,
        help=f"Collection 名称 (默认: {DEFAULT_COLLECTION_NAME})",
    )

    args = parser.parse_args(argv)

    importer = MilvusImporter(
        milvus_uri=args.uri,
        collection_name=args.collection,
    )
    importer.ensure_collection(force_recreate=args.force_recreate)
    result = importer.batch_insert(
        input_dir=args.input_dir,
        skip_uploaded=not args.no_skip,
        retry_from_log=args.retry_from_log,
        sync_checkpoint=args.sync_checkpoint,
    )

    importer.close()
    return result


if __name__ == "__main__":
    main()
