"""
Knowledge Base Service

提供用户知识库 CRUD 和文件上传处理管道。
使用已有 fp_knowledge_base / fp_document 表（对齐 fp_user 权限体系），
用户 KB 使用简化 Milvus collection schema（不含金融报告专属字段）。

用法:
    from data.kb_service import create_kb, list_kbs, upload_to_kb, delete_kb
"""

import os
import uuid
import time
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Optional

from pymilvus import DataType

from data.mysqlClient import MySQLClient
from parse.pdf2md import Pdf2Md

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
UPLOADS_DIR = _PROJECT_ROOT / "uploads"

ALLOWED_EXTENSIONS = {".pdf", ".md", ".txt"}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

# ═══════════════════════════════════════════════════════════════════════
# MySQL 连接辅助
# ═══════════════════════════════════════════════════════════════════════

def _get_db() -> MySQLClient:
    """创建新的 MySQL 连接。"""
    return MySQLClient()


# ═══════════════════════════════════════════════════════════════════════
# KB CRUD（使用 fp_knowledge_base 表）
# ═══════════════════════════════════════════════════════════════════════

def create_kb(
    creator_id: int,
    name: str,
    description: str,
    importer,  # MilvusImporter
) -> dict:
    """创建知识库：写 fp_knowledge_base + 创建简化 Milvus collection。

    Parameters
    ----------
    creator_id : int
        创建者 ID（对应 fp_user.id）。
    name : str
        知识库名称（同一用户下唯一）。
    description : str
        描述。
    importer : MilvusImporter
        共享的 Milvus 导入器实例。

    Returns
    -------
    dict
        {kb_id, name, description, milvus_collection_name, document_count, ...}
    """
    db = _get_db()
    try:
        # 检查同名 KB 是否已存在
        db.execute(
            "SELECT id FROM fp_knowledge_base WHERE creator_id = %s AND name = %s",
            (creator_id, name),
        )
        if db.fetchone():
            raise DuplicateError(f"知识库 '{name}' 已存在")

        # 插入记录（milvus_collection_name 先占位，后面 UPDATE）
        db.execute(
            """INSERT INTO fp_knowledge_base
               (chunk_count, description, document_count, enabled,
                milvus_collection_name, name, type, creator_id)
               VALUES (0, %s, 0, 1, '', %s, 'user', %s)""",
            (description, name, creator_id),
        )
        db.commit()
        kb_id = db.cursor.lastrowid

        milvus_collection = _get_db().get_collection_name_by_kb_id(kb_id)

        # 更新 milvus_collection_name
        db.execute(
            "UPDATE fp_knowledge_base SET milvus_collection_name = %s WHERE id = %s",
            (milvus_collection, kb_id),
        )
        db.commit()

        # 创建简化 Milvus collection
        _ensure_user_kb_collection(kb_id, importer)

        # 查询并返回
        db.execute(
            """SELECT id, chunk_count, created_at, description, document_count,
                      enabled, milvus_collection_name, name, type, updated_at, creator_id
               FROM fp_knowledge_base WHERE id = %s""",
            (kb_id,),
        )
        row = db.fetchone()
        return _row_to_kb_dict(row)

    finally:
        db.close()


def list_kbs(creator_id: int) -> list[dict]:
    """列出用户的所有知识库。"""
    db = _get_db()
    try:
        db.execute(
            """SELECT id, chunk_count, created_at, description, document_count,
                      enabled, milvus_collection_name, name, type, updated_at, creator_id
               FROM fp_knowledge_base
               WHERE creator_id = %s AND enabled = 1
               ORDER BY created_at DESC""",
            (creator_id,),
        )
        rows = db.fetchall()
        return [_row_to_kb_dict(r) for r in rows]
    finally:
        db.close()


def get_kb(kb_id: int, creator_id: int) -> dict | None:
    """获取单个知识库详情。KB 不存在或不属于该用户时返回 None。"""
    db = _get_db()
    try:
        db.execute(
            """SELECT id, chunk_count, created_at, description, document_count,
                      enabled, milvus_collection_name, name, type, updated_at, creator_id
               FROM fp_knowledge_base
               WHERE id = %s AND creator_id = %s""",
            (kb_id, creator_id),
        )
        row = db.fetchone()
        if not row:
            return None
        return _row_to_kb_dict(row)
    finally:
        db.close()


def delete_kb(kb_id: int, creator_id: int, importer) -> dict:
    """删除知识库：删除 Milvus collection + MySQL 记录（CASCADE 自动删除文档）。"""
    db = _get_db()
    try:
        kb = get_kb(kb_id, creator_id)
        if not kb:
            raise NotFoundError(f"知识库 {kb_id} 不存在或无权访问")

        # 删除 Milvus collection
        _drop_user_kb_collection(kb_id, importer)

        # 删除 MySQL 记录（CASCADE 自动删除 fp_document）
        db.execute("DELETE FROM fp_knowledge_base WHERE id = %s", (kb_id,))
        db.commit()

        return {"success": True, "message": f"知识库 '{kb['name']}' 已删除"}

    finally:
        db.close()


def _row_to_kb_dict(row) -> dict:
    """fp_knowledge_base row → dict。
    Columns: id, chunk_count, created_at, description, document_count,
             enabled, milvus_collection_name, name, type, updated_at, creator_id
    """
    def _fmt(v):
        if isinstance(v, datetime):
            return v.strftime("%Y-%m-%d %H:%M:%S")
        if isinstance(v, bytes):  # BIT(1) returns bytes
            return v[0] == 1 if v else False
        return v

    return {
        "kb_id": row[0],
        "chunk_count": row[1],
        "created_at": _fmt(row[2]),
        "description": row[3] or "",
        "document_count": row[4],
        "enabled": _fmt(row[5]),
        "milvus_collection_name": row[6] or "",
        "name": row[7],
        "type": row[8],
        "updated_at": _fmt(row[9]),
        "creator_id": row[10],
    }


# ═══════════════════════════════════════════════════════════════════════
# 用户 KB 的简化 Milvus Collection Schema
# ═══════════════════════════════════════════════════════════════════════

def _ensure_user_kb_collection(kb_id: int, importer) -> str:
    """为用户 KB 创建简化 schema 的 Milvus collection（无金融报告字段）。

    简化 schema:
      chunk_id (PK), doc_id, title, title_path, text,
      embedding (1024-d FLOAT_VECTOR), sparse_embedding (SPARSE_FLOAT_VECTOR)
    """
    collection_name = _get_db().get_collection_name_by_kb_id(kb_id)

    if importer.client.has_collection(collection_name=collection_name):
        return collection_name

    schema = importer.client.create_schema(
        description=f"User knowledge base {kb_id}",
        auto_id=False,
    )
    schema.add_field(field_name="chunk_id", datatype=DataType.VARCHAR, max_length=64, is_primary=True)
    schema.add_field(field_name="doc_id", datatype=DataType.VARCHAR, max_length=256)
    schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=1024)
    schema.add_field(field_name="title_path", datatype=DataType.VARCHAR, max_length=2048)
    schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)
    schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=1024)
    schema.add_field(field_name="sparse_embedding", datatype=DataType.SPARSE_FLOAT_VECTOR)

    index_params = importer.client.prepare_index_params()
    index_params.add_index(
        field_name="embedding",
        metric_type="COSINE",
        index_type="HNSW",
        params={"M": 32, "efConstruction": 200},
    )
    index_params.add_index(
        field_name="sparse_embedding",
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="IP",
    )

    importer.client.create_collection(
        collection_name=collection_name,
        schema=schema,
        index_params=index_params,
    )
    print(f"[kb_service] 用户 KB collection 已创建: {collection_name}")
    return collection_name


def _drop_user_kb_collection(kb_id: int, importer) -> None:
    """删除用户 KB 对应的 Milvus collection。"""
    collection_name = _get_db().get_collection_name_by_kb_id(kb_id)
    try:
        if importer.client.has_collection(collection_name=collection_name):
            importer.client.drop_collection(collection_name=collection_name)
            print(f"[kb_service] Collection 已删除: {collection_name}")
    except Exception as e:
        logger.warning(f"删除 Milvus collection 失败: {collection_name}, {e}")


# ═══════════════════════════════════════════════════════════════════════
# 文件上传管道
# ═══════════════════════════════════════════════════════════════════════

def upload_to_kb(
    kb_id: int,
    uploader_id: int,
    original_filename: str,
    file_bytes: bytes,
    importer,  # MilvusImporter
) -> dict:
    """上传文件到知识库的完整管道。

    流程: 校验 → 保存 → 转换 → 分块 → 编码 → 插入 Milvus（简化 schema） → 更新 MySQL

    Returns
    -------
    dict
        {document_id, title, original_file_name, format, file_size,
         status, chunk_count, doc_id}
    """
    # ── ① 校验 ──
    ext = Path(original_filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(f"不支持的文件类型: {ext}，仅支持 {', '.join(ALLOWED_EXTENSIONS)}")

    if len(file_bytes) == 0:
        raise ValueError("文件为空")

    if len(file_bytes) > MAX_FILE_SIZE:
        raise ValueError(f"文件过大: {len(file_bytes)} bytes，上限 {MAX_FILE_SIZE} bytes")

    kb = get_kb(kb_id, uploader_id)
    if not kb:
        raise NotFoundError(f"知识库 {kb_id} 不存在或无权访问")

    # ── ② 生成标识 ──
    doc_id = f"kb_{kb_id}_{uuid.uuid4().hex[:12]}"
    file_type = ext.lstrip(".")
    title = Path(original_filename).stem  # 文件名（不含扩展名）作为标题
    collection_name = _get_db().get_collection_name_by_kb_id(kb_id)
    doc_record_id = None  # 初始化，用于异常处理

    # ── ③ 保存临时文件 ──
    upload_dir = UPLOADS_DIR / str(kb_id)
    upload_dir.mkdir(parents=True, exist_ok=True)
    temp_path = upload_dir / f"{doc_id}_{original_filename}"

    try:
        with open(temp_path, "wb") as f:
            f.write(file_bytes)

        # ── ④ 转换为 Markdown 文本 ──
        try:
            markdown_text = _convert_to_markdown(str(temp_path), original_filename, file_type)
        except Exception as e:
            raise RuntimeError(f"文件转换失败: {e}")

        if not markdown_text or not markdown_text.strip():
            raise RuntimeError("文件转换后内容为空")

        # ── ⑤ 插入 MySQL 文档记录 ──
        db = _get_db()
        try:
            db.execute(
                """INSERT INTO fp_document
                   (chunk_count, content, file_path, file_size, format,
                    original_file_name, status, title, knowledge_base_id, uploader_id)
                   VALUES (0, %s, %s, %s, %s, %s, 'processing', %s, %s, %s)""",
                (markdown_text, str(temp_path), len(file_bytes), file_type,
                 original_filename, title, kb_id, uploader_id),
            )
            db.commit()
            doc_record_id = db.cursor.lastrowid
        finally:
            db.close()

        # ── ⑥ 分块（使用 MarkdownChunker，金融字段填占位值） ──
        chunks = importer.chunker.chunk(
            markdown=markdown_text,
            company_name="",           # 用户 KB 不使用金融字段
            stock_code="",
            report_type="用户上传",
            report_year=date.today().year,
            report_date=date.today(),
            doc_id=doc_id,
        )

        if not chunks:
            raise RuntimeError("文档分块结果为空")

        # ── ⑦ token 安全截断 ──
        for chunk in chunks:
            tl = len(importer.model.tokenizer.encode(chunk.text, add_special_tokens=False))
            if tl > 7500:
                chunk.text = _safe_truncate_text(chunk.text, importer.model.tokenizer, 7500)

        # ── ⑧ 编码 ──
        texts = [c.text for c in chunks]
        output = importer.model.encode(
            texts,
            batch_size=64,
            return_dense=True,
            return_sparse=True,
        )
        dense_vecs = output["dense_vecs"]
        sparse_weights = output["lexical_weights"]

        # ── ⑨ 确保 collection 存在 ──
        _ensure_user_kb_collection(kb_id, importer)

        # ── ⑩ 构建简化 entity 并插入 Milvus ──
        entities = []
        for chunk, dv, sw in zip(chunks, dense_vecs, sparse_weights):
            entities.append({
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "title": chunk.title or title,
                "title_path": " > ".join(chunk.title_path) if chunk.title_path else "",
                "text": chunk.text,
                "embedding": dv.tolist(),
                "sparse_embedding": sw,
            })

        _insert_with_retry(importer, collection_name, doc_id, entities)

        # ── ⑪ 更新 MySQL ──
        db = _get_db()
        try:
            db.execute(
                """UPDATE fp_document
                   SET status = 'completed', chunk_count = %s, updated_at = NOW()
                   WHERE id = %s""",
                (len(chunks), doc_record_id),
            )
            db.execute(
                """UPDATE fp_knowledge_base
                   SET document_count = document_count + 1, chunk_count = chunk_count + %s
                   WHERE id = %s""",
                (len(chunks), kb_id),
            )
            db.commit()
        finally:
            db.close()

        return {
            "document_id": doc_record_id,
            "title": title,
            "original_file_name": original_filename,
            "format": file_type,
            "file_size": len(file_bytes),
            "status": "completed",
            "chunk_count": len(chunks),
            "doc_id": doc_id,
        }

    except Exception as e:
        _mark_document_failed(doc_record_id, str(e))
        raise

    finally:
        # ── ⑫ 清理临时文件 ──
        _cleanup_temp(str(temp_path), upload_dir)


# ═══════════════════════════════════════════════════════════════════════
# 内部辅助函数
# ═══════════════════════════════════════════════════════════════════════

def _insert_with_retry(importer, collection_name: str, doc_id: str, entities: list[dict]) -> None:
    """幂等插入：先删除同 doc_id 数据，再插入，失败自动重试。"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            importer.client.delete(
                collection_name=collection_name,
                filter=f'doc_id == "{doc_id}"',
            )
            importer.client.insert(
                collection_name=collection_name,
                data=entities,
            )
            return
        except Exception as e:
            if attempt == max_retries - 1:
                raise RuntimeError(f"Milvus 插入失败（已重试 {max_retries} 次）: {e}")
            time.sleep(2 ** attempt)


def _convert_to_markdown(file_path: str, filename: str, file_type: str) -> str:
    """根据文件类型将文件转换为 Markdown 文本。"""
    if file_type == "md":
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    elif file_type == "pdf":
        output_dir = str(Path(file_path).parent)
        pdf2md = Pdf2Md()
        pdf2md.parse(file_path, output_dir)

        # 找到生成的 .md 文件
        stem = Path(filename).stem
        expected_md = Path(output_dir) / f"{stem}.md"
        if expected_md.exists():
            with open(expected_md, "r", encoding="utf-8") as f:
                return f.read()

        # 如果文件名不匹配，尝试找最新的 .md 文件
        md_files = sorted(
            Path(output_dir).glob("*.md"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if md_files:
            with open(md_files[0], "r", encoding="utf-8") as f:
                return f.read()

        raise RuntimeError("PDF 转换后未找到 Markdown 输出文件")

    elif file_type == "txt":
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        name_stem = Path(filename).stem
        return f"# {name_stem}\n\n{content}"

    else:
        raise ValueError(f"不支持的文件类型: {file_type}")


def _safe_truncate_text(text: str, tokenizer, limit: int) -> str:
    """使用 tokenizer 内置截断安全截断文本。"""
    encoded = tokenizer(text, truncation=True, max_length=limit, return_tensors=None)
    return tokenizer.decode(encoded["input_ids"], skip_special_tokens=True)


def _mark_document_failed(doc_record_id: int | None, error_message: str) -> None:
    """将文档记录标记为失败状态。"""
    if doc_record_id is None:
        logger.warning(f"无法标记失败（无 doc_record_id）: {error_message}")
        return
    try:
        db = _get_db()
        try:
            db.execute(
                """UPDATE fp_document
                   SET status = 'failed', summary = %s, updated_at = NOW()
                   WHERE id = %s""",
                (error_message[:2000], doc_record_id),
            )
            db.commit()
        finally:
            db.close()
    except Exception:
        logger.warning(f"无法更新文档失败状态: {error_message}")


def _cleanup_temp(temp_path: str, upload_dir: Path) -> None:
    """删除临时文件。"""
    try:
        if os.path.isfile(temp_path):
            os.remove(temp_path)
        base = Path(temp_path)
        # 同时删除 PDF 转换可能产生的其他临时文件
        for f in base.parent.glob(f"{base.stem}*"):
            if f.is_file():
                os.remove(f)
        if upload_dir.exists() and not any(upload_dir.iterdir()):
            upload_dir.rmdir()
    except Exception:
        logger.warning(f"清理临时文件失败: {temp_path}")


# ═══════════════════════════════════════════════════════════════════════
# 文档管理（使用 fp_document 表）
# ═══════════════════════════════════════════════════════════════════════

def list_documents(kb_id: int, creator_id: int) -> list[dict]:
    """列出知识库中的所有文档。"""
    kb = get_kb(kb_id, creator_id)
    if not kb:
        raise NotFoundError(f"知识库 {kb_id} 不存在或无权访问")

    db = _get_db()
    try:
        db.execute(
            """SELECT id, chunk_count, content, created_at, file_path, file_size,
                      format, original_file_name, status, summary, title, updated_at,
                      knowledge_base_id, uploader_id
               FROM fp_document
               WHERE knowledge_base_id = %s
               ORDER BY created_at DESC""",
            (kb_id,),
        )
        rows = db.fetchall()
        return [_row_to_doc_dict(r) for r in rows]
    finally:
        db.close()


def delete_document(kb_id: int, doc_id: int, creator_id: int, importer) -> dict:
    """从知识库中删除指定文档（doc_id 对应 fp_document.id）。"""
    kb = get_kb(kb_id, creator_id)
    if not kb:
        raise NotFoundError(f"知识库 {kb_id} 不存在或无权访问")

    collection_name = _get_db().get_collection_name_by_kb_id(kb_id)

    db = _get_db()
    try:
        # 先查文档记录
        db.execute(
            "SELECT id, chunk_count, file_path FROM fp_document WHERE id = %s AND knowledge_base_id = %s",
            (doc_id, kb_id),
        )
        doc_row = db.fetchone()
        if not doc_row:
            raise NotFoundError(f"文档 {doc_id} 不存在")
        if doc_row[2]:
            # 从文件路径提取 Milvus doc_id（格式: uploads/{kb_id}/kb_{kb_id}_{uuid}_filename）
            milvus_doc_id = _extract_doc_id_from_path(doc_row[2])
            if milvus_doc_id:
                try:
                    importer.client.delete(
                        collection_name=collection_name,
                        filter=f'doc_id == "{milvus_doc_id}"',
                    )
                except Exception as e:
                    logger.warning(f"Milvus 删除失败: {e}")

        chunk_count = doc_row[1]

        # 删除 MySQL 记录
        db.execute("DELETE FROM fp_document WHERE id = %s", (doc_id,))
        if chunk_count > 0:
            db.execute(
                """UPDATE fp_knowledge_base
                   SET document_count = GREATEST(document_count - 1, 0),
                       chunk_count = GREATEST(chunk_count - %s, 0)
                   WHERE id = %s""",
                (chunk_count, kb_id),
            )
        db.commit()
        return {"success": True, "message": f"文档 {doc_id} 已删除"}
    finally:
        db.close()


def _extract_doc_id_from_path(file_path: str) -> str | None:
    """从文件路径中提取 Milvus doc_id。
    路径格式: uploads/{kb_id}/kb_{kb_id}_{uuid12}_{filename}.ext
    doc_id 格式: kb_{kb_id}_{uuid12}
    """
    import re
    # 匹配 kb_{kb_id}_{12位hexuuid}
    m = re.search(r'(kb_\d+_[a-f0-9]{12})', file_path)
    return m.group(1) if m else None


def _row_to_doc_dict(row) -> dict:
    """fp_document row → dict。
    Columns: id, chunk_count, content, created_at, file_path, file_size,
             format, original_file_name, status, summary, title, updated_at,
             knowledge_base_id, uploader_id
    """
    def _fmt(v):
        if isinstance(v, datetime):
            return v.strftime("%Y-%m-%d %H:%M:%S")
        return v

    return {
        "document_id": row[0],
        "chunk_count": row[1],
        "content_preview": (row[2][:500] + "...") if row[2] and len(row[2]) > 500 else (row[2] or ""),
        "created_at": _fmt(row[3]),
        "file_path": row[4] or "",
        "file_size": row[5],
        "format": row[6] or "",
        "original_file_name": row[7],
        "status": row[8],
        "summary": row[9] or "",
        "title": row[10],
        "updated_at": _fmt(row[11]),
        "knowledge_base_id": row[12],
        "uploader_id": row[13],
    }


# ═══════════════════════════════════════════════════════════════════════
# 自定义异常
# ═══════════════════════════════════════════════════════════════════════

class DuplicateError(Exception):
    """资源已存在（如重复名称）。"""
    pass


class NotFoundError(Exception):
    """资源不存在。"""
    pass
