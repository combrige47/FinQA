import uuid
import re


from parse.chunker import Chunk



MAX_TOKEN = 800
OVERLAP = 120


class MarkdownChunker:

    heading_pattern = re.compile(r"^(#+)\s+(.*)")

    # PDF转换可能丢失 # 前缀的中文编号标题，如"六、主要会计数据和财务指标"
    _chinese_heading_pattern = re.compile(
        r"^([一二三四五六七八九十]+)、(.+)$"
    )

    # 无实际内容的行：图片引用、纯页码等
    _skip_patterns = [
        re.compile(r"^!\[image", re.IGNORECASE),
        re.compile(r"^\s*\d+\s*$"),
    ]

    def __init__(self, tokenizer,
                 max_tokens=800,
                 overlap=120,
                 min_tokens=50):

        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.overlap = overlap
        self.min_tokens = min_tokens

    def token_len(self, text):

        return len(
            self.tokenizer.encode(
                text,
                add_special_tokens=False
            )
        )

    @staticmethod
    def _enrich_chunk_text(text, title_path):
        """为 chunk text 添加章节路径前缀，提供 embedding 模型上下文信息"""
        if not title_path:
            return text
        path_str = " > ".join(title_path)
        return f"【章节】{path_str}\n\n{text}"
    @classmethod
    def _is_chinese_heading(cls, line, prev_line_empty):
        """检测是否为PDF转换后丢失#标记的中文编号标题

        条件:
        1. 匹配"数字+、+文字"模式
        2. 长度 ≤ 50 字符
        3. 不含表格标记
        4. 前一行是空行（独立性信号）
        """
        if not prev_line_empty:
            return False
        if len(line) > 50:
            return False
        if "|" in line or "<br>" in line:
            return False
        return bool(cls._chinese_heading_pattern.match(line.strip()))

    def parse_sections(self, markdown):

        sections = []

        stack = []

        buffer = []

        last_heading_level = 0

        prev_line_empty = True

        for line in markdown.splitlines():

            m = self.heading_pattern.match(line)

            if m:

                if buffer:

                    sections.append(
                        (
                            stack.copy(),
                            "\n".join(buffer).strip()
                        )
                    )

                    buffer = []

                level = len(m.group(1))

                last_heading_level = level

                title = m.group(2)

                stack = stack[:level-1]

                stack.append(title)

            elif self._is_chinese_heading(line, prev_line_empty):

                # PDF转换丢失#的中文编号标题，推断层级
                if buffer:

                    sections.append(
                        (
                            stack.copy(),
                            "\n".join(buffer).strip()
                        )
                    )

                    buffer = []

                # 推断为上一显式标题的同级（至少为3级=###）
                inferred_level = max(3, last_heading_level) if last_heading_level > 0 else 3

                title = line.strip()

                stack = stack[:inferred_level-1]

                stack.append(title)

            else:

                buffer.append(line)

            prev_line_empty = (line.strip() == "")

        if buffer:

            sections.append(
                (
                    stack.copy(),
                    "\n".join(buffer).strip()
                )
            )

        return sections
    def split_paragraph(self, text):

        paragraphs = text.split("\n\n")

        chunks = []

        current = ""

        for para in paragraphs:

            candidate = current + "\n\n" + para

            if self.token_len(candidate) <= self.max_tokens:

                current = candidate

            else:

                if current:

                    chunks.append(current.strip())

                current = para

        if current:

            chunks.append(current.strip())

        return chunks
    def split_window(self, text):
        """基于 token 滑动窗口切分，并保护表格行不被切断"""

        lines = text.split("\n")

        # 先尝试按行构建 chunk，保护表格块的完整性
        chunks = []
        current_lines = []
        current_tokens = 0
        in_table = False

        for i, line in enumerate(lines):
            stripped = line.strip()
            is_table_line = stripped.startswith("|")

            line_tokens = self.token_len(line)

            # 判断表格块边界
            if is_table_line and not in_table:
                in_table = True
            elif not is_table_line and in_table:
                in_table = False

            # 当累计 token 超出限制且有内容时，考虑 flush
            if current_tokens + line_tokens > self.max_tokens and current_lines:
                if in_table and current_tokens > 0:
                    # 正在表格内部：暂不 flush，等表格块结束再处理
                    # 但如果当前行单独就超限，先 flush 之前的内容
                    if line_tokens > self.max_tokens:
                        chunks.append("\n".join(current_lines).strip())
                        current_lines = []
                        current_tokens = 0
                        in_table = False
                else:
                    chunks.append("\n".join(current_lines).strip())
                    current_lines = []
                    current_tokens = 0

            # 处理超长单行：回退到 token 窗口
            if line_tokens > self.max_tokens:
                if current_lines:
                    chunks.append("\n".join(current_lines).strip())
                    current_lines = []
                    current_tokens = 0
                    in_table = False

                line_ids = self.tokenizer.encode(line, add_special_tokens=False)
                s = 0
                while s < len(line_ids):
                    e = s + self.max_tokens
                    chunks.append(self.tokenizer.decode(line_ids[s:e]))
                    s += self.max_tokens - self.overlap
            else:
                current_lines.append(line)
                current_tokens += line_tokens

        if current_lines:
            chunks.append("\n".join(current_lines).strip())

        return chunks
    @classmethod
    def _is_skip_chunk(cls, text):
        """检测是否为应跳过的无效chunk（图片引用、纯页码等）"""
        stripped = text.strip()
        if not stripped:
            return True
        for pat in cls._skip_patterns:
            if pat.match(stripped):
                return True
        return False

    def chunk(
            self,
            markdown,
            company_name,
            stock_code,
            report_type,
            report_year,
            report_date,
            doc_id
    ):

        results = []

        sections = self.parse_sections(markdown)

        for title_path, text in sections:

            if not text:

                continue

            pieces = [text]

            if self.token_len(text) > self.max_tokens:

                pieces = self.split_paragraph(text)

            final = []

            for p in pieces:

                if self.token_len(p) > self.max_tokens:

                    final.extend(
                        self.split_window(p)
                    )

                else:

                    final.append(p)

            for p in final:

                # 过滤无效 chunk：图片引用、纯页码等
                if self._is_skip_chunk(p):
                    continue

                # 构建完整标题路径用于 title 字段和 text 上下文
                full_title = " > ".join(title_path) if title_path else ""
                # Milvus title 字段 max_length=256，超出截断
                display_title = full_title[:250] + "..." if len(full_title) > 256 else full_title

                enriched = self._enrich_chunk_text(p, title_path)

                # 过滤过短的 chunk
                if self.token_len(enriched) < self.min_tokens:
                    continue

                chunk = Chunk(

                    chunk_id=str(uuid.uuid4()),

                    doc_id=doc_id,

                    text=enriched,

                    company_name=company_name,

                    stock_code=stock_code,

                    report_type=report_type,

                    report_year=report_year,

                    report_date=report_date,

                    title=display_title,

                    title_path=title_path.copy(),

                    page_start=None,

                    page_end=None,

                    metadata={}
                )

                results.append(chunk)

        total = len(results)

        for i, chunk in enumerate(results):

            chunk.metadata["chunk_index"] = i

            chunk.metadata["chunk_count"] = total

        return results