import uuid
import re


from parse.chunker import Chunk



MAX_TOKEN = 800
OVERLAP = 120


class MarkdownChunker:

    heading_pattern = re.compile(r"^(#+)\s+(.*)")

    def __init__(self, tokenizer,
                 max_tokens=800,
                 overlap=120):

        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.overlap = overlap

    def token_len(self, text):

        return len(
            self.tokenizer.encode(
                text,
                add_special_tokens=False
            )
        )
    def parse_sections(self, markdown):

        sections = []

        stack = []

        buffer = []

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

                title = m.group(2)

                stack = stack[:level-1]

                stack.append(title)

            else:

                buffer.append(line)

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

        ids = self.tokenizer.encode(
            text,
            add_special_tokens=False
        )

        chunks = []

        start = 0

        while start < len(ids):

            end = start + self.max_tokens

            piece = ids[start:end]

            chunks.append(
                self.tokenizer.decode(piece)
            )

            start += self.max_tokens - self.overlap

        return chunks
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

                chunk = Chunk(

                    chunk_id=str(uuid.uuid4()),

                    doc_id=doc_id,

                    text=p,

                    company_name=company_name,

                    stock_code=stock_code,

                    report_type=report_type,

                    report_year=report_year,

                    report_date=report_date,

                    title=title_path[-1] if title_path else "",

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