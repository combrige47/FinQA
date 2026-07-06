from datetime import date
from pathlib import Path

from transformers import AutoTokenizer

from parse.parse_md import MarkdownChunker
MODEL_PATH = str(Path(__file__).parent.parent / "model" / "bge-m3")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True
)
with open("../vector/2026-03-31_000488_ST晨鸣_2025年年度报告.md", encoding="utf-8") as f:
    md = f.read()

chunker = MarkdownChunker(tokenizer)

chunks = chunker.chunk(
    markdown=md,
    company_name="ST晨鸣",
    stock_code="000488",
    report_type="年度报告",
    report_year=2025,
    report_date=date(2026,3,31),
    doc_id="000001_2025_annual"
)
lengths = []

for chunk in chunks:
    lengths.append(
        len(tokenizer.encode(
            chunk.text,
            add_special_tokens=False
        ))
    )

print(min(lengths))
print(max(lengths))
print(sum(lengths)/len(lengths))