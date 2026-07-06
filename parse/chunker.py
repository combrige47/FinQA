from dataclasses import dataclass, field
from datetime import date


@dataclass
class Chunk:
    chunk_id : str
    doc_id : str
    text : str
    company_name: str
    stock_code: str
    report_type: str
    report_year: int
    report_date: date
    title: str
    title_path: list[str]
    page_start: int | None = None
    page_end: int | None = None
    metadata: dict = field(default_factory=dict)
    chunk_index: int = 0
    chunk_count: int = 0