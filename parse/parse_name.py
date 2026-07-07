import re

_CHINESE_TYPES = [
    "年度报告全文及摘要", "半年度报告全文及摘要",
    "年度报告摘要", "半年度报告摘要",
    "年度报告全文", "半年度报告全文",
    "年度报告", "半年度报告",
    "第三季度报告", "第一季度报告",
    "三季度报告", "一季度报告", "季度报告",
    "中期报告", "年度业绩", "中期业绩",
    "第一季度财务报表", "第三季度财务报表",
    "一季度财务报表", "三季度财务报表",
    "年度财务报表", "季度财务报表",
    "第一季度财务报告", "第三季度财务报告",
    "年度财务报告", "季度财务报告",
    "年报全文", "半年报全文",
    "年报", "半年报", "季报", "第一季度报", "第三季度报",
]

_ENGLISH_TYPES = [
    "Abstract of the Annual Report",
    "Abstract of the Semi-Annual Report",
    "Abstract of the Semi-annual Report",
    "Annual Report",
    "Semi-annual Report",
    "Semi-Annual Report",
    "First Quarterly Report",
    "First Quarter Report",
    "Quarterly Report",
    "Report of Q1",
]


def extract_report_meta(filename: str) -> dict | None:
    if filename.endswith(".md"):
        filename = filename[:-3]

    m = re.match(r"^(\d{4}-\d{2}-\d{2})_(.+)", filename)
    if not m:
        return None
    date = m.group(1)
    rest = m.group(2)

    code_m = re.search(r"(\d{6})", rest)
    if not code_m:
        return None
    code = code_m.group(1)
    a, b = code_m.start(), code_m.end()

    before_code = rest[:a].rstrip("_ ")
    after_code = rest[b:].lstrip("_ ")

    result = _find_report_info(after_code)
    if result:
        year, report_type, report_start = result
        name_from_after = after_code[:report_start].strip("_ -：: ")
        parts = [p for p in (before_code, name_from_after) if p]
        company_name = "_".join(parts)
    else:
        result = _find_report_info(before_code)
        if result:
            year, report_type, report_start = result
            company_name = before_code[:report_start].strip("_ -：: ")
        else:
            year = None
            report_type = "其他公告"
            company_name = (before_code + " " + after_code).strip()

    if not company_name:
        company_name = f"未知公司({code})"

    company_name = _clean_company_name(company_name, code)
    report_type = _clean_report_type(report_type)

    return {
        "report_date": date,
        "stock_code": code,
        "company_name": company_name,
        "report_year": year,
        "report_type": report_type,
    }


def _find_report_info(text: str):
    if not text:
        return None

    pattern_a = (
        r"(\d{4})\s*年[_]?\s*("
        + "|".join(re.escape(t) for t in _CHINESE_TYPES)
        + r")"
    )
    m = re.search(pattern_a, text)
    if m:
        year = m.group(1)
        rtype = m.group(2)
        suffix = _capture_suffix(text[m.end():])
        if suffix:
            rtype += suffix
        return (year, rtype, m.start())

    for ct in _CHINESE_TYPES:
        idx = text.find(ct)
        if idx < 0:
            continue
        before = text[max(0, idx - 10):idx]
        ym = re.search(r"(\d{4})\s*年?", before)
        if ym:
            year = ym.group(1)
            year_pos = max(0, idx - 10) + ym.start()
            start = year_pos
        else:
            year = None
            start = idx
        suffix = _capture_suffix(text[idx + len(ct):])
        rtype = ct
        if suffix:
            rtype += suffix
        return (year, rtype, start)

    pattern_c = (
        r"(\d{4})\s+("
        + "|".join(re.escape(t) for t in _ENGLISH_TYPES)
        + r")"
    )
    m = re.search(pattern_c, text, re.IGNORECASE)
    if m:
        year = m.group(1)
        rtype = m.group(2)
        suffix = _capture_suffix(text[m.end():])
        if suffix:
            rtype += suffix
        return (year, rtype, m.start())

    pattern_d = (
        r"("
        + "|".join(re.escape(t) for t in _ENGLISH_TYPES)
        + r")\s+(\d{4})"
    )
    m = re.search(pattern_d, text, re.IGNORECASE)
    if m:
        rtype = m.group(1)
        year = m.group(2)
        return (year, rtype, m.start())

    return None


def _capture_suffix(text: str) -> str:
    if not text:
        return ""
    m = re.match(r"^[（(]([^）)]*)[）)]", text)
    if m:
        return f"（{m.group(1)}）"
    m = re.match(
        r"^(_(?:[a-zA-Z\d一-鿿]+))(?:_(?:[a-zA-Z\d一-鿿]+))*", text
    )
    if m:
        return text[:m.end()]
    return ""


def _clean_company_name(name: str, code: str) -> str:
    name = re.sub(r"^\d+-", "", name)
    name = re.sub(r"[-_]?(?:H股)?公告[-—]*$", "", name)
    name = re.sub(r"[-_]公司$", "", name)
    name = name.strip("_ -：: ")

    if "_" in name:
        parts = [p for p in name.split("_") if p and not re.match(r"^\d{6}$", p)]
        if len(parts) >= 2:
            if len(parts[-1]) >= 6:
                name = parts[-1]
            else:
                name = parts[0]
        elif parts:
            name = parts[0]

    if not name:
        name = f"未知公司({code})"
    return name


def _clean_report_type(rtype: str) -> str:
    rtype = rtype.strip("_ -：: ")
    rtype = re.sub(r"_摘要$", "摘要", rtype)
    rtype = re.sub(r"_全文$", "（全文）", rtype)
    return rtype
