from __future__ import annotations

from datetime import date, datetime
from typing import Any

import pandas as pd

from core.utils import normalize_whitespace
from ingestion.crossref import PaperRecord


_COLUMNS = [
    "paper_id",
    "title",
    "summary",
    "authors",
    "authors_joined",
    "categories",
    "categories_joined",
    "primary_category",
    "published",
    "updated",
    "age_days",
    "summary_chars",
    "abs_url",
    "pdf_url",
    "comment",
    "text_for_embedding",
]


def _clean_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return normalize_whitespace(value)


def _clean_text_list(value: Any) -> list[str]:
    if isinstance(value, (list, tuple)):
        values = value
    else:
        values = [value]
    return [cleaned for item in values if (cleaned := _clean_text(item))]


def _parse_utc_datetime(value: Any) -> datetime | None:
    if not isinstance(value, (str, date, datetime, pd.Timestamp)):
        return None
    if isinstance(value, str):
        value = normalize_whitespace(value)
        if not value:
            return None
    try:
        parsed = pd.to_datetime(value, errors="coerce", utc=True)
    except (OverflowError, TypeError, ValueError):
        return None
    if pd.isna(parsed):
        return None
    try:
        return parsed.to_pydatetime()
    except (AttributeError, OverflowError, TypeError, ValueError):
        return None


def _record_value(record: PaperRecord, field: str) -> Any:
    if isinstance(record, dict):
        return record.get(field)
    return getattr(record, field, None)


def build_clean_dataframe(records: list[PaperRecord], run_date: datetime) -> pd.DataFrame:
    """Normalize Crossref records and build stable, embedding-ready rows."""
    run_datetime = _parse_utc_datetime(run_date)
    if run_datetime is None:
        raise ValueError("run_date must be a valid datetime.")

    rows: list[dict[str, Any]] = []
    seen_paper_ids: set[str] = set()

    for record in records:
        paper_id = _clean_text(_record_value(record, "paper_id"))
        title = _clean_text(_record_value(record, "title"))
        if not paper_id or not title:
            continue

        published_datetime = _parse_utc_datetime(_record_value(record, "published"))
        if published_datetime is None:
            continue

        dedupe_key = paper_id.casefold()
        if dedupe_key in seen_paper_ids:
            continue
        seen_paper_ids.add(dedupe_key)

        summary = _clean_text(_record_value(record, "summary"))
        authors = _clean_text_list(_record_value(record, "authors"))
        categories = _clean_text_list(_record_value(record, "categories"))
        authors_joined = ", ".join(authors)
        categories_joined = ", ".join(categories)
        published = published_datetime.date().isoformat()

        updated_datetime = _parse_utc_datetime(_record_value(record, "updated"))
        updated = updated_datetime.date().isoformat() if updated_datetime is not None else ""

        text_sections = [
            ("Title", title),
            ("Authors", authors_joined),
            ("Published", published),
            ("Categories", categories_joined),
            ("Summary", summary),
        ]
        text_for_embedding = "\n".join(
            f"{label}: {value}" if value else f"{label}:" for label, value in text_sections
        )

        rows.append(
            {
                "paper_id": paper_id,
                "title": title,
                "summary": summary,
                "authors": authors,
                "authors_joined": authors_joined,
                "categories": categories,
                "categories_joined": categories_joined,
                "primary_category": _clean_text(_record_value(record, "primary_category")),
                "published": published,
                "updated": updated,
                "age_days": (run_datetime - published_datetime).days,
                "summary_chars": len(summary),
                "abs_url": _clean_text(_record_value(record, "abs_url")),
                "pdf_url": _clean_text(_record_value(record, "pdf_url")),
                "comment": _clean_text(_record_value(record, "comment")),
                "text_for_embedding": text_for_embedding,
            }
        )

    dataframe = pd.DataFrame(rows, columns=_COLUMNS)
    if dataframe.empty:
        return dataframe
    return dataframe.sort_values("paper_id", kind="mergesort", ignore_index=True)
