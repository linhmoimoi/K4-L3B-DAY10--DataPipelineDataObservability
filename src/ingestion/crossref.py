from __future__ import annotations

from dataclasses import asdict, dataclass
from html import unescape
import json
from pathlib import Path
import re
import time
from datetime import UTC, datetime
from typing import Any
from urllib.parse import quote

import requests

from core.config import Settings
from core.utils import normalize_whitespace, read_json, write_json


_CROSSREF_ENDPOINT = "https://api.crossref.org/works"
_REQUEST_TIMEOUT = (5, 30)
_MAX_ATTEMPTS = 3
_OPENALEX_ENDPOINT = "https://api.openalex.org/works/https://doi.org/"


def enrich_missing_categories(records: list[PaperRecord], raw_records_path: Path) -> tuple[list[PaperRecord], dict[str, Any]]:
    """Fill missing categories only from Crossref/OpenAlex metadata and persist DOI lineage."""
    lineage_path = raw_records_path.parent / "category_enrichment.json"
    try:
        cached_payload = read_json(lineage_path)
        cached = cached_payload.get("records", {}) if isinstance(cached_payload, dict) else {}
        if not isinstance(cached, dict):
            cached = {}
    except (OSError, json.JSONDecodeError, TypeError):
        cached = {}

    enriched: list[PaperRecord] = []
    lineage: dict[str, Any] = dict(cached)
    for record in records:
        if record.categories:
            enriched.append(record)
            continue
        doi_key = record.paper_id.casefold()
        entry = cached.get(doi_key)
        if not isinstance(entry, dict) or entry.get("status") not in {"success", "failed"}:
            entry = {
                "doi": record.paper_id,
                "paper_id": record.paper_id,
                "categories": [],
                "source": None,
                "source_record_id": None,
                "source_url": None,
                "queried_at": datetime.now(UTC).isoformat(),
                "status": "failed",
                "error": None,
            }
            for attempt in range(_MAX_ATTEMPTS):
                response: requests.Response | None = None
                try:
                    response = requests.get(
                        _OPENALEX_ENDPOINT + quote(record.paper_id, safe="/:"),
                        timeout=_REQUEST_TIMEOUT,
                        headers={"Accept": "application/json", "User-Agent": "K4-L3B-DataObservability/1.0"},
                    )
                    if response.status_code == 429 or 500 <= response.status_code < 600:
                        if attempt + 1 < _MAX_ATTEMPTS:
                            time.sleep(_retry_delay(response, attempt))
                            continue
                    response.raise_for_status()
                    payload = response.json()
                    categories = []
                    primary = payload.get("primary_topic")
                    if isinstance(primary, dict) and isinstance(primary.get("display_name"), str):
                        categories.append(primary["display_name"].strip())
                    topics = payload.get("topics")
                    if isinstance(topics, list):
                        categories.extend(
                            topic["display_name"].strip()
                            for topic in topics
                            if isinstance(topic, dict) and isinstance(topic.get("display_name"), str)
                        )
                    categories = list(dict.fromkeys(value for value in categories if value))
                    if categories:
                        entry.update(
                            categories=categories,
                            source="OpenAlex",
                            source_record_id=_clean_text(payload.get("id")),
                            source_url=_clean_text(payload.get("id")) or _OPENALEX_ENDPOINT + quote(record.paper_id, safe="/:"),
                            status="success",
                            error=None,
                        )
                    else:
                        entry["error"] = "OpenAlex record has no category metadata."
                    break
                except (requests.RequestException, ValueError, TypeError) as exc:
                    entry["error"] = f"{type(exc).__name__}: {exc}"[:300]
                    retryable = not isinstance(exc, requests.HTTPError)
                    if isinstance(exc, requests.HTTPError):
                        status = exc.response.status_code if exc.response is not None else None
                        retryable = status == 429 or (isinstance(status, int) and 500 <= status < 600)
                    if retryable and attempt + 1 < _MAX_ATTEMPTS:
                        time.sleep(_retry_delay(response, attempt))
                        continue
                    break
            lineage[doi_key] = entry

        categories = entry.get("categories") if entry.get("status") == "success" else []
        if categories:
            enriched.append(PaperRecord(**{**asdict(record), "categories": categories, "primary_category": categories[0]}))
        else:
            enriched.append(record)

    sidecar = {
        "source": "Crossref subject when present; otherwise OpenAlex metadata",
        "records": lineage,
        "queried_at": datetime.now(UTC).isoformat(),
    }
    write_json(lineage_path, sidecar)
    _write_records(raw_records_path, enriched)
    return enriched, sidecar


@dataclass(frozen=True)
class PaperRecord:
    paper_id: str
    title: str
    summary: str
    authors: list[str]
    categories: list[str]
    primary_category: str
    published: str
    updated: str
    abs_url: str
    pdf_url: str
    comment: str


def _clean_text(value: Any, *, strip_markup: bool = False) -> str:
    if not isinstance(value, str):
        return ""
    text = unescape(value)
    if strip_markup:
        # Crossref abstracts commonly contain JATS/XML tags. Decode entities a
        # second time before removing tags, then decode any remaining entities.
        text = unescape(text)
        text = re.sub(r"<!--.*?-->|<[^>]*>", " ", text, flags=re.DOTALL)
        text = unescape(text)
    return normalize_whitespace(text)


def _first_text(value: Any) -> str:
    candidates = value if isinstance(value, list) else [value]
    for candidate in candidates:
        cleaned = _clean_text(candidate)
        if cleaned:
            return cleaned
    return ""


def _date_string(value: Any) -> str:
    if isinstance(value, dict):
        parts = value.get("date-parts")
        if isinstance(parts, list) and parts and isinstance(parts[0], (list, tuple)):
            date_parts = parts[0]
            if date_parts:
                year = _clean_text(str(date_parts[0]))
                if year:
                    result = year.zfill(4)
                    if len(date_parts) > 1 and date_parts[1] not in (None, ""):
                        result += f"-{str(date_parts[1]).zfill(2)}"
                    if len(date_parts) > 2 and date_parts[2] not in (None, ""):
                        result += f"-{str(date_parts[2]).zfill(2)}"
                    return result
        for key in ("date-time", "timestamp"):
            text = _clean_text(value.get(key))
            if text:
                match = re.match(r"^(\d{4}-\d{2}-\d{2})", text)
                return match.group(1) if match else text
        return ""
    if isinstance(value, (list, tuple)):
        return _date_string({"date-parts": [value]})
    text = _clean_text(value)
    if not text:
        return ""
    match = re.match(r"^(\d{4}(?:-\d{1,2})?(?:-\d{1,2})?)", text)
    return match.group(1) if match else text


def _date_from(item: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        date = _date_string(item.get(key))
        if date:
            return date
    return ""


def _as_text_list(value: Any) -> list[str]:
    values = value if isinstance(value, (list, tuple)) else [value]
    return [text for entry in values if (text := _clean_text(entry))]


def _primary_category(item: dict[str, Any], categories: list[str]) -> str:
    for key in ("primary_category", "primary-category", "primarySubject", "category"):
        value = item.get(key)
        if isinstance(value, (list, tuple)):
            value = value[0] if value else None
        category = _clean_text(value)
        if category:
            return category
    return categories[0] if categories else ""


def _pdf_url(item: dict[str, Any]) -> str:
    for key in ("pdf_url", "pdf-url", "pdf"):
        raw_value = item.get(key)
        if isinstance(raw_value, dict):
            raw_value = raw_value.get("URL") or raw_value.get("url")
        value = _clean_text(raw_value)
        if value:
            return value

    links = item.get("link")
    if isinstance(links, list):
        for link in links:
            if not isinstance(link, dict):
                continue
            content_type = _clean_text(link.get("content-type")).lower()
            url = _clean_text(link.get("URL") or link.get("url"))
            if url and ("pdf" in content_type or url.lower().split("?", 1)[0].endswith(".pdf")):
                return url

    resource = item.get("resource")
    if isinstance(resource, dict):
        primary = resource.get("primary")
        if isinstance(primary, dict):
            content_type = _clean_text(primary.get("content-type") or primary.get("type")).lower()
            if "pdf" in content_type:
                return _clean_text(primary.get("pdf_url") or primary.get("URL") or primary.get("url"))
    return ""


def _comment(item: dict[str, Any]) -> str:
    value = item.get("comment")
    if isinstance(value, list):
        parts: list[str] = []
        for entry in value:
            if isinstance(entry, dict):
                entry = entry.get("text") or entry.get("comment")
            text = _clean_text(entry)
            if text:
                parts.append(text)
        return normalize_whitespace(" ".join(parts))
    if isinstance(value, dict):
        value = value.get("text") or value.get("comment")
    return _clean_text(value)


def parse_crossref_payload(payload: dict) -> list[PaperRecord]:
    """Parse Crossref's works response into normalized ``PaperRecord`` objects."""
    if not isinstance(payload, dict):
        return []
    message = payload.get("message")
    if not isinstance(message, dict):
        return []
    items = message.get("items")
    if not isinstance(items, list):
        return []

    records: list[PaperRecord] = []
    for item in items:
        if not isinstance(item, dict):
            continue

        paper_id = _clean_text(item.get("DOI") or item.get("doi") or item.get("id") or item.get("URL"))
        title = _first_text(item.get("title"))
        if not paper_id or not title:
            continue

        author_entries = item.get("author")
        authors: list[str] = []
        if isinstance(author_entries, list):
            for author in author_entries:
                if not isinstance(author, dict):
                    continue
                given = _clean_text(author.get("given"))
                family = _clean_text(author.get("family"))
                name = normalize_whitespace(" ".join(part for part in (given, family) if part))
                if not name:
                    name = _clean_text(author.get("name"))
                if name:
                    authors.append(name)

        categories = _as_text_list(item.get("subject"))
        if not categories:
            categories = _as_text_list(item.get("categories"))

        abs_url = ""
        for key in ("abs_url", "abstract_url", "URL", "url"):
            abs_url = _clean_text(item.get(key))
            if abs_url:
                break

        records.append(
            PaperRecord(
                paper_id=paper_id,
                title=title,
                summary=_clean_text(item.get("abstract"), strip_markup=True),
                authors=authors,
                categories=categories,
                primary_category=_primary_category(item, categories),
                published=_date_from(
                    item,
                    ("published-print", "published-online", "published", "issued", "published-other", "first-online"),
                ),
                updated=_date_from(item, ("updated", "indexed", "deposited")),
                abs_url=abs_url,
                pdf_url=_pdf_url(item),
                comment=_comment(item),
            )
        )
    return records


def _validate_crossref_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Crossref response must be a JSON object.")
    message = payload.get("message")
    if not isinstance(message, dict) or not isinstance(message.get("items"), list):
        raise ValueError("Crossref response is missing message.items.")
    return payload


def _endpoint_from_settings(settings: Settings) -> str:
    endpoint = getattr(settings, "source_api_endpoint", None)
    if not isinstance(endpoint, str) or not endpoint.strip().lower().startswith(("http://", "https://")):
        endpoint = getattr(settings, "source_api", None)
    if isinstance(endpoint, str) and endpoint.strip().lower().startswith(("http://", "https://")):
        return endpoint.strip()
    return _CROSSREF_ENDPOINT


def _retry_delay(response: requests.Response | None, attempt: int) -> float:
    if response is not None:
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            try:
                return min(10.0, max(0.0, float(retry_after)))
            except ValueError:
                pass
    return float(2**attempt)


def _write_records(path: Path, records: list[PaperRecord]) -> None:
    write_json(path, [asdict(record) for record in records])


def _load_response_snapshot(path: Path) -> dict[str, Any]:
    payload = _validate_crossref_payload(read_json(path))
    return payload


def fetch_source_records(settings: Settings) -> list[PaperRecord]:
    """Fetch Crossref records, preserving the response and falling back to its snapshot."""
    response_path = Path(settings.paths.raw_api_response)
    records_path = Path(settings.paths.raw_records_json)
    response_path.parent.mkdir(parents=True, exist_ok=True)
    records_path.parent.mkdir(parents=True, exist_ok=True)

    params = {
        "query": settings.source_query,
        "filter": settings.source_filter,
        "rows": settings.max_results,
    }
    last_error: Exception | None = None
    for attempt in range(_MAX_ATTEMPTS):
        response: requests.Response | None = None
        try:
            response = requests.get(
                _endpoint_from_settings(settings),
                params=params,
                timeout=_REQUEST_TIMEOUT,
                headers={"Accept": "application/json"},
            )
            if response.status_code in (429, 503):
                last_error = requests.HTTPError(f"Crossref temporarily returned HTTP {response.status_code}.")
                if attempt + 1 < _MAX_ATTEMPTS:
                    time.sleep(_retry_delay(response, attempt))
                    continue
                break

            response.raise_for_status()
            payload = _validate_crossref_payload(response.json())
            records = parse_crossref_payload(payload)
            # Store the decoded response as received from the API, without
            # normalizing or removing any fields from the raw payload.
            write_json(response_path, payload)
            _write_records(records_path, records)
            return records
        except (requests.RequestException, ValueError, TypeError) as exc:
            last_error = exc
            retryable = not isinstance(exc, requests.HTTPError)
            if isinstance(exc, requests.HTTPError):
                status = exc.response.status_code if exc.response is not None else None
                retryable = status == 503 or (isinstance(status, int) and 500 <= status < 600)
            if retryable and attempt + 1 < _MAX_ATTEMPTS:
                time.sleep(_retry_delay(response, attempt))
                continue
            break

    try:
        snapshot = _load_response_snapshot(response_path)
    except (OSError, json.JSONDecodeError, ValueError, TypeError) as snapshot_error:
        cause = type(last_error).__name__ if last_error is not None else "unknown API error"
        if isinstance(snapshot_error, FileNotFoundError):
            snapshot_state = "snapshot does not exist"
        else:
            snapshot_state = "snapshot is unreadable or invalid"
        raise RuntimeError(
            f"Crossref request/response failed ({cause}) and {snapshot_state}: {response_path}"
        ) from None

    records = parse_crossref_payload(snapshot)
    _write_records(records_path, records)
    return records


def load_raw_records(path: Path) -> list[PaperRecord]:
    """Load a Crossref response snapshot or serialized ``PaperRecord`` list."""
    payload = read_json(path)
    if isinstance(payload, dict):
        return parse_crossref_payload(payload)
    if not isinstance(payload, list):
        return []

    records: list[PaperRecord] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        paper_id = _clean_text(item.get("paper_id"))
        title = _clean_text(item.get("title"))
        if not paper_id or not title:
            continue
        records.append(
            PaperRecord(
                paper_id=paper_id,
                title=title,
                summary=_clean_text(item.get("summary")),
                authors=_as_text_list(item.get("authors")),
                categories=_as_text_list(item.get("categories")),
                primary_category=_clean_text(item.get("primary_category")),
                published=_clean_text(item.get("published")),
                updated=_clean_text(item.get("updated")),
                abs_url=_clean_text(item.get("abs_url")),
                pdf_url=_clean_text(item.get("pdf_url")),
                comment=_clean_text(item.get("comment")),
            )
        )
    return records
