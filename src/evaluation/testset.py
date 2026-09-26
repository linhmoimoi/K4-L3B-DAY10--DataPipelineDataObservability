from __future__ import annotations

from itertools import combinations
import json
from pathlib import Path
import re
from typing import Any

import pandas as pd

from core.utils import normalize_whitespace


_QUESTION_TYPES = ("summary", "authors", "date", "categories")
def _normalize_text(value: Any) -> str:
    if value is None or value is pd.NA or value is pd.NaT:
        return ""
    if isinstance(value, str):
        return normalize_whitespace(value)
    if isinstance(value, (list, tuple)):
        return ", ".join(text for item in value if (text := _normalize_text(item)))
    try:
        missing = pd.isna(value)
        if isinstance(missing, bool) and missing:
            return ""
        if type(missing).__name__ == "bool_" and bool(missing):
            return ""
    except (TypeError, ValueError):
        pass
    if hasattr(value, "isoformat") and callable(value.isoformat):
        try:
            value = value.isoformat()
        except (TypeError, ValueError):
            pass
    return normalize_whitespace(str(value))


def _summary_answer(summary: str) -> str:
    normalized = normalize_whitespace(summary)
    if not normalized:
        return ""
    first_sentence = re.split(r"(?<=[.!?])\s+", normalized, maxsplit=1)[0]
    return normalize_whitespace(first_sentence)


def _joined_value(row: dict[str, Any], joined_column: str, list_column: str) -> str:
    joined = _normalize_text(row.get(joined_column))
    if joined:
        return joined
    return _normalize_text(row.get(list_column))


def _normalized_documents(df: pd.DataFrame) -> list[dict[str, Any]]:
    normalized_rows: list[dict[str, Any]] = []
    for source in df.to_dict(orient="records"):
        paper_id = _normalize_text(source.get("paper_id"))
        title = _normalize_text(source.get("title"))
        if not paper_id or not title:
            continue

        summary = _summary_answer(_normalize_text(source.get("summary")))
        authors = _joined_value(source, "authors_joined", "authors")
        published = _normalize_text(source.get("published"))
        categories = (
            _normalize_text(source.get("categories_joined"))
            or _normalize_text(source.get("primary_category"))
            or _normalize_text(source.get("categories"))
        )

        normalized_rows.append(
            {
                "paper_id": paper_id,
                "_paper_key": paper_id.casefold(),
                "title": title,
                "ground_truths": {
                    "summary": summary,
                    "authors": authors,
                    "date": published,
                    "categories": categories,
                },
            }
        )

    normalized_rows.sort(
        key=lambda row: (
            -sum(bool(value) for value in row["ground_truths"].values()),
            row["_paper_key"],
            row["paper_id"],
            row["title"].casefold(),
            row["title"],
            *(row["ground_truths"][kind] for kind in _QUESTION_TYPES),
        )
    )

    # Treat case variants of a DOI as one document and prefer the most
    # informative, then lexically stable, source row when duplicates exist.
    unique_rows: dict[str, dict[str, Any]] = {}
    for row in normalized_rows:
        unique_rows.setdefault(row["_paper_key"], row)
    return sorted(unique_rows.values(), key=lambda row: (row["_paper_key"], row["paper_id"], row["title"]))


def _quotas(extra_types: tuple[str, str]) -> dict[str, int]:
    return {question_type: 3 if question_type in extra_types else 2 for question_type in _QUESTION_TYPES}


def _question_schedule(quotas: dict[str, int]) -> tuple[str, ...]:
    remaining = dict(quotas)
    schedule: list[str] = []
    while any(remaining.values()):
        for question_type in _QUESTION_TYPES:
            if remaining[question_type] > 0:
                schedule.append(question_type)
                remaining[question_type] -= 1
    return tuple(schedule)


def _match_distinct_documents(
    candidates: dict[str, list[dict[str, Any]]], quotas: dict[str, int]
) -> dict[str, list[dict[str, Any]]] | None:
    slots = [question_type for question_type in _QUESTION_TYPES for _ in range(quotas[question_type])]
    slots.sort(key=lambda question_type: (len(candidates[question_type]), _QUESTION_TYPES.index(question_type)))
    document_assignment: dict[str, str] = {}

    def assign(question_type: str, visited: set[str]) -> bool:
        for row in candidates[question_type]:
            key = row["_paper_key"]
            if key in visited:
                continue
            visited.add(key)
            previous_type = document_assignment.get(key)
            if previous_type is None or assign(previous_type, visited):
                document_assignment[key] = question_type
                return True
        return False

    for question_type in slots:
        if not assign(question_type, set()):
            return None

    return {
        question_type: [
            row for row in candidates[question_type] if document_assignment.get(row["_paper_key"]) == question_type
        ]
        for question_type in _QUESTION_TYPES
    }


def _choose_documents(candidates: dict[str, list[dict[str, Any]]]) -> tuple[dict[str, int], dict[str, list[dict[str, Any]]]]:
    quota_options = [_quotas(pair) for pair in combinations(_QUESTION_TYPES, 2)]
    for quotas in quota_options:
        distinct = _match_distinct_documents(candidates, quotas)
        if distinct is not None:
            return quotas, distinct

    # When the source has fewer than ten suitably complete papers, reuse real
    # documents only after preferring every still-unused candidate.
    quotas = quota_options[0]
    used_keys: set[str] = set()
    usage_count: dict[str, int] = {}
    selected: dict[str, list[dict[str, Any]]] = {kind: [] for kind in _QUESTION_TYPES}
    for question_type in _question_schedule(quotas):
        if len(selected[question_type]) >= quotas[question_type]:
            continue
        pool = candidates[question_type]
        row = next((item for item in pool if item["_paper_key"] not in used_keys), None)
        if row is None:
            row = min(pool, key=lambda item: (usage_count.get(item["_paper_key"], 0), pool.index(item)))
        selected[question_type].append(row)
        used_keys.add(row["_paper_key"])
        usage_count[row["_paper_key"]] = usage_count.get(row["_paper_key"], 0) + 1
    return quotas, selected


def _make_question(question_type: str, title: str) -> str:
    prompts = {
        "summary": f'What is the main idea of the paper "{title}"?',
        "authors": f'Who are the authors of the paper "{title}"?',
        "date": f'When was the paper "{title}" published?',
        "categories": f'Which subject areas or categories does the paper "{title}" cover?',
    }
    return prompts[question_type]


def build_test_set(df: pd.DataFrame, output_path) -> list[dict[str, Any]]:
    """Create and persist a balanced ten-question ground-truth evaluation set."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("build_test_set requires a pandas DataFrame.")

    required_columns = {"paper_id", "title", "summary", "published"}
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        raise ValueError(f"Cannot build the evaluation set; missing required columns: {', '.join(missing_columns)}.")
    if not ({"authors_joined", "authors"} & set(df.columns)):
        raise ValueError("Cannot build the evaluation set; provide 'authors_joined' or 'authors'.")
    if not ({"categories_joined", "primary_category", "categories"} & set(df.columns)):
        raise ValueError("Cannot build the evaluation set; provide categories_joined, primary_category, or categories.")

    documents = _normalized_documents(df)
    candidates = {
        question_type: [
            row for row in documents if row["ground_truths"][question_type]
        ]
        for question_type in _QUESTION_TYPES
    }
    unavailable = [question_type for question_type in _QUESTION_TYPES if not candidates[question_type]]
    if unavailable:
        counts = ", ".join(f"{kind}={len(candidates[kind])}" for kind in _QUESTION_TYPES)
        raise ValueError(
            f"Cannot build ten questions; no valid ground truth for {', '.join(unavailable)} ({counts})."
        )

    quotas, selected_documents = _choose_documents(candidates)
    if any(len(selected_documents[kind]) != quotas[kind] for kind in _QUESTION_TYPES):
        counts = ", ".join(f"{kind}={len(candidates[kind])}" for kind in _QUESTION_TYPES)
        raise ValueError(f"Cannot build ten questions from the valid dataframe rows ({counts}).")

    selected_offsets = {kind: 0 for kind in _QUESTION_TYPES}
    test_set: list[dict[str, Any]] = []
    for question_type in _question_schedule(quotas):
        row = selected_documents[question_type][selected_offsets[question_type]]
        selected_offsets[question_type] += 1
        ground_truth = row["ground_truths"][question_type]
        if not ground_truth:
            raise ValueError(f"Selected paper {row['paper_id']!r} has empty {question_type} ground truth.")
        test_set.append(
            {
                "id": f"eval_{len(test_set) + 1:03d}",
                "question_type": question_type,
                "question": _make_question(question_type, row["title"]),
                "ground_truth": ground_truth,
                "ground_truth_doc_ids": [row["paper_id"]],
            }
        )

    if len(test_set) != 10:
        raise ValueError(f"Expected to create ten evaluation questions, created {len(test_set)}.")

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(test_set, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return test_set


def assess_test_set_coverage(df: pd.DataFrame) -> dict[str, Any]:
    """Count real, non-empty ground-truth values available for each question type."""
    documents = _normalized_documents(df)
    counts = {
        kind: sum(bool(row["ground_truths"][kind]) for row in documents)
        for kind in _QUESTION_TYPES
    }
    minimum = 2
    missing = {kind: max(0, minimum - count) for kind, count in counts.items()}
    return {
        "valid_ground_truth_counts": counts,
        "minimum_per_type": minimum,
        "required_questions": 10,
        "status": "passed" if not any(missing.values()) else "failed",
        "missing_per_type": missing,
    }
