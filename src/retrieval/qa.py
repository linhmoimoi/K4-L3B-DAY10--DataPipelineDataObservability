from __future__ import annotations

from dataclasses import dataclass
import re

from core.config import Settings
from core.utils import first_sentence
from retrieval.index import LocalEmbeddingIndex, SearchResult


@dataclass(frozen=True)
class AnswerResult:
    question: str
    answer: str
    retrieved_doc_ids: list[str]
    retrieved_contexts: list[str]
    retrieved_titles: list[str]


def _extract_answer(question: str, top_result: SearchResult, question_type: str | None = None) -> str:
    lowered = question.lower()
    metadata = top_result.metadata
    kind = question_type
    if kind is None:
        if any(term in lowered for term in ("author", "authored")):
            kind = "authors"
        elif any(term in lowered for term in ("when was", "publication date", "published on")):
            kind = "date"
        elif any(term in lowered for term in ("categor", "subject area", "field of study")):
            kind = "categories"
        else:
            kind = "summary"
    if kind == "authors":
        return str(metadata.get("authors_joined") or "Author metadata is unavailable for this paper.")
    if kind == "date":
        return str(metadata.get("published") or "Publication date metadata is unavailable for this paper.")
    if kind == "categories":
        return str(metadata.get("categories_joined") or "Category metadata is unavailable for this paper.")
    if kind == "summary":
        summary = metadata.get("summary")
        return first_sentence(str(summary)) if summary else "Summary metadata is unavailable for this paper."
    raise ValueError(f"Unsupported question_type: {question_type}")


def answer_question(
    question: str,
    settings: Settings,
    index: LocalEmbeddingIndex,
    top_k: int | None = None,
    question_type: str | None = None,
) -> AnswerResult:
    title_match = re.search(r"['\"]([^'\"]+)['\"]", question)
    exact = index.lookup(title_match.group(1)) if title_match else None
    retrieved = index.search(question, top_k=top_k)
    if exact:
        exact_result = SearchResult(
            paper_id=exact["paper_id"],
            title=exact["title"],
            score=1.0,
            content=exact["content"],
            metadata=exact["metadata"],
        )
        deduped = [exact_result] + [item for item in retrieved if item.paper_id != exact_result.paper_id]
        retrieved = deduped[: (top_k or settings.top_k)]
    if not retrieved:
        answer = "I don't know from the indexed corpus."
    else:
        answer = _extract_answer(question, retrieved[0], question_type)
    return AnswerResult(
        question=question,
        answer=answer,
        retrieved_doc_ids=[item.paper_id for item in retrieved],
        retrieved_contexts=[item.content for item in retrieved],
        retrieved_titles=[item.title for item in retrieved],
    )
