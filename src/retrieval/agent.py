from __future__ import annotations

import re
from typing import Any

from langchain.agents import create_agent
from langchain.tools import tool

from core.config import Settings, normalized_provider
from retrieval.index import LocalEmbeddingIndex
from retrieval.llm import build_llm
from retrieval.qa import INSUFFICIENT_CONTEXT, _question_kind, answer_is_extractive, answer_question, validate_cited_answer


class _MockPaperCorpusAgent:
    """Deterministic corpus-backed implementation for offline QA checks."""

    def __init__(self, settings: Settings, index: LocalEmbeddingIndex):
        self.settings = settings
        self.index = index

    def invoke(self, payload: dict[str, Any]) -> dict[str, Any]:
        messages = payload.get("messages", [])
        question = str(messages[-1].get("content", "")) if messages else ""
        result = answer_question(question, self.settings, self.index)
        return {"messages": [{"content": result.answer}]}


def build_agent(settings: Settings, index: LocalEmbeddingIndex):
    if normalized_provider(settings) == "mock":
        return _MockPaperCorpusAgent(settings, index)

    @tool
    def semantic_search_papers(query: str, top_k: int = 4) -> str:
        """Search the local paper corpus with embeddings and return the most relevant papers."""
        results = index.search(query, top_k=top_k)
        lines = []
        for result in results:
            lines.append(
                f"paper_id: {result.paper_id}\n"
                f"title: {result.title}\n"
                f"source: {result.metadata.get('abs_url') or result.paper_id}\n"
                f"score: {result.score:.4f}\n"
                f"{result.content}"
            )
        return "\n\n".join(lines)

    @tool
    def lookup_paper(paper_id_or_title: str) -> str:
        """Look up a paper by exact paper_id or exact title from the local corpus."""
        record = index.lookup(paper_id_or_title)
        if not record:
            return "No exact paper match found."
        return (
            f"paper_id: {record['paper_id']}\n"
            f"title: {record['title']}\n"
            f"source: {record['metadata'].get('abs_url') or record['paper_id']}\n"
            f"{record['content']}"
        )

    llm = build_llm(settings=settings, temperature=0.0)
    return create_agent(
        model=llm,
        tools=[semantic_search_papers, lookup_paper],
        system_prompt=(
            "You answer questions about the indexed scholarly paper corpus sourced from Crossref. "
            "Use tools before answering factual questions. Base every claim only on returned paper context, "
            "cite supporting paper_id and title, and say the context is insufficient when it does not "
            "support an answer. Use citations exactly as [Source: paper_id | title]. Copy complete authors, "
            "publication dates, and categories from the tool result. For summaries and other questions, "
            "copy one exact text span from a cited tool result. Never guess or imply certainty without evidence."
        ),
        name="paper_corpus_agent",
    )


def run_agent_question(agent: Any, question: str) -> str:
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    messages = result.get("messages", [])
    if not messages:
        return INSUFFICIENT_CONTEXT
    final_message = messages[-1]
    content = final_message.get("content", "") if isinstance(final_message, dict) else getattr(final_message, "content", "")
    if isinstance(content, list):
        content = " ".join(
            str(block.get("text", "")) if isinstance(block, dict) else str(block)
            for block in content
        )
    answer = str(content)
    if isinstance(agent, _MockPaperCorpusAgent):
        return answer

    # LangChain agents can issue a final message without calling a tool. Accept
    # factual claims only when a tool actually returned the cited documents.
    sources: list[dict[str, str]] = []
    tool_records: dict[tuple[str, str], str] = {}
    for message in messages:
        if isinstance(message, dict):
            role = message.get("role") or message.get("type")
        else:
            role = getattr(message, "type", "")
        if role != "tool":
            continue
        tool_content = message.get("content", "") if isinstance(message, dict) else getattr(message, "content", "")
        tool_text = str(tool_content)
        found = list(re.finditer(r"(?m)^paper_id: ([^\r\n]+)\r?\ntitle: ([^\r\n]+)", tool_text))
        for position, match in enumerate(found):
            paper_id, title = match.groups()
            source = {"paper_id": paper_id, "title": title}
            if source not in sources:
                sources.append(source)
            end = found[position + 1].start() if position + 1 < len(found) else len(tool_text)
            tool_records[(paper_id, title)] = tool_text[match.start():end]
    validated, cited_sources = validate_cited_answer(answer, sources)
    quoted = re.search(r"['\"]([^'\"]+)['\"]", question)
    if quoted and cited_sources and not any(
        quoted.group(1) in (source["paper_id"], source["title"]) for source in cited_sources
    ):
        return INSUFFICIENT_CONTEXT
    kind = _question_kind(question, None)
    if cited_sources and kind in {"authors", "date", "categories"}:
        label = {"authors": "Authors", "date": "Published", "categories": "Categories"}[kind]
        if not any(
            (field := re.search(
                rf"(?m)^{label}: ([^\r\n]+)",
                tool_records.get((source["paper_id"], source["title"]), ""),
            )) and field.group(1).casefold() in validated.casefold()
            for source in cited_sources
        ):
            return INSUFFICIENT_CONTEXT
    elif cited_sources and not answer_is_extractive(
        validated,
        [tool_records.get((source["paper_id"], source["title"]), "") for source in cited_sources],
    ):
        return INSUFFICIENT_CONTEXT
    return validated
