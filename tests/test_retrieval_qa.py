from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import patch

from core.config import normalized_provider
from retrieval import qa
from retrieval.agent import build_agent, run_agent_question
from retrieval.index import SearchResult
from retrieval.llm import build_llm


PAPER_ID = "10.1234/example"
TITLE = "Evidence Based Retrieval for Safety"
SUMMARY = "This paper evaluates retrieval for industrial safety. It reports a case study."


class CorpusIndex:
    def __init__(self, *, categories: str = "Information Retrieval"):
        self.document = {
            "paper_id": PAPER_ID,
            "title": TITLE,
            "content": (
                f"Title: {TITLE}\nAuthors: Ada Lovelace\nPublished: 2026-01-02\n"
                f"Categories: {categories}\nSummary: {SUMMARY}"
            ),
            "metadata": {
                "authors_joined": "Ada Lovelace",
                "published": "2026-01-02",
                "categories_joined": categories,
                "summary": SUMMARY,
            },
        }

    def lookup(self, value: str):
        return self.document if value == TITLE else None

    def search(self, query: str, top_k: int | None = None):
        return [SearchResult(**self.document, score=0.75)]


def settings(provider: str = "mock", key: str | None = None):
    return SimpleNamespace(llm_provider=provider, top_k=4, openai_api_key=key)


class RetrievalQATests(unittest.TestCase):
    def test_provider_router_and_credential_errors(self):
        cases = (
            ("google", "google_api_key", "GOOGLE_API_KEY", "ChatGoogleGenerativeAI"),
            ("gemini", "google_api_key", "GOOGLE_API_KEY", "ChatGoogleGenerativeAI"),
            ("openai", "openai_api_key", "OPENAI_API_KEY", "ChatOpenAI"),
            ("anthropic", "anthropic_api_key", "ANTHROPIC_API_KEY", "ChatAnthropic"),
        )
        for provider, field, key_name, client_class in cases:
            with self.subTest(provider=provider):
                config = SimpleNamespace(
                    llm_provider=provider,
                    model_name="test-model",
                    google_api_key=None,
                    openai_api_key=None,
                    anthropic_api_key=None,
                )
                with self.assertRaisesRegex(RuntimeError, key_name):
                    build_llm(config)
                setattr(config, field, "dummy-never-sent")
                client = object()
                with patch(f"retrieval.llm.{client_class}", return_value=client) as constructor:
                    self.assertIs(build_llm(config), client)
                self.assertEqual(constructor.call_args.kwargs["model"], "test-model")
        self.assertEqual(normalized_provider(SimpleNamespace(llm_provider="gemini")), "google")

    def test_mock_answers_only_supported_facts_with_source(self):
        index = CorpusIndex()
        question = f'Who are the authors of "{TITLE}"?'
        result = qa.answer_question(question, settings(), index)
        self.assertEqual(result.answer, f"Ada Lovelace [Source: {PAPER_ID} | {TITLE}]")
        self.assertEqual(result.sources, [{"paper_id": PAPER_ID, "title": TITLE}])
        self.assertEqual(run_agent_question(build_agent(settings(), index), question), result.answer)

        unsupported = qa.answer_question(f'What is the budget of "{TITLE}"?', settings(), index)
        self.assertEqual(unsupported.answer, qa.INSUFFICIENT_CONTEXT)
        self.assertEqual(unsupported.sources, [])
        unknown_title = qa.answer_question('Who authored "Evidence Based Retrieval for Space"?', settings(), index)
        self.assertEqual(unknown_title.answer, qa.INSUFFICIENT_CONTEXT)
        missing_category = qa.answer_question(
            f'What categories does "{TITLE}" have?', settings(), CorpusIndex(categories="")
        )
        self.assertEqual(missing_category.answer, qa.INSUFFICIENT_CONTEXT)

    def test_real_provider_answer_requires_retrieved_citation(self):
        class FakeLLM:
            def __init__(self, text: str):
                self.text = text

            def invoke(self, prompt: str):
                assert f"[Source: {PAPER_ID} | {TITLE}]" in prompt
                return SimpleNamespace(content=self.text)

        index = CorpusIndex()
        provider = settings("openai", "dummy-never-sent")
        question = f'Summarize "{TITLE}".'
        with patch("retrieval.qa.build_llm", return_value=FakeLLM("It reports a case study.")):
            self.assertEqual(qa.answer_question(question, provider, index).answer, qa.INSUFFICIENT_CONTEXT)

        cited = f"It reports a case study. [Source: {PAPER_ID} | {TITLE}]"
        with patch("retrieval.qa.build_llm", return_value=FakeLLM(cited)):
            result = qa.answer_question(question, provider, index)
        self.assertEqual(result.answer, cited)
        self.assertEqual(result.sources, [{"paper_id": PAPER_ID, "title": TITLE}])

        with patch("retrieval.qa.build_llm", return_value=FakeLLM("False claim. [Source: invented | Unknown]")):
            self.assertEqual(qa.answer_question(question, provider, index).answer, qa.INSUFFICIENT_CONTEXT)
        with patch("retrieval.qa.build_llm", return_value=FakeLLM(f"Unsupported claim. [Source: {PAPER_ID} | {TITLE}]")):
            self.assertEqual(qa.answer_question(question, provider, index).answer, qa.INSUFFICIENT_CONTEXT)

        date_question = f'When was "{TITLE}" published?'
        with patch("retrieval.qa.build_llm", return_value=FakeLLM(f"2025-01-01 [Source: {PAPER_ID} | {TITLE}]")):
            self.assertEqual(qa.answer_question(date_question, provider, index).answer, qa.INSUFFICIENT_CONTEXT)
        with patch("retrieval.qa.build_llm", return_value=FakeLLM(f"2026-01-02 [Source: {PAPER_ID} | {TITLE}]")):
            self.assertIn("2026-01-02", qa.answer_question(date_question, provider, index).answer)

    def test_missing_provider_key_fails_before_retrieval(self):
        with self.assertRaisesRegex(RuntimeError, "OPENAI_API_KEY"):
            qa.answer_question("Unrelated question", settings("openai"), CorpusIndex())

    def test_agent_requires_tool_output_and_valid_source(self):
        answer = f"Ada Lovelace wrote it. [Source: {PAPER_ID} | {TITLE}]"
        no_tool = SimpleNamespace(invoke=lambda _: {"messages": [{"role": "assistant", "content": answer}]})
        self.assertEqual(run_agent_question(no_tool, "Who wrote it?"), qa.INSUFFICIENT_CONTEXT)

        tool_message = {
            "role": "tool",
            "content": f"paper_id: {PAPER_ID}\ntitle: {TITLE}\nAuthors: Ada Lovelace\nSummary: {SUMMARY}",
        }
        with_tool = SimpleNamespace(invoke=lambda _: {"messages": [tool_message, {"role": "assistant", "content": answer}]})
        self.assertEqual(run_agent_question(with_tool, "Who wrote it?"), answer)

        wrong = f"Grace Hopper wrote it. [Source: {PAPER_ID} | {TITLE}]"
        wrong_author = SimpleNamespace(invoke=lambda _: {"messages": [tool_message, {"role": "assistant", "content": wrong}]})
        self.assertEqual(run_agent_question(wrong_author, "Who wrote it?"), qa.INSUFFICIENT_CONTEXT)
        self.assertEqual(
            run_agent_question(with_tool, 'Who wrote "Evidence Based Retrieval for Space"?'),
            qa.INSUFFICIENT_CONTEXT,
        )
        summary = f"This paper evaluates retrieval for industrial safety. [Source: {PAPER_ID} | {TITLE}]"
        summary_agent = SimpleNamespace(invoke=lambda _: {"messages": [tool_message, {"role": "assistant", "content": summary}]})
        self.assertEqual(run_agent_question(summary_agent, f'Summarize "{TITLE}".'), summary)
        unsupported = f"Unsupported claim. [Source: {PAPER_ID} | {TITLE}]"
        unsupported_agent = SimpleNamespace(invoke=lambda _: {"messages": [tool_message, {"role": "assistant", "content": unsupported}]})
        self.assertEqual(run_agent_question(unsupported_agent, f'Summarize "{TITLE}".'), qa.INSUFFICIENT_CONTEXT)


if __name__ == "__main__":
    unittest.main()
