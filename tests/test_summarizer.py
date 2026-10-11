"""Tests for report assembly in Summarizer."""

import pytest

from ai_news_reporter.llm.base import BaseLLM
from ai_news_reporter.models.article import Article
from ai_news_reporter.processors.summarizer import Summarizer


class CountingLLM(BaseLLM):
    def __init__(self):
        self.calls = 0
        self.language = None

    @property
    def provider_name(self) -> str:
        return "Fake"

    async def summarize(self, articles, prompt=None, focus="", language="English") -> str:
        self.calls += 1
        self.language = language
        return f"summary #{self.calls}"


@pytest.mark.asyncio
async def test_summary_generated_once_and_reused_in_report():
    llm = CountingLLM()
    articles = [
        Article(title=f"A{i}", url=f"https://example.com/{i}", source="src", content="body")
        for i in range(3)
    ]

    report = await Summarizer(llm).generate_report(articles, highlight_count=2, language="Japanese")

    assert llm.calls == 1
    assert llm.language == "Japanese"
    assert report.summary == "summary #1"
    assert "summary #1" in report.content_markdown
    assert "**2.2.** [A1]" in report.content_markdown
    assert "**3.1.** [A2]" in report.content_markdown
