"""Tests for summary prompt building, truncation and the report.language setting."""

from ai_news_reporter.core.config import ReportConfig
from ai_news_reporter.llm.base import MAX_SUMMARY_CHARS, BaseLLM
from ai_news_reporter.llm.claude import DEFAULT_SUMMARY_PROMPT
from ai_news_reporter.models.article import Article


class StubLLM(BaseLLM):
    """Minimal concrete LLM for testing the shared helpers."""

    async def summarize(self, articles, prompt=None, focus="", language="English"):
        return ""

    @property
    def provider_name(self) -> str:
        return "Stub"


ARTICLES = [Article(title="Title", url="https://example.com", source="Example", content="Body")]


class TestBuildSummaryPrompt:
    """Tests for applying report.language to the summary prompt."""

    def test_default_prompt_includes_language_once(self):
        prompt = StubLLM()._build_summary_prompt(DEFAULT_SUMMARY_PROMPT, ARTICLES, "", "Japanese")
        assert prompt.count("Write in Japanese.") == 1

    def test_custom_prompt_without_placeholder_gets_language(self):
        template = "Summarize these:\n{focus}\n{articles}"
        prompt = StubLLM()._build_summary_prompt(template, ARTICLES, "", "Japanese")
        assert prompt.rstrip().endswith("Write in Japanese.")


class TestTruncateSummary:
    """Tests for cutting long summaries."""

    def test_short_text_is_unchanged(self):
        assert BaseLLM._truncate_summary("Short. Text.") == "Short. Text."

    def test_decimal_point_is_not_a_sentence_end(self):
        text = "Model 3.8 launched. " + "x" * MAX_SUMMARY_CHARS
        assert BaseLLM._truncate_summary(text) == "Model 3.8 launched."

    def test_japanese_full_stop_is_a_sentence_end(self):
        text = "新モデルが発表された。" + "あ" * MAX_SUMMARY_CHARS
        assert BaseLLM._truncate_summary(text) == "新モデルが発表された。"

    def test_period_inside_link_text_is_not_a_sentence_end(self):
        text = (
            "First sentence. "
            + "a" * (MAX_SUMMARY_CHARS - 40)
            + " [Acme Inc. raises $1B](https://example.com/acme) and more"
        )
        assert BaseLLM._truncate_summary(text) == "First sentence."

    def test_period_inside_nested_brackets_is_not_a_sentence_end(self):
        text = (
            "First sentence. "
            + "a" * (MAX_SUMMARY_CHARS - 40)
            + " [Foo [beta]. x](https://x.com/y) and more"
        )
        assert BaseLLM._truncate_summary(text) == "First sentence."

    def test_no_sentence_end_cuts_before_split_link(self):
        prefix = "a" * (MAX_SUMMARY_CHARS - 10)
        text = prefix + " [OpenAI](https://openai.com/index/some-long-path)"
        assert BaseLLM._truncate_summary(text) == prefix + "…"


class TestReportLanguageConfig:
    """Tests for the report.language setting."""

    def test_defaults_to_english(self):
        assert ReportConfig({}).language == "English"

    def test_blank_value_falls_back_to_english(self):
        assert ReportConfig({"language": None}).language == "English"
        assert ReportConfig({"language": ""}).language == "English"

    def test_explicit_value(self):
        assert ReportConfig({"language": "Japanese"}).language == "Japanese"
