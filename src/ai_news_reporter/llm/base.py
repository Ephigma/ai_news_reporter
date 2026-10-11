"""Abstract base class for LLM providers."""

import re
from abc import ABC, abstractmethod

from ..models.article import Article

MAX_SUMMARY_CHARS = 800
# A period followed by whitespace (so "3.8" is not a sentence end), or a Japanese full stop
SENTENCE_END = re.compile(r"[.!?](?=\s)|[。！？]")
MARKDOWN_LINK = re.compile(r"\[[^\]]*\]\([^)]*\)")


class BaseLLM(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    async def summarize(
        self,
        articles: list[Article],
        prompt: str | None = None,
        focus: str = "",
        language: str = "English",
    ) -> str:
        """Generate a summary from collected articles.

        Args:
            articles: List of articles to summarize.
            prompt: Optional custom prompt template.
            focus: Optional focus instructions for the report.
            language: Language to write in.

        Returns:
            Generated summary text.
        """
        pass

    @abstractmethod
    async def generate_report(
        self,
        articles: list[Article],
        title: str,
        prompt: str | None = None,
        highlight_count: int = 10,
        focus: str = "",
        language: str = "English",
    ) -> str:
        """Generate a full report from collected articles.

        Args:
            articles: List of articles to include in report.
            title: Report title.
            prompt: Optional custom prompt template.
            highlight_count: Number of articles to feature in Highlight News.
            focus: Optional focus instructions for the report.
            language: Language to write in.

        Returns:
            Generated report in markdown format.
        """
        pass

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Get the provider name."""
        pass

    def _format_articles_for_context(self, articles: list[Article]) -> str:
        """Format articles as context for LLM.

        Args:
            articles: List of articles.

        Returns:
            Formatted string of articles.
        """
        formatted = []
        for i, article in enumerate(articles, 1):
            image_line = ""
            if article.image_url:
                image_line = f"- **Image**: {article.image_url}\n"

            entry = f"""
### Article {i}: {article.title}
- **Source**: {article.source}
- **URL**: {article.url}
{image_line}- **Published**: {article.published_at or "Unknown"}

{article.content}
"""
            formatted.append(entry)
        return "\n".join(formatted)

    def _build_summary_prompt(
        self, template: str, articles: list[Article], focus: str, language: str
    ) -> str:
        """Fill the summary prompt template.

        A custom template may predate {language}; the language instruction is
        appended in that case so report.language still takes effect.
        """
        focus_text = f"\nFOCUS: {focus}\n" if focus else ""
        full_prompt = template.format(
            articles=self._format_articles_for_context(articles),
            focus=focus_text,
            language=language,
        )
        if "{language}" not in template:
            full_prompt += f"\nWrite in {language}.\n"
        return full_prompt

    @staticmethod
    def _truncate_summary(text: str) -> str:
        """Cut text longer than MAX_SUMMARY_CHARS without breaking markdown links.

        Prefers the last sentence end that is not inside a [text](url) link.
        Without one, cuts before any link that would be split and adds an ellipsis.
        """
        if len(text) <= MAX_SUMMARY_CHARS:
            return text

        links = [m.span() for m in MARKDOWN_LINK.finditer(text)]

        def in_link(pos: int) -> bool:
            return any(start <= pos < end for start, end in links)

        ends = [
            m.end()
            for m in SENTENCE_END.finditer(text[:MAX_SUMMARY_CHARS])
            if not in_link(m.start())
        ]
        if ends:
            return text[: ends[-1]].strip()

        cut = MAX_SUMMARY_CHARS
        for start, end in links:
            if start < cut < end:
                cut = start
        return text[:cut].rstrip() + "…"
