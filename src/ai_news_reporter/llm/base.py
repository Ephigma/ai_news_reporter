"""Abstract base class for LLM providers."""

from abc import ABC, abstractmethod

from ..models.article import Article


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
