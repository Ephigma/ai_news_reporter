"""OpenAI LLM implementation."""

import re

from openai import OpenAI

from ..core.exceptions import LLMError
from ..models.article import Article
from .base import BaseLLM

DEFAULT_SUMMARY_PROMPT = """You are a professional news analyst specializing in artificial intelligence and technology.

Write a brief 2-3 sentence executive summary of the key AI developments this week.

IMPORTANT RULES:
- Write ONLY 2-3 SHORT sentences. Be concise.
- NO headers, NO subsections, NO bullet points, NO numbered lists.
- Include inline source links using [Source Name](URL) format.
- Write in {language}.
- Example: "This week saw major advances in AI healthcare with [Utah's prescription AI initiative](url) and [OpenAI's Torch acquisition](url)."
{focus}
Here are the articles to analyze:

{articles}
"""


class OpenAILLM(BaseLLM):
    """OpenAI LLM provider."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4.1-mini",
        max_tokens: int = 4096,
        temperature: float = 0.3,
    ):
        """Initialize OpenAI LLM.

        Args:
            api_key: OpenAI API key.
            model: Model to use.
            max_tokens: Maximum tokens in response.
            temperature: Sampling temperature.
        """
        if not api_key:
            raise LLMError("OpenAI API key is required")

        self._client = OpenAI(api_key=api_key)
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature

    @property
    def provider_name(self) -> str:
        return "OpenAI"

    async def summarize(
        self,
        articles: list[Article],
        prompt: str | None = None,
        focus: str = "",
        language: str = "English",
    ) -> str:
        """Generate a summary using OpenAI.

        Args:
            articles: List of articles to summarize.
            prompt: Optional custom prompt.
            focus: Optional focus instructions for the report.
            language: Language to write in.

        Returns:
            Generated summary.
        """
        if not articles:
            return "No articles to summarize."

        full_prompt = self._build_summary_prompt(
            prompt or DEFAULT_SUMMARY_PROMPT, articles, focus, language
        )

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
                messages=[{"role": "user", "content": full_prompt}],
            )
            summary = response.choices[0].message.content or ""
            return self._clean_summary(summary)

        except Exception as e:
            raise LLMError(f"OpenAI API error: {e}") from e

    def _clean_summary(self, summary: str) -> str:
        """Remove markdown headers, images, lists, and extra formatting from summary.

        Keep only the first paragraph (2-3 sentences max).
        """
        lines = summary.split("\n")
        cleaned_lines = []
        for line in lines:
            # Skip lines that are markdown headers
            if re.match(r"^#{1,6}\s", line):
                continue
            # Skip lines that are just "Executive Summary" or similar titles
            if re.match(
                r"^\*{0,2}(Executive Summary|Key Developments|Industry Impact|Trends to Watch|Weekly.*Report)\*{0,2}\s*$",
                line,
                re.IGNORECASE,
            ):
                continue
            # Skip image lines
            if re.match(r"^!\[", line):
                continue
            # Skip numbered list items (1. **Title**: ...)
            if re.match(r"^\d+\.\s+\*\*", line):
                continue
            # Skip lines that are just source references
            if re.match(r"^\*Source:", line):
                continue
            cleaned_lines.append(line)

        # Join and remove multiple consecutive blank lines
        result = "\n".join(cleaned_lines)
        result = re.sub(r"\n{3,}", "\n\n", result)
        result = result.strip()

        # Keep only first paragraph (should be 2-3 sentences)
        paragraphs = [p.strip() for p in result.split("\n\n") if p.strip()]
        if paragraphs:
            # Return only the first substantial paragraph, cut at a sentence end if too long
            return self._truncate_summary(paragraphs[0])
        return result
