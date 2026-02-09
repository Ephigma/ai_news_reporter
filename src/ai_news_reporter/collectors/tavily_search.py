"""Tavily API collector for news search."""

import asyncio
from datetime import datetime

from tavily import TavilyClient

from ..core.exceptions import CollectorError
from ..models.article import Article
from .base import BaseCollector
from .image_extractor import extract_og_image


class TavilyCollector(BaseCollector):
    """Collector that uses Tavily API for news search."""

    def __init__(self, api_key: str, fetch_images: bool = True):
        """Initialize Tavily collector.

        Args:
            api_key: Tavily API key.
            fetch_images: Whether to fetch og:image from each article.
        """
        if not api_key:
            raise CollectorError("Tavily API key is required")
        self._client = TavilyClient(api_key=api_key)
        self._fetch_images = fetch_images

    @property
    def name(self) -> str:
        return "Tavily Search"

    async def collect(
        self,
        query: str,
        time_range: str = "week",
        max_results: int = 10,
        include_domains: list[str] | None = None,
        exclude_domains: list[str] | None = None,
        **kwargs,
    ) -> list[Article]:
        """Search for news articles using Tavily API.

        Args:
            query: Search query.
            time_range: Time range for search (day, week, month, year).
            max_results: Maximum number of results.
            include_domains: Only include results from these domains.
            exclude_domains: Exclude results from these domains.

        Returns:
            List of articles found.
        """
        try:
            # Tavily search is synchronous, but we wrap it for async interface
            response = self._client.search(
                query=query,
                topic="news",
                days=self._time_range_to_days(time_range),
                max_results=max_results,
                include_domains=include_domains or [],
                exclude_domains=exclude_domains or [],
            )

            articles = []
            for result in response.get("results", []):
                article = Article(
                    title=result.get("title", "Untitled"),
                    url=result.get("url"),
                    content=result.get("content", ""),
                    source="Tavily Search",
                    image_url=None,  # Will be fetched below
                    published_at=self._parse_date(result.get("published_date")),
                    collected_at=datetime.now(),
                    keywords=[query],
                    score=result.get("score"),
                )
                articles.append(article)

            # Fetch og:image for each article in parallel
            if self._fetch_images and articles:
                image_tasks = [extract_og_image(str(article.url)) for article in articles]
                images = await asyncio.gather(*image_tasks, return_exceptions=True)
                for article, image in zip(articles, images):
                    if isinstance(image, str):
                        article.image_url = image

            return articles

        except Exception as e:
            raise CollectorError(f"Tavily search failed: {e}") from e

    def _time_range_to_days(self, time_range: str) -> int:
        """Convert time range string to days."""
        mapping = {
            "day": 1,
            "week": 7,
            "month": 30,
            "year": 365,
        }
        return mapping.get(time_range, 7)

    def _parse_date(self, date_str: str | None) -> datetime | None:
        """Parse date string to datetime."""
        if not date_str:
            return None
        try:
            return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        except ValueError:
            return None
