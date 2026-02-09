"""Web scraper for collecting articles from specific sites."""

from datetime import datetime
from urllib.parse import urljoin

import httpx
from bs4 import BeautifulSoup

from ..core.config import SiteConfig
from ..core.exceptions import CollectorError
from ..models.article import Article
from .base import BaseCollector


class WebScraper(BaseCollector):
    """Collector that scrapes configured websites."""

    def __init__(self, timeout: int = 30):
        """Initialize web scraper.

        Args:
            timeout: Request timeout in seconds.
        """
        self._timeout = timeout

    @property
    def name(self) -> str:
        return "Web Scraper"

    async def collect(
        self,
        query: str,
        site_config: SiteConfig | None = None,
        **kwargs,
    ) -> list[Article]:
        """Scrape articles from a website.

        Args:
            query: URL to scrape (or site name if site_config provided).
            site_config: Site configuration with selectors.

        Returns:
            List of scraped articles.
        """
        if site_config:
            url = site_config.url
            selectors = site_config.selectors
            source_name = site_config.name
        else:
            url = query
            selectors = kwargs.get("selectors", {})
            source_name = kwargs.get("source_name", "Web")

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.get(
                    url,
                    headers={
                        "User-Agent": (
                            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                            "AppleWebKit/537.36 (KHTML, like Gecko) "
                            "Chrome/120.0.0.0 Safari/537.36"
                        )
                    },
                    follow_redirects=True,
                )
                response.raise_for_status()

            soup = BeautifulSoup(response.text, "lxml")
            articles = self._extract_articles(soup, url, selectors, source_name)
            return articles

        except httpx.HTTPError as e:
            raise CollectorError(f"Failed to fetch {url}: {e}") from e
        except Exception as e:
            raise CollectorError(f"Scraping failed for {url}: {e}") from e

    def _extract_articles(
        self,
        soup: BeautifulSoup,
        base_url: str,
        selectors: dict[str, str],
        source_name: str,
    ) -> list[Article]:
        """Extract articles from parsed HTML.

        Args:
            soup: Parsed HTML.
            base_url: Base URL for resolving relative links.
            selectors: CSS selectors for article elements.
            source_name: Name of the source site.

        Returns:
            List of extracted articles.
        """
        articles = []
        article_selector = selectors.get("article", "article")
        title_selector = selectors.get("title", "h2 a, h3 a")
        link_selector = selectors.get("link", "a")
        content_selector = selectors.get("content", "p")
        date_selector = selectors.get("date", "time")
        image_selector = selectors.get("image", "img")

        article_elements = soup.select(article_selector)

        for element in article_elements:
            try:
                # Extract title
                title_elem = element.select_one(title_selector)
                if not title_elem:
                    continue
                title = title_elem.get_text(strip=True)

                # Extract link
                link_elem = element.select_one(link_selector)
                if link_elem and link_elem.get("href"):
                    url = urljoin(base_url, link_elem["href"])
                else:
                    continue

                # Extract content/description
                content_elem = element.select_one(content_selector)
                content = content_elem.get_text(strip=True) if content_elem else ""

                # Extract date
                date_elem = element.select_one(date_selector)
                published_at = None
                if date_elem:
                    datetime_attr = date_elem.get("datetime")
                    if datetime_attr:
                        try:
                            published_at = datetime.fromisoformat(
                                datetime_attr.replace("Z", "+00:00")
                            )
                        except ValueError:
                            pass

                # Extract image
                image_url = None
                img_elem = element.select_one(image_selector)
                if img_elem:
                    # Try different image attributes
                    image_src = (
                        img_elem.get("src")
                        or img_elem.get("data-src")
                        or img_elem.get("data-lazy-src")
                    )
                    if image_src:
                        image_url = urljoin(base_url, image_src)

                article = Article(
                    title=title,
                    url=url,
                    content=content,
                    source=source_name,
                    image_url=image_url,
                    published_at=published_at,
                    collected_at=datetime.now(),
                )
                articles.append(article)

            except Exception:
                # Skip malformed articles
                continue

        return articles

    async def collect_from_sites(self, sites: list[SiteConfig]) -> list[Article]:
        """Collect articles from multiple configured sites.

        Args:
            sites: List of site configurations.

        Returns:
            Combined list of articles from all sites.
        """
        all_articles = []
        for site in sites:
            if not site.enabled:
                continue
            try:
                articles = await self.collect(site.url, site_config=site)
                all_articles.extend(articles)
            except CollectorError:
                # Log error but continue with other sites
                continue
        return all_articles
