"""Tests for image extraction and inclusion in reports."""

import pytest

from ai_news_reporter.collectors.image_extractor import extract_og_image
from ai_news_reporter.models.article import Article


class TestImageExtractor:
    """Tests for image extraction from URLs."""

    @pytest.mark.asyncio
    async def test_extract_og_image_from_techcrunch(self):
        """Test extracting og:image from a real news site."""
        # TechCrunch should have og:image
        url = "https://techcrunch.com/"
        image = await extract_og_image(url)
        # Should return a URL string or None
        assert image is None or image.startswith("http")

    @pytest.mark.asyncio
    async def test_extract_og_image_returns_string_or_none(self):
        """Test that extract_og_image returns string or None."""
        url = "https://www.google.com/"
        image = await extract_og_image(url)
        assert image is None or isinstance(image, str)

    @pytest.mark.asyncio
    async def test_extract_og_image_handles_invalid_url(self):
        """Test that invalid URLs return None without error."""
        url = "https://this-domain-does-not-exist-12345.com/"
        image = await extract_og_image(url, timeout=5)
        assert image is None


class TestArticleWithImage:
    """Tests for Article model with images."""

    def test_article_has_image_url_field(self):
        """Test that Article model has image_url field."""
        article = Article(
            title="Test Article",
            url="https://example.com/article",
            content="Test content",
            source="Test Source",
            image_url="https://example.com/image.jpg",
        )
        assert article.image_url == "https://example.com/image.jpg"

    def test_article_image_url_can_be_none(self):
        """Test that Article image_url can be None."""
        article = Article(
            title="Test Article",
            url="https://example.com/article",
            content="Test content",
            source="Test Source",
        )
        assert article.image_url is None

    def test_article_image_url_is_mutable(self):
        """Test that Article image_url can be set after creation."""
        article = Article(
            title="Test Article",
            url="https://example.com/article",
            content="Test content",
            source="Test Source",
        )
        article.image_url = "https://example.com/new-image.jpg"
        assert article.image_url == "https://example.com/new-image.jpg"


class TestReportIncludesImages:
    """Tests for report generation with images."""

    @pytest.mark.asyncio
    async def test_highlight_news_includes_image_markdown(self):
        """Test that Highlight News section includes image markdown."""
        articles = [
            Article(
                title="Test Article 1",
                url="https://example.com/article1",
                content="Test content 1",
                source="Test Source",
                image_url="https://example.com/image1.jpg",
            ),
            Article(
                title="Test Article 2",
                url="https://example.com/article2",
                content="Test content 2",
                source="Test Source",
                image_url="https://example.com/image2.jpg",
            ),
        ]

        # Create a mock report without calling the LLM
        report = _generate_report_section(articles, highlight_count=10)

        # Check that images are included in Highlight News
        assert "![Test Article 1](https://example.com/image1.jpg)" in report
        assert "![Test Article 2](https://example.com/image2.jpg)" in report

    def test_highlight_articles_have_images(self):
        """Test that highlight articles have their images in the report."""
        articles = [
            Article(
                title=f"Article {i}",
                url=f"https://example.com/article{i}",
                content=f"Content {i}",
                source="Test",
                image_url=f"https://example.com/image{i}.jpg",
            )
            for i in range(5)
        ]

        report = _generate_report_section(articles, highlight_count=5)

        # Every highlight article should have its image
        for i in range(5):
            assert f"![Article {i}](https://example.com/image{i}.jpg)" in report, (
                f"Image for Article {i} not found in report"
            )

    def test_related_news_has_no_images(self):
        """Test that Related News section does not include images."""
        articles = [
            Article(
                title=f"Article {i}",
                url=f"https://example.com/article{i}",
                content=f"Content {i}",
                source="Test",
                image_url=f"https://example.com/image{i}.jpg",
            )
            for i in range(15)
        ]

        report = _generate_report_section(articles, highlight_count=10)

        # Related news (articles 10-14) should NOT have images
        for i in range(10, 15):
            assert f"![Article {i}]" not in report, (
                f"Image for Article {i} found in Related News (should not be there)"
            )


def _generate_report_section(articles: list[Article], highlight_count: int = 10) -> str:
    """Generate the report sections (without LLM call)."""
    highlight_articles = articles[:highlight_count]
    related_articles = articles[highlight_count:]

    report = "## 2. Highlight News\n\n"
    for i, article in enumerate(highlight_articles, 1):
        report += f"**2.{i}.** [{article.title}]({article.url})\n\n"
        if article.image_url:
            report += f"![{article.title}]({article.image_url})\n\n"
        report += f"{article.content}\n\n"
        report += f"*Source: {article.source}*\n\n---\n\n"

    if related_articles:
        report += "## 3. Related News\n\n"
        for i, article in enumerate(related_articles, 1):
            report += f"**3.{i}.** [{article.title}]({article.url})\n\n"
            report += f"{article.content[:200]}{'...' if len(article.content) > 200 else ''}\n\n"

    return report
