"""Image extraction utilities for articles."""

import httpx
from bs4 import BeautifulSoup


async def extract_og_image(url: str, timeout: int = 10) -> str | None:
    """Extract Open Graph image from a URL.

    Args:
        url: The article URL to fetch.
        timeout: Request timeout in seconds.

    Returns:
        The og:image URL if found, None otherwise.
    """
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
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

        # Try og:image first (most common)
        og_image = soup.find("meta", property="og:image")
        if og_image and og_image.get("content"):
            return og_image["content"]

        # Try twitter:image
        twitter_image = soup.find("meta", attrs={"name": "twitter:image"})
        if twitter_image and twitter_image.get("content"):
            return twitter_image["content"]

        # Try first large image in article
        for img in soup.find_all("img"):
            src = img.get("src") or img.get("data-src")
            if src and not _is_icon_or_logo(src):
                return src

        return None

    except Exception:
        return None


def _is_icon_or_logo(src: str) -> bool:
    """Check if image URL looks like an icon or logo."""
    src_lower = src.lower()
    return any(
        x in src_lower for x in ["icon", "logo", "avatar", "favicon", "sprite", "1x1", "pixel"]
    )
