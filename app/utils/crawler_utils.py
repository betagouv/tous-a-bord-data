import logging
from typing import List, Optional

import nest_asyncio

# Initialize the event loop before importing crawl4ai
# flake8: noqa: E402
nest_asyncio.apply()

from crawl4ai import AsyncWebCrawler
from crawl4ai.async_configs import BrowserConfig, CrawlerRunConfig
from crawl4ai.deep_crawling import DFSDeepCrawlStrategy
from crawl4ai.deep_crawling.filters import FilterChain, URLPatternFilter
from crawl4ai.deep_crawling.scorers import KeywordRelevanceScorer


class CrawlerManager:
    """Crawler manager with initialization and reset."""

    def __init__(self):
        self.crawler: Optional[AsyncWebCrawler] = None
        self.logger = logging.getLogger("crawler_manager")

    async def init_crawler(self) -> AsyncWebCrawler:
        """Initialize the crawler if it doesn't already exist."""
        if self.crawler is None:
            browser_config = BrowserConfig(
                browser_type="chromium", headless=True, verbose=True
            )
            self.crawler = AsyncWebCrawler(config=browser_config)
        return self.crawler

    def _get_exclude_patterns(self):
        """Return the list of URL patterns to exclude."""
        return [
            "*/actualites*",
            "*/cookie*",
            "*/mentions-legales*",
            "*/confidentialite*",
            "*/donnees-personnelles*",
            "*/contact*",
            "*/faq*",
            "*/a-propos*",
            "*/plan-du-site*",
            "*/jegeremacartenavigo*",
            "*/actu*",
            "*/grands-comptes*",
            "*/images*",
            "*/correspondance*",
            "*/en/*",
            "*/es/*",
            "*/de/*",
            "*/it/*",
            "*/zh/*",
            "*/plan*",
            "*/info-trafic*",
            "*/horaire*",
            "*/itineraire*",
            "*/rss*",
            "*/sitemap*",
            "*/depart*",
            "*/arrivee*",
            "*/arret*",
            "*/ligne*",
            "*/nous*",
            "*/velo/*",
            "*/autopartage/*",
            "*/a-la-demande/*",
            "*/trotinette/*",
            "*/carte-interactive/*",
            "*login*",
            "*cookie*",
        ]

    def _should_exclude_url(self, url, patterns):
        """Check if the URL should be excluded according to the patterns."""
        return any(pattern in url.lower() for pattern in patterns)

    async def fetch_content(self, url: str, keywords: List[str]):
        """Fetch the content of a URL with deep crawling."""
        exclude_patterns = self._get_exclude_patterns()
        # Check if the starting URL should be excluded
        if self._should_exclude_url(url, exclude_patterns):
            self.logger.warning(f"Starting URL excluded by filters: {url}")
            return None
        # Create the exclusion filter
        exclude_filter = URLPatternFilter(
            patterns=exclude_patterns,
            reverse=True,
        )
        # Create the keyword scorer
        scorer = KeywordRelevanceScorer(keywords=keywords, weight=1000.0)

        scraping_strategy = DFSDeepCrawlStrategy(
            max_depth=5,
            max_pages=5,
            include_external=False,
            url_scorer=scorer,
            filter_chain=FilterChain([exclude_filter]),
            # score_threshold=0.1,
        )

        run_config = CrawlerRunConfig(
            deep_crawl_strategy=scraping_strategy,
            wait_until="domcontentloaded",
            # If True, auto-scroll the page to load dynamic content
            scan_full_page=True,
            wait_for_images=True,
            # Add delays to avoid detection
            mean_delay=1.0,
            # Additional pause (seconds) before final HTML is captured.
            delay_before_return_html=1.0,
            scroll_delay=1.0,
            # Simulate a user
            simulate_user=True,
            # Automatic handling of popups/consent banners. Experimental
            magic=True,
            # Override navigator properties in JS for stealth.
            override_navigator=True,
            # Removes potential modals/popups blocking the main content
            remove_overlay_elements=True,
            # Inlines iframe content for single-page extraction
            process_iframes=True,
            verbose=True,
        )

        try:
            crawler = await self.init_crawler()
            return await crawler.arun(url=url, config=run_config)
        except Exception as e:
            self.logger.warning(f"Erreur lors du crawling: {str(e)}")
            raise e
