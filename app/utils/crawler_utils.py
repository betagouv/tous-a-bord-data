import logging
import sys
from typing import List, Optional

from crawl4ai import AsyncWebCrawler
from crawl4ai.async_configs import BrowserConfig, CrawlerRunConfig
from crawl4ai.deep_crawling import DFSDeepCrawlStrategy
from crawl4ai.deep_crawling.filters import FilterChain, URLPatternFilter
from crawl4ai.deep_crawling.scorers import KeywordRelevanceScorer


class DebugDFSStrategy(DFSDeepCrawlStrategy):
    """Strategy for DFS with detailed logging for debugging."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = self._setup_logger()

    def _setup_logger(self):
        """Configure and return a logger for debugging."""
        logger = logging.getLogger("link_discovery_debug")
        logger.setLevel(logging.DEBUG)
        # Avoid duplicate handlers
        if not logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

    async def link_discovery(
        self, result, source_url, current_depth, visited, next_links, depths
    ):
        """Override the link discovery method with logging."""
        self.logger.debug(
            f"Link discovery for {source_url} at depth {current_depth}"
        )
        self.logger.debug(f"Already visited links: {visited}")
        await super().link_discovery(
            result, source_url, current_depth, visited, next_links, depths
        )
        self.logger.debug(f"After discovery - new links: {next_links}")
        self.logger.debug("-------------------")


class CrawlerManager:
    """Crawler manager with initialization and reset."""

    def __init__(self, on_crawler_reset=None):
        self.crawler: Optional[AsyncWebCrawler] = None
        self.on_crawler_reset = on_crawler_reset
        self.logger = logging.getLogger("crawler_manager")

    async def init_crawler(self) -> AsyncWebCrawler:
        """Initialize the crawler if it doesn't already exist."""
        if self.crawler is None:
            browser_config = BrowserConfig(
                browser_type="chromium", headless=True, verbose=True
            )
            self.crawler = AsyncWebCrawler(config=browser_config)
            await self.crawler.start()
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
            max_depth=4,
            max_pages=10,
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
            mean_delay=2.0,
            # Additional pause (seconds) before final HTML is captured.
            delay_before_return_html=5.0,
            scroll_delay=2.0,
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
            if self.crawler:
                await self.crawler.stop()
                self.crawler = None
                if self.on_crawler_reset:
                    self.on_crawler_reset()
            raise
