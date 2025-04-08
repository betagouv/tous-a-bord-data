from typing import List, Optional

from crawl4ai import AsyncWebCrawler
from crawl4ai.async_configs import BrowserConfig, CrawlerRunConfig
from crawl4ai.deep_crawling import BestFirstCrawlingStrategy
from crawl4ai.deep_crawling.filters import FilterChain, URLPatternFilter
from crawl4ai.deep_crawling.scorers import KeywordRelevanceScorer


class CrawlerManager:
    def __init__(self, on_crawler_reset=None):
        self.crawler: Optional[AsyncWebCrawler] = None
        self.on_crawler_reset = on_crawler_reset

    async def init_crawler(self) -> AsyncWebCrawler:
        if self.crawler is None:
            browser_config = BrowserConfig(
                browser_type="chromium", headless=True, verbose=True
            )
            self.crawler = AsyncWebCrawler(config=browser_config)
            await self.crawler.start()
        return self.crawler

    async def fetch_content(self, url: str, keywords: List[str]):
        import logging
        import sys

        # blacklist of url patterns
        exclude_filter = URLPatternFilter(
            patterns=[
                "*actualites*",
                "*cookie*",
                "*mentions-legales*",
                "*confidentialite*",
                "*donnees-personnelles*",
                "*contact*",
                "*faq*",
                "*a-propos*",
                "*me-deplacer*",
                "*se-deplacer*",
                "*reseau*",
                "*plan-du-site*",
                "*jegeremacartenavigo*",
                "*actu*",
                "*grands-comptes*",
                "*images*",
                "*correspondance*",
                "*/en/*",
                "*/es/*",
                "*/de/*",
                "*/it/*",
                "*/zh/*",
                "*plan*",
                "*info-trafic*",
                "*horaire*",
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
            ],
            reverse=True,
        )
        # Configuration du logger
        logger = logging.getLogger("link_discovery_debug")
        logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        scorer = KeywordRelevanceScorer(keywords=keywords, weight=10000.0)

        class DebugBestFirstStrategy(BestFirstCrawlingStrategy):
            async def link_discovery(
                self,
                result,
                source_url,
                current_depth,
                visited,
                next_links,
                depths,
            ):
                logger.debug(
                    f"Link discovery pour {source_url}",
                    f"à profondeur {current_depth}",
                )
                logger.debug(f"Liens déjà visités: {visited}")
                logger.debug(f"Prochains liens à visiter: {next_links}")
                logger.debug(f"Profondeurs actuelles: {depths}")
                await super().link_discovery(
                    result,
                    source_url,
                    current_depth,
                    visited,
                    next_links,
                    depths,
                )
                logger.debug(f"Après discovery - nouveaux liens: {next_links}")
                logger.debug("-------------------")

        scraping_strategy = DebugBestFirstStrategy(
            max_depth=4,
            max_pages=20,
            include_external=False,
            url_scorer=scorer,
            filter_chain=FilterChain([exclude_filter]),
            logger=logger,
        )
        # js_code = """
        # function clickElements() {
        #     var elements = document."""
        # js_code += """
        #     querySelectorAll('.accordion, button, [aria-expanded="false"]');
        #     for (var i = 0; i < elements.length; i++) {
        #         try { elements[i].click(); } catch(e) {}
        #     }
        #     return elements.length;
        #     }
        #     clickElements();
        # """
        run_config = CrawlerRunConfig(
            deep_crawl_strategy=scraping_strategy,
            # Wait until all network calls are finished
            # wait_until="networkidle",
            # Wait until the DOM is fully loaded
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
            # JavaScript to run after load (click on menu)
            # js_code=js_code,
            verbose=True,
        )

        try:
            crawler = await self.init_crawler()
            return await crawler.arun(url=url, config=run_config)
        except Exception as e:
            logging.warning(f"Erreur lors du crawling: {str(e)}")
            if self.crawler:
                await self.crawler.stop()
                self.crawler = None
                if self.on_crawler_reset:
                    self.on_crawler_reset()
            raise
