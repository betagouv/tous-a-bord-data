from typing import List, Optional

from crawl4ai import AsyncWebCrawler
from crawl4ai.async_configs import BrowserConfig, CrawlerRunConfig
from crawl4ai.deep_crawling import BestFirstCrawlingStrategy
from crawl4ai.deep_crawling.filters import FilterChain, URLPatternFilter
from crawl4ai.deep_crawling.scorers import KeywordRelevanceScorer

# help(CrawlerRunConfig)


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
            ],
            reverse=True,
        )

        scorer = KeywordRelevanceScorer(keywords=keywords, weight=2)

        scraping_strategy = BestFirstCrawlingStrategy(
            max_depth=6,
            max_pages=10,
            include_external=False,
            url_scorer=scorer,
            filter_chain=FilterChain([exclude_filter]),
        )
        js_code = """
        function clickElements() {
            var elements = document."""
        js_code += """
            querySelectorAll('.accordion, button, [aria-expanded="false"]');
            for (var i = 0; i < elements.length; i++) {
                try { elements[i].click(); } catch(e) {}
            }
            return elements.length;
            }
            clickElements();
        """
        run_config = CrawlerRunConfig(
            deep_crawl_strategy=scraping_strategy,
            # Key parameters for the JavaScript
            # Wait until all network calls are finished
            wait_until="networkidle",
            # 90 seconds timeout
            page_timeout=5000,
            # Wait 2s before returning the HTML
            delay_before_return_html=2.0,
            # Scan the full page
            scan_full_page=True,
            # Simulate a user
            simulate_user=True,
            # Remove overlays
            remove_overlay_elements=True,
            # Scroll delay
            scroll_delay=1.0,
            # Process iframes
            process_iframes=True,
            js_code=js_code,
            verbose=True,
        )

        try:
            crawler = await self.init_crawler()
            return await crawler.arun(url=url, config=run_config)
        except Exception as e:
            if self.crawler:
                await self.crawler.stop()
                self.crawler = None
                if self.on_crawler_reset:
                    self.on_crawler_reset()
            raise e
