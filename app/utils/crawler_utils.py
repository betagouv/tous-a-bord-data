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
        # Filtre pour les URLs à inclure - ajout de patterns plus spécifiques
        include_filter = URLPatternFilter(
            patterns=[f"*{k}*" for k in keywords]
        )

        # Filtre pour les URLs à exclure
        exclude_filter = URLPatternFilter(
            patterns=[
                "*cookie*",
                "*mentions-legales*",
                "*confidentialite*",
                "*donnees-personnelles*",
            ],
            reverse=True,
        )

        scorer = KeywordRelevanceScorer(keywords=keywords, weight=0.5)

        scraping_strategy = BestFirstCrawlingStrategy(
            max_depth=3,
            max_pages=20,
            include_external=False,
            url_scorer=scorer,
            filter_chain=FilterChain([include_filter, exclude_filter]),
        )

        run_config = CrawlerRunConfig(
            word_count_threshold=0,
            wait_until="networkidle",
            page_timeout=60000,
            scan_full_page=True,
            process_iframes=True,
            remove_overlay_elements=True,
            simulate_user=True,
            magic=True,
            deep_crawl_strategy=scraping_strategy,
            semaphore_count=5,
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
