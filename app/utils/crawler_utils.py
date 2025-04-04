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
        url_filter = URLPatternFilter(
            patterns=[
                "*boutique*",
                "*tarif*",
                "*abonnement*",
                "*ticket*",
                "*pass*",
                "*carte*",
                "*titre*",
            ]
        )

        scorer = KeywordRelevanceScorer(keywords=keywords, weight=1)

        scraping_strategy = BestFirstCrawlingStrategy(
            max_depth=2,
            max_pages=10,
            include_external=False,
            url_scorer=scorer,
            filter_chain=FilterChain([url_filter]),
        )

        run_config = CrawlerRunConfig(
            word_count_threshold=10,
            exclude_external_links=True,
            excluded_tags=[
                "form",
                "header",
                "footer",
                "nav",
                "aside",
                "trafic",
            ],
            remove_overlay_elements=True,
            process_iframes=True,
            deep_crawl_strategy=scraping_strategy,
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
