from typing import List, Optional
from urllib.parse import urljoin

from crawl4ai import AsyncWebCrawler
from crawl4ai.async_configs import BrowserConfig, CrawlerRunConfig
from crawl4ai.deep_crawling import BestFirstCrawlingStrategy
from crawl4ai.deep_crawling.filters import FilterChain, URLPatternFilter
from crawl4ai.deep_crawling.scorers import KeywordRelevanceScorer

# help(CrawlerRunConfig)


async def image_error_handler(base_url, error, page=None):
    error_msg = str(error)
    if "Error opening" in (error_msg) and error_msg.startswith(
        "Error opening '/"
    ):
        try:
            relative_path = error_msg.split("Error opening '")[1].split("'")[0]
            if relative_path.startswith("/"):
                corrected_url = urljoin(base_url, relative_path)
                print(
                    f"🔄 Correction d'URL d'image: "
                    f"{relative_path} → {corrected_url}"
                )
                if page:
                    await page.evaluate(
                        f"""
                        (function() {{
                            var imgs = document.querySelectorAll(
                                'img[src="{relative_path}"]'
                            );
                            for(var i=0; i<imgs.length; i++) {{
                                imgs[i].src = "{corrected_url}";
                            }}
                        }})()
                    """
                    )
                return True
        except Exception as correction_error:
            print(f"⚠️ Échec de correction d'URL: {correction_error}")
    if any(
        ext in error_msg
        for ext in [".png", ".jpg", ".jpeg", ".svg", ".gif", ".webp"]
    ):
        print("ℹ️ Ignoré: Erreur d'image non corrigible")
        return True
    return False


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
                "*grands-comptes*",
                "*images*",
                "*correspondance*",
                "*/en/*",
                "*/es/*",
                "*/de/*",
                "*/it/*",
                "*/zh/*",
            ],
            reverse=True,
        )

        scorer = KeywordRelevanceScorer(keywords=keywords, weight=10)

        scraping_strategy = BestFirstCrawlingStrategy(
            max_depth=4,
            max_pages=20,
            include_external=False,
            url_scorer=scorer,
            filter_chain=FilterChain([exclude_filter]),
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
            wait_until="domcontentloaded",
            # If True, auto-scroll the page to load dynamic content
            scan_full_page=True,
            wait_for_images=True,
            # Add delays to avoid detection
            mean_delay=3.0,
            # Additional pause (seconds) before final HTML is captured.
            delay_before_return_html=3.0,
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
            if self.crawler:
                await self.crawler.stop()
                self.crawler = None
                if self.on_crawler_reset:
                    self.on_crawler_reset()
            raise e
