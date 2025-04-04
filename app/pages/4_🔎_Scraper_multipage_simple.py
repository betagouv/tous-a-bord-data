import asyncio

import nest_asyncio
import streamlit as st

# Initialize the event loop before importing crawl4ai
# flake8: noqa: E402
nest_asyncio.apply()
from crawl4ai import AsyncWebCrawler
from crawl4ai.async_configs import BrowserConfig

# Init crawler
if "crawler" not in st.session_state:
    st.session_state.crawler = None
    st.session_state.loop = asyncio.new_event_loop()
    asyncio.set_event_loop(st.session_state.loop)


async def init_crawler():
    if st.session_state.crawler is None:
        browser_config = BrowserConfig(
            browser_type="chromium", headless=True, verbose=True
        )
        crawler = AsyncWebCrawler(config=browser_config)
        await crawler.start()
        st.session_state.crawler = crawler
    return st.session_state.crawler


async def fetch_content():
    try:
        crawler = await init_crawler()
        result = await crawler.arun(
            url="https://www.nbcnews.com/business",
        )
        return result
    except Exception as e:
        # If error, reset the crawler
        if st.session_state.crawler:
            await st.session_state.crawler.stop()
            st.session_state.crawler = None
        raise e


if st.button("Lancer le crawl"):
    try:
        loop = st.session_state.loop
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(fetch_content())
        st.markdown(result.markdown)
    except Exception as e:
        st.error(f"Erreur : {str(e)}")
