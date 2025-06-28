
from googlesearch import search
import asyncio
from crawl4ai import *
import asyncio
from typing import List, Union

import asyncio

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from crawl4ai import LLMConfig
from crawl4ai.extraction_strategy import (
    LLMExtractionStrategy)
from crawl4ai.content_filter_strategy import PruningContentFilter
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator

def get_search_results(query: str, num_results: int = 10) -> list[dict]:
    """
    Return up to `num_results` Google search results as a list of dicts.

    Each dict has keys: 'url', 'title', 'description'.
    """
    results_iter = search(
        query,
        advanced=True,         # include title + snippet
        region="Singapore",    # Google SG domain
        unique=True,           # skip duplicates
        num_results=num_results
    )

    data = [
        {"url": r.url, "title": r.title, "description": r.description}
        for r in results_iter
    ][:num_results]
    

    return data

async def crawl(results: list[dict | str]) -> list[str]:
    # normalize to strings
    url_strings = [r["url"] if isinstance(r, dict) else r for r in results]

    async with AsyncWebCrawler() as crawler:
        tasks   = [crawler.arun(url=u) for u in url_strings]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        markdown_pages = []
        for url, res in zip(url_strings, results):
            if isinstance(res, Exception):
                print(f"⚠️  crawl failed: {url} → {res}")
                markdown_pages.append("")
            else:
                markdown_pages.append(res.markdown)
        return markdown_pages

api_key = ''

model = "deepseek-r1-distill-llama-70b" 




# 2️⃣  build an LLM extraction strategy that calls Groq
deepseek_strategy = LLMExtractionStrategy(
    input_format="fit_markdown",            # better than raw HTML
    instruction=(
        "Extract structured information about the company: "
        "services, industry focus, notable clients, partnerships, "
        "and recent announcements. Return **only** JSON."
    ),
    llm_config=LLMConfig(
        provider="ollama/llama3",        # ← model name inside provider
        base_url="http://localhost:11434"   # default Ollama endpoint
                    # optional
        # api_token is NOT needed for local Ollama
    ),
)


async def crawl_with_strategy(
    urls: List[Union[str, dict]],
    strategy,
    browser_config: BrowserConfig | None = None,
) -> List[dict]:
    """
    Crawl many URLs concurrently with *any* extraction strategy.

    Returns one dict per URL:
        { "url": <url>, "success": bool, "markdown": str, "extracted": Any }
    """
    # Normalize to plain strings
    url_strings = [u["url"] if isinstance(u, dict) else u for u in urls]

    results_out: list[dict] = []

    async with AsyncWebCrawler(config=browser_config or BrowserConfig(headless=True)) as crawler:
        cfg = CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,
            extraction_strategy=strategy,
            markdown_generator=DefaultMarkdownGenerator(
                content_filter=PruningContentFilter()
            ),
        )

        tasks = [crawler.arun(url=u, config=cfg) for u in url_strings]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for url, res in zip(url_strings, results):
            if isinstance(res, Exception):
                print(f"⚠️  crawl failed: {url} → {res}")
                results_out.append(
                    {"url": url, "success": False, "markdown": "", "extracted": None}
                )
            else:
                results_out.append(
                    {
                        "url": url,
                        "success": True,
                        "markdown": res.markdown.raw_markdown,
                        "extracted": res.extracted_content,  # JSON from the strategy
                    }
                )
    return results_out

 
if __name__ == "__main__":
    async def main():
        # 1️⃣ get URL list (strings or dicts)
        urls = get_search_results("What does verztec do", num_results=2)
        for i in urls:
            print(f"🔗 {i['url']} - {i['title']}\n   {i['description']}")

        # 2️⃣ run crawler with DeepSeek strategy
        results = await crawl_with_strategy(urls, deepseek_strategy)

        # 3️⃣ print results
        for r in results:
            print(f"\n🔍 {r['url']}")
            if r["success"]:
                print("✅ Extracted:\n", r["extracted"])
            else:
                print("❌ Failed.")

    asyncio.run(main())