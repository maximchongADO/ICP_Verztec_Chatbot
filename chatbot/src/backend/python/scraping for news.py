
from googlesearch import search
import asyncio
from crawl4ai import *
import asyncio
from typing import List, Union
import faiss
from pathlib import Path
import asyncio
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from crawl4ai import LLMConfig
from crawl4ai.extraction_strategy import (
    LLMExtractionStrategy)
from crawl4ai.content_filter_strategy import PruningContentFilter
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS     
import json
embedding_model = SentenceTransformer('BAAI/bge-large-en-v1.5')
browser_cfg = BrowserConfig(
    headless=False,  # opens a visible browser window so you can see page loads
    verbose=True,     # enables detailed logging from Playwright & the crawler
    text_mode=True,
    user_agent_mode="random"
    
    
  
)




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





# 2️⃣  build an LLM extraction strategy that calls Groq
deepseek_strategy = LLMExtractionStrategy(
    input_format="fit_markdown",            # better than raw HTML
    instruction=(
    "Extract structured information about the company, grouped by category. "
    "Return one JSON object with top-level keys like 'Vision', 'Brand Promise', 'Corporate Values', etc., "
    "and under each, provide a list of relevant text items."
    "Return only valid JSON."
    ),
    llm_config=LLMConfig(
        provider="ollama/llama3",        # ← model name inside provider
        base_url="http://localhost:11434"   # default Ollama endpoint
                    # optional
        # api_token is NOT needed for local Ollama
    ),
)

filter_chain = FilterChain([
    DomainFilter(blocked_domains=["linkedin.com"]),
    # or: URLPatternFilter(patterns=["*linkedin*"])
])
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy   # breadth-first search
deep_crawl = BFSDeepCrawlStrategy(
    max_depth=0,           # 0 = only start URL, 1 = +direct links, 2 = +links-of-links
    include_external=True, # stay on the same domain
    max_pages=10,        # optional overall cap
    filter_chain=filter_chain

    
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
    url_strings = [
    (u["url"] if isinstance(u, dict) else u)
    for u in urls
    if "linkedin.com" not in (u["url"] if isinstance(u, dict) else u)
    ]
    results_out: list[dict] = []

    async with AsyncWebCrawler(config=browser_config or BrowserConfig(headless=True)) as crawler:
        cfg = CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,
            deep_crawl_strategy = deep_crawl,  
            extraction_strategy=strategy,
            markdown_generator=DefaultMarkdownGenerator(
                content_filter=PruningContentFilter()
            ),
        )

        tasks = [crawler.arun(url=u, config=cfg) for u in url_strings]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for url, res in zip(url_strings, results):
            # ── 1. Task-level failures ───────────────────────────────────────────
            if isinstance(res, Exception):
                print(f"⚠️  crawl failed: {url} → {res}")
                results_out.append(
                    {"url": url, "success": False, "markdown": "", "extracted": None}
                )
                continue
            

            # ── 2. res might be a list (deep-crawl) or a single CrawlResult ──────
            crawl_results = res if isinstance(res, list) else [res]

            for sub in crawl_results:
                md_text = sub.markdown.raw_markdown if sub.markdown else ""
                extracted_json = sub.extracted_content if sub.extracted_content else {}

                results_out.append(
                    {
                        "url": sub.url,          # sub-page’s own URL
                        "success": bool(extracted_json),
                        "markdown": md_text,
                        "extracted": extracted_json,
                    }
                )
    return results_out

async def main():
    # 1️⃣ get URL list (strings or dicts)
    urls = get_search_results("What does verztec do", num_results=1)
    for i in urls:
        print(f"🔗 {i['url']} - {i['title']}\n   {i['description']}")

    # 2️⃣ run crawler with DeepSeek strategy
    results = await crawl_with_strategy(urls, deepseek_strategy,browser_cfg)

    # 3️⃣ print results
    for r in results:
        print(f"\n🔍 {r['url']}")
        if r["success"]:
            print("✅ Extracted:\n", r["extracted"])
            
        else:
            print("❌ Failed.")
            
            
            
dimension = 384  # embedding dimension for MiniLM
index = faiss.IndexFlatIP(dimension)  # for cosine, use normalized vectors

# Store metadata alongside
doc_metadata = []
async def main():
    # 1️⃣ Seed URLs
    seeds = get_search_results("What does verztec do", num_results=1)
    urls  = [s["url"] for s in seeds]
    print("🔗 Seeds:", urls)

    # 2️⃣ Crawl & extract
    crawl_results = await crawl_with_strategy(urls, deepseek_strategy, browser_cfg)

    # 3️⃣ Build LangChain docs
    docs: list[Document] = []
    for r in crawl_results:
        if not r["success"]:
            continue

        extracted = r["extracted"]               # could be str, dict, or list

        # ── a. String → parse JSON
        if isinstance(extracted, str):
            try:
                extracted = json.loads(extracted)
            except json.JSONDecodeError:
                print("⚠️  Plain string, not JSON — skipping")
                continue

        # ── b. Dict   → convert to list of blocks
        if isinstance(extracted, dict):
            extracted = [
                {"tag": k, "content": v, "error": False}
                for k, v in extracted.items()
                if isinstance(v, list)
            ]

        # ── c. Guard: must be list now
        if not isinstance(extracted, list):
            print(f"⚠️  Unsupported type {type(extracted)} — skipping")
            continue

        # ── d. Iterate blocks
        for block in extracted:
            if block.get("error"):
                continue
            tag = block.get("tag", "Unknown").strip()
            for txt in block.get("content", []):
                docs.append(
                    Document(
                        page_content=f"[{tag}] {txt}",
                        metadata={"url": r["url"], "tag": tag},
                    )
                )

    if not docs:
        print("⚠️  No valid passages found; nothing to embed.")
        return

    print(f"✅ Loaded {len(docs)} passages for embedding")

    # 4️⃣ Embed & create FAISS store
    hf_embed = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en-v1.5",
        encode_kwargs={"normalize_embeddings": True},
    )
    vecstore = FAISS.from_documents(docs, hf_embed)

    # 5️⃣ Save to disk
    out_dir = Path("chatbot/src/backend/python/faiss_GK_index")
    out_dir.mkdir(parents=True, exist_ok=True)
    vecstore.save_local(str(out_dir))
    print(f"📦 Vector store saved to {out_dir} ({vecstore.index.ntotal} vectors)")

    
if __name__ == "__main__":
    
    asyncio.run(main())