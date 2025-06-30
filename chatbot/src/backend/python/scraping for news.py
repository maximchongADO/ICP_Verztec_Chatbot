
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

from search import get_search_results

embedding_model = SentenceTransformer('BAAI/bge-large-en-v1.5')
browser_cfg = BrowserConfig(
    headless=False,  # opens a visible browser window so you can see page loads
    verbose=True,     # enables detailed logging from Playwright & the crawler
    text_mode=False,
    user_agent_mode="random",
    
)






# 2️⃣  build an LLM extraction strategy that calls Groq
deepseek_strategy = LLMExtractionStrategy(
    input_format="fit_markdown",            # better than raw HTML
    instruction = (
    "You are given well-formed Markdown taken from a web page. "
    "Read the entire content carefully, then extract **detailed, retrieval-ready facts** "
    "about the company, organised by category. "

    "• Produce **one JSON object only**.  \n"
    "• Each **top-level key** must be a clear category name—e.g. "
    "\"Vision\", \"Brand Promise\", \"Corporate Values\", \"Services Offered\", "
    "\"Notable Projects\", \"Industry Focus\", \"Testimonials\", etc.  \n"
    "• Under every key, return a **list of richly written text items**. Each item should be "
    "two-to-four full sentences that:  \n"
    "  – give precise facts and context (dates, metrics, client names, outcomes, locations)  \n"
    "  – are self-contained so they can be fed directly into a retrieval-augmented generation (RAG) system.  \n"
    "• Aim for **comprehensive coverage**: capture all distinct services, values, case studies, awards, and any other salient information. "
    "It is better to be verbose and exhaustive than brief.  \n"
    "• Do **not** invent information. Paraphrase faithfully.  \n"
    "• Output must be valid JSON—no comments, no trailing commas, no markdown."
),
    llm_config=LLMConfig(
        provider="ollama/llama3",        # ← model name inside provider
        base_url="http://localhost:11434"   # default Ollama endpoint
                    # optional
        # api_token is NOT needed for local Ollama
    ),
    extra_args={
        "temperature": 0.7
    },
    apply_chunking=False

)

filter_chain = FilterChain([
    DomainFilter(blocked_domains=["linkedin.com"]),
    # or: URLPatternFilter(patterns=["*linkedin*"])
])
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy   # breadth-first search
deep_crawl = BFSDeepCrawlStrategy(
    max_depth=0,   # 0 = only start URL, 1 = +direct links, 2 = +links-of-links
    include_external=False, # stay on the same domain      # optional overall cap
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

# Store metadata alongside
doc_metadata = []
async def main():
    # 1️⃣ Seed URLs
    seeds = get_search_results("What does verztec do", num_results=5)
    urls  = [s["url"] for s in seeds]
    urls.append("https://www.verztec.com")  # add a known page
    print("🔗 Seeds:", urls)

    # 2️⃣ Crawl & extract
    crawl_results = await crawl_with_strategy(urls, deepseek_strategy, browser_cfg)

    # 3️⃣ Build LangChain docs
    seen_texts: set[str] = set()

    # 3️⃣ Build LangChain docs
    docs: list[Document] = []
    for r in crawl_results:
        if not r["success"]:
            continue

        extracted = r["extracted"]
        if r["success"]:
            print("✅ Extracted:\n", r["extracted"])

        # a) JSON-string → dict/list
        if isinstance(extracted, str):
            try:
                extracted = json.loads(extracted)
            except json.JSONDecodeError:
                continue

        # b) dict → list-of-blocks
        if isinstance(extracted, dict):
            extracted = [
                {"tag": k, "content": v, "error": False}
                for k, v in extracted.items() if isinstance(v, list)
            ]

        # c) guard
        if not isinstance(extracted, list):
            continue

        # d) iterate blocks
        for block in extracted:
            # ✨ NORMALISE block ➜ dict shape ------------------------------------
            if isinstance(block, list):
                # unknown tag; wrap as a dict
                block = {"tag": "Unknown", "content": block, "error": False}
            elif not isinstance(block, dict):
                # anything else we can’t handle → skip
                continue

            if block.get("error"):
                continue

            tag = str(block.get("tag", "Unknown")).strip()

            for txt in block.get("content", []):
                tagged_txt = f"[{tag}] {txt}"

                if tagged_txt in seen_texts:
                    continue
                seen_texts.add(tagged_txt)

                docs.append(
                    Document(
                        page_content=tagged_txt,
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