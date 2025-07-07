
from googlesearch import search
import asyncio
from crawl4ai import *
import asyncio
from typing import List, Union
import faiss
from pathlib import Path
import asyncio
from langchain_huggingface import HuggingFaceEmbeddings
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
    "Extract key information from this webpage about the company. "
    "Return a JSON object with these categories as keys, each containing a list of relevant text snippets: "
    "\"Company_Info\", \"Services\", \"Projects\", \"News\", \"Testimonials\", \"Contact\". "
    "Keep each text snippet clear and informative. "
    "Example: {\"Company_Info\": [\"Founded in 2000\", \"Global leader in translation\"], \"Services\": [\"Translation services\", \"eLearning solutions\"]}"
),
    llm_config=LLMConfig(
        provider="ollama/llama3",        # ← model name inside provider
        base_url="http://localhost:11434"   # default Ollama endpoint
                    # optional
        # api_token is NOT needed for local Ollama
    ),
    extra_args={
        "temperature": 0.3  # Lower temperature for more consistent results
    },
    apply_chunking=False

)

filter_chain = FilterChain([
    #DomainFilter(blocked_domains=["linkedin.com"]),
    URLPatternFilter(patterns=["*linkedin*"])
])

# Note: Deep crawl strategies are now created dynamically per URL
# - Verztec sites: depth=1 (crawl main page + direct links)  
# - Other sites: depth=0 (main page only) 


async def crawl_with_strategy(
    urls: List[Union[str, dict]],
    strategy,
    browser_config: BrowserConfig | None = None,
) -> List[dict]:
    """
    Crawl many URLs concurrently with *any* extraction strategy.
    Uses depth=1 for Verztec website, depth=0 for all other sites.

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
        # Create tasks with different crawl strategies based on URL
        tasks = []
        for url in url_strings:
            # Check if it's a Verztec website
            is_verztec = "verztec.com" in url.lower()
            
            # Create appropriate deep crawl strategy
            if is_verztec:
                # Deep crawl for Verztec sites (depth=1)
                url_deep_crawl = BFSDeepCrawlStrategy(
                    max_depth=1,
                    include_external=False,
                    filter_chain=filter_chain
                )
            else:
                # Shallow crawl for other sites (depth=0)
                url_deep_crawl = BFSDeepCrawlStrategy(
                    max_depth=0,
                    include_external=False,
                    filter_chain=filter_chain
                )
            
            cfg = CrawlerRunConfig(
                cache_mode=CacheMode.BYPASS,
                deep_crawl_strategy=url_deep_crawl,
                extraction_strategy=strategy,
                markdown_generator=DefaultMarkdownGenerator(
                    content_filter=PruningContentFilter()
                ),
            )
            
            tasks.append(crawler.arun(url=url, config=cfg))
            print(f"🔗 Crawling {url} with depth={'1 (Verztec)' if is_verztec else '0 (External)'}")

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
    seeds = get_search_results("What does verztec do", num_results=10)
    urls  = [s["url"] for s in seeds]
    urls.append("https://www.verztec.com")  # add a known page
    print("🔗 Seeds:", urls)

    # 2️⃣ Crawl & extract
    crawl_results = await crawl_with_strategy(urls, deepseek_strategy, browser_cfg)
    
    # Debug: Count results by success/failure
    total_results = len(crawl_results)
    successful_results = len([r for r in crawl_results if r["success"]])
    empty_results = len([r for r in crawl_results if not r["extracted"] or r["extracted"] == []])
    
    print(f"📊 Crawl Summary:")
    print(f"   Total pages crawled: {total_results}")
    print(f"   Successful extractions: {successful_results}")
    print(f"   Empty extractions: {empty_results}")
    print(f"   Failed extractions: {total_results - successful_results}")

    # 3️⃣ Build LangChain docs
    seen_texts: set[str] = set()
    duplicate_count = 0

    # 3️⃣ Build LangChain docs
    docs: list[Document] = []
    for r in crawl_results:
        if not r["success"]:
            continue

        extracted = r["extracted"]
        
        # Skip completely empty extractions early
        if not extracted or extracted == []:
            print(f"⚠️ Skipping URL with empty extraction: {r['url']}")
            continue
            
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

            # Enhanced error filtering - check multiple error indicators
            if (block.get("error") or 
                block.get("error") is True or
                "error" in block.get("tags", []) or
                (isinstance(block.get("tag"), str) and block.get("tag", "").lower() in ["error", "errors"])):
                print(f"⚠️ Skipping error block: {block}")
                continue

            # Get tag from either 'tag' field or first item in 'tags' array
            tag = block.get("tag", "")
            if isinstance(tag, list):
                tag = tag[0] if tag else ""
            tag = str(tag).strip()
            
            if not tag and block.get("tags"):
                tags_list = block.get("tags", [])
                tag = str(tags_list[0] if tags_list else "").strip()
            if not tag:
                tag = "General"  # Default fallback tag
            
            # Additional check for error-related tags
            if tag.lower() in ["error", "errors"]:
                print(f"⚠️ Skipping error tag block: {tag}")
                continue

            content_list = block.get("content", [])
            
            # Skip blocks with empty content
            if not content_list:
                print(f"⚠️ Skipping empty content block for tag: {tag}")
                continue

            for txt in content_list:
                # Skip empty or error-related content
                if not txt or (isinstance(txt, dict) and txt.get("error")):
                    continue
                
                # Skip low-quality content (cookies, spam, etc.)
                txt_lower = str(txt).lower()
                if any(skip_phrase in txt_lower for skip_phrase in [
                    "cookie preferences", "we use cookies", "manage consent",
                    "government officials will never ask", "scamshield helpline",
                    "your browser does not support", "loading content",
                    "sorry, we're having trouble"
                ]):
                    continue
                    
                # Include URL in the content for better context
                tagged_txt = f"[{tag}] {txt}\n\nSource: {r['url']}"

                if tagged_txt in seen_texts:
                    duplicate_count += 1
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

    print(f"✅ Final Statistics:")
    print(f"   Total passages created: {len(docs)}")
    print(f"   Duplicates filtered out: {duplicate_count}")
    print(f"   Unique passages for embedding: {len(docs)}")

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