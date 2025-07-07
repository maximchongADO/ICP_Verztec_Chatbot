import asyncio
from crawl4ai import *
from crawl4ai import LLMConfig
from crawl4ai.extraction_strategy import LLMExtractionStrategy
import json

# Simplified instruction for testing
test_strategy = LLMExtractionStrategy(
    input_format="fit_markdown",
    instruction = (
        "Extract key information from this webpage content. "
        "Return a simple JSON object with company name, services, and description. "
        "Example: {\"company\": \"CompanyName\", \"services\": [\"service1\", \"service2\"], \"description\": \"what they do\"}"
    ),
    llm_config=LLMConfig(
        provider="ollama/llama3",
        base_url="http://localhost:11434"
    ),
    extra_args={"temperature": 0.3},
    apply_chunking=False
)

async def test_single_url():
    browser_cfg = BrowserConfig(headless=True, verbose=True)
    
    async with AsyncWebCrawler(config=browser_cfg) as crawler:
        # Test with Verztec homepage
        result = await crawler.arun(
            url="https://www.verztec.com",
            config=CrawlerRunConfig(
                extraction_strategy=test_strategy,
                cache_mode=CacheMode.BYPASS
            )
        )
        
        print(f"URL: {result.url}")
        print(f"Success: {result.success}")
        
        if result.markdown:
            print(f"Markdown length: {len(result.markdown.raw_markdown)}")
            print(f"Markdown preview: {result.markdown.raw_markdown[:500]}...")
        else:
            print("No markdown content!")
            
        print(f"Extracted content: {result.extracted_content}")
        
        if result.extracted_content:
            try:
                parsed = json.loads(result.extracted_content) if isinstance(result.extracted_content, str) else result.extracted_content
                print(f"Parsed extraction: {parsed}")
            except:
                print("Failed to parse extracted content as JSON")

if __name__ == "__main__":
    asyncio.run(test_single_url())
