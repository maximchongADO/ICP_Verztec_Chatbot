import asyncio
from crawl4ai import *

async def main():
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(
            url="https://www.giving.sg/organisation/profile/ee0cacf2-668c-4344-b995-e767fa3ff779",
        )
        print(result.markdown)
       

async def main():
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(
            url="https://www.verztec.com/"
        )

        if not result.success:
            print("❌ Crawl failed:", result.error_message)
            return

        print("🌐 URL:", result.url)
        print("🔧 Cleaned HTML (first 500 chars):")
        print((result.cleaned_html or "")[:500] + "...\n")

        # Handle markdown output
        if isinstance(result.markdown, str):
            md_text = result.markdown
        else:
            md_text = result.markdown.raw_markdown

        print("📘 Markdown (first 500 chars):")
        print(md_text[:500] + "...\n")

        print("🧠 Structured Content (if any):")
        print(result.extracted_content or "None")

        print("🧾 Metadata keys:", list(result.metadata.keys()) if result.metadata else "None")

if __name__ == "__main__":
    asyncio.run(main())