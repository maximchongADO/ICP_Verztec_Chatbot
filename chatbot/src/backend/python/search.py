
from crawl4ai import *
from googlesearch import search
from urllib.parse import urlparse
def get_search_results(query: str, num_results: int = 10) -> list[dict]:
    """
    Google search → list of ≤ num_results dicts, with:
        • blocked_prefixes removed
        • only one result per hostname
    """
    blocked_prefixes = [
        "https://www.verztec.com",
        "https://x.com",
        "https://facebook.com"
        
        # add more here if needed
    ]

    # grab a large pool so filtering won’t run dry
    raw_iter = search(
        query,
        advanced=True,
        region="Singapore",
        unique=True,
        num_results=max(num_results * 5, 50),  # overshoot
    )

    results: list[dict] = []
    seen_hosts: set[str] = set()

    for r in raw_iter:
        # 1️⃣ block specific prefixes
        if any(r.url.startswith(pfx) for pfx in blocked_prefixes):
            continue

        # 2️⃣ limit to one link per host
        host = urlparse(r.url).hostname or ""
        if host in seen_hosts:
            continue
        seen_hosts.add(host)

        # 3️⃣ add cleaned result
        results.append({"url": r.url, "title": r.title, "description": r.description})

        if len(results) >= num_results:
            break

    return results


if __name__ == "__main__":
    x=get_search_results("verztec news latest 2025", 10)
    for i in x:
        print(f"🔗 {i['url']} - {i['title']}\n   {i['description']}")