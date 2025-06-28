
from crawl4ai import *
from googlesearch import search

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

if __name__ == "__main__":
    x=get_search_results("verztec news and projects 2025", 10)
    for i in x:
        print(f"🔗 {i['url']} - {i['title']}\n   {i['description']}")