import requests
from bs4 import BeautifulSoup
import re
import numpy as np               # For array handling
import faiss                     # Facebook AI Similarity Search library
from sentence_transformers import SentenceTransformer

def get_search_results(query, num_results=10):
    api_key = 'AIzaSyC-peG7i7Wz8j68MRdnvMRLKnokl7WlNnM'
    cx = 'b30ae95baf130429d'  # You will need to create a custom search engine
    search_url = f'https://www.googleapis.com/customsearch/v1?q={query}&key={api_key}&cx={cx}'
    response = requests.get(search_url)
    search_results = response.json()
    
    urls = []
    for item in search_results.get('items', []):
        urls.append(item['link'])
    
    return urls


def update_verztec_index():
    # 1. Define a list of source URLs (news articles, blog posts, press releases) 
    urls = [
        "https://www.portfoliomagsg.com/article/a-personal-touch.html",   # Magazine article (Profile)
        "https://nvpc.org.sg/articles/verztec-selects-charity-partners-that-share-its-focus/",  # Org blog/news
        "https://www.prweb.com/releases/verztec-teams-up-with-hoopis-performance-network-hpn-in-financial-online-learning-for-asia-866377590.html"  # Press release
    ]

    # Placeholder: list to hold extracted article data
    articles = []
    for url in urls:
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract title (e.g., from <h1> or <title>)
        title_tag = soup.find('h1')
        title = title_tag.get_text().strip() if title_tag else (soup.title.string if soup.title else "")

        # Extract publication date if available (try <time> tag or common text patterns)
        date = ""
        time_tag = soup.find('time')
        if time_tag:
            date = time_tag.get_text()
        else:
            # Fallback: search for a date pattern (e.g., "28 August 2017" or "Sep 1 2021") in text
            text = soup.get_text()
            match = re.search(r'\d{1,2}\s+\w+\s+\d{4}', text)
            if match:
                date = match.group(0)

        # Extract main content text by collecting all paragraph texts
        content_container = soup.find('article') or soup.find('div', {'class': 'post-content'}) or soup
        paragraphs = [p.get_text() for p in content_container.find_all('p')]
        article_text = "\n\n".join([p.strip() for p in paragraphs if p.strip()])

        articles.append({
            'title': title,
            'date': date,
            'url': url,
            'text': article_text
        })

    # 2. Chunk each article's text into smaller segments for embedding
    chunks = []
    chunk_metadata = []  # parallel list for metadata corresponding to each chunk
    max_words = 300
    overlap_words = 50
    for article in articles:
        words = article['text'].split()
        # Slide a window over the word list with the defined overlap
        start = 0
        while start < len(words):
            end = min(start + max_words, len(words))
            chunk_words = words[start:end]
            chunk_text = " ".join(chunk_words)
            # Save chunk text and its metadata (title, URL, date)
            chunks.append(chunk_text)
            chunk_metadata.append({
                'title': article['title'],
                'url': article['url'],
                'date': article['date']
            })
            if end == len(words):
                break  # reached end of article text
            start += max_words - overlap_words

    # 3. Embed each chunk using a pre-trained transformer model (e.g., MiniLM)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(chunks)  # Compute vector embeddings for all chunks

    # 4. Create a FAISS index and add all embeddings to it
    embedding_dim = len(embeddings[0])
    index = faiss.IndexFlatL2(embedding_dim)       # start with a basic L2-distance index
    index = faiss.IndexIDMap(index)               # wrap it to allow mapping back to IDs
    ids = np.arange(len(embeddings))
    index.add_with_ids(np.array(embeddings).astype('float32'), ids)

    # (Optional) Save the index to a file for reuse in future searches
    faiss.write_index(index, 'verztec_index.faiss')

    return index, chunk_metadata


if __name__ =='__main__':
    results=get_search_results('Verztec news 2025 OR Verztec updates OR Verztec partnership', 10)
    print(results)
    
    