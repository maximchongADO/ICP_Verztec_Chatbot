import requests
from bs4 import BeautifulSoup
import re
import numpy as np               # For array handling
import faiss                     # Facebook AI Similarity Search library
from sentence_transformers import SentenceTransformer
from collections import Counter
import json
def get_search_results(query, num_results=10):
    api_key = 'AIzaSyC-peG7i7Wz8j68MRdnvMRLKnokl7WlNnM'  # Your Google API key
    cx = 'b30ae95baf130429d'  # Your Custom Search Engine ID (CX)
    
    # Prepare to collect all URLs
    all_urls = []
    
    # Google API allows up to 10 results per page; to get more results, we need to handle pagination
    start_index = 1
    while len(all_urls) < num_results:
        # Build search URL with the start index for pagination
        search_url = f'https://www.googleapis.com/customsearch/v1?q={query}&key={api_key}&cx={cx}&start={start_index}'
        
        response = requests.get(search_url)
        search_results = response.json()
        
        # Extract the URLs from the search results
        for item in search_results.get('items', []):
            all_urls.append(item['link'])
        
        # If fewer results than the requested number are returned, break out of the loop
        if len(search_results.get('items', [])) < 10:
            break
        
        # Update the start index for the next page of results
        start_index += 10
    
    # Return only the requested number of URLs
    return all_urls[:num_results]

def update_verztec_index():
    # 1. Define a list of source URLs (news articles, blog posts, press releases) 
    urls= get_search_results('Verztec news 2025 OR Verztec updates OR Verztec partnership')
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
            #print(chunk_text)
            chunk_metadata.append({
                'title': article['title'],
                'url': article['url'],
                'date': article['date']
            })
            if end == len(words):
                break  # reached end of article text
            start += max_words - overlap_words
            
    with open("verztec_chunks.txt", "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(chunk.replace("\n", " ") + "\n")


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

def update_verztec_index():
    urls = get_search_results('Verztec news 2025 OR Verztec updates OR Verztec partnership')
    articles = []

    for url in urls:
        try:
            response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=5)
            soup = BeautifulSoup(response.text, 'html.parser')
            

            # Title
            title_tag = soup.find('h1')
            print(title)
            title = title_tag.get_text().strip() if title_tag else (soup.title.string if soup.title else "")
            print(title)

            # Date
            date = ""
            time_tag = soup.find('time')
            if time_tag:
                date = time_tag.get_text()
            else:
                match = re.search(r'\d{1,2}\s+\w+\s+\d{4}', soup.get_text())
                if match:
                    date = match.group(0)

            # Content
            content_container = soup.find('article') or soup.find('div', {'class': 'post-content'}) or soup
            paragraphs = [p.get_text() for p in content_container.find_all('p')]
            article_text = "\n\n".join([p.strip() for p in paragraphs if p.strip()])
            print(article_text)

            # Filter out junk
            if len(article_text.split()) < 100:
                print(f"Skipped (too short): {url}")
                continue
            if title.lower() in ["news & events", "verztec", ""]:
                print(f"Skipped (bad title): {url}")
                continue

            articles.append({
                'title': title,
                'date': date,
                'url': url,
                'text': article_text
            })

        except Exception as e:
            print(f"Error scraping {url}: {e}")
            continue

    # Chunking
    chunks = []
    chunk_metadata = []
    max_words = 300
    overlap_words = 50

    for article in articles:
        words = article['text'].split()
        start = 0
        while start < len(words):
            end = min(start + max_words, len(words))
            chunk_words = words[start:end]
            chunk_text = " ".join(chunk_words)

            # Clean weak chunks
            if len(chunk_text.split()) < 30:
                start += max_words - overlap_words
                continue
            if chunk_text.lower() in ["news & events", "verztec", article['title'].lower()]:
                start += max_words - overlap_words
                continue

            chunks.append(chunk_text)
            chunk_metadata.append({
                'title': article['title'],
                'url': article['url'],
                'date': article['date']
            })

            if end == len(words):
                break
            start += max_words - overlap_words

    # Deduplicate overused chunks
    chunk_counts = Counter(chunks)
    cleaned_chunks, cleaned_metadata = [], []
    for c, m in zip(chunks, chunk_metadata):
        if chunk_counts[c] > 3:
            continue
        cleaned_chunks.append(c)
        cleaned_metadata.append(m)

    chunks = cleaned_chunks
    chunk_metadata = cleaned_metadata

    # Save chunks to file
    with open("verztec_chunks.txt", "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(chunk.replace("\n", " ") + "\n")

    # Save metadata
    with open("verztec_metadata.json", "w", encoding="utf-8") as f:
        json.dump(chunk_metadata, f, indent=2)

    # Embedding
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(chunks)

    # FAISS indexing
    embedding_dim = len(embeddings[0])
    index = faiss.IndexFlatL2(embedding_dim)
    index = faiss.IndexIDMap(index)
    ids = np.arange(len(embeddings))
    index.add_with_ids(np.array(embeddings).astype('float32'), ids)

    faiss.write_index(index, 'verztec_index.faiss')
    return index, chunk_metadata

 
def idk():
    urls = get_search_results('Verztec news 2025 OR Verztec updates OR Verztec partnership')
    articles = []

    for url in urls:
        try:
            print("--------------------------------------------------------------------------------------------------------------------------------------")
            response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=5)
            soup = BeautifulSoup(response.text, 'html.parser')
            

            # Title
            title_tag = soup.find('h1')
            title = title_tag.get_text().strip() if title_tag else (soup.title.string if soup.title else "")

            # Date
            date = ""
            time_tag = soup.find('time')
            if time_tag:
                date = time_tag.get_text()
                print('Date found: '+date)
            else:
                match = re.search(r'\d{1,2}\s+\w+\s+\d{4}', soup.get_text())
                if match:
                    date = match.group(0)

            # Content
            content_container = soup.find('article') or soup.find('div', {'class': 'post-content'}) or soup
            paragraphs = [p.get_text() for p in content_container.find_all('p')]
            article_text = "\n\n".join([p.strip() for p in paragraphs if p.strip()])
            #print(article_text)

            # Filter out junk
            if len(article_text.split()) < 100:
                print(f"Skipped (too short): {url}")
                continue
            if title.lower() in ["news & events", ""]:
                print(f"Skipped (bad title): {url}")
                continue

            articles.append({
                'title': title,
                'date': date,
                'url': url,
                'text': article_text
            })

        except Exception as e:
            print(f"Error scraping {url}: {e}")
            continue
        
        
if __name__ == '__maifefn__':
    index, chunk_metadata = update_verztec_index()

    # Save chunks to a file (if not already done)
    with open("verztec_chunks.txt", "w", encoding="utf-8") as f:
        for meta in chunk_metadata:
            f.write(meta['title'].replace("\n", " ") + "\n")  # or use the raw text if needed

    # Load chunks for querying
    with open("verztec_chunks.txt", "r", encoding="utf-8") as f:
        chunks = [line.strip() for line in f.readlines()]

    # Prepare query and embed
    model = SentenceTransformer("all-MiniLM-L6-v2")
    query = "What does Verztec do?"
    query_embedding = model.encode([query]).astype("float32")

    # Search FAISS
    top_k = 5
    distances, indices = index.search(query_embedding, top_k)
    for idx, score in zip(indices[0], distances[0]):
        if idx == -1:
            continue

        chunk_text = chunks[idx]

        print(f"Score: {score:.4f}")
        print("Chunk Text:")
        print(chunk_text)  # This is your full ~300 word chunk
        print("---")        
if __name__ == '__main__':
    idk()