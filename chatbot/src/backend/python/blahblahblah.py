import mysql.connector
import logging
import tiktoken  # For counting tokens reliably
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage
import re

# Logger setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# DB Config
DB_CONFIG = {
    'host': 'localhost',
    'user': 'chatbot_user',
    'password': 'strong_password',
    'database': 'chatbot_db',
    'raise_on_warnings': True
}

# GPT Token Limit Config
TOKEN_LIMIT = 4000  # Safely under the 6000 TPM limit

# Load tokenizer (for DeepSeek's tokenizer use GPT-3.5's encoding as fallback)
encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

def count_tokens(text: str) -> int:
    return len(encoding.encode(text))

def chunk_text(text: str, token_limit: int) -> list:
    words = text.split()
    chunks = []
    current_chunk = []

    for word in words:
        current_chunk.append(word)
        if count_tokens(' '.join(current_chunk)) >= token_limit:
            chunks.append(' '.join(current_chunk))
            current_chunk = []

    if current_chunk:
        chunks.append(' '.join(current_chunk))
    return chunks

def get_combined_cleaned_text():
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor(dictionary=True)

    select_query = 'SELECT text_content FROM extracted_texts'
    cursor.execute(select_query)
    results = cursor.fetchall()
    cursor.close()
    conn.close()

    combined_text = ' '.join(row['text_content'] for row in results if row['text_content'])
    logger.info(f"Combined {len(results)} entries into a single text block.")
    return combined_text

# Initialize LLM
api_key = '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'
model = "deepseek-r1-distill-llama-70b"
deepseek = ChatGroq(api_key=api_key, model=model)

def summarize_chunks(text: str):
    chunks = chunk_text(text, TOKEN_LIMIT)
    logger.info(f"Splitting into {len(chunks)} chunks due to token limits.")

    all_summaries = []

    for i, chunk in enumerate(chunks):
        prompt = (
            "Summarize the following internal company text into a short context description "
            "useful for understanding pantry rules, HR protocols, and internal guidelines:\n\n"
            f"{chunk}"
        )
        messages = [HumanMessage(content=prompt)]
        response = deepseek.generate([messages])
        summary = response.generations[0][0].text.strip()
        logger.info(f"Chunk {i+1}/{len(chunks)} summarized.")
        all_summaries.append(summary)

    return ' '.join(all_summaries)

# === Run It ===
text = get_combined_cleaned_text()
final_summary = summarize_chunks(text)
think_block_pattern = re.compile(r"<think>.*?</think>", flags=re.DOTALL)
final = think_block_pattern.sub("", final_summary).strip()

print("\n📄 Final Context Summary for LLM:")
print(final)

### verztec geeneral knowledege stuffings 
import requests
from bs4 import BeautifulSoup
import re
import numpy as np               # For array handling
import faiss                     # Facebook AI Similarity Search library
from sentence_transformers import SentenceTransformer

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


