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
