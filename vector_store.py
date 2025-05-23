import os
import re
from pathlib import Path
from dotenv import load_dotenv
import spacy
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document as langDocument
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser

# Load environment variables and spaCy model
load_dotenv()
nlp = spacy.load("en_core_web_sm")

# Initialize models
api_key = "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"  # Set in your .env file
model = "deepseek-r1-distill-llama-70b"
deepseek = ChatGroq(api_key=api_key, model_name=model)
deepseek_chain = deepseek | StrOutputParser()

embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-en-v1.5",
    encode_kwargs={'normalize_embeddings': True}
)

# Helper: Get all file paths recursively

def get_all_file_paths(directory, extensions=[".pdf", ".doc", ".docx", ".txt"]):
    file_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            if any(file.lower().endswith(ext) for ext in extensions):
                file_paths.append(os.path.abspath(os.path.join(root, file)))
    return file_paths

# Text splitters
fallback_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=50)
bullet_pattern = r'(?m)(?:^\s*(\d+[\.\)]|[a-zA-Z][\.\)]|[-\u2022\*])\s+)(.*?)(?=^\s*(\d+[\.\)]|[a-zA-Z][\.\)]|[-\u2022\*])\s+|$)'

def spacy_fallback_splitter(text, chunk_size=300, overlap=50):
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    chunks, current_chunk = [], ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= chunk_size:
            current_chunk += " " + sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
    if current_chunk:
        chunks.append(current_chunk.strip())

    if overlap > 0:
        overlapped_chunks = []
        for i in range(len(chunks)):
            start = max(0, i - 1)
            merged = " ".join(chunks[start:i+1])
            overlapped_chunks.append(merged.strip())
        return overlapped_chunks

    return chunks

def split_text(text):
    matches = list(re.finditer(bullet_pattern, text, flags=re.DOTALL | re.MULTILINE))
    chunks = [m.group(0).strip() for m in matches if len(m.group(0).strip()) > 5 and not m.group(0).strip().isdigit()]
    final_chunks = []
    for chunk in chunks:
        final_chunks.extend(fallback_splitter.split_text(chunk) if len(chunk) > 600 else [chunk])
    return final_chunks or spacy_fallback_splitter(text)

# Paths
root_dir = Path.cwd()
data_dir = root_dir / "data/cleaned"
cleaned_files = get_all_file_paths(data_dir)

# Chunk and embed all docs
all_chunks = []

for file_path in cleaned_files:
    indv_chunks = []
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read().lower()

    query = (
        'The following is an internal document for Verztec, a consulting company. '
        'Provide a 45-word description mentioning its contents, use, and potential queries it can answer: ' + text
    )
    raw_response = deepseek_chain.invoke(query)
    cleaned_response = re.sub(r"<think>.*?</think>", "", raw_response, flags=re.DOTALL).strip()

    # Use the cleaned response
    description = cleaned_response
    print(description)
    smart_chunks = split_text(text)

    for i, chunk in enumerate(smart_chunks):
        # Enriched version for global index
        enriched_chunk = f"[Description: {description}] [Document: {base_name}] {chunk}"
        all_chunks.append(langDocument(
            page_content=enriched_chunk,
            metadata={"source": base_name, "chunk_id": f"{base_name}_{i}", "clean_chunk": chunk}
        ))

        # Raw version for individual index
        indv_chunks.append(langDocument(
            page_content=chunk,
            metadata={"source": base_name, "chunk_id": f"{base_name}_{i}"}
        ))

    # Save individual document vector index
    indv_db = FAISS.from_documents(indv_chunks, embedding_model)
    indv_db.save_local(f"indv_doc_faiss/{base_name}_index")

# Save combined vector index
global_db = FAISS.from_documents(all_chunks, embedding_model)
global_db.save_local("faiss_index3")

# Test similarity search
query = "pantry rules"
results = global_db.similarity_search_with_score(query, k=10)

print('\n--- SIMILARITY SEARCH RESULTS ---')
for doc, score in results:
    print("\n---")
    print(doc.page_content)
    print(doc.metadata)
    print(f"Score: {score}")