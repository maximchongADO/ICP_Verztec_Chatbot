import os
import re
import mimetypes
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

import ftfy
import fitz  # pymupdf
import spacy
import win32com.client
from PIL import Image, ImageGrab
import imagehash
from pptx import Presentation
from docx import Document
from docx.oxml.ns import qn

from dotenv import load_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings

from langchain.schema import Document as langDocument
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser

from MySQLDatabase.Inserting_data import insert_knowledge_chunks


import os
import re
import ftfy
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from langchain.schema import Document as langDocument
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser

## define root directory and subsequent sub directories
root_dir = Path(os.getcwd())  # Get the current working directory (the root of your repository)
data_dir = root_dir / "data"  # Gets the "data" folder relative to the root directory
images_dir = data_dir / "images"
pdf_dir = data_dir / "pdf"
docx_dir = data_dir / "word"
pptx_dir = data_dir / "pptx"
cleaned_dir=data_dir/ "cleaned"
vertztec_collection= data_dir/ "verztec_logo"

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

# Text splitters
fallback_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=50)

def spacy_fallback_splitter(text, chunk_size=400, overlap=50):
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

def replace_image_tags_with_placeholders(text):
    pattern = r"<\|image_start\|>(.*?)<\|image_end\|>"
    matches = re.findall(pattern, text)
    replacements = {}
    image_name_map = {}
    for i, match in enumerate(matches):
        key = f"[[IMAGE_TAG_{i}]]"
        full_tag = f"<|image_start|>{match}<|image_end|>"
        text = text.replace(full_tag, key)
        replacements[key] = full_tag
        image_name_map[key] = match  # Just the image file name
    return text, replacements, image_name_map

def restore_image_placeholders_and_collect_metadata(chunks, replacements, image_name_map):
    restored_chunks = []
    chunk_image_metadata = []

    for chunk in chunks:
        images_in_chunk = []
        for placeholder, original in replacements.items():
            if placeholder in chunk:
                chunk = chunk.replace(placeholder, original)
                images_in_chunk.append(image_name_map[placeholder])
        restored_chunks.append(chunk)
        chunk_image_metadata.append(images_in_chunk)
    
    return restored_chunks, chunk_image_metadata

def load_single_file(file_path, embedding_model, faiss_index_path=None):
    """
    Processes a single file, generates embeddings, splits text into chunks,
    and updates the FAISS index if given, or creates a new one if no index is provided.
    """

    # Ensure the file is valid
    file_path = Path(file_path)
    if not file_path.exists():
        print(f"[ERROR] File not found: {file_path}")
        return None

    base_name = file_path.stem
    cleaned_filename = base_name.lower().replace(" ", "_") + ".txt"  # Cleaned filename

    # Read the file content
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read().lower()

    # Generate semantic description with LLM (Deepseek)
    query = (
        "Summarize this internal document in 50 words. Describe its purpose, main contents, "
        "and types of internal questions it can help answer. "
        "Respond concisely and in a single paragraph:\n\n" + text
    )
    raw_response = deepseek_chain.invoke(query)
    description = re.sub(r"<think>.*?</think>", "", raw_response, flags=re.DOTALL).strip()

    # Replace image tags with placeholders
    text, image_replacements, image_name_map = replace_image_tags_with_placeholders(text)

    # Split text into chunks
    smart_chunks = spacy_fallback_splitter(text)

    # Restore image tags and get images per chunk
    smart_chunks, chunk_image_lists = restore_image_placeholders_and_collect_metadata(
        smart_chunks, image_replacements, image_name_map
    )

    # Build langDocument objects for new chunks
    all_chunks = []
    for i, chunk in enumerate(smart_chunks):
        image_list = chunk_image_lists[i]

        enriched_chunk = f"[Description: {description}] [Document: {base_name}] {chunk}"
        all_chunks.append(langDocument(
            page_content=enriched_chunk,
            metadata={
                "source": base_name,
                "chunk_id": f"{base_name}_{i}",
                "clean_chunk": chunk,
                "images": image_list
            }
        ))

    # Check if FAISS index exists
    if faiss_index_path and os.path.exists(faiss_index_path):
        print(f"[INFO] Loading existing FAISS index from {faiss_index_path}")
        faiss_db = FAISS.load_local(
            faiss_index_path, 
            embeddings=embedding_model,
            allow_dangerous_deserialization=True  # Add this parameter
        )
    else:
        # Create a new FAISS index
        print(f"[INFO] Creating new FAISS index for {base_name}")
        faiss_db = FAISS.from_documents(all_chunks, embedding_model)

    # Add the new chunks to the FAISS index
    faiss_db.add_documents(all_chunks)

    # Save updated FAISS index
    faiss_db.save_local(faiss_index_path or f"faiss_index_{base_name}")

    print(f"[INFO] Processed and updated index for: {cleaned_filename}")

    return faiss_db


from Documents_Totext import process_single_file

# Example usage for a single file (updating the FAISS index)
file_to_process = r"C:\Users\ethan\OneDrive\Desktop\ICP_Verztec_Chatbot-6\data\cleaned\3_offboarding_process_on_clean_desk_policy_150125_.txt"
result = process_single_file(file_to_process, images_dir, cleaned_dir, vertztec_collection)
existing_faiss_index = "faiss_index2"  # Path to existing FAISS index
faiss_index = load_single_file(file_to_process, embedding_model, faiss_index_path=existing_faiss_index)