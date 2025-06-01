import os
import re
import mimetypes
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

import ftfy
import fitz
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
api_key = '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'  # Set in your .env file
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

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
from Documents_Totext import process_single_file

def unified_document_pipeline(file_path, images_dir, cleaned_dir, vertztec_collection, faiss_index_path, embedding_model):
    """
    Unified pipeline to process documents and update FAISS index:
    1. Extract and clean document content
    2. Process images and text
    3. Generate embeddings and update FAISS index
    
    Args:
        file_path: Path to input document
        images_dir: Directory for extracted images
        cleaned_dir: Directory for cleaned text
        vertztec_collection: Directory with Verztec logos
        faiss_index_path: Path to FAISS index
        embedding_model: HuggingFace embedding model
    
    Returns:
        dict: Processing results including FAISS update status
    """
    try:
        # Step 1: Process document and extract content
        processing_result = process_single_file(
            file_path=file_path,
            images_dir=images_dir,
            cleaned_dir=cleaned_dir,
            vertztec_collection=vertztec_collection
        )
        
        if not processing_result["success"]:
            print(f"[ERROR] Document processing failed: {processing_result['error']}")
            return processing_result

        # Step 2: Update FAISS index with processed content
        cleaned_text_path = processing_result["cleaned_text_path"]
        faiss_db = load_single_file(
            file_path=cleaned_text_path,
            embedding_model=embedding_model,
            faiss_index_path=faiss_index_path
        )
        
        if faiss_db is None:
            processing_result["success"] = False
            processing_result["error"] = "Failed to update FAISS index"
            return processing_result

        # Return processing details for the Node.js backend to handle database operations
        return {
            **processing_result,
            "faiss_index_path": faiss_index_path,
            "chunks": [doc.page_content for doc in faiss_db.docstore._dict.values()],
            "metadata": [doc.metadata for doc in faiss_db.docstore._dict.values()]
        }

    except Exception as e:
        error_msg = f"Pipeline failed: {str(e)}"
        print(f"[ERROR] {error_msg}")
        return {
            "original_path": file_path,
            "cleaned_text_path": None,
            "success": False,
            "error": error_msg
        }

# Example usage:
if __name__ == "__main__":
    # File to process
    file_to_process = r"C:\Users\ethan\OneDrive\Desktop\ICP_Verztec_Chatbot-6\data\pdf\example.pdf"
    
    # FAISS index path
    existing_faiss_index = "faiss_index2"
    
    # Process document and update FAISS index
    result = unified_document_pipeline(
        file_path=file_to_process,
        images_dir=images_dir,
        cleaned_dir=cleaned_dir,
        vertztec_collection=vertztec_collection,
        faiss_index_path=existing_faiss_index,
        embedding_model=embedding_model
    )
    
    if result["success"]:
        print(f"Pipeline completed successfully!")
    else:
        print(f"Pipeline failed: {result['error']}")