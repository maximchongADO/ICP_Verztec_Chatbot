import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from pathlib import Path

# Load embedding model
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

root_dir = Path(__file__).parent
data_dir = root_dir / "data"
cleaned_dir = data_dir / "cleaned"

# Define function for loading and splitting texts into chunks
def load_cleaned_texts(folder_path):
    texts, metadatas = [], []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            with open(os.path.join(folder_path, filename), "r", encoding="utf-8") as f:
                raw_text = f.read()
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            chunks = splitter.split_text(raw_text)
            texts.extend(chunks)
            metadatas.extend([{"source": filename}] * len(chunks))
    return texts, metadatas

# FAISS Vector Store Setup (Run Once)
def setup_faiss_vector_store():
    texts, metadatas = load_cleaned_texts(cleaned_dir)
    
    # Create the FAISS vector store and save it locally
    vector_store = FAISS.from_texts(texts=texts, embedding=embedding_model, metadatas=metadatas)
    vector_store.save_local("verztec_vector_store")



# FAISS Vector Store Setup (Run Once)
setup_faiss_vector_store()

