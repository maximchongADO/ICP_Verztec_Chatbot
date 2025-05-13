


import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import Document as langDocument
from langchain.chains import RetrievalQA
from pathlib import Path
import re
import time

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
embedding_model2 = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    encode_kwargs={'normalize_embeddings': True}  
)
embedding_model2 = HuggingFaceEmbeddings(
    model_name = "BAAI/bge-large-en-v1.5",

    encode_kwargs={'normalize_embeddings': True}  
)
# --- Step 1: Load FAISS Vector Store ---
start_faiss = time.time()
vector_store = FAISS.load_local(
    "faiss_index3",
    embedding_model2,
    allow_dangerous_deserialization=True
)
end_faiss = time.time()
print(f"FAISS loaded in {end_faiss - start_faiss:.2f} seconds")

# --- Step 2: Set up retriever ---
start_retriever = time.time()
retriever = vector_store.as_retriever(search_type="similarity", k=1)

retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 5,              # total number of documents to retrieve
        "fetch_k": 15,       # candidate pool size
        "lambda_mult": 1   # relevance-diversity trade-off 
    }
)

#retriever = vector_store.as_retriever(search_type="similarity", k=1)
end_retriever = time.time()
print(f"Retriever set up in {end_retriever - start_retriever:.2f} seconds")

llm = ChatOpenAI(
    model_name="tngtech/deepseek-r1t-chimera:free",
    temperature=0.1, # randomness of response
    openai_api_key="sk-or-v1-189b8b7afc43328f8cbebe03cf4e73955bb9bfefe7690f1e051f63e8c9f31758wh",
    openai_api_base="https://openrouter.ai/api/v1"
)


qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)


try:
    
    query= input("question(0 to exit): ")
    while query !="0":
        start_llm = time.time()
        response = qa_chain.invoke({"query": query})
        end_llm = time.time()
        print(f"LLM loaded + query answered in {end_llm - start_llm:.2f} seconds")

        # --- Output ---
        print("\nQuery:", query)
        print("Answer:", response["result"])
        print("\n Sources:")
        for doc in response["source_documents"]:
            print("-", doc.metadata["source"])
            print("--", doc.page_content)
        query= input("question(0 to exit): ")
        
except Exception as e:
    print("LLM or Retrieval failed:", e)