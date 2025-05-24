import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage

# Initialize embedding model (make sure you loaded this in your environment)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize Groq LLM client
api_key = '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'
model_name = "deepseek-r1-distill-llama-70b"
deepseek = ChatGroq(api_key=api_key, model_name=model_name)

# Load FAISS index and metadata
def load_faiss_and_metadata(index_path, meta_path):
    index = faiss.read_index(index_path)
    with open(meta_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    return index, metadata

# Refine the user query (fix grammar/spelling) using Groq LLM
def refine_prompt(user_query: str) -> str:
    prompt = f"Your only job is to correct grammar and spelling mistakes and rephrase the question clearly. The questions are based for a internal helpdesk chatbot for a company:\n\n{user_query}"
    messages = [HumanMessage(content=prompt)]
    response = deepseek.generate([messages])
    refined_query = response.generations[0][0].text.strip()
    return refined_query

# Search FAISS index with the refined query and get relevant chunks
def search_faiss_and_get_context(refined_query, index, metadata, top_k=5):
    query_emb = embedding_model.encode([refined_query], convert_to_numpy=True).astype('float32')
    distances, indices = index.search(query_emb, top_k)
    retrieved_chunks = [metadata[idx]['text'] for idx in indices[0]]
    return retrieved_chunks

# Generate final answer with original query and retrieved context using Groq LLM
def generate_final_answer(original_query: str, context_chunks: list[str]) -> str:
    context_text = "\n---\n".join(context_chunks)
    prompt = (
        f"You are an internal company helpdesk chatbot for employees that answers questions based on the given context (Company Information) and is not allowed to access information outside of the given context.\n\n"
        f"Context:\n{context_text}\n\n"
        f"Question: {original_query}\n\n"
        f"Please provide a clear and concise answer."
    )
    messages = [HumanMessage(content=prompt)]
    response = deepseek.generate([messages])
    answer = response.generations[0][0].text.strip()
    return answer

# === Main Execution ===

if __name__ == "__main__":
    user_query = "wat to do after i get fired"

    # Load your FAISS index and metadata files
    faiss_index_path = r"C:\Users\ethan\OneDrive\Desktop\ICP_Verztec_Chatbot-2\improvefaiss\faiss_local.idx"
    faiss_meta_path = r"C:\Users\ethan\OneDrive\Desktop\ICP_Verztec_Chatbot-2\improvefaiss\faiss_local_meta.json"
    index, metadata = load_faiss_and_metadata(faiss_index_path, faiss_meta_path)

    # Step 1: Refine user query
    clean_query = refine_prompt(user_query)

    # Step 2: Retrieve relevant chunks from FAISS
    context_chunks = search_faiss_and_get_context(clean_query, index, metadata, top_k=5)

    # Step 3: Generate final answer from context + original query
    final_answer = generate_final_answer(clean_query, context_chunks)

    print("User Query:", user_query)
    print("Refined Query:", clean_query)
    print("Final Answer:", final_answer)
