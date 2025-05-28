from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import json
import traceback
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage
import re
from numpy import dot
from numpy.linalg import norm
import os

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Express.js server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize embedding model
embedding_model = SentenceTransformer('BAAI/bge-large-en-v1.5')

# Initialize Groq LLM client
api_key = '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'
model_name = "compound-beta"
deepseek = ChatGroq(api_key=api_key, model_name=model_name)

# Request/Response models
class ChatRequest(BaseModel):
    message: str
    user_id: Optional[str] = None
    chat_history: Optional[List[str]] = []

class ChatResponse(BaseModel):
    message: str
    user_id: Optional[str] = None
    timestamp: str
    success: bool
    error: Optional[str] = None

# Load FAISS index and metadata on startup
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    faiss_index_path = os.path.join(script_dir, "faiss_local_BAAI.idx")
    faiss_meta_path = os.path.join(script_dir, "faiss_local_BAAI_meta.json")
    
    index = faiss.read_index(faiss_index_path)
    with open(faiss_meta_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    print("FAISS index and metadata loaded successfully")
except Exception as e:
    print(f"Failed to load FAISS index: {str(e)}")
    index = None
    metadata = None

def load_faiss_and_metadata(index_path, meta_path):
    index = faiss.read_index(index_path)
    with open(meta_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    return index, metadata

def refine_prompt(user_query: str) -> str:
    prompt = (
        f"Your only job is to correct grammar and spelling mistakes and rephrase the prompt clearly for an internal helpdesk chatbot for a company:\n\n{user_query}"
    )
    messages = [HumanMessage(content=prompt)]
    response = deepseek.generate([messages])
    refined_query = response.generations[0][0].text.strip()
    return refined_query

def search_faiss_and_get_context(refined_query, index, metadata, top_k=4):
    query_emb = embedding_model.encode([refined_query], convert_to_numpy=True).astype('float32')
    distances, indices = index.search(query_emb, top_k)
    retrieved_chunks = [metadata[idx]['text'] for idx in indices[0]]
    return retrieved_chunks, distances[0]

def is_related_to_previous(current_query: str, previous_queries: List[str], threshold: float = 0.6) -> bool:
    if not previous_queries:
        return False
    current_vec = embedding_model.encode(current_query)
    similarities = [dot(current_vec, embedding_model.encode(p)) / (norm(current_vec) * norm(embedding_model.encode(p))) for p in previous_queries]
    max_similarity = max(similarities)
    return max_similarity >= threshold

def generate_answer(user_query: str, chat_history: List[str]) -> str:
    try:
        if index is None or metadata is None:
            return "Chatbot is not properly initialized. Please check FAISS files."
        
        # Step 1: Check relation to past queries
        is_related = is_related_to_previous(user_query, chat_history)
        context_query = " ".join(chat_history + [user_query]) if is_related else user_query

        # Step 2: Refine query
        clean_query = refine_prompt(context_query)

        # Step 3: Search FAISS for context
        context_chunks, scores = search_faiss_and_get_context(clean_query, index, metadata, top_k=4)
        avg_score = float(np.mean(scores)) if len(scores) > 0 else 1.0

        if avg_score > 2.5:
            fallback = (
                f"The user's question is likely unrelated to internal company topics.\n"
                f"Query: {clean_query}\n"
                f"Please advise the user that this query is out of scope."
            )
            messages = [HumanMessage(content=fallback)]
            response = deepseek.generate([messages])
            return response.generations[0][0].text.strip()

        # Step 4: Prepare full prompt and return LLM output
        recent_history_text = ""
        if is_related:
            for q in chat_history[-2:]:
                recent_history_text += f"User: {q}\n"

        recent_history_text += f"User: {user_query}\n"
        combined_context = "\n---\n".join(context_chunks)

        final_prompt = (
            "You are an internal helpdesk chatbot answering questions ONLY based on the given company context.\n\n"
            f"Conversation history:\n{recent_history_text}\n"
            f"Company documents:\n{combined_context}\n\n"
            f"Question: {clean_query}\n"
            "Please provide a clear and concise answer."
        )
        messages = [HumanMessage(content=final_prompt)]
        response = deepseek.generate([messages])
        return response.generations[0][0].text.strip()
    
    except Exception as e:
        return f"I encountered an error while processing your request: {str(e)}"

@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "Chatbot API is running"}

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        if not request.message.strip():
            raise HTTPException(status_code=400, detail="Message cannot be empty")
        
        # Generate response using chatbot logic
        response_message = generate_answer(request.message, request.chat_history or [])
        
        return ChatResponse(
            message=response_message,
            user_id=request.user_id,
            timestamp="2024-01-01T12:00:00Z",
            success=True
        )
        
    except Exception as e:
        print(f"Error processing chat request: {str(e)}")
        print(traceback.format_exc())
        return ChatResponse(
            message="I'm sorry, I encountered an error while processing your request.",
            user_id=request.user_id,
            timestamp="2024-01-01T12:00:00Z",
            success=False,
            error=str(e)
        )

@app.delete("/history")
async def clear_chat_history():
    try:
        # For now, just return success
        # In a real implementation, you'd clear user-specific chat history from database
        return {"success": True, "message": "Chat history cleared successfully"}
    except Exception as e:
        print(f"Error clearing chat history: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to clear chat history")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
