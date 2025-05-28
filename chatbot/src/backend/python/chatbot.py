from fastapi import FastAPI, HTTPException, Request
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
from langchain.memory import ConversationBufferMemory  # Updated import
from langchain.chains import ConversationalRetrievalChain
from langchain_core.output_parsers import StrOutputParser
from langchain.embeddings import HuggingFaceEmbeddings
 # Updated import
from langchain_community.vectorstores import FAISS
import logging
from datetime import datetime
from fastapi.responses import JSONResponse
app = FastAPI()

# Update CORS middleware to allow all origins in development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize embedding model
embedding_model = SentenceTransformer('BAAI/bge-large-en-v1.5')

# Initialize Groq LLM client
api_key = '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'
model_name = "compound-beta"
model = "deepseek-r1-distill-llama-70b" 
deepseek = ChatGroq(api_key=api_key, model=model)
compound =ChatGroq(api_key=api_key, model=model_name)

## setting up memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    input_key="question",
    output_key="answer",
    return_messages=True
)

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
    #comment back in if you want to use images, not in rn cos idw break the front 
    #contains_image:bool = False
    #image_url: Optional[List[str]] = None
    error: Optional[str] = None
    
    

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global error handler: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ChatResponse(
            message="An internal server error occurred",
            timestamp=datetime.utcnow().isoformat(),
            success=False,
            error=str(exc)
        ).dict()
    )

# Load FAISS index and metadata on startup
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    faiss_index_path = os.path.join(script_dir, "faiss_index3")
    
    if not os.path.exists(faiss_index_path):
        raise FileNotFoundError(f"FAISS index not found at {faiss_index_path}")
    
    embedding_model2 = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en-v1.5",
        encode_kwargs={'normalize_embeddings': True}
    )
    index = FAISS.load_local(
        faiss_index_path,
        embedding_model2,
        allow_dangerous_deserialization=True
    )
    logger.info("FAISS index and metadata loaded successfully")
except Exception as e:
    logger.error(f"Failed to load FAISS index: {str(e)}", exc_info=True)
    index = None
    metadata = None


## clean query and ensure it is suitable for processing
def refine_prompt(user_query: str) -> str:
    prompt = (
        f"You are an assistant that improves grammar and spelling for an internal helpdesk chatbot. "
        f"If the input is offensive, unclear, or meaningless, return the phrase: [UNCLEAR].\n\n"
        f"Input:\n{user_query}"
    )
    messages = [HumanMessage(content=prompt)]
    response = compound.generate([messages])
    refined_query = response.generations[0][0].text.strip()

    # Safety net
    if "[UNCLEAR]" in refined_query.upper() or len(refined_query) < 5:
        return "[UNCLEAR]"
    if "[UNCLEAR]" in refined_query.upper():
        refined_query= 'The user has sent something unintellible, please clarify    '
    return refined_query



def generate_answer(user_query: str, chat_history: ConversationBufferMemory) -> str:
    try:
        parser = StrOutputParser()

        # Chain the Groq model with the parser
        deepseek_chain = deepseek | parser
        ## FAISS retriever setup
        retriever = index.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 10}
        )
        ## QA chain setup with mrmory 
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=deepseek_chain,
            retriever=retriever,
            memory=chat_history,
            return_source_documents=True,
            output_key="answer"
        
        )

        # Step 1: Check relation to past queries
        #is_related = is_related_to_previous(user_query, chat_history)
        #context_query = " ".join(chat_history + [user_query]) if is_related else user_query

        # Step 2: Refine query
        clean_query = refine_prompt(user_query)

        # Step 3: Search FAISS for context 
        # for images, as well as for context relevance checks 
        results = index.similarity_search_with_score(clean_query, k=5)
        scores = [score for _, score in results]
        avg_score = float(np.mean(scores)) if scores else 1.0
        
        ## retrieving images from top 3 chunks (if any)
        top_3_img=[]
        for i, (doc, score) in enumerate(results, 1):
            if len(top_3_img) < 3:
                top_3_img.append(doc.metadata['images'])


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
        response = qa_chain.invoke({"question": clean_query})
        answer = re.sub(r"<think>.*?</think>", "", response['answer'], flags=re.DOTALL).strip()
        return answer
    
    except Exception as e:
        return f"I encountered an error while processing your request: {str(e)}"

@app.get("/health")
async def health_check():
    try:
        return JSONResponse(
            content={"status": "healthy", "message": "Chatbot API is running"},
            status_code=200
        )
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return JSONResponse(
            content={"status": "unhealthy", "error": str(e)},
            status_code=500
        )

@app.post("/chatbot")
async def chat_endpoint(request: ChatRequest):
    logger.info(f"Received chat request: {request}")
    try:
        if not request.message.strip():
            raise HTTPException(status_code=400, detail="Message cannot be empty")
        
        if index is None:
            raise HTTPException(status_code=503, detail="Search index is not available")
        
        # Generate response using chatbot logic
        response_message = generate_answer(request.message, memory)
        logger.info(f"Generated response: {response_message}")
        
        return ChatResponse(
            message=response_message,
            user_id=request.user_id,
            timestamp=datetime.utcnow().isoformat(),
            success=True
        )
        
    except Exception as e:
        logger.error(f"Error processing chat request: {str(e)}", exc_info=True)
        return ChatResponse(
            message="An error occurred while processing your request. Please try again later.",
            user_id=request.user_id,
            timestamp=datetime.utcnow().isoformat(),
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

def find_available_port(start_port: int = 8000, max_attempts: int = 10) -> int:
    import socket
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('0.0.0.0', port))
                return port
        except OSError:
            continue
    raise RuntimeError(f"Could not find an available port after {max_attempts} attempts")

if __name__ == "__main__":
    import uvicorn
    try:
        print("Starting server on port 3000")
        uvicorn.run(app, host="0.0.0.0", port=3000, log_level="debug")
    except Exception as e:
        print(f"Failed to start server: {e}")
