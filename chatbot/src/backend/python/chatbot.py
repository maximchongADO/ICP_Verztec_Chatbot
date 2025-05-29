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
from langchain.vectorstores.base import VectorStoreRetriever
from langchain.schema import Document
#from numpy import dot
#from numpy.linalg import norm
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
#from langchain.schema import BaseRetriever
from better_profanity import profanity
import spacy
from spacy.matcher import PhraseMatcher
app = FastAPI()

# Update CORS middleware to allow all origins in development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600
)


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
    
    embedding_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en-v1.5",
        encode_kwargs={'normalize_embeddings': True}
    )
    index = FAISS.load_local(
        faiss_index_path,
        embedding_model,
        allow_dangerous_deserialization=True
    )
    logger.info("FAISS index and metadata loaded successfully")
except Exception as e:
    logger.error(f"Failed to load FAISS index: {str(e)}", exc_info=True)
    index = None
    metadata = None

try:
    # Load spaCy English model
    nlp = spacy.load("en_core_web_sm")
    casual_phrases = ["hi", "hello", "thanks", "thank you", "good morning", "goodbye", "hey", "yo", "sup"]
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    patterns = [nlp.make_doc(text) for text in casual_phrases]
    matcher.add("CASUAL", patterns)
    logger.info("SpaCy model and matcher initialized successfully")
except Exception as e:
    logger.error(f"Failed to load spacymodel:  {str(e)}", exc_info=True)
    

def is_casual_message(text: str) -> bool:
    doc = nlp(text)
    matches = matcher(doc)
    return bool(matches)



def is_query(text: str) -> bool:
    try:
        doc = nlp(text)
        # Check for known casual question patterns
        casual_phrases = {"how are you", "what's up", "how's it going", "are you there"}
        normalized = text.lower().strip("!?.,")
        if normalized in casual_phrases:
            return False

        # Check for bot refusal or rubbish responses
        refusal_patterns = [
            "i'm sorry", "i cannot", "i can't", "as an ai", "i am unable", "not able to", 
            "i do not understand", "i don't know", "no comment", "n/a", "not applicable",
            "this is not a question", "please rephrase", "unclear", "meaningless"
        ]
        if any(pattern in normalized for pattern in refusal_patterns):
            return False

        # Check for question words or action verbs
        for token in doc:
            if token.dep_ in ("aux", "ROOT") and token.tag_ in ("VB", "VBP", "VBZ", "MD"):
                if token.lemma_ in {"do", "can", "could", "help", "need", "upload", "reset", "access"}:
                    return True
            if token.dep_ == "nsubj" and token.lemma_ in {"problem", "issue"}:
                return True
            if token.lower_ in {"how", "what", "when", "where", "why", "who"}:
                return True

        # If it looks like a statement or rubbish, not a query
        return False
    except Exception as e:
        logger.error(f"is_query exception: {e} for text: {text!r}")
        return False


profanity.load_censor_words()

def is_profane(text: str) -> bool:
    return profanity.contains_profanity(text)

## clean query and ensure it is suitable for processing
def refine_prompt(user_query: str) -> str:
    is_profane_query = is_profane(user_query)
    logger.info(f"Is Profane: {is_profane_query}")
    logger.info(f"User Query: {user_query}")
    is_casual = is_query(user_query)
    logger.info(f"Is Question: {is_casual}")
    prompt2 = (
        f"ONLY RETURN THE CLEANED QUERY. You are an assistant that improves grammar and spelling for an internal helpdesk chatbot. "
        f"If the input is offensive, unclear, or meaningless, please clean it.\n\n"
        f"Input:\n{user_query}"
    )
    messages = [HumanMessage(content=prompt2)]
    response = compound.generate([messages])
    refined_query = response.generations[0][0].text.strip()
    logger.info(f"User Query cleaned: {refined_query}")

    # Safety net
    
    isquery = is_query(refined_query)
    logger.warning(f'New query is a question: {isquery}')
    if not isquery:
        refined_query = user_query
        logger.warning(f"Refined query was not a valid question, reverting to original: {refined_query}")
    
    

   
    return refined_query

def get_avg_score(index, embedding_model, query, k=10):
    # Embed using LangChain-style embedding model
    query_vec = embedding_model.embed_query(query)
    
    # Get the internal FAISS index
    raw_faiss_index = index.index
    
    # Prepare for search
    query_np = np.array([query_vec], dtype=np.float32)
    D, _ = raw_faiss_index.search(query_np, k)
    
    avg_score = np.mean(D[0]) if D.shape[1] > 0 else float('inf')
    logger.info(f"[L2] Average distance for query '{query}': {avg_score}")
    return avg_score


AVG_SCORE_THRESHOLD = 0.98


class CustomPromptRetriever(VectorStoreRetriever):
    def __init__(self, base_retriever, modify_query_fn):
        self.base_retriever = base_retriever
        self.modify_query_fn = modify_query_fn

    def get_relevant_documents(self, query: str):
        modified = self.modify_query_fn(query)
        return self.base_retriever.get_relevant_documents(modified)

    async def aget_relevant_documents(self, query: str):
        modified = self.modify_query_fn(query)
        return await self.base_retriever.aget_relevant_documents(modified)
def simplify_for_retrieval(query: str) -> str:
    # Example: strip out casual or irrelevant context
    query = query.lower().strip()
    if "please" in query:
        query = query.replace("please", "")
    if query.endswith("?"):
        query = query[:-1]
    return query


def generate_answer(user_query: str, chat_history: ConversationBufferMemory) -> str:
    try:
        parser = StrOutputParser()
        

        # Chain the Groq model with the parser
        deepseek_chain = deepseek | parser
        
        retriever = index.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 10}
        )
        avg_score = get_avg_score(index, embedding_model, user_query)
        
      
       
        ## FAISS retriever setup
        
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
        #clean_query=user_query

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
        is_task_query = is_query(user_query)
        
        logger.info(f"Clean Query at qa chain: {clean_query}")
        if not is_task_query and avg_score >= AVG_SCORE_THRESHOLD:
            logger.info("Bypassing QA chain for non-query with weak retrieval.")
            fallback_prompt = f"The user said: \"{clean_query}\". Respond appropriately as a polite assistant."
            messages = [HumanMessage(content=fallback_prompt)]
            response = deepseek.generate([messages])
            raw_fallback = response.generations[0][0].text.strip()

            # Remove <think> block if present
            think_block_pattern = re.compile(r"<think>.*?</think>", flags=re.DOTALL)
            cleaned_fallback = think_block_pattern.sub("", raw_fallback).strip()
            return cleaned_fallback

        # Step 4: Prepare full prompt and return LLM output
        response = qa_chain.invoke({"question": clean_query})
        raw_answer = response['answer']
        logger.info(f"Full response before cleanup: {raw_answer}")

        # Define regex pattern to match full <think>...</think> block
        think_block_pattern = re.compile(r"<think>.*?</think>", flags=re.DOTALL)

        # Check if full <think> block exists
        has_think_block = bool(think_block_pattern.search(raw_answer))

        # Clean the <think> block regardless
        cleaned_answer = think_block_pattern.sub("", raw_answer).strip()

        if not has_think_block:
            logger.warning("Missing full <think> block — retrying query once...")
            response_retry = qa_chain.invoke({"question": clean_query})
            raw_answer_retry = response_retry['answer']
            logger.info(f"Retry response: {raw_answer_retry}")
            cleaned_answer = think_block_pattern.sub("", raw_answer_retry).strip()
        else:
            logger.info("Full <think> block found and removed successfully")
        
        return cleaned_answer
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
