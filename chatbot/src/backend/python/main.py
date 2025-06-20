from fastapi import FastAPI, HTTPException, Request, File, UploadFile, Security, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
import os
import uvicorn

from chatbot import (
    generate_answer,
    generate_answer_histoy_retrieval,
    memory, 
    logger, 
    index
)
from memory_retrieval import retrieve_user_messages_and_scores
from Freq_queries import (get_suggestions)
from fileUpload import process_upload
from fastapi.staticfiles import StaticFiles
app = FastAPI()
app.mount("/images", StaticFiles(directory="data/images"), name="images")

app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600
)

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
    images: Optional[List[str]] = None  # Add images field

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
        
@app.post("/history")
async def history_retreival(request: Request):
    body = await request.json()
    user_id = body.get("user_id")
    chat_id = body.get("chat_id")

    if not user_id or not chat_id:
        return JSONResponse(
            status_code=400,
            content={"error": "Missing 'user_id' or 'chat_id'"}
        )

    results = retrieve_user_messages_and_scores(user_id, chat_id)
    results.reverse()  # Optional: oldest to newest
    return JSONResponse(content=results)


@app.post("/chatbot")
async def chat_endpoint(request: ChatRequest):
    logger.info(f"Received chat request: {request}")
    try:
        if not request.message.strip():
            raise HTTPException(status_code=400, detail="Message cannot be empty")
        
        if index is None:
            raise HTTPException(status_code=503, detail="Search index is not available")
        
        response_message, image_list = generate_answer(request.message, memory)
        # response_message, image_list= generate_answer_histoy_retrieval(request.message , request.user_id, request.chat_id)
        logger.info(f"Generated response: {response_message}")
        logger.info(f"Image list: {image_list}")
        
        
        
        return ChatResponse(
            message=response_message,
            user_id=request.user_id,
            timestamp=datetime.utcnow().isoformat(),
            success=True,
            images=image_list
        )
        

    except Exception as e:
        logger.error(f"Error processing chat request: {str(e)}", exc_info=True)
        return ChatResponse(
            message="An error occurred while processing your request. Please try again later.",
            user_id=request.user_id,
            timestamp=datetime.utcnow().isoformat(),
            success=False,
            error=str(e),
            images=None
        )
        
    from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from fastapi.responses import JSONResponse

@app.post("/frequent")
async def get_frequent_queries():
    logger.info("Received request for frequent queries")

    try:
        if index is None:
            logger.warning("Search index is not available")
            return JSONResponse(content=[], status_code=200)

        top_queries = get_suggestions()
        logger.info(f"Top frequent queries: {top_queries}")

        # Fallback if no queries
        fallback = [
            "What are the pantry rules?",
            "What is the leave policy?",
            "How do I upload e-invoices?"
        ]
        if not isinstance(top_queries, list) or not top_queries:
            top_queries = fallback

        return JSONResponse(content=top_queries, status_code=200)

    except Exception as e:
        logger.error(f"Error retrieving frequent queries: {str(e)}")
        # Also fallback on error
        fallback = [
            "What are the pantry rules?",
            "What is the leave policy?",
            "How do I upload e-invoices?"
        ]
        return JSONResponse(content=fallback, status_code=200)

@app.delete("/history")
async def clear_chat_history():
    try:
        return {"success": True, "message": "Chat history cleared successfully"}
    except Exception as e:
        logger.error(f"Error clearing chat history: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to clear chat history")

@app.post("/internal/upload")
async def upload_file(
    file: UploadFile = File(...),
    authorization: str = Header(None)
):
    """Internal endpoint to handle file uploads"""
    if not authorization or not authorization.startswith('Bearer '):
        raise HTTPException(status_code=403, detail="Invalid authorization")
    return await process_upload(file)


if __name__ == "__main__":
    try:
        print("Starting server on port 3000")
        uvicorn.run(app, host="0.0.0.0", port=3000, log_level="debug")
    except Exception as e:
        print(f"Failed to start server: {e}")
