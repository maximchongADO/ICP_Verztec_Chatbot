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
    index,agentic_bot_v1,
    execute_confirmed_tool
)
from memory_retrieval import (retrieve_user_messages_and_scores,get_all_chats_with_messages_for_user)
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
    chat_history: Optional[List[str]] = []
    user_id: Optional[str] = None
    chat_id: Optional[str] = None
    tool_identified: Optional[str] = None  # For tool confirmation requests
    tool_confidence: Optional[str] = None  # For tool confirmation requests
    user_description: Optional[str] = None  # For HR escalation description

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

@app.get("/chat_history")
async def get_all_chats(user_id: str):
    """
    Retrieve all chat sessions and their messages for a user.
    Returns a list of chat sessions with their chat_id and messages.
    """
    try:
        results = get_all_chats_with_messages_for_user(user_id)
        results.reverse()  # Optional: oldest to newest
        return JSONResponse(content=results)
    except Exception as e:
        logger.error(f"Error retrieving chats for user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve chat history")
@app.post('/chatbot_avatar')
async def avatar_endpoint(request:ChatRequest):
    logger.info(f"Received chat request: {request}")
    try:
        if not request.message.strip():
            raise HTTPException(status_code=400, detail="Message cannot be empty")
        
        if index is None:
            raise HTTPException(status_code=503, detail="Search index is not available")
        
        #response_message, image_list = generate_answer(request.message, memory)
        response_message, image_list= generate_answer_histoy_retrieval(request.message , request.user_id, request.chat_id)
        logger.info(f"Generated response:S {response_message}")
        logger.info(f"Image list: {image_list}")
        
        
        
        return ChatResponse(
            message=response_message,
            user_id=request.user_id,
            timestamp=datetime.utcnow().isoformat(),
            success=True,
            images=None
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
@app.post("/tool_confirmation")
async def tool_confirmation(request: ChatRequest):
    """
    Endpoint to execute confirmed tool and return response with original user and chat ID.
    """
    logger.info(f"Received tool confirmation request: {request}")
    
    try:
        if not request.message.strip():
            raise HTTPException(status_code=400, detail="Message cannot be empty")
        
        if index is None:
            raise HTTPException(status_code=503, detail="Search index is not available")
        
        # Use the cached tool identification from the frontend
        tool_identified = request.tool_identified or "none"
        tool_confidence = request.tool_confidence or "unknown"
        user_description = request.user_description or None
        
        logger.info(f"Using cached tool identification: {tool_identified} (confidence: {tool_confidence})")
        if user_description:
            logger.info(f"User provided description: {user_description[:100]}...")
        
        # Execute the confirmed tool using the cached identification
        if tool_identified and tool_identified != 'none':
            logger.info(f"Executing confirmed tool: {tool_identified}")
            response_data = execute_confirmed_tool(tool_identified, request.message, request.user_id, request.chat_id, user_description)
        else:
            # No tool to execute, return error
            logger.warning("No tool identified for execution in confirmation request")
            raise HTTPException(status_code=400, detail="No tool identified for execution")
        
        # Handle both old tuple format and new structured format
        if isinstance(response_data, dict):
            # New structured format
            response_message = response_data.get('text', '')
            image_list = response_data.get('images', [])
            sources = response_data.get('sources', [])
            tool_used = response_data.get('tool_used', True)  # Default to True since this is tool confirmation
            final_tool_identified = response_data.get('tool_identified', tool_identified)
            final_tool_confidence = response_data.get('tool_confidence', tool_confidence)
        else:
            # Old tuple format (fallback)
            response_message, image_list = response_data
            sources = []
            tool_used = True  # Tool was confirmed and executed
            final_tool_identified = tool_identified
            final_tool_confidence = tool_confidence
        
        logger.info(f"Generated tool confirmation response: {response_message}")
        logger.info(f"Image list: {image_list}")
        logger.info(f"Sources: {sources}")
        logger.info(f"Tool executed - identified: {final_tool_identified}, used: {tool_used}")
        
        return {
            "message": response_message,
            "user_id": request.user_id,
            "chat_id": request.chat_id,
            "timestamp": datetime.utcnow().isoformat(),
            "success": True,
            "images": image_list,
            "sources": sources,
            "tool_used": tool_used,
            "tool_identified": final_tool_identified,
            "tool_confidence": final_tool_confidence
        }
        
    except Exception as e:
        logger.error(f"Error processing tool confirmation request: {str(e)}", exc_info=True)
        return {
            "message": "An error occurred while processing your request. Please try again later.",
            "user_id": request.user_id,
            "chat_id": request.chat_id,
            "timestamp": datetime.utcnow().isoformat(),
            "success": False,
            "error": str(e),
            "images": [],
            "sources": [],
            "tool_used": False,
            "tool_identified": "none",
            "tool_confidence": "error"
        }
@app.post("/chatbot")
async def chat_endpoint(request: ChatRequest):
    logger.info(f"Received chat request: {request}")
    try:
        if not request.message.strip():
            raise HTTPException(status_code=400, detail="Message cannot be empty")
        
        if index is None:
            raise HTTPException(status_code=503, detail="Search index is not available")
        
       
        #response_message, image_list = agentic_bot_v1(request.message, request.user_id, request.chat_id)
        response_data = generate_answer_histoy_retrieval(request.message, request.user_id, request.chat_id)
        
        # Handle both old tuple format and new structured format
        if isinstance(response_data, dict):
            # New structured format
            response_message = response_data.get('text', '')
            image_list = response_data.get('images', [])
            sources = response_data.get('sources', [])
            tool_used = response_data.get('tool_used', False)
            tool_identified = response_data.get('tool_identified', 'none')
            tool_confidence = response_data.get('tool_confidence', '')
        else:
            # Old tuple format (fallback)
            response_message, image_list = response_data
            sources = []
            tool_used = False
            tool_identified = 'none'
            tool_confidence = ''
        
        logger.info(f"Generated response: {response_message}")
        logger.info(f"Image list: {image_list}")
        logger.info(f"Sources: {sources}")
        logger.info(f"Tool identified: {tool_identified}, Tool used: {tool_used}")
        
        return {
            "message": response_message,
            "user_id": request.user_id,
            "timestamp": datetime.utcnow().isoformat(),
            "success": True,
            "images": image_list,
            "sources": sources,
            "tool_used": tool_used,
            "tool_identified": tool_identified,
            "tool_confidence": tool_confidence
        }
        

    except Exception as e:
        logger.error(f"Error processing chat request: {str(e)}", exc_info=True)
        return {
            "message": "An error occurred while processing your request. Please try again later.",
            "user_id": request.user_id,
            "timestamp": datetime.utcnow().isoformat(),
            "success": False,
            "error": str(e),
            "images": [],
            "sources": [],
            "tool_used": False,
            "tool_identified": "none",
            "tool_confidence": "error"
        }
        
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
