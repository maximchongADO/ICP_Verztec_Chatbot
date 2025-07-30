from fastapi import FastAPI, HTTPException, Request, File, UploadFile, Security, Header, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
import pymysql
from datetime import datetime, timedelta
from fastapi.concurrency import run_in_threadpool
from datetime import datetime
import time
import pymysql
import asyncio

import os
import uvicorn
import uuid
from chatbot import (
   # generate_answer,
    generate_answer_histoy_retrieval,
    memory, 
    logger, 
    index,agentic_bot_v1,
    global_tools
)
from tool_executors import execute_confirmed_tool
from memory_retrieval import (retrieve_user_messages_and_scores,get_all_chats_with_messages_for_user, delete_messages_by_user_and_chat)
from Freq_queries import (get_suggestions)
from fileUpload import process_upload, get_system_config
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

class MeetingConfirmationRequest(BaseModel):
    user_response: str
    meeting_request_id: str
    user_id: str
    chat_id: str
    original_details: dict

class FrequentRequest(BaseModel):
    user_id: Optional[str] = None

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

@app.get("/api/chatbot/history")
async def get_chatbot_history(user_id: str):
    """
    Frontend-compatible endpoint for retrieving all chat sessions for a user.
    This endpoint matches the frontend's expected URL pattern.
    """
    try:
        results = get_all_chats_with_messages_for_user(user_id)
        results.reverse()  # Optional: oldest to newest
        return JSONResponse(content=results)
    except Exception as e:
        logger.error(f"Error retrieving chats for user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve chat history")

@app.get("/api/chatbot/history/{chat_id}")
async def get_specific_chatbot_history(chat_id: str, user_id: str):
    """
    Frontend-compatible endpoint for retrieving a specific chat's messages.
    This endpoint matches the frontend's expected URL pattern.
    """
    try:
        if not user_id or not chat_id:
            raise HTTPException(status_code=400, detail="Missing user_id or chat_id")
        
        results = retrieve_user_messages_and_scores(user_id, chat_id)
        results.reverse()  # Optional: oldest to newest
        return JSONResponse(content=results)
    except Exception as e:
        logger.error(f"Error retrieving chat {chat_id} for user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve chat messages")

@app.delete("/api/chatbot/history/{chat_id}")
async def delete_chatbot_history(chat_id: str, user_id: str):
    """
    Frontend-compatible endpoint for deleting a specific chat.
    This endpoint matches the frontend's expected URL pattern.
    """
    try:
        if not user_id or not chat_id:
            raise HTTPException(status_code=400, detail="Missing user_id or chat_id")
        
        logger.info(f"Deleting chat {chat_id} for user {user_id}")
        
        # Use the existing function from memory_retrieval.py
        delete_messages_by_user_and_chat(user_id, chat_id)
        
        logger.info(f"Successfully deleted chat {chat_id} for user {user_id}")
        return {"success": True, "message": f"Chat {chat_id} deleted successfully"}
        
    except Exception as e:
        logger.error(f"Error deleting chat {chat_id} for user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete chat: {str(e)}")

@app.post("/api/chatbot/newchat")
async def create_new_chat(request: Request):
    """
    Frontend-compatible endpoint for creating a new chat session.
    This endpoint matches the frontend's expected URL pattern.
    """
    try:
        body = await request.json()
        user_id = body.get("user_id", "defaultUser")
        
        import uuid
        new_chat_id = str(uuid.uuid4())
        
        logger.info(f"Created new chat {new_chat_id} for user {user_id}")
        return JSONResponse(content={
            "success": True, 
            "chat_id": new_chat_id,
            "message": "New chat session created"
        })
        
    except Exception as e:
        logger.error(f"Error creating new chat: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create new chat: {str(e)}")

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
async def avatar_endpoint(request: ChatRequest):
    logger.info(f"Received avatar chat request: {request}")
    try:
        if not request.message.strip():
            raise HTTPException(status_code=400, detail="Message cannot be empty")
        
        if index is None:
            raise HTTPException(status_code=503, detail="Search index is not available")
        
        # Get response from chatbot with proper structure
        response_data = generate_answer_histoy_retrieval(request.message, request.user_id, request.chat_id)
        
        # Handle both old tuple format and new structured format
        if isinstance(response_data, dict):
            response_message = response_data.get('text', '')
            image_list = response_data.get('images', [])
            sources = response_data.get('sources', [])
        else:
            # Old tuple format (fallback)
            response_message, image_list = response_data
            sources = []
        
        logger.info(f"Generated avatar response: {response_message}")
        logger.info(f"Image list: {image_list}")
        
        # Return structured response for avatar
        return {
            "messages": [{
                "type": "bot",
                "text": response_message,
                "id": f"msg_{int(time.time() * 1000)}",  # Add unique ID
                "timestamp": datetime.utcnow().isoformat(),
                "audio": None,  # Will be generated by TTS controller
                "lipsync": None  # Will be generated by TTS controller
            }],
            "success": True,
            "sources": sources,
            "images": image_list
        }

    except Exception as e:
        logger.error(f"Error processing avatar chat request: {str(e)}", exc_info=True)
        return {
            "messages": [{
                "type": "bot", 
                "text": "An error occurred while processing your request. Please try again later.",
                "id": f"error_{int(time.time() * 1000)}",
                "timestamp": datetime.utcnow().isoformat(),
                "audio": None,
                "lipsync": None
            }],
            "success": False,
            "error": str(e)
        }

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
            response_data = execute_confirmed_tool(
                tool_identified, 
                request.message, 
                request.user_id, 
                request.chat_id, 
                user_description,
                global_tools
            )
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
DB_CONFIG = {
    'host': 'localhost',
    'user': 'chatbot_user',
    'password': 'strong_password',
    'database': 'chatbot_db',
    'cursorclass': pymysql.cursors.Cursor,
    'autocommit': True
}

def query_matching_response(message: str, after_time: str):
    conn = pymysql.connect(**DB_CONFIG)
    cursor = conn.cursor()
    try:
        cursor.execute("""
            SELECT bot_response FROM chat_logs
            WHERE user_message = %s AND timestamp > %s
            ORDER BY timestamp DESC
            LIMIT 1
        """, (message, after_time))
        result = cursor.fetchone()
        return result[0] if result else None
    finally:
        cursor.close()
        conn.close()

@app.post("/avatar_msgmatchchatbot")
async def avatar_msgmatchchatbot(request: ChatRequest):
    logger.info(f"Received avatar message match request: {request}")
    msg = request.message.strip()
    time_now = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')

    timeout_seconds = 60
    polling_interval = 2  # how often to check, in seconds
    end_time = datetime.utcnow() + timedelta(seconds=timeout_seconds)

    while datetime.utcnow() < end_time:
        try:
            matched_response = await run_in_threadpool(query_matching_response, msg, time_now)
            if matched_response:
                return {
                    "messages": [{
                        "type": "bot",
                        "text": matched_response,
                        "id": f"match_{int(time.time() * 1000)}",
                        "timestamp": datetime.utcnow().isoformat(),
                        "audio": None,
                        "lipsync": None
                    }],
                    "success": True,
                    "error": None
                }
        except Exception as e:
            logger.error(f"Database error during polling: {e}")
            raise HTTPException(status_code=500, detail="Database polling error.")

        await asyncio.sleep(polling_interval)

    # Timeout occurred
    return {
        "messages": [{
            "type": "bot",
            "text": "An error occurred while processing your request. Please try again later.",
            "id": f"error_{int(time.time() * 1000)}",
            "timestamp": datetime.utcnow().isoformat(),
            "audio": None,
            "lipsync": None
        }],
        "success": False,
        "error": "No matching record found within timeout."
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
            meeting_confirmation = response_data.get('meeting_confirmation', None)
        else:
            # Old tuple format (fallback)
            response_message, image_list = response_data
            sources = []
            tool_used = False
            tool_identified = 'none'
            tool_confidence = ''
            meeting_confirmation = None
        
        logger.info(f"Generated response: {response_message}")
        logger.info(f"Image list: {image_list}")
        logger.info(f"Sources: {sources}")
        logger.info(f"Tool identified: {tool_identified}, Tool used: {tool_used}")
        
        response_dict = {
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
        
        # Add meeting_confirmation data if present
        if meeting_confirmation:
            response_dict["meeting_confirmation"] = meeting_confirmation
            
        return response_dict
        

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
        
@app.post("/frequent")
async def get_frequent_queries(request: FrequentRequest):
    logger.info(f"Received request for frequent queries from user: {request.user_id}")

    try:
        if index is None:
            logger.warning("Search index is not available")
            return JSONResponse(content=[], status_code=200)

        # Pass user_id to get_suggestions for regional filtering
        top_queries = get_suggestions(user_id=request.user_id)
        logger.info(f"Top frequent queries for user {request.user_id}: {top_queries}")

        # Regional fallbacks based on user info if available
        fallback = get_regional_fallback(request.user_id)
        
        if not isinstance(top_queries, list) or not top_queries:
            top_queries = fallback

        return JSONResponse(content=top_queries, status_code=200)

    except Exception as e:
        logger.error(f"Error retrieving frequent queries for user {request.user_id}: {str(e)}")
        # Also fallback on error
        fallback = get_regional_fallback(request.user_id)
        return JSONResponse(content=fallback, status_code=200)


def get_regional_fallback(user_id: Optional[str]) -> List[str]:
    """Get fallback suggestions - always returns generic suggestions since we can't predict FAISS content"""
    return [
        "What are the pantry rules?",
        "What is the leave policy?",
        "How do I upload e-invoices?",
        "How do I use the phone system?"
    ]

@app.delete("/history")
async def clear_chat_history():
    try:
        return {"success": True, "message": "Chat history cleared successfully"}
    except Exception as e:
        logger.error(f"Error clearing chat history: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to clear chat history")

@app.delete("/history/{chat_id}")
async def delete_specific_chat(chat_id: str, user_id: str):
    """
    Delete a specific chat by chat_id and user_id from the database.
    """
    try:
        if not user_id or not chat_id:
            raise HTTPException(status_code=400, detail="Missing user_id or chat_id")
        
        logger.info(f"Deleting chat {chat_id} for user {user_id}")
        
        # Use the existing function from memory_retrieval.py
        delete_messages_by_user_and_chat(user_id, chat_id)
        
        logger.info(f"Successfully deleted chat {chat_id} for user {user_id}")
        return {"success": True, "message": f"Chat {chat_id} deleted successfully"}
        
    except Exception as e:
        logger.error(f"Error deleting chat {chat_id} for user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete chat: {str(e)}")

@app.post("/internal/upload")
async def upload_file(
    file: UploadFile = File(...),
    country: str = Form(...),
    department: str = Form(...),
    authorization: str = Header(None)
):
    """Internal endpoint to handle file uploads with country and department"""
    if not authorization or not authorization.startswith('Bearer '):
        raise HTTPException(status_code=403, detail="Invalid authorization")
    return await process_upload(file, country, department)

@app.get("/internal/upload/config")
async def get_upload_config(authorization: str = Header(None)):
    """Internal endpoint to get upload system configuration"""
    if not authorization or not authorization.startswith('Bearer '):
        raise HTTPException(status_code=403, detail="Invalid authorization")
    return await get_system_config()

@app.post("/meeting_confirmation")
async def meeting_confirmation(request: MeetingConfirmationRequest):
    """
    Endpoint to handle meeting confirmation responses (confirm, modify, cancel).
    """
    logger.info(f"Received meeting confirmation request: {request}")
    
    try:
        if not request.user_response.strip():
            raise HTTPException(status_code=400, detail="User response cannot be empty")
        
        if not request.meeting_request_id:
            raise HTTPException(status_code=400, detail="Meeting request ID is required")
        
        # Import the meeting confirmation handler from tool_executors
        from tool_executors import handle_meeting_confirmation_response
        
        logger.info(f"Processing meeting confirmation: {request.user_response} for meeting {request.meeting_request_id}")
        
        # Execute the meeting confirmation response handler
        response_data = handle_meeting_confirmation_response(
            user_response=request.user_response,
            meeting_request_id=request.meeting_request_id,
            user_id=request.user_id,
            chat_id=request.chat_id,
            original_details=request.original_details,
            store_chat_log_updated_func=None  # Will be imported dynamically if needed
        )
        
        # Extract response data
        if isinstance(response_data, dict):
            response_message = response_data.get('text', '')
            image_list = response_data.get('images', [])
            sources = response_data.get('sources', [])
            tool_used = response_data.get('tool_used', True)
            tool_identified = response_data.get('tool_identified', 'meeting_response')
            tool_confidence = response_data.get('tool_confidence', 'executed')
        else:
            # Fallback for unexpected format
            response_message = str(response_data)
            image_list = []
            sources = []
            tool_used = True
            tool_identified = 'meeting_response'
            tool_confidence = 'executed'
        
        logger.info(f"Generated meeting confirmation response: {response_message[:100]}...")
        logger.info(f"Meeting action executed - identified: {tool_identified}, used: {tool_used}")
        
        return {
            "message": response_message,
            "user_id": request.user_id,
            "chat_id": request.chat_id,
            "timestamp": datetime.utcnow().isoformat(),
            "success": True,
            "images": image_list,
            "sources": sources,
            "tool_used": tool_used,
            "tool_identified": tool_identified,
            "tool_confidence": tool_confidence
        }
        
    except Exception as e:
        logger.error(f"Error processing meeting confirmation: {str(e)}", exc_info=True)
        return {
            "message": "Sorry, there was an error processing your meeting response. Please try again or contact support.",
            "user_id": request.user_id,
            "chat_id": request.chat_id,
            "timestamp": datetime.utcnow().isoformat(),
            "success": False,
            "error": str(e),
            "images": [],
            "sources": [],
            "tool_used": False,
            "tool_identified": "meeting_confirmation_error",
            "tool_confidence": "error"
        }

if __name__ == "__main__":
    try:
        print("Starting server on port 3000")
        uvicorn.run(app, host="0.0.0.0", port=3000, log_level="debug")
    except Exception as e:
        print(f"Failed to start server: {e}")
