import streamlit as st
import uuid
from datetime import datetime
from models.chat import Chat

def create_new_chat() -> str:
    """Create a new chat and set it as current"""
    if not st.session_state.authenticated:
        return None
        
    chat = Chat(
        title=f"Chat {len(st.session_state.chats) + 1}",
        user=st.session_state.username
    )
    
    st.session_state.chats[chat.id] = chat.to_dict()
    st.session_state.current_chat_id = chat.id
    return chat.id

def delete_chat(chat_id: str) -> None:
    """Delete a chat by ID"""
    if chat_id in st.session_state.chats:
        del st.session_state.chats[chat_id]
        
        # If current chat was deleted, switch to another chat or create new one
        if st.session_state.current_chat_id == chat_id:
            if st.session_state.chats:
                # Get first chat belonging to current user
                user_chats = [
                    cid for cid, chat in st.session_state.chats.items()
                    if chat["user"] == st.session_state.username
                ]
                if user_chats:
                    st.session_state.current_chat_id = user_chats[0]
                else:
                    create_new_chat()
            else:
                create_new_chat()

def rename_chat(chat_id: str, new_title: str) -> None:
    """Rename a chat by ID"""
    if chat_id in st.session_state.chats:
        st.session_state.chats[chat_id]["title"] = new_title

def switch_chat(chat_id: str) -> None:
    """Switch to a different chat"""
    if chat_id in st.session_state.chats:
        st.session_state.current_chat_id = chat_id

def add_message(chat_id: str, role: str, text: str) -> None:
    """Add a message to the specified chat"""
    if chat_id in st.session_state.chats:
        st.session_state.chats[chat_id]["messages"].append({
            "role": role,
            "text": text
        })

def generate_bot_response(user_message: str) -> str:
    """Generate a response from the bot"""
    # In a real app, this would call an AI service or API
    # For now, we'll just echo the user's message
    return f"I heard you say: {user_message}"