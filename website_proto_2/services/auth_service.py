import streamlit as st
from data.dummy_data import USERS
from services.chat_service import create_new_chat

def authenticate(username: str, password: str) -> bool:
    """Validate user credentials and authenticate user"""
    if username in USERS and USERS[username] == password:
        st.session_state.authenticated = True
        st.session_state.username = username

#Create a new chat if user doesn't have any
        user_has_chats = any(
            chat["user"] == username 
            for chat in st.session_state.chats.values()
        )

        if not user_has_chats:
            create_new_chat()

        return True
    return False

def logout() -> None:
    """Log user out and reset session state"""
    st.session_state.authenticated = False
    st.session_state.username = None
    st.session_state.current_chat_id = None