import streamlit as st

def initialize_session_state() -> None:
    """Initialize all required session state variables if they don't exist"""
    # Authentication state
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

#Current user
    if "username" not in st.session_state:
        st.session_state.username = None

#Currently selected chat
    if "current_chat_id" not in st.session_state:
        st.session_state.current_chat_id = None

#All chats (dictionary of chat_id -> chat_data)
    if "chats" not in st.session_state:
        st.session_state.chats = {}

#Chat being edited (for rename functionality)
    if "edit_chat_id" not in st.session_state:
        st.session_state.edit_chat_id = None