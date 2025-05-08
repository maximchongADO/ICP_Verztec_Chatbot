import streamlit as st
from config.settings import setup_page_config
from utils.session_helper import initialize_session_state
from ui.auth_ui import render_login
from ui.sidebar_ui import render_sidebar
from ui.chat_ui import render_chat_interface

def main():
    """Main application entry point"""
    # Set up page configuration
    setup_page_config()

#Initialize session state variables
    initialize_session_state()

#Display login screen if not authenticated
    if not st.session_state.authenticated:
        render_login()
    else:
        # Render sidebar with chat management options
        render_sidebar()

#Render main chat interface
        render_chat_interface()

if __name__ == "__main__":
    main()