import streamlit as st

def setup_page_config():
    """Configure page settings"""
    st.set_page_config(
        page_title="Enhanced Chatbot",
        page_icon="💬",
        layout="wide",
        initial_sidebar_state="expanded"
    )