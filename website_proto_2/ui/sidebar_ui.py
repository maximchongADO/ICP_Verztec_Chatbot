import streamlit as st
from services.auth_service import logout
from services.chat_service import create_new_chat, delete_chat, rename_chat, switch_chat

def render_sidebar() -> None:
    """Render sidebar with chat management options"""
    with st.sidebar:
        st.title(f"Welcome, {st.session_state.username}")
        
        # New chat button
        if st.button("New Chat", use_container_width=True):
            create_new_chat()
            st.rerun()
        
        st.divider()
        st.subheader("Your Chats")
        
        # List all chats for the current user
        user_chats = {
            k: v for k, v in st.session_state.chats.items() 
            if v["user"] == st.session_state.username
        }
        
        if not user_chats:
            create_new_chat()
            st.rerun()
        
        # Display chats with options to switch, edit, or delete
        for chat_id, chat_data in user_chats.items():
            # Highlight current chat
            is_current = chat_id == st.session_state.current_chat_id
            container = st.container(border=is_current)
            
            with container:
                col1, col2, col3 = st.columns([5, 1, 1])
                
                # Chat title - clickable to switch
                with col1:
                    title_text = chat_data["title"]
                    if is_current:
                        title_text = f"➤ {title_text}"
                        
                    if st.button(title_text, key=f"btn_{chat_id}", use_container_width=True):
                        switch_chat(chat_id)
                        st.rerun()
                
                # Edit button
                with col2:
                    if st.button("✏️", key=f"edit_{chat_id}"):
                        st.session_state.edit_chat_id = chat_id
                        st.session_state.edit_chat_title = chat_data["title"]
                        st.rerun()
                
                # Delete button
                with col3:
                    if st.button("🗑️", key=f"delete_{chat_id}"):
                        delete_chat(chat_id)
                        st.rerun()
                
                # Show timestamp in smaller text
                st.caption(f"Created: {chat_data['timestamp']}")
        
        # Chat renaming form
        if "edit_chat_id" in st.session_state and st.session_state.edit_chat_id:
            with st.form("rename_chat_form"):
                st.subheader("Rename Chat")
                new_title = st.text_input(
                    "New title", 
                    value=st.session_state.edit_chat_title,
                    key="new_chat_title"
                )
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.form_submit_button("Save"):
                        rename_chat(st.session_state.edit_chat_id, new_title)
                        st.session_state.edit_chat_id = None
                        st.rerun()
                with col2:
                    if st.form_submit_button("Cancel"):
                        st.session_state.edit_chat_id = None
                        st.rerun()
        
        # Logout button at bottom of sidebar
        st.divider()
        if st.button("Logout", use_container_width=True):
            logout()
            st.rerun()