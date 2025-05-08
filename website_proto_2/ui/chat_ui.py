import streamlit as st
from services.chat_service import add_message, generate_bot_response, create_new_chat

def render_chat_message(role: str, text: str) -> None:
    """Render a single chat message with appropriate styling"""
    if role == "user":
        st.markdown(f"""
        <div style="background-color: #e6f7ff; padding: 10px; border-radius: 10px; 
                    margin: 5px 0 5px auto; max-width: 80%; text-align: right;">
            <strong>You:</strong> {text}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background-color: #f0f0f0; padding: 10px; border-radius: 10px; 
                    margin: 5px auto 5px 0; max-width: 80%;">
            <strong>Bot:</strong> {text}
        </div>
        """, unsafe_allow_html=True)

def render_chat_interface() -> None:
    """Render the main chat interface"""
    # Ensure a chat is selected
    if st.session_state.current_chat_id is None and st.session_state.chats:
        # Find a chat for current user
        user_chats = [
            cid for cid, chat in st.session_state.chats.items()
            if chat["user"] == st.session_state.username
        ]
        if user_chats:
            st.session_state.current_chat_id = user_chats[0]
        else:
            create_new_chat()
    
    # Display current chat if available
    if st.session_state.current_chat_id is not None:
        current_chat = st.session_state.chats[st.session_state.current_chat_id]
        
        # Chat title and info
        st.title(current_chat["title"])
        st.caption(f"Chat ID: {st.session_state.current_chat_id[:8]}... | Created: {current_chat['timestamp']}")
        
        # Display chat messages in scrollable container
        chat_container = st.container(height=400, border=False)
        with chat_container:
            for msg in current_chat["messages"]:
                render_chat_message(msg["role"], msg["text"])
        
        # User input area
        with st.container():
            with st.form("chat_input_form", clear_on_submit=True):
                user_input = st.text_input("Message:", key="user_message")
                submit = st.form_submit_button("Send")
                
                if submit and user_input:
                    # Add user message
                    add_message(st.session_state.current_chat_id, "user", user_input)
                    
                    # Generate and add bot response
                    bot_response = generate_bot_response(user_input)
                    add_message(st.session_state.current_chat_id, "bot", bot_response)
                    
                    # Refresh UI
                    st.rerun()
    
    else:
        # This should only happen if all chats were deleted
        st.info("No chats available. Create a new chat from the sidebar.")