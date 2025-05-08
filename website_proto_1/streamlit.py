# Import library 
import streamlit as st
import uuid
from datetime import datetime

# Set page configuration
st.set_page_config(
    page_title="Enhanced Chatbot",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "username" not in st.session_state:
    st.session_state.username = None
if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = None
if "chats" not in st.session_state:
    st.session_state.chats = {}

# Dummy user data
USERS = {
    "user1": "password1",
    "user2": "password2",
    "admin": "admin123"
}

def login():
    st.session_state.authenticated = True
    st.session_state.username = st.session_state.input_username
    if not any(chat["user"] == st.session_state.username for chat in st.session_state.chats.values()):
        create_new_chat()

def logout():
    st.session_state.authenticated = False
    st.session_state.username = None
    st.session_state.current_chat_id = None

def create_new_chat():
    chat_id = str(uuid.uuid4())
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    st.session_state.chats[chat_id] = {
        "title": f"Chat {len(st.session_state.chats) + 1}",
        "timestamp": timestamp,
        "messages": [],
        "user": st.session_state.username
    }
    st.session_state.current_chat_id = chat_id
    return chat_id

def delete_chat(chat_id):
    if chat_id in st.session_state.chats:
        del st.session_state.chats[chat_id]
        if st.session_state.current_chat_id == chat_id:
            if st.session_state.chats:
                st.session_state.current_chat_id = next(iter(st.session_state.chats))
            else:
                create_new_chat()

def rename_chat(chat_id, new_title):
    if chat_id in st.session_state.chats:
        st.session_state.chats[chat_id]["title"] = new_title

def switch_chat(chat_id):
    st.session_state.current_chat_id = chat_id

# Login sidebar
if not st.session_state.authenticated:
    with st.sidebar:
        st.title("Login")
        st.text_input("Username", key="input_username")
        password = st.text_input("Password", type="password", key="input_password")
        
        if st.button("Login"):
            if (st.session_state.input_username in USERS and 
                USERS[st.session_state.input_username] == st.session_state.input_password):
                login()
                st.success("Login successful!")
                st.rerun()
            else:
                st.error("Invalid username or password")
        
        st.info("Try usernames: user1, user2, admin with matching passwords")

# Main application after authentication
else:
    # Sidebar for chat management
    with st.sidebar:
        st.title(f"Welcome, {st.session_state.username}")
        
        if st.button("New Chat"):
            create_new_chat()
            st.rerun()
        
        st.divider()
        st.subheader("Your Chats")
        
        # List all chats for the current user
        user_chats = {k: v for k, v in st.session_state.chats.items() if v["user"] == st.session_state.username}
        
        if not user_chats:
            create_new_chat()
            st.rerun()
        
        for chat_id, chat_data in user_chats.items():
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                if st.button(f"{chat_data['title']}", key=f"btn_{chat_id}"):
                    switch_chat(chat_id)
                    st.rerun()
            with col2:
                if st.button("✏️", key=f"edit_{chat_id}"):
                    st.session_state.edit_chat_id = chat_id
                    st.session_state.edit_chat_title = chat_data["title"]
                    st.rerun()
            with col3:
                if st.button("🗑️", key=f"delete_{chat_id}"):
                    delete_chat(chat_id)
                    st.rerun()
            
            # Show timestamp in smaller text
            st.caption(f"Created: {chat_data['timestamp']}")
        
        # Chat renaming form
        if "edit_chat_id" in st.session_state and st.session_state.edit_chat_id:
            st.text_input(
                "New title", 
                value=st.session_state.edit_chat_title,
                key="new_chat_title"
            )
            if st.button("Save"):
                rename_chat(st.session_state.edit_chat_id, st.session_state.new_chat_title)
                st.session_state.edit_chat_id = None
                st.rerun()
            if st.button("Cancel"):
                st.session_state.edit_chat_id = None
                st.rerun()
        
        st.divider()
        if st.button("Logout"):
            logout()
            st.rerun()

    # Main chat interface
    if st.session_state.current_chat_id is None and st.session_state.chats:
        st.session_state.current_chat_id = next(iter(st.session_state.chats))
    
    if st.session_state.current_chat_id is not None:
        current_chat = st.session_state.chats[st.session_state.current_chat_id]
        
        # Chat title and info
        st.title(current_chat["title"])
        st.caption(f"Chat ID: {st.session_state.current_chat_id[:8]}... | Created: {current_chat['timestamp']}")
        
        # Display chat messages with better styling
        chat_container = st.container()
        with chat_container:
            for msg in current_chat["messages"]:
                if msg["role"] == "user":
                    st.markdown(f"""
                    <div style="background-color: #e6f7ff; padding: 10px; border-radius: 10px; margin: 5px 0 5px auto; max-width: 80%; text-align: right;">
                        <strong>You:</strong> {msg["text"]}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="background-color: #f0f0f0; padding: 10px; border-radius: 10px; margin: 5px auto 5px 0; max-width: 80%;">
                        <strong>Bot:</strong> {msg["text"]}
                    </div>
                    """, unsafe_allow_html=True)
        
        # User input
        user_input = st.text_input("Message:", key="user_message")
        
        if user_input:
            # Add user message
            current_chat["messages"].append({"role": "user", "text": user_input})
            
            # Bot response (replace with actual logic or API calls)
            bot_response = f"I heard you say: {user_input}"
            current_chat["messages"].append({"role": "bot", "text": bot_response})
            
            # Clear input and refresh
            st.rerun()

    else:
        # This should only happen if all chats were deleted
        st.info("No chats available. Create a new chat from the sidebar.")
