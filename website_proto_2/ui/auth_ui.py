import streamlit as st
from services.auth_service import authenticate

def render_login() -> None:
    """Render login interface"""
    with st.sidebar:
        st.title("Login")

#Login form
        st.text_input("Username", key="input_username")
        password = st.text_input("Password", type="password", key="input_password")

        if st.button("Login"):
            if authenticate(st.session_state.input_username, st.session_state.input_password):
                st.success("Login successful!")
                st.rerun()
            else:
                st.error("Invalid username or password")

#Help information
        with st.expander("Need help?"):
            st.info("""
            Try these demo accounts:
            
Username: user1, Password: password1
Username: user2, Password: password2
Username: admin, Password: admin123""")