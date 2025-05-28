# ICP_Verztec_Chatbot
This repository contains the code for an AI-powered chatbot designed for internal helpdesk support. The chatbot handles frequently asked questions, provides automated responses, and assists users with various queries. It is built with a modular and scalable architecture, making it easy to customize and extend for different use cases.

## Dependencies

To run this project, you will need to install the following Python packages:

- `pymupdf`
- `re`
- `ftfy`
- `os`
- `langchain`
- `langchain_groq`
- `langchain_core`
- `langchain_community`
- `dotenv`
- `langchain_huggingface`
- `langchain_community.chat_models`
### Installation (chatbot) *updated*

Website structure
chatbot/
├── index.html                   # Frontend UI entry point
├── app.js, view.js              # Frontend logic and rendering
├── model.js, controller.js      # Chatbot state and controller logic
├── styles.css                   # Chat UI styling
├── main.py                      # FastAPI backend server
├── chatbot_backend.py           # Python chatbot logic (FAISS + Groq)
├── faiss_local_BAAI.idx         # FAISS vector index file
├── faiss_local_BAAI_meta.json   # Metadata corresponding to FAISS index
├── requirements.txt             # Python dependencies
├── __pycache__                 # Python cache files (can be ignored)

Install dependencies:
pip install -r requirements.txt


How to run the application:
1. change directory of project to chatbot folder (cd chatbot) in terminal
2. "uvicorn main:app --reload" within terminal to start the FastAPI backend server
3. double click "index.html" file or go live using live server extension




### Installation (website_proto_2) *outdated*

You can install all the required dependencies by running:

```bash
pip install -r requirements.txt
'



Website Structure

This follows a variant of the MVC pattern, but it's not strictly MVC. It most closely resembles a modular MVVM (Model-View-ViewModel) or MVC-like architecture, customized for a Python app like Streamlit.

```text
chatbot_app/
├── app.py                  # Main application entry point
├── config/
│   └── settings.py         # Configuration settings
├── data/
│   └── dummy_data.py       # Dummy user data
├── models/
│   └── chat.py             # Chat data models
├── services/
│   ├── auth_service.py     # Authentication logic
│   └── chat_service.py     # Chat management logic
├── ui/
│   ├── auth_ui.py          # Login UI components
│   ├── chat_ui.py          # Chat interface UI components
│   └── sidebar_ui.py       # Sidebar UI components
└── utils/
    └── session_helper.py   # Session state management helpers


Install dependencies:
pip install -r requirements.txt


How to run the application:
1. Head to "view" on top bar and select "Terminal"
2. change directory using "cd website_proto_2" in terminal
3. run "streamlit run main.py"


Demo Accounts
You can use these accounts to test the application:
Username: user1, Password: password1
Username: user2, Password: password2
Username: admin, Password: admin123


