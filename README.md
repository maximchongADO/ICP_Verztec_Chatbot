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
cd chatbot first
1. cd src
2. npm install - > only need for first time or when adding new packages
3. npm run seed - > only need for first time or when updating database 
4. npm start

then go to the localhost:3000 to access the website

to enable the chatbot
open a new terminal, then
1. cd src
2. cd backend
3. cd python
4. python -m venv .venv
5. .venv\Scripts\activate.bat
6. pip install -r requirements.txt - > only need for first time OR when u add new requirements
7. python main.py

if u want to change the python chatbot logic make sure to include:
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Express.js server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

and ensure that it is a fastapi app and routes properly to the existing backend router so that the frontend can access the chatbot




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



Demo Accounts
You can use these accounts to test the application:
Username: Maxim , Password : maximchong1
=

