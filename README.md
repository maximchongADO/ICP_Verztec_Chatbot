# ICP_Verztec_Chatbot

An AI-powered chatbot designed for internal helpdesk support, featuring:
- Automated responses to FAQs
- Document processing and analysis
- User authentication
- File upload capabilities
- Multi-format document support (PDF, DOCX, etc.)

## Project Structure

```
chatbot/
├── src/                          # Source directory
│   ├── app.js                    # Express server entry point
│   ├── package.json              # Node.js dependencies
│   │
│   ├── backend/
│   │   ├── controllers/          # Express.js controllers
│   │   │   ├── chatbotController.js    # Chatbot logic
│   │   │   ├── fileUploadController.js  # File upload handling
│   │   │   └── userController.js        # User management
│   │   │
│   │   ├── database/            # Database configuration
│   │   │   ├── dbConfig.js      # Database settings
│   │   │   └── seedScript.js    # Database seeding
│   │   │
│   │   ├── middleware/          # Express middleware
│   │   │   └── authenticateToken.js
│   │   │
│   │   ├── models/             # Data models
│   │   │   ├── fileUpload.js   # File upload model
│   │   │   └── user.js         # User model
│   │   │
│   │   ├── python/            # Python backend
│   │   │   ├── main.py        # FastAPI server
│   │   │   ├── chatbot.py     # Chatbot logic
│   │   │   ├── Documents_Totext.py  # Document processing
│   │   │   └── requirements.txt      # Python dependencies
│   │   │
│   │   └── routes/            # API routes
│   │
│   └── public/                # Frontend assets
│       ├── index.html         # Landing page
│       ├── login.html         # Login page
│       ├── chatbot.html       # Chat interface
│       ├── fileUpload.html    # File upload interface
│       │
│       ├── styles/           # CSS stylesheets
│       │   ├── main.css      # Global styles
│       │   ├── chatbot.css   # Chat interface styles
│       │   └── fileUpload.css # Upload page styles
│       │
│       ├── scripts/          # JavaScript files
│       │   ├── chatbot.js    # Chat functionality
│       │   └── fileUpload.js # Upload functionality
│       │
│       └── images/           # Image assets
│           ├── avatar-open.png
│           ├── avatar-closed.png
│           └── verztec-logo.png
```


## Dependencies

### Backend (Python)
- FastAPI
- langchain
- langchain_groq
- langchain_core
- langchain_community
- pymupdf
- python-dotenv

### Frontend (Node.js)
- Express.js
- Node.js
- npm packages (defined in package.json)

*Will also need to install MySQL and set user: 'chatbot_user', 
password: 'strong_password', database: 'chatbot_db'

How to run the application:
 ch
1. cd chatbot/src
2. npm install - > only need for first time or when adding new packages
3. npm run seed - > only need for first time or when updating database 
4. npm start

then go to the localhost:3000 to access the website

to enable the chatbot
open a new terminal, then
1. cd src/backend/python
4. python -m venv .venv -> only to create virtual environment first time
5. .venv\Scripts\activate.bat
6. pip install -r requirements.txt - > only need for first time OR when u add new requirements
7.  

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


Demo Accounts
You can use these accounts to test the application:
Username: Toby , Password : password1234


