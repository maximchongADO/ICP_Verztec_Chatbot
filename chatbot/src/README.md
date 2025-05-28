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
7. python chatbot.py


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