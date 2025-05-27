# main.py (new file)
from fastapi import FastAPI, Request
from pydantic import BaseModel
from chatbot_backend import generate_answer  # refactor your chatbot logic to this function

app = FastAPI()

class ChatRequest(BaseModel):
    query: str
    history: list[str] = []

@app.post("/chat")
def chat_endpoint(data: ChatRequest):
    answer = generate_answer(data.query, data.history)
    return {"answer": answer}
