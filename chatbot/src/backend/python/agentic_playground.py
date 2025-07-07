"""
agentic_playground.py
---------------------
A sandbox for experimenting with agentic AI, LangChain agents, tools, and prompt engineering.

You can use this file to try out new chains, agents, or custom logic without affecting production code.
"""

import os
import logging
from datetime import datetime
import uuid
import numpy as np
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from sentence_transformers import CrossEncoder
from langchain_core.retrievers import BaseRetriever
from typing import List
from pydantic import ConfigDict
from langchain.agents import initialize_agent, Tool
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize models and clients (same as chatbot.py)
api_key = ''
model = "deepseek-r1-distill-llama-70b"
model="meta-llama/llama-4-scout-17b-16e-instruct"
deepseek = ChatGroq(api_key=api_key, model=model) # type: ignore

# Initialize memory (migrated to langchain_core)
memory = ConversationBufferMemory(
    memory_key="chat_history",
    input_key="question",
    output_key="answer",
    return_messages=True
)

# Load FAISS index (use your actual path)
script_dir = os.path.dirname(os.path.abspath(__file__))
faiss_index_path = os.path.join(script_dir, "faiss_master_index")
faiss_index_path2 = os.path.join(script_dir, "faiss_GK_index")
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-en-v1.5",
    encode_kwargs={'normalize_embeddings': True},
    model_kwargs={'device': 'cpu'}
)
index = FAISS.load_local(
    faiss_index_path,
    embedding_model,
    allow_dangerous_deserialization=True
)
index2 = FAISS.load_local(
    faiss_index_path2,
    embedding_model,
    allow_dangerous_deserialization=True
)

retr_direct = index.as_retriever(search_kwargs={"k": 8})
retr_bg = index2.as_retriever(search_kwargs={"k": 20})
cross_encoder = CrossEncoder("BAAI/bge-reranker-large")

class HybridRetriever(BaseRetriever):
    retr_direct: BaseRetriever
    retr_bg: BaseRetriever
    cross_encoder: object
    top_k_direct: int = 8
    top_k_bg: int = 20
    top_k_final: int = 10
    KEEP_BG_IF_DIRECT_WINS: int = 2
    KEEP_BG_IF_BG_WINS: int = 5
    MARGIN: float = 0.1
    model_config = ConfigDict(arbitrary_types_allowed=True)
    def _get_relevant_documents(self, query: str):
        direct_docs = self.retr_direct._get_relevant_documents(query)
        bg_docs = self.retr_bg._get_relevant_documents(query)
        best_d = max((getattr(d, "score", 0.0) for d in direct_docs), default=0.0)
        best_b = max((getattr(d, "score", 0.0) for d in bg_docs), default=0.0)
        if best_b >= best_d + self.MARGIN:
            selected_dir = direct_docs[: max(self.KEEP_BG_IF_DIRECT_WINS, 1)]
            selected_bg = bg_docs[: self.KEEP_BG_IF_BG_WINS]
        else:
            selected_dir = direct_docs
            selected_bg = bg_docs[: self.KEEP_BG_IF_DIRECT_WINS]
        seen = {id(d) for d in direct_docs}
        return direct_docs + [d for d in selected_bg if id(d) not in seen]
    async def _aget_relevant_documents(self, query: str):
        return self._get_relevant_documents(query)

# Initialize the hybrid retriever
hybrid_retriever_obj = HybridRetriever(
    retr_direct=retr_direct,
    retr_bg=retr_bg,
    cross_encoder=cross_encoder,
    top_k_direct=8,
    top_k_bg=20,
    top_k_final=5
)

# Example tool for agentic behavior
def search_docs_tool(query: str) -> str:
    """Search the FAISS index for relevant context and return the top result."""
    docs = index.similarity_search(query, k=1)
    if docs:
        return docs[0].page_content
    return "No relevant document found."

def escalate_to_hr_tool(issue: str) -> str:
    """Simulate escalating an issue to HR and return a confirmation."""
    # In a real system, this would trigger a workflow or send a message to HR
    return f"Your issue has been escalated to HR: '{issue}'. HR will review and follow up as needed. (Simulated)"

def create_meeting_request_tool(details: str) -> str:
    """Simulate creating a meeting request and return a confirmation."""
    # In a real system, this would create a calendar event or send an invite
    return f"A meeting request has been created with the following details: '{details}'. (Simulated)"

# Register as LangChain Tools

escalate_hr_tool = Tool(
    name="escalate_to_hr",
    func=escalate_to_hr_tool,
    description="Escalates a user issue or concern to HR for review. Use when the user reports a serious workplace, policy, or personal issue. (Simulated)"
)
create_meeting_tool = Tool(
    name="create_meeting_request",
    func=create_meeting_request_tool,
    description="Creates a meeting request with provided details. Use when the user wants to schedule a meeting. (Simulated)"
)

tools = [ escalate_hr_tool, create_meeting_tool]

def strip_think_tags(text: str) -> str:
    """Remove <think>...</think> tags from LLM output."""
    import re
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

# Custom system prompt for ReAct format
def get_react_prompt():
    return ChatPromptTemplate.from_messages([
        (
            "system",
            "You are an AI assistant that must follow the ReAct format **verbatim**.\n"
            "Allowed formats:\n"
            "1. For tool use:\n"
            "Thought: <reason>\nAction: <tool-name>\nAction Input: <input>\n"
            "2. For direct reply:\n"
            "Thought: <reason>\nAction: Final Answer\nAction Input: <answer>\n"
            "DO NOT output <think> tags, observations, or any extra text."
        ),
        ("human", "{input}")
    ])
    
def get_react_prompt():
    return ChatPromptTemplate.from_messages([
        ("system", (
            "You are an AI assistant that uses the ReAct format.\n"
            "In every response, ONLY output ONE of the following:\n"
            "1. A tool call:\n"
            "Thought: <reasoning>\n"
            "Action: <tool name>\n"
            "Action Input: <input string>\n"
            "2. A final answer:\n"
            "Thought: <reasoning>\n"
            "Action: Final Answer\n"
            "Action Input: <your answer to the user>\n"
            "NEVER output both a tool action AND a final answer in the same step.\n"
            "NEVER output an extra 'Observation:' line.\n"
            "Strictly follow the format with no additional text or tags."
        )),
        ("human", "{input}")
    ])


# Main chat loop
if __name__ == "__main__":
    parser = StrOutputParser()
    # Compose the prompt and LLM chain
    react_prompt = get_react_prompt()
    deepseek_chain = react_prompt | deepseek | parser

    # Patch the agent to clean LLM output before parsing
    class CleanedAgent:
        def __init__(self, agent):
            self.agent = agent
        def invoke(self, inputs):
            # Clean the user input and LLM output
            if isinstance(inputs, dict) and "input" in inputs:
                inputs["input"] = strip_think_tags(inputs["input"])
            response = self.agent.invoke(inputs)
            # If response is a string and contains 'Final Answer:', extract it as 'answer'
            if isinstance(response, str) and "Final Answer:" in response:
                answer = response.split("Final Answer:", 1)[1].strip()
                return {"answer": answer}
            if isinstance(response, dict):
                for k in ["answer", "output"]:
                    if k in response and isinstance(response[k], str):
                        response[k] = strip_think_tags(response[k])
            return response

    agent = initialize_agent(
        tools,
        deepseek_chain,
        agent="zero-shot-react-description",
        memory=memory,
        verbose=True,
        handle_parsing_errors=True,  # Fix output parsing errors gracefully
        max_iterations=5  # Prevent infinite loops
    )
    agent = CleanedAgent(agent)
    print("Agentic Playground Chat (type 'exit' to quit)")
    while True:
        user_query = input("You: ").strip()
        if user_query.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break
        invalid_format_count = 0
        max_invalid = 2
        while True:
            try:
                response = agent.invoke({"input": user_query, "question": user_query})
                if isinstance(response, dict):
                    if "answer" in response:
                        print("AI:", response["answer"])
                    elif "output" in response:
                        print("AI:", response["output"])
                    else:
                        # Try to find a string value in the dict
                        for v in response.values():
                            if isinstance(v, str):
                                print("AI:", v)
                                break
                        else:
                            print("AI:", response)
                elif isinstance(response, str):
                    print("AI:", response)
                else:
                    print("AI:", str(response))
                break  # Success, break the loop
            except Exception as e:
                if "Invalid Format" in str(e):
                    invalid_format_count += 1
                    if invalid_format_count > max_invalid:
                        print("AI: Sorry, I had trouble understanding the format. Please try rephrasing your question.")
                        break
                    continue  # Try again
                else:
                    print(f"AI: An error occurred: {e}")
                    break
