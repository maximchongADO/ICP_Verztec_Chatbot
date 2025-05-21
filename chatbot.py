import os
import re
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from pathlib import Path
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.memory import ConversationBufferWindowMemory

# Load environment variables
load_dotenv()
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

api_key = "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"
model = "deepseek-r1-distill-llama-70b"  # Update model as required
deepseek = ChatGroq(api_key=api_key, model_name=model)

# Define the output parser for handling results
parser = StrOutputParser()

# Chain the Groq model with the parser
deepseek_chain = deepseek | parser

# Print the result from the chainD
#print(deepseek_chain.invoke("hello"))

# Load embedding model
embedding_model2 = HuggingFaceEmbeddings(
    model_name = "BAAI/bge-large-en-v1.5",

    encode_kwargs={'normalize_embeddings': True}  
)
# --- Step 1: Load FAISS Vector Store ---

root_dir = Path(__file__).parent
data_dir = root_dir / "data"
cleaned_dir = data_dir / "cleaned"
chat_history = []  # Store (user, bot) message pairs
memory = ConversationBufferMemory(
    memory_key="chat_history",
    input_key="question",        # Input key in your chain
    output_key="answer",         # Explicit output key
    return_messages=True
)


# Query function to handle the data retrieval and LLM invocation
def query_faiss_llm(query):
    # Load FAISS vector store
    vector_store = FAISS.load_local("verztec_vector_store", embedding_model, allow_dangerous_deserialization=True)

    # Create retriever for fetching documents
    retriever = vector_store.as_retriever(search_type="similarity", k=3)

    # Now replace OpenAI API usage with the Groq model setup
    # Build the retrieval QA chain using Groq model instead of OpenAI
    qa_chain = RetrievalQA.from_chain_type(
        llm=deepseek_chain,  # Use the Groq model chain here
        retriever=retriever,
        return_source_documents=True
    )

    # Retrieve documents for context
    docs = retriever.get_relevant_documents(query)

    # Prepare the context from the retrieved documents
    context = "\n".join([doc.page_content for doc in docs])  # Concatenate text from all retrieved docs

    # Now, invoke the Groq chain using the query and context
    response = qa_chain.invoke({"query": query, "context": context})

    # Output the response and sources
    print("Answer:", response["result"])
    print("\nSources:")
    for doc in response["source_documents"]:
        print("-", doc.metadata["source"])

# Main interactive loop for user input
def interactive_bot():
    print("Welcome to the Verztec Helpdesk Bot! You can ask questions anytime.")
    print("Type 'exit' to quit.")

    chat_history = []

    while True:
        query = input("Ask a question: ")
        if query.lower() == "exit":
            print("Goodbye!")
            break
        chat_history=query_faiss_llm(query, chat_history)


# Start the interactive bot
if __name__ == "__main__":
    interactive_bot()


