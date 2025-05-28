from typing import List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage
import re
import os
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_core.output_parsers import StrOutputParser
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize models and clients
embedding_model = SentenceTransformer('BAAI/bge-large-en-v1.5')
api_key = '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'
model_name = "compound-beta"
model = "deepseek-r1-distill-llama-70b" 
deepseek = ChatGroq(api_key=api_key, model=model)
compound = ChatGroq(api_key=api_key, model=model_name)

# Initialize memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    input_key="question",
    output_key="answer",
    return_messages=True
)

# Load FAISS index
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    faiss_index_path = os.path.join(script_dir, "faiss_index3")
    
    if not os.path.exists(faiss_index_path):
        raise FileNotFoundError(f"FAISS index not found at {faiss_index_path}")
    
    # Configure FAISS to explicitly use CPU
    import faiss
    faiss.omp_set_num_threads(4)  # Optimize CPU thread usage
    
    embedding_model2 = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en-v1.5",
        encode_kwargs={'normalize_embeddings': True},
        model_kwargs={'device': 'cpu'}  # Explicitly set device to CPU
    )
    
    index = FAISS.load_local(
        faiss_index_path,
        embedding_model2,
        allow_dangerous_deserialization=True
    )
    logger.info("FAISS index loaded successfully on CPU")
    
except Exception as e:
    logger.error(f"Failed to load FAISS index: {str(e)}", exc_info=True)
    index = None
    metadata = None


## clean query and ensure it is suitable for processing
def refine_prompt(user_query: str) -> str:
    prompt = (
        f"You are an assistant that improves grammar and spelling for an internal helpdesk chatbot. "
        f"If the input is offensive, unclear, or meaningless, return the phrase: [UNCLEAR].\n\n"
        f"Input:\n{user_query}"
    )
    messages = [HumanMessage(content=prompt)]
    response = compound.generate([messages])
    refined_query = response.generations[0][0].text.strip()

    # Safety net
    if "[UNCLEAR]" in refined_query.upper() or len(refined_query) < 5:
        return "[UNCLEAR]"
    if "[UNCLEAR]" in refined_query.upper():
        refined_query= 'The user has sent something unintellible, please clarify    '
    return refined_query



def generate_answer(user_query: str, chat_history: ConversationBufferMemory) -> str:
    try:
        parser = StrOutputParser()

        # Chain the Groq model with the parser
        deepseek_chain = deepseek | parser
        ## FAISS retriever setup
        retriever = index.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 10}
        )
        ## QA chain setup with mrmory 
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=deepseek_chain,
            retriever=retriever,
            memory=chat_history,
            return_source_documents=True,
            output_key="answer"
        
        )

        # Step 1: Check relation to past queries
        #is_related = is_related_to_previous(user_query, chat_history)
        #context_query = " ".join(chat_history + [user_query]) if is_related else user_query

        # Step 2: Refine query
        clean_query = refine_prompt(user_query)

        # Step 3: Search FAISS for context 
        # for images, as well as for context relevance checks 
        results = index.similarity_search_with_score(clean_query, k=5)
        scores = [score for _, score in results]
        avg_score = float(np.mean(scores)) if scores else 1.0
        
        ## retrieving images from top 3 chunks (if any)
        top_3_img=[]
        for i, (doc, score) in enumerate(results, 1):
            if len(top_3_img) < 3:
                top_3_img.append(doc.metadata['images'])


        if avg_score > 2.5:
            fallback = (
                f"The user's question is likely unrelated to internal company topics.\n"
                f"Query: {clean_query}\n"
                f"Please advise the user that this query is out of scope."
            )
            messages = [HumanMessage(content=fallback)]
            response = deepseek.generate([messages])
            return response.generations[0][0].text.strip()

        # Step 4: Prepare full prompt and return LLM output
        response = qa_chain.invoke({"question": clean_query})
        answer = re.sub(r"<think>.*?</think>", "", response['answer'], flags=re.DOTALL).strip()
        return answer
    
    except Exception as e:
        return f"I encountered an error while processing your request: {str(e)}"
