import os
import re
import sys
import time
import logging
from datetime import datetime
from time import sleep
import uuid
import numpy as np
import mysql.connector
import spacy
from spacy.matcher import PhraseMatcher
from typing import List
import os
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from langchain.tools import tool
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, AIMessage, Document
from langchain.vectorstores.base import VectorStoreRetriever
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_core.output_parsers import StrOutputParser
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import uuid
from memory_retrieval import build_memory_from_results, retrieve_user_messages_and_scores
from langchain.retrievers import ContextualCompressionRetriever
from langchain.llms import HuggingFaceHub 
from sentence_transformers import CrossEncoder
from langchain.schema import BaseRetriever
# Load reranker model locally


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize models and clients
embedding_model = SentenceTransformer('BAAI/bge-large-en-v1.5')
api_key = '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'

model = "deepseek-r1-distill-llama-70b" 
deepseek = ChatGroq(api_key=api_key, model=model) # type: ignore


# Initialize memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    input_key="question",
    output_key="answer",
    return_messages=True
)
# config db 
DB_CONFIG = {
    'host': 'localhost',
    'user': 'chatbot_user',
    'password': 'strong_password',
    'database': 'chatbot_db',
    'raise_on_warnings': True
}

# Load FAISS index
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    faiss_index_path = os.path.join(script_dir, "faiss_master_index")
    faiss_index_path2=os.path.join(script_dir, "faiss_GK_index")
    
    if not os.path.exists(faiss_index_path):
        raise FileNotFoundError(f"FAISS index not found at {faiss_index_path}")
    
    embedding_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en-v1.5",
        encode_kwargs={'normalize_embeddings': True},
        model_kwargs={'device': 'cpu'}  # Explicitly set device to CPU
    )
    
    index = FAISS.load_local(
        faiss_index_path,
        embedding_model,
        allow_dangerous_deserialization=True
    )
    
    index2=  FAISS.load_local(
        faiss_index_path2,
        embedding_model,
        allow_dangerous_deserialization=True
    )
    logger.info("FAISS index loaded successfully on CPU")
except Exception as e:
    logger.error(f"Failed to load FAISS index: {str(e)}", exc_info=True)
    index = None
    metadata = None
    
    
    
    
# load in spacy model and matcher for query scoring and casual phrase matching
try:
    # Load spaCy English model
    nlp = spacy.load("en_core_web_sm")
    casual_phrases = ["hi", "hello", "thanks", "thank you", "good morning", "goodbye", "hey", "yo", "sup"]
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    patterns = [nlp.make_doc(text) for text in casual_phrases]
    tokenizer = AutoTokenizer.from_pretrained("pszemraj/flan-t5-large-grammar-synthesis")
    model = AutoModelForSeq2SeqLM.from_pretrained("pszemraj/flan-t5-large-grammar-synthesis")
    reranker_model = CrossEncoder("BAAI/bge-reranker-large")  # will auto-download once
    matcher.add("CASUAL", patterns)
    logger.info("SpaCy model and matcher initialized successfully")
except Exception as e:
    logger.error(f"Failed to load spacymodel:  {str(e)}", exc_info=True)
  

def store_chat_log(user_message, bot_response, session_id,
                   query_score, relevance_score):
    conn = mysql.connector.connect(**DB_CONFIG)
    session_id = str(uuid.uuid4()) if session_id is None else session_id
    cursor = conn.cursor()

    timestamp = datetime.utcnow()

    insert_query = '''
        INSERT INTO chat_logs (timestamp, user_message, bot_response,
                               query_score, relevance_score)
        VALUES (%s, %s, %s, %s, %s)
    '''  # ─────────── five placeholders ──────────^

    cursor.execute(
        insert_query,
        (timestamp, user_message, bot_response,
         query_score, relevance_score)          # five values
    )
    conn.commit()
    logger.info("Stored chat log for session %s at %s", session_id, timestamp)

    cursor.close()
    conn.close()

# for stroing with user and chat id 
def store_chat_log_updated(user_message, bot_response, query_score, relevance_score,chat_id, user_id):
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor()
    

    timestamp = datetime.utcnow()

    insert_query = '''
        INSERT INTO chat_logs (timestamp, user_message, bot_response, feedback,query_score, relevance_score,user_id ,chat_id)
        VALUES (%s, %s, %s, %s, %s, %s)
    '''
    cursor.execute(insert_query, (timestamp, user_message, bot_response,query_score, relevance_score,user_id,chat_id))
    conn.commit()
    logger.info("Stored chat log for session %s %s at %s", user_id, chat_id, timestamp)

    cursor.close()
    conn.close()
    
# checking how alike a query the users question is
def is_query_score(text: str) -> float:
    """
    Determines how likely a given text is a task-related question for the Verztec assistant.
    Returns a float score from 0.0 to 1.0:
        - 1.0 = Strong query
        - 0.5 = Possibly a query, ambiguous
        - 0.0 = Casual talk or irrelevant
    """

    try:
        # ---------- Step 1: Clean the input ----------
        original = text
        text = text.lower().strip()
        text = re.sub(r'[^\w\s]', '', text)  # remove punctuation
        text = re.sub(r'\s+', ' ', text)     # collapse multiple spaces

        # ---------- Step 2: Early exits ----------
        refusal_phrases = [
            "i'm sorry", "as an ai", "i cannot answer", "i do not understand", "outside my scope",
            "not a question", "not relevant", "i cannot help", "please clarify", "not applicable"
        ]
        if any(phrase in text for phrase in refusal_phrases):
            return 0.0  # Definite non-query

        casual_phrases = {
            "hi", "hello", "hey", "how are you", "whats up", "thanks", "thank you",
            "good morning", "good evening", "ok", "okay", "yo", "sup",
            "what do i eat", "how do i go to bishan park", "what is ngee ann polytechnic",
            "ngee ann polytechnic", "bishan park", "park"
        }
        if text in casual_phrases:
            return 0.0  # Too casual

        if original.strip().endswith("?"):
            return 1.0  # Strong signal: explicitly a question

        # ---------- Step 3: Linguistic analysis ----------
        doc = nlp(text)

        # Match WH-words
        wh_words = {"what", "why", "who", "where", "when", "how", "which", "whom"}
        if any(token.lower_ in wh_words for token in doc):
            return 1.0

        # Explicit task keywords
        task_keywords = {"leave", "policy", "submit", "upload", "reset", "salary", "claim", "benefit", "holiday", "sick", "urgent", "apply", "deadline"}
        if any(token.lemma_ in task_keywords for token in doc):
            return 1.0

        # Related domain-specific hints (custom logic)
        related_terms = {'pantry', 'bereavement', 'rom', 'mc', 'clinic', 'finance'}
        if any(token.lower_ in related_terms for token in doc):
            return 1.0

        # Imperative verbs or command patterns
        command_verbs = {"tell", "explain", "show", "list", "describe", "give", "find", "fetch"}
        if any(token.lemma_ in command_verbs and token.pos_ == "VERB" for token in doc):
            return 0.9

        # Modal + auxiliary check
        aux_verbs = {"can", "could", "would", "should", "will", "do", "does", "did"}
        if any(token.lower_ in aux_verbs and token.tag_ in {"MD", "VB", "VBP", "VBZ"} for token in doc):
            return 0.8

        # ---------- Step 4: Fallback heuristics ----------
        if len(text.split()) >= 4:
            return 0.5  # Ambiguous but maybe informative

        return 0.0  # Short and uninformative

    except Exception as e:
        print(f"[Error in is_query_score]: {e}")
        return 0.0


def clean_with_grammar_model(user_query: str) -> str:
    """
    Uses a GEC-tuned model to clean up grammar, spelling, and clarity issues from user input.
    """
    
    input_text = f"gec: {user_query}"
    #input_text = f"gec: {user_query}"  # GEC = Grammar Error Correction instruction
    inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=128, truncation=True)
    outputs = model.generate(inputs, max_length=128, num_beams=5, early_stopping=True)
    corrected = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return user_query


def get_avg_score(index, embedding_model, query, k=10):
    # Embed using LangChain-style embedding model
    query_vec = embedding_model.embed_query(query)
    
    # Get the internal FAISS index
    raw_faiss_index = index.index
    
    # Prepare for search
    query_np = np.array([query_vec], dtype=np.float32)
    D, _ = raw_faiss_index.search(query_np, k)
    
    avg_score = np.mean(D[0]) if D.shape[1] > 0 else float('inf')
    logger.info(f"[L2] Average distance for query '{query}': {avg_score}")
    return avg_score


def build_memory_prompt(memory_obj, current_prompt):
    history = memory_obj.load_memory_variables({}).get("chat_history", [])

    # Filter out messages to just Human and AI
    dialogue = [msg for msg in history if isinstance(msg, (HumanMessage, AIMessage))]

    # Get the last 2 turns (i.e., 4 messages if it's alternating)
    last_turns = dialogue[-4:]  # 2 Human + 2 AI messages max

    # Add the current user prompt as the final turn
    last_turns.append(HumanMessage(content=current_prompt))

    return last_turns



def append_sources(cleaned_response: str, docs: list) -> str:
    def format_source_name(source_path: str) -> str:
        filename = os.path.basename(source_path)
        name, _ = os.path.splitext(filename)
        name = re.sub(r'^\d+\s*', '', name)  # Remove leading numbers
        return name.replace("_", " ").title().strip()

    sources = []
    seen_sources = set()
    for doc in docs:
        source = doc.metadata.get("source", None)
        if source and source not in seen_sources:
            seen_sources.add(source)
            sources.append(format_source_name(source))

    if not sources:
        return cleaned_response

    # Clean final formatting
    source_block = "\n\n📂 Source Document Used:\n" + "\n".join(f"• {src}" for src in sources)
    return cleaned_response.strip() + source_block


retr_direct  = index.as_retriever(search_kwargs={"k": 8})
retr_bg      = index2.as_retriever(   search_kwargs={"k": 20})

cross_encoder = CrossEncoder("BAAI/bge-reranker-large")



from pydantic import Field, ConfigDict

class HybridRetriever(BaseRetriever):
    # ---- Pydantic-declared fields --------------------------------
    retr_direct: BaseRetriever
    retr_bg: BaseRetriever
    cross_encoder: object

    top_k_direct: int = 8
    top_k_bg: int     = 20
    top_k_final: int  = 10
    
    KEEP_BG_IF_DIRECT_WINS: int = 2   # when direct is stronger
    KEEP_BG_IF_BG_WINS:     int = 5   # when BG is clearly stronger


    MARGIN: float = 0.1              # e.g. BG needs to be 0.05 better

    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    # --------------------------------------------------------------

    # optional: keep __init__ if you want positional kwargs, but not required
    # Pydantic will auto-generate one using the type hints above.

    def _best_score(self, docs: List[Document]) -> float:
        return max((getattr(d, "score", 0.0) for d in docs), default=0.0)

    def get_relevant_documents(self, query: str) -> List[Document]:
        direct_docs = self.retr_direct.get_relevant_documents(query, k=self.top_k_direct)
        bg_docs = self.retr_bg.get_relevant_documents(query, k=self.top_k_bg)

        best_d = self._best_score(direct_docs)
        best_b = self._best_score(bg_docs)


        # -------- selection logic -----------------------------
        if best_b >= best_d + self.MARGIN:
            # BG is clearly better
            selected_dir = direct_docs[: max(self.MIN_DIRECT, 1)]
            selected_bg  = bg_docs[: self.KEEP_BG_IF_BG_WINS]
        else:
            # Direct wins (or scores are close)
            selected_dir = direct_docs                       # keep ALL direct
            selected_bg  = bg_docs[: self.KEEP_BG_IF_DIRECT_WINS]

        seen = {id(d) for d in direct_docs}
        return direct_docs + [d for d in selected_bg if id(d) not in seen]
    
    
    async def aget_relevant_documents(self, query: str) -> List[Document]:
        return self.get_relevant_documents(query)



def generate_answer(user_query: str, chat_history: ConversationBufferMemory ):
    """
    Returns a tuple: (answer_text, image_list)
    """
    session_id = str(uuid.uuid4())
    try:
        total_start_time = time.time()  # Start timing for the whole query

        parser = StrOutputParser()
    
        # Chain the Groq model with the parser
        deepseek_chain = deepseek | parser
        
        
        
        ## INCASE HAVE TO SWITCH BACK TO SINGLE RETRIEVER 
        #  THIS HAS NO GENERAL KNOWLEDGE RETRIEVER
        retriever = index.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 10}
        )
        
        
        hybrid_retriever_obj = HybridRetriever(
            retr_direct=retr_direct,
            retr_bg=retr_bg,
            cross_encoder=cross_encoder,
            top_k_direct=8,
            top_k_bg=20,
            top_k_final=5
        )
        
        avg_score = get_avg_score(index, embedding_model, user_query)
        avg_score_gk = get_avg_score(index2, embedding_model, user_query)
        
        
        ## QA chain setup with mrmory 
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=deepseek_chain,
            retriever=hybrid_retriever_obj,
            memory=chat_history,
            return_source_documents=True,
            output_key="answer"
        )
        logger.info(memory)
    
    
        # Refine query
       
        clean_query = clean_with_grammar_model(user_query)
       
        # Step 3: Search FAISS for context 
        # for images, as well as for context relevance checks 
        results = index.similarity_search_with_score(clean_query, k=5)
        scores = [score for _, score in results]
        avg_score = float(np.mean(scores)) if scores else 1.0
        
        
        results_gk = index2.similarity_search_with_score(clean_query, k=5)
        scores_gk = [score for _, score in results_gk]
        avg_score_gk = float(np.mean(scores_gk)) if scores_gk else 1.0
        logger.info(f"Average score for GK index: {avg_score_gk:.4f}")
        
        
        
        seen = set()
        unique_docs = []

        for doc, _ in results:
            content = doc.page_content
            if content not in seen:
                seen.add(content)
                unique_docs.append(doc)

        logger.info(f"Retrieved {len(unique_docs)} documents with average score: {avg_score:.4f}")
    
        
        ## retrieving images from top 3 chunks (if any)
        # Retrieve images from top 3 chunks (if any)
        THRESHOLD      = 0.70          # keep docs whose score < threshold   (lower L2 → more similar)
        MAX_IMAGES     = 3             # cap images
        sources_set    = set()         # remember unique source filenames
        sources_list   = []            # ordered list of unique sources
        seen_contents  = set()         # avoid duplicate page_content
        top_docs       = []            # docs we actually keep
        top_3_img      = []            # up to 3 image paths / URLs

        for doc, score in results:
            if score >= THRESHOLD:      # skip weak matches
                continue

            # ---- source handling ------------------------------------
            src = doc.metadata.get("source")
            if src and src not in sources_set:
                sources_set.add(src)
                sources_list.append(src)

            # ---- unique doc content ---------------------------------
            if doc.page_content not in seen_contents:
                seen_contents.add(doc.page_content)
                top_docs.append(doc)

            # ---- gather up to 3 images ------------------------------
            if len(top_3_img) < MAX_IMAGES:
                for img in doc.metadata.get("images", []):
                    top_3_img.append(img)
                    if len(top_3_img) >= MAX_IMAGES:
                        break

                    


            # Ensures a flat list of strings
        top_3_img = list(set(top_3_img))  # Remove duplicates
        logger.info(f"Top 3 images: {top_3_img}")


       
        is_task_query = is_query_score(user_query)
        logger.info(f"Query Score: {is_task_query}")
        logger.info(f"Average Score: {avg_score}")
        
        soft_threshold = 0.7
        SOFT_QUERY_THRESHOLD = 0.5
        STRICT_QUERY_THRESHOLD = 0.2
        HARD_AVG_SCORE_THRESHOLD = 1.01
        
        #handle irrelevant query
        logger.info(f"Clean Query at qa chain: {clean_query}")
        if (
            (is_task_query < SOFT_QUERY_THRESHOLD and avg_score >= soft_threshold) or
            avg_score >= HARD_AVG_SCORE_THRESHOLD or
            is_task_query < STRICT_QUERY_THRESHOLD
        ):
            if is_task_query < SOFT_QUERY_THRESHOLD and avg_score >= soft_threshold:
                logger.info("[BYPASS_REASON] Tag: low_task_high_score — Query intent is kinda weak and query is slighlty irrelevant.")
            elif avg_score >= HARD_AVG_SCORE_THRESHOLD:
                logger.info("[BYPASS_REASON] Tag: high_score_threshold — Query is highly irrelevant to FAISS documents.")
            elif is_task_query < STRICT_QUERY_THRESHOLD:
                logger.info("[BYPASS_REASON] Tag: very_low_task_intent — Query clearly not a task query.")

            logger.info("Bypassing QA chain for non-query with weak retrieval.")
           # fallback_prompt = f"The user said, this query is out of scope: \"{clean_query}\". Respond appropriately as a POLITELY VERZTEC assistant, and ask how else you can help"
            fallback_prompt = (
                f'The user said: "{clean_query}". '
                'As a HELPFUL and FRIENDLY VERZTEC helpdesk assistant, respond with a light-hearted or polite reply — '
                'even if the message is small talk or out of scope (e.g., "how are you", "do you like pizza"). '
                'Keep it human and warm (e.g., "I’m doing great, thanks for asking!"), then ***gently guide the user back to Verztec-related helpdesk topics***.'
                'Do not answer any questions that are not related to Verztec helpdesk topics, and do not use any of the provided documents in your response. '
            )

            #modified_query = "You are a verztec helpdesk assistant. You will only use the provided documents in your response. If the query is out of scope, say so.\n\n" + clean_query
            #messages = [HumanMessage(content=fallback_prompt)]
            messages = build_memory_prompt(memory, fallback_prompt)
            response = deepseek.generate([messages])

            raw_fallback = response.generations[0][0].text.strip()

            # Remove <think> block if present
            think_block_pattern = re.compile(r"<think>.*?</think>", flags=re.DOTALL)
            cleaned_fallback = think_block_pattern.sub("", raw_fallback).strip()
            top_3_img = []  # No images for fallback response
            # Store the fallback response in chat memory
         
            memory.chat_memory.add_user_message(fallback_prompt)
            memory.chat_memory.add_ai_message(cleaned_fallback) 
            # Clean up chat memory to keep it manageable
            # Limit chat memory to last 4 turns (8 messages)
            MAX_TURNS = 4
            if len(memory.chat_memory.messages) > 2 * MAX_TURNS:
                memory.chat_memory.messages = memory.chat_memory.messages[-2 * MAX_TURNS:]
            store_chat_log(user_message=user_query, bot_response=cleaned_fallback, session_id=session_id, query_score=is_task_query, relevance_score=avg_score)
    
            return cleaned_fallback, top_3_img
        
        
        
        
        logger.info("QA chain activated for query processing.")
        # Step 4: Prepare full prompt and return LLM output
        modified_query = "You are a  HELPFUL AND NICE verztec helpdesk assistant. You will only use the provided documents in your response. If the query is out of scope, say so.\n\n" + clean_query
        modified_query = (
            "You are a HELPFUL AND NICE Verztec helpdesk assistant. "
            "You will only use the provided documents in your response. "
            "If the query is out of scope, say so. "
            "If there are any image tags or screenshots mentioned in the documents, "
            "please reference them in your response where appropriate, such as 'See Screenshot 1' or 'Refer to the image above'.\n\n"
            + clean_query
        )

        qa_start_time = time.time()
        response = qa_chain.invoke({"question": modified_query})
        qa_elapsed_time = time.time() - qa_start_time
        raw_answer = response['answer']
        source_docs = response['source_documents']
        logger.info(f"Source docs from QA chain: {source_docs}")
        logger.info(f"Full response before cleanup: {raw_answer} (QA chain time taken: {qa_elapsed_time:.2f}s)")
        
        # Clean the chat memory to keep it manageable
        # Limit chat memory to last 4 turns (8 messages)
        MAX_TURNS = 4
        if len(memory.chat_memory.messages) > 2 * MAX_TURNS:
            memory.chat_memory.messages = memory.chat_memory.messages[-2 * MAX_TURNS:]

        # Define regex pattern to match full <think>...</think> block
        think_block_pattern = re.compile(r"<think>.*?</think>", flags=re.DOTALL)
       
        

        # Check if full <think> block exists
        has_think_block = bool(think_block_pattern.search(raw_answer))

        # Clean the <think> block regardless
        cleaned_answer = think_block_pattern.sub("", raw_answer).strip()
        has_tag = False
        import concurrent.futures

        # Timeout logic for retry
        def retry_qa_chain():
            return qa_chain.invoke({"question": clean_query})['answer']

        i = 1  # Ensure i is defined
        while has_tag and i == 1:
            if not has_think_block:
                logger.warning("Missing full <think> block — retrying query once...")
                try:
                    retry_start = time.time()
                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                        future = executor.submit(retry_qa_chain)
                        raw_answer_retry = future.result(timeout=30)
                    retry_elapsed = time.time() - retry_start
                    logger.info(f"Retry response: {raw_answer_retry} (Retry time taken: {retry_elapsed:.2f}s)")
                    sleep(1)  # Optional: wait a bit before retrying
                    cleaned_answer = think_block_pattern.sub("", raw_answer_retry).strip()
                except concurrent.futures.TimeoutError:
                    logger.error("QA chain retry timed out after 30 seconds.")
                    cleaned_answer = "Sorry, the system took too long to generate a response. Please try again in a moment."
                    break
                except Exception as e:
                    logger.error(f"Error during QA chain retry: {e}")
                    cleaned_answer = f"Sorry, an error occurred while generating a response: {e}"
                    break
            else:
                logger.info("Full <think> block found and removed successfully")
            block_tag_pattern = re.compile(r"<([a-zA-Z0-9_]+)>.*?</\1>", flags=re.DOTALL)

            # Check if there are any block tags at all
            has_any_block_tag = bool(block_tag_pattern.search(raw_answer))
            if has_any_block_tag:
                logger.info("Block tags found in response, cleaning them up")

            # Remove all block tags
            cleaned_answer = block_tag_pattern.sub("", raw_answer).strip()
            cleaned_answer = re.sub(r"^\s+", "", cleaned_answer)
            has_any_block_tag = bool(block_tag_pattern.search(raw_answer))
            if not has_any_block_tag:
                has_tag = False
            i += 1
        ## one last cleanup to ensure no <think> tags remain
        # Remove any remaining <think> tagsbetter have NO MOR NO MORE NO MO NO MOMRE 
        cleaned_answer = re.sub(r"</?think>", "", cleaned_answer).strip()
        cleaned_answer = re.sub(r"</?think>", "", cleaned_answer).strip()
        cleaned_answer = re.sub(r'[\*#]+', '', cleaned_answer).strip()

    
        store_chat_log(user_message=user_query, bot_response=cleaned_answer, session_id=session_id, query_score=is_task_query, relevance_score=avg_score)
        # After generating the bot's response
        final_response = append_sources(cleaned_answer, top_docs)
        print(final_response)

        total_elapsed_time = time.time() - total_start_time
        logger.info(f"Total time taken for query processing: {total_elapsed_time:.2f}s")

        return final_response, top_3_img
    except Exception as e:
        return f"I encountered an error while processing your request: {str(e)}", []

memory_store = {}

def generate_answer_histoy_retrieval(user_query: str, user_id:str, chat_id:str):
    """
    Returns a tuple: (answer_text, image_list)
    """
    
    key = f"{user_id}_{chat_id}"  # Use a separator to avoid accidental key collisions

    if key in memory_store:
        chat_history = memory_store[key]
        logger.info(f"Memory object found at key: {key}")
    else:
        msgs = retrieve_user_messages_and_scores(user_id, chat_id)
        chat_history = build_memory_from_results(msgs)
        memory_store[key] = chat_history
        logger.info(f"No memory object found. Created and saved for key: {key}")

    
        
    
    try:
        total_start_time = time.time()  # Start timing for the whole query

        parser = StrOutputParser()
    
        # Chain the Groq model with the parser
        deepseek_chain = deepseek | parser
        
        retriever = index.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 10}
        )
        avg_score = get_avg_score(index, embedding_model, user_query)
        
        
        ## QA chain setup with mrmory 
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=deepseek_chain,
            retriever=retriever,
            memory=chat_history,
            return_source_documents=True,
            output_key="answer"
        )
        logger.info(memory)
    
    
        # Refine query
       
        clean_query = clean_with_grammar_model(user_query)
       
        # Step 3: Search FAISS for context 
        # for images, as well as for context relevance checks 
        results = index.similarity_search_with_score(clean_query, k=5)
        scores = [score for _, score in results]
        avg_score = float(np.mean(scores)) if scores else 1.0
        seen = set()
        unique_docs = []

        for doc, _ in results:
            content = doc.page_content
            if content not in seen:
                seen.add(content)
                unique_docs.append(doc)

        logger.info(f"Retrieved {len(unique_docs)} documents with average score: {avg_score:.4f}")
    
        
        ## retrieving images from top 3 chunks (if any)
        # Retrieve images from top 3 chunks (if any)
        top_3_img = []
        sources_set = set()
        sources_list = []
        top_docs = []
        seen_contents = set()

        for doc, score in results:
            if score < 0.7:
                # Use the filename or document name as the source
                source = doc.metadata.get('source')
                if source and source not in sources_set:
                    sources_set.add(source)
                    sources_list.append(source)

                # Track unique docs (based on content) for appending source metadata later
                if doc.page_content not in seen_contents:
                    seen_contents.add(doc.page_content)
                    top_docs.append(doc)

                # Add up to 3 images only
                images = doc.metadata.get('images', [])
                for img in images:
                    if len(top_3_img) < 3:
                        top_3_img.append(img)
                    else:
                        break

                if len(top_3_img) >= 3:
                    break


            # Ensures a flat list of strings
        top_3_img = list(set(top_3_img))  # Remove duplicates
        logger.info(f"Top 3 images: {top_3_img}")


       
        is_task_query = is_query_score(user_query)
        logger.info(f"Query Score: {is_task_query}")
        logger.info(f"Average Score: {avg_score}")
        
        soft_threshold = 0.7
        SOFT_QUERY_THRESHOLD = 0.5
        STRICT_QUERY_THRESHOLD = 0.2
        HARD_AVG_SCORE_THRESHOLD = 1.01
        
        #handle irrelevant query
        logger.info(f"Clean Query at qa chain: {clean_query}")
        if (
            (is_task_query < SOFT_QUERY_THRESHOLD and avg_score >= soft_threshold) or
            avg_score >= HARD_AVG_SCORE_THRESHOLD or
            is_task_query < STRICT_QUERY_THRESHOLD
        ):
            if is_task_query < SOFT_QUERY_THRESHOLD and avg_score >= soft_threshold:
                logger.info("[BYPASS_REASON] Tag: low_task_high_score — Query intent is kinda weak and query is slighlty irrelevant.")
            elif avg_score >= HARD_AVG_SCORE_THRESHOLD:
                logger.info("[BYPASS_REASON] Tag: high_score_threshold — Query is highly irrelevant to FAISS documents.")
            elif is_task_query < STRICT_QUERY_THRESHOLD:
                logger.info("[BYPASS_REASON] Tag: very_low_task_intent — Query clearly not a task query.")

            logger.info("Bypassing QA chain for non-query with weak retrieval.")
           # fallback_prompt = f"The user said, this query is out of scope: \"{clean_query}\". Respond appropriately as a POLITELY VERZTEC assistant, and ask how else you can help"
            fallback_prompt = (
                f'The user said: "{clean_query}". '
                'As a HELPFUL and FRIENDLY VERZTEC helpdesk assistant, respond with a light-hearted or polite reply — '
                'even if the message is small talk or out of scope (e.g., "how are you", "do you like pizza"). '
                'Keep it human and warm (e.g., "I’m doing great, thanks for asking!"), then ***gently guide the user back to Verztec-related helpdesk topics***.'
                'Do not answer any questions that are not related to Verztec helpdesk topics, and do not use any of the provided documents in your response. '
            )

            #modified_query = "You are a verztec helpdesk assistant. You will only use the provided documents in your response. If the query is out of scope, say so.\n\n" + clean_query
            #messages = [HumanMessage(content=fallback_prompt)]
            messages = build_memory_prompt(memory, fallback_prompt)
            response = deepseek.generate([messages])

            raw_fallback = response.generations[0][0].text.strip()

            # Remove <think> block if present
            think_block_pattern = re.compile(r"<think>.*?</think>", flags=re.DOTALL)
            cleaned_fallback = think_block_pattern.sub("", raw_fallback).strip()
            top_3_img = []  # No images for fallback response
            # Store the fallback response in chat memory
         
            memory.chat_memory.add_user_message(fallback_prompt)
            memory.chat_memory.add_ai_message(cleaned_fallback) 
            # Clean up chat memory to keep it manageable
            # Limit chat memory to last 4 turns (8 messages)
            MAX_TURNS = 4
            if len(memory.chat_memory.messages) > 2 * MAX_TURNS:
                memory.chat_memory.messages = memory.chat_memory.messages[-2 * MAX_TURNS:]
            #store_chat_log(user_message=user_query, bot_response=cleaned_fallback, session_id=session_id, query_score=is_task_query, relevance_score=avg_score)
            store_chat_log_updated(user_message=user_query, bot_response=cleaned_answer, query_score=is_task_query, relevance_score=avg_score, user_id=user_id, chat_id=chat_id)
    
            return cleaned_fallback, top_3_img
        
        
        
        
        logger.info("QA chain activated for query processing.")
        # Step 4: Prepare full prompt and return LLM output
        modified_query = "You are a  HELPFUL AND NICE verztec helpdesk assistant. You will only use the provided documents in your response. If the query is out of scope, say so.\n\n" + clean_query
        modified_query = (
            "You are a HELPFUL AND NICE Verztec helpdesk assistant. "
            "You will only use the provided documents in your response. "
            "If the query is out of scope, say so. "
            "If there are any image tags or screenshots mentioned in the documents, "
            "please reference them in your response where appropriate, such as 'See Screenshot 1' or 'Refer to the image above'.\n\n"
            + clean_query
        )

        qa_start_time = time.time()
        response = qa_chain.invoke({"question": modified_query})
        qa_elapsed_time = time.time() - qa_start_time
        raw_answer = response['answer']
        logger.info(f"Full response before cleanup: {raw_answer} (QA chain time taken: {qa_elapsed_time:.2f}s)")
        
        # Clean the chat memory to keep it manageable
        # Limit chat memory to last 4 turns (8 messages)
        MAX_TURNS = 4
        if len(memory.chat_memory.messages) > 2 * MAX_TURNS:
            memory.chat_memory.messages = memory.chat_memory.messages[-2 * MAX_TURNS:]

        # Define regex pattern to match full <think>...</think> block
        think_block_pattern = re.compile(r"<think>.*?</think>", flags=re.DOTALL)
       
        

        # Check if full <think> block exists
        has_think_block = bool(think_block_pattern.search(raw_answer))

        # Clean the <think> block regardless
        cleaned_answer = think_block_pattern.sub("", raw_answer).strip()
        has_tag = False
        import concurrent.futures

        # Timeout logic for retry
        def retry_qa_chain():
            return qa_chain.invoke({"question": clean_query})['answer']

        i = 1  # Ensure i is defined
        while has_tag and i == 1:
            if not has_think_block:
                logger.warning("Missing full <think> block — retrying query once...")
                try:
                    retry_start = time.time()
                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                        future = executor.submit(retry_qa_chain)
                        raw_answer_retry = future.result(timeout=30)
                    retry_elapsed = time.time() - retry_start
                    logger.info(f"Retry response: {raw_answer_retry} (Retry time taken: {retry_elapsed:.2f}s)")
                    sleep(1)  # Optional: wait a bit before retrying
                    cleaned_answer = think_block_pattern.sub("", raw_answer_retry).strip()
                except concurrent.futures.TimeoutError:
                    logger.error("QA chain retry timed out after 30 seconds.")
                    cleaned_answer = "Sorry, the system took too long to generate a response. Please try again in a moment."
                    break
                except Exception as e:
                    logger.error(f"Error during QA chain retry: {e}")
                    cleaned_answer = f"Sorry, an error occurred while generating a response: {e}"
                    break
            else:
                logger.info("Full <think> block found and removed successfully")
            block_tag_pattern = re.compile(r"<([a-zA-Z0-9_]+)>.*?</\1>", flags=re.DOTALL)

            # Check if there are any block tags at all
            has_any_block_tag = bool(block_tag_pattern.search(raw_answer))
            if has_any_block_tag:
                logger.info("Block tags found in response, cleaning them up")

            # Remove all block tags
            cleaned_answer = block_tag_pattern.sub("", raw_answer).strip()
            cleaned_answer = re.sub(r"^\s+", "", cleaned_answer)
            has_any_block_tag = bool(block_tag_pattern.search(raw_answer))
            if not has_any_block_tag:
                has_tag = False
            i += 1
        ## one last cleanup to ensure no <think> tags remain
        # Remove any remaining <think> tagsbetter have NO MOR NO MORE NO MO NO MOMRE 
        cleaned_answer = re.sub(r"</?think>", "", cleaned_answer).strip()
        cleaned_answer = re.sub(r"</?think>", "", cleaned_answer).strip()
        cleaned_answer = re.sub(r'[\*#]+', '', cleaned_answer).strip()

    
        store_chat_log_updated(user_message=user_query, bot_response=cleaned_answer, query_score=is_task_query, relevance_score=avg_score, user_id=user_id, chat_id=chat_id)##brian u need to update sql for this to work
        # also need to update the store_chat_bot method, to incoude user id and chat id
        # After generating the bot's response
        final_response = append_sources(cleaned_answer, top_docs)
        print(final_response)

        total_elapsed_time = time.time() - total_start_time
        logger.info(f"Total time taken for query processing: {total_elapsed_time:.2f}s")

        return final_response, top_3_img
    except Exception as e:
        return f"I encountered an error while processing your request: {str(e)}", []

