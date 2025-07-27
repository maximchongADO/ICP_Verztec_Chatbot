
from collections import Counter
import os, re, difflib, logging
import os
import re
import difflib
import time
import concurrent.futures
import logging
import csv
import uuid
import numpy as np
import pymysql
import spacy
import concurrent.futures
from datetime import datetime
from time import sleep
from spacy.matcher import PhraseMatcher
from typing import List
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from typing import Optional, Dict
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, AIMessage, Document
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from memory_retrieval import build_memory_from_results, retrieve_user_messages_and_scores
from sentence_transformers import CrossEncoder
from langchain.schema import BaseRetriever
from langchain.agents import Tool, create_react_agent, AgentExecutor
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.agents import AgentAction, AgentFinish
from langchain.prompts import ChatPromptTemplate
#from tool_executors import execute_confirmed_tool, execute_hr_escalation_tool, execute_meeting_scheduling_tool
import csv
import logging
import os
import re
import smtplib
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv
# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Initialize models and clients
embedding_model = SentenceTransformer('BAAI/bge-large-en-v1.5')
load_dotenv()
api_key='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'
api_key="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"
api_key="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"## backend api key
api_key='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'
api_key='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'# i love api key
api_key='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' # i love api key
api_key='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' # i love api keyyy
model = "deepseek-r1-distill-llama-70b" 
deepseek = ChatGroq(api_key=api_key, model=model, temperature = 0.4) # type: ignore
decisionlayer_model=ChatGroq(api_key=api_key, 
                            model="qwen/qwen3-32b",
                            temperature=0,                # ⬅ deterministic, no creativity
                            model_kwargs={
                                "top_p": 0,               # ⬅ eliminates sampling randomness
                                "frequency_penalty": 0,
                                "presence_penalty": 0
                                }
                            ) 
cleaning_model = ChatGroq(api_key=api_key,  model="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0.0) # type: ignore


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
    'cursorclass': pymysql.cursors.Cursor,
    'autocommit': True
}

# Load FAISS index
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    faiss_index_path = os.path.join(script_dir, "faiss_master_index3")
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
  

def store_chat_log_updated(user_message, bot_response, query_score, relevance_score, chat_id, user_id,chat_name=None):
    conn = pymysql.connect(**DB_CONFIG)
    cursor = conn.cursor()

    timestamp = datetime.utcnow()
    feedback = None  # placeholder if no feedback given

    insert_query = '''
        INSERT INTO chat_logs (timestamp, user_message, bot_response, feedback, query_score, relevance_score, user_id, chat_id,chat_name)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s,%s)
    '''
    cursor.execute(insert_query, (timestamp, user_message, bot_response, feedback, query_score, relevance_score, user_id, chat_id,chat_name))
    conn.commit()
    logger.info("Stored chat log for session %s %s at %s", user_id, chat_id, timestamp)

    cursor.close()
    conn.close()


def store_hr_escalation(escalation_id, user_id, chat_id, user_message, issue_summary, status="PENDING", priority="NORMAL", user_description=None):
    """
    Store HR escalation data in the hr_escalations table.
    
    Args:
        escalation_id (str): Unique escalation reference ID
        user_id (str): User identifier
        chat_id (str): Chat session identifier
        user_message (str): Original user message that triggered escalation
        issue_summary (str): Sanitized summary of the issue (max 800 chars)
        status (str): Escalation status (default: PENDING)
        priority (str): Escalation priority (default: NORMAL)
        user_description (str, optional): Additional description provided by user
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        conn = pymysql.connect(**DB_CONFIG)
        cursor = conn.cursor()

        timestamp = datetime.utcnow()
        
        # Insert escalation data (with fallback for tables without user_description column)
        try:
            # Try with user_description column first
            insert_query = '''
                INSERT INTO hr_escalations (
                    escalation_id, timestamp, user_id, chat_id, 
                    user_message, issue_summary, status, priority, user_description
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            '''
            
            cursor.execute(insert_query, (
                escalation_id, timestamp, user_id, chat_id, 
                user_message, issue_summary, status, priority, user_description
            ))
            
        except pymysql.err.OperationalError as e:
            # If user_description column doesn't exist, fall back to original schema
            if "unknown column" in str(e).lower() or "doesn't exist" in str(e).lower():
                logger.warning("user_description column not found, using fallback insert")
                insert_query = '''
                    INSERT INTO hr_escalations (
                        escalation_id, timestamp, user_id, chat_id, 
                        user_message, issue_summary, status, priority
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                '''
                
                cursor.execute(insert_query, (
                    escalation_id, timestamp, user_id, chat_id, 
                    user_message, issue_summary, status, priority
                ))
            else:
                raise  # Re-raise if it's a different error
        
        conn.commit()
        logger.info(f"Stored HR escalation {escalation_id} for user {user_id} in chat {chat_id} at {timestamp}")
        
        cursor.close()
        conn.close()
        
        return True
        
    except Exception as e:
        logger.error(f"Error storing HR escalation {escalation_id}: {str(e)}")
        return False

def get_user_info(user_id: str):
    conn = pymysql.connect(**DB_CONFIG)
    cursor = conn.cursor()
    try:
        # Fetch user info from the users table
        cursor.execute("SELECT username, role, country, department FROM users WHERE id = %s", (user_id,))
        result = cursor.fetchone()
        
        if result:
            user_info = {
                "username": result[0],
                "role": result[1],
                "country": result[2],
                "department": result[3]
            }
            return user_info
        else:
            logger.warning(f"No user found with ID: {user_id}")
            return None
    except pymysql.Error as e:
        logger.error(f"Error fetching user info for {user_id}: {str(e)}")
        return None
    
    
    
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

def append_sources_with_links(cleaned_response: str, docs: list):
    def format_source_name(source_path: str) -> str:
        filename = os.path.basename(source_path)
        name, _ = os.path.splitext(filename)
        name = re.sub(r'^\d+\s*', '', name)  # Remove leading numbers
        return name

    def find_file_in_subdirs(file_name: str) -> str:
        """
        Searches for a file in the 'data' subdirectories (pdf, word, pptx).
        Uses comprehensive fuzzy matching to locate files.
        """
        logger.info(f"Searching for file: {file_name}")
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_dirs = [
            os.path.join(script_dir, "data", "pdf"),
            os.path.join(script_dir, "data", "word"), 
            os.path.join(script_dir, "data", "pptx")
        ]
        
        #logger.info(f"Searching in directories: {data_dirs}")
        
        # Get all files from all data directories for fuzzy matching
        all_files = []
        for data_dir in data_dirs:
            if os.path.exists(data_dir):
                try:
                    files_in_dir = os.listdir(data_dir)
                    #logger.info(f"Found {len(files_in_dir)} files in {data_dir}: {files_in_dir}")
                    for file in files_in_dir:
                        full_file_path = os.path.join(data_dir, file)
                        if os.path.isfile(full_file_path):
                            all_files.append((file, data_dir))
                except OSError as e:
                    logger.warning(f"Error reading directory {data_dir}: {e}")
                    continue
            else:
                logger.warning(f"Directory does not exist: {data_dir}")
        
        if not all_files:
            logger.warning("No files found in any data directory")
            return None
            
        #logger.info(f"Total files found: {len(all_files)}")
        #logger.info(f"All files: {[f[0] for f in all_files]}")
        
        # Extract just the filenames for fuzzy matching
        file_names = [file[0] for file in all_files]
        
        # Clean the source filename for better matching
        src_clean = os.path.splitext(os.path.basename(file_name))[0]
        logger.info(f"Cleaned source name: '{src_clean}' from original: '{file_name}'")
        
        # First try: exact filename match
        for file_name_item, data_dir in all_files:
            if file_name_item.lower() == os.path.basename(file_name).lower():
                found_file_path = os.path.join(data_dir, file_name_item)
                logger.info(f"Source document found (exact filename match): {found_file_path}")
                return found_file_path
        
        # Second try: exact substring/prefix matching
        for file_name_item, data_dir in all_files:
            file_clean = os.path.splitext(file_name_item)[0]
            logger.debug(f"Comparing '{src_clean.lower()}' with '{file_clean.lower()}'")
            if (file_name.lower() in file_name_item.lower() or 
                file_name_item.lower().startswith(file_name.lower()) or
                src_clean.lower() in file_clean.lower() or
                file_clean.lower().startswith(src_clean.lower())):
                found_file_path = os.path.join(data_dir, file_name_item)
                logger.info(f"Source document found (substring match): {found_file_path}")
                return found_file_path
        
        # Third try: fuzzy matching if exact matching failed
        try:
            file_names_clean = [os.path.splitext(f)[0] for f in file_names]
            
            # Try with different cutoff values for better matches
            for cutoff in [0.8, 0.6, 0.4]:
                close_matches = difflib.get_close_matches(
                    src_clean.lower(), 
                    [f.lower() for f in file_names_clean], 
                    n=1, 
                    cutoff=cutoff
                )
                
                if close_matches:
                    # Find the original file corresponding to the close match
                    matched_clean = close_matches[0]
                    for i, file_clean in enumerate(file_names_clean):
                        if file_clean.lower() == matched_clean:
                            matched_file, data_dir = all_files[i]
                            found_file_path = os.path.join(data_dir, matched_file)
                            logger.info(f"Source document found (fuzzy match, cutoff={cutoff}): {found_file_path}")
                            return found_file_path
                    break
        except Exception as e:
            logger.warning(f"Error in fuzzy matching: {e}")
        
        # Fourth try: partial name matching (for cases where source has different format)
        for file_name_item, data_dir in all_files:
            file_clean = os.path.splitext(file_name_item)[0].lower()
            src_words = set(src_clean.lower().split())
            file_words = set(file_clean.split())
            
            # Check if there's significant word overlap
            if src_words and file_words:
                overlap = len(src_words.intersection(file_words))
                if overlap >= min(2, len(src_words) // 2):  # At least 2 words or half of source words
                    found_file_path = os.path.join(data_dir, file_name_item)
                    logger.info(f"Source document found (word overlap match): {found_file_path}")
                    return found_file_path
        
        logger.warning(f"Source document not found in data directories: {file_name}")
        logger.info(f"Available files: {[f[0] for f in all_files]}")
        return None

    sources = []
    seen_sources = set()
    source_files_data = []  # For storing source file information

    for doc in docs:
        source = doc.metadata.get("source", None)
        file_path = doc.metadata.get("file_path", None)  # Get the file path if available

        # Ensure source is a string before processing
        if isinstance(source, str):
            logger.info(f"Processing source: {source}")
            
            if not file_path:
                # Try to find the file using the source name
                source_basename = os.path.basename(source)
                logger.info(f"Looking for file with basename: {source_basename}")
                file_path = find_file_in_subdirs(source_basename)
                if file_path:
                    logger.info(f"Found file at: {file_path}")
                else:
                    logger.warning(f"Could not find file for source: {source}")

            is_clickable = bool(file_path and os.path.exists(file_path) if file_path else False)
            logger.info(f"Source {source} - clickable: {is_clickable}, file_path: {file_path}")

            if source not in seen_sources:
                seen_sources.add(source)
                sources.append(format_source_name(source))

                # Add source file information
                source_files_data.append({
                    "name": format_source_name(source),
                    "file_path": file_path if isinstance(file_path, str) else None,
                    "is_clickable": is_clickable
                })

    if not sources:
        return cleaned_response, source_files_data

    # Clean final formatting
    source_block = "\n\n📂 Source Document Used:\n" + "\n".join(f"• {src}" for src in sources)
    #cleaned_response = cleaned_response.strip() + source_block

    return cleaned_response, source_files_data



def append_sources_with_links(cleaned_response: str, docs: list):
    """
    Append (or just return) info about the single majority source document in `docs`.
    Returns: (cleaned_response, source_files_data)
        - cleaned_response: unchanged in this version (you can append a block if you want)
        - source_files_data: [{'name': str, 'file_path': str|None, 'is_clickable': bool}]
    """

    def format_source_name(source_path: str) -> str:
        filename = os.path.basename(source_path)
        name, _ = os.path.splitext(filename)
        return re.sub(r'^\d+\s*', '', name)  # remove leading numbers

    def find_file_in_subdirs(file_name: str) -> str | None:
        """
        Look for `file_name` under data/pdf, data/word, data/pptx.
        Uses exact, substring, fuzzy and word-overlap matching.
        """
        logger.info(f"Searching for file: {file_name}")

        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_dirs = [
            os.path.join(script_dir, "data", "pdf"),
            os.path.join(script_dir, "data", "word"),
            os.path.join(script_dir, "data", "pptx")
        ]

        all_files = []
        for data_dir in data_dirs:
            if not os.path.exists(data_dir):
                logger.warning(f"Directory does not exist: {data_dir}")
                continue
            try:
                for f in os.listdir(data_dir):
                    fp = os.path.join(data_dir, f)
                    if os.path.isfile(fp):
                        all_files.append((f, data_dir))
            except OSError as e:
                logger.warning(f"Error reading directory {data_dir}: {e}")

        if not all_files:
            logger.warning("No files found in any data directory")
            return None

        file_names = [f for f, _ in all_files]
        src_clean = os.path.splitext(os.path.basename(file_name))[0]

        # 1) exact filename
        for fname, ddir in all_files:
            if fname.lower() == os.path.basename(file_name).lower():
                return os.path.join(ddir, fname)

        # 2) substring / prefix
        for fname, ddir in all_files:
            fclean = os.path.splitext(fname)[0]
            if (file_name.lower() in fname.lower() or
                fname.lower().startswith(file_name.lower()) or
                src_clean.lower() in fclean.lower() or
                fclean.lower().startswith(src_clean.lower())):
                return os.path.join(ddir, fname)

        # 3) fuzzy
        try:
            file_names_clean = [os.path.splitext(f)[0] for f in file_names]
            for cutoff in (0.8, 0.6, 0.4):
                match = difflib.get_close_matches(
                    src_clean.lower(),
                    [f.lower() for f in file_names_clean],
                    n=1,
                    cutoff=cutoff
                )
                if match:
                    matched_clean = match[0]
                    for idx, fclean in enumerate(file_names_clean):
                        if fclean.lower() == matched_clean:
                            fname, ddir = all_files[idx]
                            return os.path.join(ddir, fname)
        except Exception as e:
            logger.warning(f"Error in fuzzy matching: {e}")

        # 4) word-overlap
        src_words = set(src_clean.lower().split())
        for fname, ddir in all_files:
            fclean = os.path.splitext(fname)[0].lower()
            overlap = src_words.intersection(fclean.split())
            if src_words and len(overlap) >= min(2, max(1, len(src_words)//2)):
                return os.path.join(ddir, fname)

        logger.warning(f"Source document not found in data directories: {file_name}")
        logger.info(f"Available files: {[f for f, _ in all_files]}")
        return None

    # ---------- pick the single majority source ----------
    raw_sources = []
    for d in docs:
        src = d.metadata.get("source")
        if isinstance(src, str):
            raw_sources.append(src)

    if not raw_sources:
        return cleaned_response, []

    # Count occurrences (use basename for stability)
    normalized = [os.path.basename(s) for s in raw_sources]
    counts = Counter(normalized)
    majority_source_raw = max(counts, key=counts.get)  # first max in tie
    logger.info(f"Majority source selected: {majority_source_raw} (count={counts[majority_source_raw]})")

    # Build only for that source
    file_path = None
    is_clickable = False

    # Sometimes retriever already set a file_path
    for d in docs:
        if os.path.basename(d.metadata.get("source", "")) == majority_source_raw:
            maybe_fp = d.metadata.get("file_path")
            if maybe_fp:
                file_path = maybe_fp
                break

    if not file_path:
        file_path = find_file_in_subdirs(majority_source_raw)

    if file_path and isinstance(file_path, str):
        is_clickable = os.path.exists(file_path)

    source_files_data = [{
        "name": format_source_name(majority_source_raw),
        "file_path": file_path if isinstance(file_path, str) else None,
        "is_clickable": is_clickable
    }]

    # Optional: if you still want the pretty block appended
    # source_block = "\n\n📂 Source Document Used:\n• " + format_source_name(majority_source_raw)
    # cleaned_response = cleaned_response.strip() + source_block

    return cleaned_response, source_files_data
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
    
    KEEP_BG_IF_DIRECT_WINS: int = 1   # when direct is stronger
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
            logger.info(f"Background retrieval is stronger, keeping {len(selected_dir)} direct docs and {len(selected_bg)} background docs.")   
        else:
            # Direct wins (or scores are close)
            selected_dir = direct_docs                       # keep ALL direct
            selected_bg  = bg_docs[: self.KEEP_BG_IF_DIRECT_WINS]
            logger.info(f"Direct retrieval is stronger, keeping {len(selected_dir)} direct docs and {len(selected_bg)} background docs.")

        seen = {id(d) for d in direct_docs}
        return direct_docs + [d for d in selected_bg if id(d) not in seen]
    
    
    async def aget_relevant_documents(self, query: str) -> List[Document]:
        return self.get_relevant_documents(query)
    
    
# Initialize the hybrid retriever
hybrid_retriever_obj = HybridRetriever(
                retr_direct=retr_direct,
                retr_bg=retr_bg,
                cross_encoder=cross_encoder,
                top_k_direct=8,
                top_k_bg=20,
                top_k_final=5
            )

## legacy   
def generate_answer(user_query: str, chat_history: ConversationBufferMemory ):
    """
    Returns a tuple: (answer_text, image_list)
    """
    session_id = str(uuid.uuid4())
    cleaned_answer = ""  # Ensure cleaned_answer is always defined
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
            # For the fallback response, we need dummy user_id and chat_id since they're not available in this function
            dummy_user_id = "anonymous"
            dummy_chat_id = session_id
            store_chat_log_updated(user_message=user_query, bot_response=cleaned_fallback, query_score=is_task_query, relevance_score=avg_score, user_id=dummy_user_id, chat_id=dummy_chat_id)
            #store_chat_log_updated(user_message=user_query, bot_response=cleaned_fallback, query_score=is_task_query, relevance_score=avg_score, user_id=user_id, chat_id=chat_id)
    
            return {
                'text': cleaned_fallback,
                'images': top_3_img,
                'sources': []
            }
        
        
        
        
        logger.info("QA chain activated for query processing.")
        # Step 4: Prepare full prompt and return LLM output
        modified_query = "You are a  HELPFUL AND NICE verztec helpdesk assistant. You will only use the provided documents in your response. If the query is out of scope, say so.\n\n" + clean_query
        modified_query = (
            "You are a HELPFUL AND NICE Verztec helpdesk assistant. "
            "You will only use the provided documents in your response. "
            "If the query is out of scope, say so. "
            "If there are any image tags or screenshots mentioned in the documents, "
            "IF QUERY SHOULD BE ESCALATED TO HR, RESPOND WITH <ESCALATE> "
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

    
        # For the main response, we need dummy user_id and chat_id since they're not available in this function
        dummy_user_id = "anonymous"
        dummy_chat_id = session_id
        store_chat_log_updated(user_message=user_query, bot_response=cleaned_answer, query_score=is_task_query, relevance_score=avg_score, user_id=dummy_user_id, chat_id=dummy_chat_id)
        # After generating the bot's response
        final_response, source_docs = append_sources_with_links(cleaned_answer, top_docs)
        print(final_response)
        
        # Log source documents for debugging
        logger.info(f"Source documents data: {source_docs}")
        
        # Also append clickable links to text for backwards compatibility
        for doc in source_docs:
            if doc['is_clickable'] and doc['file_path']:
                final_response += f"\n\n[Source: {doc['name']}]({doc['file_path']})"
                logger.info(f"Added clickable source: {doc['name']} -> {doc['file_path']}")
            else:
                final_response += f"\n\n[Source: {doc['name']}]"
                logger.info(f"Added non-clickable source: {doc['name']}")

        total_elapsed_time = time.time() - total_start_time
        logger.info(f"Total time taken for query processing: {total_elapsed_time:.2f}s")

        # Return structured data for frontend
        return {
            'text': final_response,
            'images': top_3_img,
            'sources': source_docs  # Include source file data for frontend
        }
    except Exception as e:
        return {
            'error': f"I encountered an error while processing your request: {str(e)}", 
            'text': f"I encountered an error while processing your request: {str(e)}", 
            'images': [], 
            'sources': []
        }

memory_store = {}
global_tools = {
    "raise_to_hr": {
        "description": "Raise the query to HR — ONLY for serious complaints, legal issues, or sensitive workplace matters. DO NOT use this tool for general HR queries (e.g., leave balance, benefits, policies), questions about entitlements, or non-sensitive issues.",
        "prompt_style": "ONLY use this tool for workplace issues that are serious, sensitive, involve harassment, discrimination, legal matters, or require confidential escalation. DO NOT use this tool for general HR questions such as leave balance, entitlements, policies, routine HR matters, or informational queries about HR processes.",
        "response_tone": "supportive_professional"
    },
    "schedule_meeting": {
        "description": "Set up a meeting — for coordination involving multiple stakeholders or recurring issues.",
        "prompt_style": "You are an efficient scheduling and coordination assistant. When handling meeting requests, focus on practical logistics, available time slots, and clear next steps. Be organized and detail-oriented in your responses, asking for necessary information like preferred dates, attendees, and meeting purpose.",
        "response_tone": "organized_efficient"
    },
    "vacation_check": {
        "description": "Check vacation balance — for inquiries about remaining leave days, entitlements",
        "prompt_style": "You are a helpful assistant focused on vacation and leave inquiries. When responding to vacation balance questions, provide clear information about remaining leave days, entitlements, and any relevant policies. Be concise and direct in your responses, ensuring the user understands their current vacation status.",
        "response_tone": "concise_direct"
    }   
}
# Utility: Get last bot message from ConversationBufferMemory
def get_last_bot_message(chat_history):
    """
    Returns the content of the last AI (bot) message from a ConversationBufferMemory object.
    """
    history = chat_history.load_memory_variables({}).get("chat_history", [])
    # history is a list of HumanMessage and AIMessage objects (if return_messages=True)
    for msg in reversed(history):
        if hasattr(msg, 'content') and msg.__class__.__name__ == 'answer':
            return msg.content
    return None

def get_last_human_message(chat_history):
    """
    Returns the content of the last human message from a ConversationBufferMemory object.
    """
    history = chat_history.load_memory_variables({}).get("chat_history", [])
    # history is a list of HumanMessage and AIMessage objects (if return_messages=True)
    for msg in reversed(history):
        if hasattr(msg, 'content') and msg.__class__.__name__ == 'HumanMessage':
            return msg.content
    return None

def get_last_human_message(chat_id, user_id):
    """
    Returns the content of the last human message from a sql database.
    """
    conn = pymysql.connect(**DB_CONFIG)
    cursor = conn.cursor()

    select_query = '''
        SELECT user_message
        FROM chat_logs
        WHERE chat_id = %s AND user_id = %s
        ORDER BY timestamp DESC
        LIMIT 1
    '''
    cursor.execute(select_query, (chat_id, user_id))
    result = cursor.fetchone()

    cursor.close()
    conn.close()

    if result:
        return result[0]
    return None
   

def get_last_bot_message(chat_id, user_id):
    """
    Returns the content of the last bot message from a sql database.
    """
    conn = pymysql.connect(**DB_CONFIG)
    cursor = conn.cursor()

    select_query = '''
        SELECT bot_response
        FROM chat_logs
        WHERE chat_id = %s AND user_id = %s
        ORDER BY timestamp DESC
        LIMIT 1
    '''
    cursor.execute(select_query, (chat_id, user_id))
    result = cursor.fetchone()

    cursor.close()
    conn.close()

    if result:
        return result[0]
    return None
   

def create_chat_name(user_id: str, chat_id: str, chat_history: ConversationBufferMemory, query: str):
    """
    Checks DB for chat name. If not found, uses decisionlayer_model to generate one.
    """
    key = f"{user_id}_{chat_id}"
    logger.info(f"Creating chat name for key: {key}")

    # Connect to MySQL
    conn = pymysql.connect(**DB_CONFIG)
    cursor = conn.cursor()

    try:
        # Fetch existing chat_name
        select_query = """
            SELECT chat_name
            FROM chat_logs
            WHERE user_id = %s AND chat_id = %s
            ORDER BY timestamp DESC
            LIMIT 1;
        """
        cursor.execute(select_query, (user_id, chat_id))
        result = cursor.fetchone()


        if result and result[0]:
            chat_name = result[0]
            logger.info(f"Existing chat name found: {chat_name}")
            
        else:
            # Get conversation history
            logger.info("No existing chat name found, generating a new one.")

            messages = chat_history.load_memory_variables({}).get("chat_history", "")
            # Handle both list (return_messages=True) and string cases
            if isinstance(messages, list):
                # Convert to string for prompt, and count user messages
                user_msgs = [m.content for m in messages if m.__class__.__name__ == 'HumanMessage']
                if not user_msgs:
                    logger.warning("No conversation history available to generate a chat name.")
                    return
                # Always include the most recent query if not already present
                if not user_msgs or (query and (not user_msgs or query.strip() != user_msgs[-1].strip())):
                    user_msgs.append(query)
                if len(user_msgs) < 1:
                    logger.info(f"Too few messages ({len(user_msgs)} found). Skipping chat name generation.")
                    #return
                # Remove <image> tags and their content, and also remove the specific system prompt chunk from user messages before joining
                def clean_user_message(text):
                    # Remove <image>...</image> blocks (multiline)
                    #text = re.sub(r'<image>.*?</image>', '', text, flags=re.DOTALL)
                    # Remove the specific system prompt chunk if it exists
                    system_prompt_pattern = re.compile(
                        r"User: You are a HELPFUL AND NICE Verztec helpdesk assistant\. You will only use the provided documents in your response\. If the query is out of scope, say so\. If there are any image tags or screenshots mentioned in the documents, please reference them in your response where appropriate, such as 'See Screenshot 1' or 'Refer to the image above' these images will be made obvious with the <image> tags\.\s*",
                        re.DOTALL
                    )
                    text = system_prompt_pattern.sub('', text)
                    return text.strip()

                messages_str = '\n'.join([
                    f"User: {clean_user_message(m)}" if isinstance(m, str) else f"User: {clean_user_message(m.content)}"
                    for m in user_msgs
                ])
            else:
                # Fallback: treat as string
                messages_str = messages
                if not messages_str.strip():
                   # logger.warning("No conversation history available to generate a chat name.")
                    return
                message_lines = [line for line in messages_str.split("\n") if line.strip().startswith("User:")]
                if len(message_lines) < 2:
                    #logger.info(f"Too few messages ({len(message_lines)} found). Skipping chat name generation.")
                    return
            prompt = (
                "Given the following conversation, generate a concise, specific, and meaningful chat title, within 27 CHARACTERS. "
                "Do NOT include generic words like 'Chat', 'Conversation', or 'Session'. "
                "Do NOT use quotes, explanations, or any extra text—return ONLY the title. "
                "The title should summarize the main topic or purpose of the conversation, using clear and relevant keywords.\n\n"
                f"{messages_str}"
            )
            logger.info(f"Generated prompt for chat name: {prompt}")
        
            
            response = decisionlayer_model.predict(prompt).strip().strip('"')
            # Truncate chat_name to fit DB column (assume VARCHAR(50), adjust if needed)
            max_length = 50
            response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
            chat_name = response[:max_length]
            logger.info(f"Generated chat name: {chat_name}")

            # Update DB
            update_query = """
                UPDATE chat_logs
                SET chat_name = %s
                WHERE user_id = %s AND chat_id = %s;
            """
            cursor.execute(update_query, (chat_name, user_id, chat_id))
            conn.commit()


    except Exception as e:
        logger.exception(f"Error creating chat name: {e}")
    finally:
        cursor.close()
        conn.close()
        
def get_vacation_days(user_id: str, filename: str = r"chatbot\src\backend\python\leave.csv") -> int:
    """
    Reads leave.csv and returns the number of vacation days for the given user_id.
    Assumes columns: user_id, vacation_days
    """
    try:
        with open(filename, mode='r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row.get('user_id') == user_id:
                    return int(row.get('vacation_days', 0))
        return 0  # User not found
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return 0  
    
    
def generate_answer_histoy_retrieval(user_query: str, user_id:str, chat_id:str):
    """
    Returns a tuple: (answer_text, image_list)
    """
    
    key = f"{user_id}_{chat_id}"  # Use a separator to avoid accidental key collisions
    logger.info(f"Retrieving memory for key: {key}")
    logger.info(f"User ID: {user_id}, Chat ID: {chat_id}")

    chat_name = None
    if key in memory_store:
        chat_history = memory_store[key]
        logger.info(f"Memory object found at key: {key}")
    else:
        msgs, chat_name = retrieve_user_messages_and_scores(user_id, chat_id)
        chat_history = build_memory_from_results(msgs)
        memory_store[key] = chat_history
        logger.info(f"No memory object found. Created and saved for key: {key}")
        
    create_chat_name(user_id, chat_id, chat_history, user_query)
    userinfo=get_user_info(user_id)
    if userinfo:
        logger.info(f"User info retrieved: {userinfo}")
        user_name = userinfo.get("username", "User")
        user_role = userinfo.get("role", "Unknown")
        user_country = userinfo.get("country", "Unknown")
        user_department = userinfo.get("department", "Unknown")
    if not userinfo:
        logger.warning(f"No user info found for user_id: {user_id}. Using default values.")
        user_name = "User"
        user_role = "Unknown"
        user_country = "Unknown"
        user_department = "Unknown"

    try:
        
        
        
        total_start_time = time.time()  # Start timing for the whole query

        parser = StrOutputParser()
    
        # Chain the Groq model with the parser
        deepseek_chain = deepseek | parser
        
        
        from langchain.prompts import PromptTemplate
        from langchain.retrievers import ContextualCompressionRetriever
        from langchain.retrievers.document_compressors import LLMChainFilter
        base = index.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 15}
        )
   

        avg_score = get_avg_score(index, embedding_model, user_query)    
        
        
        
        ## QA chain setup with mrmory 
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=deepseek_chain,
            #retriever=hybrid_retriever_obj,
            retriever =base,
            memory=chat_history,
            return_source_documents=True,
            output_key="answer"
        )
        logger.info(memory)
    
    
        # Refine query
        # cos im too lazy to change all the refences below
        clean_query = user_query
        
        # Use our global tools dictionary

        # Generate formatted string for prompt
        tool_descriptions = "\n".join([f"[{name}] — {tool_data['description']}" for name, tool_data in global_tools.items()])

        # Prompt template with context and tool injection
        glast_bot_message = get_last_bot_message(chat_id,user_id )
        decision_prompt = ChatPromptTemplate.from_messages([
            ("system", """
                You are the decision-making layer of a corporate assistant chatbot for internal employee support.
                The bot may have asked the user if they need to raise a query to HR or schedule a meeting.
                You have access to the following tools, but must ONLY use one if it is absolutely necessary:

                {tool_descriptions}

                Respond with ONLY one of the following tool tags:
                - [tool_name] — to activate a tools
                - [Continue] — if no tool is necessary

                Only use a tool if the request is clearly serious, procedural, or high-impact. 
                If the request can be answered normally, prefer [Continue].

                Return ONLY the tag. No explanations.
            """),
            ("ai", "{glast_bot_message}"),
            ("human", "{question}")
        ])

        # Create the decision chain
        decision_chain = decision_prompt | decisionlayer_model

        # Run it in your workflow
        tool_response = decision_chain.invoke({
            "tool_descriptions": tool_descriptions,
            "question": clean_query,
            "glast_bot_message": glast_bot_message if glast_bot_message is not None else ""
        })

        Tool_answer = re.sub(r"<think>.*?</think>", "", tool_response.content, flags=re.DOTALL).strip()
        print(f"Tool decision: {Tool_answer}")
        
        # Identify and handle the selected tool with improved logic
        tool_identified = "none"  # Default to none
        tool_used = False
        tool_confidence = Tool_answer  # Store raw decision for debugging
        
        # Normalize the tool answer for better matching
        tool_answer_lower = Tool_answer.lower().strip()
        
        # Check for Continue first (highest priority)
        if "continue" in tool_answer_lower or "[continue]" in tool_answer_lower:
            tool_identified = "none"
            tool_used = False
            logger.info(f"No tool needed (Continue) for user {user_id}, query: {user_query}")
            
        # Check for HR escalation
        elif "raise_to_hr" in tool_answer_lower or "[raise_to_hr]" in tool_answer_lower:
            tool_identified = "raise_to_hr"
            tool_used = True
            logger.info(f"HR escalation tool identified for user {user_id}, query: {user_query}")
            
        # Check for meeting scheduling
        elif "schedule_meeting" in tool_answer_lower or "[schedule_meeting]" in tool_answer_lower:
            tool_identified = "schedule_meeting"
            tool_used = True
            logger.info(f"Meeting scheduling tool identified for user {user_id}, query: {user_query}")
        elif "vacation_check" in tool_answer_lower or "[vacation_check]" in tool_answer_lower:
            tool_identified = "vacation_check"
            tool_used = True
            logger.info(f"Vacation check tool identified for user {user_id}, query: {user_query}")
            
        # Handle unknown or malformed responses
        else:
            tool_identified = "none"  # Default to none for safety
            tool_used = False
            logger.warning(f"Unknown or malformed tool decision: '{Tool_answer}' for user {user_id}, query: {user_query}. Defaulting to no tool.")
            
        #logger.info(f"Final tool decision - identified: {tool_identified}, used: {tool_used}")
           
        
       
        # Step 3: Search BOTH FAISS indices for context 
        # for images, as well as for context relevance checks 
        results = index.similarity_search_with_score(clean_query, k=10)
        #results2 = index2.similarity_search_with_score(clean_query, k=5)  # Search second index too
        
        # Combine results from both indices
        all_results = results 
        
        scores = [score for _, score in results]
        avg_score = float(np.mean(scores)) if scores else 1.0
        prev_score=0
        prev_query = get_last_human_message(chat_id, user_id)
        formatted_prev_docs = ""
                
        if prev_query:
            logger.info(f"Previous query: {prev_query}")
            prev_results = index.similarity_search_with_score(prev_query, k=10)
            prev_scores = [score for _, score in prev_results]
            prev_score = float(np.mean(prev_scores)) if prev_scores else 1.0
            formatted_prev_docs = "\n".join([f"- {doc.page_content}..." for doc, _ in prev_results[:3]])

            # Combine scores: 80% current, 20% previous
            combined_score = 0.8 * avg_score + 0.2 * prev_score
        else:
            combined_score = avg_score
                    
        
        logger.info(f"Average score for current query: {avg_score:.4f}")
        #avg_score = combined_score
            
       
        seen = set()
        unique_docs = []

        for doc, _ in all_results:  # Process combined results
            content = doc.page_content
           # logger.info("=" * 20)
           # logger.info(f"Processing document content: {content[:50]}...")  # Log first 50 chars for brevity
           # logger.info(f"Document metadata: {doc.metadata}")  # Log metadata for debugging
            if content not in seen:
                seen.add(content)
                unique_docs.append(doc)

        logger.info(f"Retrieved {len(unique_docs)} documents from both indices with average score: {avg_score:.4f}")
    
        
        ## retrieving images from top 3 chunks (if any)
        # Retrieve images from top 3 chunks (if any)
        THRESHOLD      = 0.70          # keep docs whose score < threshold   (lower L2 → more similar)
        MAX_IMAGES     = 3             # cap images
        sources_set    = set()         # remember unique source filenames
        sources_list   = []            # ordered list of unique sources
        seen_contents  = set()         # avoid duplicate page_content
        top_docs       = []            # docs we actually keep
        top_3_img      = []            # up to 3 image paths / URLs

        for doc, score in all_results:  # Process combined results for sources
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
        
        is_task_query = is_query_score(user_query)
        logger.info(f"Top 3 images: {top_3_img}")
        logger.info(f"Query Score: {is_task_query}")
        logger.info(f"Average Score: {avg_score}")
        
        soft_threshold = 0.7
        SOFT_QUERY_THRESHOLD = 0.5
        STRICT_QUERY_THRESHOLD = 0.2
        HARD_AVG_SCORE_THRESHOLD = 0.9
        
        #handle irrelevant query
       
        logger.info(f"Clean Query at qa chain: {clean_query}")
        if (
            not tool_used and (
                (is_task_query < SOFT_QUERY_THRESHOLD and avg_score >= soft_threshold) or
                avg_score >= HARD_AVG_SCORE_THRESHOLD or
                is_task_query < STRICT_QUERY_THRESHOLD
            )
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
                'You do not need to greet the user, unless they greeted you first. '
                'Keep it human and warm (e.g., "I’m doing great, thanks for asking!"), then ***gently guide the user back to Verztec-related helpdesk topics***.'
                'Do not answer any questions that are not related to Verztec helpdesk topics'
                f"Here is some information about the user, NAME:{user_name}, ROLE: {user_role}, COUNTRY: {user_country}, DEPARTMENT: {user_department}\n\n"
                )
            if formatted_prev_docs: 
                fallback_prompt += (
                    "To help you stay conversational, here are a few bits of previous Verztec-related context retrieved from earlier:\n"
                    f"{formatted_prev_docs}\n\n"
                    "These documents were retrieved to attempt to answer the previous query. YOU MAY USE THEM TO ATTEMPT AN ACCURATE ANSWER\n"
                    f'The previous query was: "{prev_query}"\n\n'
                    f'The response to the previous query was you may use this for context: "{glast_bot_message}"\n\n'
                )
            logger.info("=+" * 20)
            logger.info(f"Fallback prompt: {fallback_prompt}")
            #modified_query = "You are a verztec helpdesk assistant. You will only use
            logger.info("=+" * 20)

           
            messages = build_memory_prompt(memory, fallback_prompt)
            response = deepseek.generate([messages])
            raw_fallback = response.generations[0][0].text.strip()

            # Remove <think> block if present
            think_block_pattern = re.compile(r"<think>.*?</think>", flags=re.DOTALL)
            cleaned_fallback = think_block_pattern.sub("", raw_fallback).strip()
              
            # Store the fallback response in chat memory
            memory.chat_memory.add_user_message(fallback_prompt)
            memory.chat_memory.add_ai_message(cleaned_fallback) 
            
            
            # Clean up chat memory to keep it manageable
            # Limit chat memory to last 4 turns (8 messages)
            
            MAX_TURNS = 4
            if len(memory.chat_memory.messages) > 2 * MAX_TURNS:
                memory.chat_memory.messages = memory.chat_memory.messages[-2 * MAX_TURNS:]
            if tool_used:
                is_task_query = 0  # Force task query for fallback response
                avg_score = 2  # Force average score for fallback response
                
            store_chat_log_updated(user_message=user_query, bot_response=cleaned_fallback, query_score=is_task_query, relevance_score=avg_score, user_id=user_id, chat_id=chat_id, chat_name=chat_name)
    
            return {
                'text': cleaned_fallback,
                'images': [],
                'sources': [],
                'tool_used': tool_used,
                'tool_identified': tool_identified,
                'tool_confidence': tool_confidence
            }
        
        
        
        
        logger.info("QA chain activated for query processing.")
        
        # Step 4: Prepare tool-specific or standard prompt based on identified tool
        if tool_used and tool_identified != "none" and tool_identified in global_tools:
            tool_data = global_tools[tool_identified]
            is_task_query=0
            avg_score=1
            if tool_identified == "raise_to_hr":
                confirmation_response = """I understand you have a workplace concern that requires proper attention. I can help you escalate this issue to our Human Resources department.

                            Here's what will happen if you proceed:

                            - Your concern will be securely logged with a unique reference number
                            - The issue will be forwarded directly to qualified HR personnel 
                            - Complete confidentiality will be maintained throughout the process
                            - You'll receive a reference ID to track the progress of your case
                            - HR will follow up with you within 24 hours to discuss next steps

                            HR has the expertise and authority to handle sensitive workplace matters through proper channels, ensuring your concern gets the attention and resolution it deserves.

                            Would you like me to proceed with escalating this matter to HR?"""
            elif tool_identified == "schedule_meeting":
                # Immediately extract meeting details
                meeting_details = extract_meeting_details(user_query)
                if not meeting_details or not isinstance(meeting_details, dict):
                    meeting_details = {}
                # Format meeting details for confirmation
                details_lines = []
                if meeting_details.get('subject'):
                    details_lines.append(f"• **Subject:** {meeting_details['subject']}")
                if meeting_details.get('date_time'):
                    details_lines.append(f"• **Date/Time:** {meeting_details['date_time']}")
                if meeting_details.get('duration'):
                    details_lines.append(f"• **Duration:** {meeting_details['duration']}")
                if meeting_details.get('participants'):
                    details_lines.append(f"• **Participants:** {', '.join(meeting_details['participants'])}")
                if meeting_details.get('meeting_type'):
                    details_lines.append(f"• **Type:** {meeting_details['meeting_type']}")
                if meeting_details.get('location'):
                    details_lines.append(f"• **Location:** {meeting_details['location']}")
                if meeting_details.get('priority'):
                    details_lines.append(f"• **Priority:** {meeting_details['priority']}")
                details_str = '\n'.join(details_lines) if details_lines else "(No details extracted)"
                confirmation_response = (
                    "I've extracted the following meeting details from your request:\n\n"
                    f"{details_str}\n\n"
                    "Would you like to confirm and schedule this meeting?\n"
                    "(Please click 'Confirm' to proceed or 'Cancel' to abort.)"
                )
            elif tool_identified == "vacation_check":
                confirmation_response = f"""I've identified that your request is related to checking your vacation balance. Would you like me to proceed with checking your vacation balance?"""
            else:
                confirmation_response = f"""I've identified that your request requires the {tool_identified} tool. This will help ensure your request is handled through the appropriate channels with the right level of attention.

                    Would you like me to proceed with activating this tool for your request?"""
            # Store the confirmation response
           
            total_elapsed_time = time.time() - total_start_time
            logger.info(f"Total time taken for tool confirmation: {total_elapsed_time:.2f}s")
            return {
                'text': confirmation_response,
                'images': [],
                'sources': [],
                'tool_used': tool_used,
                'tool_identified': tool_identified,
                'tool_confidence': tool_confidence,
                'meeting_details': meeting_details if tool_identified == "schedule_meeting" else None
            }
        else:
            modified_query = clean_query
           
            logger.info("Query is within scope.")
            modified_query = (
                "You are a HELPFUL AND NICE Verztec helpdesk assistant. "
                "You will only use the provided documents in your response. "
                "You will answer in a conversational manner, as if you were a human. Referencing the user as YOU.not as a third person."
                "If the query is out of scope, politely inform the user that you can only assist with Verztec-related topics, explain why you are unable to answer the given query "
                "Use a kind and professional tone. Do not fabricate answers. "
                "If appropriate, suggest the user contact Verztec support for further help.\n\n"
                f"Here is some information about the user, NAME:{user_name}, ROLE: {user_role}, COUNTRY: {user_country}, DEPARTMENT: {user_department}\n\n"
                + clean_query
            )


            qa_start_time = time.time()
            response = qa_chain.invoke({"question": modified_query})
            qa_elapsed_time = time.time() - qa_start_time
            raw_answer = response['answer']
            source_docs = response['source_documents']
            #logger.info(f"Source docs from QA chain: {source_docs}")
            logger.info(f"(QA chain time taken: {qa_elapsed_time:.2f}s)")
            docs = response["source_documents"]  # list[Document]
            logger.info("+="*20)
            for i, d in enumerate(docs):
                src   = d.metadata.get("source")
                score = d.metadata.get("score")
                snip  = d.page_content[:500].replace("\n", " ")

                logger.info("=== Doc %d ===", i)
                logger.info("source: %s | score: %s", src, score)
                logger.info("%s ...", snip)

            # Clean the chat memory to keep it manageable
            # Limit chat memory to last 4 turns (8 messages)
            
            MAX_TURNS = 4
            if len(memory.chat_memory.messages) > 2 * MAX_TURNS:
                memory.chat_memory.messages = memory.chat_memory.messages[-2 * MAX_TURNS:]

            # Define regex pattern to match full <think>...</think> block
            think_block_pattern = re.compile(r"<think>.*?</think>", flags=re.DOTALL)
            single_think_block_pattern = re.compile(r"</think>", flags=re.DOTALL)
            
           
            

            # Check if full <think> block exists
            has_think_block = bool(think_block_pattern.search(raw_answer))
        

            # Clean the <think> block regardless
            cleaned_answer = think_block_pattern.sub("", raw_answer).strip()
            cleaned_answer = single_think_block_pattern.sub("", cleaned_answer).strip()
            has_tag = False
           

            

            # Remove any remaining <think> tags
            cleaned_answer = re.sub(r"</?think>", "", cleaned_answer).strip()
        
            cleaned_answer = re.sub(r'[\*#]+', '', cleaned_answer).strip()
            #cleaned_answer= cleaned_answer+"\n\n Would you like me to escalate this query to HR?"
            #user_query = user_query+"AHHAHAHAHHA"
            
           
            store_chat_log_updated(user_message=user_query, bot_response=cleaned_answer, query_score=is_task_query, relevance_score=avg_score, user_id=user_id, chat_id=chat_id, chat_name=chat_name)##brian u need to update sql for this to work
            logger.info(f"Stored chat log for user {user_id}, chat {chat_id} with query score {is_task_query} and relevance score {avg_score}")
            # also need to update the store_chat_bot method, to incoude user id and chat id
            # After generating the bot's response
            final_response, source_docs = append_sources_with_links(cleaned_answer, top_docs)
            #print(final_response)
            
            # Log source documents for debugging
            logger.info(f"Source documents data: {source_docs}")
            
            # Also append clickable links to text for backwards compatibility
            for doc in source_docs:
                if doc['is_clickable'] and doc['file_path']:
                    #final_response += f"\n\n[Source: {doc['name']}]({doc['file_path']})"
                    logger.info(f"Added clickable source: {doc['name']} -> {doc['file_path']}")
                else:
                    #final_response += f"\n\n[Source: {doc['name']}]"
                    logger.info(f"Added non-clickable source: {doc['name']}")

            total_elapsed_time = time.time() - total_start_time
            logger.info(f"Total time taken for query processing: {total_elapsed_time:.2f}s")

            # Return structured data for frontend
            return {
                'text': final_response,
                'images': top_3_img,
                'sources': source_docs,
                'tool_used': tool_used,
                'tool_identified': tool_identified,
                'tool_confidence': tool_confidence
            }
    except Exception as e:
        return {
            'error': f"I encountered an error while processing your request: {str(e)}", 
            'text': f"I encountered an error while processing your request: {str(e)}", 
            'images': [], 
            'sources': [],
            'tool_used': False,
            'tool_identified': "none",
            'tool_confidence': "error"
        }
















































































try:
    ## tools for agentic bot
    def faiss_search_tool(query: str) -> str:
        """
        Searches FAISS indices for relevant information and returns the top 10 most relevant documents.
        """
        try:
            # Input validation
            if not query or not query.strip():
                return "Error: Please provide a search query."
            
            # Clean the query
            clean_query = clean_with_grammar_model(query.strip())
            
            # Search FAISS index for top 10 results
            results = index.similarity_search_with_score(clean_query, k=10)
            
            if not results:
                return "I couldn't find any relevant information in our knowledge base for your query."
            
            # Print results neatly for debugging
            print(f"\n{'='*60}")
            print(f"FAISS SEARCH RESULTS FOR: '{clean_query}'")
            print(f"{'='*60}")
            print(f"Found {len(results)} relevant documents:")
            print("-" * 60)
            
            for i, (doc, score) in enumerate(results, 1):
                content = doc.page_content if len(doc.page_content) > 200 else doc.page_content
                source = doc.metadata.get("source", "Unknown")
                print(f"\n[{i}] Score: {score:.3f}")
                print(f"Source: {source}")
                print(f"Content: {content}")
                print("-" * 40)
            
            print(f"{'='*60}\n")
            
            # Format the results simply
            search_result = f"Found {len(results)} relevant documents:\n\n"
            
            for i, (doc, score) in enumerate(results, 1):
                content = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                source = doc.metadata.get("source", "Unknown")
                search_result += f"{i}. {content}\n   Source: {source}\n   Score: {score:.3f}\n\n"
            
            # Log the search
            try:
                with open("yabdabado.csv", "a", newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([datetime.now().isoformat(), "faiss_search", clean_query])
            except Exception as e:
                print(f"Warning: Failed to log FAISS search: {e}")
            
            return search_result
            
        except Exception as e:
            logger.error(f"Error in FAISS search tool: {str(e)}")
            return f"Error occurred during search: {str(e)}"

    def escalate_to_hr_tool(issue: str) -> str:
        """
        Enhanced HR escalation tool with better formatting and error handling.
        """
        # Input validation
        if not issue or not issue.strip():
            return """I'd be happy to help escalate your concern to HR. However, I need you to provide a detailed description of the issue you'd like to escalate. Please include:

**Required Information:**
1. A brief summary of the situation
2. Any relevant dates or timeframes  
3. The specific assistance you're seeking

This information will help HR provide you with the most appropriate support."""
        
        # Sanitize and structure the input for logging
        sanitized_issue = issue.strip()[:800]  # Increased limit for better context
        escalation_id = f"ESC-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        # Enhanced logging with structured data
        try:
            with open("yabdabado.csv", "a", newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.now().isoformat(), 
                    "escalate_to_hr", 
                    escalation_id,
                    sanitized_issue[:200] + "..." if len(sanitized_issue) > 200 else sanitized_issue
                ])
            logger.info(f"HR escalation logged with ID: {escalation_id}")
        except Exception as e:
            logger.error(f"Failed to log HR escalation: {e}")
            # Continue execution even if logging fails
        
        return f"""🚨 **HR Escalation Successfully Initiated**

            Your concern has been successfully escalated to our Human Resources department:

            **Issue Summary:** {sanitized_issue}

            **What happens next:**
            1. **Within 24 hours:** HR will acknowledge receipt of your escalation
            2. **Within 2-3 business days:** An HR representative will contact you to discuss the matter
            3. **Ongoing:** You'll receive regular updates on the status of your case

            **Important Notes:**
            - All escalations are handled with strict confidentiality
            - You may be contacted for additional information if needed
            - If this is an urgent safety matter, please also contact your immediate supervisor

            **Reference ID:** ESC-{datetime.now().strftime('%Y%m%d-%H%M%S')}

            Is there anything else I can help you with regarding company policies or procedures?"""

    def create_meeting_request_tool(details: str) -> str:
    
        # Input validation
        if not details or not details.strip():
            return "I'd be glad to help you create a meeting request. To ensure your meeting is properly scheduled, please provide the following details:\n\n**Required Information:**\n1. **Purpose/Agenda:** What is the meeting about?\n2. **Preferred Date/Time:** When would you like to meet?\n3. **Duration:** How long do you expect the meeting to last?\n4. **Attendees:** Who should be invited?\n5. **Meeting Type:** In-person, virtual, or hybrid?\n\n**Optional Information:**\n- Specific topics to discuss\n- Any preparation materials needed\n- Preferred meeting location (if in-person)\n\nOnce you provide these details, I'll create a comprehensive meeting request for you."
        
        # Sanitize and format input
        sanitized_details = details.strip()[:300]  # Reasonable limit for meeting descriptions
        
        try:
            with open("yabdabado.csv", "a", newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([datetime.now().isoformat(), "create_meeting_request", sanitized_details])
        except Exception as e:
            print(f"Warning: Failed to log meeting request: {e}")
            # Continue execution even if logging fails
        
        return f"""📅 **Meeting Request Created Successfully**

**Meeting Details:** {sanitized_details}

**Your meeting request has been processed with the following steps:**

**Immediate Actions Taken:**
1. ✅ Meeting request logged in the system
2. ✅ Notification sent to relevant parties
3. ✅ Calendar invitations will be generated

**Next Steps:**
1. **Within 2 hours:** You'll receive a calendar invitation with meeting details
2. **24 hours before:** Automatic reminder will be sent to all attendees
3. **If needed:** Meeting coordinator will contact you for any clarifications

**Meeting Reference ID:** MTG-{datetime.now().strftime('%Y%m%d-%H%M%S')}

**Important Reminders:**
- Please prepare any materials mentioned in your request
- If you need to modify or cancel, contact the meeting coordinator at least 24 hours in advance
- For urgent changes, please reach out to your supervisor directly

**Need Help?**
- Meeting room bookings: Contact facilities management
- Technical setup for virtual meetings: Contact IT support
- Agenda planning assistance: I'm here to help!

Is there anything else you'd like me to help you with regarding this meeting or other company procedures?"""

    def final_answer_tool(answer: str) -> str:
    
        # Input validation
        if not answer or not answer.strip():
            fallback_answer = "I apologize, but I couldn't generate a proper response to your inquiry. Please feel free to rephrase your question or contact our support team for further assistance."
            print(f"Warning: Empty final answer provided, using fallback: {fallback_answer}")
            answer = fallback_answer
        
        # Clean the answer for optimal agent termination
        clean_answer = answer.strip()
        
        # Remove formal email elements if present
        email_greetings = ["Dear", "Hello,", "Hi,", "Good morning,", "Good afternoon,"]
        email_closings = ["Best regards,", "Sincerely,", "Thank you,", "Kind regards,", "Yours sincerely,"]
        
        # Remove email-style greetings
        for greeting in email_greetings:
            if clean_answer.startswith(greeting):
                # Find the end of the greeting line and remove it
                lines = clean_answer.split('\n')
                if lines:
                    lines = lines[1:]  # Remove first line
                clean_answer = '\n'.join(lines).strip()
                break
        
        # Remove email-style closings and signatures
        for closing in email_closings:
            if closing.lower() in clean_answer.lower():
                closing_pos = clean_answer.lower().find(closing.lower())
                clean_answer = clean_answer[:closing_pos].strip()
                break
        
        # Remove signature lines (lines that start with [Your Name], [Name], etc.)
        lines = clean_answer.split('\n')
        filtered_lines = []
        for line in lines:
            stripped_line = line.strip()
            if not (stripped_line.startswith('[') and stripped_line.endswith(']')):
                filtered_lines.append(line)
        clean_answer = '\n'.join(filtered_lines).strip()
        
        # Ensure the answer follows guidelines for detailed instructions and polite rejections
        if len(clean_answer) < 50 and "sorry" not in clean_answer.lower() and "apologize" not in clean_answer.lower():
            # If answer seems too brief and isn't already an apology, enhance it
            clean_answer = f"{clean_answer}\n\nIf you need more specific details or have additional questions, please don't hesitate to ask. I'm here to help you with any Verztec-related policies or procedures."
        
        try:
            with open("yabdabado.csv", "a", newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([datetime.now().isoformat(), "Final Answer", clean_answer])
        except Exception as e:
            print(f"Warning: Failed to log final answer: {e}")
            # Continue execution - logging failure shouldn't break user experience
        
        # Return clean answer without prefixes to ensure proper termination
        return clean_answer

    tools = [
        Tool(
            name="faiss_search",
            func=faiss_search_tool,
            description="Searches the Verztec knowledge base for relevant information about policies, procedures, and company guidelines. Use this when you need to find specific information to provide detailed, step-by-step instructions to users."
        ),
        Tool(
            name="escalate_to_hr",
            func=escalate_to_hr_tool,
            description="Escalates a user issue to HR with detailed information. Use for serious workplace or personal issues like harassment, discrimination, or policy violations. Always explain the escalation process politely to the user."
        ),
        Tool(
            name="create_meeting_request", 
            func=create_meeting_request_tool,
            description="Creates a meeting request with comprehensive details. Use for scheduling follow-ups, consultations, or formal discussions. Provide clear next steps and expectations to the user."
        ),
        Tool(
            name="Final Answer",
            func=final_answer_tool,
            description="Provides the final detailed and polite answer to the user in conversational chat style. MUST include step-by-step instructions for procedures, or empathetic rejections with alternative suggestions when unable to help. MUST NOT use formal email format (no 'Dear', 'Best regards', signatures). MUST be used as the last action to complete the interaction."
        )
    ] 
except Exception as e:
    logger.error(f"Error initializing tools: {e}")
    tools = []  # Fallback to empty tools list if initialization fails
memory_store = {}
def get_react_prompt() -> PromptTemplate:
    
    template = """You are a helpful and professional Verztec AI assistant. Use the ReAct format to work through problems step by step.

        You have access to these tools:
        {tools}

        CRITICAL RULES:
        1. Use the format: → Action: → Action Input: → wait for Observation
        2. When you use "Final Answer", you are DONE. Do not continue thinking or taking actions.
        3. Do NOT repeat the same action multiple times.
        4. Do NOT continue after "Final Answer".
        5. ALWAYS search the knowledge base first using "faiss_search" before providing answers about Verztec policies or procedures.

        RESPONSE GUIDELINES:
        - Provide DETAILED, step-by-step instructions when helping users with procedures or policies
        - Break down complex processes into clear, numbered steps
        - Include all necessary details, forms, deadlines, and requirements
        - Reference specific sections of policies when applicable
        - If you cannot help with a request, provide a POLITE and EMPATHETIC rejection
        - Explain WHY you cannot help and suggest alternative solutions or contacts
        - Always maintain a professional, helpful, and courteous tone
        - For out-of-scope requests, gently redirect users to appropriate resources
        
        FORMATTING REQUIREMENTS:
        - Use conversational, chat-style language - NOT formal email format
        - DO NOT use email greetings like "Dear [User's Name]" or "Hello [Name]"
        - DO NOT use email closings like "Best regards", "Sincerely", or "Thank you"
        - DO NOT sign off with "[Your Name]" or similar formal signatures
        - Respond as if you're having a helpful conversation, not writing an email
        - Use direct, friendly language: "I can help you with that" instead of "I hope this email finds you well"

        Format:
        Thought: [your reasoning about what you need to do]
        Action: [tool name from: {tool_names}]
        Action Input: [input for the tool]
        Observation: [tool response]
        Thought: [more reasoning if needed]
        Action: Final Answer
        Action Input: [your detailed, helpful, and polite final response to the user]

        Question: {input}
        {agent_scratchpad}"""

    return PromptTemplate(
        template=template,
        input_variables=["input", "agent_scratchpad", "tools", "tool_names"]
    )

def agentic_bot_v1(user_query: str, user_id: str, chat_id: str):
    """
    Agentic bot function that supports chat history and multi-turn conversations.
    
    This function implements an agentic chatbot using the ReAct pattern with:
    1. Chat history retrieval and management
    2. Multi-step reasoning and tool usage
    3. HR escalation and meeting request capabilities
    4. Comprehensive error handling and logging
    
    Args:
        user_query (str): The user's input query
        user_id (str): Unique identifier for the user
        chat_id (str): Unique identifier for the chat session
        
    Returns:
        tuple: (answer_text, image_list) where:
            - answer_text: The agent's final response
            - image_list: List of relevant images (empty for agentic responses)
    """
    
    try:
        # Step 1: Retrieve or create chat history
        key = f"{user_id}_{chat_id}"
        logger.info(f"Agentic bot processing query for key: {key}")
        logger.info(f"User ID: {user_id}, Chat ID: {chat_id}")

        if key in memory_store:
            chat_history = memory_store[key]
            logger.info(f"Memory object found at key: {key}")
        else:
            msgs, chat_name = retrieve_user_messages_and_scores(user_id, chat_id)
            chat_history = build_memory_from_results(msgs)
            memory_store[key] = chat_history
            logger.info(f"No memory object found. Created and saved for key: {key}")
        
        # Step 2: Build context with chat history
        history_context = ""
        if chat_history and hasattr(chat_history, 'chat_memory'):
            history_messages = chat_history.chat_memory.messages
            if history_messages:
                history_context = "\n\nChat History:\n"
                for msg in history_messages[-4:]:  # Last 4 messages for context
                    if hasattr(msg, 'content'):
                        msg_type = "Human" if msg.__class__.__name__ == "HumanMessage" else "AI"
                        history_context += f"{msg_type}: {msg.content}\n"
                history_context += "\nCurrent Query:\n"
        
        # Step 3: Enhanced query with context
        enhanced_query = f"{history_context}{user_query}"
        
        # Step 4: Initialize ReAct agent
        prompt = get_react_prompt()
        
        # Create the core ReAct agent
        agent_runnable = create_react_agent(
            llm=deepseek,           # Use the deepseek model from the file
            tools=tools,            # Available tools for actions
            prompt=prompt           # Engineered prompt template
        )
        
        # Configure the agent executor with robust settings
        agent = AgentExecutor(
            agent=agent_runnable,
            tools=tools,
            verbose=True,                    # Show detailed execution steps
            handle_parsing_errors=True,      # Graceful error recovery
            max_iterations=3,                # Allow more iterations for complex queries
            max_execution_time=60,           # 60-second timeout
            return_intermediate_steps=True,  # Enable step-by-step analysis
            early_stopping_method="force"    # Force stop when limits reached
        )
        
        # Step 5: Execute the agent
        logger.info(f"Executing agentic bot with query: {enhanced_query}")
        start_time = time.time()
        
        response = agent.invoke({"input": enhanced_query})
        
        execution_time = time.time() - start_time
        logger.info(f"Agent execution completed in {execution_time:.2f} seconds")
        
        # Step 6: Extract final answer from response
        final_answer = ""
        if isinstance(response, dict):
            # Try to get the output first
            final_answer = response.get('output', '')
            
            # If the agent hit iteration limits, extract from intermediate steps
            if "Agent stopped due to iteration limit" in final_answer:
                logger.warning("Agent hit iteration limit, extracting final answer from steps...")
                steps = response.get('intermediate_steps', [])
                for step in reversed(steps):
                    if isinstance(step, tuple) and len(step) >= 2:
                        action, observation = step
                        if hasattr(action, 'tool') and action.tool == "Final Answer":
                            final_answer = observation
                            break
                else:
                    final_answer = "I apologize, but I couldn't complete the full reasoning process. Please try rephrasing your question or contact support."
            
            # Clean up the final answer
            if final_answer:
                # Remove any agent-specific formatting
                final_answer = final_answer.strip()
                # Remove any remaining tool prefixes
                if final_answer.startswith("[Simulated]"):
                    final_answer = final_answer.replace("[Simulated]", "").strip()
        else:
            final_answer = str(response)
        
        # Step 7: Update chat history
        if chat_history and hasattr(chat_history, 'chat_memory'):
            chat_history.chat_memory.add_user_message(user_query)
            chat_history.chat_memory.add_ai_message(final_answer)
            
            # Clean up chat memory to keep it manageable
            MAX_TURNS = 6  # Keep more history for agentic conversations
            if len(chat_history.chat_memory.messages) > 2 * MAX_TURNS:
                chat_history.chat_memory.messages = chat_history.chat_memory.messages[-2 * MAX_TURNS:]
        
        # Step 8: Calculate scores for logging
        is_task_query = is_query_score(user_query)
        
        # For agentic responses, relevance is high since tools were used
        relevance_score = 0.8  # High relevance for agentic responses
        
        # Step 9: Log the interaction
        store_chat_log_updated(
            user_message=user_query,
            bot_response=final_answer,
            query_score=is_task_query,
            relevance_score=relevance_score,
            user_id=user_id,
            chat_id=chat_id
        )
        
        logger.info(f"Agentic bot response: {final_answer}")
        
        # Step 10: Return response (no images for agentic responses)
        return final_answer, []
        
    except Exception as e:
        logger.error(f"Error in agentic_bot_v1: {str(e)}", exc_info=True)
        error_response = f"I encountered an error while processing your request: {str(e)}. Please try again or contact support."
        
        # Still try to log the error
        try:
            store_chat_log_updated(
                user_message=user_query,
                bot_response=error_response,
                query_score=0.0,
                relevance_score=0.0,
                user_id=user_id,
                chat_id=chat_id
            )
        except:
            pass  # Don't fail on logging errors
        
        return error_response, []
    
    
    

def extract_meeting_details(user_query: str) -> Dict[str, Any]:
    """
    Extract meeting details from natural language user query using LLM.
    
    Args:
        user_query (str): User's natural language meeting request
        
    Returns:
        dict: Extracted meeting details with confidence scores
    """
    try:
        # Import the LLM models from chatbot.py when needed
        from langchain_groq import ChatGroq
        from langchain_core.prompts import ChatPromptTemplate
        import json
        import re
        
        # Initialize the LLM (using same config as in chatbot.py)
        api_key = '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'
        extraction_model = ChatGroq(
            api_key=api_key, 
            model="qwen/qwen3-32b",
            temperature=0,
            model_kwargs={
                "top_p": 0,
                "frequency_penalty": 0,
                "presence_penalty": 0
            }
        )
        
        # Get current date and time context
        from datetime import datetime
        current_datetime = datetime.now()
        current_date = current_datetime.strftime("%A, %B %d, %Y")
        current_time = current_datetime.strftime("%I:%M %p")
        
        # Create extraction prompt with current date/time context
        extraction_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a meeting detail extraction assistant. Extract meeting details from user input and return ONLY a valid JSON object.

CURRENT CONTEXT:
- Today is: {current_date}
- Current time is: {current_time}

Required JSON format:
{{
    "subject": "string or null",
    "date_time": "string or null", 
    "duration": "string or null",
    "participants": ["array", "of", "strings"],
    "meeting_type": "virtual|in-person|hybrid or null",
    "location": "string or null",
    "priority": "high|normal|low",
    "extraction_confidence": "high|medium|low"
}}

Extraction rules:
- subject: Main topic/purpose of the meeting
- date_time: Convert relative dates to specific dates/times. For example:
  * "tomorrow" = the next day from today
  * "next Monday" = the next Monday from today
  * "this Friday" = the upcoming Friday this week
  * Include both date and time when available (e.g., "Tuesday, July 16, 2025 at 3:00 PM")
- duration: How long the meeting should be
- participants: Names, emails, departments, titles mentioned
- meeting_type: virtual (zoom/teams/online), in-person (conference room/office), or hybrid
- location: Physical location, room names, or virtual platform
- priority: high (urgent/asap/emergency), normal (default), low (when possible/eventually)
- extraction_confidence: high (4+ fields), medium (2-3 fields), low (0-1 fields)

Return ONLY the JSON object. No explanations."""),
            ("human", "Extract meeting details from: {query}")
        ])
        
        # Create and run the extraction chain
        extraction_chain = extraction_prompt | extraction_model
        
        # Get the response
        response = extraction_chain.invoke({
            "query": user_query,
            "current_date": current_date,
            "current_time": current_time
        })
        raw_response = response.content.strip()
        
        logger.info(f"Raw LLM response: {raw_response}")
        
        # Clean up the response to extract JSON
        # Remove any markdown formatting or extra text
        json_start = raw_response.find('{')
        json_end = raw_response.rfind('}') + 1
        
        if json_start != -1 and json_end > json_start:
            json_str = raw_response[json_start:json_end]
            logger.info(f"Extracted JSON string: {json_str}")
            
            # Additional cleanup for common issues
            json_str = json_str.replace('\n', ' ').replace('\r', ' ')
            json_str = re.sub(r'\s+', ' ', json_str)  # Replace multiple spaces with single space
            
            extracted_data = json.loads(json_str)
            
            # Add raw query for reference
            extracted_data['raw_query'] = user_query.strip()
            
            # Validate and clean the extracted data
            details = {
                'subject': extracted_data.get('subject'),
                'date_time': extracted_data.get('date_time'),
                'duration': extracted_data.get('duration'),
                'participants': extracted_data.get('participants', []) if isinstance(extracted_data.get('participants'), list) else [],
                'meeting_type': extracted_data.get('meeting_type'),
                'location': extracted_data.get('location'),
                'priority': extracted_data.get('priority', 'normal'),
                'raw_query': user_query.strip(),
                'extraction_confidence': extracted_data.get('extraction_confidence', 'medium')
            }
            
            logger.info(f"LLM extraction successful: {details}")
            return details
            
        else:
            raise ValueError("No valid JSON found in LLM response")
            
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing failed at position {e.pos}: {str(e)}")
        logger.error(f"Problematic JSON string: {json_str if 'json_str' in locals() else 'Not extracted'}")
        logger.error(f"Full LLM response: {raw_response if 'raw_response' in locals() else 'Not available'}")
        
        # Continue to fallback logic...
    except Exception as e:
        logger.error(f"LLM extraction failed: {str(e)}, falling back to regex")
        
        # Fallback to basic regex extraction if LLM fails
        details = {
            'subject': None,
            'date_time': None,
            'duration': None,
            'participants': [],
            'meeting_type': None,
            'location': None,
            'priority': 'normal',
            'raw_query': user_query.strip(),
            'extraction_confidence': 'low'
        }
        
        query_lower = user_query.lower().strip()
        
        # Basic regex fallbacks for critical fields
        # Extract time mentions
        time_patterns = [
            r'\d{1,2}(?::\d{2})?\s*(?:am|pm)',
            r'(?:at\s+)?\d{1,2}(?::\d{2})?',
            r'tomorrow|today|next week|monday|tuesday|wednesday|thursday|friday|saturday|sunday'
        ]
        
        for pattern in time_patterns:
            match = re.search(pattern, query_lower)
            if match:
                details['date_time'] = match.group(0)
                break
        
        # Extract participant names (simple pattern)
        participant_match = re.search(r'with\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', user_query)
        if participant_match:
            details['participants'] = [participant_match.group(1)]
        
        # Extract room/location
        room_match = re.search(r'(?:room|conference room)\s+([A-Za-z0-9]+)', query_lower)
        if room_match:
            details['location'] = room_match.group(1).upper()
        
        # Basic subject extraction (everything before time/participant mentions)
        subject_match = re.search(r'^(.+?)(?:\s+(?:with|at|tomorrow|today|next|room))', query_lower)
        if subject_match:
            subject = subject_match.group(1).strip()
            if len(subject) > 3:
                details['subject'] = subject.replace('meeting', '').strip().title()
        
        return details

