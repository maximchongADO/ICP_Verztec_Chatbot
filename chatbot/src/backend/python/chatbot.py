from typing import List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage
import re
from langchain.vectorstores.base import VectorStoreRetriever
from langchain.schema import Document
#from numpy import dot
#from numpy.linalg import norm
import os
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_core.output_parsers import StrOutputParser
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import logging
#from better_profanity import profanity
import spacy
from spacy.matcher import PhraseMatcher


from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize models and clients
embedding_model = SentenceTransformer('BAAI/bge-large-en-v1.5')
api_key = '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'
model_name = "compound-beta"
model = "deepseek-r1-distill-llama-70b" 
deepseek = ChatGroq(api_key=api_key, model=model) # type: ignore
compound = ChatGroq(api_key=api_key, model=model_name) # type: ignore

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
    logger.info("FAISS index loaded successfully on CPU")
    
except Exception as e:
    logger.error(f"Failed to load FAISS index: {str(e)}", exc_info=True)
    index = None
    metadata = None

try:
    # Load spaCy English model
    nlp = spacy.load("en_core_web_sm")
    casual_phrases = ["hi", "hello", "thanks", "thank you", "good morning", "goodbye", "hey", "yo", "sup"]
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    patterns = [nlp.make_doc(text) for text in casual_phrases]
    tokenizer = AutoTokenizer.from_pretrained("prithivida/grammar_error_correcter_v1")
    model = AutoModelForSeq2SeqLM.from_pretrained("prithivida/grammar_error_correcter_v1")
    matcher.add("CASUAL", patterns)
    logger.info("SpaCy model and matcher initialized successfully")
    
except Exception as e:
    logger.error(f"Failed to load spacymodel:  {str(e)}", exc_info=True)
    

def is_casual_message(text: str) -> bool:
    doc = nlp(text)
    matches = matcher(doc)
    return bool(matches)



def is_query(text: str) -> bool:
    try:
        doc = nlp(text)
        # Check for known casual question patterns
        casual_phrases = {"how are you", "what's up", "how's it going", "are you there"}
        normalized = text.lower().strip("!?.,")
        if normalized in casual_phrases:
            return False

        # Check for bot refusal or rubbish responses
        refusal_patterns = [
            "i'm sorry", "i cannot", "i can't", "as an ai", "i am unable", "not able to", 
            "i do not understand", "i don't know", "no comment", "n/a", "not applicable",
            "this is not a question", "please rephrase", "unclear", "meaningless"
        ]
        if any(pattern in normalized for pattern in refusal_patterns):
            return False

        # Check for question words or action verbs
        for token in doc:
            if token.dep_ in ("aux", "ROOT") and token.tag_ in ("VB", "VBP", "VBZ", "MD"):
                if token.lemma_ in {"do", "can", "could", "help", "need", "upload", "reset", "access"}:
                    return True
            if token.dep_ == "nsubj" and token.lemma_ in {"problem", "issue"}:
                return True
            if token.lower_ in {"how", "what", "when", "where", "why", "who"}:
                return True

        # If it looks like a statement or rubbish, not a query
        return False
    except Exception as e:
        logger.error(f"is_query exception: {e} for text: {text!r}")
        return False


def is_query(text: str) -> bool:
    try:
        # Normalize text
        normalized = text.lower().strip()
        normalized = re.sub(r'[^\w\s]', '', normalized)  # remove punctuation

        # 1. Refusal or non-query phrases (LLM-like outputs or conversational redirections)
        refusal_phrases = [
            "i won't engage", "i will not engage", "not a question", "this isn't a question",
            "is there something else", "can i help you with something else", "i'm unable to answer",
            "i cannot help with that", "i'm sorry", "as an ai", "i do not understand",
            "this is unclear", "this seems incomplete", "please clarify", "please rephrase",
            "i don't have an answer", "i cannot provide", "that's outside my scope",
            "no comment", "n/a", "not applicable", "i'm just an ai", "not relevant"
        ]
        if any(phrase in normalized for phrase in refusal_phrases):
            return False

        # 2. Casual conversational small talk — not queries
        casual_talk = [
            "how are you", "what's up", "good morning", "hello", "hi", "are you there", "thanks", "okay"
        ]
        if normalized in casual_talk:
            return False

        # 3. If it ends with a question mark, it’s *very likely* a question
        if text.strip().endswith("?"):
            return True

        # 4. Use spaCy parsing to find question intent
        doc = nlp(text)

        # Question starters like: what, how, why, where...
        question_words = {"what", "why", "who", "where", "when", "how", "which", "whom"}
        if any(token.lower_ in question_words for token in doc):
            return True

        # Imperative: "Tell me", "Explain", "Show me", etc.
        imperative_verbs = {"tell", "explain", "show", "list", "describe", "give", "find", "fetch"}
        if any(token.lemma_ in imperative_verbs and token.pos_ == "VERB" for token in doc):
            return True

        # Auxiliary verbs indicating a query
        query_aux_verbs = {"can", "could", "would", "do", "does", "did", "is", "are", "will", "should"}
        if any(token.lemma_ in query_aux_verbs and token.tag_ in {"MD", "VB", "VBP", "VBZ"} for token in doc):
            return True

        # Heuristic fallback: is it short and not informative?
        if len(normalized.split()) <= 2:
            return False

        return False

    except Exception as e:
        print(f"[Error in is_query]: {e}")
        return False
def is_query_score(text: str) -> float:
    try:
        score = 0.0
        normalized = text.lower().strip()
        normalized = re.sub(r'[^\w\s]', '', normalized)

        # 1. Refusal or AI-like responses (strong negative)
        refusal_phrases = [
            "i won't engage", "i will not engage", "not a question", "this isn't a question",
            "is there something else", "can i help you with something else", "i'm unable to answer",
            "i cannot help with that", "i'm sorry", "as an ai", "i do not understand",
            "this is unclear", "this seems incomplete", "please clarify", "please rephrase",
            "i don't have an answer", "i cannot provide", "that's outside my scope",
            "no comment", "n/a", "not applicable", "i'm just an ai", "not relevant"
        ]
        if any(phrase in normalized for phrase in refusal_phrases):
            return 0.0  # Immediate disqualifier

        # 2. Casual phrases (low query likelihood)
        casual_talk = [
            "how are you", "what's up", "good morning", "hello", "hi", "are you there", "thanks", "okay"
        ]
        if normalized in casual_talk:
            score += 0.1

        # 3. Ends with a question mark (strong signal)
        if text.strip().endswith("?"):
            score += 0.4

        # 4. Use spaCy to analyze structure
        doc = nlp(text)

        question_words = {"what", "why", "who", "where", "when", "how", "which", "whom"}
        if any(token.lower_ in question_words for token in doc):
            score += 0.3

        imperative_verbs = {"tell", "explain", "show", "list", "describe", "give", "find", "fetch"}
        if any(token.lemma_ in imperative_verbs and token.pos_ == "VERB" for token in doc):
            score += 0.2

        query_aux_verbs = {"can", "could", "would", "do", "does", "did", "is", "are", "will", "should"}
        if any(token.lemma_ in query_aux_verbs and token.tag_ in {"MD", "VB", "VBP", "VBZ"} for token in doc):
            score += 0.2

        # Slight penalty if it's too short
        if len(normalized.split()) <= 2:
            score -= 0.2

        # Clamp between 0 and 1
        return max(0.0, min(score, 1.0))

    except Exception as e:
        print(f"[Error in is_query_score]: {e}")
        return 0.0




def clean_with_grammar_model(user_query: str) -> str:
    input_text = f"gec: {user_query}"
    inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=128, truncation=True)
    outputs = model.generate(inputs, max_length=128, num_beams=5, early_stopping=True)
    corrected = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return corrected

## clean query and ensure it is suitable for processing
def refine_prompt(user_query: str) -> str:
    #is_profane_query = is_profane(user_query)
    #logger.info(f"Is Profane: {is_profane_query}")
    logger.info(f"User Query: {user_query}")
    is_casual = is_query(user_query)
    logger.info(f"Is Question: {is_casual}")
    prompt2 = (
        f"ONLY RETURN THE CLEANED QUERY. You are an assistant that improves grammar and spelling for an internal helpdesk chatbot. "
        f"If the input is offensive, unclear, or meaningless, please clean it.\n\n"
        f"Input:\n{user_query}"
    )
    messages = [HumanMessage(content=prompt2)]
    #response = compound.generate([messages]) # type: ignore
    
    #refined_query = response.generations[0][0].text.strip()
    refined_query = clean_with_grammar_model(user_query)
    logger.info(f"User Query cleaned: {refined_query}")

    # Safety net
    
    isquery = is_query(refined_query)
    logger.warning(f'New query is a question: {isquery}')
    if not isquery:
        refined_query = user_query
        logger.warning(f"Refined query was not a valid question, reverting to original: {refined_query}")
    
    

   
    return refined_query

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


AVG_SCORE_THRESHOLD = 0.967


def generate_answer(user_query: str, chat_history: ConversationBufferMemory) -> str:
    try:
        parser = StrOutputParser()
        

        # Chain the Groq model with the parser
        deepseek_chain = deepseek | parser
        
        retriever = index.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 10}
        )
        avg_score = get_avg_score(index, embedding_model, user_query)
        
      
       
        ## FAISS retriever setup
        
        ## QA chain setup with mrmory 
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=deepseek_chain,
            retriever=retriever,
            memory=chat_history,
            return_source_documents=True,
            output_key="answer"
        )
        logger.info(memory)
        
      

        # Step 1: Check relation to past queries
        #is_related = is_related_to_previous(user_query, chat_history)
        #context_query = " ".join(chat_history + [user_query]) if is_related else user_query

        # Step 2: Refine query
        clean_query = refine_prompt(user_query)
        #clean_query=user_query

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


       
        is_task_query = is_query(user_query)
        
        logger.info(f"Clean Query at qa chain: {clean_query}")
        if is_task_query<0.4 and avg_score >= AVG_SCORE_THRESHOLD:
            logger.info("Bypassing QA chain for non-query with weak retrieval.")
            fallback_prompt = f"The user said: \"{clean_query}\". Respond appropriately as a polite VERZTEC assustant, without"
            messages = [HumanMessage(content=fallback_prompt)]
            response = deepseek.generate([messages])
            raw_fallback = response.generations[0][0].text.strip()

            # Remove <think> block if present
            think_block_pattern = re.compile(r"<think>.*?</think>", flags=re.DOTALL)
            cleaned_fallback = think_block_pattern.sub("", raw_fallback).strip()
            return cleaned_fallback
        logger.info("QA chain activated for query processing.")
        # Step 4: Prepare full prompt and return LLM output
        modified_query = "You are a verztec helpdesk assistant. You will only use the provided documents in your response. If the query is out of scope, say so.\n\n" + clean_query
        response = qa_chain.invoke({"question": modified_query})
        #response = qa_chain.invoke({"question": clean_query})
        raw_answer = response['answer']
        logger.info(f"Full response before cleanup: {raw_answer}")

        # Define regex pattern to match full <think>...</think> block
        think_block_pattern = re.compile(r"<think>.*?</think>", flags=re.DOTALL)

        # Check if full <think> block exists
        has_think_block = bool(think_block_pattern.search(raw_answer))

        # Clean the <think> block regardless
        cleaned_answer = think_block_pattern.sub("", raw_answer).strip()

        if not has_think_block:
            logger.warning("Missing full <think> block — retrying query once...")
            response_retry = qa_chain.invoke({"question": clean_query})
            raw_answer_retry = response_retry['answer']
            logger.info(f"Retry response: {raw_answer_retry}")
            cleaned_answer = think_block_pattern.sub("", raw_answer_retry).strip()
        else:
            logger.info("Full <think> block found and removed successfully")
        block_tag_pattern = re.compile(r"<([a-zA-Z0-9_]+)>.*?</\1>", flags=re.DOTALL)

        # Check if there are any block tags at all
        has_any_block_tag = bool(block_tag_pattern.search(raw_answer))
        if has_any_block_tag:
            logger.info("Block tags found in response, cleaning them up")

        # Remove all block tags
        cleaned_answer = block_tag_pattern.sub("", raw_answer).strip()
        
        return cleaned_answer
    except Exception as e:
        return f"I encountered an error while processing your request: {str(e)}"
