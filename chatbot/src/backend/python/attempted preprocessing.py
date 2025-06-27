from transformers import LogitsProcessor
import torch

from typing import List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from typing import List, Dict
from math import isclose

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
model_name = "pszemraj/flan-t5-large-grammar-synthesis"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
model.eval()    


MAX_BOOST = 7.0   

class BoostWordProcessor(LogitsProcessor):
    def __init__(self, tokenizer, boost_words, boost_value=5.0):
        self.tokenizer = tokenizer
        self.boost_token_ids = set()
        for w in boost_words:
            self.boost_token_ids.update(tokenizer.encode(w, add_special_tokens=False))
        self.boost_value = boost_value

    def __call__(self, input_ids, scores):
        for tok_id in self.boost_token_ids:
            if 0 <= tok_id < scores.size(-1):
                scores[:, tok_id] += self.boost_value
        return scores
    
    
class WeightedBoostProcessor(LogitsProcessor):
    """
    Boost selected tokens by *individual* amounts.

    Parameters
    ----------
    tokenizer : PreTrainedTokenizer
    boost_map : dict[str, float]
        word/phrase → boost value (already scaled 0‒max_boost)
    """

    def __init__(self, tokenizer, boost_map: Dict[str, float]):
        self.tokenizer = tokenizer
        # map token-id → boost
        self.token_boost: Dict[int, float] = {}

        for word, boost in boost_map.items():
            # encode without special tokens so “meeting room” → [tokA, tokB]
            for tok_id in tokenizer.encode(word, add_special_tokens=False):
                # keep the *largest* boost if the token shows up twice
                self.token_boost[tok_id] = max(self.token_boost.get(tok_id, 0.0),
                                               boost)

    def __call__(self, input_ids, scores):
        # scores shape: (batch, vocab)
        for tok_id, boost in self.token_boost.items():
            if 0 <= tok_id < scores.size(-1):
                scores[:, tok_id] += boost
        return scores

 


from typing import List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import re


def boosted_weight(wt, token, filename_words,
                   base_boost=2.5, acronym_boost=5.0):
    """
    Apply stronger boost to tokens from filename if they:
    - appear in filename_words
    - AND are likely acronyms or technical terms:
        - all uppercase (e.g. "SOP")
        - contain digits (e.g. "11a")
        - OR are short (3–5 chars) and uncommon-looking (e.g. "abss", "xtranet")
    """
    token_lc = token.lower()

    if token_lc in filename_words:
        is_all_caps = token.isupper()
        has_digits = any(c.isdigit() for c in token_lc)
        is_short_technical = 3 <= len(token_lc) <= 5 and token_lc.isalpha() and not token_lc in {"this", "that", "from", "with", "file"}

        if is_all_caps or has_digits or is_short_technical:
            return wt * acronym_boost
        return wt * base_boost

    return wt



def extract_keywords(
    docs: List[Dict[str, str]],
    top_n: int = 5,
    ngram_range: tuple = (1, 2),
    stop_words: str = "english",
    filename_boost: float = 2.5
) -> List[Dict[str, float]]:
    """
    Extract top_n keywords from each document with TF-IDF weights,
    and boost terms that appear in the document's filename.

    Parameters
    ----------
    docs : list of dict
        Each dict should have 'text_content' and 'doc_filename'.
    filename_boost : float
        Multiplier to apply to keywords found in filename.

    Returns
    -------
    results : list of dict
        Each element maps keyword → boosted weight for a single doc.
    """
    texts = [d["text_content"] for d in docs]
    filenames = [d["doc_filename"] for d in docs]

    # 1) Fit TF-IDF on all text contents
    vectorizer = TfidfVectorizer(
        ngram_range=ngram_range,
        stop_words=stop_words,
        norm="l2"
    )
    tfidf_matrix = vectorizer.fit_transform(texts)
    feature_names = np.array(vectorizer.get_feature_names_out())

    results: List[Dict[str, float]] = []

    # 2) For each document
    for i, row in enumerate(tfidf_matrix):
        if row.nnz == 0:
            results.append({})
            continue

        scores = row.data
        indices = row.indices

        # Pick top-N
        top_idx = scores.argsort()[-top_n:][::-1]
        keywords = feature_names[indices[top_idx]]
        weights  = scores[top_idx]
        ## extrac
        filename = filenames[i].rsplit('.', 1)[0].lower()

        # split on underscores, dashes, spaces, etc.
        parts = re.split(r'[_\W]+', filename)

        # keep tokens that are (a) not pure digits and (b) at least 4 chars long
        filename_words = [
            word
            for word in parts
            if word and not word.isdigit() and len(word) >= 3
        ]

        print(filename_words)
        # Build keyword-weight dict, apply filename boost if matched
        kw_map = {}
        for kw, wt in zip(keywords, weights):
            wt = boosted_weight(wt, kw, filename_words,
                                base_boost=3, acronym_boost=7.0)
            kw_map[kw] = round(wt, 4)
            
        for word in filename_words:
            if word not in kw_map:
                wt = boosted_weight(1.0, word, filename_words,
                                    base_boost=3.0, acronym_boost=7.0)
                kw_map[word] = round(wt, 4)
            else:
            # bump up the weight if already present
                bumped = boosted_weight(kw_map[word], word, filename_words,
                                        base_boost=3.0, acronym_boost=7.0)
                kw_map[word] = round(max(kw_map[word], bumped), 3)
                

        results.append(kw_map)
        
    print(results)
    return results

import mysql.connector
DB_CONFIG = {
    'host': 'localhost',
    'user': 'chatbot_user',
    'password': 'strong_password',
    'database': 'chatbot_db',
    'raise_on_warnings': True
}

def retrieve_user_messages_and_scores():
    """Fetch all text_content values from extracted_texts and return them as a list of strings."""
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor(dictionary=True)

    query = """
        SELECT text_content, doc_filename
        FROM extracted_texts;
    """
    cursor.execute(query)
    rows = cursor.fetchall()          # rows → list[dict]

    cursor.close()
    conn.close()
    return rows




def build_boost_map(
    tfidf_dicts: List[Dict[str, float]],
    max_boost: float = 7.0,
    min_boost: float = 3.0,
    method: str = "sqrt"  # options: "sqrt", "log", or "linear"
) -> Dict[str, float]:
    """
    Flatten list of TF-IDF dictionaries, take max weight per word,
    and rescale to [min_boost, max_boost] using non-linear scaling.

    Parameters
    ----------
    tfidf_dicts : list of dict
        Each dict maps keyword → raw TF-IDF weight.
    max_boost : float
        Desired upper bound for boosted logits.
    min_boost : float
        Lower bound to keep even weak terms slightly boosted.
    method : str
        One of: "sqrt", "log", "linear"

    Returns
    -------
    dict[str, float] : word → scaled boost
    """
    # Step 1 — Merge all TF-IDF dicts, keep max value per word
    weight_map: Dict[str, float] = {}
    for d in tfidf_dicts:
        for k, w in d.items():
            weight_map[k] = max(weight_map.get(k, 0.0), w)

    if not weight_map:
        return {}

    values = np.array(list(weight_map.values()))
    
    # Step 2 — Apply scaling
    if method == "sqrt":
        scaled = np.sqrt(values)
    elif method == "log":
        scaled = np.log1p(values)
    elif method == "linear":
        scaled = values
    else:
        raise ValueError("method must be 'sqrt', 'log', or 'linear'")

    # Step 3 — Normalize to [min_boost, max_boost]
    scaled = (scaled - scaled.min()) / (scaled.max() - scaled.min() + 1e-8)
    scaled = scaled * (max_boost - min_boost) + min_boost

    # Step 4 — Map back to keywords
    return {k: float(v) for k, v in zip(weight_map.keys(), scaled)}


def clean_with_grammar_model(user_query: str) -> str:
    input_text = f"gec: {user_query}"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=128)

    # Define your bias words
    boost_words = ["pantry"]
    boost_processor = BoostWordProcessor(tokenizer, boost_words, boost_value=10.0)

    outputs = model.generate(
        **inputs,
        max_new_tokens=64,
        num_beams=5,
        early_stopping=True,
        logits_processor=[boost_processor]
    )

    corrected = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return corrected


def clean_with_grammar_model(user_query: str,
                             boost_map: Dict[str, float],
                             max_boost: float = MAX_BOOST) -> str:

    input_text = f"gec: {user_query}"
    inputs = tokenizer(input_text, return_tensors="pt",
                       truncation=True, max_length=128)

    
    logits_proc = WeightedBoostProcessor(tokenizer, boost_map)
    

    outputs = model.generate(
        **inputs,
        max_new_tokens=64,
        num_beams=5,
        early_stopping=True,
        logits_processor=[logits_proc]   # single object, very fast
    )

    corrected = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if ':' in corrected:
        corrected = corrected.split(':', 1)[1].strip()
    return corrected




res = retrieve_user_messages_and_scores()
words =extract_keywords(
    docs= res
) 

boost_map = build_boost_map(words, MAX_BOOST)
print("---------------------------------------------------------------------")
print(boost_map)

origin = 'how many days of lave do i have ah'
result = clean_with_grammar_model(origin, boost_map)
print("Original: "+origin)
print("Cleaned: "+result)

origin = 'what are the pantey rules'
result = clean_with_grammar_model(origin, boost_map)
print("Original: "+origin)
print("Cleaned: "+result)


origin = 'what are the panty rules'
result = clean_with_grammar_model(origin, boost_map)
print("Original: "+origin)
print("Cleaned: "+result)

origin = 'how do i upload files to the abss system'
result = clean_with_grammar_model(origin, boost_map)

print("Original: "+origin)
print("Cleaned: "+result)