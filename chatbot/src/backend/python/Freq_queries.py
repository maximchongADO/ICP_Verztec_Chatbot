import re
from rapidfuzz import fuzz
from collections import defaultdict, Counter
import re
import string
import pymysql
from typing import List

DB_CONFIG = {
    'host': 'localhost',
    'user': 'chatbot_user',
    'password': 'strong_password',
    'database': 'chatbot_db',
    'cursorclass': pymysql.cursors.DictCursor,
    'autocommit': True
}
## some stuff to generate suggestionsm i might just use the same api pathway cos idk how ot redo ir HHHAHA
def retrieve_user_messages_and_scores():
    conn = pymysql.connect(**DB_CONFIG)
    cursor = conn.cursor()

    select_query = '''
        SELECT user_message, query_score, relevance_score
        FROM chat_logs
        ORDER BY timestamp ASC
    '''
    cursor.execute(select_query)
    results = cursor.fetchall()

    cursor.close()
    conn.close()

    return results


def normalize_text(text: str) -> str:
    # Lowercase, remove punctuation, strip spaces
    return re.sub(r'[^\w\s]', '', text.lower().strip())


def format_query(text: str) -> str:
    """Cleans and formats user queries with proper spacing, capitalization, and punctuation."""
    if not text or not isinstance(text, str):
        return ""

    # Step 1: Normalize whitespace
    text = re.sub(r'\s+', ' ', text.strip())

    # Step 2: Remove stray punctuation from the start (e.g., "?? help?")
    text = re.sub(r'^[^\w]+', '', text)

    # Step 3: Capitalize first word only, leave rest as-is
    if len(text) > 1:
        text = text[0].upper() + text[1:]

    # Step 4: Add a question mark if it doesn't end in punctuation
    if not text.endswith(('.', '?', '!')):
        text += '?'

    # Optional: replace multiple punctuation at the end (e.g., "What is this??") with a single ?
    text = re.sub(r'[.?!]{2,}$', '?', text)

    return text


def get_suggestions(query: str = "") -> List[str]:
    """
    Returns a list of the most frequent, semantically distinct user queries.
    Groups similar queries (e.g., 'pantry rules?' vs 'what are the pantry rules?') together
    and returns the most representative (longest) canonical form for each group.
    """
    all_results = retrieve_user_messages_and_scores()

    # Filter based on your criteria
    filtered_results = [
        result for result in all_results
        if result['user_message'] and result['query_score'] > 0.79 and result['relevance_score'] < 1.01
    ]

    # Group by normalized form to avoid semantic overlaps
    query_map = defaultdict(list)  # {normalized: [originals]}
    for result in filtered_results:
        message = result['user_message']
        norm = normalize_text(message)
        matched = False
        for canon_norm in query_map:
            # Use fuzzy match to group semantically similar queries
            if fuzz.ratio(norm, canon_norm) > 85:
                query_map[canon_norm].append(message)
                matched = True
                break
        if not matched:
            query_map[norm].append(message)

    # For each group, pick the most representative (longest) query
    canonical_queries = []
    for variants in query_map.values():
        # Pick the variant with the most words, or the longest
        best = max(variants, key=lambda x: (len(x.split()), len(x)))
        canonical_queries.append(best)

    # Count occurrences by group
    grouped_counts = [(q, sum(q in v for v in query_map.values())) for q in canonical_queries]
    top_groups = sorted(grouped_counts, key=lambda x: x[1], reverse=True)[:4]

    # Format and deduplicate
    top_queries = []
    seen = set()
    for item in top_groups:
        formatted = format_query(item[0])
        if formatted not in seen:
            seen.add(formatted)
            top_queries.append(formatted)

    return top_queries
