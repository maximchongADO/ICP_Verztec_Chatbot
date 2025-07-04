

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
    all_results = retrieve_user_messages_and_scores()

    # Filter based on your criteria
    filtered_results = [
        result for result in all_results
        if result['user_message'] and result['query_score'] > 0.79 and result['relevance_score'] < 1.01
    ]

    # Fuzzy group similar queries
    grouped_queries = []
    query_map = defaultdict(list)  # {canonical_query: [variants]}

    for result in filtered_results:
        message = result['user_message']
        matched = False

        # Try to match against existing groups
        for canon in query_map:
            if fuzz.ratio(message.lower(), canon.lower()) > 85:  # similarity threshold
                query_map[canon].append(message)
                matched = True
                break

        # If no match, start new group
        if not matched:
            query_map[message] = [message]

    # Count occurrences by group
    grouped_counts = [(canon, len(variants)) for canon, variants in query_map.items()]
    top_3_groups = sorted(grouped_counts, key=lambda x: x[1], reverse=True)[:4]

    # Return the canonical representative from each top group
    top_3_queries = [item[0] for item in top_3_groups]
    for i in range(len(top_3_queries)):
        top_3_queries[i] = format_query(normalize_text(top_3_queries[i]))

    return top_3_queries
