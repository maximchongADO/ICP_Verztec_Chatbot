import re
from rapidfuzz import fuzz
from collections import defaultdict, Counter
import re
import string
import pymysql
from typing import List, Optional
import logging

# Set up logging
logger = logging.getLogger(__name__)

DB_CONFIG = {
    'host': 'localhost',
    'user': 'chatbot_user',
    'password': 'strong_password',
    'database': 'chatbot_db',
    'cursorclass': pymysql.cursors.DictCursor,
    'autocommit': True
}

def get_user_info(user_id: str):
    """Get user information from the users table"""
    conn = pymysql.connect(**DB_CONFIG)
    cursor = conn.cursor()
    try:
        # Fetch user info from the users table
        cursor.execute("SELECT username, role, country, department FROM users WHERE id = %s", (user_id,))
        result = cursor.fetchone()
        
        if result:
            user_info = {
                "username": result['username'],
                "role": result['role'],
                "country": result['country'],
                "department": result['department']
            }
            return user_info
        else:
            logger.warning(f"No user found with ID: {user_id}")
            return None
    except pymysql.Error as e:
        logger.error(f"Error fetching user info for {user_id}: {str(e)}")
        return None
    finally:
        cursor.close()
        conn.close()

## some stuff to generate suggestionsm i might just use the same api pathway cos idk how ot redo ir HHHAHA
def retrieve_user_messages_and_scores(user_id: Optional[str] = None):
    """
    Retrieve user messages and scores with optional regional filtering.
    
    Args:
        user_id: If provided, filters based on user's role, country, and department
                - Admin users: see all queries
                - Regional users: see only their region's queries  
                - Department users: see only their department's queries
    """
    conn = pymysql.connect(**DB_CONFIG)
    cursor = conn.cursor()
    
    try:
        # If no user_id provided, return all results (backward compatibility)
        if not user_id:
            select_query = '''
                SELECT user_message, query_score, relevance_score
                FROM chat_logs
                WHERE user_message IS NOT NULL AND user_message != ''
                ORDER BY timestamp ASC
            '''
            cursor.execute(select_query)
            results = cursor.fetchall()
            logger.info(f"Retrieved {len(results)} messages for all users")
            return results
        
        # Get user information to determine filtering
        user_info = get_user_info(user_id)
        if not user_info:
            logger.warning(f"No user info found for {user_id}, using global results")
            select_query = '''
                SELECT user_message, query_score, relevance_score
                FROM chat_logs
                WHERE user_message IS NOT NULL AND user_message != ''
                ORDER BY timestamp ASC
            '''
            cursor.execute(select_query)
            results = cursor.fetchall()
            return results
        
        role = user_info.get('role', '').lower() if user_info.get('role') else ''
        country = user_info.get('country', '').lower() if user_info.get('country') else ''
        department = user_info.get('department', '').lower() if user_info.get('department') else ''
        
        logger.info(f"Filtering frequent queries for user {user_id}: role={role}, country={country}, department={department}")
        
        # Admin users see everything
        if role in ['admin', 'administrator', 'super_admin', 'superadmin']:
            select_query = '''
                SELECT cl.user_message, cl.query_score, cl.relevance_score
                FROM chat_logs cl
                WHERE cl.user_message IS NOT NULL AND cl.user_message != ''
                ORDER BY cl.timestamp ASC
            '''
            cursor.execute(select_query)
            results = cursor.fetchall()
            logger.info(f"Admin user {user_id}: Retrieved {len(results)} messages from all regions")
            return results
        
        # Regional filtering for non-admin users
        # Map country and department variations to standard names
        country_mapping = {
            'singapore': 'singapore',
            'sg': 'singapore', 
            'china': 'china',
            'cn': 'china',
            'prc': 'china'
        }
        
        dept_mapping = {
            'hr': 'hr',
            'human resources': 'hr',
            'human_resources': 'hr',
            'it': 'it', 
            'information technology': 'it',
            'information_technology': 'it',
            'tech': 'it',
            'technology': 'it'
        }
        
        mapped_country = country_mapping.get(country, country)
        mapped_dept = dept_mapping.get(department, department)
        
        # Build query with regional filtering
        if mapped_country and mapped_dept:
            # Filter by exact country and department match
            select_query = '''
                SELECT cl.user_message, cl.query_score, cl.relevance_score
                FROM chat_logs cl
                JOIN users u ON cl.user_id = u.id
                WHERE cl.user_message IS NOT NULL AND cl.user_message != ''
                AND LOWER(u.country) = %s AND LOWER(u.department) = %s
                ORDER BY cl.timestamp ASC
            '''
            cursor.execute(select_query, (mapped_country, mapped_dept))
            results = cursor.fetchall()
            logger.info(f"Regional user {user_id}: Retrieved {len(results)} messages for {mapped_country}/{mapped_dept}")
            
        elif mapped_country:
            # Filter by country only
            select_query = '''
                SELECT cl.user_message, cl.query_score, cl.relevance_score
                FROM chat_logs cl
                JOIN users u ON cl.user_id = u.id
                WHERE cl.user_message IS NOT NULL AND cl.user_message != ''
                AND LOWER(u.country) = %s
                ORDER BY cl.timestamp ASC
            '''
            cursor.execute(select_query, (mapped_country,))
            results = cursor.fetchall()
            logger.info(f"Country user {user_id}: Retrieved {len(results)} messages for {mapped_country}")
            
        else:
            # Fallback to all results if no clear regional mapping
            select_query = '''
                SELECT cl.user_message, cl.query_score, cl.relevance_score
                FROM chat_logs cl
                WHERE cl.user_message IS NOT NULL AND cl.user_message != ''
                ORDER BY cl.timestamp ASC
            '''
            cursor.execute(select_query)
            results = cursor.fetchall()
            logger.info(f"User {user_id}: No clear regional mapping, retrieved {len(results)} messages from all regions")
        
        return results
        
    except Exception as e:
        logger.error(f"Error retrieving user messages for {user_id}: {str(e)}")
        # Fallback to basic query on error
        select_query = '''
            SELECT user_message, query_score, relevance_score
            FROM chat_logs
            WHERE user_message IS NOT NULL AND user_message != ''
            ORDER BY timestamp ASC
        '''
        cursor.execute(select_query)
        results = cursor.fetchall()
        return results
        
    finally:
        cursor.close()
        conn.close()


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


def get_suggestions(user_id: Optional[str] = None, query: str = "") -> List[str]:
    """
    Returns a list of the most frequent, semantically distinct user queries.
    Groups similar queries (e.g., 'pantry rules?' vs 'what are the pantry rules?') together
    and returns the most representative (longest) canonical form for each group.
    
    Args:
        user_id: If provided, filters suggestions based on user's regional access
        query: Unused parameter for backward compatibility
    """
    all_results = retrieve_user_messages_and_scores(user_id)

    # Filter based on your criteria
    filtered_results = [
        result for result in all_results
        if result['user_message'] and result['query_score'] > 0.8 and result['relevance_score'] < 0.7
    ]

    logger.info(f"Filtered {len(filtered_results)} relevant queries from {len(all_results)} total for user {user_id}")

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

    logger.info(f"Generated {len(top_queries)} suggestions for user {user_id}: {top_queries}")
    return top_queries
