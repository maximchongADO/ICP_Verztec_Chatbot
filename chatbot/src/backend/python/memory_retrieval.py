from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage
import re
from datetime import datetime
import pymysql


DB_CONFIG = {
    'host': 'localhost',
    'user': 'chatbot_user',
    'password': 'strong_password',
    'database': 'chatbot_db',
    'cursorclass': pymysql.cursors.DictCursor,
    'autocommit': True
}
## SQL DB needs to be updated to include user and chatid 
def retrieve_user_messages_and_scores(User_id, chat_id):
    conn = pymysql.connect(**DB_CONFIG)
    try:
        with conn.cursor() as cursor:
            select_query = '''
                SELECT timestamp, user_message, bot_response, query_score, relevance_score
                FROM chat_logs
                WHERE user_id = %s AND chat_id = %s
            '''
            cursor.execute(select_query, (User_id, chat_id))
            results = cursor.fetchall()
            for row in results:
                if isinstance(row['timestamp'], datetime):
                    row['timestamp'] = row['timestamp'].isoformat()
        return results
    finally:
        conn.close()

def build_memory_from_results(results):
    # Sort results by timestamp descending and take top 10
    def parse_timestamp(ts):
        try:
            return datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%S")
    sorted_results = sorted(
        results,
        key=lambda x: parse_timestamp(x["timestamp"]),
        reverse=True
    )[:10]

    # Create memory object with specified keys
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        input_key="question",
        output_key="answer",
        return_messages=True
    )

    # Insert messages in chronological order (oldest to newest)
    for item in reversed(sorted_results):
        user_msg = item["user_message"]
        bot_msg = item["bot_response"]
        
        memory.chat_memory.messages.append(HumanMessage(content=user_msg))
        memory.chat_memory.messages.append(AIMessage(content=bot_msg))

    return memory
## DELETES WHOLE CHAT CAREFUL WHEN CALLING THIS 
def delete_messages_by_user_and_chat(User_id, chat_id):
    conn = pymysql.connect(**DB_CONFIG)
    try:
        with conn.cursor() as cursor:
            delete_query = '''
                DELETE FROM chat_logs
                WHERE user_id = %s AND chat_id = %s
            '''
            cursor.execute(delete_query, (User_id, chat_id))
        conn.commit()
    finally:
        conn.close()
    
    
    
def gather_for_analytics(user_id):
    conn = pymysql.connect(**DB_CONFIG)  # Fixed to use pymysql and DictCursor
    cursor = conn.cursor()

    select_query = '''
        SELECT timestamp ,user_message, bot_response, query_score, feedback,  relevance_score, user_id, chat_id
        FROM chat_logs
        WHERE user_id = %s
    '''

    cursor.execute(select_query, (user_id,))
    results = cursor.fetchall()

    cursor.close()
    conn.close()

    return results

def build_chatname_from_user_id(user_id, chat_id):
    allmsgs = retrieve_user_messages_and_scores(user_id, chat_id)
    if not allmsgs:
        return "No messages found for this user and chat ID."
    else:
        return "meoqw"    

def all_user_chatid(user_id):
    """
    Retrieve all unique chat IDs for a given user from the chat_logs table.
    Args:
        user_id (str or int): The user's unique identifier.
    Returns:
        list: List of chat_id values (str or int) for the user.
    """
    conn = pymysql.connect(**DB_CONFIG)
    cursor = conn.cursor()
    select_query = '''
        SELECT DISTINCT chat_id
        FROM chat_logs
        WHERE user_id = %s
    '''
    cursor.execute(select_query, (user_id,))
    results = cursor.fetchall()
    cursor.close()
    conn.close()
    return [row['chat_id'] for row in results]

def get_all_chats_with_messages_for_user(user_id):
    """
    Retrieve all chat sessions and their messages for a user, formatted for frontend/API use.
    Each chat session is represented as a dict with its chat_id and a list of messages.
    Messages are sorted by timestamp (oldest first) and only relevant fields are included.
    Args:
        user_id (str or int): The user's unique identifier.
    Returns:
        list: List of dicts, each with 'chat_id' and 'messages' (list of dicts).
    Example:
        [
            {
                'chat_id': 'abc123',
                'messages': [
                    {'timestamp': ..., 'user_message': ..., 'bot_response': ..., ...},
                    ...
                ]
            },
            ...
        ]
    """
    chat_ids = all_user_chatid(user_id)
    all_chats = []
    for chat_id in chat_ids:
        messages = retrieve_user_messages_and_scores(user_id, chat_id)
        # Sort messages by timestamp ascending (oldest first)
        messages_sorted = sorted(
            messages,
            key=lambda x: x['timestamp']
        )
        # Only include relevant fields for frontend
        formatted_msgs = [
            {
                'timestamp': m['timestamp'],
                'user_message': m['user_message'],
                'bot_response': m['bot_response'],
                'query_score': m.get('query_score'),
                'relevance_score': m.get('relevance_score')
            }
            for m in messages_sorted
        ]
        all_chats.append({
            'chat_id': chat_id,
            'messages': formatted_msgs
        })
    return all_chats
    
if __name__=='__main__':
    # Test and print all chats/messages for user 2 as JSON
    all_chats = get_all_chats_with_messages_for_user("2")
    import json
    print(json.dumps(all_chats, indent=2))