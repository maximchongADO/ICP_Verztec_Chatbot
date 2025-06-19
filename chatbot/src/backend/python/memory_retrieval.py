from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage
import re
from datetime import datetime
import mysql.connector

DB_CONFIG = {
    'host': 'localhost',
    'user': 'chatbot_user',
    'password': 'strong_password',
    'database': 'chatbot_db',
    'raise_on_warnings': True
}
## SQL DB needs to be updated to include user and chatid 
def retrieve_user_messages_and_scores(User_id, chat_id):
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor(dictionary=True)

    select_query = '''
        SELECT timestamp, user_message, bot_response, query_score, relevance_score
        FROM chat_logs
        WHERE user_id = %s AND chat_id = %s
    '''

    cursor.execute(select_query, (User_id, chat_id))
    results = cursor.fetchall()

    cursor.close()
    conn.close()

    return results

def build_memory_from_results(results):
    # Sort results by timestamp descending and take top 10
    sorted_results = sorted(
        results,
        key=lambda x: datetime.strptime(x["timestamp"], "%Y-%m-%d %H:%M:%S"),
        reverse=True
    )[:10]

    # Create memory object
    memory = ConversationBufferMemory(return_messages=True)

    # Insert messages in chronological order
    for item in reversed(sorted_results):  # oldest to newest
        user_msg = item["user_message"]
        bot_msg = item["bot_response"]
        
        memory.chat_memory.messages.append(HumanMessage(content=user_msg))
        memory.chat_memory.messages.append(AIMessage(content=bot_msg))

    return memory

if __name__=='__main__':
    print('moew')