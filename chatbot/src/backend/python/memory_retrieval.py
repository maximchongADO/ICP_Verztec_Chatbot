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
    
    for row in results:
        row['timestamp'] = row['timestamp'].isoformat()

    cursor.close()
    conn.close()

    return results

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
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor()

    delete_query = '''
        DELETE FROM chat_logs
        WHERE user_id = %s AND chat_id = %s
    '''

    cursor.execute(delete_query, (User_id, chat_id))
    conn.commit()

    cursor.close()
    conn.close()
    
    
    
if __name__=='__main__':
    delete_messages_by_user_and_chat("2", "chat123")
    
    r= retrieve_user_messages_and_scores("2", "chat123")
    for each in r:
        print("-" * 20)
        print(f"Timestamp: {each['timestamp']}, \n\nUser: {each['user_message']},\n\nBot: {each['bot_response']}, \n\nQuery Score: {each['query_score']}, Relevance Score: {each['relevance_score']}")