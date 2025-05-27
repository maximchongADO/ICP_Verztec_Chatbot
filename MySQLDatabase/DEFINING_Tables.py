import os
import mysql.connector
from datetime import datetime

DB_CONFIG = {
    'host': 'localhost',
    'user': 'chatbot_user',
    'password': 'strong_password',
    'database': 'chatbot_db',
    'raise_on_warnings': True
}

def create_tables():
    conn = mysql.connector.connect(**DB_CONFIG)
    c = conn.cursor()

    statements = [
            '''
    CREATE TABLE IF NOT EXISTS documents (
        doc_id INT AUTO_INCREMENT PRIMARY KEY,
        filename VARCHAR(255),
        file_type VARCHAR(20),
        filepath VARCHAR(500),
        uploaded_at DATETIME
    )
    ''',
    '''
    CREATE TABLE knowledge_chunks (
        chunk_id VARCHAR(255) PRIMARY KEY,
        text LONGTEXT,
        source VARCHAR(255),
        images JSON,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    );
    ''',

    '''
    CREATE TABLE IF NOT EXISTS chat_logs (
        log_id INT AUTO_INCREMENT PRIMARY KEY,
        timestamp DATETIME,
        user_message LONGTEXT,
        bot_response LONGTEXT,
        session_id VARCHAR(255)
    )
    ''',

    '''
    CREATE TABLE IF NOT EXISTS extracted_texts (
        id INT AUTO_INCREMENT PRIMARY KEY,
        doc_filename VARCHAR(255),
        text_content LONGTEXT,
        created_at DATETIME
    )
    ''',
     # RBAC tables
    '''
    CREATE TABLE IF NOT EXISTS roles (
        role_id INT AUTO_INCREMENT PRIMARY KEY,
        role_name VARCHAR(50) UNIQUE NOT NULL
    )
    ''',

    '''
    CREATE TABLE IF NOT EXISTS users (
        user_id INT AUTO_INCREMENT PRIMARY KEY,
        username VARCHAR(100) UNIQUE NOT NULL,
        password_hash VARCHAR(255) NOT NULL,
        email VARCHAR(255) UNIQUE NOT NULL,
        role_id INT,
        department VARCHAR(100),
        country VARCHAR(100),
        FOREIGN KEY (role_id) REFERENCES roles(role_id)
    )
    '''
    ]
    for stmt in statements:
        c.execute(stmt)
        # Explicitly ensure no unread results remain
        try:
            c.fetchall()
        except mysql.connector.errors.InterfaceError:
            # No results to fetch, ignore
            pass

    conn.commit()
    c.close()
    conn.close()

# def main():
#     # Create tables once
#     create_tables()

# if __name__ == '__main__':
#     main()

