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


def insert_documents_from_folder(conn, folder_path, file_extensions):
    cursor = conn.cursor()
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(file_extensions):
            filepath = os.path.abspath(os.path.join(folder_path, filename))

            # Check if this file is already stored
            query = "SELECT COUNT(*) FROM documents WHERE filename = %s AND filepath = %s"
            cursor.execute(query, (filename, filepath))
            exists = cursor.fetchone()[0]

            if exists == 0:
                uploaded_at = datetime.utcnow()
                insert_query = '''
                    INSERT INTO documents (filename, file_type, filepath, uploaded_at)
                    VALUES (%s, %s, %s, %s)
                '''
                cursor.execute(insert_query, (filename, file_extensions[0].strip('.'), filepath, uploaded_at))
            else:
                print(f"Skipping duplicate file: {filename}")
    conn.commit()
    cursor.close()

def insert_all_extracted_texts(conn, folder_path='data/cleaned'):
    cursor = conn.cursor()
    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.txt'):
            filepath = os.path.join(folder_path, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                text_content = f.read().strip()

            # Check if already inserted
            query = "SELECT COUNT(*) FROM extracted_texts WHERE doc_filename = %s"
            cursor.execute(query, (filename,))
            exists = cursor.fetchone()[0]

            if exists == 0:
                insert_query = '''
                    INSERT INTO extracted_texts (doc_filename, text_content, created_at)
                    VALUES (%s, %s, %s)
                '''
                cursor.execute(insert_query, (filename, text_content, datetime.utcnow()))
                print(f"Inserted extracted text for {filename}")
            else:
                print(f"Skipped duplicate for {filename}")
    conn.commit()
    cursor.close()

def store_chat_log(user_message, bot_response, session_id):
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor()

    timestamp = datetime.utcnow()

    insert_query = '''
        INSERT INTO chat_logs (timestamp, user_message, bot_response, session_id)
        VALUES (%s, %s, %s, %s)
    '''
    cursor.execute(insert_query, (timestamp, user_message, bot_response, session_id))
    conn.commit()

    cursor.close()
    conn.close()

def insert_image_blob(conn, image_path, linked_chunk_id=None):
    with open(image_path, 'rb') as file:
        binary_data = file.read()

    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO images (filename, image_data, linked_chunk_id, uploaded_at)
        VALUES (%s, %s, %s, %s)
    ''', (os.path.basename(image_path), binary_data, linked_chunk_id, datetime.utcnow()))
    conn.commit()
    cursor.close()
    print(f"Inserted image {os.path.basename(image_path)} successfully.")
# def main():

#     conn = mysql.connector.connect(**DB_CONFIG)

#     print("Inserting PDF files metadata...")
#     insert_documents_from_folder(conn, 'data/pdf', ('.pdf',))

#     print("Inserting Word files metadata...")
#     insert_documents_from_folder(conn, 'data/word', ('.doc', '.docx'))

#     print("Inserting extracted text files...")
#     insert_all_extracted_texts(conn, 'data/cleaned')

#     print("Done inserting all metadata and extracted text.")

#     conn.close()

# if __name__ == '__main__':
#     main()