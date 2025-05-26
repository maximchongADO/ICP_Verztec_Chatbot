import sqlite3


DB_PATH = 'Database/chatbot.db'
def clear_documents_table():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('DELETE FROM documents')
    conn.commit()
    conn.close()
    print("Cleared all records from documents table.")

if __name__ == '__main__':
    clear_documents_table()