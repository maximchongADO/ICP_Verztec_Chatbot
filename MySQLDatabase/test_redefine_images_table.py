import mysql.connector
DB_CONFIG = {
    'host': 'localhost',
    'user': 'chatbot_user',
    'password': 'strong_password',
    'database': 'chatbot_db',
    'raise_on_warnings': True
}


def recreate_images_table_with_blob():
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor()

    cursor.execute('DROP TABLE IF EXISTS images;')

    cursor.execute('''
    CREATE TABLE images (
        image_id INT AUTO_INCREMENT PRIMARY KEY,
        filename VARCHAR(255),
        image_data LONGBLOB,
        linked_chunk_id INT,
        uploaded_at DATETIME,
        FOREIGN KEY (linked_chunk_id) REFERENCES knowledge_chunks(chunk_id) ON DELETE SET NULL
    );
    ''')

    conn.commit()
    cursor.close()
    conn.close()
    print("Recreated 'images' table with BLOB storage.")

recreate_images_table_with_blob()
