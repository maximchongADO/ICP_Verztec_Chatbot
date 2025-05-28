const mysql = require('mysql2/promise');
const bcrypt = require("bcrypt");
const dbConfig = require("./dbConfig");
const path = require('path');
const fs = require('fs').promises;

const createTablesSQL = {
  documents: `
    CREATE TABLE IF NOT EXISTS documents (
      doc_id INT AUTO_INCREMENT PRIMARY KEY,
      filename VARCHAR(255),
      file_type VARCHAR(20),
      filepath VARCHAR(500),
      uploaded_at DATETIME
    )`,
    
  knowledge_chunks: `
    CREATE TABLE knowledge_chunks (
      chunk_id VARCHAR(255) PRIMARY KEY,
      text LONGTEXT,
    
      source VARCHAR(255),
      images JSON,
      created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )`,
    
  chat_logs: `
    CREATE TABLE IF NOT EXISTS chat_logs (
      log_id INT AUTO_INCREMENT PRIMARY KEY,
      timestamp DATETIME,
      user_message LONGTEXT,
      bot_response LONGTEXT,
      session_id VARCHAR(255)
    )`,
    
  extracted_texts: `
    CREATE TABLE IF NOT EXISTS extracted_texts (
      id INT AUTO_INCREMENT PRIMARY KEY,
      doc_filename VARCHAR(255),
      text_content LONGTEXT,
      created_at DATETIME
    )`,
    
  roles: `
    CREATE TABLE IF NOT EXISTS roles (
      role_id INT AUTO_INCREMENT PRIMARY KEY,
      role_name VARCHAR(50) UNIQUE NOT NULL
    )`,
    
  users: `
    CREATE TABLE IF NOT EXISTS users (
      user_id INT AUTO_INCREMENT PRIMARY KEY,
      username VARCHAR(100) UNIQUE NOT NULL,
      password_hash VARCHAR(255) NOT NULL,
      email VARCHAR(255) UNIQUE NOT NULL,
      role_id INT,
      department VARCHAR(100),
      country VARCHAR(100),
      FOREIGN KEY (role_id) REFERENCES roles(role_id)
    )`
};

async function createTables(connection) {
  try {
    // Create tables in order (roles first, then users due to foreign key)
    const tableOrder = ['documents', 'knowledge_chunks', 'chat_logs', 'extracted_texts', 'roles', 'users'];
    
    for (const tableName of tableOrder) {
      await connection.execute(createTablesSQL[tableName]);
      console.log(`Created ${tableName} table`);
    }
  } catch (err) {
    console.error('Error creating tables:', err);
    throw err;
  }
}

async function insertDocumentsFromFolder(connection, folderPath, fileExtensions) {
  try {
    const files = await fs.readdir(folderPath);
    for (const filename of files) {
      if (fileExtensions.some(ext => filename.toLowerCase().endsWith(ext))) {
        const filepath = path.resolve(folderPath, filename);
        
        // Check for duplicates
        const [rows] = await connection.execute(
          'SELECT COUNT(*) as count FROM documents WHERE filename = ? AND filepath = ?',
          [filename, filepath]
        );
        
        if (rows[0].count === 0) {
          await connection.execute(
            'INSERT INTO documents (filename, file_type, filepath, uploaded_at) VALUES (?, ?, ?, ?)',
            [filename, path.extname(filename).slice(1), filepath, new Date()]
          );
          console.log(`Inserted document: ${filename}`);
        }
      }
    }
  } catch (err) {
    console.error('Error inserting documents:', err);
    throw err;
  }
}

async function insertExtractedTexts(connection, folderPath) {
  try {
    const files = await fs.readdir(folderPath);
    for (const filename of files) {
      if (filename.toLowerCase().endsWith('.txt')) {
        const filepath = path.join(folderPath, filename);
        const content = await fs.readFile(filepath, 'utf-8');
        
        // Check for duplicates
        const [rows] = await connection.execute(
          'SELECT COUNT(*) as count FROM extracted_texts WHERE doc_filename = ?',
          [filename]
        );
        
        if (rows[0].count === 0) {
          await connection.execute(
            'INSERT INTO extracted_texts (doc_filename, text_content, created_at) VALUES (?, ?, ?)',
            [filename, content.trim(), new Date()]
          );
          console.log(`Inserted extracted text: ${filename}`);
        }
      }
    }
  } catch (err) {
    console.error('Error inserting extracted texts:', err);
    throw err;
  }
}

async function insertInitialData(connection) {
  try {
    // Insert roles first
    await connection.execute(`INSERT INTO roles (role_name) VALUES ('admin'), ('user'), ('manager')`);
    console.log('Roles inserted successfully');

    // Insert users with role references
    const hashedPassword1 = await bcrypt.hash('password1234', 10);
    const hashedPassword2 = await bcrypt.hash('maximchong1', 10);
    
    const insertUserSQL = `
      INSERT INTO users (username, email, password_hash, role_id) 
      SELECT ?, ?, ?, role_id FROM roles WHERE role_name = ?`;
    
    await connection.execute(insertUserSQL, ['Toby', 'toby@gmail.com', hashedPassword1, 'user']);
    await connection.execute(insertUserSQL, ['Maxim', 'maxim@gmail.com', hashedPassword2, 'admin']);
    console.log('Users inserted successfully');
    
    // Update data directory path to be relative to project root
    const dataDir = path.join(__dirname, '..', '..', '..', '..', 'data');
    await insertDocumentsFromFolder(connection, path.join(dataDir, 'pdf'), ['.pdf']);
    await insertDocumentsFromFolder(connection, path.join(dataDir, 'word'), ['.doc', '.docx']);
    
    // Insert extracted texts with updated path
    await insertExtractedTexts(connection, path.join(dataDir, 'cleaned'));
    
  
    
  
    
    console.log('All initial data inserted successfully');
  } catch (err) {
    console.error('Error inserting initial data:', err);
    throw err;
  }
}

async function resetDatabase(connection) {
  try {
    // Disable foreign key checks to allow dropping tables with dependencies
    await connection.execute('SET FOREIGN_KEY_CHECKS = 0');
    
    // Get all tables in the database
    const [tables] = await connection.execute('SHOW TABLES');
    
    // Drop each table
    for (const tableRow of tables) {
      const tableName = tableRow[`Tables_in_${dbConfig.database}`];
      await connection.execute(`DROP TABLE IF EXISTS ${tableName}`);
      console.log(`Dropped table: ${tableName}`);
    }
    
    // Re-enable foreign key checks
    await connection.execute('SET FOREIGN_KEY_CHECKS = 1');
    
    console.log('Database reset completed');
  } catch (err) {
    console.error('Error resetting database:', err);
    throw err;
  }
}

async function run() {
  let connection;
  try {
    connection = await mysql.createConnection(dbConfig);
    
    await resetDatabase(connection);
    await createTables(connection);
    await insertInitialData(connection);
    
    console.log('Database setup completed successfully');
  } catch (err) {
    console.error('Setup error:', err);
  } finally {
    if (connection) {
      await connection.end();
    }
  }
}

run();