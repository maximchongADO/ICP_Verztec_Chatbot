const mysql = require('mysql2/promise');
const bcrypt = require("bcrypt");
const dbConfig = require("./dbConfig");
const path = require('path');
const fs = require('fs').promises;  // Using promises version
const fsSync = require('fs');  // Add this for sync operations

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
    feedback VARCHAR(20),
    query_score FLOAT,
    relevance_score FLOAT,
    user_id VARCHAR(255),
    chat_id VARCHAR(255),
    chat_name VARCHAR(255)
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
  
  mailing_list: `
    CREATE TABLE IF NOT EXISTS mailing_list (
      id INT AUTO_INCREMENT PRIMARY KEY,
      email VARCHAR(255) NOT NULL UNIQUE,
      name VARCHAR(100) NULL,
      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
      updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
    )`,

  users: `
    CREATE TABLE IF NOT EXISTS Users (
      id INT AUTO_INCREMENT,
      username VARCHAR(40) NOT NULL,
      email VARCHAR(50) NOT NULL UNIQUE,
      password VARCHAR(100) NOT NULL,
      role ENUM('user', 'admin','manager') NOT NULL,
      country VARCHAR(100),
      department VARCHAR(100),
      PRIMARY KEY (id)
    )`,
    
  cleaned_texts: `
    CREATE TABLE IF NOT EXISTS cleaned_texts (
      id INT AUTO_INCREMENT PRIMARY KEY,
      filename VARCHAR(255) NOT NULL,
      cleaned_content LONGTEXT NOT NULL,
      uploaded_by INT,
      created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
      FOREIGN KEY (uploaded_by) REFERENCES Users(id) ON DELETE SET NULL
    )`,
    
  hr_escalations: `
    CREATE TABLE IF NOT EXISTS hr_escalations (
      id INT AUTO_INCREMENT PRIMARY KEY,
      escalation_id VARCHAR(50) UNIQUE NOT NULL,
      timestamp DATETIME NOT NULL,
      user_id VARCHAR(100) NOT NULL,
      chat_id VARCHAR(100) NOT NULL,
      user_message TEXT NOT NULL,
      issue_summary TEXT NOT NULL,
      user_description TEXT NULL,
      status ENUM('PENDING', 'ACKNOWLEDGED', 'IN_PROGRESS', 'RESOLVED', 'CLOSED') DEFAULT 'PENDING',
      priority ENUM('LOW', 'NORMAL', 'HIGH', 'URGENT') DEFAULT 'NORMAL',
      assigned_to VARCHAR(100) NULL,
      notes TEXT NULL,
      resolved_at DATETIME NULL,
      created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
      updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
      INDEX idx_escalation_id (escalation_id),
      INDEX idx_user_id (user_id),
      INDEX idx_status (status),
      INDEX idx_timestamp (timestamp)
    )`
  };

async function createTables(connection) {
  try {
    // Create tables in order (roles first, then users due to foreign key)
    const tableOrder = ['documents', 'knowledge_chunks', 'chat_logs', 'extracted_texts', 'roles', 'users', 'cleaned_texts', 'hr_escalations', 'mailing_list'];
    
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
        // Insert users
        const hashedPassword1 = await bcrypt.hash('password1234', 10);
        const hashedPassword2 = await bcrypt.hash('maximchong1', 10);
        const hashedPassword3 = await bcrypt.hash('manager123', 10);
        
        const insertSQL = `
            INSERT INTO Users (username, email, password, role) 
            VALUES (?, ?, ?, ?)`;
        
        await connection.execute(insertSQL, ['Toby', 'toby@gmail.com', hashedPassword1, 'user']);
        await connection.execute(insertSQL, ['Maxim', 'maxim@gmail.com', hashedPassword2, 'admin']);
        await connection.execute(insertSQL, ['Manager', 'manager@gmail.com', hashedPassword3, 'manager']);
        console.log('Users inserted successfully');
        
        // Update data directory path to be relative to src folder
        const scriptDir = path.dirname(__filename);
        const srcDir = path.resolve(scriptDir, '..');
        const pythonDataDir = path.join(srcDir, 'python', 'data');
        
        // Check if directories exist before trying to read them
        const pdfDir = path.join(pythonDataDir, 'pdf');
        const wordDir = path.join(pythonDataDir, 'word');
        const cleanedDir = path.join(pythonDataDir, 'cleaned');

        // Create directories if they don't exist
        [pdfDir, wordDir, cleanedDir].forEach(dir => {
            if (!fsSync.existsSync(dir)) {  // Use synchronous version
                fsSync.mkdirSync(dir, { recursive: true });  // Use synchronous version
                console.log(`Created directory: ${dir}`);
            }
        });

        // Only try to insert documents if directories exist
        if (fsSync.existsSync(pdfDir)) {  // Use synchronous version
            await insertDocumentsFromFolder(connection, pdfDir, ['.pdf']);
            console.log('PDF documents inserted successfully');
        } else {
            console.log('PDF directory not found, skipping PDF documents');
        }

        if (fsSync.existsSync(wordDir)) {
          await insertDocumentsFromFolder(connection, wordDir, ['.doc', '.docx']);
          console.log('Word documents inserted successfully');
        } else {
          console.log('Word directory not found, skipping Word documents');
        }

        if (fsSync.existsSync(cleanedDir)) {
          await insertExtractedTexts(connection, cleanedDir);
          console.log('Extracted texts inserted successfully');
        } else {
          console.log('Cleaned directory not found, skipping extracted texts');
        }
        
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