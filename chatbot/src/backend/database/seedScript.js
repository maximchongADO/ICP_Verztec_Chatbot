const mysql = require('mysql2/promise');
const bcrypt = require("bcrypt");
const dbConfig = require("./dbConfig");

const createTableSQL = `
CREATE TABLE IF NOT EXISTS Users (
  id INT AUTO_INCREMENT,
  username VARCHAR(40) NOT NULL,
  email VARCHAR(50) NOT NULL UNIQUE,
  password VARCHAR(100) NOT NULL,
  role ENUM('student', 'admin') NOT NULL,
  PRIMARY KEY (id)
)`;

async function insertUsers(connection) {
  const hashedPassword1 = await bcrypt.hash('password1234', 10);
  const hashedPassword2 = await bcrypt.hash('maximchong1', 10);
  
  const insertSQL = `
    INSERT INTO Users (username, email, password, role) 
    VALUES (?, ?, ?, ?)`;
  
  try {
    await connection.execute(insertSQL, ['Toby', 'toby@gmail.com', hashedPassword1, 'student']);
    await connection.execute(insertSQL, ['Maxim', 'maxim@gmail.com', hashedPassword2, 'admin']);
    console.log('Users inserted successfully');
  } catch (err) {
    console.error('Error inserting users:', err);
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
    
    // Reset the database first
    await resetDatabase(connection);
    
    // Create new table
    await connection.execute(createTableSQL);
    console.log('Created Users table');
    
    // Insert users
    await insertUsers(connection);
    
    console.log('Seeding completed successfully');
  } catch (err) {
    console.error('Seeding error:', err);
  } finally {
    if (connection) {
      await connection.end();
    }
  }
}

run();