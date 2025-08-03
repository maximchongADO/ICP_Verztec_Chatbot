const mysql = require('mysql2/promise');
const fs = require('fs');
const path = require('path');
const dbConfig = require('./dbConfig.js');

async function runMigration() {
    let connection;
    
    try {
        // Create connection
        connection = await mysql.createConnection(dbConfig);
        console.log('Connected to MySQL database');
        
        // Read the migration SQL file
        const migrationPath = path.join(__dirname, 'migration_add_country_department.sql');
        const migrationSQL = fs.readFileSync(migrationPath, 'utf8');
        
        // Split SQL file into individual statements
        const statements = migrationSQL
            .split(';')
            .map(stmt => stmt.trim())
            .filter(stmt => stmt.length > 0 && !stmt.startsWith('--'));
        
        console.log(`Found ${statements.length} SQL statements to execute`);
        
        // Execute each statement
        for (let i = 0; i < statements.length; i++) {
            const statement = statements[i];
            if (statement) {
                try {
                    console.log(`Executing statement ${i + 1}:`, statement.substring(0, 50) + '...');
                    await connection.execute(statement);
                    console.log(`✅ Statement ${i + 1} executed successfully`);
                } catch (error) {
                    if (error.code === 'ER_DUP_FIELDNAME') {
                        console.log(`⚠️  Column already exists, skipping: ${error.message}`);
                    } else if (error.code === 'ER_DUP_KEYNAME') {
                        console.log(`⚠️  Index already exists, skipping: ${error.message}`);
                    } else {
                        console.error(`❌ Error executing statement ${i + 1}:`, error.message);
                        throw error;
                    }
                }
            }
        }
        
        // Verify the table structure
        console.log('\n📋 Final table structure:');
        const [rows] = await connection.execute('DESCRIBE cleaned_texts');
        console.table(rows);
        
        console.log('\n🎉 Migration completed successfully!');
        
    } catch (error) {
        console.error('❌ Migration failed:', error.message);
        process.exit(1);
    } finally {
        if (connection) {
            await connection.end();
            console.log('Database connection closed');
        }
    }
}

// Run the migration
runMigration();
