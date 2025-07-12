const mysql = require('mysql2/promise');
const dbConfig = require('./dbConfig');

/**
 * Migration script to add country and department columns to the Users table
 * Run this script to update existing databases
 */
async function addCountryDepartmentColumns() {
    let connection;
    
    try {
        connection = await mysql.createConnection(dbConfig);
        
        console.log('Connected to database, running migration...');
        
        // Check if columns already exist
        const [columns] = await connection.execute(`
            SELECT COLUMN_NAME 
            FROM INFORMATION_SCHEMA.COLUMNS 
            WHERE TABLE_SCHEMA = ? AND TABLE_NAME = 'Users'
        `, [dbConfig.database]);
        
        const existingColumns = columns.map(col => col.COLUMN_NAME);
        
        // Add country column if it doesn't exist
        if (!existingColumns.includes('country')) {
            await connection.execute(`
                ALTER TABLE Users 
                ADD COLUMN country VARCHAR(100) AFTER role
            `);
            console.log('✓ Added country column to Users table');
        } else {
            console.log('- Country column already exists, skipping');
        }
        
        // Add department column if it doesn't exist
        if (!existingColumns.includes('department')) {
            await connection.execute(`
                ALTER TABLE Users 
                ADD COLUMN department VARCHAR(100) AFTER country
            `);
            console.log('✓ Added department column to Users table');
        } else {
            console.log('- Department column already exists, skipping');
        }
        
        console.log('Migration completed successfully!');
        
    } catch (error) {
        console.error('Migration failed:', error);
        throw error;
    } finally {
        if (connection) {
            await connection.end();
        }
    }
}

// Run migration if this script is executed directly
if (require.main === module) {
    addCountryDepartmentColumns()
        .then(() => {
            console.log('Migration script completed');
            process.exit(0);
        })
        .catch((error) => {
            console.error('Migration script failed:', error);
            process.exit(1);
        });
}

module.exports = { addCountryDepartmentColumns };
