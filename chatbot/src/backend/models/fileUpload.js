// const mysql = require('mysql2/promise');
// const dbConfig = require('../database/dbConfig.js');

// class FileUpload {
//     static async query(sql, params = []) {
//         const connection = await mysql.createConnection(dbConfig);
//         try {
//             const [result] = await connection.execute(sql, params);
//             return result;
//         } finally {
//             await connection.end();
//         }
//     }

//     static async createFileRecord({ filename, cleanedContent, uploadedBy, country, department, faissIndexPath }) {
//         const sql = `
//             INSERT INTO cleaned_texts (
//                 filename, 
//                 cleaned_content, 
//                 uploaded_by, 
//                 country,
//                 department,
//                 faiss_index_path,
//                 created_at
//             ) VALUES (?, ?, ?, ?, ?, ?, NOW())`;
//         
//         try {
//             console.log('Creating file record:', { 
//                 filename, 
//                 contentLength: cleanedContent?.length, 
//                 uploadedBy, 
//                 country, 
//                 department,
//                 faissIndexPath 
//             });
//             
//             // Ensure we have required values
//             if (!filename || !cleanedContent) {
//                 throw new Error('Missing required fields: filename and cleanedContent');
//             }

//             const result = await this.query(sql, [
//                 filename,
//                 cleanedContent,
//                 uploadedBy || null,  // Convert undefined to null
//                 country || null,
//                 department || null,
//                 faissIndexPath || null
//             ]);

//             return { success: true, fileId: result.insertId };
//         } catch (error) {
//             console.error('Error in createFileRecord:', error);
//             throw error;
//         }
//     }

//     static async getFileById(fileId) {
//         const sql = 'SELECT * FROM cleaned_texts WHERE id = ?';
//         try {
//             const rows = await this.query(sql, [fileId]);
//             return rows[0];
//         } catch (error) {
//             console.error('Error in getFileById:', error);
//             throw error;
//         }
//     }

//     static async deleteFile(fileId) {
//         const sql = 'DELETE FROM cleaned_texts WHERE id = ?';
//         try {
//             const result = await this.query(sql, [fileId]);
//             return result.affectedRows > 0;
//         } catch (error) {
//             console.error('Error in deleteFile:', error);
//             throw error;
//         }
//     }
// }

// module.exports = FileUpload;
