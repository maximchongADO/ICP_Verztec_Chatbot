const path = require('path');
const fs = require('fs');
const multer = require('multer');
const xlsx = require('xlsx');
const csv = require('csv-parser');
const bcrypt = require('bcrypt');
const mysql = require('mysql2/promise');
const dbConfig = require('../database/dbConfig.js');

// Utility functions
const hashPassword = async (password) => {
  const saltRounds = 10;
  return await bcrypt.hash(password, saltRounds);
};

const validateEmail = (email) => {
  const pattern = /^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$/;
  return pattern.test(email);
};

const validatePassword = (password) => {
  if (password.length < 6) {
    return { valid: false, message: "Password must be at least 6 characters long" };
  }
  return { valid: true, message: "Password is valid" };
};

const validateUserType = (userType) => {
  const validTypes = ['user', 'admin'];
  return validTypes.includes(userType.toLowerCase());
};

const validateUsername = (username) => {
  if (username.length < 3) {
    return { valid: false, message: "Username must be at least 3 characters long" };
  }
  if (username.includes(' ')) {
    return { valid: false, message: "Username cannot contain spaces" };
  }
  return { valid: true, message: "Username is valid" };
};

// Parse Excel file
const parseExcelFile = (filePath) => {
  try {
    const workbook = xlsx.readFile(filePath);
    const sheetName = workbook.SheetNames[0];
    const worksheet = workbook.Sheets[sheetName];
    const jsonData = xlsx.utils.sheet_to_json(worksheet);
    return { success: true, data: jsonData };
  } catch (error) {
    return { success: false, error: error.message };
  }
};

// Parse CSV file
const parseCSVFile = (filePath) => {
  return new Promise((resolve, reject) => {
    const results = [];
    fs.createReadStream(filePath)
      .pipe(csv())
      .on('data', (data) => results.push(data))
      .on('end', () => {
        resolve({ success: true, data: results });
      })
      .on('error', (error) => {
        reject({ success: false, error: error.message });
      });
  });
};

// Database operations
const checkUserExists = async (connection, username, email) => {
  try {
    // Check by username
    const [usernameRows] = await connection.execute(
      'SELECT id FROM users WHERE username = ?', 
      [username]
    );
    if (usernameRows.length > 0) {
      return { exists: true, message: `User with username '${username}' already exists` };
    }

    // Check by email
    const [emailRows] = await connection.execute(
      'SELECT id FROM users WHERE email = ?', 
      [email]
    );
    if (emailRows.length > 0) {
      return { exists: true, message: `User with email '${email}' already exists` };
    }

    return { exists: false, message: "User does not exist" };
  } catch (error) {
    return { exists: true, message: `Database error while checking user: ${error.message}` };
  }
};

const createUser = async (connection, userRecord) => {
  try {
    const hashedPassword = await hashPassword(userRecord.password);
    
    const insertQuery = `
      INSERT INTO users (username, email, password, role) 
      VALUES (?, ?, ?, ?)
    `;
    
    const values = [
      userRecord.username,
      userRecord.email,
      hashedPassword,
      userRecord.user_type
    ];

    await connection.execute(insertQuery, values);
    return { success: true, message: "User created successfully" };
  } catch (error) {
    return { success: false, message: `Database error: ${error.message}` };
  }
};

const validateUserRecord = (row, rowIndex) => {
  const errors = [];
  const requiredColumns = ['username', 'email', 'password', 'user_type'];

  // Check for missing required fields
  for (const col of requiredColumns) {
    if (!row[col] || String(row[col]).trim() === '') {
      errors.push(`Missing required field '${col}'`);
    }
  }

  if (errors.length > 0) {
    return { valid: false, errors, userRecord: null };
  }

  // Create user record
  const userRecord = {
    username: String(row.username).trim(),
    email: String(row.email).trim(),
    password: String(row.password).trim(),
    user_type: String(row.user_type).trim()
  };

  // Validate email format
  if (!validateEmail(userRecord.email)) {
    errors.push(`Invalid email format: ${userRecord.email}`);
  }

  // Validate password strength
  const passwordValidation = validatePassword(userRecord.password);
  if (!passwordValidation.valid) {
    errors.push(passwordValidation.message);
  }

  // Validate user type
  if (!validateUserType(userRecord.user_type)) {
    errors.push(`Invalid user type: ${userRecord.user_type}. Must be 'user' or 'admin'`);
  }

  // Validate username
  const usernameValidation = validateUsername(userRecord.username);
  if (!usernameValidation.valid) {
    errors.push(usernameValidation.message);
  }

  return { 
    valid: errors.length === 0, 
    errors, 
    userRecord: errors.length === 0 ? userRecord : null 
  };
};

// Configure multer for file uploads
const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    const uploadDir = path.join(__dirname, '..', '..', 'uploads');
    // Create uploads directory if it doesn't exist
    if (!fs.existsSync(uploadDir)) {
      fs.mkdirSync(uploadDir, { recursive: true });
    }
    cb(null, uploadDir);
  },
  filename: function (req, file, cb) {
    // Generate unique filename with timestamp
    const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1E9);
    cb(null, 'batch-users-' + uniqueSuffix + path.extname(file.originalname));
  }
});

const upload = multer({
  storage: storage,
  fileFilter: function (req, file, cb) {
    // Accept only Excel and CSV files
    const allowedTypes = [
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', // .xlsx
      'application/vnd.ms-excel', // .xls
      'text/csv' // .csv
    ];
    
    const allowedExtensions = ['.xlsx', '.xls', '.csv'];
    const fileExtension = path.extname(file.originalname).toLowerCase();
    
    if (allowedTypes.includes(file.mimetype) || allowedExtensions.includes(fileExtension)) {
      cb(null, true);
    } else {
      cb(new Error('Only Excel (.xlsx, .xls) and CSV files are allowed'), false);
    }
  },
  limits: {
    fileSize: 5 * 1024 * 1024 // 5MB limit
  }
});

const batchUploadUsers = async (req, res) => {
  const startTime = Date.now();
  let connection;
  
  try {
    if (!req.file) {
      return res.status(400).json({
        success: false,
        message: 'No file uploaded. Please select an Excel or CSV file.'
      });
    }

    const filePath = req.file.path;
    const fileExtension = path.extname(req.file.originalname).toLowerCase();
    
    // Initialize result object
    const result = {
      total_records: 0,
      successful_uploads: 0,
      failed_uploads: 0,
      errors: [],
      duplicates_handled: 0,
      processing_time: 0
    };

    // Parse file based on extension
    let parseResult;
    if (fileExtension === '.csv') {
      parseResult = await parseCSVFile(filePath);
    } else {
      parseResult = parseExcelFile(filePath);
    }

    // Clean up uploaded file
    if (fs.existsSync(filePath)) {
      fs.unlinkSync(filePath);
    }

    if (!parseResult.success) {
      return res.status(400).json({
        success: false,
        message: 'Failed to parse file',
        error: parseResult.error
      });
    }

    const data = parseResult.data;
    
    // Validate data structure
    if (!data || data.length === 0) {
      return res.status(400).json({
        success: false,
        message: 'The uploaded file is empty or contains no valid data'
      });
    }

    // Check for required columns
    const requiredColumns = ['username', 'email', 'password', 'user_type'];
    const firstRow = data[0];
    const missingColumns = requiredColumns.filter(col => !(col in firstRow));
    
    if (missingColumns.length > 0) {
      return res.status(400).json({
        success: false,
        message: `Missing required columns: ${missingColumns.join(', ')}`
      });
    }

    result.total_records = data.length;

    // Connect to database
    connection = await mysql.createConnection(dbConfig);
    await connection.beginTransaction();

    // Process each row
    for (let index = 0; index < data.length; index++) {
      const row = data[index];
      const rowNumber = index + 2; // Excel row number (header is row 1)

      try {
        // Validate user record
        const validation = validateUserRecord(row, index);
        
        if (!validation.valid) {
          result.failed_uploads++;
          for (const error of validation.errors) {
            result.errors.push({
              row: rowNumber,
              error: error
            });
          }
          continue;
        }

        // Check if user exists
        const existsCheck = await checkUserExists(connection, validation.userRecord.username, validation.userRecord.email);
        
        if (existsCheck.exists) {
          result.duplicates_handled++;
          result.failed_uploads++;
          result.errors.push({
            row: rowNumber,
            error: `Skipped - ${existsCheck.message}`
          });
          continue;
        }

        // Create user
        const createResult = await createUser(connection, validation.userRecord);
        
        if (createResult.success) {
          result.successful_uploads++;
        } else {
          result.failed_uploads++;
          result.errors.push({
            row: rowNumber,
            error: createResult.message
          });
        }

      } catch (error) {
        result.failed_uploads++;
        result.errors.push({
          row: rowNumber,
          error: `Unexpected error: ${error.message}`
        });
      }
    }

    // Commit transaction
    await connection.commit();
    
    // Calculate processing time
    result.processing_time = (Date.now() - startTime) / 1000;

    res.json({
      success: true,
      message: 'Batch upload completed',
      result: result
    });

  } catch (error) {
    console.error('Batch upload error:', error);
    
    // Rollback transaction if exists
    if (connection) {
      try {
        await connection.rollback();
      } catch (rollbackError) {
        console.error('Rollback error:', rollbackError);
      }
    }
    
    // Clean up uploaded file if it exists
    if (req.file && fs.existsSync(req.file.path)) {
      fs.unlinkSync(req.file.path);
    }

    res.status(500).json({
      success: false,
      message: 'Internal server error during batch upload',
      error: error.message
    });
  } finally {
    if (connection) {
      await connection.end();
    }
  }
};

const generateSampleFile = async (req, res) => {
  try {
    const { format = 'xlsx' } = req.query;
    const uploadDir = path.join(__dirname, '..', '..', 'uploads');
    
    // Create uploads directory if it doesn't exist
    if (!fs.existsSync(uploadDir)) {
      fs.mkdirSync(uploadDir, { recursive: true });
    }

    const fileName = `sample_users_${Date.now()}.${format}`;
    const outputPath = path.join(uploadDir, fileName);

    // Sample data
    const sampleData = [
      {
        username: 'john_doe',
        email: 'john.doe@company.com',
        password: 'password123',
        user_type: 'user'
      },
      {
        username: 'jane_smith',
        email: 'jane.smith@company.com',
        password: 'securepass456',
        user_type: 'user'
      },
      {
        username: 'admin_user',
        email: 'admin@company.com',
        password: 'adminpass789',
        user_type: 'admin'
      }
    ];

    if (format === 'csv') {
      // Generate CSV
      const csvHeader = 'username,email,password,user_type\n';
      const csvRows = sampleData.map(row => 
        `${row.username},${row.email},${row.password},${row.user_type}`
      ).join('\n');
      
      fs.writeFileSync(outputPath, csvHeader + csvRows);
    } else {
      // Generate Excel
      const worksheet = xlsx.utils.json_to_sheet(sampleData);
      const workbook = xlsx.utils.book_new();
      xlsx.utils.book_append_sheet(workbook, worksheet, 'Users');
      xlsx.writeFile(workbook, outputPath);
    }

    if (!fs.existsSync(outputPath)) {
      return res.status(500).json({
        success: false,
        message: 'Sample file was not created'
      });
    }

    // Send file for download
    res.download(outputPath, fileName, (err) => {
      if (err) {
        console.error('Error sending file:', err);
      }
      
      // Clean up file after download
      setTimeout(() => {
        if (fs.existsSync(outputPath)) {
          fs.unlinkSync(outputPath);
        }
      }, 5000);
    });

  } catch (error) {
    console.error('Sample file generation error:', error);
    res.status(500).json({
      success: false,
      message: 'Internal server error',
      error: error.message
    });
  }
};

module.exports = {
  batchUploadUsers,
  generateSampleFile,
  upload
};
