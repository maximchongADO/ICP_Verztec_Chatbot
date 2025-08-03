/**
 * Internal API Routes for Python FastAPI Communication
 * These routes handle communication between the Node.js backend and Python FastAPI
 */

const express = require('express');
const multer = require('multer');
const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');
const router = express.Router();

// Configure multer for file uploads
const storage = multer.memoryStorage();
const upload = multer({ 
    storage: storage,
    limits: {
        fileSize: 10 * 1024 * 1024 // 10MB limit
    }
});

// Middleware for internal API authentication
const authenticateInternal = (req, res, next) => {
    const token = req.headers.authorization?.split(' ')[1];
    const expectedToken = process.env.INTERNAL_API_KEY || 'default-key';
    
    if (token !== expectedToken) {
        return res.status(401).json({
            success: false,
            message: 'Unauthorized internal API access'
        });
    }
    
    next();
};

// Internal upload endpoint that calls Python FastAPI
router.post('/upload', authenticateInternal, upload.single('file'), async (req, res) => {
    try {
        if (!req.file) {
            return res.status(400).json({
                success: false,
                error: 'No file provided'
            });
        }

        const country = req.body.country;
        const department = req.body.department;

        if (!country || !department) {
            return res.status(400).json({
                success: false,
                error: 'Country and department are required'
            });
        }

        console.log(`Internal upload: ${req.file.originalname} for ${country}/${department}`);

        // Save file temporarily
        const tempDir = path.join(__dirname, '../backend/python/temp_uploads');
        if (!fs.existsSync(tempDir)) {
            fs.mkdirSync(tempDir, { recursive: true });
        }

        const tempFilePath = path.join(tempDir, req.file.originalname);
        fs.writeFileSync(tempFilePath, req.file.buffer);

        try {
            // Call Python script directly
            const pythonScript = path.join(__dirname, '../backend/python/enhanced_upload_handler.py');
            
            const result = await new Promise((resolve, reject) => {
                const pythonProcess = spawn('python', [
                    pythonScript,
                    tempFilePath,
                    country,
                    department
                ], {
                    cwd: path.join(__dirname, '../backend/python')
                });

                let stdout = '';
                let stderr = '';

                pythonProcess.stdout.on('data', (data) => {
                    stdout += data.toString();
                });

                pythonProcess.stderr.on('data', (data) => {
                    stderr += data.toString();
                });

                pythonProcess.on('close', (code) => {
                    if (code === 0) {
                        try {
                            const result = JSON.parse(stdout);
                            resolve(result);
                        } catch (parseError) {
                            reject(new Error(`Failed to parse Python output: ${parseError.message}`));
                        }
                    } else {
                        reject(new Error(`Python script failed with code ${code}: ${stderr}`));
                    }
                });

                pythonProcess.on('error', (error) => {
                    reject(new Error(`Failed to start Python process: ${error.message}`));
                });
            });

            // Clean up temp file
            if (fs.existsSync(tempFilePath)) {
                fs.unlinkSync(tempFilePath);
            }

            return res.json(result);

        } catch (error) {
            // Clean up temp file on error
            if (fs.existsSync(tempFilePath)) {
                fs.unlinkSync(tempFilePath);
            }
            throw error;
        }

    } catch (error) {
        console.error('Internal upload error:', error);
        return res.status(500).json({
            success: false,
            error: error.message
        });
    }
});

// Get upload configuration
router.get('/upload/config', authenticateInternal, async (req, res) => {
    try {
        // Call Python script to get configuration
        const pythonScript = path.join(__dirname, '../backend/python/get_config.py');
        
        const result = await new Promise((resolve, reject) => {
            const pythonProcess = spawn('python', [pythonScript], {
                cwd: path.join(__dirname, '../backend/python')
            });

            let stdout = '';
            let stderr = '';

            pythonProcess.stdout.on('data', (data) => {
                stdout += data.toString();
            });

            pythonProcess.stderr.on('data', (data) => {
                stderr += data.toString();
            });

            pythonProcess.on('close', (code) => {
                if (code === 0) {
                    try {
                        const result = JSON.parse(stdout);
                        resolve(result);
                    } catch (parseError) {
                        // Return default config if parsing fails
                        resolve({
                            success: true,
                            supported_countries: ['china', 'singapore'],
                            supported_departments: ['hr', 'it'],
                            supported_file_types: ['.pdf', '.doc', '.docx', '.txt', '.pptx'],
                            total_combinations: 4
                        });
                    }
                } else {
                    // Return default config if Python script fails
                    resolve({
                        success: true,
                        supported_countries: ['china', 'singapore'],
                        supported_departments: ['hr', 'it'],
                        supported_file_types: ['.pdf', '.doc', '.docx', '.txt', '.pptx'],
                        total_combinations: 4
                    });
                }
            });

            pythonProcess.on('error', (error) => {
                // Return default config if process fails
                resolve({
                    success: true,
                    supported_countries: ['china', 'singapore'],
                    supported_departments: ['hr', 'it'],
                    supported_file_types: ['.pdf', '.doc', '.docx', '.txt', '.pptx'],
                    total_combinations: 4
                });
            });
        });

        return res.json(result);

    } catch (error) {
        console.error('Config retrieval error:', error);
        // Return default configuration
        return res.json({
            success: true,
            supported_countries: ['china', 'singapore'],
            supported_departments: ['hr', 'it'],
            supported_file_types: ['.pdf', '.doc', '.docx', '.txt', '.pptx'],
            total_combinations: 4
        });
    }
});

// Get available indices
router.get('/upload/indices', authenticateInternal, async (req, res) => {
    try {
        // Call Python script to get available indices
        const pythonScript = path.join(__dirname, '../backend/python/get_indices.py');
        
        const result = await new Promise((resolve, reject) => {
            const pythonProcess = spawn('python', [pythonScript], {
                cwd: path.join(__dirname, '../backend/python')
            });

            let stdout = '';
            let stderr = '';

            pythonProcess.stdout.on('data', (data) => {
                stdout += data.toString();
            });

            pythonProcess.stderr.on('data', (data) => {
                stderr += data.toString();
            });

            pythonProcess.on('close', (code) => {
                if (code === 0) {
                    try {
                        const result = JSON.parse(stdout);
                        resolve(result);
                    } catch (parseError) {
                        resolve({
                            success: true,
                            available_indices: {},
                            total_indices: 0
                        });
                    }
                } else {
                    resolve({
                        success: true,
                        available_indices: {},
                        total_indices: 0
                    });
                }
            });

            pythonProcess.on('error', (error) => {
                resolve({
                    success: true,
                    available_indices: {},
                    total_indices: 0
                });
            });
        });

        return res.json(result);

    } catch (error) {
        console.error('Indices retrieval error:', error);
        return res.json({
            success: true,
            available_indices: {},
            total_indices: 0
        });
    }
});

module.exports = router;
