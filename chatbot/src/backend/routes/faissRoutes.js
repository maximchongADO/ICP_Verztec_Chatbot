const express = require('express');
const { spawn } = require('child_process');
const path = require('path');
const router = express.Router();

// FAISS extraction endpoint
router.post('/extract', async (req, res) => {
    try {
        const { command, query, limit } = req.body;
        
        if (!command) {
            return res.status(400).json({
                error: 'Command is required'
            });
        }
        
        const pythonScriptPath = path.join(__dirname, '../python/faiss_extractor.py');
        
        // Build command arguments
        const args = [pythonScriptPath, command];
        
        if (command === 'search') {
            if (!query) {
                return res.status(400).json({
                    error: 'Query is required for search command'
                });
            }
            args.push('--query', query);
            if (limit) {
                args.push('--limit', limit.toString());
            }
        }
        
        // Execute Python script
        const pythonProcess = spawn('python', args);
        
        let output = '';
        let errorOutput = '';
        
        pythonProcess.stdout.on('data', (data) => {
            output += data.toString();
        });
        
        pythonProcess.stderr.on('data', (data) => {
            errorOutput += data.toString();
        });
        
        pythonProcess.on('close', (code) => {
            if (code === 0) {
                try {
                    const result = JSON.parse(output);
                    res.json(result);
                } catch (parseError) {
                    console.error('Failed to parse Python output:', parseError);
                    res.status(500).json({
                        error: 'Failed to parse FAISS extraction results',
                        details: parseError.message,
                        output: output
                    });
                }
            } else {
                console.error('Python script failed:', errorOutput);
                res.status(500).json({
                    error: 'FAISS extraction failed',
                    details: errorOutput,
                    code: code
                });
            }
        });
        
        pythonProcess.on('error', (error) => {
            console.error('Failed to start Python process:', error);
            res.status(500).json({
                error: 'Failed to start FAISS extraction process',
                details: error.message
            });
        });
        
        // Set timeout for the process
        setTimeout(() => {
            pythonProcess.kill();
            res.status(408).json({
                error: 'FAISS extraction timed out'
            });
        }, 30000); // 30 seconds timeout
        
    } catch (error) {
        console.error('Error in FAISS extraction endpoint:', error);
        res.status(500).json({
            error: 'Internal server error',
            details: error.message
        });
    }
});

// Get FAISS statistics endpoint
router.get('/stats', async (req, res) => {
    try {
        const pythonScriptPath = path.join(__dirname, '../python/faiss_extractor.py');
        const pythonProcess = spawn('python', [pythonScriptPath, 'stats']);
        
        let output = '';
        let errorOutput = '';
        
        pythonProcess.stdout.on('data', (data) => {
            output += data.toString();
        });
        
        pythonProcess.stderr.on('data', (data) => {
            errorOutput += data.toString();
        });
        
        pythonProcess.on('close', (code) => {
            if (code === 0) {
                try {
                    const result = JSON.parse(output);
                    res.json(result);
                } catch (parseError) {
                    res.status(500).json({
                        error: 'Failed to parse statistics results',
                        details: parseError.message
                    });
                }
            } else {
                res.status(500).json({
                    error: 'Failed to get FAISS statistics',
                    details: errorOutput
                });
            }
        });
        
    } catch (error) {
        console.error('Error in FAISS stats endpoint:', error);
        res.status(500).json({
            error: 'Internal server error',
            details: error.message
        });
    }
});

// Delete file from FAISS endpoint
router.delete('/file/:filename', async (req, res) => {
    try {
        const filename = req.params.filename;
        
        if (!filename) {
            return res.status(400).json({
                error: 'Filename is required'
            });
        }
        
        const pythonScriptPath = path.join(__dirname, '../python/faiss_extractor.py');
        const pythonProcess = spawn('python', [pythonScriptPath, 'delete', '--filename', filename]);
        
        let output = '';
        let errorOutput = '';
        
        pythonProcess.stdout.on('data', (data) => {
            output += data.toString();
        });
        
        pythonProcess.stderr.on('data', (data) => {
            errorOutput += data.toString();
        });
        
        pythonProcess.on('close', (code) => {
            if (code === 0) {
                try {
                    const result = JSON.parse(output);
                    if (result.success) {
                        res.json({
                            success: true,
                            message: result.message,
                            deletedChunks: result.deleted_chunks,
                            filename: result.filename
                        });
                    } else {
                        res.status(400).json({
                            success: false,
                            error: result.error
                        });
                    }
                } catch (parseError) {
                    console.error('Failed to parse deletion result:', parseError);
                    res.status(500).json({
                        error: 'Failed to parse deletion results',
                        details: parseError.message,
                        output: output
                    });
                }
            } else {
                console.error('Python deletion script failed:', errorOutput);
                res.status(500).json({
                    error: 'FAISS file deletion failed',
                    details: errorOutput,
                    code: code
                });
            }
        });
        
        pythonProcess.on('error', (error) => {
            console.error('Failed to start Python deletion process:', error);
            res.status(500).json({
                error: 'Failed to start file deletion process',
                details: error.message
            });
        });
        
        // Set timeout for the process
        setTimeout(() => {
            pythonProcess.kill();
            res.status(408).json({
                error: 'File deletion timed out'
            });
        }, 30000); // 30 seconds timeout
        
    } catch (error) {
        console.error('Error in FAISS file deletion endpoint:', error);
        res.status(500).json({
            error: 'Internal server error',
            details: error.message
        });
    }
});

module.exports = router;
