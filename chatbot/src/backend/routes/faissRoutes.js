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
        
        const pythonScriptPath = path.join(__dirname, '../python/faiss_extractor_optimized.py');
        
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
        let responseSent = false;
        
        pythonProcess.stdout.on('data', (data) => {
            output += data.toString();
        });
        
        pythonProcess.stderr.on('data', (data) => {
            errorOutput += data.toString();
        });
        
        pythonProcess.on('close', (code) => {
            if (responseSent) return;
            responseSent = true;
            
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
            if (responseSent) return;
            responseSent = true;
            
            console.error('Failed to start Python process:', error);
            res.status(500).json({
                error: 'Failed to start FAISS extraction process',
                details: error.message
            });
        });
        
        // Set timeout for the process - increased for first-time model loading
        const timeoutDuration = command === 'list' ? 120000 : 60000; // 2 minutes for list, 1 minute for others
        const timeoutId = setTimeout(() => {
            if (responseSent) return;
            responseSent = true;
            
            pythonProcess.kill();
            res.status(408).json({
                error: 'FAISS extraction timed out',
                suggestion: 'Try the warmup endpoint first: POST /api/faiss/warmup'
            });
        }, timeoutDuration);
        
        // Clear timeout if process completes normally
        pythonProcess.on('close', () => {
            clearTimeout(timeoutId);
        });
        
    } catch (error) {
        console.error('Error in FAISS extraction endpoint:', error);
        res.status(500).json({
            error: 'Internal server error',
            details: error.message
        });
    }
});

// Warmup endpoint to pre-load the embedding model
router.post('/warmup', async (req, res) => {
    try {
        const pythonScriptPath = path.join(__dirname, '../python/faiss_extractor_optimized.py');
        const pythonProcess = spawn('python', [pythonScriptPath, 'warmup']);
        
        let output = '';
        let errorOutput = '';
        let responseSent = false;
        
        pythonProcess.stdout.on('data', (data) => {
            output += data.toString();
        });
        
        pythonProcess.stderr.on('data', (data) => {
            errorOutput += data.toString();
        });
        
        pythonProcess.on('close', (code) => {
            if (responseSent) return;
            responseSent = true;
            
            if (code === 0) {
                try {
                    const result = JSON.parse(output);
                    res.json(result);
                } catch (parseError) {
                    res.status(500).json({
                        error: 'Failed to parse warmup results',
                        details: parseError.message
                    });
                }
            } else {
                res.status(500).json({
                    error: 'Failed to warm up FAISS model',
                    details: errorOutput
                });
            }
        });
        
        pythonProcess.on('error', (error) => {
            if (responseSent) return;
            responseSent = true;
            
            console.error('Error in FAISS warmup endpoint:', error);
            res.status(500).json({
                error: 'Failed to start warmup process',
                details: error.message
            });
        });
        
        // Set timeout for warmup (allow more time for first load)
        const timeoutId = setTimeout(() => {
            if (responseSent) return;
            responseSent = true;
            
            pythonProcess.kill();
            res.status(408).json({
                error: 'Warmup request timed out',
                message: 'This is normal for the first load. Try again.'
            });
        }, 150000); // 2.5 minutes timeout for warmup
        
        // Clear timeout if process completes normally
        pythonProcess.on('close', () => {
            clearTimeout(timeoutId);
        });
        
    } catch (error) {
        console.error('Error in FAISS warmup endpoint:', error);
        res.status(500).json({
            error: 'Internal server error',
            details: error.message
        });
    }
});

// Get FAISS statistics endpoint
router.get('/stats', async (req, res) => {
    try {
        const pythonScriptPath = path.join(__dirname, '../python/faiss_extractor_optimized.py');
        const pythonProcess = spawn('python', [pythonScriptPath, 'stats']);
        
        let output = '';
        let errorOutput = '';
        let responseSent = false;
        
        pythonProcess.stdout.on('data', (data) => {
            output += data.toString();
        });
        
        pythonProcess.stderr.on('data', (data) => {
            errorOutput += data.toString();
        });
        
        pythonProcess.on('close', (code) => {
            if (responseSent) return;
            responseSent = true;
            
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
        
        pythonProcess.on('error', (error) => {
            if (responseSent) return;
            responseSent = true;
            
            console.error('Error in FAISS stats endpoint:', error);
            res.status(500).json({
                error: 'Failed to start statistics process',
                details: error.message
            });
        });
        
        // Set timeout for the process
        const timeoutId = setTimeout(() => {
            if (responseSent) return;
            responseSent = true;
            
            pythonProcess.kill();
            res.status(408).json({
                error: 'Statistics request timed out',
                suggestion: 'Try the warmup endpoint first: POST /api/faiss/warmup'
            });
        }, 120000); // 2 minutes timeout for stats (first time may need model loading)
        
        // Clear timeout if process completes normally
        pythonProcess.on('close', () => {
            clearTimeout(timeoutId);
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
        let responseSent = false;
        
        pythonProcess.stdout.on('data', (data) => {
            output += data.toString();
        });
        
        pythonProcess.stderr.on('data', (data) => {
            errorOutput += data.toString();
        });
        
        pythonProcess.on('close', (code) => {
            if (responseSent) return;
            responseSent = true;
            
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
            if (responseSent) return;
            responseSent = true;
            
            console.error('Failed to start Python deletion process:', error);
            res.status(500).json({
                error: 'Failed to start file deletion process',
                details: error.message
            });
        });
        
        // Set timeout for the process
        const timeoutId = setTimeout(() => {
            if (responseSent) return;
            responseSent = true;
            
            pythonProcess.kill();
            res.status(408).json({
                error: 'File deletion timed out'
            });
        }, 60000); // 60 seconds timeout (increased from 30)
        
        // Clear timeout if process completes normally
        pythonProcess.on('close', () => {
            clearTimeout(timeoutId);
        });
        
    } catch (error) {
        console.error('Error in FAISS file deletion endpoint:', error);
        res.status(500).json({
            error: 'Internal server error',
            details: error.message
        });
    }
});

module.exports = router;
