const axios = require('axios');
const FormData = require('form-data');
const { Readable } = require('stream');
const FileUpload = require('../models/fileUpload.js');

const uploadFile = async (req, res) => {
    try {
        if (!req.user) {
            return res.status(403).json({
                success: false,
                message: 'Authentication required'
            });
        }

        if (!req.file) {
            return res.status(400).json({ 
                success: false, 
                message: 'No file uploaded' 
            });
        }

        // Create form data for FastAPI
        const formData = new FormData();
        const stream = Readable.from(req.file.buffer);
        formData.append('file', stream, {
            filename: req.file.originalname,
            contentType: req.file.mimetype
        });

        console.log('Sending file to FastAPI:', req.file.originalname);

        const response = await axios.post(
            'http://localhost:3000/internal/upload',
            formData,
            {
                headers: {
                    ...formData.getHeaders(),
                    'Authorization': `Bearer ${process.env.INTERNAL_API_KEY || 'default-key'}`
                }
            }
        );

        if (!response.data.success) {
            throw new Error(response.data.error || 'FastAPI processing failed');
        }

        // Save to database with proper null checks
        const fileRecord = await FileUpload.createFileRecord({
            filename: req.file.originalname,
            cleanedContent: response.data.cleaned_content || '',  // Ensure not null
            uploadedBy: req.user.id || null  // Ensure not undefined
        });

        return res.status(200).json({
            success: true,
            message: 'File processed and stored successfully',
            fileId: fileRecord.fileId
        });

    } catch (error) {
        console.error('Error uploading file:', error);
        return res.status(500).json({ 
            success: false, 
            message: 'Failed to upload file',
            error: error.message 
        });
    }
};

const getFile = async (req, res) => {
    try {
        const fileId = req.params.id;
        const file = await FileUpload.getFileById(fileId);
        
        if (!file) {
            return res.status(404).json({
                success: false,
                message: 'File not found'
            });
        }

        res.status(200).json({
            success: true,
            data: file
        });
    } catch (error) {
        console.error('Error getting file:', error);
        res.status(500).json({
            success: false,
            message: 'Failed to get file',
            error: error.message
        });
    }
};

const deleteFile = async (req, res) => {
    try {
        const fileId = req.params.id;
        const result = await FileUpload.deleteFile(fileId);
        
        if (!result) {
            return res.status(404).json({
                success: false,
                message: 'File not found'
            });
        }

        res.status(200).json({
            success: true,
            message: 'File deleted successfully'
        });
    } catch (error) {
        console.error('Error deleting file:', error);
        res.status(500).json({
            success: false,
            message: 'Failed to delete file',
            error: error.message
        });
    }
};

module.exports = {
    uploadFile,
    getFile,
    deleteFile
};
