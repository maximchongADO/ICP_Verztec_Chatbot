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

        // Get country and department from form data
        const country = req.body.country;
        const department = req.body.department;
        
        if (!country || !department) {
            return res.status(400).json({
                success: false,
                message: 'Country and department are required'
            });
        }

        // Create form data for FastAPI
        const formData = new FormData();
        const stream = Readable.from(req.file.buffer);
        formData.append('file', stream, {
            filename: req.file.originalname,
            contentType: req.file.mimetype
        });
        formData.append('country', country);
        formData.append('department', department);

        console.log(`Sending file to FastAPI: ${req.file.originalname} for ${country}/${department}`);

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
            uploadedBy: req.user.id || null,  // Ensure not undefined
            country: country,
            department: department,
            faissIndexPath: response.data.faiss_index_path || null
        });

        return res.status(200).json({
            success: true,
            message: `File processed and stored successfully in ${country.toUpperCase()}/${department.toUpperCase()} knowledge base`,
            fileId: fileRecord.fileId,
            country: country,
            department: department,
            faissIndexPath: response.data.faiss_index_path
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

const getUploadConfig = async (req, res) => {
    try {
        // Get configuration from FastAPI
        const response = await axios.get(
            'http://localhost:3000/internal/upload/config',
            {
                headers: {
                    'Authorization': `Bearer ${process.env.INTERNAL_API_KEY || 'default-key'}`
                }
            }
        );

        if (response.data.success) {
            return res.status(200).json(response.data);
        } else {
            // Fallback to default configuration
            return res.status(200).json({
                success: true,
                supported_countries: ['china', 'singapore'],
                supported_departments: ['hr', 'it'],
                supported_file_types: ['.pdf', '.doc', '.docx', '.txt', '.pptx'],
                total_combinations: 4
            });
        }
    } catch (error) {
        console.warn('Could not get config from FastAPI, using defaults:', error.message);
        // Return default configuration
        return res.status(200).json({
            success: true,
            supported_countries: ['china', 'singapore'],
            supported_departments: ['hr', 'it'],
            supported_file_types: ['.pdf', '.doc', '.docx', '.txt', '.pptx'],
            total_combinations: 4
        });
    }
};

const getAvailableIndices = async (req, res) => {
    try {
        const response = await axios.get(
            'http://localhost:3000/internal/upload/indices',
            {
                headers: {
                    'Authorization': `Bearer ${process.env.INTERNAL_API_KEY || 'default-key'}`
                }
            }
        );

        return res.status(200).json(response.data);
    } catch (error) {
        console.error('Error getting available indices:', error);
        return res.status(500).json({
            success: false,
            message: 'Failed to get available indices',
            error: error.message
        });
    }
};

module.exports = {
    uploadFile,
    getFile,
    deleteFile,
    getUploadConfig,
    getAvailableIndices
};
