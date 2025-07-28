const axios = require('axios');
const FormData = require('form-data');
const { Readable } = require('stream');
const FileUpload = require('../models/fileUpload.js');

// Admin bulk upload function
const handleAdminBulkUpload = async (req, res) => {
    try {
        const results = {
            success: true,
            message: 'Admin bulk upload completed',
            uploadResults: [],
            errors: [],
            totalUploads: 0,
            successfulUploads: 0,
            failedUploads: 0
        };

        // Define all country/department combinations
        const combinations = [
            { country: 'china', department: 'hr' },
            { country: 'china', department: 'it' },
            { country: 'singapore', department: 'hr' },
            { country: 'singapore', department: 'it' }
        ];

        console.log(`Admin bulk upload: Processing file ${req.file.originalname} for ${combinations.length} indices`);

        // Upload to each combination
        for (const combo of combinations) {
            try {
                // Create form data for FastAPI
                const formData = new FormData();
                const stream = Readable.from(req.file.buffer);
                formData.append('file', stream, {
                    filename: req.file.originalname,
                    contentType: req.file.mimetype
                });
                formData.append('country', combo.country);
                formData.append('department', combo.department);
                formData.append('replaceIfExists', 'true'); // Admin can overwrite

                console.log(`Uploading to ${combo.country}/${combo.department}...`);

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

                if (response.data.success) {
                    // Save to database
                    const fileRecord = await FileUpload.createFileRecord({
                        filename: req.file.originalname,
                        cleanedContent: response.data.cleaned_content || '',
                        uploadedBy: req.user.id || null,
                        country: combo.country,
                        department: combo.department,
                        faissIndexPath: response.data.faiss_index_path || null
                    });

                    results.uploadResults.push({
                        country: combo.country,
                        department: combo.department,
                        success: true,
                        fileId: fileRecord.fileId,
                        message: `Successfully uploaded to ${combo.country.toUpperCase()}/${combo.department.toUpperCase()}`
                    });
                    results.successfulUploads++;
                } else {
                    throw new Error(response.data.error || 'FastAPI processing failed');
                }

            } catch (error) {
                console.error(`Error uploading to ${combo.country}/${combo.department}:`, error.message);
                results.uploadResults.push({
                    country: combo.country,
                    department: combo.department,
                    success: false,
                    error: error.message,
                    message: `Failed to upload to ${combo.country.toUpperCase()}/${combo.department.toUpperCase()}`
                });
                results.errors.push(`${combo.country}/${combo.department}: ${error.message}`);
                results.failedUploads++;
            }
            results.totalUploads++;
        }

        // Also upload to admin master index
        try {
            const formData = new FormData();
            const stream = Readable.from(req.file.buffer);
            formData.append('file', stream, {
                filename: req.file.originalname,
                contentType: req.file.mimetype
            });
            formData.append('country', 'admin');
            formData.append('department', 'master');
            formData.append('replaceIfExists', 'true');

            console.log('Uploading to admin master index...');

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

            if (response.data.success) {
                const fileRecord = await FileUpload.createFileRecord({
                    filename: req.file.originalname,
                    cleanedContent: response.data.cleaned_content || '',
                    uploadedBy: req.user.id || null,
                    country: 'admin',
                    department: 'master',
                    faissIndexPath: response.data.faiss_index_path || null
                });

                results.uploadResults.push({
                    country: 'admin',
                    department: 'master',
                    success: true,
                    fileId: fileRecord.fileId,
                    message: 'Successfully uploaded to ADMIN MASTER index'
                });
                results.successfulUploads++;
            }
        } catch (error) {
            console.error('Error uploading to admin master:', error.message);
            results.uploadResults.push({
                country: 'admin',
                department: 'master',
                success: false,
                error: error.message,
                message: 'Failed to upload to ADMIN MASTER index'
            });
            results.errors.push(`admin/master: ${error.message}`);
            results.failedUploads++;
        }
        results.totalUploads++;

        // Update overall success status
        results.success = results.failedUploads === 0;
        if (results.failedUploads > 0) {
            results.message = `Admin bulk upload completed with ${results.failedUploads} failures out of ${results.totalUploads} total uploads`;
        }

        return res.status(results.success ? 200 : 207).json(results); // 207 = Multi-Status

    } catch (error) {
        console.error('Error in admin bulk upload:', error);
        return res.status(500).json({
            success: false,
            message: 'Admin bulk upload failed',
            error: error.message
        });
    }
};

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

        // Check if this is an admin bulk upload
        const uploadMode = req.body.uploadMode;
        const userRole = req.user.user_type;

        // For admin bulk uploads
        if (uploadMode === 'all' && userRole === 'admin') {
            return await handleAdminBulkUpload(req, res);
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
    handleAdminBulkUpload,
    getFile,
    deleteFile,
    getUploadConfig,
    getAvailableIndices
};
