const axios = require('axios');
const FormData = require('form-data');
const { Readable } = require('stream');
const FileUpload = require('../models/fileUpload.js');

// Admin single file upload to all indices function
const handleAdminSingleFileToAllIndices = async (req, res) => {
    try {
        const results = {
            success: true,
            message: 'Admin file upload to all indices completed',
            filename: req.file.originalname,
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

        console.log(`Admin single file upload: Processing file ${req.file.originalname} for ${combinations.length + 1} indices (including admin master)`);

        // First upload to admin master index
        try {
            const fileBuffer = Buffer.from(req.file.buffer);
            const formData = new FormData();
            const stream = Readable.from(fileBuffer);
            formData.append('file', stream, {
                filename: req.file.originalname,
                contentType: req.file.mimetype
            });
            formData.append('country', 'admin');
            formData.append('department', 'master');
            formData.append('replaceIfExists', 'true');

            console.log('Uploading to admin master index first...');

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
                results.uploadResults.push({
                    country: 'admin',
                    department: 'master',
                    success: true,
                    message: 'Successfully uploaded to ADMIN MASTER index'
                });
                results.successfulUploads++;
            } else {
                throw new Error(response.data.error || 'FastAPI processing failed');
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

        // Then upload to each specific combination
        for (const combo of combinations) {
            try {
                // Create a fresh buffer for each upload
                const fileBuffer = Buffer.from(req.file.buffer);
                const formData = new FormData();
                const stream = Readable.from(fileBuffer);
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
                    results.uploadResults.push({
                        country: combo.country,
                        department: combo.department,
                        success: true,
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

        // Update overall success status
        results.success = results.failedUploads === 0;
        if (results.failedUploads > 0) {
            results.message = `Admin file upload completed with ${results.failedUploads} failures out of ${results.totalUploads} total uploads`;
        }

        return res.status(results.success ? 200 : 207).json(results); // 207 = Multi-Status

    } catch (error) {
        console.error('Error in admin single file upload:', error);
        return res.status(500).json({
            success: false,
            message: 'Admin file upload failed',
            error: error.message,
            filename: req.file ? req.file.originalname : 'unknown'
        });
    }
};

// Handle single file upload to specific index with admin master replication
const handleSpecificIndexUpload = async (req, res) => {
    try {
        const results = {
            success: true,
            message: 'File uploaded successfully',
            filename: req.file.originalname,
            uploadResults: [],
            errors: [],
            totalUploads: 0,
            successfulUploads: 0,
            failedUploads: 0
        };

        const country = req.body.country;
        const department = req.body.department;
        const userRole = req.user.user_type;

        console.log(`Specific index upload: Processing file ${req.file.originalname} for ${country}/${department}`);

        // First upload to the intended index
        try {
            const fileBuffer = Buffer.from(req.file.buffer);
            const formData = new FormData();
            const stream = Readable.from(fileBuffer);
            formData.append('file', stream, {
                filename: req.file.originalname,
                contentType: req.file.mimetype
            });
            formData.append('country', country);
            formData.append('department', department);
            formData.append('replaceIfExists', userRole === 'admin' ? 'true' : 'false');

            console.log(`Uploading to ${country}/${department}...`);

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
                results.uploadResults.push({
                    country: country,
                    department: department,
                    success: true,
                    message: `Successfully uploaded to ${country.toUpperCase()}/${department.toUpperCase()}`
                });
                results.successfulUploads++;
            } else {
                throw new Error(response.data.error || 'FastAPI processing failed');
            }
        } catch (error) {
            console.error(`Error uploading to ${country}/${department}:`, error.message);
            results.uploadResults.push({
                country: country,
                department: department,
                success: false,
                error: error.message,
                message: `Failed to upload to ${country.toUpperCase()}/${department.toUpperCase()}`
            });
            results.errors.push(`${country}/${department}: ${error.message}`);
            results.failedUploads++;
        }
        results.totalUploads++;

        // Then upload to admin master index (unless it's already admin/master)
        if (!(country === 'admin' && department === 'master')) {
            try {
                const fileBuffer = Buffer.from(req.file.buffer);
                const formData = new FormData();
                const stream = Readable.from(fileBuffer);
                formData.append('file', stream, {
                    filename: req.file.originalname,
                    contentType: req.file.mimetype
                });
                formData.append('country', 'admin');
                formData.append('department', 'master');
                formData.append('replaceIfExists', 'true');

                console.log('Also uploading to admin master index...');

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
                    results.uploadResults.push({
                        country: 'admin',
                        department: 'master',
                        success: true,
                        message: 'Successfully uploaded to ADMIN MASTER index'
                    });
                    results.successfulUploads++;
                } else {
                    throw new Error(response.data.error || 'FastAPI processing failed');
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
        }

        // Update overall success status
        results.success = results.failedUploads === 0;
        if (results.failedUploads > 0) {
            results.message = `File upload completed with ${results.failedUploads} failures out of ${results.totalUploads} total uploads`;
        }

        return res.status(results.success ? 200 : 207).json(results); // 207 = Multi-Status

    } catch (error) {
        console.error('Error in specific index upload:', error);
        return res.status(500).json({
            success: false,
            message: 'File upload failed',
            error: error.message,
            filename: req.file ? req.file.originalname : 'unknown'
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
        
        // Get country and department from form data (needed for condition check)
        const country = req.body.country;
        const department = req.body.department;

        // For admin bulk uploads: upload to all indices (including admin master)
        if (uploadMode === 'all' && userRole === 'admin') {
            return await handleAdminSingleFileToAllIndices(req, res);
        }

        // For specific uploads: upload to intended index + admin master
        if (uploadMode === 'specific') {
            return await handleSpecificIndexUpload(req, res);
        }

        if (!country || !department) {
            return res.status(400).json({
                success: false,
                message: 'Country and department are required'
            });
        }        // Upload to the selected index
        const fileBuffer = Buffer.from(req.file.buffer);
        const formData = new FormData();
        const stream = Readable.from(fileBuffer);
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

        // Always also upload to admin master index (unless already uploading to admin/master)
        let adminUploadResult = null;
        if (!(country === 'admin' && department === 'master')) {
            try {
                const adminFileBuffer = Buffer.from(req.file.buffer);
                const adminFormData = new FormData();
                const adminStream = Readable.from(adminFileBuffer);
                adminFormData.append('file', adminStream, {
                    filename: req.file.originalname,
                    contentType: req.file.mimetype
                });
                adminFormData.append('country', 'admin');
                adminFormData.append('department', 'master');
                adminFormData.append('replaceIfExists', 'true');

                console.log('Also uploading to admin master index...');

                const adminResponse = await axios.post(
                    'http://localhost:3000/internal/upload',
                    adminFormData,
                    {
                        headers: {
                            ...adminFormData.getHeaders(),
                            'Authorization': `Bearer ${process.env.INTERNAL_API_KEY || 'default-key'}`
                        }
                    }
                );
                adminUploadResult = adminResponse.data;
            } catch (adminError) {
                console.error('Error uploading to admin master:', adminError.message);
                adminUploadResult = { success: false, error: adminError.message };
            }
        }

        // File record DB logic removed
        return res.status(200).json({
            success: true,
            message: `File processed and stored successfully in ${country.toUpperCase()}/${department.toUpperCase()} knowledge base` +
                ((adminUploadResult && adminUploadResult.success) ? ' and ADMIN MASTER index' : ''),
            country: country,
            department: department,
            faissIndexPath: response.data.faiss_index_path,
            adminMasterUpload: adminUploadResult
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
    // File DB logic removed
    return res.status(501).json({
        success: false,
        message: 'File DB logic removed'
    });
};

const deleteFile = async (req, res) => {
    // File DB logic removed
    return res.status(501).json({
        success: false,
        message: 'File DB logic removed'
    });
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
    handleAdminSingleFileToAllIndices,
    handleSpecificIndexUpload,
    getFile,
    deleteFile,
    getUploadConfig,
    getAvailableIndices
};
