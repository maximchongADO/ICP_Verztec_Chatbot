const fileUploadController = require("../controllers/fileUploadController.js");
const authenticateToken = require("../middleware/authenticateToken.js");
const multer = require('multer');
const cors = require('cors');

const storage = multer.memoryStorage();
const upload = multer({ storage: storage });

const fileUploadRoute = (app) => {
  // Add CORS options specifically for file upload endpoints
  const corsOptions = {
    origin: 'http://localhost:8000',
    credentials: true,
    methods: ['POST', 'GET', 'DELETE', 'OPTIONS'],
    allowedHeaders: ['Content-Type', 'Authorization']
  };

  // Require admin or manager for all file upload endpoints
  const requireAdminOrManager = authenticateToken.requireAdminOrManager;

  // Protected chatbot endpoints - require authentication
  app.post(
    "/api/fileUpload/upload",
    cors(corsOptions),
    authenticateToken,  // First check auth
    requireAdminOrManager,        // Check admin or manager rights
    upload.single('file'),  // Then handle file
    (req, res, next) => {
      // Debug logging
      console.log("Request User:", req.user);
      console.log("File received:", req.file ? "Yes" : "No");
      console.log("Country:", req.body.country);
      console.log("Department:", req.body.department);
      next();
    },
    fileUploadController.uploadFile
  );
  
  app.get(
    "/api/fileUpload/config",
    cors(corsOptions),
    authenticateToken,
    requireAdminOrManager,
    fileUploadController.getUploadConfig
  );
  
  app.get(
    "/api/fileUpload/indices",
    cors(corsOptions),
    authenticateToken,
    requireAdminOrManager,
    fileUploadController.getAvailableIndices
  );
  
  app.get(
    "/api/fileUpload/getFile/:id",
    authenticateToken,
    requireAdminOrManager,
    fileUploadController.getFile
  );
  app.delete(
    "/api/fileUpload/deleteFile/:id",
    authenticateToken,
    requireAdminOrManager,
    fileUploadController.deleteFile
  );
};

module.exports = fileUploadRoute;

