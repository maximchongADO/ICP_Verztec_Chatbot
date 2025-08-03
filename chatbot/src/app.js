require('dotenv').config();
const express = require("express");
const session = require("express-session");
const app = express();
const cors = require("cors");
const fs = require("fs");
const path = require("path");
const PORT = process.env.PORT || 8000;
const dbConfig = require("./backend/database/dbConfig.js");
const mysql = require("mysql2/promise");
const route = require("./backend/routes/routes.js");
const bodyParser = require("body-parser");

app.use(cors({
  origin: 'http://localhost:8000',
  credentials: true,
  methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Authorization', 'Accept'],
  exposedHeaders: ['*'],
  maxAge: 3600
}));

const staticMiddleware = express.static("public");
app.use(staticMiddleware);

app.use(
  session({
    secret: "your-secret-key",
    resave: false,
    saveUninitialized: true,
    cookie: { secure: false },
  })
);
app.use("/avatar", express.static(path.join(__dirname, "public/avatar")));
// Always return index.html for any /avatar/* route (for React Router)

app.use(bodyParser.urlencoded({ extended: true }));
app.use(bodyParser.json());

route(app);

// Internal API routes for Python FastAPI communication
app.use('/internal', require('./backend/routes/internalRoutes.js'));

// Serve Python image directory to frontend
app.use('/data/images', express.static(path.join(__dirname, 'backend/python/data/images')));

// Custom document serving with proper headers
app.get('/documents/:filename', (req, res) => {
  const filename = req.params.filename;
  const decodedFilename = decodeURIComponent(filename);
  const ext = path.extname(decodedFilename).toLowerCase();
  
  console.log(`Document request for: ${decodedFilename}`);
  
  // Define data directories
  const dataDirs = [
    path.join(__dirname, 'backend/python/data/pdf'),
    path.join(__dirname, 'backend/python/data/word'),
    path.join(__dirname, 'backend/python/data/pptx')
  ];
  
  // Find the file in one of the directories
  let filePath = null;
  let foundDir = null;
  
  for (const dir of dataDirs) {
    try {
      if (fs.existsSync(dir)) {
        const files = fs.readdirSync(dir);
        
        // First try exact match
        const exactMatch = files.find(file => file.toLowerCase() === decodedFilename.toLowerCase());
        if (exactMatch) {
          const fullPath = path.join(dir, exactMatch);
          if (fs.existsSync(fullPath)) {
            filePath = fullPath;
            foundDir = dir;
            console.log(`Found exact match: ${filePath}`);
            break;
          }
        }
        
        // Then try partial match
        const partialMatch = files.find(file => 
          file.toLowerCase().includes(decodedFilename.toLowerCase()) ||
          decodedFilename.toLowerCase().includes(file.toLowerCase())
        );
        if (partialMatch) {
          const fullPath = path.join(dir, partialMatch);
          if (fs.existsSync(fullPath)) {
            filePath = fullPath;
            foundDir = dir;
            console.log(`Found partial match: ${filePath}`);
            break;
          }
        }
      }
    } catch (error) {
      console.error(`Error checking directory ${dir}:`, error);
    }
  }
  
  if (!filePath) {
    console.error(`File not found: ${decodedFilename}`);
    return res.status(404).json({ 
      error: 'File not found',
      filename: decodedFilename,
      searched_directories: dataDirs.map(dir => ({
        path: dir,
        exists: fs.existsSync(dir)
      }))
    });
  }
  
  // Verify file exists and is readable
  let stats;
  try {
    stats = fs.statSync(filePath);
    if (!stats.isFile()) {
      return res.status(404).json({ error: 'Path is not a file' });
    }
  } catch (error) {
    console.error(`Error accessing file ${filePath}:`, error);
    return res.status(500).json({ error: 'File access error' });
  }
  
  // Add CORS headers for external viewers
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
  
  // Set appropriate headers based on file type
  const mimeTypes = {
    '.pdf': 'application/pdf',
    '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    '.doc': 'application/msword',
    '.pptx': 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
    '.ppt': 'application/vnd.ms-powerpoint',
    '.txt': 'text/plain',
    '.rtf': 'application/rtf'
  };
  
  const contentType = mimeTypes[ext] || 'application/octet-stream';
  res.setHeader('Content-Type', contentType);
  res.setHeader('Content-Disposition', `inline; filename="${path.basename(filePath)}"`);
  
  // Add cache headers for better performance
  res.setHeader('Cache-Control', 'public, max-age=3600'); // Cache for 1 hour
  res.setHeader('ETag', `"${stats.mtime.getTime()}-${stats.size}"`);
  
  // Handle range requests for large files
  const range = req.headers.range;
  if (range) {
    const parts = range.replace(/bytes=/, "").split("-");
    const start = parseInt(parts[0], 10);
    const end = parts[1] ? parseInt(parts[1], 10) : stats.size - 1;
    const chunkSize = (end - start) + 1;
    
    res.status(206);
    res.setHeader('Content-Range', `bytes ${start}-${end}/${stats.size}`);
    res.setHeader('Accept-Ranges', 'bytes');
    res.setHeader('Content-Length', chunkSize);
    
    const stream = fs.createReadStream(filePath, { start, end });
    stream.pipe(res);
  } else {
    // Send the whole file
    res.sendFile(filePath, (err) => {
      if (err) {
        console.error(`Error sending file ${filePath}:`, err);
        if (!res.headersSent) {
          res.status(500).json({ error: 'Error sending file' });
        }
      } else {
        console.log(`Successfully served file: ${filePath}`);
      }
    });
  }
});

// Debug endpoint to test file serving
app.get('/test-documents', (req, res) => {
  const testResults = {
    timestamp: new Date().toISOString(),
    server_info: {
      node_version: process.version,
      platform: process.platform,
      working_directory: process.cwd(),
      server_directory: __dirname
    },
    directories: []
  };
  
  const dataDirs = [
    { name: 'PDF', path: path.join(__dirname, 'backend/python/data/pdf') },
    { name: 'Word', path: path.join(__dirname, 'backend/python/data/word') },
    { name: 'PPTX', path: path.join(__dirname, 'backend/python/data/pptx') }
  ];
  
  dataDirs.forEach(dir => {
    const dirResult = {
      directory: dir.name,
      path: dir.path,
      exists: fs.existsSync(dir.path)
    };
    
    if (dirResult.exists) {
      try {
        const files = fs.readdirSync(dir.path);
        dirResult.file_count = files.length;
        dirResult.files = files.map(file => {
          const filePath = path.join(dir.path, file);
          const stats = fs.statSync(filePath);
          return {
            name: file,
            size: stats.size,
            modified: stats.mtime.toISOString(),
            link: `/documents/${encodeURIComponent(file)}`,
            test_link: `http://localhost:${PORT}/documents/${encodeURIComponent(file)}`
          };
        });
      } catch (error) {
        dirResult.error = error.message;
        dirResult.files = [];
      }
    } else {
      dirResult.error = 'Directory not found';
      dirResult.files = [];
    }
    
    testResults.directories.push(dirResult);
  });
  
  res.json(testResults);
});

// Serve static files from the public directory
app.listen(PORT, async () => {
  let connection;
  try {
    // Validate database configuration
    if (!dbConfig || !dbConfig.host) {
      throw new Error(
        "Database configuration is missing or invalid. Please check dbConfig.js"
      );
    }

    // Test database connection
    connection = await mysql.createConnection(dbConfig);
    console.log("Connected to MySQL database successfully");
    await connection.end();
  } catch (error) {
    console.error("Database connection error:", error.message);
    console.log(
      "Please ensure your MySQL configuration is correct in ./backend/database/dbConfig.js"
    );
    process.exit(1);
  }
  console.log(`Server is running on port ${PORT}`);
});

// Clean up on app termination
process.on("SIGINT", () => {
  console.log("Server shutting down...");
  process.exit(0);
});
