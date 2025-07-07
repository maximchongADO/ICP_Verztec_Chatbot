require('dotenv').config();
const express = require("express");
const session = require("express-session");
const app = express();
const cors = require("cors");
const PORT = process.env.PORT || 8000;
const dbConfig = require("./backend/database/dbConfig.js");
const mysql = require("mysql2/promise");
const route = require("./backend/routes/routes.js");
const bodyParser = require("body-parser");
const path = require("path");
const fs = require("fs");

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
// Serve Python image directory to frontend
app.use('/data/images', express.static(path.join(__dirname, 'backend/python/data/images')));

// Custom document serving with proper headers
app.get('/documents/:filename', (req, res) => {
  const filename = req.params.filename;
  const ext = path.extname(filename).toLowerCase();
  
  // Define data directories
  const dataDirs = [
    path.join(__dirname, 'backend/python/data/pdf'),
    path.join(__dirname, 'backend/python/data/word'),
    path.join(__dirname, 'backend/python/data/pptx')
  ];
  
  // Find the file in one of the directories
  let filePath = null;
  for (const dir of dataDirs) {
    const fullPath = path.join(dir, filename);
    if (require('fs').existsSync(fullPath)) {
      filePath = fullPath;
      break;
    }
  }
  
  if (!filePath) {
    return res.status(404).send('File not found');
  }
  
  // Add CORS headers for external viewers
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
  
  // Set appropriate headers based on file type
  if (ext === '.pdf') {
    res.setHeader('Content-Type', 'application/pdf');
    res.setHeader('Content-Disposition', 'inline; filename="' + filename + '"');
  } else if (ext === '.docx') {
    res.setHeader('Content-Type', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document');
    res.setHeader('Content-Disposition', 'inline; filename="' + filename + '"');
  } else if (ext === '.doc') {
    res.setHeader('Content-Type', 'application/msword');
    res.setHeader('Content-Disposition', 'inline; filename="' + filename + '"');
  } else if (ext === '.pptx') {
    res.setHeader('Content-Type', 'application/vnd.openxmlformats-officedocument.presentationml.presentation');
    res.setHeader('Content-Disposition', 'inline; filename="' + filename + '"');
  } else if (ext === '.ppt') {
    res.setHeader('Content-Type', 'application/vnd.ms-powerpoint');
    res.setHeader('Content-Disposition', 'inline; filename="' + filename + '"');
  } else {
    res.setHeader('Content-Disposition', 'inline; filename="' + filename + '"');
  }
  
  // Send the file
  res.sendFile(filePath);
});

// Debug endpoint to test file serving
app.get('/test-documents', (req, res) => {
  const fs = require('fs');
  const testResults = [];
  
  const dataDirs = [
    { name: 'PDF', path: path.join(__dirname, 'backend/python/data/pdf') },
    { name: 'Word', path: path.join(__dirname, 'backend/python/data/word') },
    { name: 'PPTX', path: path.join(__dirname, 'backend/python/data/pptx') }
  ];
  
  dataDirs.forEach(dir => {
    if (fs.existsSync(dir.path)) {
      const files = fs.readdirSync(dir.path);
      testResults.push({
        directory: dir.name,
        path: dir.path,
        files: files.map(file => ({
          name: file,
          link: `/documents/${file}`
        }))
      });
    } else {
      testResults.push({
        directory: dir.name,
        path: dir.path,
        error: 'Directory not found'
      });
    }
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
