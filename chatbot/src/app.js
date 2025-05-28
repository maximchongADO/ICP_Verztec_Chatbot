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

app.use(cors({
  origin: ['http://localhost:8000', 'http://localhost:3000'],
  credentials: true,
  methods: ['GET', 'POST', 'PUT', 'DELETE'],
  allowedHeaders: ['Content-Type', 'Authorization']
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

app.use(bodyParser.urlencoded({ extended: true }));
app.use(bodyParser.json());

route(app);

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
