const userController = require("../controllers/userController.js");
const batchUploadController = require("../controllers/batchUploadController.js");
const authenticateToken = require("../middleware/authenticateToken.js");
require("dotenv").config();

const userRoute = (app) => {
  // User registration route
  app.post("/users", userController.createUser);
  // User login route - matches frontend expectation
  app.post("/api/login", userController.loginUser);
  // JWT decode route
  app.get("/api/decode-jwt", userController.decodeJWT);
  // Get all users route
  app.get("/users", userController.getAllUsers);
  // Get user by ID route
  app.get("/users/:id", userController.getUserById);
  // Admin-only: create user (POST /api/users)
  app.post(
    "/api/users",
    authenticateToken,
    authenticateToken.requireAdmin,
    userController.adminCreateUser
  );
  // Admin-only: update user (PATCH /api/users/:id)
  app.patch(
    "/api/users/:id",
    authenticateToken,
    authenticateToken.requireAdmin,
    userController.adminUpdateUser
  );
  // Admin-only: delete user (DELETE /api/users/:id)
  app.delete(
    "/api/users/:id",
    authenticateToken,
    authenticateToken.requireAdmin,
    userController.adminDeleteUser
  );
  
  // Manager-only: get users in same department/country (GET /api/manager/users)
  app.get(
    "/api/manager/users",
    authenticateToken,
    authenticateToken.requireManager,
    userController.managerGetUsers
  );
  
  // Manager-only: create user in same department/country (POST /api/manager/users)
  app.post(
    "/api/manager/users",
    authenticateToken,
    authenticateToken.requireManager,
    userController.managerCreateUser
  );
  
  // Manager-only: update user in same department/country (PATCH /api/manager/users/:id)
  app.patch(
    "/api/manager/users/:id",
    authenticateToken,
    authenticateToken.requireManager,
    userController.managerUpdateUser
  );
  
  // Manager-only: delete user in same department/country (DELETE /api/manager/users/:id)
  app.delete(
    "/api/manager/users/:id",
    authenticateToken,
    authenticateToken.requireManager,
    userController.managerDeleteUser
  );
  
  // Admin-only: batch upload users via Excel/CSV
  app.post(
    "/api/users/batch-upload",
    authenticateToken,
    authenticateToken.requireAdmin,
    batchUploadController.upload.single("file"),
    batchUploadController.batchUploadUsers
  );
  // Admin-only: generate sample Excel file
  app.get(
    "/api/users/sample-file",
    authenticateToken,
    authenticateToken.requireAdmin,
    batchUploadController.generateSampleFile
  );
  // User analytics dashboard (authenticated)
  app.get(
    "/api/users/analytics",
    authenticateToken,
    userController.getUserAnalytics
  );
  // Recent chats for analytics
  app.get(
    "/api/users/chats",
    authenticateToken,
    userController.getUserChats
  );
  // Recent feedback for analytics
  app.get(
    "/api/users/feedback",
    authenticateToken,
    userController.getUserFeedback
  );
  // Admin or Manager: company-wide analytics
  app.get(
    "/api/users/company-analytics",
    authenticateToken,
    authenticateToken.requireAdminOrManager,
    userController.getCompanyAnalytics
  );
  // Admin or Manager: all users' analytics
  app.get(
    "/api/users/all-analytics",
    authenticateToken,
    authenticateToken.requireAdminOrManager,
    userController.getAllUsersAnalytics
  );

  console.log("User routes mounted successfully");
};

module.exports = userRoute;
