const userController = require("../controllers/userController.js");
const authenticateToken = require("../middleware/authenticateToken.js");
require("dotenv").config();

const userRoute = (app) => {
  // User registration route
  app.post("/users", userController.createUser);
  // User login route - matches frontend expectation
  app.post("/api/login", userController.loginUser);
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
  // Admin-only: company-wide analytics
  app.get(
    "/api/users/company-analytics",
    authenticateToken,
    authenticateToken.requireAdmin,
    userController.getCompanyAnalytics
  );
  // Admin-only: all users' analytics
  app.get(
    "/api/users/all-analytics",
    authenticateToken,
    authenticateToken.requireAdmin,
    userController.getAllUsersAnalytics
  );

  console.log("User routes mounted successfully");
};

module.exports = userRoute;
