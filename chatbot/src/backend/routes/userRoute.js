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

  console.log("User routes mounted successfully");
};

module.exports = userRoute;
