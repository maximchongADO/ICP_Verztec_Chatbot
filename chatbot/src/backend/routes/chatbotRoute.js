const chatbotController = require("../controllers/chatbotController.js");
const authenticateToken = require("../middleware/authenticateToken.js");

const chatbotRoute = (app) => {
  // Protected chatbot endpoints - require authentication
  app.post(
    "/api/chatbot/message",
    authenticateToken,
    chatbotController.processMessage
  );
  app.get(
    "/api/chatbot/history",
    authenticateToken,
    chatbotController.getChatHistory
  );
  app.delete(
    "/api/chatbot/history",
    authenticateToken,
    chatbotController.clearChatHistory
  );
};

module.exports = chatbotRoute;
