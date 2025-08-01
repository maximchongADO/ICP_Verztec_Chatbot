
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
  // Add support for /api/chatbot/history/:chat_id
  app.get(
    "/api/chatbot/history/:chat_id",
    authenticateToken,
    chatbotController.getChatHistory
  );
  // Support POST for clearing chat history (for frontend compatibility)
  app.post(
    "/api/chatbot/history",
    authenticateToken,
    chatbotController.clearChatHistory
  );
  app.delete(
    "/api/chatbot/history",
    authenticateToken,
    chatbotController.clearChatHistory
  );
  // Add DELETE endpoint for specific chat_id
  app.delete(
    "/api/chatbot/history/:chat_id",
    authenticateToken,
    chatbotController.deleteChatById
  );
  // Add new feedback endpoint
  app.post(
    "/api/chatbot/feedback",
    authenticateToken,
    chatbotController.handleFeedback
  );
  // New Chat endpoint
  app.post(
    "/api/chatbot/newchat",
    authenticateToken,
    chatbotController.newChat
  );
  // Avatar chatbot endpoint
  app.post("/chatbot_avatar", chatbotController.processAvatarMessage);

  // Test avatar endpoint without authentication
  app.post(
    "/chatbot_avatar_test",
    chatbotController.processAvatarMessage
  );
};

module.exports = chatbotRoute;
