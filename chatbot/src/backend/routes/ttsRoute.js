const ttsController = require("../controllers/ttsController.js");
const authenticateToken = require("../middleware/authenticateToken.js");

const ttsRoute = (app) => {
  // Test endpoint without authentication
  app.get("/api/tts/test", (req, res) => {
    res.json({ 
      success: true, 
      message: "TTS endpoint is working without authentication",
      timestamp: new Date().toISOString()
    });
  });

  // Debug endpoint to test authentication
  app.post("/api/tts/debug-auth", (req, res) => {
    const authHeader = req.headers["authorization"];
    const token = authHeader && authHeader.split(" ")[1];
    
    console.log('TTS Debug - Auth header:', authHeader);
    console.log('TTS Debug - Token:', token);
    
    res.json({
      success: true,
      message: "Debug auth endpoint reached",
      hasAuthHeader: !!authHeader,
      hasToken: !!token,
      tokenPreview: token ? token.substring(0, 20) + '...' : 'No token'
    });
  });

  // Temporary TTS synthesis endpoint WITHOUT authentication for testing
  app.post("/api/tts/synthesize-test", ttsController.synthesizeSpeech);

  // TTS synthesis endpoint - requires authentication
  app.post(
    "/api/tts/synthesize",
    authenticateToken,
    ttsController.synthesizeSpeech
  );

  // Get available voices endpoint - requires authentication
  app.get(
    "/api/tts/voices",
    authenticateToken,
    ttsController.getVoices
  );
};

module.exports = ttsRoute;
