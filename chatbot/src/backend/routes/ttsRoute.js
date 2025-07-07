const ttsController = require("../controllers/ttsController.js");
const ttsEnhanced = require("../controllers/ttsControllerEnhanced.js");
const { processAudioWithLipSync, getLipSyncData } = require("../controllers/rhubarbController.js");
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

  // **Enhanced TTS Endpoints (following reference implementation pattern)**
  
  // Enhanced TTS synthesis with comprehensive lip sync support
  app.post(
    "/api/tts/chat",
    authenticateToken,
    ttsEnhanced.chatWithLipSync
  );

  // Enhanced single message TTS with lip sync (replaces standard synthesize)
  app.post(
    "/api/tts/synthesize-enhanced",
    authenticateToken,
    ttsEnhanced.synthesizeSpeechWithLipSync
  );

  // Test enhanced endpoints without authentication
  app.post("/api/tts/chat-test", ttsEnhanced.chatWithLipSync);
  app.post("/api/tts/synthesize-enhanced-test", ttsEnhanced.synthesizeSpeechWithLipSync);

  // **Original TTS Endpoints (backward compatibility)**
  
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

  // **Rhubarb Lip Sync Routes**
  
  // Process audio file with lip sync generation
  app.post(
    "/api/tts/lipsync/generate",
    authenticateToken,
    processAudioWithLipSync
  );

  // Get existing lip sync data by filename
  app.get(
    "/api/tts/lipsync/:filename",
    authenticateToken,
    getLipSyncData
  );

  // Test lip sync endpoint without authentication for debugging
  app.post("/api/tts/lipsync/generate-test", processAudioWithLipSync);
};

module.exports = ttsRoute;
