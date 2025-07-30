const express = require('express');
const router = express.Router();
const { transcribeAudio, transcribeStream, getSupportedLanguages } = require('../controllers/sttController');

// POST /api/stt/transcribe - Transcribe audio data
router.post('/transcribe', transcribeAudio);

// POST /api/stt/stream - Set up streaming transcription (for future implementation)
router.post('/stream', transcribeStream);

// GET /api/stt/languages - Get supported languages and models
router.get('/languages', getSupportedLanguages);

module.exports = router;
