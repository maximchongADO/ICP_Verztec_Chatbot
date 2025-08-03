const speech = require('@google-cloud/speech');
const path = require('path');
const fs = require('fs');
require('dotenv').config();

// Initialize the Google Cloud Speech client with the same authentication as TTS
const speechClient = new speech.SpeechClient({
  keyFilename: path.resolve(__dirname, 'service-account-key.json'), // Same key as TTS
  projectId: 'golden-frame-461314-e6' // Same project as TTS
});

// Function to transcribe audio from base64 data
const transcribeAudio = async (req, res) => {
  try {
    const { 
      audioData, 
      languageCode = 'en-GB', 
      encoding = 'WEBM_OPUS',
      sampleRateHertz = 48000,
      enableAutomaticPunctuation = true,
      model = 'latest_long'
    } = req.body;

    if (!audioData) {
      return res.status(400).json({
        success: false,
        error: 'Audio data is required'
      });
    }

    console.log('STT Request:', { 
      languageCode, 
      encoding, 
      sampleRateHertz,
      audioDataLength: audioData.length 
    });

    // Configure the recognition request
    const request = {
      audio: {
        content: audioData, // Base64 encoded audio
      },
      config: {
        encoding: encoding,
        sampleRateHertz: sampleRateHertz,
        languageCode: languageCode,
        enableAutomaticPunctuation: enableAutomaticPunctuation,
        model: model,
        useEnhanced: true, // Use enhanced model for better accuracy
        // Alternative language codes for better recognition
        alternativeLanguageCodes: ['en-US', 'en-AU'],
        // Enable word time offsets if needed
        enableWordTimeOffsets: false,
        // Enable speaker diarization for multiple speakers
        enableSpeakerDiarization: false,
        // Profanity filter
        profanityFilter: false,
      },
    };

    // Perform the speech recognition
    const [response] = await speechClient.recognize(request);
    
    if (!response.results || response.results.length === 0) {
      return res.json({
        success: true,
        transcript: '',
        confidence: 0,
        message: 'No speech detected'
      });
    }

    // Get the transcription from the first result
    const transcription = response.results
      .map(result => result.alternatives[0])
      .filter(alternative => alternative.transcript)
      .map(alternative => ({
        transcript: alternative.transcript,
        confidence: alternative.confidence || 0
      }));

    const finalTranscript = transcription
      .map(item => item.transcript)
      .join(' ')
      .trim();

    const averageConfidence = transcription.length > 0 
      ? transcription.reduce((sum, item) => sum + item.confidence, 0) / transcription.length
      : 0;

    console.log('STT Response:', { 
      transcript: finalTranscript, 
      confidence: averageConfidence,
      resultsCount: response.results.length 
    });

    res.json({
      success: true,
      transcript: finalTranscript,
      confidence: averageConfidence,
      alternatives: transcription,
      resultsCount: response.results.length
    });

  } catch (error) {
    console.error('STT transcription error:', error);
    
    // Handle specific Google Cloud errors
    let errorMessage = 'Failed to transcribe audio';
    let statusCode = 500;

    if (error.code === 3) {
      errorMessage = 'Invalid audio format or request parameters';
      statusCode = 400;
    } else if (error.code === 16) {
      errorMessage = 'Authentication failed. Please check your Google Cloud credentials.';
      statusCode = 401;
    } else if (error.code === 7) {
      errorMessage = 'Permission denied. Please check your Google Cloud Speech API permissions.';
      statusCode = 403;
    } else if (error.code === 8) {
      errorMessage = 'Resource quota exceeded. Please check your Google Cloud Speech usage limits.';
      statusCode = 429;
    } else if (error.code === 11) {
      errorMessage = 'Audio file too large or too long. Please use shorter audio clips.';
      statusCode = 413;
    }

    res.status(statusCode).json({
      success: false,
      error: errorMessage,
      details: process.env.NODE_ENV === 'development' ? error.message : undefined
    });
  }
};

// Function to transcribe streaming audio (for real-time recognition)
const transcribeStream = async (req, res) => {
  try {
    const { languageCode = 'en-GB' } = req.body;

    // Set up WebSocket or Server-Sent Events for streaming
    // This would require additional setup for real-time streaming
    res.json({
      success: true,
      message: 'Streaming transcription endpoint - requires WebSocket implementation'
    });

  } catch (error) {
    console.error('Streaming STT error:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to set up streaming transcription'
    });
  }
};

// Get supported languages and models
const getSupportedLanguages = async (req, res) => {
  try {
    // Common supported languages for Speech-to-Text
    const supportedLanguages = [
      { code: 'en-US', name: 'English (US)' },
      { code: 'en-GB', name: 'English (UK)' },
      { code: 'en-AU', name: 'English (Australia)' },
      { code: 'es-ES', name: 'Spanish (Spain)' },
      { code: 'es-US', name: 'Spanish (US)' },
      { code: 'fr-FR', name: 'French (France)' },
      { code: 'de-DE', name: 'German (Germany)' },
      { code: 'it-IT', name: 'Italian (Italy)' },
      { code: 'pt-BR', name: 'Portuguese (Brazil)' },
      { code: 'ja-JP', name: 'Japanese (Japan)' },
      { code: 'ko-KR', name: 'Korean (South Korea)' },
      { code: 'zh-CN', name: 'Chinese (Simplified)' },
      { code: 'zh-TW', name: 'Chinese (Traditional)' }
    ];

    const supportedModels = [
      { name: 'latest_long', description: 'Latest model optimized for long audio files' },
      { name: 'latest_short', description: 'Latest model optimized for short audio files' },
      { name: 'command_and_search', description: 'Optimized for commands and search queries' },
      { name: 'phone_call', description: 'Optimized for phone call audio' },
      { name: 'video', description: 'Optimized for video audio' },
      { name: 'default', description: 'Default model' }
    ];

    res.json({
      success: true,
      languages: supportedLanguages,
      models: supportedModels,
      encodings: [
        'LINEAR16',
        'FLAC',
        'MULAW',
        'AMR',
        'AMR_WB',
        'OGG_OPUS',
        'SPEEX_WITH_HEADER_BYTE',
        'WEBM_OPUS'
      ]
    });

  } catch (error) {
    console.error('Error fetching supported languages:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to fetch supported languages'
    });
  }
};

module.exports = {
  transcribeAudio,
  transcribeStream,
  getSupportedLanguages
};
