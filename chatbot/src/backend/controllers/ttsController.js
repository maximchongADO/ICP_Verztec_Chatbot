const textToSpeech = require('@google-cloud/text-to-speech');
const path = require('path');
require('dotenv').config();

// Initialize the Google Cloud TTS client
const ttsClient = new textToSpeech.TextToSpeechClient({
  keyFilename: path.resolve(__dirname, 'service-account-key.json'),
  projectId: process.env.GOOGLE_CLOUD_PROJECT_ID || 'golden-frame-461314-e6'
});


const synthesizeSpeech = async (req, res) => {
  try {
    const { text, voice = 'en-GB-Standard-A', languageCode = 'en-GB' } = req.body;

    if (!text || !text.trim()) {
      return res.status(400).json({
        success: false,
        error: 'Text is required for synthesis'
      });
    }

    // Configure the synthesis request
    const request = {
      input: { text: text },
      voice: {
        languageCode: languageCode,
        name: voice,
        ssmlGender: 'FEMALE'
      },
      audioConfig: {
        audioEncoding: 'MP3',
        speakingRate: 1.0,
        pitch: 0.0,
        volumeGainDb: 0.0
      }
    };

    console.log('TTS Request:', { text: text.substring(0, 100), voice, languageCode });

    // Perform the text-to-speech synthesis
    const [response] = await ttsClient.synthesizeSpeech(request);

    if (!response.audioContent) {
      throw new Error('No audio content received from Google TTS');
    }

    // Convert the audio content to base64
    const audioBase64 = response.audioContent.toString('base64');

    res.json({
      success: true,
      audio: audioBase64,
      contentType: 'audio/mpeg'
    });

  } catch (error) {
    console.error('TTS synthesis error:', error);
    
    // Handle specific Google Cloud errors
    let errorMessage = 'Failed to synthesize speech';
    let statusCode = 500;

    if (error.code === 3) {
      errorMessage = 'Invalid request parameters';
      statusCode = 400;
    } else if (error.code === 16) {
      errorMessage = 'Authentication failed. Please check your Google Cloud credentials.';
      statusCode = 401;
    } else if (error.code === 7) {
      errorMessage = 'Permission denied. Please check your Google Cloud TTS API permissions.';
      statusCode = 403;
    } else if (error.code === 8) {
      errorMessage = 'Resource quota exceeded. Please check your Google Cloud TTS usage limits.';
      statusCode = 429;
    }

    res.status(statusCode).json({
      success: false,
      error: errorMessage,
      details: process.env.NODE_ENV === 'development' ? error.message : undefined
    });
  }
};

const getVoices = async (req, res) => {
  try {
    const { languageCode } = req.query;

    const request = languageCode ? { languageCode } : {};
    const [response] = await ttsClient.listVoices(request);

    const voices = response.voices.map(voice => ({
      name: voice.name,
      languageCodes: voice.languageCodes,
      ssmlGender: voice.ssmlGender,
      naturalSampleRateHertz: voice.naturalSampleRateHertz
    }));

    res.json({
      success: true,
      voices: voices
    });

  } catch (error) {
    console.error('Error fetching voices:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to fetch available voices'
    });
  }
};

module.exports = {
  synthesizeSpeech,
  getVoices
};
