const textToSpeech = require('@google-cloud/text-to-speech');
const path = require('path');
const fs = require('fs');
const { generateLipSync } = require('./rhubarbController');
require('dotenv').config();

// Initialize the Google Cloud TTS client
const ttsClient = new textToSpeech.TextToSpeechClient({
  keyFilename: path.resolve(__dirname, 'service-account-key.json'),
  projectId: process.env.GOOGLE_CLOUD_PROJECT_ID || 'golden-frame-461314-e6'
});

// Create audio directory if it doesn't exist
const audioDir = path.resolve(__dirname, '../../public/audio');
if (!fs.existsSync(audioDir)) {
  fs.mkdirSync(audioDir, { recursive: true });
}


const synthesizeSpeech = async (req, res) => {
  try {
    const { text, voice = 'en-GB-Standard-A', languageCode = 'en-GB', generateLipSyncData = false } = req.body;

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
        speakingRate: 1.25,
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

    // Generate unique filename based on text hash and timestamp
    const crypto = require('crypto');
    const textHash = crypto.createHash('md5').update(text).digest('hex').substring(0, 8);
    const timestamp = Date.now();
    const filename = `tts_${textHash}_${timestamp}.mp3`;
    const audioFilePath = path.join(audioDir, filename);

    // Save audio file to disk (THIS IS CRUCIAL)
    fs.writeFileSync(audioFilePath, response.audioContent);
    console.log('Audio saved to:', audioFilePath);

    // Convert the audio content to base64
    const audioBase64 = response.audioContent.toString('base64');

    // Generate lip sync data if requested
    let lipSyncData = null;
    let lipSyncPath = null;
    if (generateLipSyncData) {
      try {
        console.log('Starting lip sync generation...');
        const lipSyncResult = await generateLipSync(filename, text);
        lipSyncData = lipSyncResult.lipSyncData;
        lipSyncPath = lipSyncResult.lipSyncPath;
        console.log('Lip sync data generated successfully');
      } catch (error) {
        console.error('Failed to generate lip sync data:', error.message);
        // Continue without lip sync data rather than failing completely
      }
    }

    res.json({
      success: true,
      audio: audioBase64,
      contentType: 'audio/mpeg',
      filename: filename,
      audioPath: `/audio/${filename}`,
      lipSyncData: lipSyncData,
      lipSyncPath: lipSyncPath,
      text: text
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
