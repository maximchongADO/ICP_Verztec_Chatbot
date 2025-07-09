const textToSpeech = require('@google-cloud/text-to-speech');
const path = require('path');
const fs = require('fs');
const { generateLipSync, audioFileToBase64, readJsonTranscript } = require('./rhubarbController');
require('dotenv').config();

// Initialize the Google Cloud TTS client with proper authentication
const ttsClient = new textToSpeech.TextToSpeechClient({
  keyFilename: path.resolve(__dirname, 'service-account-key.json'), // Updated to correct path
  projectId:'golden-frame-461314-e6'
});

// Create audio directory if it doesn't exist
const audioDir = path.resolve(__dirname, '../../public/audio');
const lipSyncDir = path.resolve(__dirname, '../../public/audio');

[audioDir, lipSyncDir].forEach(dir => {
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
  }
});

/**
 * Enhanced TTS endpoint that follows the reference implementation pattern
 * Generates both audio and lip sync data in a single request
 */
const synthesizeSpeechWithLipSync = async (req, res) => {
  const startTime = new Date().getTime();
  
  try {
    const { 
      text, 
      voice = 'en-GB-Standard-A', 
      languageCode = 'en-GB',
      generateLipSyncData = false,
      facialExpression = 'default',
      animation = 'Talking_1'
    } = req.body;

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

    console.log(`Starting TTS synthesis for: "${text.substring(0, 50)}..."`);

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

    // Save audio file to disk
    fs.writeFileSync(audioFilePath, response.audioContent);
    console.log(`Audio saved: ${filename} in ${new Date().getTime() - startTime}ms`);

    // Convert the audio content to base64
    const audioBase64 = response.audioContent.toString('base64');

    // Prepare response object
    const responseObj = {
      success: true,
      text: text,
      audio: audioBase64,
      contentType: 'audio/mpeg',
      filename: filename,
      audioPath: `/audio/${filename}`,
      facialExpression: facialExpression,
      animation: animation,
      processingTime: new Date().getTime() - startTime
    };

    // Generate lip sync data if requested
    if (generateLipSyncData) {
      try {
        console.log('Generating lip sync data...');
        const lipSyncResult = await generateLipSync(filename, text);
        
        responseObj.lipSyncData = lipSyncResult.lipSyncData;
        responseObj.lipSyncPath = lipSyncResult.lipSyncPath;
        responseObj.lipSyncProcessingTime = lipSyncResult.processingTime;
        
        console.log(`Total processing time: ${new Date().getTime() - startTime}ms`);
      } catch (error) {
        console.warn('Failed to generate lip sync data:', error.message);
        responseObj.lipSyncError = error.message;
        // Continue without lip sync data rather than failing completely
      }
    }

    res.json(responseObj);

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
      details: process.env.NODE_ENV === 'development' ? error.message : undefined,
      processingTime: new Date().getTime() - startTime
    });
  }
};

/**
 * Chat endpoint that mimics the reference implementation
 * Generates multiple messages with TTS and lip sync
 */
const chatWithLipSync = async (req, res) => {
  try {
    const { messages } = req.body;
    
    if (!messages || !Array.isArray(messages)) {
      return res.status(400).json({
        success: false,
        error: 'Messages array is required'
      });
    }

    const processedMessages = [];

    for (let i = 0; i < messages.length; i++) {
      const message = messages[i];
      
      if (!message.text) {
        continue;
      }

      console.log(`Processing message ${i}: "${message.text.substring(0, 50)}..."`);

      // Generate audio file
      const ttsRequest = {
        input: { text: message.text },
        voice: {
          languageCode: 'en-GB',
          name: 'en-GB-Standard-A',
          ssmlGender: 'FEMALE'
        },
        audioConfig: {
          audioEncoding: 'MP3',
          speakingRate: 1.25,
          pitch: 0.0,
          volumeGainDb: 0.0
        }
      };

      const [response] = await ttsClient.synthesizeSpeech(ttsRequest);
      
      // Generate unique filename
      const crypto = require('crypto');
      const textHash = crypto.createHash('md5').update(message.text).digest('hex').substring(0, 8);
      const timestamp = Date.now();
      const filename = `message_${i}_${textHash}_${timestamp}.mp3`;
      const audioFilePath = path.join(audioDir, filename);

      // Save audio file
      fs.writeFileSync(audioFilePath, response.audioContent);

      // Generate lip sync
      const lipSyncResult = await generateLipSync(filename, message.text);

      // Convert audio to base64
      const audioBase64 = await audioFileToBase64(audioFilePath);

      // Add processed data to message
      processedMessages.push({
        text: message.text,
        audio: audioBase64,
        lipsync: lipSyncResult.lipSyncData,
        facialExpression: message.facialExpression || 'default',
        animation: message.animation || 'Talking_1',
        filename: filename,
        audioPath: `/audio/${filename}`,
        lipSyncPath: lipSyncResult.lipSyncPath
      });
    }

    res.json({
      success: true,
      messages: processedMessages
    });

  } catch (error) {
    console.error('Chat processing error:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to process chat messages',
      details: process.env.NODE_ENV === 'development' ? error.message : undefined
    });
  }
};

// Keep the original functions for backward compatibility
const synthesizeSpeech = async (req, res) => {
  // This maintains the original API but adds lip sync support
  return synthesizeSpeechWithLipSync(req, res);
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
  synthesizeSpeechWithLipSync,
  chatWithLipSync,
  getVoices
};
