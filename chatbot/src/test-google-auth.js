const textToSpeech = require('@google-cloud/text-to-speech');
const path = require('path');
require('dotenv').config();

// Test Google Cloud authentication
console.log('Testing Google Cloud TTS authentication...');

const ttsClient = new textToSpeech.TextToSpeechClient({
  keyFilename: path.resolve(__dirname, 'backend/controllers/service-account-key.json'),
  projectId: 'golden-frame-461314-e6'
});

async function testAuthentication() {
  try {
    console.log('Service account key path:', path.resolve(__dirname, 'backend/controllers/service-account-key.json'));
    
    // Simple test request
    const request = {
      input: { text: 'Hello, testing authentication' },
      voice: {
        languageCode: 'en-GB',
        name: 'en-GB-Standard-A',
        ssmlGender: 'FEMALE'
      },
      audioConfig: {
        audioEncoding: 'MP3'
      }
    };

    console.log('Making TTS request...');
    const [response] = await ttsClient.synthesizeSpeech(request);
    
    if (response.audioContent) {
      console.log('✅ Authentication successful! TTS response received.');
      console.log('Audio content length:', response.audioContent.length);
    } else {
      console.log('❌ No audio content in response');
    }
    
  } catch (error) {
    console.error('❌ Authentication failed:');
    console.error('Error code:', error.code);
    console.error('Error message:', error.message);
    console.error('Error details:', error.details);
    
    if (error.message.includes('authentication')) {
      console.log('\n🔍 Troubleshooting suggestions:');
      console.log('1. Check if the service account key file exists and has correct permissions');
      console.log('2. Verify the service account has Text-to-Speech API enabled');
      console.log('3. Check if the project ID is correct');
      console.log('4. Ensure the service account has the Cloud Text-to-Speech API User role');
    }
  }
}

testAuthentication();
