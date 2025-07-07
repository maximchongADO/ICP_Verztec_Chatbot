// Test script to verify lip sync generation is working
const fetch = require('node-fetch');
const path = require('path');
const fs = require('fs');

async function testLipSyncGeneration() {
  console.log('🧪 Testing Lip Sync Generation...');
  
  try {
    // Test the enhanced TTS endpoint with lip sync
    const response = await fetch('http://localhost:8000/api/tts/synthesize-enhanced-test', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        text: 'Hello, this is a test message for lip sync generation.',
        voice: 'en-GB-Standard-A',
        languageCode: 'en-GB',
        generateLipSyncData: true
      })
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    console.log('✅ API Response received');
    console.log('📁 Audio filename:', data.filename);
    console.log('🎵 Audio path:', data.audioPath);
    console.log('💋 Lip sync data:', data.lipSyncData ? 'Generated' : 'Not generated');
    console.log('📊 Mouth cues count:', data.lipSyncData?.mouthCues?.length || 0);

    // Check if files were created
    const audioDir = path.resolve(__dirname, '../public/audio');
    const audioFile = path.join(audioDir, data.filename);
    const wavFile = path.join(audioDir, data.filename.replace('.mp3', '.wav'));
    const jsonFile = path.join(audioDir, data.filename.replace('.mp3', '.json'));

    console.log('\n📁 File Check:');
    console.log('MP3 file exists:', fs.existsSync(audioFile));
    console.log('WAV file exists:', fs.existsSync(wavFile));
    console.log('JSON file exists:', fs.existsSync(jsonFile));

    if (fs.existsSync(jsonFile)) {
      console.log('✅ Lip sync JSON file was created!');
    } else {
      console.log('❌ Lip sync JSON file was NOT created');
    }

  } catch (error) {
    console.error('❌ Test failed:', error.message);
  }
}

// Run the test
testLipSyncGeneration();
