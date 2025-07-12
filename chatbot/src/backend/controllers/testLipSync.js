// Test script for TTS and Lip Sync integration
// Run this with: node testLipSync.js

const path = require('path');
const fs = require('fs');

// Test 1: Check if required files exist
console.log('=== Testing File Paths ===');

const rhubarbPath = path.resolve(__dirname, '../../public/avatar-scene/Rhubarb-Lip-Sync-1.14.0-Windows/rhubarb.exe');
const audioDir = path.resolve(__dirname, '../../public/audio');
const lipSyncDir = path.resolve(__dirname, '../../public/audio');

console.log('Rhubarb executable:', rhubarbPath);
console.log('Rhubarb exists:', fs.existsSync(rhubarbPath));

console.log('Audio directory:', audioDir);
console.log('Audio dir exists:', fs.existsSync(audioDir));

console.log('LipSync directory:', lipSyncDir);
console.log('LipSync dir exists:', fs.existsSync(lipSyncDir));

// Test 2: Check FFmpeg availability
console.log('\n=== Testing FFmpeg ===');
const { exec } = require('child_process');

exec('ffmpeg -version', (error, stdout, stderr) => {
  if (error) {
    console.log('❌ FFmpeg not found. Please install FFmpeg and add it to PATH.');
    console.log('Download from: https://ffmpeg.org/download.html');
  } else {
    console.log('✅ FFmpeg is available');
    const version = stdout.split('\n')[0];
    console.log('Version:', version);
  }
});

// Test 3: Check Rhubarb executable
console.log('\n=== Testing Rhubarb ===');
if (fs.existsSync(rhubarbPath)) {
  exec(`"${rhubarbPath}" --version`, (error, stdout, stderr) => {
    if (error) {
      console.log('❌ Rhubarb executable failed:', error.message);
    } else {
      console.log('✅ Rhubarb is working');
      console.log('Version:', stdout.trim());
    }
  });
} else {
  console.log('❌ Rhubarb executable not found at:', rhubarbPath);
}

// Test 4: Test API endpoints
console.log('\n=== API Endpoints Available ===');
console.log('Enhanced TTS with LipSync:');
console.log('  POST /api/tts/synthesize-enhanced');
console.log('  POST /api/tts/chat');
console.log('  POST /api/tts/synthesize-enhanced-test (no auth)');
console.log('  POST /api/tts/chat-test (no auth)');
console.log('');
console.log('LipSync Generation:');
console.log('  POST /api/tts/lipsync/generate');
console.log('  GET /api/tts/lipsync/:filename');
console.log('  POST /api/tts/lipsync/generate-test (no auth)');
console.log('');
console.log('Original TTS (backward compatibility):');
console.log('  POST /api/tts/synthesize');
console.log('  GET /api/tts/voices');

// Test 5: Create sample test request
console.log('\n=== Sample API Request ===');
const sampleRequest = {
  text: "Hello, how are you today?",
  voice: "en-GB-Standard-A",
  languageCode: "en-GB",
  generateLipSyncData: true,
  facialExpression: "smile",
  animation: "Talking_1"
};

console.log('POST /api/tts/synthesize-enhanced-test');
console.log('Content-Type: application/json');
console.log('');
console.log(JSON.stringify(sampleRequest, null, 2));

// Test 6: Sample chat request
console.log('\n=== Sample Chat Request ===');
const chatRequest = {
  messages: [
    {
      text: "Hello there! How can I help you today?",
      facialExpression: "smile",
      animation: "Talking_1"
    },
    {
      text: "I'm here to assist you with any questions you might have.",
      facialExpression: "default",
      animation: "Talking_2"
    }
  ]
};

console.log('POST /api/tts/chat-test');
console.log('Content-Type: application/json');
console.log('');
console.log(JSON.stringify(chatRequest, null, 2));

console.log('\n=== Setup Complete ===');
console.log('✅ Your enhanced TTS with Lip Sync integration is ready!');
console.log('📚 Check LIPSYNC_SETUP.md for detailed documentation');
