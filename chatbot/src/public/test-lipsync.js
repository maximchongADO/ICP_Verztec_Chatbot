// Simple test script to verify lip sync generation
// Run this in the browser console after the chatbot loads

async function testLipSyncGeneration() {
  console.log('🧪 Testing Lip Sync Generation...');
  
  try {
    // Test 1: Check if GoogleTTS is available
    if (typeof window.googleTTS === 'undefined') {
      console.error('❌ GoogleTTS not found. Make sure the chatbot is loaded.');
      return;
    }
    
    console.log('✅ GoogleTTS found');
    
    // Test 2: Try to speak with lip sync
    console.log('🎤 Speaking test message...');
    
    await window.googleTTS.speak('Hello, this is a test message for lip sync generation.', {
      onstart: () => {
        console.log('🎵 Audio started playing');
        
        // Check if lip sync data was generated
        const lipSyncData = window.googleTTS.getLastLipSyncData();
        if (lipSyncData) {
          console.log('✅ Lip sync data generated:', lipSyncData);
          console.log('📊 Mouth cues count:', lipSyncData.mouthCues?.length || 0);
        } else {
          console.log('❌ No lip sync data found');
        }
      },
      onend: () => {
        console.log('🎵 Audio finished playing');
      }
    });
    
    // Test 3: Check audio folder
    console.log('📁 Check your audio folder at: chatbot/src/public/audio');
    console.log('   You should see both .mp3 and .wav files');
    console.log('   And also .json files with lip sync data');
    
  } catch (error) {
    console.error('❌ Test failed:', error);
  }
}

// Run the test
testLipSyncGeneration();
