// Test script to verify Rhubarb lip sync integration
// Run this from the browser console or as a standalone test

async function testRhubarbIntegration() {
  console.log('🧪 Testing Rhubarb Lip Sync Integration...');
  
  try {
    // Test the enhanced TTS endpoint with lip sync
    const response = await fetch('/api/tts/synthesize-enhanced-test', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        text: "Hello, this is a test of the Rhubarb lip sync integration.",
        voice: 'en-GB-Standard-A',
        languageCode: 'en-GB',
        generateLipSyncData: true,
        facialExpression: 'smile',
        animation: 'Talking_1'
      })
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    console.log('✅ Enhanced TTS Response:', data);

    if (data.success) {
      console.log('📊 TTS Success:', {
        filename: data.filename,
        hasAudio: !!data.audio,
        hasLipSync: !!data.lipSyncData,
        lipSyncPath: data.lipSyncPath,
        processingTime: data.processingTime
      });

      if (data.lipSyncData && data.lipSyncData.mouthCues) {
        console.log('🎭 Lip Sync Data:', {
          mouthCues: data.lipSyncData.mouthCues.length,
          sampleCues: data.lipSyncData.mouthCues.slice(0, 5)
        });
        
        // Test audio playback
        const audioBlob = base64ToBlob(data.audio, 'audio/mpeg');
        const audioUrl = URL.createObjectURL(audioBlob);
        const audio = new Audio(audioUrl);
        
        console.log('🔊 Playing test audio...');
        audio.play();
        
        // Simulate lip sync animation
        animateLipSync(data.lipSyncData.mouthCues);
        
        return {
          success: true,
          message: 'Rhubarb lip sync is working correctly!',
          data: data
        };
      } else {
        return {
          success: false,
          message: 'TTS worked but no lip sync data was generated'
        };
      }
    } else {
      return {
        success: false,
        message: data.error || 'Unknown error occurred'
      };
    }

  } catch (error) {
    console.error('❌ Test failed:', error);
    return {
      success: false,
      message: error.message
    };
  }
}

function base64ToBlob(base64, mimeType) {
  const byteCharacters = atob(base64);
  const byteNumbers = new Array(byteCharacters.length);
  
  for (let i = 0; i < byteCharacters.length; i++) {
    byteNumbers[i] = byteCharacters.charCodeAt(i);
  }
  
  const byteArray = new Uint8Array(byteNumbers);
  return new Blob([byteArray], { type: mimeType });
}

function animateLipSync(mouthCues) {
  console.log('🎬 Animating lip sync...');
  
  mouthCues.forEach((cue, index) => {
    setTimeout(() => {
      console.log(`Mouth shape: ${cue.value} (${cue.start}s - ${cue.end}s)`);
    }, cue.start * 1000);
  });
}

// Auto-run test
console.log('🚀 Starting Rhubarb integration test...');
testRhubarbIntegration().then(result => {
  console.log('📋 Test Result:', result);
});
