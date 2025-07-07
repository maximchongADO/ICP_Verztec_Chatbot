// Temporary test TTS without authentication
async function testTTSWithoutAuth() {
  try {
    console.log('Testing TTS without authentication...');
    
    const response = await fetch("http://localhost:8000/api/tts/synthesize-test", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        text: "Hello, this is a test without authentication with lip sync generation",
        voice: "en-GB-Standard-A",
        languageCode: "en-GB",
        generateLipSyncData: true
      }),
    });

    console.log('TTS Test Response status:', response.status);
    
    if (!response.ok) {
      const errorText = await response.text();
      console.log('TTS Test Error response:', errorText);
      throw new Error(`TTS test failed: ${response.status} - ${errorText}`);
    }

    const data = await response.json();
    console.log('TTS Test Success:', data.success);
    console.log('Has lip sync data:', !!data.lipSyncData);
    console.log('Audio filename:', data.filename);
    
    if (data.success) {
      // Create audio from base64
      const byteCharacters = atob(data.audio);
      const byteNumbers = new Array(byteCharacters.length);

      for (let i = 0; i < byteCharacters.length; i++) {
        byteNumbers[i] = byteCharacters.charCodeAt(i);
      }

      const byteArray = new Uint8Array(byteNumbers);
      const audioBlob = new Blob([byteArray], { type: "audio/mpeg" });
      const audioUrl = URL.createObjectURL(audioBlob);

      const audio = new Audio(audioUrl);
      audio.play();
      
      console.log('✅ TTS Test successful! Audio is playing.');
      
      if (data.lipSyncData) {
        console.log('🎭 Lip sync data generated:');
        console.log('- Mouth cues:', data.lipSyncData.mouthCues.length);
        console.log('- Duration:', data.lipSyncData.metadata?.duration || 'unknown');
        console.log('- Lip sync path:', data.lipSyncPath);
      }
    }
    
  } catch (error) {
    console.error('❌ TTS Test failed:', error);
  }
}

// Run the test
testTTSWithoutAuth();
