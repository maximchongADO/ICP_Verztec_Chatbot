// Google Cloud TTS utility functions

class GoogleTTS {
  constructor() {
    this.currentAudio = null;
    this.isPlaying = false;
    this.currentLipSyncData = null;
    this.currentLipSyncPath = null;
  }

  async speak(text, options = {}) {
    try {
      // Stop any currently playing audio
      this.stop();

      const token = localStorage.getItem("token");
      
      // Use enhanced TTS endpoint if lip sync is requested
      const endpoint = options.generateLipSync ? 
        '/api/tts/synthesize-enhanced' : 
        '/api/tts/synthesize';
      
      const response = await fetch(endpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({
          text: text,
          voice: options.voice || 'en-GB-Standard-A',
          languageCode: options.languageCode || 'en-GB',
          generateLipSyncData: options.generateLipSync || false,
          facialExpression: options.facialExpression || 'default',
          animation: options.animation || 'Talking_1'
        })
      });

      if (!response.ok) {
        throw new Error(`TTS request failed: ${response.status}`);
      }

      const data = await response.json();
      
      if (!data.success) {
        throw new Error(data.error || 'TTS synthesis failed');
      }

      console.log('TTS Response:', {
        filename: data.filename,
        hasLipSync: !!data.lipSyncData,
        lipSyncPath: data.lipSyncPath
      });

      // Create audio from base64
      const audioBlob = this.base64ToBlob(data.audio, 'audio/mpeg');
      const audioUrl = URL.createObjectURL(audioBlob);
      
      this.currentAudio = new Audio(audioUrl);
      this.isPlaying = true;

      // Store lip sync data for external use
      this.currentLipSyncData = data.lipSyncData;
      this.currentLipSyncPath = data.lipSyncPath;

      // Set up event listeners
      this.currentAudio.onended = () => {
        this.isPlaying = false;
        URL.revokeObjectURL(audioUrl);
        if (options.onended) options.onended();
      };

      this.currentAudio.onerror = (error) => {
        this.isPlaying = false;
        URL.revokeObjectURL(audioUrl);
        console.error('Audio playback error:', error);
        if (options.onerror) options.onerror(error);
      };

      // Call lip sync callback if provided
      if (options.onLipSync && data.lipSyncData) {
        options.onLipSync(data.lipSyncData);
      }

      // Play the audio
      await this.currentAudio.play();

    } catch (error) {
      console.error('TTS Error:', error);
      this.isPlaying = false;
      if (options.onerror) options.onerror(error);
    }
  }

  stop() {
    if (this.currentAudio && this.isPlaying) {
      this.currentAudio.pause();
      this.currentAudio.currentTime = 0;
      this.isPlaying = false;
    }
  }

  base64ToBlob(base64, mimeType) {
    const byteCharacters = atob(base64);
    const byteNumbers = new Array(byteCharacters.length);
    
    for (let i = 0; i < byteCharacters.length; i++) {
      byteNumbers[i] = byteCharacters.charCodeAt(i);
    }
    
    const byteArray = new Uint8Array(byteNumbers);
    return new Blob([byteArray], { type: mimeType });
  }

  // Compatibility methods for existing code
  cancel() {
    this.stop();
  }

  // Method to get current lip sync data
  getCurrentLipSyncData() {
    return this.currentLipSyncData;
  }

  // Method to get current lip sync path
  getCurrentLipSyncPath() {
    return this.currentLipSyncPath;
  }
}

// Global instance for compatibility
window.googleTTS = new GoogleTTS();

// Compatibility function to replace responsiveVoice.speak
window.speak = function(text, voice, options = {}) {
  return window.googleTTS.speak(text, {
    voice: voice === "UK English Female" ? 'en-GB-Standard-A' : 'en-US-Standard-C',
    languageCode: voice === "UK English Female" ? 'en-GB' : 'en-US',
    ...options
  });
};

// Compatibility function to replace responsiveVoice.cancel
window.cancelSpeech = function() {
  window.googleTTS.stop();
};
