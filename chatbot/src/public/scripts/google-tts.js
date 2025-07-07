class GoogleTTS {
  constructor() {
    this.isPlaying = false;
    this.currentAudio = null;
    this.onEndCallback = null;
    this.onStartCallback = null;
  }

  async speak(text, options = {}) {
    if (!text || !text.trim()) return;

    // Stop any currently playing audio
    this.cancel();    const {
      voice = 'en-GB-Standard-A',  // British English female voice
      languageCode = 'en-GB',      // British English
      volume = 1,
      onend = null,
      onstart = null
    } = options;

    this.onEndCallback = onend;
    this.onStartCallback = onstart;    try {
      const token = localStorage.getItem("token");
      console.log('TTS Request - Token exists:', !!token);
      console.log('TTS Request - Token preview:', token ? token.substring(0, 20) + '...' : 'No token');
      
      const response = await fetch("/api/tts/synthesize", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({
          text: text,
          voice: voice,
          languageCode: languageCode,
        }),
      });

      console.log('TTS Response status:', response.status);
      console.log('TTS Response headers:', [...response.headers.entries()]);

      if (!response.ok) {
        const errorText = await response.text();
        console.log('TTS Error response body:', errorText);
        throw new Error(`TTS request failed: ${response.status} - ${errorText}`);
      }

      const data = await response.json();

      if (!data.success) {
        throw new Error(data.error || "TTS synthesis failed");
      }

      // Create audio from base64
      const byteCharacters = atob(data.audio);
      const byteNumbers = new Array(byteCharacters.length);

      for (let i = 0; i < byteCharacters.length; i++) {
        byteNumbers[i] = byteCharacters.charCodeAt(i);
      }

      const byteArray = new Uint8Array(byteNumbers);
      const audioBlob = new Blob([byteArray], { type: "audio/mpeg" });
      const audioUrl = URL.createObjectURL(audioBlob);

      this.currentAudio = new Audio(audioUrl);
      this.currentAudio.volume = volume;
      
      this.currentAudio.onloadeddata = () => {
        this.isPlaying = true;
        if (this.onStartCallback) this.onStartCallback();
      };

      this.currentAudio.onended = () => {
        this.isPlaying = false;
        URL.revokeObjectURL(audioUrl);
        if (this.onEndCallback) this.onEndCallback();
        this.currentAudio = null;
      };

      this.currentAudio.onerror = () => {
        this.isPlaying = false;
        URL.revokeObjectURL(audioUrl);
        console.error('Audio playback error');
        if (this.onEndCallback) this.onEndCallback();
        this.currentAudio = null;
      };

      await this.currentAudio.play();
    } catch (error) {
      console.error("TTS Error:", error);
      this.isPlaying = false;
      if (this.onEndCallback) this.onEndCallback();
    }
  }

  cancel() {
    if (this.currentAudio) {
      this.currentAudio.pause();
      this.currentAudio.currentTime = 0;
      this.currentAudio = null;
    }
    this.isPlaying = false;
  }

  pause() {
    if (this.currentAudio && !this.currentAudio.paused) {
      this.currentAudio.pause();
    }
  }

  resume() {
    if (this.currentAudio && this.currentAudio.paused) {
      this.currentAudio.play();
    }
  }
}

// Create global instance for backward compatibility
window.googleTTS = new GoogleTTS();

// Create ResponsiveVoice-compatible interface for easier migration
window.responsiveVoice = {
  speak: (text, voice, options = {}) => {
    const voiceMap = {
      "UK English Female": "en-GB-Standard-A",
      "US English Female": "en-US-Standard-C",
      "US English Male": "en-US-Standard-B",
      "Asian English Female": "en-US-Standard-A",    // Google's Asian-accented female voice
      "Indian English Female": "en-IN-Standard-A",   // Indian female voice
      "Japanese Female": "ja-JP-Standard-A",         // Japanese female voice
      "Korean Female": "ko-KR-Standard-A",           // Korean female voice
      "Chinese Female": "cmn-CN-Standard-A"          // Chinese Mandarin female voice
    };
    
    const googleVoice = voiceMap[voice] || "en-GB-Standard-A";  // Default to British female
    
    // Determine language code based on voice
    let detectedLanguageCode = 'en-US';
    if (googleVoice.includes('en-GB')) detectedLanguageCode = 'en-GB';
    else if (googleVoice.includes('en-IN')) detectedLanguageCode = 'en-IN';
    else if (googleVoice.includes('ja-JP')) detectedLanguageCode = 'ja-JP';
    else if (googleVoice.includes('ko-KR')) detectedLanguageCode = 'ko-KR';
    else if (googleVoice.includes('cmn-CN')) detectedLanguageCode = 'cmn-CN';
    
    window.googleTTS.speak(text, {
      voice: googleVoice,
      languageCode: detectedLanguageCode,
      volume: options.volume || 1,
      onend: options.onend,
      onstart: options.onstart
    });
  },
  
  cancel: () => {
    window.googleTTS.cancel();
  },
  
  pause: () => {
    window.googleTTS.pause();
  },
  
  resume: () => {
    window.googleTTS.resume();
  }
};
