// ==========================================
// BROWSER SPEECH-TO-TEXT FUNCTIONALITY
// ==========================================

class BrowserSpeechToText {
  constructor() {
    this.recognition = null;
    this.isListening = false;
    this.onResult = null;
    this.onError = null;
    this.onStart = null;
    this.onEnd = null;
    
    this.initializeSpeechRecognition();
  }

  initializeSpeechRecognition() {
    // Check if browser supports speech recognition
    if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
      console.warn('Speech recognition not supported in this browser');
      return;
    }

    // Create speech recognition instance
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    this.recognition = new SpeechRecognition();

    // Configure speech recognition
    this.recognition.continuous = false; // Stop after one result
    this.recognition.interimResults = true; // Show interim results
    this.recognition.lang = 'en-GB'; // Default language
    this.recognition.maxAlternatives = 1;

    // Set up event listeners
    this.recognition.onstart = () => {
      console.log('Speech recognition started');
      this.isListening = true;
      if (this.onStart) this.onStart();
    };

    this.recognition.onresult = (event) => {
      let finalTranscript = '';
      let interimTranscript = '';

      for (let i = event.resultIndex; i < event.results.length; i++) {
        const transcript = event.results[i][0].transcript;
        if (event.results[i].isFinal) {
          finalTranscript += transcript;
        } else {
          interimTranscript += transcript;
        }
      }

      console.log('Speech result:', { finalTranscript, interimTranscript });
      
      if (this.onResult) {
        this.onResult(finalTranscript || interimTranscript, event.results[event.results.length - 1].isFinal);
      }
    };

    this.recognition.onerror = (event) => {
      console.error('Speech recognition error:', event.error);
      this.isListening = false;
      if (this.onError) this.onError(event);
    };

    this.recognition.onend = () => {
      console.log('Speech recognition ended');
      this.isListening = false;
      if (this.onEnd) this.onEnd();
    };
  }

  startListening(language = 'en-GB') {
    if (!this.recognition) {
      throw new Error('Speech recognition not available');
    }

    if (this.isListening) {
      console.log('Already listening...');
      return;
    }

    this.recognition.lang = language;
    this.recognition.start();
  }

  stopListening() {
    if (!this.recognition || !this.isListening) {
      return;
    }

    this.recognition.stop();
  }

  isSupported() {
    return this.recognition !== null;
  }
}

// ==========================================
// GOOGLE CLOUD SPEECH-TO-TEXT FUNCTIONALITY
// ==========================================

class GoogleCloudSpeechToText {
  constructor() {
    this.mediaRecorder = null;
    this.audioChunks = [];
    this.isRecording = false;
    this.onResult = null;
    this.onError = null;
    this.onStart = null;
    this.onEnd = null;
  }

  async startListening() {
    try {
      // Request microphone access
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      
      this.mediaRecorder = new MediaRecorder(stream, {
        mimeType: 'audio/webm;codecs=opus'
      });

      this.audioChunks = [];

      this.mediaRecorder.ondataavailable = (event) => {
        this.audioChunks.push(event.data);
      };

      this.mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(this.audioChunks, { type: 'audio/webm;codecs=opus' });
        await this.sendAudioToServer(audioBlob);
        
        // Stop all tracks to release microphone
        stream.getTracks().forEach(track => track.stop());
        
        this.isRecording = false;
        if (this.onEnd) this.onEnd();
      };

      this.mediaRecorder.onerror = (event) => {
        console.error('MediaRecorder error:', event.error);
        this.isRecording = false;
        if (this.onError) this.onError(event.error);
      };

      this.mediaRecorder.start();
      this.isRecording = true;
      
      if (this.onStart) this.onStart();

    } catch (error) {
      console.error('Failed to start recording:', error);
      this.isRecording = false;
      if (this.onError) this.onError(error);
    }
  }

  stopListening() {
    if (this.mediaRecorder && this.isRecording) {
      this.mediaRecorder.stop();
    }
  }

  async sendAudioToServer(audioBlob) {
    try {
      // Convert audio blob to base64
      const base64Audio = await this.blobToBase64(audioBlob);
      const audioData = base64Audio.split(',')[1]; // Remove data:audio/webm;base64, prefix

      console.log('Sending audio to server...', { size: audioBlob.size });

      // Send to your STT API
      const response = await fetch('/api/stt/transcribe', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          audioData: audioData,
          languageCode: 'en-GB',
          encoding: 'WEBM_OPUS',
          sampleRateHertz: 48000
        })
      });

      const result = await response.json();
      
      if (result.success && result.transcript) {
        console.log('Transcription result:', result.transcript);
        if (this.onResult) this.onResult(result.transcript, true);
      } else {
        console.log('No speech detected or transcription failed');
        if (this.onError) this.onError('No speech detected');
      }

    } catch (error) {
      console.error('Failed to transcribe audio:', error);
      if (this.onError) this.onError(error);
    }
  }

  blobToBase64(blob) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result);
      reader.onerror = reject;
      reader.readAsDataURL(blob);
    });
  }

  isSupported() {
    return navigator.mediaDevices && navigator.mediaDevices.getUserMedia && window.MediaRecorder;
  }
}

// ==========================================
// UNIFIED SPEECH-TO-TEXT CLASS
// ==========================================

class SpeechToText {
  constructor(preferGoogleCloud = false) {
    this.preferGoogleCloud = preferGoogleCloud;
    this.activeService = null;
    this.onResult = null;
    this.onError = null;
    this.onStart = null;
    this.onEnd = null;

    this.initializeService();
  }

  initializeService() {
    if (this.preferGoogleCloud && new GoogleCloudSpeechToText().isSupported()) {
      this.activeService = new GoogleCloudSpeechToText();
      console.log('Using Google Cloud Speech-to-Text service');
    } else if (new BrowserSpeechToText().isSupported()) {
      this.activeService = new BrowserSpeechToText();
      console.log('Using Browser Speech Recognition API');
    } else {
      console.error('No speech recognition service available');
      return;
    }

    // Set up event forwarding
    this.activeService.onResult = (transcript, isFinal) => {
      if (this.onResult) this.onResult(transcript, isFinal);
    };

    this.activeService.onError = (error) => {
      if (this.onError) this.onError(error);
    };

    this.activeService.onStart = () => {
      if (this.onStart) this.onStart();
    };

    this.activeService.onEnd = () => {
      if (this.onEnd) this.onEnd();
    };
  }

  startListening(language = 'en-GB') {
    if (!this.activeService) {
      throw new Error('No speech recognition service available');
    }

    return this.activeService.startListening(language);
  }

  stopListening() {
    if (this.activeService) {
      this.activeService.stopListening();
    }
  }

  isListening() {
    return this.activeService && this.activeService.isListening;
  }

  isSupported() {
    return this.activeService !== null;
  }

  getServiceType() {
    if (this.activeService instanceof GoogleCloudSpeechToText) {
      return 'google-cloud';
    } else if (this.activeService instanceof BrowserSpeechToText) {
      return 'browser';
    }
    return 'none';
  }
}
