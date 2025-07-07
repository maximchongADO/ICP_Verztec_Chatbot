# Lip Sync Setup Guide

## Prerequisites

### 1. Install FFmpeg
FFmpeg is required to convert MP3 audio files to WAV format for better Rhubarb processing.

**Download and Install:**
1. Go to https://ffmpeg.org/download.html
2. Download the Windows build
3. Extract to a folder (e.g., `C:\ffmpeg`)
4. Add `C:\ffmpeg\bin` to your system PATH environment variable

**Adding FFmpeg to PATH (Windows):**

**Method 1: Using System Properties (Recommended)**
1. Right-click on "This PC" or "My Computer" and select "Properties"
2. Click "Advanced system settings"
3. Click "Environment Variables..." button
4. In the "System Variables" section, find and select "Path"
5. Click "Edit..."
6. Click "New" and add: `C:\ffmpeg\bin`
7. Click "OK" to close all dialogs
8. **Restart your command prompt or IDE** for changes to take effect

**Method 2: Using Windows Settings (Windows 10/11)**
1. Open Windows Settings (Win + I)
2. Search for "Environment Variables"
3. Click "Edit the system environment variables"
4. Click "Environment Variables..." button
5. In the "System Variables" section, find and select "Path"
6. Click "Edit..."
7. Click "New" and add: `C:\ffmpeg\bin`
8. Click "OK" to close all dialogs
9. **Restart your command prompt or IDE** for changes to take effect

**Method 3: Using Command Prompt (Temporary)**
```cmd
set PATH=%PATH%;C:\ffmpeg\bin
```
*Note: This method only works for the current session*

**Verify Installation:**
```bash
ffmpeg -version
```

You should see output similar to:
```
ffmpeg version 4.4.2 Copyright (c) 2000-2021 the FFmpeg developers
built with gcc 10.3.0 (Rev5, Built by MSYS2 project)
```

**Troubleshooting PATH Issues:**
- If `ffmpeg -version` doesn't work, restart your terminal/command prompt
- If still not working, restart your computer
- Check that the path `C:\ffmpeg\bin\ffmpeg.exe` exists
- Make sure there are no typos in the PATH entry

### 2. Verify Rhubarb Executable
The Rhubarb executable should be located at:
```
chatbot/src/public/avatar-scene/Rhubarb-Lip-Sync-1.14.0-Windows/rhubarb.exe
```

**Test Rhubarb:**
```bash
cd chatbot/src/public/avatar-scene/Rhubarb-Lip-Sync-1.14.0-Windows
./rhubarb.exe --version
```

## API Endpoints

### 1. TTS with Lip Sync
```http
POST /api/tts/synthesize
Content-Type: application/json
Authorization: Bearer <token>

{
  "text": "Hello, how are you today?",
  "voice": "en-GB-Standard-A",
  "languageCode": "en-GB",
  "generateLipSyncData": true
}
```

**Response:**
```json
{
  "success": true,
  "audio": "base64AudioData...",
  "contentType": "audio/mpeg",
  "filename": "tts_abc123_1234567890.mp3",
  "audioPath": "/audio/tts_abc123_1234567890.mp3",
  "lipSyncData": {
    "mouthCues": [
      {"start": 0.00, "end": 0.05, "value": "X"},
      {"start": 0.05, "end": 0.27, "value": "D"},
      {"start": 0.27, "end": 0.31, "value": "C"},
      {"start": 0.31, "end": 0.43, "value": "B"},
      {"start": 0.43, "end": 0.47, "value": "X"}
    ]
  },
  "lipSyncPath": "/lipsync/tts_abc123_1234567890.json",
  "processingTime": 1250
}
```

### 2. Generate Lip Sync for Existing Audio
```http
POST /api/tts/lipsync/generate
Content-Type: application/json
Authorization: Bearer <token>

{
  "audioFilename": "tts_abc123_1234567890.mp3",
  "text": "Hello, how are you today?"
}
```

### 3. Get Existing Lip Sync Data
```http
GET /api/tts/lipsync/tts_abc123_1234567890.json
Authorization: Bearer <token>
```

## Mouth Shape Reference

Rhubarb generates mouth shapes using these labels:
- **A** - Rest position, mouth closed
- **B** - Slightly open, for sounds like "m", "b", "p"
- **C** - More open, for sounds like "e", "i"
- **D** - Open, for sounds like "a"
- **E** - Wide open, for sounds like "ah"
- **F** - Lips forward, for sounds like "f", "v"
- **G** - Extended shape (optional)
- **H** - Extended shape (optional)
- **X** - Rest/silence

## Integration with Avatar

Use the lip sync data to animate your avatar's mouth:

```javascript
// Example usage in frontend
const response = await fetch('/api/tts/synthesize', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${token}`
  },
  body: JSON.stringify({
    text: "Hello world!",
    generateLipSyncData: true
  })
});

const data = await response.json();

// Play audio
const audio = new Audio(`data:audio/mpeg;base64,${data.audio}`);
audio.play();

// Animate mouth using lip sync data
animateMouth(data.lipSyncData.mouthCues);
```

## Troubleshooting

### FFmpeg Issues
- Ensure FFmpeg is in your PATH
- Try using absolute path to ffmpeg.exe in the command
- Check Windows permissions

### Rhubarb Issues
- Ensure the executable has proper permissions
- Check that audio files are accessible
- Verify file paths are correct

### Performance Tips
- Use phonetic recognizer (-r phonetic) for faster processing
- Provide dialog text for better accuracy
- Clean up temporary files regularly
