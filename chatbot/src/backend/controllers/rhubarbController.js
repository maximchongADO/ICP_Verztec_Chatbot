const path = require('path');
const fs = require('fs');
const { exec } = require('child_process');
const util = require('util');

const execAsync = util.promisify(exec);

// Path to Rhubarb executable
const rhubarbPath = path.resolve(__dirname, '../../public/avatar-scene/Rhubarb-Lip-Sync-1.14.0-Windows/rhubarb.exe');
const audioDir = path.resolve(__dirname, '../../public/audio');
const lipSyncDir = path.resolve(__dirname, '../../public/lipsync');

// Create directories if they don't exist
if (!fs.existsSync(audioDir)) {
  fs.mkdirSync(audioDir, { recursive: true });
}
if (!fs.existsSync(lipSyncDir)) {
  fs.mkdirSync(lipSyncDir, { recursive: true });
}

/**
 * Execute command with promise wrapper (like in reference code)
 * @param {string} command - Command to execute
 * @returns {Promise<string>} Command output
 */
const execCommand = (command) => {
  return new Promise((resolve, reject) => {
    exec(command, (error, stdout, stderr) => {
      if (error) reject(error);
      resolve(stdout);
    });
  });
};

/**
 * Generate lip sync for a message (like in reference code)
 * @param {string} message - Message identifier or filename
 * @param {boolean} keepInAudioFolder - Whether to keep files in audio folder
 * @returns {Promise<void>}
 */
const lipSyncMessage = async (message, keepInAudioFolder = true) => {
  const time = new Date().getTime();
  console.log(`Starting conversion for message ${message}`);
  
  const audioPath = keepInAudioFolder ? audioDir : audioDir;
  const lipSyncPath = keepInAudioFolder ? audioDir : lipSyncDir;
  
  // Convert MP3 to WAV
  await execCommand(
    `ffmpeg -y -i "${audioPath}/message_${message}.mp3" "${audioPath}/message_${message}.wav"`
  );
  console.log(`Conversion done in ${new Date().getTime() - time}ms`);
  
  // Generate lip sync data
  await execCommand(
    `"${rhubarbPath}" -f json -o "${lipSyncPath}/message_${message}.json" "${audioPath}/message_${message}.wav" -r phonetic`
  );
  console.log(`Lip sync done in ${new Date().getTime() - time}ms`);
};

/**
 * Convert audio file from MP3 to WAV using FFmpeg
 * @param {string} inputPath - Path to input MP3 file
 * @param {string} outputPath - Path to output WAV file
 * @returns {Promise<void>}
 */
const convertMp3ToWav = async (inputPath, outputPath) => {
  const command = `ffmpeg -y -i "${inputPath}" "${outputPath}"`;
  console.log('Converting MP3 to WAV:', command);
  
  try {
    const { stdout, stderr } = await execAsync(command, { timeout: 30000 });
    console.log('Audio conversion completed');
    if (stderr && !stderr.includes('size=')) {
      console.warn('FFmpeg stderr:', stderr);
    }
  } catch (error) {
    throw new Error(`Audio conversion failed: ${error.message}`);
  }
};

/**
 * Generate lip sync data using Rhubarb (updated to match reference pattern)
 * @param {string} audioFilename - Name of the audio file (MP3)
 * @param {string} text - The spoken text (optional, improves accuracy)
 * @param {boolean} keepInAudioFolder - Whether to keep WAV and JSON in audio folder
 * @returns {Promise<Object>} Lip sync data
 */
const generateLipSync = async (audioFilename, text = '', keepInAudioFolder = true) => {
  const time = new Date().getTime();
  console.log(`Starting lip sync generation for: ${audioFilename}`);
  
  try {
    const audioFilePath = path.join(audioDir, audioFilename);
    const wavFilename = audioFilename.replace('.mp3', '.wav');
    const jsonFilename = audioFilename.replace('.mp3', '.json');
    
    // Determine output paths based on keepInAudioFolder flag
    const wavFilePath = keepInAudioFolder 
      ? path.join(audioDir, wavFilename)
      : path.join(audioDir, wavFilename);
    const outputFilePath = keepInAudioFolder 
      ? path.join(audioDir, jsonFilename)
      : path.join(lipSyncDir, jsonFilename);

    // Check if audio file exists
    if (!fs.existsSync(audioFilePath)) {
      throw new Error(`Audio file not found: ${audioFilePath}`);
    }

    // Check if Rhubarb executable exists
    if (!fs.existsSync(rhubarbPath)) {
      throw new Error(`Rhubarb executable not found: ${rhubarbPath}`);
    }

    console.log(`Audio file exists: ${audioFilePath}`);
    console.log(`Will convert to WAV: ${wavFilePath}`);
    console.log(`Will output to: ${outputFilePath}`);

    // Step 1: Convert MP3 to WAV (Rhubarb works better with WAV)
    console.log(`Converting audio started at ${new Date().getTime() - time}ms`);
    await execCommand(`ffmpeg -y -i "${audioFilePath}" "${wavFilePath}"`);
    console.log(`Conversion completed at ${new Date().getTime() - time}ms`);

    // Verify WAV file was created
    if (!fs.existsSync(wavFilePath)) {
      throw new Error(`WAV conversion failed - output file not found: ${wavFilePath}`);
    }

    // Step 2: Build Rhubarb command
    let command = `"${rhubarbPath}" -f json -o "${outputFilePath}" -r phonetic`;
    
    // Add dialog text if provided (improves accuracy)
    let tempTextFile = null;
    if (text && text.trim()) {
      const tempDir = keepInAudioFolder ? audioDir : lipSyncDir;
      tempTextFile = path.join(tempDir, `temp_${Date.now()}.txt`);
      fs.writeFileSync(tempTextFile, text.trim());
      command += ` --dialogFile "${tempTextFile}"`;
      console.log(`Created dialog file: ${tempTextFile}`);
    }
    
    command += ` "${wavFilePath}"`;
    console.log('Running Rhubarb command:', command);

    // Step 3: Execute Rhubarb
    await execCommand(command);
    console.log(`Rhubarb execution completed at ${new Date().getTime() - time}ms`);

    // Step 4: Check if output file was created
    if (!fs.existsSync(outputFilePath)) {
      throw new Error(`Rhubarb did not generate output file: ${outputFilePath}`);
    }

    // Step 5: Read the generated lip sync data
    const lipSyncData = JSON.parse(fs.readFileSync(outputFilePath, 'utf8'));
    console.log(`Lip sync generation completed in ${new Date().getTime() - time}ms`);
    console.log(`Generated ${lipSyncData.mouthCues ? lipSyncData.mouthCues.length : 0} mouth cues`);
    
    // Step 6: Cleanup temporary files (but keep WAV and JSON as requested)
    try {
      // Remove temporary text file only
      if (tempTextFile && fs.existsSync(tempTextFile)) {
        fs.unlinkSync(tempTextFile);
        console.log('Cleaned up temp text file');
      }
    } catch (cleanupError) {
      console.warn('Cleanup warning:', cleanupError.message);
    }

    return {
      success: true,
      lipSyncData: lipSyncData,
      outputFile: jsonFilename,
      wavFile: wavFilename,
      audioPath: keepInAudioFolder ? `/audio/${wavFilename}` : `/audio/${wavFilename}`,
      lipSyncPath: keepInAudioFolder ? `/audio/${jsonFilename}` : `/lipsync/${jsonFilename}`,
      processingTime: new Date().getTime() - time
    };

  } catch (error) {
    console.error('Lip sync generation error:', error);
    throw error;
  }
};

/**
 * Read JSON transcript file (helper function like in the reference implementation)
 * @param {string} filePath - Path to the JSON file
 * @returns {Promise<Object>} Parsed JSON data
 */
const readJsonTranscript = async (filePath) => {
  try {
    const data = fs.readFileSync(filePath, 'utf8');
    return JSON.parse(data);
  } catch (error) {
    throw new Error(`Failed to read JSON transcript: ${error.message}`);
  }
};

/**
 * Convert audio file to base64 (helper function like in the reference implementation)
 * @param {string} filePath - Path to the audio file
 * @returns {Promise<string>} Base64 encoded audio
 */
const audioFileToBase64 = async (filePath) => {
  try {
    const data = fs.readFileSync(filePath);
    return data.toString('base64');
  } catch (error) {
    throw new Error(`Failed to convert audio to base64: ${error.message}`);
  }
};

/**
 * Process TTS audio with lip sync generation
 */
const processAudioWithLipSync = async (req, res) => {
  try {
    const { audioFilename, text } = req.body;

    if (!audioFilename) {
      return res.status(400).json({
        success: false,
        error: 'Audio filename is required'
      });
    }

    const result = await generateLipSync(audioFilename, text);
    
    res.json(result);

  } catch (error) {
    console.error('Error processing audio with lip sync:', error);
    res.status(500).json({
      success: false,
      error: error.message || 'Failed to generate lip sync data'
    });
  }
};

/**
 * Get existing lip sync data (updated to check both audio and lipsync folders)
 */
const getLipSyncData = async (req, res) => {
  try {
    const { filename } = req.params;
    
    // First try audio folder, then lipsync folder
    let lipSyncFilePath = path.join(audioDir, filename);
    if (!fs.existsSync(lipSyncFilePath)) {
      lipSyncFilePath = path.join(lipSyncDir, filename);
    }

    if (!fs.existsSync(lipSyncFilePath)) {
      return res.status(404).json({
        success: false,
        error: 'Lip sync file not found in audio or lipsync directories'
      });
    }

    const lipSyncData = JSON.parse(fs.readFileSync(lipSyncFilePath, 'utf8'));
    
    res.json({
      success: true,
      lipSyncData: lipSyncData,
      location: lipSyncFilePath.includes(audioDir) ? 'audio' : 'lipsync'
    });

  } catch (error) {
    console.error('Error reading lip sync data:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to read lip sync data'
    });
  }
};

module.exports = {
  generateLipSync,
  processAudioWithLipSync,
  getLipSyncData,
  readJsonTranscript,
  audioFileToBase64,
  convertMp3ToWav,
  execCommand,
  lipSyncMessage
};
