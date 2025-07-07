const path = require('path');
const fs = require('fs');
const { exec } = require('child_process');
const util = require('util');

const execAsync = util.promisify(exec);

// Path to Rhubarb executable
const rhubarbPath = path.resolve(__dirname, '../../public/avatar-scene/Rhubarb-Lip-Sync-1.14.0-Windows/rhubarb.exe');
const audioDir = path.resolve(__dirname, '../../public/audio');
const lipSyncDir = path.resolve(__dirname, '../../public/lipsync');

// Create lip sync directory if it doesn't exist
if (!fs.existsSync(lipSyncDir)) {
  fs.mkdirSync(lipSyncDir, { recursive: true });
}

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
 * Generate lip sync data using Rhubarb
 * @param {string} audioFilename - Name of the audio file (MP3)
 * @param {string} text - The spoken text (optional, improves accuracy)
 * @returns {Promise<Object>} Lip sync data
 */
const generateLipSync = async (audioFilename, text = '') => {
  const time = new Date().getTime();
  console.log(`Starting lip sync generation for: ${audioFilename}`);
  
  try {
    const audioFilePath = path.join(audioDir, audioFilename);
    const wavFilename = audioFilename.replace('.mp3', '.wav');
    const wavFilePath = path.join(audioDir, wavFilename);
    const outputFilename = audioFilename.replace('.mp3', '.json');
    const outputFilePath = path.join(lipSyncDir, outputFilename);

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
    await convertMp3ToWav(audioFilePath, wavFilePath);
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
      tempTextFile = path.join(lipSyncDir, `temp_${Date.now()}.txt`);
      fs.writeFileSync(tempTextFile, text.trim());
      command += ` --dialogFile "${tempTextFile}"`;
      console.log(`Created dialog file: ${tempTextFile}`);
    }
    
    command += ` "${wavFilePath}"`;
    console.log('Running Rhubarb command:', command);

    // Step 3: Execute Rhubarb
    const { stdout, stderr } = await execAsync(command, {
      cwd: path.dirname(rhubarbPath),
      timeout: 60000 // Increased timeout to 60 seconds
    });

    console.log('Rhubarb stdout:', stdout);
    if (stderr) {
      console.log('Rhubarb stderr:', stderr);
    }

    // Step 4: Check if output file was created
    if (!fs.existsSync(outputFilePath)) {
      throw new Error(`Rhubarb did not generate output file: ${outputFilePath}`);
    }

    // Step 5: Read the generated lip sync data
    const lipSyncData = JSON.parse(fs.readFileSync(outputFilePath, 'utf8'));
    console.log(`Lip sync generation completed in ${new Date().getTime() - time}ms`);
    console.log(`Generated ${lipSyncData.mouthCues ? lipSyncData.mouthCues.length : 0} mouth cues`);
    
    // Step 6: Cleanup temporary files
    try {
      // Remove temporary WAV file
      if (fs.existsSync(wavFilePath)) {
        fs.unlinkSync(wavFilePath);
        console.log('Cleaned up WAV file');
      }
      
      // Remove temporary text file
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
      outputFile: outputFilename,
      lipSyncPath: `/lipsync/${outputFilename}`,
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
 * Get existing lip sync data
 */
const getLipSyncData = async (req, res) => {
  try {
    const { filename } = req.params;
    const lipSyncFilePath = path.join(lipSyncDir, filename);

    if (!fs.existsSync(lipSyncFilePath)) {
      return res.status(404).json({
        success: false,
        error: 'Lip sync file not found'
      });
    }

    const lipSyncData = JSON.parse(fs.readFileSync(lipSyncFilePath, 'utf8'));
    
    res.json({
      success: true,
      lipSyncData: lipSyncData
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
  convertMp3ToWav
};
