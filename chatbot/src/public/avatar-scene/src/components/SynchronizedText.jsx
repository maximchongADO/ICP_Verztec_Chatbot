import React, { useEffect, useState, useRef } from 'react';

export const SynchronizedText = ({ text, audio, lipsync, isPlaying }) => {
  const [displayedWords, setDisplayedWords] = useState([]);
  const [currentWordIndex, setCurrentWordIndex] = useState(-1);
  const [words, setWords] = useState([]);
  const [hasFinishedPlaying, setHasFinishedPlaying] = useState(false);
  const audioRef = useRef(null);
  const intervalRef = useRef(null);
  const startTimeRef = useRef(null);

  // Initialize words and reset display
  useEffect(() => {
    if (text) {
      const wordArray = text.split(' ').filter(word => word.trim() !== '');
      setWords(wordArray);
      setDisplayedWords([]);
      setCurrentWordIndex(-1);
      setHasFinishedPlaying(false);
      console.log('Text initialized:', wordArray.length, 'words');
    }
  }, [text]);

  // Create audio element when audio data is available
  useEffect(() => {
    if (audio) {
      if (audioRef.current) {
        audioRef.current.pause();
        audioRef.current = null;
      }
      audioRef.current = new Audio("data:audio/mp3;base64," + audio);
      console.log('Audio element created');
    }
  }, [audio]);

  // Main synchronization logic
  useEffect(() => {
    if (isPlaying && words.length > 0) {
      console.log('Starting synchronized text display with', words.length, 'words');
      
      // Reset state
      setDisplayedWords([]);
      setCurrentWordIndex(-1);
      setHasFinishedPlaying(false);
      startTimeRef.current = Date.now();
      
      // Calculate word timings
      const wordDurations = calculateWordTimings(words, lipsync);
      console.log('Word durations:', wordDurations);
      
      let wordIndex = 0;
      
      // Function to show next word
      const showNextWord = () => {
        if (wordIndex < words.length) {
          console.log(`Showing word ${wordIndex}: "${words[wordIndex]}"`);
          
          setDisplayedWords(prev => [...prev, words[wordIndex]]);
          setCurrentWordIndex(wordIndex);
          
          wordIndex++;
          
          // Schedule next word
          if (wordIndex < words.length) {
            const nextDelay = wordDurations[wordIndex] || 300;
            setTimeout(showNextWord, nextDelay);
          } else {
            // All words shown, mark as finished but keep displayed
            setTimeout(() => {
              setCurrentWordIndex(-1);
              setHasFinishedPlaying(true);
              console.log('Text display finished, keeping all words visible');
            }, 500);
          }
        }
      };
      
      // Start showing words after a brief delay
      setTimeout(showNextWord, 200);
      
    } else if (!isPlaying && !hasFinishedPlaying) {
      // Only reset if we haven't finished playing yet
      console.log('Audio stopped before completion, resetting text display');
      setDisplayedWords([]);
      setCurrentWordIndex(-1);
      
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    } else if (!isPlaying && hasFinishedPlaying) {
      // Keep all words displayed after finishing
      console.log('Keeping all words displayed after TTS completion');
      setDisplayedWords(words);
      setCurrentWordIndex(-1);
    }

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, [isPlaying, words, lipsync, hasFinishedPlaying]);

  // If we have finished playing or not currently playing, show all words
  useEffect(() => {
    if (words.length > 0 && (hasFinishedPlaying || (!isPlaying && displayedWords.length === 0))) {
      console.log('Displaying all words (finished or not playing)');
      setDisplayedWords(words);
      setCurrentWordIndex(-1);
    }
  }, [words, hasFinishedPlaying, isPlaying, displayedWords.length]);

  // Calculate timing for each word
  const calculateWordTimings = (wordArray, lipSyncData) => {
    if (!lipSyncData || !lipSyncData.mouthCues || lipSyncData.mouthCues.length === 0) {
      // Fallback: 250ms per word
      console.log('Using fallback timing: 250ms per word');
      return wordArray.map(() => 250);
    }

    const mouthCues = lipSyncData.mouthCues;
    const totalDuration = Math.max(...mouthCues.map(cue => cue.end)) * 1000; // Convert to milliseconds
    const averageWordDuration = totalDuration / wordArray.length;
    
    console.log(`Total duration: ${totalDuration}ms, Average per word: ${averageWordDuration}ms`);
    
    // Adjust timing based on word length and complexity
    return wordArray.map((word, index) => {
      let duration = averageWordDuration;
      
      // Longer words get slightly more time
      if (word.length > 6) duration *= 1.2;
      else if (word.length < 3) duration *= 0.8;
      
      // Punctuation pauses
      if (word.includes(',') || word.includes('.') || word.includes('!') || word.includes('?')) {
        duration *= 1.5;
      }
      
      // Ensure minimum and maximum durations
      duration = Math.max(150, Math.min(500, duration));
      
      return Math.round(duration);
    });
  };

  if (!text) return null;

  return (
    <div className="synchronized-text-container">
      <div className="text-display">
        {displayedWords.map((word, index) => (
          <span
            key={`${word}-${index}`}
            className={`word ${
              index === currentWordIndex ? 'current' : 
              index < displayedWords.length || hasFinishedPlaying ? 'displayed' : ''
            }`}
          >
            {word}
            {index < displayedWords.length - 1 ? ' ' : ''}
          </span>
        ))}
        {/* Show cursor only when actively playing and not finished */}
        {isPlaying && displayedWords.length < words.length && !hasFinishedPlaying && (
          <span className="typing-cursor">|</span>
        )}
      </div>
      
      {/* Progress indicator - show completion when finished */}
      <div className="progress-container">
        <div className="progress-bar">
          <div 
            className="progress-fill"
            style={{
              width: `${hasFinishedPlaying ? 100 : (displayedWords.length / words.length) * 100}%`,
            }}
          />
        </div>
        <div className="progress-text">
          {hasFinishedPlaying ? `${words.length} / ${words.length}` : `${displayedWords.length} / ${words.length}`}
        </div>
      </div>
    </div>
  );
};
