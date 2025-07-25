/**
 * Test file to verify Avatar TTS Integration
 * This demonstrates how the chatbot text input connects to TTS and Lipsync systems
 */

console.log('🧪 Testing Avatar TTS Integration...');

// Test configuration
const testConfig = {
  backendUrl: 'http://localhost:8000',
  testMessage: 'Hello! This is a test message for the avatar TTS and lipsync integration.',
  avatarWindowUrl: '/avatar-scene/index.html'
};

// Test functions
async function testTTSEndpoint() {
  console.log('\n1️⃣ Testing TTS Endpoint...');
  
  try {
    const token = localStorage.getItem('token');
    if (!token) {
      console.warn('⚠️ No authentication token found. Please login first.');
      return false;
    }

    const response = await fetch(`${testConfig.backendUrl}/api/tts/synthesize-enhanced`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${token}`
      },
      body: JSON.stringify({
        text: testConfig.testMessage,
        voice: 'en-GB-Standard-A',
        languageCode: 'en-GB',
        generateLipSyncData: true,
        facialExpression: 'default',
        animation: 'Talking_1'
      })
    });

    if (response.ok) {
      const data = await response.json();
      console.log('✅ TTS Endpoint working:', {
        hasAudio: !!data.audio,
        hasLipsync: !!data.lipSyncData,
        audioSize: data.audio ? data.audio.length : 0,
        lipsyncCues: data.lipSyncData ? data.lipSyncData.mouthCues?.length : 0
      });
      return true;
    } else {
      console.error('❌ TTS Endpoint failed:', response.status);
      return false;
    }
  } catch (error) {
    console.error('❌ TTS Endpoint error:', error);
    return false;
  }
}

async function testChatbotEndpoint() {
  console.log('\n2️⃣ Testing Chatbot Avatar Endpoint...');
  
  try {
    const response = await fetch(`${testConfig.backendUrl}/chatbot_avatar_test`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        message: testConfig.testMessage
      })
    });

    if (response.ok) {
      const data = await response.json();
      console.log('✅ Chatbot Avatar Endpoint working:', {
        messageCount: data.messages ? data.messages.length : 0,
        hasMessages: !!(data.messages && data.messages.length > 0)
      });
      return true;
    } else {
      console.error('❌ Chatbot Avatar Endpoint failed:', response.status);
      return false;
    }
  } catch (error) {
    console.error('❌ Chatbot Avatar Endpoint error:', error);
    return false;
  }
}

function testMessageCommunication() {
  console.log('\n3️⃣ Testing Message Communication...');
  
  // Check if we can send messages to avatar
  const sendMessageToAvatar = window.sendMessageToAvatar;
  if (typeof sendMessageToAvatar === 'function') {
    console.log('✅ sendMessageToAvatar function available');
    return true;
  } else {
    console.log('⚠️ sendMessageToAvatar function not found - this is normal if not on main chatbot page');
    return false;
  }
}

function testAvatarWindow() {
  console.log('\n4️⃣ Testing Avatar Window...');
  
  try {
    // Test opening avatar window
    const testWindow = window.open(testConfig.avatarWindowUrl, 'testWindow', 'width=400,height=300');
    
    if (testWindow) {
      console.log('✅ Avatar window can be opened');
      
      // Test message sending after a delay
      setTimeout(() => {
        try {
          testWindow.postMessage({
            type: 'chat_message',
            payload: 'Test message from integration test'
          }, window.location.origin);
          console.log('✅ Message sent to avatar window');
          
          // Close test window
          setTimeout(() => {
            testWindow.close();
            console.log('✅ Test window closed');
          }, 2000);
        } catch (error) {
          console.error('❌ Failed to send message to avatar window:', error);
          testWindow.close();
        }
      }, 1000);
      
      return true;
    } else {
      console.error('❌ Failed to open avatar window');
      return false;
    }
  } catch (error) {
    console.error('❌ Avatar window test error:', error);
    return false;
  }
}

// Main test function
async function runIntegrationTests() {
  console.log('🚀 Starting Avatar TTS Integration Tests...\n');
  
  const results = {
    tts: await testTTSEndpoint(),
    chatbot: await testChatbotEndpoint(),
    communication: testMessageCommunication(),
    avatarWindow: testAvatarWindow()
  };
  
  console.log('\n📊 Test Results Summary:');
  console.log('='.repeat(50));
  console.log(`TTS Endpoint:        ${results.tts ? '✅ PASS' : '❌ FAIL'}`);
  console.log(`Chatbot Endpoint:    ${results.chatbot ? '✅ PASS' : '❌ FAIL'}`);
  console.log(`Message Communication: ${results.communication ? '✅ PASS' : '⚠️ SKIP'}`);
  console.log(`Avatar Window:       ${results.avatarWindow ? '✅ PASS' : '❌ FAIL'}`);
  
  const passCount = Object.values(results).filter(Boolean).length;
  const totalTests = Object.keys(results).length;
  
  console.log(`\n📈 Overall: ${passCount}/${totalTests} tests passed`);
  
  if (results.tts && results.chatbot) {
    console.log('\n🎉 Core integration is working! You can now:');
    console.log('   1. Type messages in the main chatbot');
    console.log('   2. Open the Avatar Scene window');
    console.log('   3. Messages will be sent to avatar with TTS and lipsync');
    console.log('   4. Avatar will speak and animate based on the text');
  } else {
    console.log('\n⚠️ Some core features need attention before full integration works.');
  }
  
  return results;
}

// Auto-run tests if this file is executed directly
if (typeof window !== 'undefined') {
  // Browser environment
  window.testAvatarIntegration = runIntegrationTests;
  
  // Run tests automatically after a short delay
  setTimeout(() => {
    console.log('Running automatic integration tests...');
    runIntegrationTests();
  }, 1000);
} else {
  // Node.js environment
  console.log('This test is designed to run in a browser environment.');
  console.log('Please run it from the browser console or include it in a web page.');
}

// Export for manual testing
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { runIntegrationTests, testConfig };
}
