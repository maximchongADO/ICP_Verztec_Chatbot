// Test the /chatbot_avatar endpoint
const fetch = require('node-fetch');

async function testChatbotAvatar() {
    try {
        console.log('🧪 Testing Chatbot Avatar Integration...');
        
        // Test the chatbot avatar endpoint
        const response = await fetch('http://localhost:8000/chatbot_avatar_test', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                message: 'Hello, this is a test message for the avatar TTS and lipsync integration.'
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        console.log('✅ API Response received:', data);
        
        if (data.messages && data.messages.length > 0) {
            const message = data.messages[0];
            console.log('- text:', message.text ? 'Present' : 'Missing');
            console.log('- audio:', message.audio ? 'Present' : 'Missing');
            console.log('- lipsync:', message.lipsync ? 'Present' : 'Missing');
            
            if (message.lipsync && message.lipsync.mouthCues) {
                console.log('- mouth cues count:', message.lipsync.mouthCues.length);
            }
        } else {
            console.log('No messages in response');
        }
        
    } catch (error) {
        console.error('Test failed:', error);
    }
}

// Auto-run the test
testChatbotAvatar();
