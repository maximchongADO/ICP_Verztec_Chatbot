// Test the /chatbot_avatar endpoint
const fetch = require('node-fetch');

async function testChatbotAvatar() {
    try {
        console.log('Testing /chatbot_avatar endpoint...');
        
        const response = await fetch('http://localhost:8000/chatbot_avatar', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json',
                // For testing without auth - we'll add this to bypass auth
                'Authorization': 'Bearer test-token'
            },
            body: JSON.stringify({
                message: 'Hello, how are you?',
                user_id: 'test-user',
                chat_id: 'test-chat'
            })
        });

        console.log('Response status:', response.status);
        console.log('Response headers:', response.headers.raw());
        
        const data = await response.json();
        console.log('Response data:', JSON.stringify(data, null, 2));
        
        // Check if we got the expected fields
        console.log('\nChecking response fields:');
        console.log('- success:', data.success);
        console.log('- text:', data.text ? 'Present' : 'Missing');
        console.log('- audio:', data.audio ? 'Present' : 'Missing');
        console.log('- lipsync:', data.lipsync ? 'Present' : 'Missing');
        
    } catch (error) {
        console.error('Test failed:', error);
    }
}

testChatbotAvatar();
