// Test the /chatbot_avatar endpoint
const fetch = require('node-fetch');

async function testChatbotAvatar() {
    try {
        console.log('Testing /chatbot_avatar_test endpoint...');
        
        const response = await fetch('http://localhost:8000/chatbot_avatar_test', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            },
            body: JSON.stringify({
                message: 'Hello, how are you today?'
            })
        });

        console.log('Response status:', response.status);
        console.log('Response headers:', response.headers.raw());
        
        const data = await response.json();
        console.log('Response data:', JSON.stringify(data, null, 2));
        
        // Check if we got the expected fields
        console.log('\nChecking response fields:');
        if (data.messages && data.messages.length > 0) {
            const message = data.messages[0];
            console.log('- type:', message.type);
            console.log('- text:', message.text ? 'Present' : 'Missing');
            console.log('- audio:', message.audio ? 'Present' : 'Missing');
            console.log('- lipsync:', message.lipsync ? 'Present' : 'Missing');
        } else {
            console.log('No messages in response');
        }
        
    } catch (error) {
        console.error('Test failed:', error);
    }
}

testChatbotAvatar();
