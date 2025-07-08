#!/usr/bin/env python3
"""
Test script to verify Ollama connection and model availability.
Run this script to check if your local Ollama instance is working properly.
"""

import requests
import json
from langchain_ollama import ChatOllama

def test_ollama_connection():
    """Test direct connection to Ollama API"""
    try:
        # Test basic connection
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            models = response.json()
            print("✅ Ollama is running!")
            print(f"Available models: {[model['name'] for model in models.get('models', [])]}")
            return True
        else:
            print(f"❌ Ollama API returned status code: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to Ollama. Make sure it's running on localhost:11434")
        return False
    except Exception as e:
        print(f"❌ Error connecting to Ollama: {e}")
        return False

def test_langchain_ollama():
    """Test LangChain integration with Ollama"""
    try:
        # Test the specific model configuration from chatbot.py
        llm = ChatOllama(
            model="deepseek-r1:7b",
            base_url="http://localhost:11434"
        )
        
        # Test a simple query
        from langchain.schema import HumanMessage
        messages = [HumanMessage(content="Hello, can you respond with just 'test successful'?")]
        
        response = llm.invoke(messages)
        print("✅ LangChain-Ollama integration working!")
        print(f"Response: {response.content}")
        return True
        
    except Exception as e:
        print(f"❌ LangChain-Ollama test failed: {e}")
        return False

def main():
    print("🔍 Testing Ollama Setup for Verztec Chatbot")
    print("=" * 50)
    
    # Test 1: Direct API connection
    print("\n1. Testing Ollama API connection...")
    api_ok = test_ollama_connection()
    
    if not api_ok:
        print("\n💡 Troubleshooting tips:")
        print("   - Make sure Ollama is installed: https://ollama.ai/")
        print("   - Start Ollama: 'ollama serve'")
        print("   - Pull the model: 'ollama pull deepseek-r1:7b'")
        return
    
    # Test 2: LangChain integration
    print("\n2. Testing LangChain integration...")
    langchain_ok = test_langchain_ollama()
    
    if langchain_ok:
        print("\n🎉 All tests passed! Your chatbot should work with Ollama.")
    else:
        print("\n💡 LangChain integration failed. Try:")
        print("   - pip install langchain-ollama")
        print("   - Check if the model 'deepseek-r1:7b' is available")

if __name__ == "__main__":
    main()
