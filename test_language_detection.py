#!/usr/bin/env python3
"""
Test script for improved language detection functionality
"""

import sys
import os

# Add the chatbot directory to Python path
chatbot_dir = os.path.join(os.path.dirname(__file__), 'chatbot', 'src', 'backend', 'python')
sys.path.append(chatbot_dir)

# Import the function (note: this will also import all dependencies)
try:
    from chatbot import detect_language_improved
    print("✅ Successfully imported detect_language_improved function")
except ImportError as e:
    print(f"❌ Failed to import: {e}")
    sys.exit(1)

# Test cases
test_cases = [
    # English phrases that were previously misidentified
    "hello",
    "hi there",
    "good morning",
    "thank you",
    "how are you",
    "what is the policy",
    "can you help me",
    
    # Legitimate non-English queries
    "休假政策是什么",  # Chinese
    "Bonjour comment allez-vous",  # French
    "Hola como estas",  # Spanish
    
    # Mixed/ambiguous cases
    "ok",
    "yes",
    "no",
    "help"
]

print("\n🧪 Testing improved language detection:")
print("=" * 60)

for query in test_cases:
    try:
        detected_lang, is_english = detect_language_improved(query)
        status = "✅" if (query in ["hello", "hi there", "good morning", "thank you", "how are you", "what is the policy", "can you help me", "ok", "yes", "no", "help"] and is_english) or (query in ["休假政策是什么", "Bonjour comment allez-vous", "Hola como estas"] and not is_english) else "⚠️"
        print(f"{status} '{query}' → {detected_lang} (English: {is_english})")
    except Exception as e:
        print(f"❌ '{query}' → Error: {e}")

print("=" * 60)
print("✅ Language detection test completed!")
