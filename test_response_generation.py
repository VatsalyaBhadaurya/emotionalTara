#!/usr/bin/env python3
"""
Test script for response generation
Tests both LangChain and fallback response generation
"""

from emotional_assistant import EmotionalAssistant

def test_response_generation():
    """Test response generation with various emotions"""
    print("🧪 Testing Response Generation")
    print("=" * 50)
    
    # Test cases with different emotions
    test_cases = [
        ("I'm feeling really happy today!", "joy"),
        ("I'm so sad and lonely right now", "sadness"),
        ("I'm really angry about what happened", "anger"),
        ("I'm scared and worried about the future", "fear"),
        ("Wow, that's amazing news!", "surprise"),
        ("I'm feeling okay today", "neutral"),
    ]
    
    # Test both LangChain and fallback
    methods = [
        ("LangChain", True),
        ("Fallback Only", False)
    ]
    
    for method_name, use_langchain in methods:
        print(f"\n🔍 Testing {method_name} Response Generation:")
        print("-" * 40)
        
        try:
            assistant = EmotionalAssistant(
                use_advanced_emotion=False,  # Use keyword-based for consistency
                enable_caching=False
            )
            
            # Disable LangChain if testing fallback only
            if not use_langchain:
                assistant.qwen_available = False
            
            for user_text, expected_emotion in test_cases:
                print(f"\nUser: '{user_text}'")
                print(f"Expected Emotion: {expected_emotion}")
                
                # Detect emotion
                emotion, confidence = assistant.detect_emotion(user_text)
                print(f"Detected Emotion: {emotion} (confidence: {confidence:.2f})")
                
                # Generate response
                response = assistant.generate_response(user_text, emotion)
                print(f"Response: '{response}'")
                print(f"Response Length: {len(response)} characters")
                
                # Check if response is complete
                if len(response) < 20:
                    print("⚠️ Response seems too short")
                elif len(response) > 200:
                    print("⚠️ Response seems too long")
                else:
                    print("✅ Response length looks good")
                
                print("-" * 30)
            
        except Exception as e:
            print(f"❌ Error testing {method_name}: {e}")
    
    print("\n" + "=" * 50)
    print("🎯 Response Generation Test Complete!")

if __name__ == "__main__":
    test_response_generation() 