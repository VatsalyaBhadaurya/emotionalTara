#!/usr/bin/env python3
"""
Test script for raw emotion detection and response generation
Tests that raw RoBERTa emotions are used directly
"""

from emotional_assistant import EmotionalAssistant

def test_raw_emotions():
    """Test raw emotion detection and response generation"""
    print("🧪 Testing Raw Emotion Detection")
    print("=" * 50)
    
    # Test phrases that should trigger specific raw emotions
    test_cases = [
        ("I'm feeling really happy today!", "joy"),
        ("I'm so sad and lonely right now", "sadness"),
        ("I'm really angry about what happened", "anger"),
        ("I'm scared and worried about the future", "fear"),
        ("Wow, that's amazing news!", "surprise"),
        ("This is disgusting and awful", "disgust"),
        ("I'm feeling okay today", "neutral"),
    ]
    
    print("🔍 Testing Advanced Emotion Detection (Raw Emotions):")
    print("-" * 50)
    
    try:
        assistant = EmotionalAssistant(
            use_advanced_emotion=True,  # Use RoBERTa model
            enable_caching=False
        )
        
        for user_text, expected_emotion in test_cases:
            print(f"\nUser: '{user_text}'")
            print(f"Expected Raw Emotion: {expected_emotion}")
            
            # Detect emotion
            emotion, confidence = assistant.detect_emotion(user_text)
            print(f"Detected Raw Emotion: {emotion} (confidence: {confidence:.3f})")
            
            # Generate response
            response = assistant.generate_response(user_text, emotion)
            print(f"Response: '{response}'")
            
            # Check if emotion matches expected
            if emotion == expected_emotion:
                print("✅ Raw emotion matches expected")
            else:
                print(f"⚠️ Raw emotion differs: expected {expected_emotion}, got {emotion}")
            
            print("-" * 40)
        
    except Exception as e:
        print(f"❌ Error testing raw emotions: {e}")
    
    print("\n" + "=" * 50)
    print("🎯 Raw Emotion Test Complete!")

if __name__ == "__main__":
    test_raw_emotions() 