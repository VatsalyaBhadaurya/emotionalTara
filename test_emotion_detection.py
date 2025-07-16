#!/usr/bin/env python3
"""
Test script for emotion detection
Tests both advanced and keyword-based detection methods
"""

from emotional_assistant import EmotionalAssistant

def test_emotion_detection():
    """Test emotion detection with various phrases"""
    print("🧪 Testing Emotion Detection")
    print("=" * 50)
    
    # Test phrases with expected emotions
    test_cases = [
        ("I'm feeling really happy today!", "joy"),
        ("I'm so sad and lonely right now", "sadness"),
        ("I'm really angry about what happened", "anger"),
        ("I'm scared and worried about the future", "fear"),
        ("Wow, that's amazing news!", "surprise"),
        ("Then I lower himself side", "neutral"),  # The problematic phrase
        ("I love this so much!", "joy"),
        ("This is terrible and awful", "sadness"),
        ("I hate this stupid thing", "anger"),
        ("I'm afraid of what might happen", "fear"),
        ("I can't believe this happened!", "surprise"),
        ("I'm feeling okay today", "neutral"),
        ("This makes me so frustrated", "anger"),
        ("I'm grateful for your help", "joy"),
        ("I feel lost and alone", "sadness"),
    ]
    
    # Test both detection methods
    methods = [
        ("Advanced", True),
        ("Keyword-based", False)
    ]
    
    for method_name, use_advanced in methods:
        print(f"\n🔍 Testing {method_name} Emotion Detection:")
        print("-" * 40)
        
        try:
            assistant = EmotionalAssistant(
                use_advanced_emotion=use_advanced,
                enable_caching=False  # Disable caching for testing
            )
            
            correct = 0
            total = len(test_cases)
            
            for phrase, expected_emotion in test_cases:
                print(f"\nTesting: '{phrase}'")
                print(f"Expected: {expected_emotion}")
                
                emotion, confidence = assistant.detect_emotion(phrase)
                
                is_correct = emotion == expected_emotion
                status = "✅" if is_correct else "❌"
                
                print(f"Detected: {emotion} (confidence: {confidence:.3f}) {status}")
                
                if is_correct:
                    correct += 1
            
            accuracy = (correct / total) * 100
            print(f"\n📊 {method_name} Accuracy: {correct}/{total} ({accuracy:.1f}%)")
            
        except Exception as e:
            print(f"❌ Error testing {method_name}: {e}")
    
    print("\n" + "=" * 50)
    print("🎯 Emotion Detection Test Complete!")

if __name__ == "__main__":
    test_emotion_detection() 