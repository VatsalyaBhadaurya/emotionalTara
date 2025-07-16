#!/usr/bin/env python3
"""
Performance Test Script for Emotional AI Voice Assistant
Compares optimized vs non-optimized performance
"""

import time
import json
from emotional_assistant import EmotionalAssistant

def test_performance():
    """Test performance of optimized assistant"""
    print("🚀 Performance Test - Optimized Emotional AI Assistant")
    print("=" * 60)
    
    # Test configurations
    configs = [
        {"name": "Optimized (LangChain + Caching)", "use_advanced_emotion": True, "enable_caching": True},
        {"name": "Optimized (Direct Qwen + Caching)", "use_advanced_emotion": True, "enable_caching": True},
        {"name": "Basic (No Caching)", "use_advanced_emotion": False, "enable_caching": False},
    ]
    
    # Test phrases with different emotions
    test_phrases = [
        "I'm feeling really happy today!",
        "I'm so sad and lonely right now",
        "I'm really angry about what happened",
        "I'm scared and worried about the future",
        "Wow, that's amazing news!"
    ]
    
    results = {}
    
    for config in configs:
        print(f"\n🧪 Testing: {config['name']}")
        print("-" * 40)
        
        try:
            # Initialize assistant
            start_time = time.time()
            assistant = EmotionalAssistant(
                use_advanced_emotion=config['use_advanced_emotion'],
                enable_caching=config['enable_caching']
            )
            init_time = time.time() - start_time
            
            # Test each phrase
            phrase_results = []
            for phrase in test_phrases:
                print(f"  Testing: '{phrase}'")
                
                # Time emotion detection
                emotion_start = time.time()
                emotion, confidence = assistant.detect_emotion(phrase)
                emotion_time = time.time() - emotion_start
                
                # Time response generation
                response_start = time.time()
                response = assistant.generate_response(phrase, emotion)
                response_time = time.time() - response_start
                
                phrase_results.append({
                    "phrase": phrase,
                    "emotion": emotion,
                    "confidence": confidence,
                    "response": response,
                    "emotion_time": emotion_time,
                    "response_time": response_time,
                    "total_time": emotion_time + response_time
                })
                
                print(f"    Emotion: {emotion} ({emotion_time:.2f}s)")
                print(f"    Response: {response} ({response_time:.2f}s)")
            
            # Calculate averages
            avg_emotion_time = sum(r['emotion_time'] for r in phrase_results) / len(phrase_results)
            avg_response_time = sum(r['response_time'] for r in phrase_results) / len(phrase_results)
            avg_total_time = sum(r['total_time'] for r in phrase_results) / len(phrase_results)
            
            results[config['name']] = {
                "init_time": init_time,
                "avg_emotion_time": avg_emotion_time,
                "avg_response_time": avg_response_time,
                "avg_total_time": avg_total_time,
                "phrase_results": phrase_results
            }
            
            print(f"  📊 Averages:")
            print(f"    Emotion Detection: {avg_emotion_time:.2f}s")
            print(f"    Response Generation: {avg_response_time:.2f}s")
            print(f"    Total Processing: {avg_total_time:.2f}s")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            results[config['name']] = {"error": str(e)}
    
    # Save results
    with open("performance_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 60)
    print("📈 PERFORMANCE SUMMARY")
    print("=" * 60)
    
    for config_name, result in results.items():
        if "error" not in result:
            print(f"\n{config_name}:")
            print(f"  Init Time: {result['init_time']:.2f}s")
            print(f"  Avg Emotion Detection: {result['avg_emotion_time']:.2f}s")
            print(f"  Avg Response Generation: {result['avg_response_time']:.2f}s")
            print(f"  Avg Total Processing: {result['avg_total_time']:.2f}s")
        else:
            print(f"\n{config_name}: ❌ {result['error']}")
    
    print(f"\n📄 Detailed results saved to: performance_results.json")

if __name__ == "__main__":
    test_performance() 