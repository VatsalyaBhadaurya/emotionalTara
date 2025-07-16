import speech_recognition as sr
from google.cloud import speech
import json
import os

LANGUAGE_FILE = "lang.json"

# Supported language options with TTS voices
LANGUAGE_OPTIONS = {
    "hi-IN": {"name": "Hindi", "voice": "hi-IN-SwaraNeural"},
    "ml-IN": {"name": "Malayalam", "voice": "ml-IN-SobhanaNeural"},
    "ta-IN": {"name": "Tamil", "voice": "ta-IN-PallaviNeural"},
    "en-US": {"name": "English", "voice": "en-US-JennyNeural"},
}

# Set your Google credentials file
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "key.json"

def test_language_detection():
    recognizer = sr.Recognizer()
    print("🎤 Speak something in Hindi, Malayalam, Tamil, or English...")

    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
        audio_data = audio.get_wav_data()

    client = speech.SpeechClient()
    candidates = []

    for lang_code, data in LANGUAGE_OPTIONS.items():
        try:
            print(f"\n🔍 Trying {data['name']} [{lang_code}]...")
            audio_request = speech.RecognitionAudio(content=audio_data)
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=44100,
                language_code=lang_code
            )

            response = client.recognize(config=config, audio=audio_request)

            if response.results:
                alt = response.results[0].alternatives[0]
                transcript = alt.transcript
                confidence = alt.confidence if alt.confidence else 0.0
                print(f"🗣️ {data['name']} recognized: {transcript} (confidence: {confidence:.2f})")
                candidates.append({
                    "language_code": lang_code,
                    "voice": data['voice'],
                    "name": data['name'],
                    "transcript": transcript,
                    "confidence": confidence
                })

        except Exception as e:
            print(f"❌ {data['name']} failed: {e}")

    if not candidates:
        print("\n❌ No valid transcription detected.")
        return

    # Choose the result with highest confidence
    best = sorted(candidates, key=lambda x: x['confidence'], reverse=True)[0]
    print(f"\n✅ Best Match: {best['name']} ({best['language_code']})")
    print(f"📝 Transcript: {best['transcript']}")

    with open(LANGUAGE_FILE, 'w') as f:
        json.dump({
            'language_code': best['language_code'],
            'voice': best['voice']
        }, f, indent=4)

    print(f"💾 Saved language and voice to {LANGUAGE_FILE}")

if __name__ == "__main__":
    test_language_detection()

