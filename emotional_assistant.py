#!/usr/bin/env python3
"""
Emotional AI Voice Assistant - Optimized Version with LangChain
Fast, efficient implementation with caching and parallel processing.
"""

import warnings
import whisper
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import time
import os
import hashlib
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import json
from google.cloud import speech  # Google STT for language detection
import threading  # For real-time pipeline

# Suppress LangChain deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain")

# LangChain imports
try:
    from langchain_community.llms import HuggingFacePipeline
    from langchain.memory import ConversationBufferMemory
    from langchain.prompts import PromptTemplate
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("⚠️ LangChain not available - using direct Qwen calls")

class EmotionalAssistant:
    """Optimized emotional AI voice assistant with caching and LangChain"""
    
    def __init__(self, use_advanced_emotion=True, enable_caching=True, enable_language_detection=True):
        """Initialize the assistant with optimizations and optional language detection"""
        print("🤖 Initializing Optimized Emotional AI Voice Assistant...")
        
        # Performance tracking and settings
        self.timings = {}
        self.enable_caching = enable_caching
        self.use_advanced_emotion = use_advanced_emotion
        self.enable_language_detection = enable_language_detection
        self.detected_language = None
        self.detected_voice = None
        # Set Google credentials if available
        if os.path.exists("key.json"):
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "key.json"
        
        # Initialize components
        self._init_whisper()
        self._init_emotion_detection()
        self._init_response_system()
        self._init_tts()
        
        # Initialize thread pool and load conversation history
        self.executor = ThreadPoolExecutor(max_workers=3)
        self.conversation_history = []
        self._load_previous_conversation()
        
        print("✅ Optimized initialization complete!")
    
    def _init_whisper(self):
        """Initialize Whisper for speech recognition"""
        print("🎙️ Loading optimized Whisper model...")
        import torch
        try:
            self.whisper_model = whisper.load_model("tiny")
            if torch.cuda.is_available():
                self.whisper_model = self.whisper_model.cuda()
                print(f"🚀 Whisper model moved to GPU: {next(self.whisper_model.parameters()).device}")
            else:
                print("⚠️ CUDA not available, running Whisper on CPU (will be slow)")
        except Exception as e:
            print(f"⚠️ Whisper model failed to load on GPU: {e}")
            print("🔄 Falling back to CPU mode for Whisper...")
            self.whisper_model = whisper.load_model("tiny")
    
    def _init_emotion_detection(self):
        """Initialize emotion detection with caching"""
        # Initialize emotion keywords for both methods
        self.emotion_keywords = {
            "joy": ["happy", "excited", "great", "wonderful", "amazing", "fantastic", "love", "joy", "pleased", "delighted", "good", "nice", "awesome", "excellent", "thrilled", "cheerful", "glad", "grateful", "content", "satisfied", "optimistic", "positive", "upbeat", "better", "improved", "success", "win", "victory", "celebrate", "blessed", "lucky", "fortunate"],
            "sadness": ["sad", "depressed", "lonely", "miserable", "unhappy", "crying", "tears", "grief", "sorrow", "down", "bad", "terrible", "awful", "hopeless", "gloomy", "heartbroken", "devastated", "disappointed", "hurt", "pain", "suffering", "despair", "defeated", "lost", "empty", "worthless", "useless", "failure", "defeat", "rejected", "abandoned", "alone", "miss", "gone", "died", "death"],
            "anger": ["angry", "mad", "furious", "rage", "hate", "annoyed", "irritated", "frustrated", "upset", "livid", "dislike", "hostile", "aggressive", "outraged", "enraged", "fuming", "bothered", "disgusted", "resentful", "bitter", "hate", "despise", "loathe", "kill", "destroy", "fight", "attack", "blame", "blamed", "unfair", "injustice", "wrong", "stupid", "idiot", "dumb"],
            "fear": ["afraid", "scared", "terrified", "fear", "panic", "anxious", "worried", "nervous", "frightened", "horror", "scary", "terrifying", "alarming", "threatened", "intimidated", "overwhelmed", "stressed", "tense", "uneasy", "danger", "dangerous", "risk", "risky", "threat", "threatened", "attack", "hurt", "pain", "die", "death", "kill", "killed", "lost", "lose", "losing", "fail", "failure", "embarrassed", "embarrassing"],
            "surprise": ["surprised", "shocked", "amazed", "astonished", "wow", "unexpected", "incredible", "unbelievable", "omg", "stunned", "startled", "bewildered", "confused", "puzzled", "astounded", "speechless", "really", "actually", "seriously", "no way", "impossible", "can't believe", "never thought", "out of nowhere", "suddenly", "just", "just happened"]
        }
        
        if self.use_advanced_emotion:
            try:
                from transformers import pipeline
                print("😐 Loading advanced emotion detection model...")
                self.emotion_detector = pipeline(
                    "text-classification", 
                    model="j-hartmann/emotion-english-distilroberta-base", 
                    top_k=1
                )
                print("✅ Advanced emotion detection loaded")
            except Exception as e:
                print(f"⚠️ Advanced emotion detection failed: {e}")
                print("🔄 Falling back to keyword-based detection")
                self.use_advanced_emotion = False
        
        if not self.use_advanced_emotion:
            print("😐 Using keyword-based emotion detection...")

    def _init_response_system(self):
        """Initialize Qwen LLM with LangChain integration"""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
            import torch
            print("🤖 Loading Qwen LLM with LangChain...")
            
            # Load Qwen 1.5B model and force GPU
            checkpoint = "Qwen/Qwen1.5-1.8B-Chat"
            self.qwen_tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True, add_prefix_space=False)
            self.qwen_model = AutoModelForCausalLM.from_pretrained(
                checkpoint, 
                trust_remote_code=True,
                torch_dtype=torch.float16 if torch.cuda.is_available() else "auto",
                device_map=None  # We'll move manually
            )
            if torch.cuda.is_available():
                self.qwen_model = self.qwen_model.cuda()
                print("🚀 Qwen model moved to GPU")
            else:
                print("⚠️ CUDA not available, running Qwen on CPU (will be slow)")
            
            # Create HuggingFace pipeline for LangChain
            self.qwen_pipeline = pipeline(
                "text-generation",
                model=self.qwen_model,
                tokenizer=self.qwen_tokenizer,
                max_new_tokens=50,  # Increased for complete responses
                temperature=0.6,    # Slightly lower for more focused responses
                do_sample=True,
                pad_token_id=self.qwen_tokenizer.eos_token_id,
                eos_token_id=self.qwen_tokenizer.eos_token_id,
                device=0 if torch.cuda.is_available() else -1
            )
            
            if LANGCHAIN_AVAILABLE:
                # Initialize LangChain components
                self.llm = HuggingFacePipeline(pipeline=self.qwen_pipeline)
                self.memory = ConversationBufferMemory(return_messages=True)
                
                # Create conversation chain with emotion-aware prompt
                prompt_template = """You are Talia, a caring AI friend. User feels {emotion}. 
Give a short, warm 1-2 sentence response.

{history}
Human: {input}
AI:"""
                
                self.prompt = PromptTemplate(
                    input_variables=["history", "input", "emotion"],
                    template=prompt_template
                )
                
                self.use_langchain_simple = True
                print("✅ LangChain simple integration ready")
            
            self.qwen_available = True
            
        except Exception as e:
            print(f"⚠️ Qwen LLM not available: {e}")
            self.qwen_available = False
    
    def _init_tts(self):
        """Initialize TTS system"""
        # Map raw RoBERTa emotions to TTS voice tones
        self.emotion_prompt = {
            "joy": "happy", 
            "sadness": "sad", 
            "anger": "angry", 
            "fear": "worried", 
            "surprise": "excited", 
            "disgust": "disgusted",
            "neutral": "neutral"
        }
        self.tts_available = False
        try:
            from parler_tts import ParlerTTSForConditionalGeneration
            from transformers import AutoTokenizer
            import torch
            print("🗣️ Loading TTS model...")
            self.tts_model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler-tts-mini-expresso")
            self.tts_tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-expresso")
            self.tts_model.eval()
            if torch.cuda.is_available():
                self.tts_model = self.tts_model.cuda()
                print(f"🚀 TTS model moved to GPU: {next(self.tts_model.parameters()).device}")
            else:
                print("⚠️ CUDA not available, running TTS on CPU (will be slow)")
            self.tts_available = True
            print("✅ TTS loaded successfully")
        except Exception as e:
            print(f"⚠️ TTS not available: {e}")
    
    @lru_cache(maxsize=100)
    def _cached_transcribe(self, audio_hash):
        """Cached transcription to avoid re-processing same audio"""
        return self.whisper_model.transcribe("input.wav")
    
    def record_audio(self, duration=5, sample_rate=16000):
        """Record audio from microphone"""
        start_time = time.time()
        print(f"🎤 Recording for {duration} seconds...")
        audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
        sd.wait()
        self.timings['recording'] = time.time() - start_time
        return audio, sample_rate
    
    def save_audio(self, audio, sample_rate, filename="input.wav"):
        """Save audio to file"""
        wav.write(filename, sample_rate, audio)
    
    def transcribe_audio(self, audio_file="input.wav"):
        """Convert speech to text using Whisper with caching"""
        start_time = time.time()
        print("🔊 Transcribing audio...")
        import torch
        try:
            if self.enable_caching:
                try:
                    with open(audio_file, 'rb') as f:
                        audio_hash = hashlib.md5(f.read()).hexdigest()
                    result = self._cached_transcribe(audio_hash)
                except:
                    result = self.whisper_model.transcribe(audio_file)
            else:
                result = self.whisper_model.transcribe(audio_file)
        except RuntimeError as e:
            if 'CUDA error' in str(e):
                print(f"⚠️ CUDA error in Whisper, falling back to CPU: {e}")
                self.whisper_model = whisper.load_model("tiny")
                result = self.whisper_model.transcribe(audio_file)
            else:
                raise
        text = result["text"].strip()
        self.timings['transcription'] = time.time() - start_time
        print(f"📝 Transcribed: '{text}' ({self.timings['transcription']:.2f}s)")
        return text
    
    @lru_cache(maxsize=1000)
    def _cached_emotion_detection(self, text):
        """Cached emotion detection"""
        if self.use_advanced_emotion and hasattr(self, 'emotion_detector'):
            return self._detect_emotion_advanced(text)
        else:
            return self._detect_emotion_keywords(text)
    
    def detect_emotion(self, text):
        """Detect emotion using configured method with caching"""
        start_time = time.time()
        print("❤️ Detecting emotion...")
        
        if self.enable_caching:
            emotion, confidence = self._cached_emotion_detection(text)
        else:
            if self.use_advanced_emotion and hasattr(self, 'emotion_detector'):
                emotion, confidence = self._detect_emotion_advanced(text)
                print(f"🔍 Using raw emotion: {emotion}")
            else:
                emotion, confidence = self._detect_emotion_keywords(text)
        
        self.timings['emotion_detection'] = time.time() - start_time
        print(f"😐 Final Emotion: {emotion} (confidence: {confidence:.2f}) ({self.timings['emotion_detection']:.2f}s)")
        return emotion, confidence
    
    def _detect_emotion_advanced(self, text):
        """Advanced emotion detection using transformers"""
        try:
            result = self.emotion_detector(text)
            print(f"🔍 Raw emotion result: {result}")
            
            # Handle different result formats and get the emotion with highest score
            if isinstance(result, list) and len(result) > 0:
                first_result = result[0]
                if isinstance(first_result, list) and len(first_result) > 0:
                    emotion_data = first_result[0]
                    if isinstance(emotion_data, dict) and "label" in emotion_data:
                        emotion = emotion_data["label"].lower()
                        confidence = emotion_data.get("score", 0.5)
                    else:
                        emotion = "neutral"
                        confidence = 0.5
                elif isinstance(first_result, dict) and "label" in first_result:
                    emotion = first_result["label"].lower()
                    confidence = first_result.get("score", 0.5)
                else:
                    emotion = "neutral"
                    confidence = 0.5
            elif isinstance(result, dict) and "label" in result:
                emotion = result["label"].lower()
                confidence = result.get("score", 0.5)
            else:
                emotion = "neutral"
                confidence = 0.5
            
            print(f"🔍 Raw emotion: {emotion} (confidence: {confidence:.3f})")
            return emotion, confidence
                
        except Exception as e:
            print(f"Error in advanced emotion detection: {e}")
            return "neutral", 0.5
    
    def _detect_emotion_keywords(self, text):
        """Keyword-based emotion detection with context awareness"""
        text_lower = text.lower()
        emotion_scores = {}
        
        # Count emotion keywords with better matching
        for emotion, keywords in self.emotion_keywords.items():
            score = 0
            matched_keywords = []
            for keyword in keywords:
                if f" {keyword} " in f" {text_lower} " or text_lower.startswith(keyword) or text_lower.endswith(keyword):
                    score += 1
                    matched_keywords.append(keyword)
            emotion_scores[emotion] = score
            if score > 0:
                print(f"🔍 {emotion}: {score} matches ({', '.join(matched_keywords)})")
        
        # Context-aware scoring
        context_indicators = {
            "joy": ["feel", "feeling", "am", "is", "are", "was", "were", "been", "being"],
            "sadness": ["feel", "feeling", "am", "is", "are", "was", "were", "been", "being"],
            "anger": ["feel", "feeling", "am", "is", "are", "was", "were", "been", "being", "make", "makes", "made"],
            "fear": ["feel", "feeling", "am", "is", "are", "was", "were", "been", "being", "afraid", "scared", "worried"],
            "surprise": ["wow", "really", "actually", "seriously", "unexpected", "suddenly"]
        }
        
        # Add context bonus for emotional statements
        for emotion, indicators in context_indicators.items():
            for indicator in indicators:
                if indicator in text_lower:
                    words = text_lower.split()
                    for i, word in enumerate(words):
                        if word == indicator:
                            for j in range(max(0, i-3), min(len(words), i+4)):
                                if j != i and any(keyword in words[j] for keyword in self.emotion_keywords[emotion]):
                                    emotion_scores[emotion] += 0.5
                                    print(f"🔍 Context bonus for {emotion}: '{indicator}' + emotion keyword")
                                    break
        
        print(f"🔍 All emotion scores: {emotion_scores}")
        
        # Find emotion with highest score
        if emotion_scores:
            max_score = max(emotion_scores.values())
            if max_score > 0:
                detected_emotion = max(emotion_scores, key=emotion_scores.get)
                word_count = len(text.split())
                confidence = min(1.0, max_score / max(word_count, 1))
                print(f"🔍 Detected: {detected_emotion} (score: {max_score}, confidence: {confidence:.3f})")
                return detected_emotion, confidence
        
        print(f"🔍 No emotion keywords found, defaulting to neutral")
        return "neutral", 0.1

    def detect_language_google(self, audio_file="input.wav", sample_rate=44100):
        """
        Detect spoken language using Google Cloud Speech-to-Text.
        Returns (language_code, voice_name, language_name, transcript, confidence) or None on failure.
        """
        LANGUAGE_OPTIONS = {
            "hi-IN": {"name": "Hindi", "voice": "hi-IN-SwaraNeural"},
            "ml-IN": {"name": "Malayalam", "voice": "ml-IN-SobhanaNeural"},
            "ta-IN": {"name": "Tamil", "voice": "ta-IN-PallaviNeural"},
            "en-US": {"name": "English", "voice": "en-US-JennyNeural"},
        }
        try:
            with open(audio_file, "rb") as f:
                audio_data = f.read()
            client = speech.SpeechClient()
            candidates = []
            for lang_code, data in LANGUAGE_OPTIONS.items():
                try:
                    audio_request = speech.RecognitionAudio(content=audio_data)
                    config = speech.RecognitionConfig(
                        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                        sample_rate_hertz=sample_rate,
                        language_code=lang_code
                    )
                    response = client.recognize(config=config, audio=audio_request)
                    if response.results:
                        alt = response.results[0].alternatives[0]
                        transcript = alt.transcript
                        confidence = alt.confidence if alt.confidence else 0.0
                        candidates.append({
                            "language_code": lang_code,
                            "voice": data['voice'],
                            "name": data['name'],
                            "transcript": transcript,
                            "confidence": confidence
                        })
                except Exception:
                    continue
            if not candidates:
                print("❌ No valid transcription detected by Google STT.")
                return None
            best = sorted(candidates, key=lambda x: x['confidence'], reverse=True)[0]
            print(f"✅ Google STT detected language: {best['name']} ({best['language_code']}) | Confidence: {best['confidence']:.2f}")
            print(f"📝 Transcript: {best['transcript']}")
            self.detected_language = best['language_code']
            self.detected_voice = best['voice']
            # Optionally save to file for other components
            with open("lang.json", 'w') as f:
                json.dump({'language_code': best['language_code'], 'voice': best['voice']}, f, indent=4)
            return best['language_code'], best['voice'], best['name'], best['transcript'], best['confidence']
        except Exception as e:
            print(f"❌ Google language detection failed: {e}")
            return None

    def _clean_llm_response(self, response, full_response=None, emotion=None):
        """Helper to clean and finalize LLM response text."""
        # Remove '[HumanMessage(content=...' artifacts
        if response.startswith("[HumanMessage(content="):
            response = response.split("content=")[-1].split(",")[0].strip("'\")[] ")
        # Remove system prompts and conversation markers
        markers_to_remove = [
            "You are Talia, an empathetic AI assistant",
            "Current conversation:",
            "Human:",
            "AI:",
            "System:",
            "Assistant:",
            "[HumanMessage(content=",
        ]
        for marker in ["AI:", "Assistant:", "assistant:"]:
            if marker in response:
                parts = response.split(marker)
                if len(parts) > 1:
                    response = parts[-1].strip()
                    break
        if any(marker in response for marker in markers_to_remove):
            lines = response.split('\n')
            response_lines = []
            for line in lines:
                line = line.strip()
                if line and not any(marker in line for marker in markers_to_remove):
                    response_lines.append(line)
            response = " ".join(response_lines).strip()
        response = response.replace("<|im_start|>", "").replace("<|im_end|>", "").strip()
        # Ensure response ends with a complete sentence
        if response and response[-1] not in ".!?":
            response += "."
        # Fallback if response is unclear
        if full_response is not None and (len(response) < 5 or response == full_response or not response.strip()):
            print("⚠️ LangChain response unclear, using fallback...")
            response = self._get_fallback_response(emotion)
        return response

    def generate_response(self, user_text, emotion):
        """Generate response using LangChain or direct Qwen with timing"""
        start_time = time.time()
        
        if hasattr(self, 'qwen_available') and self.qwen_available:
            print("🧠 Generating response...")
            try:
                if LANGCHAIN_AVAILABLE and hasattr(self, 'use_langchain_simple'):
                    # Use LangChain with simple approach
                    formatted_prompt = self.prompt.format(
                        history=self.memory.buffer if hasattr(self.memory, 'buffer') else "",
                        input=user_text,
                        emotion=emotion
                    )
                    # Use invoke() instead of deprecated __call__()
                    full_response = self.llm.invoke(formatted_prompt)
                    response = str(full_response).strip()
                    response = self._clean_llm_response(response, full_response, emotion)
                    self.memory.save_context(
                        {"input": user_text},
                        {"output": response}
                    )
                else:
                    # Direct Qwen call (fallback)
                    system_prompt = f"You are Talia. User feels {emotion}. Give a short, warm response."
                    conversation = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_text}<|im_end|>\n<|im_start|>assistant\n"
                    inputs = self.qwen_tokenizer(conversation, return_tensors="pt")
                    if hasattr(self.qwen_model, 'device'):
                        inputs = {k: v.to(self.qwen_model.device) for k, v in inputs.items()}
                    outputs = self.qwen_model.generate(
                        **inputs,
                        max_new_tokens=50,  # Increased for complete responses
                        temperature=0.6,
                        do_sample=True,
                        pad_token_id=self.qwen_tokenizer.eos_token_id,
                        eos_token_id=self.qwen_tokenizer.eos_token_id
                    )
                    response = self.qwen_tokenizer.decode(outputs[0], skip_special_tokens=True)
                    if "<|im_start|>assistant\n" in response:
                        response = response.split("<|im_start|>assistant\n")[-1].strip()
                    else:
                        response = response.split("assistant\n")[-1].strip() if "assistant\n" in response else response
                    response = self._clean_llm_response(response, None, emotion)
                self.timings['response_generation'] = time.time() - start_time
                print(f"🤖 Response: '{response}' ({self.timings['response_generation']:.2f}s)")
                return response
                
            except Exception as e:
                print(f"Error generating response: {e}")
                return self._get_fallback_response(emotion)
        else:
            print("⚠️ Qwen LLM not available, falling back to default response.")
            return self._get_fallback_response(emotion)
    
    def _get_fallback_response(self, emotion):
        """Get a complete, empathetic fallback response based on raw emotion"""
        fallback_responses = {
            # Raw RoBERTa emotions - shorter responses for speed
            "joy": "I'm so happy for you! Your joy is contagious! 😊",
            "sadness": "I'm here for you. It's okay to feel sad, and you're not alone. 💙",
            "anger": "I understand you're frustrated. Let's talk about what's bothering you. 🔥",
            "fear": "You're safe here with me. I'm here to support you through this. 🛡️",
            "surprise": "Wow! That's exciting! Tell me more about what happened! 😲",
            "disgust": "I can sense something is bothering you. I'm here to help. 🤢",
            "neutral": "How are you feeling today? I'm here to listen. 🤔"
        }
        return fallback_responses.get(emotion, "I'm here to listen and support you. 💬")
    
    def synthesize_speech(self, text, emotion):
        """Convert text to speech with emotion using TTS"""
        if not self.tts_available:
            print("⚠️ TTS not available - response displayed as text only")
            return
        import torch
        start_time = time.time()
        print("🎭 Synthesizing speech...")
        print(f"📝 TTS input text: '{text}' ({len(text.split())} words)")
        try:
            # Map emotion to TTS prompt
            emotion_voice = self.emotion_prompt.get(emotion, "neutral")
            desc = f"Talia speaks in a {emotion_voice} tone with very clear audio."
            desc_tokens = self.tts_tokenizer(desc, return_tensors="pt")
            text_tokens = self.tts_tokenizer(text, return_tensors="pt")
            # Move tensors to same device as model, with CPU fallback
            try:
                device = next(self.tts_model.parameters()).device
                if torch.cuda.is_available():
                    self.tts_model = self.tts_model.cuda()
                    desc_tokens = {k: v.cuda() for k, v in desc_tokens.items()}
                    text_tokens = {k: v.cuda() for k, v in text_tokens.items()}
                    print(f"🖥️ TTS running on device: {device}")
                else:
                    desc_tokens = {k: v.to(device) for k, v in desc_tokens.items()}
                    text_tokens = {k: v.to(device) for k, v in text_tokens.items()}
                    print("🖥️ TTS running on device: cpu")
            except Exception as device_error:
                print(f"⚠️ GPU device error, falling back to CPU: {device_error}")
                self.tts_model = self.tts_model.cpu()
                desc_tokens = {k: v.cpu() for k, v in desc_tokens.items()}
                text_tokens = {k: v.cpu() for k, v in text_tokens.items()}
                print("🖥️ TTS running on device: cpu")
            ids = desc_tokens['input_ids'] if isinstance(desc_tokens, dict) else desc_tokens.input_ids
            prompt_ids = text_tokens['input_ids'] if isinstance(text_tokens, dict) else text_tokens.input_ids
            with torch.no_grad():
                try:
                    out = self.tts_model.generate(
                        input_ids=ids, 
                        prompt_input_ids=prompt_ids,
                        max_new_tokens=300,  # Increased for full response
                        do_sample=True,
                        temperature=0.7,
                        attention_mask=desc_tokens['attention_mask'] if isinstance(desc_tokens, dict) else desc_tokens.attention_mask,
                        pad_token_id=self.tts_tokenizer.eos_token_id,
                        eos_token_id=self.tts_tokenizer.eos_token_id
                    )
                except Exception as gen_error:
                    print(f"⚠️ Generation error, trying with reduced parameters: {gen_error}")
                    out = self.tts_model.generate(
                        input_ids=ids, 
                        prompt_input_ids=prompt_ids,
                        max_new_tokens=128,  # Fallback for stability
                        do_sample=False,
                        temperature=0.5,
                        attention_mask=desc_tokens['attention_mask'] if isinstance(desc_tokens, dict) else desc_tokens.attention_mask,
                        pad_token_id=self.tts_tokenizer.eos_token_id,
                        eos_token_id=self.tts_tokenizer.eos_token_id
                    )
            audio = out.cpu().numpy().squeeze()
            # Regenerate if audio is empty or too short (<1.5s)
            audio_duration = len(audio) / self.tts_model.config.sampling_rate
            if audio.size == 0 or audio_duration < 1.5:
                print("🔄 Audio too short, regenerating with more tokens...")
                with torch.no_grad():
                    out = self.tts_model.generate(
                        input_ids=ids, 
                        prompt_input_ids=prompt_ids,
                        max_new_tokens=400,
                        do_sample=True,
                        temperature=0.7,
                        attention_mask=desc_tokens['attention_mask'] if isinstance(desc_tokens, dict) else desc_tokens.attention_mask,
                        pad_token_id=self.tts_tokenizer.eos_token_id,
                        eos_token_id=self.tts_tokenizer.eos_token_id
                    )
                audio = out.cpu().numpy().squeeze()
                audio_duration = len(audio) / self.tts_model.config.sampling_rate
            # Normalize audio to prevent clipping
            if np.max(np.abs(audio)) > 0:
                audio = audio / np.max(np.abs(audio)) * 0.8
            print(f"🎵 Generated audio: {len(audio)} samples ({audio_duration:.2f}s)")
            self.timings['tts'] = time.time() - start_time
            print(f"🎭 TTS completed ({self.timings['tts']:.2f}s)")
            wav.write("reply.wav", self.tts_model.config.sampling_rate, audio.astype(np.float32))
            print("🔊 Playing response...")
            sd.play(audio, self.tts_model.config.sampling_rate, blocking=True)
        except Exception as e:
            print(f"Error in TTS: {e}")
            print("⚠️ TTS failed - response displayed as text only")
            print(f"📝 Text Response: {text}")
            try:
                import pyttsx3
                print("🔄 Trying simple TTS fallback...")
                engine = pyttsx3.init()
                engine.say(text)
                engine.runAndWait()
                print("✅ Simple TTS fallback completed")
            except ImportError:
                print("ℹ️ pyttsx3 not available for fallback TTS")
            except Exception as fallback_error:
                print(f"❌ Fallback TTS also failed: {fallback_error}")
    
    def run_conversation_loop(self):
        """Main conversation loop with performance monitoring and optional language detection"""
        print("\n🎧 Optimized Emotional AI Voice Assistant Ready!")
        print(f"Emotion detection: {'Advanced' if self.use_advanced_emotion else 'Keyword-based'}")
        print(f"TTS: {'Available' if self.tts_available else 'Not available'}")
        print(f"LangChain: {'Available' if LANGCHAIN_AVAILABLE else 'Not available'}")
        print(f"Caching: {'Enabled' if self.enable_caching else 'Disabled'}")
        print("Press Ctrl+C to exit\n")
        try:
            while True:
                loop_start = time.time()
                # Step 1: Record audio (shorter window for real-time feel)
                audio, sample_rate = self.record_audio(duration=2.5)
                self.save_audio(audio, sample_rate)
                # Step 1.5: Detect language (optional)
                if self.enable_language_detection:
                    lang_result = self.detect_language_google(audio_file="input.wav", sample_rate=44100)
                    if lang_result:
                        lang_code, voice, lang_name, transcript, conf = lang_result
                        print(f"🌐 Detected Language: {lang_name} ({lang_code}), Voice: {voice}")
                # Step 2: Transcribe
                user_text = self.transcribe_audio()
                if not user_text:
                    print("❌ No speech detected, trying again...")
                    continue
                # Step 3: Detect emotion
                emotion, confidence = self.detect_emotion(user_text)
                # Step 4: Generate response
                response = self.generate_response(user_text, emotion)
                # Step 5: Synthesize and play speech (real-time, threaded)
                if self.tts_available:
                    tts_thread = threading.Thread(target=self.synthesize_speech, args=(response, emotion))
                    tts_thread.start()
                    tts_thread.join()  # Wait for TTS to finish before next loop
                else:
                    print(f"\n🎭 Emotional Response: {response}")
                    print("(TTS not available - response displayed as text)")
                # Performance summary
                total_time = time.time() - loop_start
                print(f"\n⏱️ Performance Summary:")
                print(f"  Recording: {self.timings.get('recording', 0):.2f}s")
                print(f"  Transcription: {self.timings.get('transcription', 0):.2f}s")
                print(f"  Emotion Detection: {self.timings.get('emotion_detection', 0):.2f}s")
                print(f"  Response Generation: {self.timings.get('response_generation', 0):.2f}s")
                if self.tts_available:
                    print(f"  TTS: {self.timings.get('tts', 0):.2f}s")
                print(f"  Total Loop Time: {total_time:.2f}s")
                # Store conversation with performance data
                conversation_entry = {
                    "user": user_text,
                    "emotion": emotion,
                    "confidence": confidence,
                    "assistant": response,
                    "timestamp": time.time(),
                    "detection_method": "Advanced" if self.use_advanced_emotion else "Keyword-based",
                    "response_method": "LangChain" if LANGCHAIN_AVAILABLE else "Direct Qwen" if hasattr(self, 'qwen_available') and self.qwen_available else "Fallback",
                    "performance": {
                        "recording": self.timings.get('recording', 0),
                        "transcription": self.timings.get('transcription', 0),
                        "emotion_detection": self.timings.get('emotion_detection', 0),
                        "response_generation": self.timings.get('response_generation', 0),
                        "tts": self.timings.get('tts', 0),
                        "total_time": total_time
                    }
                }
                self.conversation_history.append(conversation_entry)
                # Save conversation after each exchange
                self._save_conversation_log()
                print("\n" + "="*50 + "\n")
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            self._save_conversation_log()
            self.executor.shutdown(wait=True)
    
    def _save_conversation_log(self):
        """Save conversation history to file with performance data"""
        if not self.conversation_history:
            return
            
        try:
            with open("conversation_log.txt", "w", encoding="utf-8") as f:
                f.write("Optimized Emotional AI Voice Assistant - Conversation Log\n")
                f.write("=" * 60 + "\n\n")
                
                # Session summary with performance
                total_exchanges = len(self.conversation_history)
                emotions_detected = {}
                avg_performance = {
                    "recording": 0, "transcription": 0, "emotion_detection": 0,
                    "response_generation": 0, "tts": 0, "total_time": 0
                }
                
                for conv in self.conversation_history:
                    emotion = conv['emotion']
                    emotions_detected[emotion] = emotions_detected.get(emotion, 0) + 1
                    
                    # Sum performance metrics
                    if 'performance' in conv:
                        for key in avg_performance:
                            avg_performance[key] += conv['performance'].get(key, 0)
                
                # Calculate averages
                for key in avg_performance:
                    avg_performance[key] /= total_exchanges
                
                f.write("📊 SESSION SUMMARY:\n")
                f.write(f"Total Exchanges: {total_exchanges}\n")
                f.write(f"Emotions Detected: {emotions_detected}\n")
                f.write(f"Detection Method: {self.conversation_history[0]['detection_method']}\n")
                f.write(f"Response Method: {self.conversation_history[0]['response_method']}\n")
                # Safe timestamp handling
                start_time = self.conversation_history[0].get('timestamp', time.time())
                end_time = self.conversation_history[-1].get('timestamp', time.time())
                f.write(f"Session Start: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}\n")
                f.write(f"Session End: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}\n")
                f.write(f"\n⏱️ AVERAGE PERFORMANCE:\n")
                f.write(f"  Recording: {avg_performance['recording']:.2f}s\n")
                f.write(f"  Transcription: {avg_performance['transcription']:.2f}s\n")
                f.write(f"  Emotion Detection: {avg_performance['emotion_detection']:.2f}s\n")
                f.write(f"  Response Generation: {avg_performance['response_generation']:.2f}s\n")
                f.write(f"  TTS: {avg_performance['tts']:.2f}s\n")
                f.write(f"  Total Time: {avg_performance['total_time']:.2f}s\n")
                f.write("\n" + "=" * 60 + "\n\n")
                
                # Detailed conversation log
                f.write("💬 DETAILED CONVERSATION:\n\n")
                for i, conv in enumerate(self.conversation_history, 1):
                    f.write(f"Exchange {i}:\n")
                    # Safe field access with defaults
                    timestamp = conv.get('timestamp', time.time())
                    user_text = conv.get('user', 'Unknown')
                    emotion = conv.get('emotion', 'unknown')
                    confidence = conv.get('confidence', 0.0)
                    assistant_text = conv.get('assistant', 'No response')
                    detection_method = conv.get('detection_method', 'Unknown')
                    response_method = conv.get('response_method', 'Unknown')
                    
                    f.write(f"⏰ Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp))}\n")
                    f.write(f"👤 User: \"{user_text}\"\n")
                    f.write(f"😐 Emotion: {emotion} (confidence: {confidence:.2f})\n")
                    f.write(f"🤖 Assistant: \"{assistant_text}\"\n")
                    f.write(f"🔧 Method: {detection_method} detection, {response_method} response\n")
                    if 'performance' in conv:
                        f.write(f"⏱️ Performance: {conv['performance'].get('total_time', 0):.2f}s total\n")
                    f.write("-" * 50 + "\n\n")
            
            print(f"📝 Conversation log saved to conversation_log.txt ({total_exchanges} exchanges)")
        except Exception as e:
            print(f"Error saving conversation log: {e}")
    
    def _load_previous_conversation(self):
        """Load previous conversation history if available"""
        try:
            if os.path.exists("conversation_log.txt"):
                print("📖 Loading previous conversation history...")
                with open("conversation_log.txt", "r", encoding="utf-8") as f:
                    content = f.read()
                
                # Extract conversation entries from the log
                lines = content.split('\n')
                current_entry = {}
                
                for line in lines:
                    if line.startswith("👤 User: "):
                        if current_entry:
                            self.conversation_history.append(current_entry)
                        current_entry = {"user": line[9:].strip('"')}
                    elif line.startswith("😐 Emotion: ") and current_entry:
                        parts = line[12:].split(" (confidence: ")
                        current_entry["emotion"] = parts[0]
                        if len(parts) > 1:
                            current_entry["confidence"] = float(parts[1].split(")")[0])
                    elif line.startswith("🤖 Assistant: ") and current_entry:
                        current_entry["assistant"] = line[13:].strip('"')
                    elif line.startswith("⏰ Time: ") and current_entry:
                        time_str = line[9:]
                        try:
                            current_entry["timestamp"] = time.mktime(time.strptime(time_str, "%Y-%m-%d %H:%M:%S"))
                        except:
                            current_entry["timestamp"] = time.time()
                    elif line.startswith("🔧 Method: ") and current_entry:
                        method_info = line[11:]
                        if "detection" in method_info:
                            detection_part = method_info.split(" detection")[0]
                            current_entry["detection_method"] = detection_part
                        if "response" in method_info:
                            response_part = method_info.split("response")[0].split(", ")[-1]
                            current_entry["response_method"] = response_part
                
                if current_entry:
                    self.conversation_history.append(current_entry)
                
                print(f"📚 Loaded {len(self.conversation_history)} previous exchanges")
                
        except Exception as e:
            print(f"Error loading previous conversation: {e}")

def main():
    """Main function to run the emotional assistant"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Emotional AI Voice Assistant")
    parser.add_argument("--simple-emotion", action="store_true", help="Use keyword-based emotion detection")
    parser.add_argument("--no-cache", action="store_true", help="Disable caching for debugging")
    
    args = parser.parse_args()
    
    # Create and run the assistant
    assistant = EmotionalAssistant(
        use_advanced_emotion=not args.simple_emotion,
        enable_caching=not args.no_cache
    )
    
    assistant.run_conversation_loop()

if __name__ == "__main__":
    main() 