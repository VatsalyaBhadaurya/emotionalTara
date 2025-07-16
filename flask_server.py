from flask import Flask, request, jsonify, send_file, make_response
from emotional_assistant import EmotionalAssistant
import os
import io
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename
from flask import Response
import json as pyjson

app = Flask(__name__)
assistant = EmotionalAssistant()

@app.route('/api/ask', methods=['POST'])
def ask():
    print('Received files:', request.files)  # Debug print
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided', 'received_keys': list(request.files.keys())}), 400
    audio_file = request.files['audio']
    audio_path = 'input.wav'
    audio_file.save(audio_path)
    # Run pipeline
    user_text = assistant.transcribe_audio(audio_path)
    if not user_text:
        return jsonify({'error': 'No speech detected'}), 400
    emotion, confidence = assistant.detect_emotion(user_text)
    response = assistant.generate_response(user_text, emotion)
    assistant.synthesize_speech(response, emotion)
    # Prepare JSON metadata
    meta = {
        'user_text': user_text,
        'emotion': emotion,
        'confidence': confidence,
        'response': response
    }
    # Prepare audio file
    if not os.path.exists('reply.wav'):
        return jsonify({'error': 'No audio generated'}), 500
    with open('reply.wav', 'rb') as f:
        audio_bytes = f.read()
    # Create multipart response (JSON + audio)
    boundary = '----WebKitFormBoundary7MA4YWxkTrZu0gW'
    multipart_body = (
        f'--{boundary}\r\n'
        'Content-Disposition: form-data; name="metadata"\r\n'
        'Content-Type: application/json\r\n\r\n'
        f'{pyjson.dumps(meta)}\r\n'
        f'--{boundary}\r\n'
        'Content-Disposition: form-data; name="audio"; filename="reply.wav"\r\n'
        'Content-Type: audio/wav\r\n\r\n'
    ).encode('utf-8') + audio_bytes + f'\r\n--{boundary}--\r\n'.encode('utf-8')
    resp = Response(multipart_body, mimetype=f'multipart/form-data; boundary={boundary}')
    return resp

@app.route('/api/audio', methods=['GET'])
def get_audio():
    if not os.path.exists('reply.wav'):
        return jsonify({'error': 'No audio available'}), 404
    return send_file('reply.wav', mimetype='audio/wav')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True) 