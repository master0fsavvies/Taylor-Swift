import wave
import json
import os
from vosk import Model, KaldiRecognizer
import subprocess


model_path = "model"
mp3_path = "audio/YBWM.mp3"
wav_path = "audio/YBWM.wav"

subprocess.run([
    "ffmpeg", "-i", mp3_path, 
    "-ac", "1",              # Mono audio
    "-ar", "16000",          # 16 kHz sample rate
    "-sample_fmt", "s16",    # 16-bit audio
    wav_path
])
song_path = wav_path

# Load Vosk model
model = Model(model_path)
rec = KaldiRecognizer(model, 16000)  # Vosk expects 16kHz audio

# Open WAV file
with wave.open(song_path, "rb") as wf:
    if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() != 16000:
        raise ValueError("Audio must be mono, 16-bit, and 16kHz")

    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        rec.AcceptWaveform(data)

# Get final transcription
result = json.loads(rec.Result())
print("Lyrics:", result["text"])
