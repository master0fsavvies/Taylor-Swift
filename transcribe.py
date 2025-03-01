import wave
import json
from vosk import Model, KaldiRecognizer

# File paths
model_path = "model"
wav_path = "audio/YBWM.wav"

# Load Vosk model
model = Model(model_path)
rec = KaldiRecognizer(model, 16000)  # Vosk expects 16kHz audio

# Open WAV file
with wave.open(wav_path, "rb") as wf:
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