import wave
import json
import os
from vosk import Model, KaldiRecognizer
import subprocess
import librosa
import numpy as np

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



# Extract tempo and average pitch with librosa
y, sr = librosa.load("audio/YBWM.wav")

tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
print(f"Tempo: {tempo} BPM")

pitches, magnitudes = librosa.piptrack(y=y, sr=sr)

pitch_values = []
for t in range(pitches.shape[1]):
    index = magnitudes[:, t].argmax()
    pitch = pitches[index, t]
    if pitch > 0:
        pitch_values.append(pitch)

avg_pitch = np.mean(pitch_values)
print(f"Average Pitch: {avg_pitch:.2f} Hz")


# Extract common musical notes
chroma = librosa.feature.chroma_stft(y=y, sr=sr)
note_labels = ['C', 'C#', 'D', 'D#', 'E', 'F', 
               'F#', 'G', 'G#', 'A', 'A#', 'B']

# Average energy of each pitch class over time
mean_chroma = np.mean(chroma, axis=1)

# Pair notes with their average energies
for i, energy in enumerate(mean_chroma):
    print(f"{note_labels[i]}: {energy:.3f}")
