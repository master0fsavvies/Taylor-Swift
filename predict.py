# predict.py

import wave
import os
import json
import subprocess
import librosa
import numpy as np
import re
import joblib
from vosk import Model as VoskModel, KaldiRecognizer
from scipy.sparse import hstack, csr_matrix

# Paths
model_bundle_path = "output/genre_classifier_bundle.joblib"
vosk_model_path = "model"  # your speech recognition model
temp_wav_path = "audio/temp.wav"

# Load trained model + vectorizer
print("Loading model and vectorizer...")
bundle = joblib.load(model_bundle_path)
model = bundle["model"]
vectorizer = bundle["vectorizer"]

# Load Vosk speech recognition model
vosk_model = VoskModel(vosk_model_path)

# Notes for chroma
note_labels = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# --- FUNCTIONS ---

def extract_audio_features(wav_path):
    # Load audio
    y, sr = librosa.load(wav_path)

    # Tempo
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    if isinstance(tempo, (list, np.ndarray)):
        tempo = float(tempo[0])

    # Average pitch
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_values = []
    for t in range(pitches.shape[1]):
        index = magnitudes[:, t].argmax()
        pitch = pitches[index, t]
        if pitch > 0:
            pitch_values.append(pitch)
    avg_pitch = np.mean(pitch_values)

    # Chroma
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    mean_chroma = np.mean(chroma, axis=1)
    chroma_features = {f"chroma_{note_labels[i]}": float(mean_chroma[i]) for i in range(12)}

    return tempo, avg_pitch, chroma_features

def transcribe_audio(wav_path):
    rec = KaldiRecognizer(vosk_model, 16000)
    with wave.open(wav_path, "rb") as wf:
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() != 16000:
            raise ValueError("Audio must be mono, 16-bit, and 16kHz")
        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            rec.AcceptWaveform(data)
    result = json.loads(rec.Result())
    return result["text"]

def extract_lyric_features(lyrics):
    words = re.findall(r'\w+', lyrics.lower())
    num_words = len(words)
    unique_words = len(set(words))
    avg_word_length = np.mean([len(word) for word in words]) if words else 0
    return num_words, unique_words, avg_word_length

def process_mp3(mp3_path):
    # Convert MP3 to WAV (mono, 16kHz, 16-bit)
    subprocess.run([
        "ffmpeg", "-i", mp3_path,
        "-ac", "1", "-ar", "16000", "-sample_fmt", "s16",
        temp_wav_path, "-y"
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Extract features
    tempo, avg_pitch, chroma_features = extract_audio_features(temp_wav_path)
    lyrics = transcribe_audio(temp_wav_path)
    lyric_length, unique_words, avg_word_length = extract_lyric_features(lyrics)

    # Prepare full feature dictionary
    song_info = {
        "tempo": tempo,
        "average_pitch": avg_pitch,
        **chroma_features,
        "lyric_length": lyric_length,
        "unique_words": unique_words,
        "avg_word_length": avg_word_length,
        "lyrics": lyrics
    }

    return song_info

def predict_song(song_info):
    # Prepare numeric features
    numeric_fields = [
        "tempo", "average_pitch",
        "chroma_C", "chroma_C#", "chroma_D", "chroma_D#",
        "chroma_E", "chroma_F", "chroma_F#", "chroma_G", "chroma_G#",
        "chroma_A", "chroma_A#", "chroma_B",
        "lyric_length", "unique_words", "avg_word_length"
    ]

    numeric_features = [song_info[field] for field in numeric_fields]

    # Lyrics vector
    lyrics_text = [song_info["lyrics"]]
    lyrics_vector = vectorizer.transform(lyrics_text)

    # Combine features
    numeric_matrix = csr_matrix([numeric_features])
    X = hstack([numeric_matrix, lyrics_vector])

    # Predict
    pred = model.predict(X)[0]
    genre_mapping = {0: "country", 1: "pop"}
    return genre_mapping[pred]

# --- MAIN ---

if __name__ == "__main__":
    mp3_path = "WeAreNeverGettingBackTogether.mp3"

    if not os.path.exists(mp3_path):
        print(f"Error: '{mp3_path}' does not exist.")
        exit()

    print("Processing audio and extracting features...")
    song_info = process_mp3(mp3_path)

    print("Predicting genre...")
    predicted_genre = predict_song(song_info)

    print(f"\nPredicted Genre: {predicted_genre}")

    # Clean up temp WAV file
    if os.path.exists(temp_wav_path):
        os.remove(temp_wav_path)