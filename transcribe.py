import wave
import csv
import os
import json
from vosk import Model, KaldiRecognizer
import subprocess
import librosa
import numpy as np

# Paths
model_path = "model"
audio_folder = "audio"
output_csv = "output/song_features.csv"

# Prepare model once globally
model = Model(model_path)

# Note labels
note_labels = ['C', 'C#', 'D', 'D#', 'E', 'F', 
               'F#', 'G', 'G#', 'A', 'A#', 'B']

# CSV header columns
csv_columns = [
    "genre", "song_name", "tempo", "average_pitch"
] + [f"chroma_{note}" for note in note_labels] + ["lyrics"]

# Ensure output folder exists
os.makedirs("output", exist_ok=True)

def process_song(mp3_path):
    """Process one song and append features to CSV."""
    basename = os.path.basename(mp3_path)

    # Check filename format
    if "_" not in basename:
        raise ValueError(f"Filename '{basename}' does not contain a genre prefix (expected format 'genre_songname.mp3')")

    genre, song_base = basename.split("_", 1)
    song_name = os.path.splitext(song_base)[0]
    wav_path = os.path.join(audio_folder, f"{song_name}.wav")

    # Convert MP3 to WAV
    subprocess.run([
        "ffmpeg", "-i", mp3_path, 
        "-ac", "1", "-ar", "16000", "-sample_fmt", "s16",
        wav_path,
        "-y"
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Transcribe lyrics
    rec = KaldiRecognizer(model, 16000)
    with wave.open(wav_path, "rb") as wf:
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() != 16000:
            raise ValueError(f"{song_name}: Audio must be mono, 16-bit, and 16kHz")

        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            rec.AcceptWaveform(data)

    result = json.loads(rec.Result())
    lyrics = result["text"]

    # Extract audio features
    y, sr = librosa.load(wav_path)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    if isinstance(tempo, (list, np.ndarray)):
        tempo = float(tempo[0])

    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_values = []
    for t in range(pitches.shape[1]):
        index = magnitudes[:, t].argmax()
        pitch = pitches[index, t]
        if pitch > 0:
            pitch_values.append(pitch)

    avg_pitch = np.mean(pitch_values)

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    mean_chroma = np.mean(chroma, axis=1)

    # Build feature row
    song_features = {
        "genre": genre,        # <- now called genre
        "song_name": song_name,
        "tempo": float(tempo),
        "average_pitch": float(avg_pitch),
    }

    for i, energy in enumerate(mean_chroma):
        song_features[f"chroma_{note_labels[i]}"] = float(energy)

    song_features["lyrics"] = lyrics

    # Append to CSV
    file_exists = os.path.isfile(output_csv)

    with open(output_csv, "a", newline='') as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=csv_columns)
        if not file_exists:
            writer.writeheader()
        writer.writerow(song_features)

    print(f"Processed {genre}_{song_name}")

    # Delete temporary WAV file
    if os.path.exists(wav_path):
        os.remove(wav_path)

def process_folder(folder_path):
    """Process all MP3 files in a folder."""
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".mp3"):
            mp3_path = os.path.join(folder_path, filename)
            process_song(mp3_path)

process_folder("./audio")
