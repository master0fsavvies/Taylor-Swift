# train.py

import pandas as pd
import numpy as np
import re
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import hstack, csr_matrix

# Paths
csv_path = "output/song_features.csv"
model_output_path = "output/genre_classifier_bundle.joblib"  # Save bundle now!

# 1. Load dataset
print("Loading dataset...")
df = pd.read_csv(csv_path)

# 2. Preprocess the data

# Function to extract simple lyric features
def extract_lyric_features(lyrics):
    words = re.findall(r'\w+', lyrics.lower())
    num_words = len(words)
    unique_words = len(set(words))
    avg_word_length = np.mean([len(word) for word in words]) if words else 0
    return num_words, unique_words, avg_word_length

# Extract lyric features if missing
if 'lyric_length' not in df.columns or 'unique_words' not in df.columns or 'avg_word_length' not in df.columns:
    print("Extracting simple lyric features...")
    features = df['lyrics'].apply(extract_lyric_features)
    df['lyric_length'] = [f[0] for f in features]
    df['unique_words'] = [f[1] for f in features]
    df['avg_word_length'] = [f[2] for f in features]

# 3. Prepare feature sets

# Numeric features
X_numeric = df.drop(columns=["genre", "song_name", "lyrics"])

# Text features (bag of words from lyrics)
print("Extracting bag-of-words features from lyrics...")
vectorizer = CountVectorizer(max_features=100)  # Top 100 words
X_text = vectorizer.fit_transform(df['lyrics'])

# Combine numeric + text features
X_combined = hstack([csr_matrix(X_numeric.values), X_text])

# Labels
y = df["genre"].map({"country": 0, "pop": 1})

# 4. Train on ALL data (no splitting yet)
print("Training Random Forest model on all data...")
model = RandomForestClassifier(random_state=42)
model.fit(X_combined, y)

# 5. Evaluate on the same data
y_pred = model.predict(X_combined)
print("\n=== Training Set Evaluation ===\n")
print(classification_report(y, y_pred, target_names=["country", "pop"]))

# 6. Save model and vectorizer together
print(f"\nSaving model and vectorizer to {model_output_path}...")
model_bundle = {
    "model": model,
    "vectorizer": vectorizer
}
joblib.dump(model_bundle, model_output_path)

print("\nModel training complete!")