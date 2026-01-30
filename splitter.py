import os
import re  # NEW: For robust name matching
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

FEATURE_DIR = "./mel_features"
OUTPUT_FILE = "dataset.npz"
CLASSES = ['Gazal', 'Lokdohori', 'Nephop', 'POP']
TEST_SIZE = 0.2
RANDOM_STATE = 42

def load_features(file_list, genre_label):
    X, y = [], []
    for f in file_list:
        mel = np.load(f)
        if mel.shape == (128, 128, 1):
            X.append(mel)
            y.append(genre_label)
    return X, y

def main():
    X_train, X_test, y_train, y_test = [], [], [], []

    for label, genre in enumerate(CLASSES):
        genre_path = os.path.join(FEATURE_DIR, genre)
        if not os.path.exists(genre_path): continue
        
        song_files = {}

        # ---------------------------------------------------------
        # UPDATED: Grouping logic to prevent Data Leakage
        # ---------------------------------------------------------
        for f in os.listdir(genre_path):
            if f.endswith(".npy"):
                # Use Regex to find the original song name.
                # This looks for the chunk index (_0, _1, or _2) and ignores 
                # everything after it (like _aug0) to get the true base song name.
                match = re.search(r'^(.*?)_[012](?:_aug.*)?\.npy$', f)
                
                if match:
                    original_song_name = match.group(1)
                    full_path = os.path.join(genre_path, f)
                    song_files.setdefault(original_song_name, []).append(full_path)

        songs = list(song_files.keys())

        # Now we split by Original Song Name
        train_songs, test_songs = train_test_split(
            songs, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )

        # Append all versions (original + augmented) to the correct split
        for s in train_songs:
            X, y = load_features(song_files[s], label)
            X_train.extend(X)
            y_train.extend(y)
            
        for s in test_songs:
            X, y = load_features(song_files[s], label)
            X_test.extend(X)
            y_test.extend(y)

        print(f"{genre}: {len(train_songs)} original songs in Train, {len(test_songs)} in Test")

    # Convert to arrays
    X_train = np.array(X_train, dtype=np.float32)
    X_test = np.array(X_test, dtype=np.float32)
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=len(CLASSES))
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=len(CLASSES))

    # Save
    np.savez(OUTPUT_FILE, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
    print(f"\n✅ Balanced dataset saved! Train: {X_train.shape}, Test: {X_test.shape}")

if __name__ == "__main__":
    main()
