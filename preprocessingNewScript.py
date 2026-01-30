import os
import librosa
import numpy as np
import tensorflow as tf

# Constants
datasetDirectory = './datasets'
classes = ['Gazal', 'Lokdohori', 'Nephop', 'POP']
OUTPUT_DIR = "./mel_features"
SR = 22050
DURATION = 30
SAMPLES = SR * DURATION
N_MELS = 128

# ---------------------------------------------------------
# AUGMENTATION REPERTOIRE
# ---------------------------------------------------------
def augment_audio(y, sr):
    """Returns a list of 3 different augmented versions of the audio."""
    augmented = []
    
    # 1. Pitch shift (Good for vocal-heavy genres like Gazal)
    y_pitch = librosa.effects.pitch_shift(y, sr=sr, n_steps=np.random.choice([-2, -1, 1, 2]))
    augmented.append(y_pitch)
    
    # 2. Time stretch (Changes tempo without changing pitch)
    rate = np.random.uniform(0.85, 1.15)
    y_stretch = librosa.effects.time_stretch(y, rate=rate)
    augmented.append(y_stretch)
    
    # 3. Add white noise (Helps model ignore background hiss)
    noise = np.random.normal(0, 0.005, len(y))
    y_noise = y + noise
    augmented.append(y_noise)
    
    return augmented

def save_mel(chunk, sampleRate, target_shape, base_name, i, save_path, aug_tag=""):
    """Handles the Mel-spectrogram conversion and saving."""
    melSpectrogram = librosa.feature.melspectrogram(
        y=chunk, 
        sr=sampleRate, 
        n_mels=N_MELS
    )
    mel_db = librosa.power_to_db(melSpectrogram, ref=np.max)
    
    # Ensure shape is exactly (128, 128, 1) for the CNN
    mel_resized = tf.image.resize(
        mel_db[..., np.newaxis].astype(np.float32),
        target_shape 
    ).numpy()
    
    save_filename = f"{base_name}_{i}{aug_tag}.npy"
    actual_savepath = os.path.join(save_path, save_filename)
    np.save(actual_savepath, mel_resized)

# ---------------------------------------------------------
# MAIN PREPROCESSING ENGINE
# ---------------------------------------------------------
def loadAndPreprocessData(datasetDirectory, classes, target_shape=(128, 128)):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for className in classes:
        classDir = os.path.join(datasetDirectory, className)
        if not os.path.exists(classDir):
            print(f"Skipping {className}: Directory not found.")
            continue
            
        print(f"--- Processing: {className} ---")
        save_path = os.path.join(OUTPUT_DIR, className)
        os.makedirs(save_path, exist_ok=True)

        for filename in os.listdir(classDir):
            if filename.endswith('.wav'):
                filepath = os.path.join(classDir, filename)
                
                # Load audio
                try:
                    audioData, sampleRate = librosa.load(filepath, sr=SR)
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
                    continue

                length = len(audioData)
                positions = [0.25, 0.5, 0.75] # Extract 3 chunks per song
                
                for i, p in enumerate(positions):
                    start = int(p * length)
                    end = start + SAMPLES
                    
                    if end <= length:
                        chunk = audioData[start:end]
                        base_name = os.path.splitext(filename)[0] 
                        
                        # 1. ALWAYS SAVE ORIGINAL
                        save_mel(chunk, sampleRate, target_shape, base_name, i, save_path)
                        
                        # -----------------------------------------------------
                        # TARGETED AUGMENTATION LOGIC (THE FIX)
                        # -----------------------------------------------------
                        
                        # IF GAZAL: Create 3 extra versions (Target: 432 * 4 = 1728)
                        if className == "Gazal":
                            aug_chunks = augment_audio(chunk, sampleRate)
                            for j, aug in enumerate(aug_chunks):
                                save_mel(aug, sampleRate, target_shape, base_name, i, save_path, aug_tag=f"_aug{j}")
                        
                        # IF LOKDOHORI: Create 1 extra version (Target: 1027 * 2 = 2054)
                        elif className == "Lokdohori":
                            # Just use the pitch shift for Lokdohori
                            y_pitch = librosa.effects.pitch_shift(chunk, sr=sampleRate, n_steps=1)
                            save_mel(y_pitch, sampleRate, target_shape, base_name, i, save_path, aug_tag="_augPitch")
                        
                        # NEPHOP and POP: Do nothing extra (Already ~1500)
                        else:
                            pass

def main():
    loadAndPreprocessData(datasetDirectory, classes)
    print("\n🎉 Preprocessing complete! Balanced dataset saved to:", OUTPUT_DIR)

if __name__ == "__main__":
    main()
