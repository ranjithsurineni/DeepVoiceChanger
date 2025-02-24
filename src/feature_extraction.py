import numpy as np
import os
import librosa

# Paths
DATA_DIR = "D:\\projects\\voice_conversion\\data\\processed\\"
OUTPUT_DIR = "D:\\projects\\voice_conversion\\data\\features\\"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_mfcc_and_mel(npy_file):
    """Extracts MFCC and Mel-Spectrogram features from the preprocessed npy file."""
    mel_spec = np.load(npy_file)

    # Convert Mel-Spectrogram to MFCC
    mfcc = librosa.feature.mfcc(S=mel_spec, n_mfcc=13)

    # Save extracted features
    feature_filename = os.path.basename(npy_file).replace(".npy", "_features.npy")
    np.save(os.path.join(OUTPUT_DIR, feature_filename), mfcc)

# Process all npy files
for file in os.listdir(DATA_DIR):
    if file.endswith(".npy"):
        extract_mfcc_and_mel(os.path.join(DATA_DIR, file))

print("✅ Feature extraction completed! Features stored in:", OUTPUT_DIR)
