import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

DATA_PATH = "../data/LJSpeech-1.1/wavs/"
OUTPUT_PATH = "../data/processed/"

def load_audio(file_path, sr=22050):
    """Loads an audio file and returns waveform and sample rate."""
    waveform, sample_rate = librosa.load(file_path, sr=sr)
    return waveform, sample_rate

def plot_waveform(waveform, sample_rate, title="Waveform"):
    """Plots waveform of the audio."""
    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(waveform, sr=sample_rate)
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.show()

def process_audio(file_name):
    """Loads, processes, and saves spectrogram for given audio file."""
    file_path = os.path.join(DATA_PATH, file_name)
    waveform, sample_rate = load_audio(file_path)

    # Convert to Mel-Spectrogram
    mel_spec = librosa.feature.melspectrogram(y=waveform, sr=sample_rate, n_mels=128)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Save processed data
    np.save(os.path.join(OUTPUT_PATH, file_name.replace(".wav", ".npy")), mel_spec_db)

# Process all files
for file in os.listdir(DATA_PATH):
    if file.endswith(".wav"):
        process_audio(file)


#This script converts .wav files to spectrograms and saves them in data/processed/ for training.