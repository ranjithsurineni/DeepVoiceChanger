# 🎤 DeepVoiceChanger: LSTM-based Voice Conversion

## 📌 Project Overview
This project focuses on Voice Conversion using deep learning. It takes an input speech waveform, extracts MFCC (Mel-Frequency Cepstral Coefficients) features, trains an LSTM-based neural network, and generates a transformed speech waveform.

The project is built using **PyTorch, Librosa, NumPy, Matplotlib, and other open-source tools.**

## 🚀 Features
- Converts a speaker's voice to another voice while maintaining speech characteristics.
- Uses **LSTM (Long Short-Term Memory)** for sequence modeling.
- Supports **training, inference, and waveform conversion**.
- Works **without a GPU**, optimized for CPU-based systems.
- Outputs the converted voice in **.wav format**.

## 🛠 Tech Stack & Dependencies
- **Python (>= 3.8)**
- **PyTorch** (For deep learning model training & inference)
- **Librosa** (For audio feature extraction & manipulation)
- **Matplotlib** (For visualization)
- **NumPy** (For numerical computations)

## 📥 Installation
```bash
# Clone the repository
git clone https://github.com/ranjithsurineni/DeepVoiceChanger.git
cd DeepVoiceChanger

# Create a virtual environment
python -m venv voice_env
source voice_env/bin/activate  # On Windows use: voice_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 📂 Project Structure
```
voice-changer-project/
│── data/                        # Original & processed datasets
│   ├── LJSpeech-1.1/            # Original LJ Speech dataset
│   │   ├── metadata.csv         # Transcriptions & filenames
│   │   ├── wavs/                # Folder containing all audio clips
│   │       ├── LJ001-0001.wav
│   │       ├── LJ001-0002.wav
│   │       ├── ...
│   ├── processed_wavs/          # Preprocessed audio (16kHz, mono)
│   ├── features/                # .npy files extracted voices
│
│── src/                         # Source code
│   ├── preprocessing.py         # Converts audio (16kHz, mono, normalized)
│   ├── extract_features.py      # Extracts Mel spectrograms
│   ├── convert_voice.py         # Converts new audio using trained model
│
│── models/                      # Saved trained models
│   ├── trained_model.pth        # Final trained RVC model
│
│── results/                     # Output converted voices
│   ├── converted_audio.wav      # Output after voice conversion
│
│── notebooks/                    # Jupyter Notebooks for debugging
│   ├── data_visualization.ipynb  # Exploratory data analysis
│   ├── model_training.ipynb      # Training experiments and train RVC model and save to ./models
│   ├── convert_voice.ipynb       # Convert .npy file to .wav file
│   ├── verify_data_npy.ipynb     # Verify processed sample data of .npy file and visualize to heatmap
│   ├── wav2png.ipynb             # Converts .wav file to plot for display wave structure of input & output
│
│── requirements.txt              # Dependencies for Python environment
│── README.md                     # Project documentation
```

## 🎙 Dataset Used: LJ Speech Dataset
### 📌 Dataset Overview
- **Name:** LJ Speech Dataset (LJSpeech-1.1)
- **Speaker:** Single female speaker
- **Total Clips:** 13,100 short audio clips
- **Duration:** ~24 hours of speech
- **Clip Length:** 1 to 10 seconds per clip
- **Transcriptions:** Each clip has an accompanying text transcript
- **Recording Year:** 2016-2017
- **Source:** Recorded by the **LibriVox** project
- **Text Source:** Passages from 7 non-fiction books published between **1884 and 1964** (public domain)

### 🎯 Why Use LJ Speech for Voice Conversion?
✅ **High-Quality Audio:** Clear, studio-recorded speech without background noise.  
✅ **Consistent Speaker:** Ideal for **voice modeling and conversion** tasks.  
✅ **Public Domain:** Freely available for commercial and non-commercial use.  
✅ **Rich Phonetic Coverage:** Covers a diverse set of phonemes for speech modeling.  

### 🔍 How It’s Used in DeepVoiceChanger
1️⃣ **Preprocessing:** Convert raw `.wav` files to **16kHz, mono, normalized format**.  
2️⃣ **Feature Extraction:** Extract **MFCC (Mel-Frequency Cepstral Coefficients)** features from the audio clips.  
3️⃣ **Model Training:** Train an **LSTM-based neural network** using MFCC features as input.  
4️⃣ **Inference:** Convert new audio files into transformed voice while retaining speech characteristics.  

## 🏗️ Step-by-Step Implementation
### 1️⃣ Preprocessing: Convert Audio to Features
```python
import librosa
import numpy as np

# Load the WAV file
def extract_mfcc(file_path, n_mfcc=13):
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return mfcc.T  # Transpose to match (time, features)
```
### 2️⃣ Model Training
Define and train an **LSTM-based neural network** for voice conversion.
```python
import torch
import torch.nn as nn

class VoiceConversionModel(nn.Module):
    def __init__(self, input_dim=13, hidden_dim=128):
        super(VoiceConversionModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, input_dim)
    
    def forward(self, x):
        x, _ = self.lstm(x)
        return self.fc(x)
```
Train the model using **Mean Squared Error Loss (MSE)**.
```python
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
```
### 3️⃣ Inference: Convert Voice & Save as Audio
```python
import soundfile as sf

def save_audio(features, output_path, sr=22050):
    reconstructed_audio = librosa.feature.inverse.mfcc_to_audio(features.T)
    sf.write(output_path, reconstructed_audio, sr)
```
### 4️⃣ Visualizing Waveforms
```python
import librosa.display
import matplotlib.pyplot as plt

def plot_waveform(file_path):
    y, sr = librosa.load(file_path, sr=None)
    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(y, sr=sr)
    plt.title("Waveform")
    plt.show()
```

## 📜 Future Improvements
✅ Enhance voice transformation quality using **GANs or Diffusion models**  
✅ Implement **real-time voice conversion**  
✅ Optimize performance for **low-end CPUs**  

## 📢 Contribution
Contributions are welcome! Feel free to fork this repository, open issues, or submit pull requests.

## 📄 License
This project is licensed under the **MIT License**.

