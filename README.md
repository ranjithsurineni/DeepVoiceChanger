# 🎤 DeepVoiceChanger: LSTM-based Voice Conversion

## 📌 Project Overview
This project focuses on **Voice Conversion** using deep learning. It takes an input speech waveform, extracts MFCC (Mel-Frequency Cepstral Coefficients) features, trains a **LSTM-based neural network**, and generates a transformed speech waveform.

The project is built using **PyTorch, Librosa, NumPy, Matplotlib**, and other open-source tools.

## 🚀 Features
- Converts a speaker's voice to another voice while maintaining speech characteristics.
- Uses **LSTM (Long Short-Term Memory)** for sequence modeling.
- Supports **training, inference, and waveform conversion**.
- Works without a GPU, optimized for **CPU-based** systems.
- Outputs the converted voice in `.wav` format.

## 🛠 Tech Stack & Dependencies
- **Python** (>= 3.8)
- **PyTorch** (For deep learning model training & inference)
- **Librosa** (For audio feature extraction & manipulation)
- **Matplotlib** (For visualization)
- **NumPy** (For numerical computations)

## 📥 Installation
```bash
# Clone the repository
git clone https://github.com/ranjithsurineni/DeepVoiceChanger.git
cd voice-conversion

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
│   │   ├── metadata.csv
│   │   ├── wavs/
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
│
│── models/                      # Saved trained models
│   ├── trained_model.pth        # Final trained RVC model
│
│── results/                     # Output converted voices
│   ├── converted_audio.wav      # Output after voice conversion
│
│── notebooks/                    # Jupyter Notebooks for debugging
│   ├── data_visualization.ipynb  # Exploratory data analysis
│   ├── model_training.ipynb      # Training experiments and train RVC model
│   ├── convert_voice.ipynb       # convert .npy file to .wav file
│   ├── verify_data_npy.ipynb     # verify processed sample data of .npy file and visualize to heatmap
│   ├── wav2png.ipynb             # converts .wav file to plot for display wave structure of input & output
│
│── requirements.txt              # Dependencies for Python environment
│── README.md                     # Project documentation

```

## 🏗️ Step-by-Step Implementation
### 1️⃣ **Preprocessing: Convert Audio to Features**
Extract **MFCC** features from input `.wav` files.
```python
import librosa
import numpy as np

# Load the WAV file
def extract_mfcc(file_path, n_mfcc=13):
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return mfcc.T  # Transpose to match (time, features)
```

### 2️⃣ **Model Training**
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

### 3️⃣ **Inference: Convert Voice & Save as Audio**
Use the trained model to convert voice and reconstruct the audio file.
```python
import soundfile as sf

def save_audio(features, output_path, sr=22050):
    reconstructed_audio = librosa.feature.inverse.mfcc_to_audio(features.T)
    sf.write(output_path, reconstructed_audio, sr)
```

### 4️⃣ **Visualizing Waveforms**
Display the waveform before and after conversion.
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

## 📌 Issues & Debugging
### **Problem: Output Audio is Too Short**
🔹 **Reason**: The model may not be processing full-length sequences properly.
🔹 **Fix**: Ensure all input MFCC feature sequences are **padded** to the same length before training.
```python
from torch.nn.utils.rnn import pad_sequence

padded_batch = pad_sequence([torch.tensor(f) for f in features], batch_first=True)
```

### **Problem: Mismatch in Input Shape During Inference**
🔹 **Reason**: The model expects an input of shape `[batch, time, features]` but gets `[time, features]`.
🔹 **Fix**: Reshape the input before passing it to the model.
```python
test_features = test_features.unsqueeze(0)  # Add batch dimension
output_features = model(test_features)
```

## 📜 Future Improvements
✅ Enhance voice transformation quality using **GANs or Diffusion models**
✅ Implement **real-time voice conversion**
✅ Optimize performance for **low-end CPUs**

## 📢 Contribution
Contributions are welcome! Feel free to **fork** this repository, open **issues**, or submit **pull requests**.

## 📄 License
This project is licensed under the **MIT License**.

