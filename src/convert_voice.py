import torch
import numpy as np
import librosa
import librosa.display
import soundfile as sf
import os

# ✅ Step 1: Load the Trained Model
# ----------------------------------
class VoiceConversionModel(torch.nn.Module):
    """Simple feedforward model for voice conversion"""
    def __init__(self):
        super(VoiceConversionModel, self).__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(13, 64),  # Input: 13 MFCC features
            torch.nn.ReLU(),
            torch.nn.Linear(64, 13)  # Output: same shape as input
        )

    def forward(self, x):
        return self.fc(x)

# Load model
model_path = "D:\\projects\\voice_conversion\\models\\voice_conversion.pth"
model = VoiceConversionModel()
model.load_state_dict(torch.load(model_path))
model.eval()  # Set to evaluation mode

print("✅ Model loaded successfully!")

# ✅ Step 2: Load Test Data
# ---------------------------
# Select an example feature file
test_file = "D:\\projects\\voice_conversion\\data\\features\\LJ001-0001_features.npy"

if not os.path.exists(test_file):
    raise FileNotFoundError(f"Test file not found: {test_file}")

# Load preprocessed features (MFCC or Mel-spectrogram)
test_features = np.load(test_file)

# Convert to PyTorch tensor
test_features = torch.tensor(test_features, dtype=torch.float32)

# Reshape: (Batch, 13, Time Frames)
test_features = test_features.unsqueeze(0)  # Add batch dimension
test_features = test_features.permute(0, 2, 1)  # Ensure correct shape

print("✅ Test features loaded. Shape:", test_features.shape)

# ✅ Step 3: Run Model for Voice Conversion
# -----------------------------------------
with torch.no_grad():  # No gradient calculation needed
    converted_features = model(test_features)

# Convert back to NumPy and reshape
converted_features = converted_features.squeeze(0).permute(1, 0).numpy()  # Shape: (Time Frames, 13)

print("✅ Voice conversion completed!")

# ✅ Step 4: Convert Features to Spectrogram
# ------------------------------------------
# Convert the generated features (MFCC) back to a spectrogram
mel_spectrogram = librosa.feature.inverse.mel_to_stft(converted_features)

# ✅ Step 5: Convert Spectrogram to Audio
# ---------------------------------------
# Use Griffin-Lim algorithm to reconstruct waveform
waveform = librosa.griffinlim(mel_spectrogram)

# ✅ Step 6: Save and Play the Audio
# -----------------------------------
output_wav_path = "D:\\projects\\voice_conversion\\results\\converted_audio.wav"
sf.write(output_wav_path, waveform, samplerate=22050)

print(f"✅ Converted voice saved at: {output_wav_path}")
