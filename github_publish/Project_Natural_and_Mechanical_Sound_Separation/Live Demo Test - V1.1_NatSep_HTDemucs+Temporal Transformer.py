import os
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np
import IPython.display as ipd

# ---------------------------------------------------------
# ⚙️ Configuration Paths (Kaggle Paths)
# ---------------------------------------------------------
# Kaggle model path
MODEL_PATH = "/kaggle/input/models/inboxhasibur/v1-1-natsep-htdemucs-temporal-transformer/pytorch/default/1/best_model"

# 👉 PASTE THE FULL PATH TO YOUR TEST AUDIO FILE HERE
INPUT_AUDIO_PATH = "/kaggle/input/datasets/inboxhasibur/v1-natsep-live-demo-test-data/V1_NatSep_Live_Demo_Test_Data/(Audio) Alan Walker ft Sabrina Carpenter and Farruko  - On My Way.m4a"


# Auto-install demucs if missing (Kaggle environment support)
try:
    import demucs.pretrained
except ImportError:
    print("📦 'demucs' library not found. Auto-installing demucs...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "demucs"])
    import demucs.pretrained
    print("✅ 'demucs' successfully installed!")

# ---------------------------------------------------------
# 🧠 Model Architecture (True Transfer Learning)
# ---------------------------------------------------------
class TemporalRhythmTransformer(nn.Module):
    def __init__(self, d_model=8, nhead=4, num_layers=2, dropout=0.15):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        B, C, F_dim, T_dim = x.shape
        x_reshaped = x.permute(0, 3, 1, 2).contiguous().view(B, T_dim, C * F_dim)
        out = self.transformer(x_reshaped)
        out = out.view(B, T_dim, C, F_dim).permute(0, 2, 3, 1)
        return out

class TrueHybridHTDemucsTRT(nn.Module):
    def __init__(self, in_channels=2, n_fft=2048, hop_length=512):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length

        print("⏳ Downloading/Loading Official Meta HTDemucs Weights...")
        import demucs.pretrained
        self.demucs = demucs.pretrained.get_model('htdemucs')

        self.trt = TemporalRhythmTransformer(d_model=8, nhead=4, num_layers=2, dropout=0.15)

        self.mixer = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 2, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def _stft(self, x):
        B, C, L = x.shape
        x_flat = x.reshape(B * C, L)
        window = torch.hann_window(self.n_fft, device=x.device)
        spec = torch.stft(x_flat, n_fft=self.n_fft, hop_length=self.hop_length, window=window, return_complex=True)
        _, F_bins, T_bins = spec.shape
        spec_mag = torch.abs(spec).reshape(B, C, F_bins, T_bins)
        spec_phase = torch.angle(spec).reshape(B, C, F_bins, T_bins)
        return spec_mag, spec_phase, L

    def _istft(self, spec_mag, spec_phase, length):
        B, C, F_bins, T_bins = spec_mag.shape
        spec_complex = torch.polar(spec_mag, spec_phase).reshape(B * C, F_bins, T_bins)
        window = torch.hann_window(self.n_fft, device=spec_mag.device)
        wav_flat = torch.istft(spec_complex, n_fft=self.n_fft, hop_length=self.hop_length, window=window, length=length)
        return wav_flat.reshape(B, C, length)

    def forward(self, x):
        with torch.no_grad():
            from demucs.apply import apply_model
            demucs_stems = apply_model(self.demucs, x) 
            
        B, S, C, L = demucs_stems.shape
        stems_merged = demucs_stems.reshape(B, S * C, L)
        stems_mag, _, _ = self._stft(stems_merged)
        
        stems_time = stems_mag.mean(dim=2, keepdim=True)
        trt_out = self.trt(stems_time)
        trt_features = stems_mag * trt_out
        mask = self.mixer(trt_features)
        
        input_mag, input_phase, _ = self._stft(x)
        pred_mag = input_mag * mask
        pred_wav = self._istft(pred_mag, input_phase, L)
        
        return pred_wav

# ---------------------------------------------------------
# 🛠️ Helper Functions
# ---------------------------------------------------------
def load_audio(file_path, target_sr=44100):
    wav, sr = torchaudio.load(file_path)
    if sr != target_sr:
        wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)(wav)
    if wav.shape[0] == 1:
        wav = wav.repeat(2, 1) # Convert to stereo
    elif wav.shape[0] > 2:
        wav = wav[:2, :]
    return wav, target_sr

def save_audio(wav, file_path, sr=44100):
    torchaudio.save(file_path, wav, sr)

def process_audio(model, audio_tensor, chunk_duration=6.0, sr=44100, device="cuda"):
    """
    Process audio in chunks to avoid OOM errors.
    """
    chunk_samples = int(chunk_duration * sr)
    total_samples = audio_tensor.shape[1]
    
    output_tensor = torch.zeros_like(audio_tensor)
    
    model.eval()
    with torch.no_grad():
        for start in range(0, total_samples, chunk_samples):
            end = min(start + chunk_samples, total_samples)
            chunk = audio_tensor[:, start:end]
            
            pad_size = 0
            if chunk.shape[1] < chunk_samples:
                pad_size = chunk_samples - chunk.shape[1]
                chunk = torch.nn.functional.pad(chunk, (0, pad_size))
                
            chunk = chunk.unsqueeze(0).to(device)
            
            # Predict
            pred_chunk = model(chunk)
            pred_chunk = pred_chunk.squeeze(0).cpu()
            
            if pad_size > 0:
                pred_chunk = pred_chunk[:, :-pad_size]
                
            output_tensor[:, start:end] = pred_chunk
            
    return output_tensor

# ---------------------------------------------------------
# 🚀 Main Execution Pipeline
# ---------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Using device: {device}")
    
    # Initialize Model
    print("⏳ Initializing TrueHybridHTDemucsTRT...")
    model = TrueHybridHTDemucsTRT().to(device)
    
    # Load Model Weights
    if os.path.exists(MODEL_PATH):
        model_file_to_load = MODEL_PATH
        if os.path.isdir(MODEL_PATH):
            print(f"⚠️ Model path is a directory (Kaggle automatically unzipped the .pth file). Re-packing to .pth...")
            import zipfile
            zip_path = "/kaggle/working/temp_model.pth"
            if os.path.exists(zip_path):
                os.remove(zip_path)
            
            # PyTorch's C++ zip reader expects the 'version' file. Adding an 'archive/' prefix
            # and using ZIP_STORED (uncompressed) ensures maximum compatibility with torch.load.
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_STORED) as zipf:
                for root, _, files in os.walk(MODEL_PATH):
                    for file in files:
                        file_path = os.path.join(root, file)
                        rel_path = os.path.relpath(file_path, MODEL_PATH)
                        # Add 'archive' prefix to match PyTorch's default zip structure
                        arcname = os.path.join("archive", rel_path)
                        zipf.write(file_path, arcname)
                        
            model_file_to_load = zip_path

        print(f"✅ Loading weights from: {model_file_to_load}")
        try:
            state_dict = torch.load(model_file_to_load, map_location=device, weights_only=True)
            model.load_state_dict(state_dict, strict=False)
            print("✅ Model weights loaded successfully.")
        except Exception as e:
            print(f"❌ Failed to load model weights: {e}")
    else:
        print(f"⚠️ Warning: Model path '{MODEL_PATH}' not found. Running with untrained weights!")

    # Setup Input Audio
    input_audio_path = INPUT_AUDIO_PATH
    
    if not os.path.exists(input_audio_path):
        print(f"❌ Error: Input file does not exist at {input_audio_path}")
        return
        
    print(f"🎧 Loading audio: {os.path.basename(input_audio_path)}")
    audio, sr = load_audio(input_audio_path)
    
    # Limit to first 30 seconds for quick testing (Optional)
    # audio = audio[:, :30 * sr] 
    
    # Display Original Audio in Kaggle
    print("\n🎵 Original Audio:")
    ipd.display(ipd.Audio(audio.numpy(), rate=sr))
    
    print("⚙️ Processing audio (separating natural/vocal sounds)... Please wait!")
    separated_audio = process_audio(model, audio, device=device)
    
    # Save the output
    output_filename = "separated_output.wav"
    save_audio(separated_audio, output_filename, sr)
    
    print(f"\n✅ Done! The separated audio has been saved to: {output_filename}")
    
    # Display Separated Audio in Kaggle
    print("🎵 Separated Target Audio (Listen Below):")
    ipd.display(ipd.Audio(separated_audio.numpy(), rate=sr))

if __name__ == "__main__":
    main()
