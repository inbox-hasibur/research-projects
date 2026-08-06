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
MODEL_PATH = "/kaggle/input/models/inboxhasibur/v1-natsep-htdemucs-temporal-transformer/pytorch/default/1/best_model"
# Fallback in case the path is slightly different
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = "/kaggle/input/v1-natsep-htdemucs-temporal-transformer/pytorch/default/1/best_model"

# 👉 PASTE THE FULL PATH TO YOUR TEST AUDIO FILE HERE
INPUT_AUDIO_PATH = "/kaggle/input/datasets/inboxhasibur/v1-natsep-live-demo-test-data/V1_NatSep_Live_Demo_Test_Data/(Audio) Alan Walker ft Sabrina Carpenter and Farruko  - On My Way.m4a"


# ---------------------------------------------------------
# 🧠 Model Architecture (Standalone for Kaggle)
# ---------------------------------------------------------
class TemporalRhythmTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_layers=2, dropout=0.15):
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

class HeavyEncoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_c, out_c * 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_c * 2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_c, out_c * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_c * 2)
        )
        
    def forward(self, x):
        x = self.conv1(x)
        x = F.glu(x, dim=1)
        x = self.conv2(x)
        x = F.glu(x, dim=1)
        return x

class HeavyDecoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c * 2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(out_c * 2)
        )
        self.conv = nn.Sequential(
            nn.Conv2d(out_c, out_c * 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_c * 2)
        )
        
    def forward(self, x):
        x = self.deconv(x)
        x = F.glu(x, dim=1)
        x = self.conv(x)
        x = F.glu(x, dim=1)
        return x

class HybridHTDemucsTransformer(nn.Module):
    def __init__(self, in_channels=2, n_fft=2048, hop_length=512):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length

        self.enc1 = HeavyEncoderBlock(in_channels, 48)
        self.enc2 = HeavyEncoderBlock(48, 96)
        self.enc3 = HeavyEncoderBlock(96, 192)
        self.enc4 = HeavyEncoderBlock(192, 384)
        self.enc5 = HeavyEncoderBlock(384, 768)

        self.trt = TemporalRhythmTransformer(d_model=768, nhead=12, num_layers=4, dropout=0.15)

        self.dec5 = HeavyDecoderBlock(768, 384)
        self.dec4 = HeavyDecoderBlock(384, 192)
        self.dec3 = HeavyDecoderBlock(192, 96)
        self.dec2 = HeavyDecoderBlock(96, 48)
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(48, in_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
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
        spec_mag, spec_phase, L = self._stft(x)

        e1 = self.enc1(spec_mag)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)

        B, C_bot, F_bot, T_bot = e5.shape
        e5_time = e5.mean(dim=2, keepdim=True)
        trt_out = self.trt(e5_time)
        bottleneck = e5 * trt_out

        d5 = self.dec5(bottleneck)
        if d5.shape != e4.shape: d5 = F.interpolate(d5, size=e4.shape[2:], mode='bilinear', align_corners=False)
        d4 = self.dec4(d5 + e4)
        if d4.shape != e3.shape: d4 = F.interpolate(d4, size=e3.shape[2:], mode='bilinear', align_corners=False)
        d3 = self.dec3(d4 + e3)
        if d3.shape != e2.shape: d3 = F.interpolate(d3, size=e2.shape[2:], mode='bilinear', align_corners=False)
        d2 = self.dec2(d3 + e2)
        if d2.shape != e1.shape: d2 = F.interpolate(d2, size=e1.shape[2:], mode='bilinear', align_corners=False)
        mask = self.dec1(d2 + e1)
        if mask.shape != spec_mag.shape: mask = F.interpolate(mask, size=spec_mag.shape[2:], mode='bilinear', align_corners=False)

        pred_mag = spec_mag * mask
        pred_wav = self._istft(pred_mag, spec_phase, L)
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
    print("⏳ Initializing HybridHTDemucsTransformer...")
    model = HybridHTDemucsTransformer().to(device)
    
    # Load Model Weights
    if os.path.exists(MODEL_PATH):
        model_file_to_load = MODEL_PATH
        if os.path.isdir(MODEL_PATH):
            print(f"⚠️ Model path is a directory (Kaggle automatically unzipped the .pth file). Re-packing to .pth...")
            # We need to zip the contents of the directory, not the directory itself
            zip_path = "/kaggle/working/temp_model"
            shutil.make_archive(zip_path, 'zip', MODEL_PATH)
            if os.path.exists(zip_path + ".pth"):
                os.remove(zip_path + ".pth")
            os.rename(zip_path + ".zip", zip_path + ".pth")
            model_file_to_load = zip_path + ".pth"

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
