# ==============================================================================
# 🎧 LIVE DEMO TEST — HYBRID U-DIT (SAM AUDIO + U-NET + VIT) MODEL
# 📌 Project: Natural and Mechanical Sound Separation (NatSep V2.1)
# 📄 Description: Real-time inference test with Hybrid U-DiT Generative Architecture.
#                 Measures audio quality, speed, lightweightness, & VRAM footprint.
# ==============================================================================

import os
import sys
import time
import shutil
import zipfile
import subprocess
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
import numpy as np
import IPython.display as ipd

# Auto-install missing packages
try:
    import soundfile
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "soundfile", "librosa"])

# ==============================================================================
# ⚙️ CONFIGURATION PATHS (KAGGLE PATHS)
# ==============================================================================

# 📌 1. HYBRID U-DIT V2.1 MODEL PATH
MODEL_PATH = "/kaggle/input/models/inboxhasibur/v2-1-hybrid-u-dit/pytorch/default/1/best_hybrid_udit"


# 📌 2. INPUT TEST AUDIO PATH (PASTE YOUR AUDIO PATH HERE)
INPUT_AUDIO_PATH = "/kaggle/input/datasets/inboxhasibur/v1-natsep-live-demo-test-data/V1_NatSep_Live_Demo_Test_Data/(Audio) Alan Walker ft Sabrina Carpenter and Farruko  - On My Way.m4a"


# 📌 3. AUDIO PREVIEW DURATION (IN SECONDS) FOR KAGGLE BROWSER PLAYER
# Prevents Chrome/Edge "Error code: Out of Memory" browser crashes!
# NOTE: Full-length audio files are ALWAYS saved completely to disk!
PREVIEW_SECONDS = 60


# ==============================================================================


# ---------------------------------------------------------
# 🧠 Hybrid U-DiT Architecture Definition
# ---------------------------------------------------------
class UNetEncoder(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.conv1a = nn.Conv2d(in_channels, 64, 3, padding=1)
        self.conv1b = nn.Conv2d(64, 64, 3, padding=1)
        self.bn1    = nn.BatchNorm2d(64)
        self.pool1  = nn.MaxPool2d(2)
        self.conv2a = nn.Conv2d(64, 128, 3, padding=1)
        self.conv2b = nn.Conv2d(128, 128, 3, padding=1)
        self.bn2    = nn.BatchNorm2d(128)
        self.pool2  = nn.MaxPool2d(2)
        self.conv3a = nn.Conv2d(128, 256, 3, padding=1)
        self.conv3b = nn.Conv2d(256, 256, 3, padding=1)
        self.bn3    = nn.BatchNorm2d(256)
        self.pool3  = nn.MaxPool2d(2)
        self.relu   = nn.ReLU(inplace=True)

    def forward(self, x):
        s1 = self.relu(self.bn1(self.conv1b(self.relu(self.conv1a(x)))))
        x  = self.pool1(s1)
        s2 = self.relu(self.bn2(self.conv2b(self.relu(self.conv2a(x)))))
        x  = self.pool2(s2)
        s3 = self.relu(self.bn3(self.conv3b(self.relu(self.conv3a(x)))))
        x  = self.pool3(s3)
        return x, s1, s2, s3

class ViTAdapter(nn.Module):
    def __init__(self, embed_dim=256, num_heads=4, dropout=0.15):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm      = nn.LayerNorm(embed_dim)
        self.dropout   = nn.Dropout(dropout)
        
    def forward(self, x):
        B, C, H, W = x.shape
        x_seq = x.mean(dim=2).permute(0, 2, 1)
        attn_out, _ = self.attention(x_seq, x_seq, x_seq)
        attn_out = self.norm(x_seq + self.dropout(attn_out))
        attn_out = attn_out.permute(0, 2, 1).unsqueeze(2)
        return x + attn_out

class DiTBackbone(nn.Module):
    def __init__(self, channels=256, depth=4, ffn_dim=1024):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=channels, nhead=8, dropout=0.1,
            dim_feedforward=ffn_dim, batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
    def forward(self, x, time_steps=None):
        B, C, H, W = x.shape
        x_flat = x.view(B, C, H*W).permute(0, 2, 1)
        out = self.transformer(x_flat)
        out = out.permute(0, 2, 1).view(B, C, H, W)
        return out

class UNetDecoder(nn.Module):
    """
    Fully flexible decoder — every upconv layer's in/out channels are passed
    explicitly so it can match any pruned checkpoint exactly.
    """
    def __init__(self,
                 up3_in=256, up3_out=256,
                 conv3_in=None, conv3_out=128,
                 up2_in=128,  up2_out=128,
                 conv2_in=None, conv2_out=64,
                 up1_in=64,   up1_out=64,
                 conv1_in=None,
                 out_channels=1):
        super().__init__()
        # auto-compute cat-merged channel sizes when not provided
        if conv3_in is None:
            conv3_in = up3_out + up3_out   # latent upsampled + skip3
        if conv2_in is None:
            conv2_in = up2_out + up2_out
        if conv1_in is None:
            conv1_in = up1_out + up1_out

        self.upconv3 = nn.ConvTranspose2d(up3_in,  up3_out, kernel_size=2, stride=2)
        self.conv3a  = nn.Conv2d(conv3_in, conv3_out, 3, padding=1)
        self.conv3b  = nn.Conv2d(conv3_out, conv3_out, 3, padding=1)
        self.bn3     = nn.BatchNorm2d(conv3_out)
        self.upconv2 = nn.ConvTranspose2d(up2_in,  up2_out, kernel_size=2, stride=2)
        self.conv2a  = nn.Conv2d(conv2_in, conv2_out, 3, padding=1)
        self.conv2b  = nn.Conv2d(conv2_out, conv2_out, 3, padding=1)
        self.bn2     = nn.BatchNorm2d(conv2_out)
        self.upconv1 = nn.ConvTranspose2d(up1_in,  up1_out, kernel_size=2, stride=2)
        self.conv1a  = nn.Conv2d(conv1_in, 32, 3, padding=1)
        self.conv1b  = nn.Conv2d(32, out_channels, 3, padding=1)
        self.relu    = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, s1, s2, s3):
        x = self.upconv3(x)
        x = torch.cat([x, s3], dim=1)
        x = self.relu(self.bn3(self.conv3b(self.relu(self.conv3a(x)))))
        x = self.upconv2(x)
        x = torch.cat([x, s2], dim=1)
        x = self.relu(self.bn2(self.conv2b(self.relu(self.conv2a(x)))))
        x = self.upconv1(x)
        x = torch.cat([x, s1], dim=1)
        x = self.relu(self.conv1a(x))
        x = self.sigmoid(self.conv1b(x))
        return x

class HybridUDiT(nn.Module):
    def __init__(self,
                 dit_channels=256, dit_ffn=1024, dit_depth=4,
                 up3_in=256, up3_out=256, conv3_in=None, conv3_out=128,
                 up2_in=128, up2_out=128, conv2_in=None, conv2_out=64,
                 up1_in=64,  up1_out=64,  conv1_in=None):
        super().__init__()
        self.encoder      = UNetEncoder()
        self.vit_adapter  = ViTAdapter(embed_dim=dit_channels)
        self.dit_backbone = DiTBackbone(channels=dit_channels, depth=dit_depth, ffn_dim=dit_ffn)
        self.decoder      = UNetDecoder(
            up3_in=up3_in, up3_out=up3_out, conv3_in=conv3_in, conv3_out=conv3_out,
            up2_in=up2_in, up2_out=up2_out, conv2_in=conv2_in, conv2_out=conv2_out,
            up1_in=up1_in, up1_out=up1_out, conv1_in=conv1_in,
        )
        
    def forward(self, x, time_steps=None):
        latent, s1, s2, s3 = self.encoder(x)
        latent = self.vit_adapter(latent)
        denoised_latent = self.dit_backbone(latent, time_steps)
        out = self.decoder(denoised_latent, s1, s2, s3)
        return out


def build_model_from_checkpoint(state_dict, device):
    """
    Reads ALL key tensor shapes directly from the pruned checkpoint and
    constructs a HybridUDiT whose architecture exactly matches the saved weights.

    ConvTranspose2d weight layout: [in_channels, out_channels, kH, kW]
    Conv2d          weight layout: [out_channels, in_channels, kH, kW]
    """
    sd = state_dict

    # --- DiTBackbone ---
    dit_channels = sd["dit_backbone.transformer.layers.0.self_attn.in_proj_weight"].shape[1]
    dit_ffn      = sd["dit_backbone.transformer.layers.0.linear1.weight"].shape[0]
    dit_depth    = sum(1 for k in sd
                       if k.startswith("dit_backbone.transformer.layers.")
                       and k.endswith(".self_attn.in_proj_weight"))

    # --- Decoder upconv shapes (ConvTranspose2d: shape = [in_c, out_c, kH, kW]) ---
    up3_in,  up3_out  = sd["decoder.upconv3.weight"].shape[:2]
    up2_in,  up2_out  = sd["decoder.upconv2.weight"].shape[:2]
    up1_in,  up1_out  = sd["decoder.upconv1.weight"].shape[:2]

    # --- Decoder conv shapes (Conv2d: shape = [out_c, in_c, kH, kW]) ---
    conv3_out = sd["decoder.conv3a.weight"].shape[0]
    conv3_in  = sd["decoder.conv3a.weight"].shape[1]   # = up3_out + skip3_ch
    conv2_out = sd["decoder.conv2a.weight"].shape[0]
    conv2_in  = sd["decoder.conv2a.weight"].shape[1]
    conv1_in  = sd["decoder.conv1a.weight"].shape[1]   # = up1_out + skip1_ch

    print(f"ℹ️  Checkpoint dims detected:")
    print(f"     DiT: channels={dit_channels}, ffn_dim={dit_ffn}, depth={dit_depth}")
    print(f"     Decoder upconv:  up3({up3_in}→{up3_out}), up2({up2_in}→{up2_out}), up1({up1_in}→{up1_out})")
    print(f"     Decoder conv3a: in={conv3_in} → out={conv3_out}")

    model = HybridUDiT(
        dit_channels=dit_channels, dit_ffn=dit_ffn, dit_depth=dit_depth,
        up3_in=up3_in, up3_out=up3_out, conv3_in=conv3_in, conv3_out=conv3_out,
        up2_in=up2_in, up2_out=up2_out, conv2_in=conv2_in, conv2_out=conv2_out,
        up1_in=up1_in, up1_out=up1_out, conv1_in=conv1_in,
    ).to(device)

    missing, unexpected = model.load_state_dict(sd, strict=True)
    return model


# ---------------------------------------------------------
# 🛠️ Helper & Signal Processing Functions
# ---------------------------------------------------------
def load_audio(file_path, target_sr=44100):
    """Loads audio and converts to stereo 44.1kHz tensor (2, L)."""
    wav, sr = torchaudio.load(file_path)
    if sr != target_sr:
        wav = T.Resample(orig_freq=sr, new_freq=target_sr)(wav)
    if wav.shape[0] == 1:
        wav = wav.repeat(2, 1)  # Mono to stereo
    elif wav.shape[0] > 2:
        wav = wav[:2, :]
    return wav, target_sr

def save_audio(wav, file_path, sr=44100):
    """Saves tensor audio to file."""
    torchaudio.save(file_path, wav, sr)

def process_audio(model, audio_tensor, sr=44100, chunk_duration=6.0, device="cuda"):
    """Processes audio through Hybrid U-DiT model in chunks."""
    chunk_samples = int(chunk_duration * sr)
    total_samples = audio_tensor.shape[1]
    output_tensor = torch.zeros_like(audio_tensor)

    n_fft = 2048
    hop_length = 512
    mel_transform = T.MelSpectrogram(sample_rate=sr, n_fft=n_fft, hop_length=hop_length, n_mels=128).to(device)
    amplitude_to_db = T.AmplitudeToDB(top_db=80).to(device)

    model.eval()
    with torch.no_grad():
        for start in range(0, total_samples, chunk_samples):
            end = min(start + chunk_samples, total_samples)
            chunk = audio_tensor[:, start:end]

            pad_size = 0
            if chunk.shape[1] < chunk_samples:
                pad_size = chunk_samples - chunk.shape[1]
                chunk = F.pad(chunk, (0, pad_size))

            chunk_device = chunk.to(device)

            window = torch.hann_window(n_fft, device=device)
            spec = torch.stft(chunk_device, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True)
            spec_mag = torch.abs(spec).unsqueeze(0)
            spec_phase = torch.angle(spec).unsqueeze(0)
            orig_L = chunk_device.shape[-1]

            mono_chunk = chunk_device.mean(dim=0, keepdim=True)
            mel = mel_transform(mono_chunk)
            mel_db = amplitude_to_db(mel)
            mel_norm = (mel_db + 80.0) / 80.0
            mel_norm = mel_norm.clamp(0.0, 1.0).unsqueeze(0)

            T_bins = mel_norm.shape[-1]
            pad_t = (8 - (T_bins % 8)) % 8
            if pad_t > 0:
                mel_norm = F.pad(mel_norm, (0, pad_t))

            dummy_time = torch.zeros(1, dtype=torch.long, device=device)
            pred_mask_mel = model(mel_norm, dummy_time)

            if pad_t > 0:
                pred_mask_mel = pred_mask_mel[..., :-pad_t]

            stft_bins = spec_mag.shape[2]
            real_T = spec_mag.shape[3]
            pred_mask_stft = F.interpolate(pred_mask_mel, size=(stft_bins, real_T), mode="bilinear", align_corners=False)

            pred_mag = spec_mag * pred_mask_stft
            spec_complex = torch.polar(pred_mag.squeeze(0), spec_phase.squeeze(0))
            pred_chunk = torch.istft(spec_complex, n_fft=n_fft, hop_length=hop_length, window=window, length=orig_L).cpu()

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

    # Initialize & Load Model — architecture rebuilt from checkpoint shapes
    print("\n⏳ Loading Hybrid U-DiT V2.1 weights & reconstructing pruned architecture...")
    model = None

    if os.path.exists(MODEL_PATH):
        model_file_to_load = MODEL_PATH
        if os.path.isdir(MODEL_PATH):
            print(f"⚠️ Model path is a directory ({MODEL_PATH}). Re-packing to temporary .pth...")
            zip_path = "/kaggle/working/temp_hybrid_udit_v21.pth"
            if os.path.exists(zip_path):
                os.remove(zip_path)
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_STORED) as zipf:
                for root, _, files in os.walk(MODEL_PATH):
                    for file in files:
                        file_path = os.path.join(root, file)
                        rel_path = os.path.relpath(file_path, MODEL_PATH)
                        zipf.write(file_path, os.path.join("archive", rel_path))
            model_file_to_load = zip_path

        print(f"✅ Loading checkpoint from: {model_file_to_load}")
        try:
            state_dict = torch.load(model_file_to_load, map_location=device, weights_only=False)
            if "model_state_dict" in state_dict:
                state_dict = state_dict["model_state_dict"]
            elif "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]

            # ✅ KEY FIX: Rebuild architecture dynamically from checkpoint shapes
            model = build_model_from_checkpoint(state_dict, device)
            print("✅ Hybrid U-DiT V2.1 model weights loaded successfully.")
        except Exception as e:
            print(f"❌ Failed to load model weights: {e}")
            model = HybridUDiT().to(device)
    else:
        print(f"⚠️ Warning: Model path '{MODEL_PATH}' not found! Using random weights.")
        model = HybridUDiT().to(device)

    # Load Input Audio
    if not os.path.exists(INPUT_AUDIO_PATH):
        print(f"❌ Error: Input file does not exist at {INPUT_AUDIO_PATH}")
        return
        
    print(f"🎧 Loading audio: {os.path.basename(INPUT_AUDIO_PATH)}")
    audio, sr = load_audio(INPUT_AUDIO_PATH)
    
    # Display Original Audio in Kaggle Notebook
    print("\n🎵 Original Audio:")
    if audio.shape[-1] > int(PREVIEW_SECONDS * sr):
        orig_preview = f"preview_original_v21_{os.path.basename(INPUT_AUDIO_PATH)}.wav"
        save_audio(audio[:, :int(PREVIEW_SECONDS * sr)], orig_preview, sr=sr)
        ipd.display(ipd.Audio(filename=orig_preview))
    else:
        ipd.display(ipd.Audio(filename=INPUT_AUDIO_PATH))
    
    # Reset GPU stats and start timer
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()
    start_time = time.perf_counter()

    print("\n⚙️ Processing audio with Hybrid U-DiT V2.1... Please wait!")
    separated_audio = process_audio(model, audio, device=device)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time = time.perf_counter()

    proc_time = end_time - start_time
    audio_duration_sec = audio.shape[-1] / sr
    rtf = proc_time / audio_duration_sec
    speedup = 1.0 / rtf if rtf > 0 else 0.0

    # Calculate model size & parameters
    params_m = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2)
    peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2) if torch.cuda.is_available() else 0.0

    # Save Full Output Audio File to Disk
    output_filename = "v2_1_separated_output_hybrid_udit.wav"
    save_audio(separated_audio, output_filename, sr)
    print(f"\n✅ Done! The full separated audio has been saved to: {output_filename}")
    
    # Display Lightweight Preview Player in Notebook
    print(f"\n🎵 Separated Target Audio (Playing first {PREVIEW_SECONDS}s preview in notebook player):")
    if separated_audio.shape[-1] > int(PREVIEW_SECONDS * sr):
        sep_preview = f"preview_{output_filename}"
        save_audio(separated_audio[:, :int(PREVIEW_SECONDS * sr)], sep_preview, sr=sr)
        ipd.display(ipd.Audio(filename=sep_preview))
    else:
        ipd.display(ipd.Audio(filename=output_filename))

    # 📊 Print Efficiency & Resource Consumption Report
    print("\n" + "="*60)
    print("⚡ MODEL EFFICIENCY & RESOURCE CONSUMPTION REPORT")
    print("="*60)
    print(f"📦 Model Name:               Hybrid U-DiT V2.1 (Edge SOTA)")
    print(f"🧩 Total Parameters:         {params_m:.2f} M")
    print(f"💾 Model Weight Size:        {size_mb:.2f} MB")
    print(f"⏱️ Audio Processing Time:    {proc_time:.2f} seconds")
    print(f"🎧 Test Audio Duration:      {audio_duration_sec:.2f} seconds")
    print(f"⚡ Real-Time Factor (RTF):   {rtf:.4f} (Lower is Faster)")
    print(f"🚀 Speedup Multiplier:       {speedup:.1f}x Real-Time")
    if torch.cuda.is_available():
        print(f"🔋 Peak GPU VRAM Memory:     {peak_vram_mb:.2f} MB")
    print("="*60)

if __name__ == "__main__":
    main()

# ==============================================================================
# 🏁 END OF SCRIPT — LIVE DEMO TEST (HYBRID U-DIT V2.1)
# ==============================================================================
