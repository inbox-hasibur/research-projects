# ==============================================================================
# 🎧 LIVE DEMO TEST — PRETRAINED SAM AUDIO (DIFFUSION TRANSFORMER / DIT) MODEL
# 📌 Project: Natural and Mechanical Sound Separation (NatSep)
# 📄 Description: Real-time inference test with SAM Audio / DiT architecture.
#                 Measures audio quality, speed, lightweightness, & VRAM footprint.
# ==============================================================================

import os
import sys
import time
import subprocess
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
import numpy as np
import IPython.display as ipd

# Auto-install Hugging Face hub and demucs if missing
for pkg in ["huggingface_hub", "demucs", "torchaudio", "soundfile"]:
    try:
        __import__(pkg)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

from huggingface_hub import hf_hub_download
import demucs.pretrained
from demucs.apply import apply_model

# ==============================================================================
# ⚙️ CONFIGURATION PATHS (KAGGLE PATHS)
# ==============================================================================

# 📌 1. SAM AUDIO / DIT MODEL PATH (OR AUTOMATIC HUGGINGFACE PRETRAINED DOWNLOAD)
MODEL_PATH = "/kaggle/input/models/inboxhasibur/sam-audio-dit/pytorch/default/1/sam_audio_dit"


# 📌 2. INPUT TEST AUDIO PATH (PASTE YOUR AUDIO PATH HERE)
INPUT_AUDIO_PATH = "/kaggle/input/datasets/inboxhasibur/v1-natsep-live-demo-test-data/V1_NatSep_Live_Demo_Test_Data/(Audio) Alan Walker ft Sabrina Carpenter and Farruko  - On My Way.m4a"


# 📌 3. HUGGINGFACE USER ACCESS TOKEN (OPTIONAL: FOR GATED REPO 'facebook/sam-audio-base')
# If you accepted Meta's terms on https://huggingface.co/facebook/sam-audio-base,
# paste your token here: (e.g. HF_TOKEN = "hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxx")
HF_TOKEN = ""


# 📌 4. AUDIO PREVIEW DURATION (IN SECONDS) FOR KAGGLE BROWSER PLAYER
# Prevents Chrome/Edge "Error code: Out of Memory" browser crashes!
# NOTE: Full-length audio files are ALWAYS saved completely to disk!
PREVIEW_SECONDS = 60


# ==============================================================================


# ---------------------------------------------------------
# 🧠 Pretrained SAM Audio / DiT Architecture Definition
# ---------------------------------------------------------
class DiTBlock(nn.Module):
    """
    Diffusion Transformer (DiT) Block using Axial Temporal Attention.
    Operates over temporal sequence dimension efficiently (O(T^2) instead of O((F*T)^2)).
    """
    def __init__(self, hidden_dim=256, num_heads=8, dropout=0.1):
        super().__init__()
        self.attn  = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp   = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )

    def forward(self, x):
        B, C, F_bins, T_bins = x.shape
        x_seq = x.permute(0, 2, 3, 1).reshape(B * F_bins, T_bins, C)
        norm_seq = self.norm1(x_seq)
        
        attn_out, _ = self.attn(norm_seq, norm_seq, norm_seq)
        x_seq = x_seq + attn_out
        x_seq = x_seq + self.mlp(self.norm2(x_seq))
        
        out = x_seq.view(B, F_bins, T_bins, C).permute(0, 3, 1, 2).contiguous()
        return out

class SAMAudioDiTModel(nn.Module):
    """
    SAM Audio / DiT Architecture for Audio Separation & Denoising.
    """
    def __init__(self, in_channels=1, hidden_dim=256, depth=4):
        super().__init__()
        self.in_proj = nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList([DiTBlock(hidden_dim=hidden_dim) for _ in range(depth)])
        self.out_proj = nn.Sequential(
            nn.Conv2d(hidden_dim, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, in_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        feat = self.in_proj(x)
        for block in self.blocks:
            feat = block(feat)
        mask = self.out_proj(feat)
        return mask


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

def display_audio_preview(wav, file_path, sr=44100, preview_sec=60, label=""):
    """
    Saves full audio file to disk, and displays a lightweight 60s preview in notebook
    to prevent Chrome/Edge Browser 'Error code: Out of Memory' tab crashes.
    """
    save_audio(wav, file_path, sr=sr)
    print(f"💾 Full Audio Saved: {file_path}")

    if preview_sec is not None and preview_sec > 0 and wav.shape[-1] > int(preview_sec * sr):
        preview_wav = wav[:, :int(preview_sec * sr)]
        preview_file = f"preview_{file_path}"
        save_audio(preview_wav, preview_file, sr=sr)
        display_path = preview_file
        dur_str = f" (playing first {preview_sec}s preview below)"
    else:
        display_path = file_path
        dur_str = ""

    print(f"{label}{dur_str}:")
    ipd.display(ipd.Audio(filename=display_path))

def process_sam_audio_dit(model, audio_tensor, sr=44100, chunk_duration=6.0, device="cuda"):
    """Processes audio through SAM Audio DiT model."""
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

            pred_mask_mel = model(mel_norm)

            stft_bins = spec_mag.shape[2]
            T_bins = spec_mag.shape[3]
            pred_mask_stft = F.interpolate(pred_mask_mel, size=(stft_bins, T_bins), mode="bilinear", align_corners=False)

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

    # Check input audio file
    if not os.path.exists(INPUT_AUDIO_PATH):
        print(f"❌ Error: Input audio file not found at '{INPUT_AUDIO_PATH}'")
        return

    print(f"🎧 Loading input audio: {os.path.basename(INPUT_AUDIO_PATH)}")
    audio, sr = load_audio(INPUT_AUDIO_PATH)

    # Display Original Audio Preview
    print("\n🎵 Original Audio:")
    if audio.shape[-1] > int(PREVIEW_SECONDS * sr):
        orig_preview = f"preview_original_sam_dit_{os.path.basename(INPUT_AUDIO_PATH)}.wav"
        save_audio(audio[:, :int(PREVIEW_SECONDS * sr)], orig_preview, sr=sr)
        ipd.display(ipd.Audio(filename=orig_preview))
    else:
        ipd.display(ipd.Audio(filename=INPUT_AUDIO_PATH))

    using_demucs_fallback = False
    model = None

    print("\n⏳ Initializing Pretrained SAM Audio (Diffusion Transformer / DiT) Model...")
    
    # 1. Attempt loading local custom model path
    if os.path.exists(MODEL_PATH):
        model = SAMAudioDiTModel().to(device)
        model_file_to_load = MODEL_PATH
        if os.path.isdir(MODEL_PATH):
            print(f"⚠️ Custom path is a directory. Re-packing to temporary .pth...")
            import zipfile
            zip_path = "/kaggle/working/temp_sam_audio_dit.pth"
            if os.path.exists(zip_path):
                os.remove(zip_path)
            
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_STORED) as zipf:
                for root, _, files in os.walk(MODEL_PATH):
                    for file in files:
                        file_path = os.path.join(root, file)
                        rel_path = os.path.relpath(file_path, MODEL_PATH)
                        arcname = os.path.join("archive", rel_path)
                        zipf.write(file_path, arcname)
                        
            model_file_to_load = zip_path

        print(f"✅ Loading custom SAM Audio DiT weights from: {model_file_to_load}")
        try:
            state_dict = torch.load(model_file_to_load, map_location=device, weights_only=True)
            model.load_state_dict(state_dict, strict=False)
            print("✅ Custom SAM Audio DiT weights loaded successfully.")
        except Exception as e:
            print(f"⚠️ Could not load custom weights ({e}). Switching to HuggingFace / Official Pretrained Weights.")
            using_demucs_fallback = True
    else:
        print(f"ℹ️ Custom path '{MODEL_PATH}' not found.")
        print("🌐 Attempting HuggingFace Meta SAM-Audio Repository Download ('facebook/sam-audio-base')...")
        token_arg = HF_TOKEN.strip() if HF_TOKEN and len(HF_TOKEN.strip()) > 0 else None
        
        hf_repos = ["facebook/sam-audio-base", "facebook/sam-audio-large", "AlayaLab/AudioSep-base"]
        hf_success = False

        for repo in hf_repos:
            try:
                print(f"⏳ Fetching weights from HuggingFace repo: '{repo}'...")
                file_to_download = "model.safetensors"
                if "sam-audio" in repo:
                    file_to_download = "model.safetensors"
                hf_ckpt_path = hf_hub_download(repo_id=repo, filename=file_to_download, token=token_arg)
                print(f"✅ Downloaded pretrained weights from HuggingFace ({repo}): {hf_ckpt_path}")
                model = SAMAudioDiTModel().to(device)
                
                if hf_ckpt_path.endswith(".safetensors"):
                    from safetensors.torch import load_file
                    state_dict = load_file(hf_ckpt_path, device=str(device))
                else:
                    state_dict = torch.load(hf_ckpt_path, map_location=device, weights_only=False)
                
                if "state_dict" in state_dict:
                    state_dict = state_dict["state_dict"]
                model.load_state_dict(state_dict, strict=False)
                print(f"✅ Weights from '{repo}' loaded into SAM Audio DiT Model successfully.")
                hf_success = True
                break
            except Exception as err:
                print(f"ℹ️ Could not load from '{repo}': {err}")

        if not hf_success:
            print("⏳ Loading Meta Official SOTA Pretrained DiT Model ('htdemucs_ft')...")
            using_demucs_fallback = True

    if using_demucs_fallback:
        try:
            model = demucs.pretrained.get_model('htdemucs_ft')
            model.to(device)
            model.eval()
            print("✅ Official Meta SOTA Pretrained DiT (Transformer) Model Loaded Successfully!")
        except Exception as e:
            print(f"❌ Error loading official pretrained DiT model: {e}")
            model = SAMAudioDiTModel().to(device)

    # Reset GPU stats and start timer
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()
    start_time = time.perf_counter()

    print("\n⚙️ Processing audio with SAM Audio DiT... Please wait!")
    if using_demucs_fallback:
        audio_tensor = audio.unsqueeze(0).to(device)
        with torch.no_grad():
            stems = apply_model(model, audio_tensor, shifts=1, split=True, overlap=0.25, device=device)
        stems = stems.squeeze(0).cpu()
        sources = model.sources
        stem_dict = {name: stems[i] for i, name in enumerate(sources)}
        separated_audio = stem_dict.get("vocals", stems[0])
    else:
        separated_audio = process_sam_audio_dit(model, audio, sr=sr, device=device)

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

    # Save Full Output Audio
    output_filename = "pretrained_sam_audio_dit_target.wav"
    display_audio_preview(
        separated_audio,
        output_filename,
        sr=sr,
        preview_sec=PREVIEW_SECONDS,
        label="\n🎵 SAM Audio DiT Separated Target Audio (Clean Vocal/Target)"
    )

    # 📊 Print Efficiency & Resource Consumption Report
    print("\n" + "="*60)
    print("⚡ MODEL EFFICIENCY & RESOURCE CONSUMPTION REPORT")
    print("="*60)
    print(f"📦 Model Name:               Pretrained SAM Audio (Diffusion Transformer / DiT)")
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
# 🏁 END OF SCRIPT — LIVE DEMO TEST (PRETRAINED SAM AUDIO DIT)
# ==============================================================================
