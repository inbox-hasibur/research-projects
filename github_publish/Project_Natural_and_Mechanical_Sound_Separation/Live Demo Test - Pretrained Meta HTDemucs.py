# ==============================================================================
# 🎧 LIVE DEMO TEST — PRETRAINED META HTDEMUCS MODEL
# 📌 Project: Natural and Mechanical Sound Separation (NatSep)
# 📄 Description: Real-time inference test with official Meta HTDemucs model.
#                 Measures audio quality, speed, lightweightness, & VRAM footprint.
# ==============================================================================

import os
import sys
import time
import subprocess
import torch
import torchaudio
import torchaudio.transforms as T
import numpy as np
import IPython.display as ipd

# Auto-install demucs if missing (Kaggle environment support)
try:
    import demucs.pretrained
    from demucs.apply import apply_model
except ImportError:
    print("📦 'demucs' library not found. Auto-installing demucs...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "demucs"])
    import demucs.pretrained
    from demucs.apply import apply_model
    print("✅ 'demucs' successfully installed!")

# ==============================================================================
# ⚙️ CONFIGURATION PATHS (KAGGLE PATHS)
# ==============================================================================

# 📌 1. INPUT TEST AUDIO PATH (PASTE YOUR AUDIO PATH HERE)
INPUT_AUDIO_PATH = "/kaggle/input/datasets/inboxhasibur/v1-natsep-live-demo-test-data/V1_NatSep_Live_Demo_Test_Data/(Audio) Alan Walker ft Sabrina Carpenter and Farruko  - On My Way.m4a"


# 📌 2. OFFICIAL PRETRAINED MODEL NAME
# Options: 'htdemucs_ft' (Meta's best fine-tuned HTDemucs), 'htdemucs', 'mdx_extra'
MODEL_NAME = "htdemucs_ft"


# 📌 3. AUDIO PREVIEW DURATION (IN SECONDS) FOR KAGGLE BROWSER PLAYER
# Prevents Chrome/Edge "Error code: Out of Memory" browser crashes!
# NOTE: Full-length audio files are ALWAYS saved completely to disk!
PREVIEW_SECONDS = 60


# ==============================================================================


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

def display_audio_preview(wav, file_path, sr=44100, preview_sec=30, label=""):
    """
    Saves full audio file to disk, and displays a lightweight 30s preview in notebook
    to prevent Chrome/Edge Browser 'Error code: Out of Memory' tab crashes.
    """
    # 1. Save full length audio file to disk
    save_audio(wav, file_path, sr=sr)
    print(f"💾 Full Audio Saved: {file_path}")

    # 2. Create lightweight preview snippet for browser player
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

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Using device: {device}")

    # Load Official Pretrained Meta HTDemucs Model
    print(f"⏳ Downloading & Loading Official Pretrained Meta Model '{MODEL_NAME}'...")
    try:
        model = demucs.pretrained.get_model(MODEL_NAME)
    except Exception as e:
        print(f"⚠️ Could not load '{MODEL_NAME}', falling back to 'htdemucs': {e}")
        model = demucs.pretrained.get_model("htdemucs")

    model.to(device)
    model.eval()
    print(f"✅ Pretrained Model Loaded! Model sources: {model.sources}")

    # Check input audio file
    if not os.path.exists(INPUT_AUDIO_PATH):
        print(f"❌ Error: Input audio file not found at '{INPUT_AUDIO_PATH}'")
        return

    print(f"🎧 Loading input audio: {os.path.basename(INPUT_AUDIO_PATH)}")
    audio, sr = load_audio(INPUT_AUDIO_PATH, target_sr=model.samplerate)

    # Display Original Audio Preview
    print("\n🎵 Original Audio:")
    if audio.shape[-1] > int(PREVIEW_SECONDS * sr):
        orig_preview = f"preview_original_{os.path.basename(INPUT_AUDIO_PATH)}.wav"
        save_audio(audio[:, :int(PREVIEW_SECONDS * sr)], orig_preview, sr=sr)
        ipd.display(ipd.Audio(filename=orig_preview))
    else:
        ipd.display(ipd.Audio(filename=INPUT_AUDIO_PATH))

    # Reshape audio for demucs apply_model: (1, channels, samples)
    audio_tensor = audio.unsqueeze(0).to(device)

    # Reset GPU stats and start timer
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()
    start_time = time.perf_counter()

    print("\n⚙️ Processing audio with official pretrained HTDemucs... Please wait!")
    with torch.no_grad():
        stems = apply_model(
            model,
            audio_tensor,
            shifts=1,
            split=True,
            overlap=0.25,
            progress=True,
            device=device
        )
    
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

    # stems shape: (1, num_sources, channels, samples)
    stems = stems.squeeze(0).cpu()  # (num_sources, 2, L)
    sources = model.sources         # ['drums', 'bass', 'other', 'vocals']

    print("\n✅ Separation complete! Saving separated stems and preparing players...")

    # Map sources to dictionary
    stem_dict = {name: stems[i] for i, name in enumerate(sources)}

    # Header for results
    print("\n" + "="*60)
    print("🎧 PRETRAINED HTDEMUCS SEPARATION RESULTS (BROWSER SAFE)")
    print("="*60)

    # 1. Vocals Only Stem
    if "vocals" in stem_dict:
        display_audio_preview(
            stem_dict["vocals"],
            "pretrained_htdemucs_vocals.wav",
            sr=sr,
            preview_sec=PREVIEW_SECONDS,
            label="\n🎤 Vocals Only (Clean Vocal - Music Removed)"
        )

    # 2. Target Audio (Vocals + Other)
    if "vocals" in stem_dict and "other" in stem_dict:
        target_wav = stem_dict["vocals"] + stem_dict["other"]
        max_val = torch.max(torch.abs(target_wav))
        if max_val > 1.0:
            target_wav = target_wav / max_val

        display_audio_preview(
            target_wav,
            "pretrained_htdemucs_target_vocals_plus_other.wav",
            sr=sr,
            preview_sec=PREVIEW_SECONDS,
            label="\n🎵 Target Audio (Vocals + Natural/Other Sounds Combined)"
        )

    # 3. Other / Instrumental Stem
    if "other" in stem_dict:
        display_audio_preview(
            stem_dict["other"],
            "pretrained_htdemucs_other.wav",
            sr=sr,
            preview_sec=PREVIEW_SECONDS,
            label="\n🎸 Other / Instrumental Stem"
        )

    # 4. Removed Drums & Bass Stem
    if "drums" in stem_dict and "bass" in stem_dict:
        bg_wav = stem_dict["drums"] + stem_dict["bass"]
        max_val = torch.max(torch.abs(bg_wav))
        if max_val > 1.0:
            bg_wav = bg_wav / max_val

        display_audio_preview(
            bg_wav,
            "pretrained_htdemucs_removed_drums_and_bass.wav",
            sr=sr,
            preview_sec=PREVIEW_SECONDS,
            label="\n🥁 Removed Drums & Bass (Backing Track)"
        )

    # 📊 Print Efficiency & Resource Consumption Report
    print("\n" + "="*60)
    print("⚡ MODEL EFFICIENCY & RESOURCE CONSUMPTION REPORT")
    print("="*60)
    print(f"📦 Model Name:               {MODEL_NAME}")
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
# 🏁 END OF SCRIPT — LIVE DEMO TEST (PRETRAINED META HTDEMUCS)
# ==============================================================================
