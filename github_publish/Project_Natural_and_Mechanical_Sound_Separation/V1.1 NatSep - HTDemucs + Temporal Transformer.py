# ==============================================================================
# HTDemucs + Temporal Transformer Pipeline (Kaggle T4 x2 Optimized)
# Project: Natural and Mechanical Sound Separation (NatSep)
# ==============================================================================

import os
import sys
import glob
import math
import random
import pathlib
import warnings
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp

# Auto-install demucs if missing (Kaggle environment support)
try:
    import demucs.pretrained
except ImportError:
    print("📦 'demucs' library not found. Auto-installing demucs for Transfer Learning...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "demucs"])
    import demucs.pretrained
    print("✅ 'demucs' successfully installed!")

# Fix Python 3.12 multiprocessing file descriptor sharing in Jupyter notebooks
try:
    mp.set_sharing_strategy('file_system')
except Exception:
    pass

warnings.filterwarnings("ignore")

# Modern PyTorch AMP compatibility
try:
    from torch.amp import autocast, GradScaler
except ImportError:
    from torch.cuda.amp import autocast, GradScaler

# ==========================================
# 🛠️ Step 1: Dataset Pipeline (The Data Strategy)
# ==========================================

POSSIBLE_DATASET_PATHS = [
    "/kaggle/input/datasets/inboxhasibur/netsep-audio-dataset/NetSep Audio Dataset",
    "/kaggle/input/netsep-audio-dataset/NetSep Audio Dataset",
    "/kaggle/input/netsep-audio-dataset",
    "./NetSep Audio Dataset",
    "./data"
]

def find_dataset_root():
    for p in POSSIBLE_DATASET_PATHS:
        if os.path.exists(p):
            print(f"✅ Found dataset root at: {p}")
            return pathlib.Path(p)
    default_p = pathlib.Path(POSSIBLE_DATASET_PATHS[0])
    print(f"⚠️ Dataset path not found directly. Defaulting to: {default_p}")
    return default_p

def get_audio_metadata(file_path):
    """Robust metadata extractor safely handling torchaudio/soundfile/librosa backends."""
    if file_path is None or not os.path.exists(str(file_path)):
        return 0, 44100
        
    # Method 1: torchaudio.info if available in backend
    try:
        if hasattr(torchaudio, "info"):
            info = torchaudio.info(str(file_path))
            return info.num_frames, info.sample_rate
    except Exception:
        pass

    # Method 2: soundfile info
    try:
        import soundfile as sf
        info = sf.info(str(file_path))
        return int(info.frames), int(info.samplerate)
    except Exception:
        pass

    # Method 3: librosa samplerate & duration
    try:
        import librosa
        sr = librosa.get_samplerate(str(file_path))
        dur = librosa.get_duration(path=str(file_path))
        return int(dur * sr), int(sr)
    except Exception:
        pass

    # Method 4: torchaudio.load full fallback
    try:
        wav, sr = torchaudio.load(str(file_path))
        return wav.shape[-1], sr
    except Exception:
        return 0, 44100

def safe_load_audio_chunk(file_path, start_frame, num_frames, target_sr=44100):
    """Bulletproof audio chunk loader supporting m4a, wav, mp3 across all backends."""
    if file_path is None or not os.path.exists(str(file_path)):
        return torch.zeros((2, num_frames))

    wav = None
    loaded_sr = target_sr

    # Method 1: torchaudio.load
    try:
        wav, loaded_sr = torchaudio.load(str(file_path), frame_offset=start_frame, num_frames=num_frames)
    except Exception:
        pass

    # Method 2: soundfile
    if wav is None:
        try:
            import soundfile as sf
            data, loaded_sr = sf.read(str(file_path), start=start_frame, frames=num_frames, dtype='float32')
            if data.ndim == 1:
                data = data[np.newaxis, :]
            else:
                data = data.T
            wav = torch.from_numpy(data)
        except Exception:
            pass

    # Method 3: librosa
    if wav is None:
        try:
            import librosa
            offset_sec = start_frame / target_sr
            duration_sec = num_frames / target_sr
            data, loaded_sr = librosa.load(str(file_path), sr=target_sr, offset=offset_sec, duration=duration_sec, mono=False)
            if data.ndim == 1:
                data = data[np.newaxis, :]
            wav = torch.from_numpy(data)
        except Exception:
            return torch.zeros((2, num_frames))

    # Resample if needed
    if loaded_sr != target_sr:
        try:
            wav = T.Resample(loaded_sr, target_sr)(wav)
        except Exception:
            pass

    # Enforce Stereo (2 channels)
    if wav.shape[0] == 1:
        wav = wav.repeat(2, 1)
    elif wav.shape[0] > 2:
        wav = wav[:2, :]

    # Pad or Crop to exact num_frames
    if wav.shape[-1] < num_frames:
        wav = F.pad(wav, (0, num_frames - wav.shape[-1]))
    else:
        wav = wav[:, :num_frames]

    return wav

class AudioAugmentor:
    """
    Beatbox Solution: Apply Pitch Shifting & Time Stretching variations
    to augment beatbox segments and mix them into input audio files.
    """
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate

    def apply_pitch_shift(self, wav, n_steps):
        try:
            return T.PitchShift(self.sample_rate, n_steps=n_steps)(wav)
        except Exception:
            return wav

    def apply_time_stretch(self, wav, rate):
        orig_len = wav.shape[-1]
        new_sr = int(self.sample_rate * rate)
        try:
            resampler = T.Resample(self.sample_rate, new_sr)
            stretched = resampler(wav)
            if stretched.shape[-1] > orig_len:
                stretched = stretched[..., :orig_len]
            else:
                stretched = F.pad(stretched, (0, orig_len - stretched.shape[-1]))
            return stretched
        except Exception:
            return wav

    def generate_variations(self, beatbox_wav):
        """Generates 4-5 variations using Pitch Shift & Time Stretch."""
        variations = []
        variations.append(self.apply_pitch_shift(beatbox_wav, n_steps=2))   # Variation 1
        variations.append(self.apply_pitch_shift(beatbox_wav, n_steps=-3))  # Variation 2
        variations.append(self.apply_time_stretch(beatbox_wav, rate=0.85))  # Variation 3
        variations.append(self.apply_time_stretch(beatbox_wav, rate=1.15))  # Variation 4
        v5 = self.apply_pitch_shift(beatbox_wav, n_steps=4)
        v5 = self.apply_time_stretch(v5, rate=0.90)
        variations.append(v5)
        return variations

class NatSepDataset(Dataset):
    """
    Dataset loader supporting dynamic target creation (vocals + naturals),
    beatbox augmentation, and dynamic 6-second chunking.
    """
    def __init__(self, root_dir=None, chunk_duration=6.0, sample_rate=44100, is_train=True):
        self.root_dir = root_dir if root_dir else find_dataset_root()
        self.chunk_duration = chunk_duration
        self.sample_rate = sample_rate
        self.chunk_samples = int(chunk_duration * sample_rate)
        self.is_train = is_train
        self.augmentor = AudioAugmentor(sample_rate)

        # Locate subdirectories
        self.input_dir = self._get_subfolder(["vocal+natural+instrumental", "vocal+instrumental"])
        self.vocals_dir = self._get_subfolder(["vocals"])
        self.naturals_dir = self._get_subfolder(["naturals"])
        self.instrumentals_dir = self._get_subfolder(["instrumentals"])

        # Gather audio files from input directory
        self.input_files = []
        if self.input_dir and self.input_dir.exists():
            for ext in ["*.m4a", "*.wav", "*.flac", "*.mp3"]:
                self.input_files.extend(list(self.input_dir.glob(ext)))
                self.input_files.extend(list(self.input_dir.glob(f"**/{ext}")))

        self.input_files = sorted(list(set(self.input_files)))
        print(f"📊 Dataset Loaded: Found {len(self.input_files)} input track(s).")

        # Find beatbox track for beatbox solution
        self.beatbox_file = None
        for f in self.input_files:
            if "beatbox" in f.name.lower():
                self.beatbox_file = f
                break
                
        # Pre-compute beatbox variations ONCE to prevent CPU bottleneck and feed GPU faster!
        self.beatbox_variations = []
        if self.beatbox_file and self.is_train:
            print("⏳ Pre-computing beatbox variations (Pitch Shift) to save CPU & boost GPU...")
            beatbox_chunk = safe_load_audio_chunk(self.beatbox_file, 0, self.chunk_samples, self.sample_rate)
            self.beatbox_variations = self.augmentor.generate_variations(beatbox_chunk)
            print("✅ Beatbox variations pre-computed!")

    def _get_subfolder(self, folder_names):
        for name in folder_names:
            sub = self.root_dir / name
            if sub.exists():
                return sub
            matches = list(self.root_dir.glob(f"**/{name}"))
            if matches:
                return matches[0]
        return None

    def _find_matching_file(self, folder, track_name):
        if not folder or not folder.exists():
            return None
        prefix = track_name.split()[0] if " " in track_name else track_name.split("-")[0]
        for f in folder.glob("*"):
            if track_name.lower() in f.name.lower() or (prefix and prefix in f.name):
                return f
        return None

    def __len__(self):
        return max(len(self.input_files), 1)

    def __getitem__(self, idx):
        if len(self.input_files) == 0:
            synth_in = torch.randn(2, self.chunk_samples)
            synth_tgt = torch.randn(2, self.chunk_samples)
            return synth_in, synth_tgt

        in_file = self.input_files[idx % len(self.input_files)]
        track_name = in_file.name

        # Dynamic 6-second Chunking offset
        total_frames, _ = get_audio_metadata(in_file)
        max_start = max(0, total_frames - self.chunk_samples)
        start_frame = random.randint(0, max_start) if self.is_train and max_start > 0 else 0

        # 1. Load Input (vocal+natural+instrumental)
        in_wav = safe_load_audio_chunk(in_file, start_frame, self.chunk_samples, self.sample_rate)

        # 2. Dynamic Target Creation: Combine Vocals + Naturals
        vocal_file = self._find_matching_file(self.vocals_dir, track_name)
        natural_file = self._find_matching_file(self.naturals_dir, track_name)

        vocal_wav = safe_load_audio_chunk(vocal_file, start_frame, self.chunk_samples, self.sample_rate)
        natural_wav = safe_load_audio_chunk(natural_file, start_frame, self.chunk_samples, self.sample_rate)

        # Ground-truth Target = vocals + naturals
        tgt_wav = vocal_wav + natural_wav

        # Normalize target range
        max_val = torch.max(torch.abs(tgt_wav))
        if max_val > 1.0:
            tgt_wav = tgt_wav / max_val

        # 3. Beatbox Augmentation Solution (Randomly mix pre-computed beatbox variations)
        if self.is_train and random.random() < 0.4 and self.beatbox_variations:
            chosen_var = random.choice(self.beatbox_variations)
            gain = random.uniform(0.2, 0.5)
            in_wav = in_wav + gain * chosen_var

        # Normalize input range
        in_max = torch.max(torch.abs(in_wav))
        if in_max > 1.0:
            in_wav = in_wav / in_max

        return in_wav, tgt_wav


# ==========================================
# 🧠 Step 2: Model Architecture (Hybrid Transfer Learning)
# ==========================================

class TemporalRhythmTransformer(nn.Module):
    """
    Novelty Layer: Temporal Rhythm Transformer (TRT)
    Scans the time-axis of spectrogram features, detects cyclic/repetitive
    patterns (e.g., beatbox), and outputs a modulation map.
    """
    def __init__(self, d_model=8, nhead=4, num_layers=2, dropout=0.15):
        super().__init__()
        # PyTorch TransformerEncoder expects shape (Batch, SeqLen, Features) if batch_first=True
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        # x is [B, C, F, T] where F is 1 (frequency pooled)
        B, C, F_dim, T_dim = x.shape
        # Permute to [B, T_dim, C * F_dim]
        x_reshaped = x.permute(0, 3, 1, 2).contiguous().view(B, T_dim, C * F_dim)
        
        # Transformer processes the sequence over time
        out = self.transformer(x_reshaped)
        
        # Reshape back to [B, C, F_dim, T_dim]
        out = out.view(B, T_dim, C, F_dim).permute(0, 2, 3, 1)
        return out

class TrueHybridHTDemucsTRT(nn.Module):
    """
    True Transfer Learning Architecture:
    1. Official Meta HTDemucs (Frozen): Extracts 4 highly-accurate stems (Vocals, Drums, Bass, Other).
    2. Temporal Rhythm Transformer (Trainable): Analyzes the stems over time to detect rhythmic artifacts (Beatboxing) vs Sustained sounds (Halal Vocals, Nature).
    3. Custom Mixer (Trainable): Combines the TRT analysis into a final mask to extract the target (Vocal + Natural).
    """
    def __init__(self, in_channels=2, n_fft=2048, hop_length=512):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length

        print("⏳ Downloading/Loading Official Meta HTDemucs Weights...")
        # Load official Demucs model and freeze it
        import demucs.pretrained
        self.demucs = demucs.pretrained.get_model('htdemucs')
        for param in self.demucs.parameters():
            param.requires_grad = False
        print("✅ Official HTDemucs loaded and Frozen!")

        # 4 stems * 2 channels = 8 channels
        self.trt = TemporalRhythmTransformer(d_model=8, nhead=4, num_layers=2, dropout=0.15)

        # Trainable Mixer to create the final 2-channel output mask
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
        # 1. Official Demucs Feature Extraction (Frozen)
        with torch.no_grad():
            from demucs.apply import apply_model
            # Returns [B, 4, 2, L] (4 stems: drums, bass, other, vocals)
            # We use apply_model as required by Demucs API
            demucs_stems = apply_model(self.demucs, x) 
            
        B, S, C, L = demucs_stems.shape
        
        # Merge stems and channels: [B, 8, L]
        stems_merged = demucs_stems.reshape(B, S * C, L)
        
        # 2. Get STFT of the separated stems
        stems_mag, _, _ = self._stft(stems_merged) # [B, 8, F, T]
        
        # 3. TRT Analysis (Trainable)
        # Pool over frequency to analyze rhythm over time
        stems_time = stems_mag.mean(dim=2, keepdim=True) # [B, 8, 1, T]
        trt_out = self.trt(stems_time) # [B, 8, 1, T]
        
        # Broadcast TRT output across frequency
        trt_features = stems_mag * trt_out # [B, 8, F, T]
        
        # 4. Custom Mixer (Trainable)
        # Predict a mask for the ORIGINAL input spectrogram
        mask = self.mixer(trt_features) # [B, 2, F, T]
        
        # 5. Apply Mask to Original Input and Reconstruct
        input_mag, input_phase, _ = self._stft(x)
        pred_mag = input_mag * mask
        pred_wav = self._istft(pred_mag, input_phase, L)
        
        return pred_wav


# ==========================================
# 📊 Step 4: Performance Metrics & Losses
# ==========================================

def multi_resolution_stft_loss(pred_wav, tgt_wav):
    """Multi-Resolution STFT Loss for fast frequency-domain evaluation."""
    loss = 0.0
    fft_sizes = [512, 1024, 2048]
    hop_sizes = [128, 256, 512]
    win_sizes = [512, 1024, 2048]

    B, C, L = pred_wav.shape
    pred_flat = pred_wav.reshape(B * C, L)
    tgt_flat = tgt_wav.reshape(B * C, L)

    for n_fft, hop, win_size in zip(fft_sizes, hop_sizes, win_sizes):
        window = torch.hann_window(win_size, device=pred_wav.device)
        p_stft = torch.stft(pred_flat, n_fft=n_fft, hop_length=hop, win_length=win_size, window=window, return_complex=True)
        t_stft = torch.stft(tgt_flat, n_fft=n_fft, hop_length=hop, win_length=win_size, window=window, return_complex=True)

        p_mag = torch.abs(p_stft)
        t_mag = torch.abs(t_stft)

        sc_loss = torch.norm(t_mag - p_mag, p="fro") / (torch.norm(t_mag, p="fro") + 1e-8)
        mag_loss = F.l1_loss(torch.log(p_mag + 1e-8), torch.log(t_mag + 1e-8))
        loss += (sc_loss + mag_loss)

    return loss / len(fft_sizes)

def compute_si_sdr_tensor(pred, target, eps=1e-8):
    """Fast Scale-Invariant SDR calculation for epoch validation."""
    target_clean = target - torch.mean(target, dim=-1, keepdim=True)
    pred_clean = pred - torch.mean(pred, dim=-1, keepdim=True)

    dot = torch.sum(pred_clean * target_clean, dim=-1, keepdim=True)
    s_target = (dot * target_clean) / (torch.sum(target_clean ** 2, dim=-1, keepdim=True) + eps)
    e_noise = pred_clean - s_target

    sdr = 10 * torch.log10(
        (torch.sum(s_target ** 2, dim=-1) + eps) / (torch.sum(e_noise ** 2, dim=-1) + eps)
    )
    return sdr.mean()


# ==========================================
# ⚙️ Step 3: DDP & Training Execution Loop
# ==========================================

def train_one_epoch(model, dataloader, optimizer, scaler, device, rank):
    model.train()
    total_loss = 0.0

    for batch_idx, (in_wav, tgt_wav) in enumerate(dataloader):
        in_wav, tgt_wav = in_wav.to(device), tgt_wav.to(device)
        optimizer.zero_grad()

        with autocast(device_type='cuda', dtype=torch.float16):
            pred_wav = model(in_wav)
            l1_loss = F.l1_loss(pred_wav, tgt_wav)
            stft_loss = multi_resolution_stft_loss(pred_wav, tgt_wav)
            loss = l1_loss + stft_loss

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    return total_loss / max(len(dataloader), 1)

def validate(model, dataloader, device):
    model.eval()
    total_si_sdr = 0.0

    with torch.no_grad():
        for in_wav, tgt_wav in dataloader:
            in_wav, tgt_wav = in_wav.to(device), tgt_wav.to(device)
            with autocast(device_type='cuda', dtype=torch.float16):
                pred_wav = model(in_wav)
            total_si_sdr += compute_si_sdr_tensor(pred_wav, tgt_wav).item()

    return total_si_sdr / max(len(dataloader), 1)


def main():
    is_ddp = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if is_ddp:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        local_rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🖥️ Single device mode: {device}")

    dataset_root = find_dataset_root()
    train_dataset = NatSepDataset(root_dir=dataset_root, chunk_duration=6.0, is_train=True)
    val_dataset = NatSepDataset(root_dir=dataset_root, chunk_duration=6.0, is_train=False)

    if is_ddp:
        num_workers = 2
        train_sampler = DistributedSampler(train_dataset)
        train_loader = DataLoader(train_dataset, batch_size=8, sampler=train_sampler, num_workers=num_workers, pin_memory=True, persistent_workers=True)
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=True)
    else:
        train_sampler = None
        # In single-device notebook mode, num_workers=0 prevents Python 3.12 multiprocessing child process assertions
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0, pin_memory=True)

    model = TrueHybridHTDemucsTRT().to(device)

    # ---------------------------------------------------------
    # 🔥 TRUE TRANSFER LEARNING 
    # ---------------------------------------------------------
    # The official HTDemucs weights are already loaded and frozen inside `TrueHybridHTDemucsTRT`!
    # We only train the TRT and Custom Mixer.
    print("✅ Ready for training! Only TRT and Mixer parameters will be trained.")

    if is_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=1e-4)
    scaler = GradScaler(enabled=torch.cuda.is_available())

    best_si_sdr = -float('inf')
    patience = 30
    patience_counter = 0
    save_path = "best_model.pth"

    epochs = 500
    print(f"🚀 Starting Training Loop on GPU rank {local_rank}...")

    for epoch in range(epochs):
        if is_ddp and train_sampler:
            train_sampler.set_epoch(epoch)

        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device, local_rank)

        if local_rank == 0:
            val_si_sdr = validate(model, val_loader, device)
            print(f"Epoch [{epoch+1:03d}/{epochs:03d}] | Train Loss: {train_loss:.4f} | Val SI-SDR: {val_si_sdr:.4f} dB")

            if val_si_sdr > best_si_sdr:
                best_si_sdr = val_si_sdr
                patience_counter = 0
                state_dict = model.module.state_dict() if is_ddp else model.state_dict()
                torch.save(state_dict, save_path)
                print(f"  💾 Best model saved with SI-SDR: {best_si_sdr:.4f} dB")
            else:
                patience_counter += 1
                print(f"  ⏳ Patience counter: {patience_counter}/{patience}")

            if patience_counter >= patience:
                print(f"🛑 Early Stopping Triggered! Validation SI-SDR did not improve for {patience} epochs.")
                break

    if is_ddp:
        dist.destroy_process_group()

    if local_rank == 0 and os.path.exists(save_path):
        print("🎨 Generating Post-Training XAI Attention Map...")
        plot_xai_attention_map(model, val_loader, device)

def plot_xai_attention_map(model, dataloader, device):
    model.eval()
    with torch.no_grad():
        for in_wav, _ in dataloader:
            in_wav = in_wav.to(device)
            raw_model = model.module if hasattr(model, 'module') else model
            from demucs.apply import apply_model
            demucs_stems = apply_model(raw_model.demucs, in_wav)
            B, S, C, L = demucs_stems.shape
            stems_merged = demucs_stems.reshape(B, S * C, L)
            stems_mag, _, _ = raw_model._stft(stems_merged)
            
            stems_time = stems_mag.mean(dim=2, keepdim=True)
            trt_map = raw_model.trt(stems_time)[0].cpu().squeeze().numpy()
            
            plt.figure(figsize=(10, 4))
            plt.imshow(trt_map, aspect='auto', cmap='magma', origin='lower')
            plt.colorbar(label='Attention Weight')
            plt.title('Temporal Rhythm Transformer (XAI) Attention Map - Beatbox/Music Masking')
            plt.xlabel('Time Frames')
            plt.ylabel('Feature Channels')
            plt.tight_layout()
            plt.savefig('xai_attention_map.png', dpi=150)
            plt.close()
            print("✅ XAI Attention Map saved to 'xai_attention_map.png'")
            break

if __name__ == "__main__":
    main()