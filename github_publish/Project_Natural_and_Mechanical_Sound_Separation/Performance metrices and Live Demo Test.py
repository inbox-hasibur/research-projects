# ============================================================
# TahirNet — Complete Evaluation + Demo Pipeline
# Fix: autocast device_type | Performance Metrics | 2x Demo
# ============================================================

import os, sys, time, json, warnings
warnings.filterwarnings('ignore')

# Install required packages
import subprocess
for pkg in ["gradio", "pesq", "pystoi", "librosa", "soundfile", "nest_asyncio"]:
    subprocess.check_call([sys.executable, "-m", "pip", 
                          "install", "-q", pkg])

import nest_asyncio
import asyncio
nest_asyncio.apply()

# Fix for Python 3.12 + Uvicorn + nest_asyncio bug where loop_factory is passed
original_run = asyncio.run
def patched_run(*args, loop_factory=None, **kwargs):
    return original_run(*args, **kwargs)
asyncio.run = patched_run

import sys
if "uvicorn.server" in sys.modules:
    import uvicorn.server
    uvicorn.server.asyncio_run = patched_run

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from tqdm import tqdm
import soundfile as sf
import librosa
import librosa.display

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as Tr
from torch.amp import autocast          # ← Fixed import

from pesq import pesq
from pystoi import stoi

# ─── Device Setup ───────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_GPUS = torch.cuda.device_count()
print(f"✅ Device: {DEVICE} | GPUs: {N_GPUS}")

# ─── Paths ──────────────────────────────────────────────────
MODEL_PATH  = Path("/kaggle/input/models/inboxhasibur/"
                   "tahirnet2-pruned/pytorch/default/1/"
                   "tahirnet_pruned")
DATA_DIR    = Path("/kaggle/input/datasets/inboxhasibur/"
                   "mecnature-audio-dataset")
OUTPUT_DIR  = Path("/kaggle/working/tahirnet_eval")
DEMO_DIR    = OUTPUT_DIR / "demo_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DEMO_DIR.mkdir(parents=True, exist_ok=True)

# ─── Config (must match training exactly) ───────────────────
CONFIG = {
    "sr"              : 22050,
    "clip_length"     : 4,
    "n_fft"           : 1024,
    "hop_length"      : 256,
    "band_boundaries" : [0, 300, 1000, 4000, 8000, 11025],
    "hidden_dim"      : 192,
    "num_rnn_layers"  : 4,
    "dropout"         : 0.1,
    "batch_size"      : 8,
}
SAMPLES = CONFIG["sr"] * CONFIG["clip_length"]

print(f"✅ Config loaded | SR: {CONFIG['sr']} | "
      f"Clip: {CONFIG['clip_length']}s")


# ══════════════════════════════════════════════════════════════
# 1. MODEL DEFINITION (identical to training)
# ══════════════════════════════════════════════════════════════

def _band_slices(boundaries, n_fft, sr):
    freq_res = sr / n_fft
    slices   = []
    for lo, hi in zip(boundaries[:-1], boundaries[1:]):
        lo_bin = int(lo / freq_res)
        hi_bin = min(int(hi / freq_res), n_fft // 2 + 1)
        if hi_bin > lo_bin:
            slices.append(slice(lo_bin, hi_bin))
    return slices


class BandSplitProjection(nn.Module):
    def __init__(self, band_bins, hidden):
        super().__init__()
        self.projs = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(b * 2),
                nn.Linear(b * 2, hidden),
                nn.GELU(),
            ) for b in band_bins
        ])

    def forward(self, S_ri, slices):
        feats = []
        for proj, sl in zip(self.projs, slices):
            band = S_ri[:, sl, :, :]
            B, bins, T, _ = band.shape
            flat = band.permute(0, 2, 1, 3).reshape(B, T, bins * 2)
            feats.append(proj(flat))
        return torch.stack(feats, dim=1)


class IntraBandGRU(nn.Module):
    def __init__(self, hidden, layers, dropout):
        super().__init__()
        self.gru  = nn.GRU(hidden, hidden // 2,
                           num_layers=layers,
                           batch_first=True,
                           bidirectional=True,
                           dropout=dropout if layers > 1 else 0)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x):
        B, K, T, H = x.shape
        x_flat = x.reshape(B * K, T, H)
        out, _ = self.gru(x_flat)
        out    = out.reshape(B, K, T, H)
        return self.norm(x + out)


class InterBandAttention(nn.Module):
    def __init__(self, hidden, num_heads=4, dropout=0.1):
        super().__init__()
        self.attn  = nn.MultiheadAttention(hidden, num_heads,
                                            dropout=dropout,
                                            batch_first=True)
        self.norm  = nn.LayerNorm(hidden)
        self.ff    = nn.Sequential(
            nn.Linear(hidden, hidden * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden),
        )
        self.norm2 = nn.LayerNorm(hidden)

    def forward(self, x):
        B, K, T, H = x.shape
        x_t = x.permute(0, 2, 1, 3).reshape(B * T, K, H)
        attn_out, _ = self.attn(x_t, x_t, x_t)
        x_t = self.norm(x_t + attn_out)
        x_t = self.norm2(x_t + self.ff(x_t))
        return x_t.reshape(B, T, K, H).permute(0, 2, 1, 3)


class BandMaskDecoder(nn.Module):
    def __init__(self, band_bins, hidden):
        super().__init__()
        self.decoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden, b * 2),
                nn.Tanh(),
            ) for b in band_bins
        ])

    def forward(self, x, slices):
        masks = []
        for i, (dec, sl) in enumerate(zip(self.decoders, slices)):
            feat = x[:, i, :, :]
            m    = dec(feat)
            B, T, _ = m.shape
            bins = sl.stop - sl.start
            m    = m.reshape(B, T, bins, 2).permute(0, 2, 1, 3)
            masks.append(m)
        return masks


class TahirNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.sr        = CONFIG["sr"]
        self.n_fft     = CONFIG["n_fft"]
        self.hop       = CONFIG["hop_length"]
        self.hidden    = CONFIG["hidden_dim"]
        self.slices    = _band_slices(
            CONFIG["band_boundaries"], self.n_fft, self.sr)
        self.band_bins = [sl.stop - sl.start for sl in self.slices]
        K              = len(self.slices)

        self.band_proj = BandSplitProjection(
            self.band_bins, self.hidden)

        n_layers = CONFIG["num_rnn_layers"]
        drop     = CONFIG["dropout"]
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                "intra": IntraBandGRU(self.hidden, 2, drop),
                "inter": InterBandAttention(
                    self.hidden, num_heads=4, dropout=drop),
            }) for _ in range(n_layers)
        ])

        self.mask_dec     = BandMaskDecoder(
            self.band_bins, self.hidden)
        self.band_weights = nn.Parameter(torch.ones(K))

    def _stft(self, x):
        win = torch.hann_window(self.n_fft).to(x.device)
        return torch.stft(x, n_fft=self.n_fft,
                         hop_length=self.hop,
                         window=win, return_complex=True)

    def _istft(self, S, length):
        win = torch.hann_window(self.n_fft).to(S.device)
        return torch.istft(S, n_fft=self.n_fft,
                          hop_length=self.hop,
                          window=win, length=length)

    def forward(self, mixture):
        length = mixture.shape[-1]
        S      = self._stft(mixture)
        S_ri   = torch.stack([S.real, S.imag], dim=-1)
        x      = self.band_proj(S_ri, self.slices)

        for block in self.blocks:
            x = block["intra"](x)
            x = block["inter"](x)

        bw  = torch.softmax(self.band_weights, dim=0)
        x   = x * bw.view(1, -1, 1, 1)

        masks = self.mask_dec(x, self.slices)
        S_out = torch.zeros_like(S_ri)
        for mask, sl in zip(masks, self.slices):
            S_out[:, sl, :, :] = S_ri[:, sl, :, :] * mask

        S_sep = torch.complex(S_out[..., 0], S_out[..., 1])
        return self._istft(S_sep, length=length)


# ══════════════════════════════════════════════════════════════
# 2. MODEL LOADER
# ══════════════════════════════════════════════════════════════

def load_model(model_path: Path) -> nn.Module:
    """
    Load TahirNet from saved checkpoint.
    Handles both DataParallel and single-GPU saves.
    """
    print(f"\n{'='*60}")
    print(f"📦 Loading TahirNet from:")
    print(f"   {model_path}")
    
    model = TahirNet().to(DEVICE)
    
    # Handle different path types
    # Type 1: Direct .pt file
    # Type 2: PyTorch format folder (your case)
    
    pt_candidates = [
        model_path,
        model_path / "tahirnet_best.pt",
        model_path.parent / "tahirnet_best.pt",
        model_path / "tahirnet_pruned.pt",
        model_path / "tahirnet2_pruned.pt",
    ]
    
    loaded = False
    for candidate in pt_candidates:
        if Path(str(candidate) + ".pt").exists():
            candidate = Path(str(candidate) + ".pt")
        
        if Path(candidate).exists():
            try:
                state = torch.load(candidate, 
                                   map_location=DEVICE,
                                   weights_only=False)
                
                # Handle various save formats
                if isinstance(state, dict):
                    # Case 1: Full checkpoint dict
                    if 'model_state' in state:
                        state = state['model_state']
                    # Case 2: Already state_dict
                    # (nothing to do)
                
                # Remove DataParallel prefix if present
                new_state = {}
                for k, v in state.items():
                    new_key = k.replace('module.', '')
                    new_state[new_key] = v
                
                model.load_state_dict(new_state, strict=False)
                loaded = True
                print(f"   ✅ Loaded from: {candidate}")
                break
            except Exception as e:
                print(f"   ⚠️  Failed {candidate}: {e}")
                continue
    
    if not loaded:
        # Handle Kaggle Model Hub automatically unpacking .pt files into directories
        if Path(model_path).is_dir() and (Path(model_path) / "data.pkl").exists():
            print(f"   📦 Detected unpacked PyTorch checkpoint. Re-zipping on the fly...")
            import zipfile
            import os
            import tempfile
            
            # PyTorch expects all files in the zip to be under a single top-level directory (e.g., 'archive/')
            tmp_zip = Path(tempfile.gettempdir()) / "temp_tahirnet_ckpt.zip"
            with zipfile.ZipFile(tmp_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
                for root_dir, _, files in os.walk(model_path):
                    for file in files:
                        full_path = os.path.join(root_dir, file)
                        rel_path  = os.path.relpath(full_path, model_path)
                        zip_path  = os.path.join("archive", rel_path)
                        zf.write(full_path, zip_path)
            
            try:
                state = torch.load(str(tmp_zip), map_location=DEVICE, weights_only=False)
                if isinstance(state, dict):
                    if 'model_state' in state:
                        state = state['model_state']
                
                # Remove DataParallel prefix if present
                new_state = {}
                for k, v in state.items():
                    new_key = k.replace('module.', '')
                    new_state[new_key] = v
                
                model.load_state_dict(new_state, strict=False)
                loaded = True
                print(f"   ✅ Successfully loaded unpacked model via temp zip!")
            except Exception as e:
                print(f"   ⚠️  Failed loading re-zipped model: {e}")
    
    if not loaded:
        print("   ⚠️  Could not load weights — using random init")
        print("   📌 Tip: Make sure model path is correct")
    
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {n_params:,}")
    print(f"{'='*60}\n")
    return model


# ══════════════════════════════════════════════════════════════
# 3. METRICS SUITE
# ══════════════════════════════════════════════════════════════

def compute_si_sdr(est: np.ndarray, 
                   tgt: np.ndarray, 
                   eps: float = 1e-8) -> float:
    """Scale-Invariant SDR."""
    tgt = tgt - tgt.mean()
    est = est - est.mean()
    dot = np.dot(est, tgt)
    s_t = dot * tgt / (np.dot(tgt, tgt) + eps)
    e_n = est - s_t
    return float(10 * np.log10(
        np.dot(s_t, s_t) / (np.dot(e_n, e_n) + eps) + eps))


def compute_sdr(est: np.ndarray, 
                tgt: np.ndarray, 
                eps: float = 1e-8) -> float:
    """Signal-to-Distortion Ratio."""
    noise = est - tgt
    return float(10 * np.log10(
        np.dot(tgt, tgt) / (np.dot(noise, noise) + eps) + eps))


def compute_snr(est: np.ndarray,
                tgt: np.ndarray,
                eps: float = 1e-8) -> float:
    """Signal-to-Noise Ratio."""
    noise = tgt - est
    return float(10 * np.log10(
        np.dot(tgt, tgt) / (np.dot(noise, noise) + eps) + eps))


def compute_pesq_score(est: np.ndarray,
                        tgt: np.ndarray,
                        sr: int) -> float:
    """
    PESQ (Perceptual Evaluation of Speech Quality).
    Requires 8000 or 16000 Hz.
    """
    try:
        # Resample to 16000 for PESQ
        if sr != 16000:
            tgt_r = librosa.resample(tgt, orig_sr=sr, target_sr=16000)
            est_r = librosa.resample(est, orig_sr=sr, target_sr=16000)
        else:
            tgt_r, est_r = tgt, est
        
        # Normalize
        tgt_r = tgt_r / (np.abs(tgt_r).max() + 1e-8)
        est_r = est_r / (np.abs(est_r).max() + 1e-8)
        
        score = pesq(16000, tgt_r.astype(np.float32),
                            est_r.astype(np.float32), 'wb')
        return float(score)
    except Exception:
        return float('nan')


def compute_stoi_score(est: np.ndarray,
                        tgt: np.ndarray,
                        sr: int) -> float:
    """
    STOI (Short-Time Objective Intelligibility).
    Score range: [0, 1] — higher is better.
    """
    try:
        if sr != 10000:
            tgt_r = librosa.resample(tgt, orig_sr=sr, target_sr=10000)
            est_r = librosa.resample(est, orig_sr=sr, target_sr=10000)
        else:
            tgt_r, est_r = tgt, est
        
        length = min(len(tgt_r), len(est_r))
        score = stoi(tgt_r[:length], est_r[:length], 10000, extended=False)
        return float(score)
    except Exception:
        return float('nan')


def compute_all_metrics(est: np.ndarray,
                        tgt: np.ndarray,
                        mix: np.ndarray,
                        sr: int) -> dict:
    """Compute full metrics suite."""
    length = min(len(est), len(tgt), len(mix))
    est = est[:length]
    tgt = tgt[:length]
    mix = mix[:length]
    
    # Improvement metrics (model vs raw mixture)
    si_sdr_imp = compute_si_sdr(est, tgt) - compute_si_sdr(mix, tgt)
    sdr_imp    = compute_sdr(est, tgt) - compute_sdr(mix, tgt)
    
    return {
        # Absolute metrics
        'SI-SDR (dB)'       : compute_si_sdr(est, tgt),
        'SDR (dB)'          : compute_sdr(est, tgt),
        'SNR (dB)'          : compute_snr(est, tgt),
        'PESQ'              : compute_pesq_score(est, tgt, sr),
        'STOI'              : compute_stoi_score(est, tgt, sr),
        
        # Improvement over mixture
        'SI-SDRi (dB)'      : si_sdr_imp,
        'SDRi (dB)'         : sdr_imp,
        
        # Mixture baseline
        'Mix SI-SDR (dB)'   : compute_si_sdr(mix, tgt),
        'Mix SDR (dB)'      : compute_sdr(mix, tgt),
    }


# ══════════════════════════════════════════════════════════════
# 4. TEST DATASET + EVALUATION LOOP
# ══════════════════════════════════════════════════════════════

class TestDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir: Path, split: str = "test"):
        self.sr = CONFIG["sr"]
        base = data_dir / split
        
        self.mix_files   = []
        self.nat_files   = []
        self.interf_files = []
        
        # Try standard folder structure
        if (base / "mixture").exists():
            self.mix_files   = sorted((base/"mixture").glob("*.wav"))
            self.nat_files   = sorted((base/"natural").glob("*.wav"))
            self.interf_files= sorted((base/"interference").glob("*.wav"))
        else:
            # Fallback: search recursively
            self.mix_files = sorted(base.rglob("mix*.wav"))
            self.nat_files = sorted(base.rglob("nat*.wav"))
            self.interf_files = sorted(base.rglob("int*.wav"))
        
        print(f"  [test] Found {len(self.mix_files)} samples")
        
        if len(self.mix_files) == 0:
            print("  ⚠️  No test files found — creating synthetic test")
            self._synthetic = True
            self._n = 50
        else:
            self._synthetic = False
            self._n = len(self.mix_files)
    
    def _load_and_prep(self, path, fixed_start=0):
        audio, sr = torchaudio.load(str(path))
        if sr != self.sr:
            audio = torchaudio.functional.resample(
                audio, sr, self.sr)
        audio = audio.mean(0)
        if audio.shape[-1] < SAMPLES:
            audio = F.pad(audio, (0, SAMPLES - audio.shape[-1]))
        else:
            audio = audio[fixed_start: fixed_start + SAMPLES]
        mx = audio.abs().max()
        return audio / (mx + 1e-8)
    
    def _make_synthetic(self, idx):
        """Synthetic audio for testing when no dataset available."""
        torch.manual_seed(idx)
        # Natural: sine waves (bird-like harmonics)
        t   = torch.linspace(0, CONFIG['clip_length'],
                             SAMPLES)
        f0  = 440 * (1 + idx % 5 * 0.2)
        nat = (torch.sin(2 * np.pi * f0 * t) * 0.5 +
               torch.sin(2 * np.pi * f0 * 2 * t) * 0.3 +
               torch.sin(2 * np.pi * f0 * 3 * t) * 0.2)
        
        # Interference: noise burst
        interf = torch.randn(SAMPLES) * 0.3
        
        mix = nat + 0.5 * interf
        mx  = mix.abs().max()
        return mix / (mx + 1e-8), nat / (nat.abs().max() + 1e-8)
    
    def __len__(self):
        return self._n
    
    def __getitem__(self, idx):
        if self._synthetic:
            mix, nat = self._make_synthetic(idx)
            return mix, nat, mix.clone()
        
        mix  = self._load_and_prep(self.mix_files[idx])
        nat  = self._load_and_prep(self.nat_files[idx])
        
        # Return mixture for baseline comparison
        return mix, nat, mix.clone()


@torch.no_grad()
def evaluate_full_test(model: nn.Module,
                       data_dir: Path) -> pd.DataFrame:
    """
    Complete test set evaluation.
    Returns DataFrame with per-sample metrics.
    """
    print(f"\n{'='*60}")
    print("🧪 FULL TEST SET EVALUATION")
    print(f"{'='*60}")
    
    test_ds = TestDataset(data_dir, "test")
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    all_metrics = []
    model.eval()
    
    for batch_idx, (mix, tgt, mix_raw) in enumerate(
            tqdm(test_loader, desc="Evaluating")):
        
        mix = mix.to(DEVICE)
        tgt = tgt.to(DEVICE)
        
        # ── Fixed autocast with device_type ──
        with autocast(device_type='cuda',
                      dtype=torch.float16,
                      enabled=torch.cuda.is_available()):
            est = model(mix)
        
        # Move to CPU for metric computation
        est_np = est.cpu().float().numpy()
        tgt_np = tgt.cpu().float().numpy()
        mix_np = mix_raw.cpu().float().numpy()
        
        for i in range(est_np.shape[0]):
            m = compute_all_metrics(
                est_np[i], tgt_np[i], mix_np[i],
                CONFIG["sr"]
            )
            m['sample_idx'] = batch_idx * CONFIG["batch_size"] + i
            all_metrics.append(m)
    
    df = pd.DataFrame(all_metrics)
    
    # ── Print Summary ────────────────────────────────────────
    print(f"\n{'='*60}")
    print("📊 TEST RESULTS SUMMARY")
    print(f"{'='*60}")
    
    key_metrics = [
        'SI-SDR (dB)', 'SDR (dB)', 'SNR (dB)',
        'SI-SDRi (dB)', 'SDRi (dB)',
        'PESQ', 'STOI'
    ]
    
    results_summary = {}
    for metric in key_metrics:
        if metric in df.columns:
            vals = df[metric].dropna()
            mean = vals.mean()
            std  = vals.std()
            results_summary[metric] = {
                'mean': mean, 'std': std,
                'min': vals.min(), 'max': vals.max()
            }
            print(f"  {metric:<20s}: "
                  f"{mean:+7.3f} ± {std:.3f} dB"
                  if 'dB' in metric
                  else f"  {metric:<20s}: "
                       f"{mean:7.4f} ± {std:.4f}")
    
    print(f"\n  Total samples evaluated: {len(df)}")
    print(f"  Baseline Mix SI-SDR: "
          f"{df['Mix SI-SDR (dB)'].mean():+.3f} dB")
    print(f"  Model SI-SDR:        "
          f"{df['SI-SDR (dB)'].mean():+.3f} dB")
    print(f"  Improvement:         "
          f"{df['SI-SDRi (dB)'].mean():+.3f} dB ↑")
    
    # Save
    df.to_csv(OUTPUT_DIR / "test_metrics.csv", index=False)
    
    with open(OUTPUT_DIR / "test_summary.json", 'w') as f:
        json.dump({k: {'mean': v['mean'], 'std': v['std']}
                  for k, v in results_summary.items()}, f, indent=2)
    
    print(f"\n  💾 Saved: {OUTPUT_DIR / 'test_metrics.csv'}")
    
    return df


def plot_metrics(df: pd.DataFrame):
    """Comprehensive metrics visualization."""
    fig = plt.figure(figsize=(20, 14))
    gs  = gridspec.GridSpec(3, 3, figure=fig, 
                            hspace=0.4, wspace=0.35)
    fig.suptitle('TahirNet — Test Evaluation Results',
                 fontsize=16, fontweight='bold')
    
    # 1. SI-SDR Distribution
    ax1 = fig.add_subplot(gs[0, 0])
    vals = df['SI-SDR (dB)'].dropna()
    ax1.hist(vals, bins=30, color='steelblue',
             edgecolor='white', alpha=0.85)
    ax1.axvline(vals.mean(), color='red', 
                linestyle='--', label=f'Mean: {vals.mean():.2f}')
    ax1.set_title('SI-SDR Distribution', fontweight='bold')
    ax1.set_xlabel('SI-SDR (dB)')
    ax1.set_ylabel('Count')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. SDR Distribution
    ax2 = fig.add_subplot(gs[0, 1])
    vals2 = df['SDR (dB)'].dropna()
    ax2.hist(vals2, bins=30, color='coral',
             edgecolor='white', alpha=0.85)
    ax2.axvline(vals2.mean(), color='darkred',
                linestyle='--', label=f'Mean: {vals2.mean():.2f}')
    ax2.set_title('SDR Distribution', fontweight='bold')
    ax2.set_xlabel('SDR (dB)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Improvement (SI-SDRi)
    ax3 = fig.add_subplot(gs[0, 2])
    imp = df['SI-SDRi (dB)'].dropna()
    colors = ['#2ecc71' if v > 0 else '#e74c3c' for v in imp]
    ax3.bar(range(min(50, len(imp))),
            imp.values[:50], color=colors[:50])
    ax3.axhline(0, color='black', linewidth=0.8)
    ax3.axhline(imp.mean(), color='blue', 
                linestyle='--', label=f'Mean: {imp.mean():.2f}')
    ax3.set_title('SI-SDR Improvement (first 50)', fontweight='bold')
    ax3.set_xlabel('Sample Index')
    ax3.set_ylabel('SI-SDRi (dB)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. PESQ Scores
    ax4 = fig.add_subplot(gs[1, 0])
    pesq_vals = df['PESQ'].dropna()
    if len(pesq_vals) > 0:
        ax4.hist(pesq_vals, bins=25, color='mediumpurple',
                 edgecolor='white', alpha=0.85)
        ax4.axvline(pesq_vals.mean(), color='purple',
                    linestyle='--',
                    label=f'Mean: {pesq_vals.mean():.3f}')
        ax4.set_title('PESQ Score Distribution', fontweight='bold')
        ax4.set_xlabel('PESQ Score')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    # 5. STOI Scores
    ax5 = fig.add_subplot(gs[1, 1])
    stoi_vals = df['STOI'].dropna()
    if len(stoi_vals) > 0:
        ax5.hist(stoi_vals, bins=25, color='teal',
                 edgecolor='white', alpha=0.85)
        ax5.axvline(stoi_vals.mean(), color='darkgreen',
                    linestyle='--',
                    label=f'Mean: {stoi_vals.mean():.3f}')
        ax5.set_title('STOI Score Distribution', fontweight='bold')
        ax5.set_xlabel('STOI Score (0-1)')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
    
    # 6. Metrics Radar / Summary Bar
    ax6 = fig.add_subplot(gs[1, 2])
    metrics_to_plot = {
        'SI-SDR': df['SI-SDR (dB)'].mean(),
        'SDR'   : df['SDR (dB)'].mean(),
        'SNR'   : df['SNR (dB)'].mean(),
        'SI-SDRi': df['SI-SDRi (dB)'].mean(),
        'SDRi'  : df['SDRi (dB)'].mean(),
    }
    bar_colors = ['#3498db' if v >= 0 else '#e74c3c'
                  for v in metrics_to_plot.values()]
    bars = ax6.bar(metrics_to_plot.keys(),
                   metrics_to_plot.values(),
                   color=bar_colors, edgecolor='white')
    ax6.axhline(0, color='black', linewidth=0.8)
    ax6.set_title('Mean Metrics Summary', fontweight='bold')
    ax6.set_ylabel('dB')
    for bar, val in zip(bars, metrics_to_plot.values()):
        ax6.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.1,
                 f'{val:.2f}', ha='center',
                 va='bottom', fontsize=9, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    
    # 7. Model vs Mixture comparison
    ax7 = fig.add_subplot(gs[2, :2])
    x = np.arange(min(30, len(df)))
    mix_si = df['Mix SI-SDR (dB)'].values[:30]
    est_si = df['SI-SDR (dB)'].values[:30]
    ax7.plot(x, mix_si, 'o--', color='gray',
             label='Mixture (baseline)', alpha=0.7)
    ax7.plot(x, est_si, 's-', color='steelblue',
             label='TahirNet Output', linewidth=1.5)
    ax7.fill_between(x, mix_si, est_si, alpha=0.2,
                     color='green', label='Improvement')
    ax7.set_title('Per-Sample: TahirNet vs Mixture Baseline',
                  fontweight='bold')
    ax7.set_xlabel('Sample Index')
    ax7.set_ylabel('SI-SDR (dB)')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # 8. Score table
    ax8 = fig.add_subplot(gs[2, 2])
    ax8.axis('off')
    table_data = [
        ['Metric', 'Mean', 'Std'],
        ['SI-SDR (dB)', f"{df['SI-SDR (dB)'].mean():.3f}",
                        f"{df['SI-SDR (dB)'].std():.3f}"],
        ['SDR (dB)',    f"{df['SDR (dB)'].mean():.3f}",
                        f"{df['SDR (dB)'].std():.3f}"],
        ['SI-SDRi (dB)',f"{df['SI-SDRi (dB)'].mean():.3f}",
                        f"{df['SI-SDRi (dB)'].std():.3f}"],
        ['PESQ',        f"{df['PESQ'].mean():.3f}",
                        f"{df['PESQ'].std():.3f}"],
        ['STOI',        f"{df['STOI'].mean():.4f}",
                        f"{df['STOI'].std():.4f}"],
    ]
    table = ax8.table(cellText=table_data[1:],
                      colLabels=table_data[0],
                      cellLoc='center', loc='center',
                      bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor('#2c3e50')
            cell.set_text_props(color='white', fontweight='bold')
        elif r % 2 == 0:
            cell.set_facecolor('#ecf0f1')
    ax8.set_title('Results Table', fontweight='bold', pad=10)
    
    plt.savefig(OUTPUT_DIR / "test_results.png",
                dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✅ Plot saved: {OUTPUT_DIR / 'test_results.png'}")


# ══════════════════════════════════════════════════════════════
# 5. DEMO TYPE 1 — File-Based Test (Audio Path Input)
# ══════════════════════════════════════════════════════════════

def visualize_separation(mix_np, est_np, tgt_np, sr, 
                          save_path: Path, title="Demo"):
    """Generate waveform + spectrogram comparison plot."""
    fig, axes = plt.subplots(3, 2, figsize=(16, 10))
    fig.suptitle(f'TahirNet Separation — {title}',
                 fontsize=14, fontweight='bold')
    
    duration = len(mix_np) / sr
    t = np.linspace(0, duration, len(mix_np))
    
    signals = [
        (mix_np, 'Input Mixture', '#e74c3c'),
        (est_np, 'TahirNet Output (Separated)', '#2ecc71'),
        (tgt_np, 'Ground Truth Target', '#3498db'),
    ]
    
    for row, (sig, label, color) in enumerate(signals):
        # Waveform
        ax = axes[row, 0]
        ax.plot(t, sig, color=color, linewidth=0.5, alpha=0.8)
        ax.set_title(f'{label} — Waveform', fontweight='bold')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.set_xlim([0, duration])
        ax.grid(True, alpha=0.3)
        
        # Spectrogram
        ax = axes[row, 1]
        D = librosa.amplitude_to_db(
            np.abs(librosa.stft(sig.astype(np.float32),
                                n_fft=1024, hop_length=256)),
            ref=np.max
        )
        img = librosa.display.specshow(
            D, sr=sr, hop_length=256,
            x_axis='time', y_axis='hz', ax=ax,
            cmap='magma'
        )
        ax.set_title(f'{label} — Spectrogram', fontweight='bold')
        plt.colorbar(img, ax=ax, format="%+2.0f dB")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches='tight')
    plt.show()
    print(f"   📊 Visualization saved: {save_path}")


def demo_file_test(model: nn.Module,
                   mix_path: str,
                   tgt_path: str = None,
                   output_name: str = "demo_output"):
    """
    DEMO TYPE 1 — File-Based Test
    
    Input:  Path to mixture audio file
    Output: Separated audio saved as .wav
    
    Usage:
        demo_file_test(model, 
                       "/path/to/mixture.wav",
                       "/path/to/target.wav")  # optional
    """
    print(f"\n{'='*60}")
    print("🎵 DEMO TYPE 1 — File-Based Audio Test")
    print(f"{'='*60}")
    
    sr = CONFIG["sr"]
    
    # ── Load mixture ─────────────────────────────────────────
    print(f"\n   📂 Loading: {mix_path}")
    audio, orig_sr = librosa.load(mix_path, sr=None, mono=True)
    
    if orig_sr != sr:
        print(f"   🔄 Resampling {orig_sr} → {sr} Hz")
        audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=sr)
    
    print(f"   Duration: {len(audio)/sr:.2f}s | SR: {sr} Hz")
    
    # ── Process in chunks if longer than clip_length ─────────
    chunk_size = SAMPLES
    overlap    = chunk_size // 4
    
    if len(audio) <= chunk_size:
        audio_pad = np.zeros(chunk_size)
        audio_pad[:len(audio)] = audio
        chunks = [audio_pad]
        original_length = len(audio)
    else:
        # Chunked processing with overlap-add
        chunks = []
        original_length = len(audio)
        for start in range(0, len(audio), chunk_size - overlap):
            end = start + chunk_size
            chunk = audio[start:end]
            if len(chunk) < chunk_size:
                chunk = np.pad(chunk, 
                              (0, chunk_size - len(chunk)))
            chunks.append(chunk)
    
    print(f"   Processing {len(chunks)} chunk(s)...")
    
    # ── Model inference ──────────────────────────────────────
    separated_chunks = []
    model.eval()
    
    with torch.no_grad():
        for chunk in tqdm(chunks, desc="   Separating"):
            x = torch.from_numpy(chunk).float().unsqueeze(0).to(DEVICE)
            
            with autocast(device_type='cuda',
                          dtype=torch.float16,
                          enabled=torch.cuda.is_available()):
                est = model(x)
            
            separated_chunks.append(
                est.squeeze(0).cpu().float().numpy())
    
    # ── Reconstruct full audio ────────────────────────────────
    if len(separated_chunks) == 1:
        separated = separated_chunks[0][:original_length]
    else:
        # Overlap-add reconstruction
        output = np.zeros(original_length + chunk_size)
        count  = np.zeros(original_length + chunk_size)
        pos = 0
        for chunk in separated_chunks:
            end = min(pos + chunk_size, len(output))
            actual = end - pos
            output[pos:end] += chunk[:actual]
            count[pos:end]  += 1
            pos += (chunk_size - overlap)
        
        count = np.maximum(count, 1)
        separated = (output / count)[:original_length]
    
    # Normalize
    mx = np.abs(separated).max()
    if mx > 0:
        separated = separated / mx * 0.95
    
    # ── Save output audio ────────────────────────────────────
    out_path = DEMO_DIR / f"{output_name}.wav"
    sf.write(str(out_path), separated, sr)
    print(f"\n   ✅ Separated audio saved: {out_path}")
    
    # ── Load target for metrics (if provided) ────────────────
    tgt_np = None
    if tgt_path and Path(tgt_path).exists():
        tgt_np, _ = librosa.load(tgt_path, sr=sr, mono=True)
        tgt_np = tgt_np[:original_length]
        
        metrics = compute_all_metrics(
            separated, tgt_np, audio[:original_length], sr)
        
        print(f"\n   📊 Performance Metrics:")
        print(f"   {'─'*40}")
        for k, v in metrics.items():
            if not np.isnan(v):
                unit = ' dB' if 'dB' in k else ''
                print(f"   {k:<22s}: {v:+.4f}{unit}")
    
    # ── Visualization ─────────────────────────────────────────
    mix_np = audio[:original_length]
    est_np = separated
    ref_np = tgt_np if tgt_np is not None else np.zeros_like(est_np)
    
    visualize_separation(
        mix_np, est_np, ref_np, sr,
        save_path=DEMO_DIR / f"{output_name}_plot.png",
        title=Path(mix_path).stem
    )
    
    print(f"\n   📁 Output files:")
    print(f"      Audio : {out_path}")
    print(f"      Plot  : {DEMO_DIR}/{output_name}_plot.png")
    
    return separated, sr


# ══════════════════════════════════════════════════════════════
# 6. DEMO TYPE 2 — Gradio Web UI
# ══════════════════════════════════════════════════════════════

def build_gradio_ui(model: nn.Module):
    """
    DEMO TYPE 2 — Gradio Web Interface
    
    একটি interactive web UI যেখানে:
    - Audio file upload করা যাবে
    - Real-time separation হবে
    - Input + Output audio side-by-side শোনা যাবে
    - Spectrogram comparison দেখা যাবে
    - Metrics download করা যাবে
    
    Usage:
        - Kaggle এ run করলে public URL পাবে
        - সেই URL browser এ open করো
        - Audio upload → Separate বাটন → শোনো!
    """
    import gradio as gr
    import tempfile
    
    sr = CONFIG["sr"]
    
    # ── Core separation function ──────────────────────────────
    def separate_audio(mix_audio, has_target, tgt_audio=None):
        """
        Gradio callback function.
        mix_audio: (sample_rate, numpy_array) tuple from gr.Audio
        """
        if mix_audio is None:
            return None, None, "❌ Please upload a mixture audio file"
        
        # Gradio returns (sr, array)
        in_sr, audio_np = mix_audio
        
        # Convert to float32 [-1, 1]
        if audio_np.dtype == np.int16:
            audio_np = audio_np.astype(np.float32) / 32768.0
        elif audio_np.dtype == np.int32:
            audio_np = audio_np.astype(np.float32) / 2147483648.0
        
        # Stereo → mono
        if audio_np.ndim == 2:
            audio_np = audio_np.mean(axis=1)
        
        # Resample if needed
        if in_sr != sr:
            audio_np = librosa.resample(
                audio_np, orig_sr=in_sr, target_sr=sr)
        
        # Normalize
        mx = np.abs(audio_np).max()
        if mx > 0:
            audio_np = audio_np / mx
        
        # ── Process in chunks with overlap-add ──────────
        chunk_size = SAMPLES
        overlap    = chunk_size // 4
        original_length = len(audio_np)
        
        if original_length <= chunk_size:
            audio_pad = np.zeros(chunk_size)
            audio_pad[:original_length] = audio_np
            chunks = [audio_pad]
        else:
            chunks = []
            for start in range(0, original_length, chunk_size - overlap):
                end = start + chunk_size
                chunk = audio_np[start:end]
                if len(chunk) < chunk_size:
                    chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
                chunks.append(chunk)
                
        separated_chunks = []
        model.eval()
        with torch.no_grad():
            for chunk in chunks:
                x = torch.from_numpy(chunk).float().unsqueeze(0).to(DEVICE)
                with autocast(device_type='cuda', dtype=torch.float16, enabled=torch.cuda.is_available()):
                    est = model(x)
                separated_chunks.append(est.squeeze(0).cpu().float().numpy())
                
        # Reconstruct full audio
        if len(separated_chunks) == 1:
            separated = separated_chunks[0][:original_length]
        else:
            output = np.zeros(original_length + chunk_size)
            count  = np.zeros(original_length + chunk_size)
            pos = 0
            for chunk in separated_chunks:
                end = min(pos + chunk_size, len(output))
                actual = end - pos
                output[pos:end] += chunk[:actual]
                count[pos:end]  += 1
                pos += (chunk_size - overlap)
            count = np.maximum(count, 1)
            separated = (output / count)[:original_length]
            
        # Normalize output
        mx = np.abs(separated).max()
        if mx > 0:
            separated = separated / mx * 0.95
        
        # ── Compute metrics if target provided ───────────────
        metrics_text = "**Metrics** (no target provided)\n"
        
        if has_target and tgt_audio is not None:
            tgt_sr, tgt_np = tgt_audio
            if tgt_np.dtype == np.int16:
                tgt_np = tgt_np.astype(np.float32) / 32768.0
            if tgt_np.ndim == 2:
                tgt_np = tgt_np.mean(axis=1)
            if tgt_sr != sr:
                tgt_np = librosa.resample(
                    tgt_np, orig_sr=tgt_sr, target_sr=sr)
            
            tgt_np = tgt_np[:original_length]
            mix_short = audio_np[:original_length]
            
            m = compute_all_metrics(
                separated, tgt_np, mix_short, sr)
            
            metrics_text = "### 📊 Performance Metrics\n\n"
            metrics_text += "| Metric | Value |\n|--------|-------|\n"
            for k, v in m.items():
                if not np.isnan(v):
                    unit = ' dB' if 'dB' in k else ''
                    metrics_text += \
                        f"| {k} | {v:+.4f}{unit} |\n"
        
        # ── Generate comparison plot ──────────────────────────
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('TahirNet Audio Separation',
                     fontsize=13, fontweight='bold')
        
        t = np.linspace(0, original_length/sr, original_length)
        
        # Waveforms
        axes[0,0].plot(t, audio_np[:original_length],
                      color='#e74c3c', linewidth=0.4)
        axes[0,0].set_title('🔴 Input Mixture',
                            fontweight='bold')
        axes[0,0].set_xlabel('Time (s)')
        axes[0,0].grid(True, alpha=0.3)
        
        axes[0,1].plot(t, separated,
                      color='#2ecc71', linewidth=0.4)
        axes[0,1].set_title('🟢 Separated Output',
                            fontweight='bold')
        axes[0,1].set_xlabel('Time (s)')
        axes[0,1].grid(True, alpha=0.3)
        
        # Spectrograms
        for ax, sig, label, cmap in [
            (axes[1,0], audio_np[:original_length],
             'Input Spectrogram', 'Reds'),
            (axes[1,1], separated,
             'Separated Spectrogram', 'Greens'),
        ]:
            D = librosa.amplitude_to_db(
                np.abs(librosa.stft(
                    sig.astype(np.float32),
                    n_fft=1024, hop_length=256)),
                ref=np.max
            )
            librosa.display.specshow(
                D, sr=sr, hop_length=256,
                x_axis='time', y_axis='hz',
                ax=ax, cmap=cmap
            )
            ax.set_title(label, fontweight='bold')
        
        plt.tight_layout()
        
        # Save plot to temp file for Gradio
        with tempfile.NamedTemporaryFile(
                suffix='.png', delete=False) as tmp:
            plt.savefig(tmp.name, dpi=100, bbox_inches='tight')
            plot_path = tmp.name
        plt.close()
        
        # Save output audio to temp file
        with tempfile.NamedTemporaryFile(
                suffix='.wav', delete=False) as tmp:
            sf.write(tmp.name, separated, sr)
            audio_path = tmp.name
        
        return (sr, separated), plot_path, metrics_text
    
    # ── Build Gradio Interface ────────────────────────────────
    with gr.Blocks(
        title="TahirNet — NatVoc Separation",
        theme=gr.themes.Soft()
    ) as demo:
        
        gr.HTML("""
        <div style="text-align:center; padding:20px; 
                    background:linear-gradient(135deg,#1a1a2e,#16213e);
                    border-radius:10px; margin-bottom:20px">
            <h1 style="color:#e94560; margin:0; font-size:2em">
                🎵 TahirNet
            </h1>
            <p style="color:#a8d8ea; margin:5px 0">
                NatVoc Separation Pipeline | 
                Band-Split RNN + Inter-Band Attention
            </p>
            <p style="color:#888; font-size:0.85em">
                Separate Natural Sounds & Vocals 
                from Music/Mechanical Noise
            </p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📤 Upload Audio")
                
                mix_input = gr.Audio(
                    label="🔴 Mixture Audio (Input)",
                    type="numpy",
                    sources=["upload", "microphone"]
                )
                
                has_target = gr.Checkbox(
                    label="I have ground truth target "
                          "(for metrics)",
                    value=False
                )
                
                tgt_input = gr.Audio(
                    label="🔵 Target Audio (Optional)",
                    type="numpy",
                    visible=False
                )
                
                has_target.change(
                    fn=lambda x: gr.update(visible=x),
                    inputs=has_target,
                    outputs=tgt_input
                )
                
                separate_btn = gr.Button(
                    "🎯 Separate Audio",
                    variant="primary",
                    size="lg"
                )
                
                gr.Markdown("""
                **📌 How to use:**
                1. Upload `.wav` / `.mp3` mixture file
                2. Click **Separate Audio**
                3. Listen to separated output below
                4. Download the output audio
                
                **✅ Best results with:**
                - 4-8 second clips
                - 22050 Hz sample rate
                - Mono audio
                """)
            
            with gr.Column(scale=1):
                gr.Markdown("### 📥 Separated Output")
                
                audio_output = gr.Audio(
                    label="🟢 Separated Natural/Vocal Audio",
                    type="numpy"
                )
                
                metrics_output = gr.Markdown(
                    "Metrics will appear here after separation...")
        
        plot_output = gr.Image(
            label="📊 Waveform & Spectrogram Comparison",
            type="filepath"
        )
        
        # ── Example files ─────────────────────────────────────
        example_files = list(DATA_DIR.rglob("*.wav"))[:3]
        if example_files:
            gr.Examples(
                examples=[[str(f)] for f in example_files],
                inputs=[mix_input],
                label="📂 Example Files from Dataset"
            )
        
        # ── Connect button ────────────────────────────────────
        separate_btn.click(
            fn=separate_audio,
            inputs=[mix_input, has_target, tgt_input],
            outputs=[audio_output, plot_output, metrics_output]
        )
    
    return demo


# ══════════════════════════════════════════════════════════════
# 7. MAIN EXECUTION
# ══════════════════════════════════════════════════════════════

def main():
    print("\n" + "="*30)
    print("  TahirNet — Test & Demo Pipeline")
    print("="*30 + "\n")
    
    # ── Step 1: Load Model ───────────────────────────────────
    model = load_model(MODEL_PATH)
    
    # Multi-GPU if available
    if N_GPUS > 1:
        model = nn.DataParallel(model)
        print(f"  ✅ DataParallel: {N_GPUS} GPUs")
    
    # ── Step 2: Full Test Evaluation ─────────────────────────
    test_df = evaluate_full_test(model, DATA_DIR)
    
    # ── Step 3: Plot Results ─────────────────────────────────
    plot_metrics(test_df)
    
    # ── Step 4: Demo Type 1 — File Test ─────────────────────
    print(f"\n{'='*60}")
    print("🎵 DEMO TYPE 1 — File-Based Test")
    print(f"{'='*60}")
    
    # Find a test file automatically
    test_mix_files = list((DATA_DIR / "test" / "mixture"
                          ).glob("*.wav"))
    test_nat_files = list((DATA_DIR / "test" / "natural"
                          ).glob("*.wav"))
    
    if test_mix_files:
        mix_path = str(test_mix_files[0])
        tgt_path = str(test_nat_files[0]) \
                   if test_nat_files else None
        
        print(f"\n  📂 Using: {Path(mix_path).name}")
        separated, sr = demo_file_test(
            model,
            mix_path  = mix_path,
            tgt_path  = tgt_path,
            output_name = "demo_type1_result"
        )
    else:
        print("\n  ⚠️  No test files found for Demo 1")
        print("  💡 Run manually:")
        print("""
        demo_file_test(
            model,
            mix_path   = "/path/to/your/mixture.wav",
            tgt_path   = "/path/to/your/target.wav",
            output_name = "my_test"
        )
        """)
    
    # ── Step 5: Demo Type 2 — Gradio UI ─────────────────────
    print(f"\n{'='*60}")
    print("🌐 DEMO TYPE 2 — Gradio Web UI")
    print(f"{'='*60}")
    
    demo = build_gradio_ui(model)
    
    print("""
    📌 Gradio UI Instructions:
    ─────────────────────────────────────────────────
    1. নিচে একটি public URL আসবে যেমন:
       Running on public URL: https://xxxx.gradio.live
    
    2. সেই URL টা browser এ open করো
       (যেকোনো device থেকে access করা যাবে)
    
    3. Interface এ:
       → "Upload Audio" এ mixture .wav file দাও
       → "Separate Audio" বাটন click করো
       → Separated audio play করো / download করো
       → Spectrogram comparison দেখো
    
    4. Ground truth থাকলে:
       → "I have ground truth target" tick দাও
       → Target audio upload করো
       → SI-SDR, PESQ, STOI metrics দেখো
    
    5. Microphone থেকেও real-time input দিতে পারো!
    ─────────────────────────────────────────────────
    """)
    
    demo.launch(
        share=True,           # Public URL generate করবে
        show_error=True,
        quiet=False
    )


# ── RUN ──────────────────────────────────────────────────────
if __name__ == "__main__":
    main()