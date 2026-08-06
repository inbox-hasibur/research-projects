# ============================================================
# MDXNet2 — Data Process & Model Train Pipeline
# Architecture: TFC-TDF U-Net (MDX-Net Base)
# Dataset: MECNature V2 (22050 Hz)
# ============================================================

import os
import time
import glob
import random
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast

# ─────────────────────────────────────────
# SEED & REPRODUCIBILITY
# ─────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark     = True

# ─────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────
CONFIG = {
    # Paths
    "data_dir"         : Path("/kaggle/input/datasets/inboxhasibur/mecnature-audio-dataset"),
    "output_dir"       : Path("/kaggle/working/mdxnet2"),
    "resume_checkpoint": "",  # Path to Kaggle uploaded checkpoint if resuming

    # Audio
    "sr"               : 22050,
    "clip_length"      : 4,      # seconds
    "n_fft"            : 1024,
    "hop_length"       : 256,
    
    # Model (TFC-TDF U-Net)
    "channels"         : 24,     # Base channels for U-Net
    "num_blocks"       : 4,      # Depth of U-Net
    
    # Training
    "batch_size"       : 16,
    "epochs"           : 200,
    "lr"               : 5e-4,
    "weight_decay"     : 1e-4,
    "patience"         : 10,
    "grad_clip"        : 3.0,
    "time_limit_hr"    : 11.5,
}

SAMPLES = CONFIG["sr"] * CONFIG["clip_length"]
OUT_DIR = CONFIG["output_dir"]
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_GPUS   = torch.cuda.device_count()

print(f"{'='*60}")
print(f"  MDXNet2 — NatVoc Separation (Train Pipeline)")
print(f"{'='*60}")
print(f"  Device   : {DEVICE}")
print(f"  GPUs     : {N_GPUS}")
print(f"  SR       : {CONFIG['sr']} Hz")
print(f"  Clip     : {CONFIG['clip_length']}s  ({SAMPLES} samples)")
print(f"{'='*60}\n")


# ══════════════════════════════════════════════════════════════
# 1. DATASET & DATALOADER (GPU Optimized)
# ══════════════════════════════════════════════════════════════

class MECNatureDataset(Dataset):
    """
    Loads MECNature mixture / natural / interference triplets.
    Uses ultra-fast volume scaling augmentation instead of CPU resampling.
    """
    def __init__(self, split="train", augment=False):
        self.augment = augment
        self.sr      = CONFIG["sr"]
        
        base = CONFIG["data_dir"] / split
        self.mix_files    = sorted((base / "mixture").glob("*.wav"))
        self.nat_files    = sorted((base / "natural").glob("*.wav"))
        self.interf_files = sorted((base / "interference").glob("*.wav"))

        assert len(self.mix_files) == len(self.nat_files), \
            f"[{split}] mixture/natural count mismatch!"
        print(f"  [{split:5s}] {len(self.mix_files)} samples loaded.")

    def __len__(self):
        return len(self.mix_files)

    def _load(self, path):
        # torchaudio loads natively; assuming files are already 22050Hz
        audio, _ = torchaudio.load(str(path))
        audio = audio.mean(0) # stereo to mono
        
        if audio.shape[-1] < SAMPLES:
            audio = F.pad(audio, (0, SAMPLES - audio.shape[-1]))
        else:
            if self.augment:
                start = random.randint(0, audio.shape[-1] - SAMPLES)
                audio = audio[start: start + SAMPLES]
            else:
                audio = audio[:SAMPLES]
        return audio

    @staticmethod
    def _norm(x, eps=1e-8):
        return x / (x.abs().max() + eps)

    def __getitem__(self, idx):
        nat = self._norm(self._load(self.nat_files[idx]))

        # Dynamic Remixing Augmentation
        if self.augment and random.random() < 0.6:
            ridx   = random.randint(0, len(self.interf_files) - 1)
            interf = self._norm(self._load(self.interf_files[ridx]))
        else:
            interf = self._norm(self._load(self.interf_files[idx]))

        # Volume Augmentation (super fast)
        if self.augment and random.random() < 0.3:
            vol_scale = random.uniform(0.5, 1.5)
            nat = self._norm(nat * vol_scale)

        snr     = random.uniform(-5, 5)
        scale   = 10 ** (-snr / 20)
        mixture = self._norm(nat + scale * interf)

        return mixture, nat

def build_loaders():
    print("Loading datasets...")
    train_ds  = MECNatureDataset("train", augment=True)
    val_ds    = MECNatureDataset("val",   augment=False)
    
    eff_batch = CONFIG["batch_size"] * max(N_GPUS, 1)
    kw        = dict(num_workers=4, pin_memory=True, persistent_workers=True)

    return (
        DataLoader(train_ds, batch_size=eff_batch, shuffle=True,  **kw),
        DataLoader(val_ds,   batch_size=eff_batch, shuffle=False, **kw)
    )


# ══════════════════════════════════════════════════════════════
# 2. MODEL — MDXNet2 (TFC-TDF U-Net)
# ══════════════════════════════════════════════════════════════

class TFCBlock(nn.Module):
    """Time-Frequency Convolution Block"""
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class TDFBlock(nn.Module):
    """Time-Distributed Fully-connected Block for Frequency Attention"""
    def __init__(self, channels, freq_bins):
        super().__init__()
        mid = max(freq_bins // 2, 1)
        # Apply dense layers across the frequency dimension
        self.fc = nn.Sequential(
            nn.Linear(freq_bins, mid, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Linear(mid, freq_bins, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x shape: (B, C, F, T)
        # apply linear across the F dimension
        out = x.transpose(-1, -2)  # (B, C, T, F)
        out = self.fc(out)         # (B, C, T, F)
        out = out.transpose(-1, -2) # (B, C, F, T)
        return x * out

class MDXNet2(nn.Module):
    def __init__(self):
        super().__init__()
        import math
        self.n_fft = CONFIG["n_fft"]
        self.hop_length = CONFIG["hop_length"]
        # Make window a buffer so it moves to device automatically
        self.register_buffer("window", torch.hann_window(self.n_fft))
        self.num_blocks = CONFIG["num_blocks"]
        
        # Track freq_bins using CEILING division — same as what Conv2d(stride=2) actually produces
        freq_bins = self.n_fft // 2 + 1
        base_ch = CONFIG["channels"]
        
        self.encoders    = nn.ModuleList()
        self.decoders    = nn.ModuleList()
        self.tdfs        = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        self.upsamples   = nn.ModuleList()
        
        # Store per-level freq sizes for decoder TDF construction
        enc_freq_bins = []
        
        in_c = 2  # Real & Imaginary channels
        
        # Encoder
        for i in range(self.num_blocks):
            out_c = base_ch * (2 ** i)
            self.encoders.append(TFCBlock(in_c, out_c))
            self.downsamples.append(nn.Conv2d(out_c, out_c, kernel_size=(2, 2), stride=(2, 2)))
            enc_freq_bins.append(freq_bins)          # freq size BEFORE this downsample
            in_c = out_c
            freq_bins = freq_bins // 2               # floor — matches Conv2d(kernel=2, stride=2) exactly
            
        # Bottleneck
        out_c = base_ch * (2 ** self.num_blocks)
        self.bottleneck_tfc = TFCBlock(in_c, out_c)
        self.bottleneck_tdf = TDFBlock(out_c, freq_bins)  # correct size now
        
        # Decoder (mirror of encoder, reversed)
        for i in range(self.num_blocks - 1, -1, -1):
            skip_freq = enc_freq_bins[i]              # freq size at skip connection level
            dec_in_c  = out_c + base_ch * (2 ** i)   # concat with skip
            dec_out_c = base_ch * (2 ** i)
            
            self.upsamples.append(nn.ConvTranspose2d(out_c, base_ch * (2 ** i), kernel_size=(2, 2), stride=(2, 2)))
            self.tdfs.append(TDFBlock(dec_in_c, skip_freq))  # matches skip connection freq
            self.decoders.append(TFCBlock(dec_in_c, dec_out_c))
            
            out_c     = dec_out_c
            freq_bins = skip_freq
            
        # Final projection to mask
        self.final_conv = nn.Conv2d(base_ch, 2, kernel_size=3, padding=1)
        
    def stft(self, x):
        X = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop_length, 
                       window=self.window, return_complex=True)
        # Convert Complex to (B, 2, F, T)
        X_ri = torch.stack([X.real, X.imag], dim=1)
        return X_ri
        
    def istft(self, X_ri, length):
        # Convert (B, 2, F, T) to Complex
        X = torch.complex(X_ri[:, 0], X_ri[:, 1])
        x = torch.istft(X, n_fft=self.n_fft, hop_length=self.hop_length, 
                        window=self.window, length=length)
        return x

    def forward(self, x):
        length = x.shape[-1]
        
        # STFT
        X_ri = self.stft(x)   # (B, 2, F, T)
        orig_f, orig_t = X_ri.shape[2], X_ri.shape[3]
        
        # NO manual padding needed: we built TDFBlock using ceil(freq/2),
        # which matches Conv2d(stride=2) exactly, so shapes are always consistent.
        
        # U-Net forward
        skips = []
        out = X_ri
        
        # Encoder
        for i in range(self.num_blocks):
            out = self.encoders[i](out)
            skips.append(out)
            out = self.downsamples[i](out)
            
        # Bottleneck
        out = self.bottleneck_tfc(out)
        out = self.bottleneck_tdf(out)
        
        # Decoder
        for i in range(self.num_blocks):
            out = self.upsamples[i](out)
            skip = skips[-(i + 1)]
            
            # Trim if ConvTranspose2d produces one extra row/col on odd sizes
            if out.shape[-2:] != skip.shape[-2:]:
                out = out[:, :, :skip.shape[2], :skip.shape[3]]
                
            out = torch.cat([out, skip], dim=1)  # concat along channel dim
            out = self.tdfs[i](out)
            out = self.decoders[i](out)
            
        # Final mask
        mask = self.final_conv(out)   # (B, 2, F, T)
        
        # Trim mask to original STFT size if U-Net altered dims
        mask = mask[:, :, :orig_f, :orig_t]
        
        # Apply mask to input STFT
        Y_ri = X_ri[:, :, :orig_f, :orig_t] * mask
        
        # iSTFT
        y = self.istft(Y_ri, length)
        return y


# ══════════════════════════════════════════════════════════════
# 3. METRICS & LOSS
# ══════════════════════════════════════════════════════════════

def calc_sisnr(est, tgt, eps=1e-8):
    tgt_energy = torch.sum(tgt ** 2, dim=-1, keepdim=True) + eps
    scale = torch.sum(tgt * est, dim=-1, keepdim=True) / tgt_energy
    proj = scale * tgt
    noise = est - proj
    sisnr = 10 * torch.log10((torch.sum(proj ** 2, dim=-1) + eps) / (torch.sum(noise ** 2, dim=-1) + eps))
    return sisnr.mean()

class MDXHybridLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("window", torch.hann_window(CONFIG["n_fft"]))
        
    def forward(self, est, tgt):
        # Time-domain L1 Loss
        l1_time = F.l1_loss(est, tgt)
        
        # Frequency-domain L1 Loss
        EST = torch.stft(est, n_fft=CONFIG["n_fft"], hop_length=CONFIG["hop_length"], window=self.window, return_complex=True)
        TGT = torch.stft(tgt, n_fft=CONFIG["n_fft"], hop_length=CONFIG["hop_length"], window=self.window, return_complex=True)
        l1_freq = F.l1_loss(torch.abs(EST), torch.abs(TGT))
        
        # Negative SI-SDR Loss
        sisnr_loss = -calc_sisnr(est, tgt) / 100.0 # Scaled down to match L1 range roughly
        
        return l1_time + l1_freq + sisnr_loss


# ══════════════════════════════════════════════════════════════
# 4. TRAINING PIPELINE & CHECKPOINTING
# ══════════════════════════════════════════════════════════════

def save_checkpoint(model, optimizer, scaler, epoch, val_loss, history, is_best=False, suffix=""):
    state = {
        "epoch": epoch,
        "val_loss": val_loss,
        "history": history,
        "opt_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict()
    }
    
    if isinstance(model, nn.DataParallel):
        state["model_state"] = model.module.state_dict()
    else:
        state["model_state"] = model.state_dict()
        
    if is_best:
        for f in glob.glob(str(OUT_DIR / "*_best.pt")):
            if "mdxnet2_best.pt" != Path(f).name:
                Path(f).unlink(missing_ok=True)
        torch.save(state, OUT_DIR / f"mdxnet2_ep{epoch:03d}_best.pt")
        torch.save(state["model_state"], OUT_DIR / "mdxnet2_best.pt")
    elif suffix:
        if suffix == "latest":
            for f in glob.glob(str(OUT_DIR / "*_latest.pt")):
                Path(f).unlink(missing_ok=True)
        torch.save(state, OUT_DIR / f"mdxnet2_ep{epoch:03d}_{suffix}.pt")

def run_epoch(model, loader, criterion, optimizer, scaler, training=True, desc=""):
    model.train() if training else model.eval()
    total_loss = total_sisnr = 0.0
    ctx = torch.enable_grad if training else torch.no_grad

    with ctx():
        for mix, tgt in tqdm(loader, desc=desc, leave=False):
            mix, tgt = mix.to(DEVICE, non_blocking=True), tgt.to(DEVICE, non_blocking=True)

            if training:
                optimizer.zero_grad(set_to_none=True)

            with autocast('cuda'):
                est  = model(mix)
                loss = criterion(est, tgt)

            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"])
                scaler.step(optimizer)
                scaler.update()

            total_loss += loss.item()
            total_sisnr += calc_sisnr(est.detach(), tgt.detach()).item()

    n = len(loader)
    return total_loss / n, total_sisnr / n

def train():
    train_loader, val_loader = build_loaders()

    model = MDXNet2().to(DEVICE)
    if N_GPUS > 1:
        print(f"\n  DataParallel across {N_GPUS} GPUs")
        model = nn.DataParallel(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=4, factor=0.5)
    criterion = MDXHybridLoss().to(DEVICE)
    scaler    = GradScaler('cuda')

    best_val_loss   = float("inf")
    patience_ctr    = 0
    history         = []
    start_ep        = 1

    # Resume Logic
    resume_path = CONFIG.get("resume_checkpoint", "")
    if resume_path and Path(resume_path).exists():
        print(f"\n🔄 Resuming from checkpoint: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=DEVICE)
        
        state_dict = checkpoint.get("model_state", checkpoint)
        new_state_dict = {}
        is_module = any(k.startswith("module.") for k in state_dict.keys())
        for k, v in state_dict.items():
            if is_module and N_GPUS <= 1:
                new_state_dict[k.replace("module.", "")] = v
            elif not is_module and N_GPUS > 1:
                new_state_dict["module." + k] = v
            else:
                new_state_dict[k] = v
                
        model.load_state_dict(new_state_dict)
        
        if "opt_state" in checkpoint: optimizer.load_state_dict(checkpoint["opt_state"])
        if "scaler_state" in checkpoint: scaler.load_state_dict(checkpoint["scaler_state"])
        if "epoch" in checkpoint: start_ep = checkpoint["epoch"] + 1
        if "val_loss" in checkpoint: best_val_loss = checkpoint["val_loss"]
        if "history" in checkpoint: history = checkpoint["history"]
        print(f"   ✅ Resumed successfully from Epoch {start_ep-1}")
    else:
        print("\n🚀 Starting fresh MDXNet2 training...")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters : {n_params:,}\n")

    t_start = time.time()
    
    print("=" * 65)
    print(f"{'Ep':>4} | {'T-Loss':>8} | {'T-SISNR':>8} | {'V-Loss':>8} | {'V-SISNR':>8} | {'LR':>8} | Time")
    print("=" * 65)

    for ep in range(start_ep, CONFIG["epochs"] + 1):
        elapsed = (time.time() - t_start) / 3600
        if elapsed >= CONFIG["time_limit_hr"]:
            print(f"\n⏱ Time limit hit. Saving safe checkpoint and stopping at epoch {ep-1}.")
            save_checkpoint(model, optimizer, scaler, ep-1, best_val_loss, history, suffix="timeout")
            break

        lr = optimizer.param_groups[0]["lr"]

        t_loss, t_sisnr = run_epoch(model, train_loader, criterion, optimizer, scaler, training=True, desc=f"Ep{ep:03d} Train")
        v_loss, v_sisnr = run_epoch(model, val_loader, criterion, optimizer, scaler, training=False, desc=f"Ep{ep:03d} Val  ")

        scheduler.step(v_loss)
        elapsed = (time.time() - t_start) / 3600

        print(f"{ep:4d} | {t_loss:8.4f} | {t_sisnr:8.3f} | {v_loss:8.4f} | {v_sisnr:8.3f} | {lr:.2e} | {elapsed:.2f}h")
        history.append(dict(epoch=ep, train_loss=t_loss, train_sisnr=t_sisnr, val_loss=v_loss, val_sisnr=v_sisnr))

        # Best checkpoint
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            patience_ctr  = 0
            save_checkpoint(model, optimizer, scaler, ep, v_loss, history, is_best=True)
            print(f"       ✅ Best model saved  (val_loss={best_val_loss:.4f})")
        else:
            patience_ctr += 1
            if patience_ctr >= CONFIG["patience"]:
                print(f"\n⏹ Early stopping triggered at epoch {ep}.")
                save_checkpoint(model, optimizer, scaler, ep, v_loss, history, suffix="early_stop")
                break

        save_checkpoint(model, optimizer, scaler, ep, v_loss, history, suffix="latest")
        if ep % 20 == 0:
            save_checkpoint(model, optimizer, scaler, ep, v_loss, history, suffix="backup")

    pd.DataFrame(history).to_csv(OUT_DIR / "history.csv", index=False)
    return model

if __name__ == "__main__":
    train()
    print(f"\n{'='*70}")
    print(f"  MDXNet2 Training Complete!")
    print(f"  Outputs     : {OUT_DIR}")
    print(f"{'='*70}")
