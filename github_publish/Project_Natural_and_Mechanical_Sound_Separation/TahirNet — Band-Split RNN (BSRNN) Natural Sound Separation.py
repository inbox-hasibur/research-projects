# ============================================================
# TahirNet — NatVoc Separation Pipeline
# Band-Split RNN (BSRNN) | MECNature Dataset | 2x T4 GPU
# AMP + DataParallel + Dynamic Remixing + Pruning
# ============================================================

import os
import time
import random
import numpy as np
import pandas as pd
from pathlib import Path
import glob
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast
import torch.nn.utils.prune as prune

# ─────────────────────────────────────────
# SEED & REPRODUCIBILITY
# ─────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = True

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
CONFIG = {
    # Paths
    "data_dir"         : Path("/kaggle/input/datasets/inboxhasibur/mecnature-audio-dataset"),
    "output_dir"       : Path("/kaggle/working/tahirnet"),
    "resume_checkpoint": "",  # e.g., "/kaggle/input/tahirnet-checkpoint/tahirnet_ep004_latest.pt"

    # Audio
    "sr"               : 22050,
    "clip_length"      : 4,
    "n_fft"            : 1024,
    "hop_length"       : 256,

    # Band boundaries in Hz
    # Sub-bass | Bass/Drums | Vocal/Bird | Air/Insect/Wind
    "band_boundaries"  : [0, 300, 1000, 4000, 8000, 11025],

    # Model
    "hidden_dim"       : 192,
    "num_rnn_layers"   : 4,
    "dropout"          : 0.1,

    # Training
    "batch_size"       : 6,        # per GPU → 12 effective on 2x T4
    "epochs"           : 200,
    "lr"               : 3e-4,
    "weight_decay"     : 1e-4,
    "patience"         : 5,
    "grad_clip"        : 5.0,
    "time_limit_hr"    : 11.5,

    # Loss
    "lambda_freq"      : 0.5,

    # Pruning
    "prune_amount"     : 0.15
}

def save_checkpoint(model, optimizer, scaler, epoch, val_loss, history, is_best=False, suffix=""):
    state = {
        "epoch": epoch,
        "val_loss": val_loss,
        "history": history,
        "opt_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict()
    }
    
    # Strip module. prefix if DataParallel is used
    if isinstance(model, nn.DataParallel):
        state["model_state"] = model.module.state_dict()
    else:
        state["model_state"] = model.state_dict()
        
    if is_best:
        # Delete old best checkpoints (except the generic one)
        for f in glob.glob(str(OUT_DIR / "*_best.pt")):
            if "tahirnet_best.pt" != Path(f).name:
                Path(f).unlink(missing_ok=True)
        filename = OUT_DIR / f"tahirnet_ep{epoch:03d}_best.pt"
        torch.save(state, filename)
        # Always keep the generic name for easy loading by the test script
        torch.save(state["model_state"], OUT_DIR / "tahirnet_best.pt")
    elif suffix:
        if suffix == "latest":
            # Delete old latest checkpoints to save Kaggle disk space
            for f in glob.glob(str(OUT_DIR / "*_latest.pt")):
                Path(f).unlink(missing_ok=True)
        filename = OUT_DIR / f"tahirnet_ep{epoch:03d}_{suffix}.pt"
        torch.save(state, filename)

SAMPLES    = CONFIG["sr"] * CONFIG["clip_length"]
N_BINS     = CONFIG["n_fft"] // 2 + 1
OUT_DIR    = CONFIG["output_dir"]
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_GPUS   = torch.cuda.device_count()

print(f"{'='*60}")
print(f"  TahirNet — NatVoc Separation Pipeline")
print(f"{'='*60}")
print(f"  Device   : {DEVICE}")
print(f"  GPUs     : {N_GPUS}")
print(f"  SR       : {CONFIG['sr']} Hz")
print(f"  Clip     : {CONFIG['clip_length']}s  ({SAMPLES} samples)")
print(f"  Bands    : {CONFIG['band_boundaries']}")
print(f"{'='*60}\n")


# ══════════════════════════════════════════════════════════════
# 1. DATASET & DATALOADER
# ══════════════════════════════════════════════════════════════

class MECNatureDataset(Dataset):
    """
    Loads MECNature mixture / natural / interference triplets.
    Dynamic Remixing: randomly swaps interference sources during training
    to prevent overfitting and improve generalisation.
    """
    def __init__(self, split="train", augment=False):
        self.augment  = augment
        self.sr       = CONFIG["sr"]

        base = CONFIG["data_dir"] / split
        self.mix_files    = sorted((base / "mixture").glob("*.wav"))
        self.nat_files    = sorted((base / "natural").glob("*.wav"))
        self.interf_files = sorted((base / "interference").glob("*.wav"))

        assert len(self.mix_files) == len(self.nat_files), \
            f"[{split}] mixture/natural count mismatch!"
        print(f"  [{split:5s}] {len(self.mix_files)} samples loaded.")

    def __len__(self):
        return len(self.mix_files)

    # ── helpers ──────────────────────────────────────────────
    def _load(self, path):
        audio, sr = torchaudio.load(str(path))
        if sr != self.sr:
            audio = torchaudio.functional.resample(audio, sr, self.sr)
        audio = audio.mean(0)                          # stereo → mono
        if audio.shape[-1] < SAMPLES:
            audio = F.pad(audio, (0, SAMPLES - audio.shape[-1]))
        else:
            # Random crop during training, fixed crop during eval
            if self.augment:
                start = random.randint(0, audio.shape[-1] - SAMPLES)
                audio = audio[start: start + SAMPLES]
            else:
                audio = audio[:SAMPLES]
        return audio

    @staticmethod
    def _norm(x, eps=1e-8):
        return x / (x.abs().max() + eps)

    @staticmethod
    def _mix(nat, interf, snr_db):
        scale  = 10 ** (-snr_db / 20)
        mixture = nat + scale * interf
        return MECNatureDataset._norm(mixture)

    # ── getitem ──────────────────────────────────────────────
    def __getitem__(self, idx):
        nat    = self._norm(self._load(self.nat_files[idx]))

        # Dynamic Remixing Augmentation
        if self.augment and random.random() < 0.6:
            ridx   = random.randint(0, len(self.interf_files) - 1)
            interf = self._norm(self._load(self.interf_files[ridx]))
        else:
            interf = self._norm(self._load(self.interf_files[idx]))

        # Volume aug (super fast) instead of slow CPU resampling
        if self.augment and random.random() < 0.3:
            vol_scale = random.uniform(0.5, 1.5)
            nat = self._norm(nat * vol_scale)

        snr     = random.uniform(-5, 5)
        mixture = self._mix(nat, interf, snr)

        return mixture, nat          # (input, target)


def build_loaders():
    print("Loading datasets...")
    train_ds  = MECNatureDataset("train", augment=True)
    val_ds    = MECNatureDataset("val",   augment=False)
    test_ds   = MECNatureDataset("test",  augment=False)

    eff_batch = CONFIG["batch_size"] * max(N_GPUS, 1)
    kw        = dict(num_workers=4, pin_memory=True, persistent_workers=True)

    return (
        DataLoader(train_ds, batch_size=eff_batch, shuffle=True,  **kw),
        DataLoader(val_ds,   batch_size=eff_batch, shuffle=False, **kw),
        DataLoader(test_ds,  batch_size=eff_batch, shuffle=False, **kw),
    )


# ══════════════════════════════════════════════════════════════
# 2. MODEL — TahirNet (Novel BSRNN + Inter-Band Attention)
# ══════════════════════════════════════════════════════════════

def _band_slices(boundaries, n_fft, sr):
    """Convert Hz boundaries → STFT bin slices."""
    freq_res = sr / n_fft
    slices   = []
    for lo, hi in zip(boundaries[:-1], boundaries[1:]):
        lo_bin = int(lo / freq_res)
        hi_bin = min(int(hi / freq_res), n_fft // 2 + 1)
        if hi_bin > lo_bin:
            slices.append(slice(lo_bin, hi_bin))
    return slices


class BandSplitProjection(nn.Module):
    """Project each frequency band (real+imag) → hidden_dim."""
    def __init__(self, band_bins, hidden):
        super().__init__()
        self.projs = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(b * 2),
                nn.Linear(b * 2, hidden),
                nn.GELU(),
            )
            for b in band_bins
        ])

    def forward(self, S_ri, slices):
        # S_ri: [B, F, T, 2]
        feats = []
        for proj, sl in zip(self.projs, slices):
            band = S_ri[:, sl, :, :]                   # [B, bins, T, 2]
            B, bins, T, _ = band.shape
            flat = band.permute(0, 2, 1, 3).reshape(B, T, bins * 2)
            feats.append(proj(flat))                    # [B, T, H]
        return torch.stack(feats, dim=1)                # [B, K, T, H]


class IntraBandGRU(nn.Module):
    """BiGRU over time within each band."""
    def __init__(self, hidden, layers, dropout):
        super().__init__()
        self.gru  = nn.GRU(hidden, hidden // 2, num_layers=layers,
                           batch_first=True, bidirectional=True,
                           dropout=dropout if layers > 1 else 0)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x):
        # x: [B, K, T, H]
        B, K, T, H = x.shape
        x_flat      = x.reshape(B * K, T, H)
        out, _      = self.gru(x_flat)
        out         = out.reshape(B, K, T, H)
        return self.norm(x + out)


class InterBandAttention(nn.Module):
    """
    Novel addition: Multi-head self-attention across bands
    at each time step — allows the model to learn cross-band
    relationships (e.g. vocal harmonics spanning mid bands).
    """
    def __init__(self, hidden, num_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden, num_heads,
                                          dropout=dropout,
                                          batch_first=True)
        self.norm = nn.LayerNorm(hidden)
        self.ff   = nn.Sequential(
            nn.Linear(hidden, hidden * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden),
        )
        self.norm2 = nn.LayerNorm(hidden)

    def forward(self, x):
        # x: [B, K, T, H]
        B, K, T, H = x.shape
        x_t        = x.permute(0, 2, 1, 3).reshape(B * T, K, H)
        attn_out, _ = self.attn(x_t, x_t, x_t)
        x_t        = self.norm(x_t + attn_out)
        x_t        = self.norm2(x_t + self.ff(x_t))
        return x_t.reshape(B, T, K, H).permute(0, 2, 1, 3)


class BandMaskDecoder(nn.Module):
    """Decode hidden features → complex soft mask per band."""
    def __init__(self, band_bins, hidden):
        super().__init__()
        self.decoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden, b * 2),
                nn.Tanh(),
            )
            for b in band_bins
        ])

    def forward(self, x, slices):
        # x: [B, K, T, H]
        masks = []
        for i, (dec, sl) in enumerate(zip(self.decoders, slices)):
            feat = x[:, i, :, :]                       # [B, T, H]
            m    = dec(feat)                            # [B, T, bins*2]
            B, T, _ = m.shape
            bins = sl.stop - sl.start
            m    = m.reshape(B, T, bins, 2).permute(0, 2, 1, 3)
            masks.append(m)                             # [B, bins, T, 2]
        return masks


class TahirNet(nn.Module):
    """
    TahirNet: Novel Band-Split RNN with Inter-Band Attention
    for Natural Sound & Vocal Preservation (NatVoc Separation).

    Architecture:
        STFT → Band-Split Projection
             → [IntraBand BiGRU → InterBand Attention] × N
             → Band Mask Decoder
             → iSTFT
    """
    def __init__(self):
        super().__init__()
        self.sr         = CONFIG["sr"]
        self.n_fft      = CONFIG["n_fft"]
        self.hop        = CONFIG["hop_length"]
        self.hidden     = CONFIG["hidden_dim"]
        self.slices     = _band_slices(
            CONFIG["band_boundaries"], self.n_fft, self.sr
        )
        self.band_bins  = [sl.stop - sl.start for sl in self.slices]
        K               = len(self.slices)

        print(f"  Bands      : {K}")
        print(f"  Bins/band  : {self.band_bins}")
        print(f"  Hidden dim : {self.hidden}")

        self.band_proj  = BandSplitProjection(self.band_bins, self.hidden)

        # Stacked Intra-GRU + Inter-Attention blocks
        n_layers = CONFIG["num_rnn_layers"]
        drop     = CONFIG["dropout"]
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                "intra": IntraBandGRU(self.hidden, 2, drop),
                "inter": InterBandAttention(self.hidden, num_heads=4,
                                            dropout=drop),
            })
            for _ in range(n_layers)
        ])

        self.mask_dec = BandMaskDecoder(self.band_bins, self.hidden)

        # Learnable band weights (novel: weight each band's contribution)
        self.band_weights = nn.Parameter(torch.ones(K))

    def _stft(self, x):
        win = torch.hann_window(self.n_fft).to(x.device)
        return torch.stft(x, n_fft=self.n_fft, hop_length=self.hop,
                          window=win, return_complex=True)

    def _istft(self, S, length):
        win = torch.hann_window(self.n_fft).to(S.device)
        return torch.istft(S, n_fft=self.n_fft, hop_length=self.hop,
                           window=win, length=length)

    def forward(self, mixture):
        length = mixture.shape[-1]

        S      = self._stft(mixture)                   # [B, F, T] complex
        S_ri   = torch.stack([S.real, S.imag], dim=-1) # [B, F, T, 2]

        # Band-split projection
        x = self.band_proj(S_ri, self.slices)          # [B, K, T, H]

        # Stacked blocks
        for block in self.blocks:
            x = block["intra"](x)
            x = block["inter"](x)

        # Learnable band weighting
        bw = torch.softmax(self.band_weights, dim=0)   # [K]
        x  = x * bw.view(1, -1, 1, 1)

        # Mask decoding
        masks   = self.mask_dec(x, self.slices)

        # Apply masks → reconstruct full spectrum
        S_out   = torch.zeros_like(S_ri)
        for mask, sl in zip(masks, self.slices):
            S_out[:, sl, :, :] = S_ri[:, sl, :, :] * mask

        S_sep   = torch.complex(S_out[..., 0], S_out[..., 1])
        return self._istft(S_sep, length=length)


# ══════════════════════════════════════════════════════════════
# 3. LOSS FUNCTIONS
# ══════════════════════════════════════════════════════════════

def si_sdr(est, tgt, eps=1e-8):
    tgt  = tgt - tgt.mean(-1, keepdim=True)
    est  = est - est.mean(-1, keepdim=True)
    dot  = (est * tgt).sum(-1, keepdim=True)
    s_t  = dot * tgt / (tgt.pow(2).sum(-1, keepdim=True) + eps)
    e_n  = est - s_t
    return 10 * torch.log10(
        s_t.pow(2).sum(-1) / (e_n.pow(2).sum(-1) + eps) + eps
    )


def si_sdr_loss(est, tgt):
    return -si_sdr(est, tgt).mean()


class MultiResSTFTLoss(nn.Module):
    CONFIGS = [
        (256,  64),
        (512,  128),
        (1024, 256),
        (2048, 512),
    ]

    def forward(self, est, tgt):
        loss = 0.0
        for n_fft, hop in self.CONFIGS:
            win   = torch.hann_window(n_fft).to(est.device)
            S_est = torch.stft(est, n_fft=n_fft, hop_length=hop,
                               window=win, return_complex=True).abs()
            S_tgt = torch.stft(tgt, n_fft=n_fft, hop_length=hop,
                               window=win, return_complex=True).abs()
            loss += F.l1_loss(S_est, S_tgt)
            loss += F.l1_loss(torch.log(S_est + 1e-8),
                              torch.log(S_tgt + 1e-8))
        return loss / len(self.CONFIGS)


class HybridLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.stft_loss = MultiResSTFTLoss()
        self.lam       = CONFIG["lambda_freq"]

    def forward(self, est, tgt):
        return si_sdr_loss(est, tgt) + self.lam * self.stft_loss(est, tgt)


# ══════════════════════════════════════════════════════════════
# 4. METRICS
# ══════════════════════════════════════════════════════════════

@torch.no_grad()
def batch_metrics(est, tgt):
    sisnr = si_sdr(est, tgt).mean().item()
    # SDR approximation (no permutation needed — single source)
    noise = est - tgt
    sdr   = 10 * torch.log10(
        tgt.pow(2).sum(-1) / (noise.pow(2).sum(-1) + 1e-8) + 1e-8
    ).mean().item()
    return sisnr, sdr


# ══════════════════════════════════════════════════════════════
# 5. TRAINING
# ══════════════════════════════════════════════════════════════

def run_epoch(model, loader, criterion, optimizer, scaler,
              training=True, desc=""):
    model.train() if training else model.eval()
    total_loss = total_sisnr = total_sdr = 0.0
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
                nn.utils.clip_grad_norm_(model.parameters(),
                                         CONFIG["grad_clip"])
                scaler.step(optimizer)
                scaler.update()

            sisnr, sdr    = batch_metrics(est.detach(), tgt.detach())
            total_loss   += loss.item()
            total_sisnr  += sisnr
            total_sdr    += sdr

    n = len(loader)
    return total_loss / n, total_sisnr / n, total_sdr / n


def train():
    train_loader, val_loader, test_loader = build_loaders()

    model = TahirNet().to(DEVICE)
    if N_GPUS > 1:
        print(f"\n  DataParallel across {N_GPUS} GPUs")
        model = nn.DataParallel(model)

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=CONFIG["lr"],
                                  weight_decay=CONFIG["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5
    )
    criterion = HybridLoss().to(DEVICE)
    scaler    = GradScaler('cuda')

    best_val_loss   = float("inf")
    patience_ctr    = 0
    history         = []
    start_ep        = 1

    # ── RESUME CHECKPOINT LOGIC ──────────────────────────────────────
    resume_path = CONFIG.get("resume_checkpoint", "")
    if resume_path and Path(resume_path).exists():
        print(f"\n🔄 Resuming from checkpoint: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=DEVICE)
        
        # Handle state dict matching (DataParallel to single or vice versa)
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
        
        if "opt_state" in checkpoint:
            optimizer.load_state_dict(checkpoint["opt_state"])
        if "scaler_state" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state"])
        if "epoch" in checkpoint:
            start_ep = checkpoint["epoch"] + 1
        if "val_loss" in checkpoint:
            best_val_loss = checkpoint["val_loss"]
        if "history" in checkpoint:
            history = checkpoint["history"]
            
        print(f"   ✅ Resumed successfully from Epoch {start_ep-1} (Best Val Loss: {best_val_loss:.4f})")
    else:
        if resume_path:
            print(f"\n⚠️  Resume path not found: {resume_path}. Starting fresh training...")
        else:
            print("\n🚀 Starting fresh training...")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters : {n_params:,}\n")

    t_start = time.time()

    print("=" * 70)
    print(f"{'Ep':>4} | {'T-Loss':>8} | {'T-SISNR':>8} | "
          f"{'V-Loss':>8} | {'V-SISNR':>8} | {'V-SDR':>7} | {'LR':>8} | Time")
    print("=" * 70)

    for ep in range(start_ep, CONFIG["epochs"] + 1):

        # Time guard
        elapsed = (time.time() - t_start) / 3600
        if elapsed >= CONFIG["time_limit_hr"]:
            print(f"\n⏱ Time limit hit ({elapsed:.2f}h >= {CONFIG['time_limit_hr']}h). Saving safe checkpoint and stopping at epoch {ep-1}.")
            save_checkpoint(model, optimizer, scaler, ep-1, best_val_loss, history, is_best=False, suffix="timeout")
            break

        lr = optimizer.param_groups[0]["lr"]

        t_loss, t_sisnr, _  = run_epoch(
            model, train_loader, criterion, optimizer, scaler,
            training=True,  desc=f"Ep{ep:03d} Train"
        )
        v_loss, v_sisnr, v_sdr = run_epoch(
            model, val_loader, criterion, optimizer, scaler,
            training=False, desc=f"Ep{ep:03d} Val  "
        )

        scheduler.step(v_loss)
        elapsed = (time.time() - t_start) / 3600

        print(f"{ep:4d} | {t_loss:8.4f} | {t_sisnr:8.3f} | "
              f"{v_loss:8.4f} | {v_sisnr:8.3f} | {v_sdr:7.3f} | "
              f"{lr:.2e} | {elapsed:.2f}h")

        history.append(dict(epoch=ep, train_loss=t_loss, train_sisnr=t_sisnr,
                            val_loss=v_loss, val_sisnr=v_sisnr, val_sdr=v_sdr))

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
                save_checkpoint(model, optimizer, scaler, ep, v_loss, history, is_best=False, suffix="early_stop")
                break

        # Periodic checkpoint every epoch (overwrites previous 'latest')
        save_checkpoint(model, optimizer, scaler, ep, v_loss, history, is_best=False, suffix="latest")
        
        # Occasional persistent checkpoint
        if ep % 20 == 0:
            save_checkpoint(model, optimizer, scaler, ep, v_loss, history, is_best=False, suffix="backup")

    pd.DataFrame(history).to_csv(OUT_DIR / "history.csv", index=False)

    return model


# ══════════════════════════════════════════════════════════════
# 6. PRUNING
# ══════════════════════════════════════════════════════════════

def apply_pruning(model):
    amt = CONFIG["prune_amount"]
    print(f"\n{'='*70}")
    print(f"PRUNING — L1 Unstructured ({amt*100:.0f}%)")
    print(f"{'='*70}")

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            prune.l1_unstructured(module, "weight", amount=amt)
            prune.remove(module, "weight")
        elif isinstance(module, nn.GRU):
            for pname in list(module._parameters.keys()):
                if "weight" in pname and module._parameters[pname] is not None:
                    prune.l1_unstructured(module, pname, amount=amt)
                    prune.remove(module, pname)

    total  = sum(p.numel()          for p in model.parameters())
    zeros  = sum((p == 0).sum().item() for p in model.parameters())
    print(f"  Total params : {total:,}")
    print(f"  Zero params  : {zeros:,}")
    print(f"  Sparsity     : {100*zeros/total:.2f}%")

    torch.save(model.state_dict(), OUT_DIR / "tahirnet_pruned.pt")
    print(f"  Pruned model saved → tahirnet_pruned.pt")
    return model


# ══════════════════════════════════════════════════════════════
# 7. ENTRY POINT
# ══════════════════════════════════════════════════════════════

model = train()
model = apply_pruning(model)

print(f"\n{'='*70}")
print(f"  TahirNet Training Complete!")
print(f"  Outputs     : {OUT_DIR}")
print(f"{'='*70}")