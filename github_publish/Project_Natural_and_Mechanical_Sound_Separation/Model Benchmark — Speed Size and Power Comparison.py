# ==============================================================================
# ⚡ MODEL BENCHMARK — SPEED, SIZE, AND POWER COMPARISON SUITE
# 📌 Project: Natural and Mechanical Sound Separation (NatSep)
# 📄 Description: Automated comparison benchmark suite across HTDemucs, MDX-Net,
#                 SAM Audio DiT, Spectrogram U-Net, BS-RoFormer, & Hybrid U-DiT.
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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import IPython.display as ipd

# Auto-install missing packages
for pkg in ["demucs", "pandas", "matplotlib"]:
    try:
        __import__(pkg)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

import demucs.pretrained
from demucs.apply import apply_model

# ==============================================================================
# 🧠 MODEL ARCHITECTURES FOR BENCHMARKING
# ==============================================================================

# --- 1. Hybrid U-DiT ---
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
    def __init__(self, channels=256, depth=4):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=channels, nhead=8, dropout=0.1,
            dim_feedforward=1024, batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
    def forward(self, x, time_steps=None):
        B, C, H, W = x.shape
        x_flat = x.view(B, C, H*W).permute(0, 2, 1)
        out = self.transformer(x_flat)
        out = out.permute(0, 2, 1).view(B, C, H, W)
        return out

class UNetDecoder(nn.Module):
    def __init__(self, out_channels=1):
        super().__init__()
        self.upconv3 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.conv3a  = nn.Conv2d(256 + 256, 128, 3, padding=1)
        self.conv3b  = nn.Conv2d(128, 128, 3, padding=1)
        self.bn3     = nn.BatchNorm2d(128)
        self.upconv2 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.conv2a  = nn.Conv2d(128 + 128, 64, 3, padding=1)
        self.conv2b  = nn.Conv2d(64, 64, 3, padding=1)
        self.bn2     = nn.BatchNorm2d(64)
        self.upconv1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.conv1a  = nn.Conv2d(64 + 64, 32, 3, padding=1)
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
    def __init__(self):
        super().__init__()
        self.encoder      = UNetEncoder()
        self.vit_adapter  = ViTAdapter()
        self.dit_backbone = DiTBackbone()
        self.decoder      = UNetDecoder()
        
    def forward(self, x, time_steps=None):
        latent, s1, s2, s3 = self.encoder(x)
        latent = self.vit_adapter(latent)
        denoised_latent = self.dit_backbone(latent, time_steps)
        out = self.decoder(denoised_latent, s1, s2, s3)
        return out


# --- 2. Spectrogram U-Net ---
class UNetBlock(nn.Module):
    def __init__(self, in_c, out_c, down=True):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
        self.pool = nn.MaxPool2d(2) if down else None

    def forward(self, x):
        feat = self.conv(x)
        out = self.pool(feat) if self.pool is not None else feat
        return out, feat

class SpectrogramUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.enc1 = UNetBlock(in_channels, 32, down=True)
        self.enc2 = UNetBlock(32, 64, down=True)
        self.enc3 = UNetBlock(64, 128, down=True)
        self.bottleneck = UNetBlock(128, 256, down=False)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = UNetBlock(256, 128, down=False)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = UNetBlock(128, 64, down=False)
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = UNetBlock(64, 32, down=False)
        self.final_conv = nn.Sequential(nn.Conv2d(32, out_channels, kernel_size=1), nn.Sigmoid())

    def forward(self, x):
        x_p, s1 = self.enc1(x)
        x_p, s2 = self.enc2(x_p)
        x_p, s3 = self.enc3(x_p)
        b_feat, _ = self.bottleneck(x_p)
        d3 = torch.cat([self.up3(b_feat), s3], dim=1)
        d3_feat, _ = self.dec3(d3)
        d2 = torch.cat([self.up2(d3_feat), s2], dim=1)
        d2_feat, _ = self.dec2(d2)
        d1 = torch.cat([self.up1(d2_feat), s1], dim=1)
        d1_feat, _ = self.dec1(d1)
        return self.final_conv(d1_feat)


# --- 3. BS-RoFormer / Band-Split Attention ---
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
            nn.Sequential(nn.LayerNorm(b * 2), nn.Linear(b * 2, hidden), nn.GELU())
            for b in band_bins
        ])

    def forward(self, S_ri, slices):
        feats = []
        for proj, sl in zip(self.projs, slices):
            band = S_ri[:, sl, :, :]
            B, bins, T_bins, _ = band.shape
            flat = band.permute(0, 2, 1, 3).reshape(B, T_bins, bins * 2)
            feats.append(proj(flat))
        return torch.stack(feats, dim=1)

class IntraBandBiGRU(nn.Module):
    def __init__(self, hidden, layers=2, dropout=0.1):
        super().__init__()
        self.gru  = nn.GRU(hidden, hidden // 2, num_layers=layers, batch_first=True, bidirectional=True)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x):
        B, K, T_bins, H = x.shape
        out, _ = self.gru(x.reshape(B * K, T_bins, H))
        return self.norm(x + out.reshape(B, K, T_bins, H))

class InterBandAttention(nn.Module):
    def __init__(self, hidden, num_heads=4, dropout=0.1):
        super().__init__()
        self.attn  = nn.MultiheadAttention(hidden, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden)
        self.ff    = nn.Sequential(nn.Linear(hidden, hidden * 2), nn.GELU(), nn.Linear(hidden * 2, hidden))
        self.norm2 = nn.LayerNorm(hidden)

    def forward(self, x):
        B, K, T_bins, H = x.shape
        x_t = x.permute(0, 2, 1, 3).reshape(B * T_bins, K, H)
        attn_out, _ = self.attn(x_t, x_t, x_t)
        x_t = self.norm1(x_t + attn_out)
        x_t = self.norm2(x_t + self.ff(x_t))
        return x_t.reshape(B, T_bins, K, H).permute(0, 2, 1, 3)

class BandMaskDecoder(nn.Module):
    def __init__(self, band_bins, hidden):
        super().__init__()
        self.decoders = nn.ModuleList([nn.Sequential(nn.Linear(hidden, b * 2), nn.Tanh()) for b in band_bins])

    def forward(self, x, slices):
        masks = []
        for i, (dec, sl) in enumerate(zip(self.decoders, slices)):
            m = dec(x[:, i, :, :])
            B, T_bins, _ = m.shape
            bins = sl.stop - sl.start
            masks.append(m.reshape(B, T_bins, bins, 2).permute(0, 2, 1, 3))
        return masks

class BSRoFormerModel(nn.Module):
    def __init__(self, sr=44100, n_fft=2048, hop_length=512, hidden_dim=128):
        super().__init__()
        self.sr = sr
        self.n_fft = n_fft
        self.hop = hop_length
        self.hidden = hidden_dim
        self.slices = _band_slices([0, 300, 1000, 4000, 8000, 16000, 22050], n_fft, sr)
        self.band_bins = [sl.stop - sl.start for sl in self.slices]
        self.proj = BandSplitProjection(self.band_bins, self.hidden)
        self.blocks = nn.ModuleList([nn.ModuleList([IntraBandBiGRU(self.hidden), InterBandAttention(self.hidden)]) for _ in range(2)])
        self.decoder = BandMaskDecoder(self.band_bins, self.hidden)

    def forward(self, S_ri):
        x = self.proj(S_ri, self.slices)
        for gru, attn in self.blocks:
            x = gru(x)
            x = attn(x)
        masks = self.decoder(x, self.slices)
        mask_full = torch.zeros_like(S_ri)
        for sl, m in zip(self.slices, masks):
            mask_full[:, sl, :, :] = m
        return mask_full


# ==============================================================================
# 📊 BENCHMARK MEASUREMENT SUITE
# ==============================================================================

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

def estimate_model_size_mb(model):
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / (1024 ** 2)

def benchmark_model_performance(model_name, model_fn, test_audio_duration_sec=10.0, device="cuda"):
    print(f"\n⏳ Benchmarking Model: {model_name}...")
    sr = 44100
    num_samples = int(test_audio_duration_sec * sr)
    dummy_audio = torch.randn(1, 2, num_samples).to(device)

    if hasattr(model_fn, 'parameters'):
        params_m = count_parameters(model_fn)
        size_mb = estimate_model_size_mb(model_fn)
    else:
        params_m = 0.0
        size_mb = 0.0

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()

    try:
        model_fn(dummy_audio)
    except Exception:
        pass

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    runs = 5
    start_time = time.perf_counter()
    with torch.no_grad():
        for _ in range(runs):
            _ = model_fn(dummy_audio)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
    end_time = time.perf_counter()

    avg_latency = (end_time - start_time) / runs
    rtf = avg_latency / test_audio_duration_sec
    speedup = 1.0 / rtf if rtf > 0 else 0.0

    if torch.cuda.is_available():
        peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    else:
        peak_vram_mb = 0.0

    return {
        "Model Name": model_name,
        "Parameters (M)": round(params_m, 2),
        "Weight Size (MB)": round(size_mb, 2),
        "Inference Latency (s)": round(avg_latency, 3),
        "RTF (Lower is Faster)": round(rtf, 4),
        "Speedup (x Real-Time)": round(speedup, 1),
        "Peak VRAM (MB)": round(peak_vram_mb, 2)
    }

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print(f"🚀 COMPREHENSIVE MULTI-MODEL EDGE BENCHMARK SUITE (Device: {device})")
    print("=" * 70)

    results = []

    # 1. Meta HTDemucs ('htdemucs_ft')
    try:
        print("\n📥 Loading Meta Pretrained HTDemucs ('htdemucs_ft')...")
        htdemucs_model = demucs.pretrained.get_model('htdemucs_ft').to(device)
        res1 = benchmark_model_performance("Meta HTDemucs (htdemucs_ft)", htdemucs_model, test_audio_duration_sec=10.0, device=device)
        results.append(res1)
    except Exception as e:
        print(f"⚠️ Error benchmarking HTDemucs: {e}")

    # 2. Meta MDX-Net ('mdx_extra')
    try:
        print("\n📥 Loading Meta Pretrained MDX-Net ('mdx_extra')...")
        mdx_model = demucs.pretrained.get_model('mdx_extra').to(device)
        res2 = benchmark_model_performance("Meta MDX-Net (mdx_extra)", mdx_model, test_audio_duration_sec=10.0, device=device)
        results.append(res2)
    except Exception as e:
        print(f"⚠️ Error benchmarking MDX-Net: {e}")

    # 3. Spectrogram U-Net
    try:
        print("\n📥 Initializing Spectrogram U-Net Architecture...")
        unet_model = SpectrogramUNet().to(device)
        mel_trans = T.MelSpectrogram(sample_rate=44100, n_fft=2048, hop_length=512, n_mels=128).to(device)
        def run_unet(x):
            mono = x.mean(dim=1, keepdim=True)
            mel = mel_trans(mono)
            return unet_model((mel + 80.0)/80.0)
        res3 = benchmark_model_performance("Spectrogram U-Net", unet_model, test_audio_duration_sec=10.0, device=device)
        results.append(res3)
    except Exception as e:
        print(f"⚠️ Error benchmarking Spectrogram U-Net: {e}")

    # 4. BS-RoFormer / Band-Split Attention
    try:
        print("\n📥 Initializing BS-RoFormer (Band-Split Attention) Architecture...")
        bs_model = BSRoFormerModel().to(device)
        def run_bsroformer(x):
            mono = x.mean(dim=1)
            spec = torch.stft(mono, n_fft=2048, hop_length=512, return_complex=True)
            S_ri = torch.stack([spec.real, spec.imag], dim=-1).unsqueeze(0)
            return bs_model(S_ri)
        res4 = benchmark_model_performance("BS-RoFormer (Band-Split)", bs_model, test_audio_duration_sec=10.0, device=device)
        results.append(res4)
    except Exception as e:
        print(f"⚠️ Error benchmarking BS-RoFormer: {e}")

    # 5. Hybrid U-DiT
    try:
        print("\n📥 Initializing Hybrid U-DiT Architecture...")
        udit_model = HybridUDiT().to(device)
        def run_udit(x):
            mono = x.mean(dim=1, keepdim=True)
            mel = mel_trans(mono)
            mel_norm = (mel + 80.0) / 80.0
            T_crop = mel_norm.shape[-1] - (mel_norm.shape[-1] % 8)
            return udit_model(mel_norm[..., :T_crop], torch.zeros(1, dtype=torch.long, device=device))
        res5 = benchmark_model_performance("Hybrid U-DiT (Edge SOTA)", udit_model, test_audio_duration_sec=10.0, device=device)
        results.append(res5)
    except Exception as e:
        print(f"⚠️ Error benchmarking Hybrid U-DiT: {e}")

    # Summary Table
    df = pd.DataFrame(results)
    print("\n" + "=" * 70)
    print("📊 MULTI-MODEL BENCHMARK COMPARISON TABLE")
    print("=" * 70)
    print(df.to_string(index=False))

    csv_file = "model_benchmark_results.csv"
    df.to_csv(csv_file, index=False)
    print(f"\n💾 Saved detailed metrics CSV to: {csv_file}")

    # Plot Comparative Bar Charts
    plt.style.use("ggplot")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("⚡ All Sound Separation Models Benchmark Comparison", fontsize=16, fontweight="bold")

    models = df["Model Name"].tolist()
    colors = ["#3498db", "#e74c3c", "#f39c12", "#1abc9c", "#2ecc71"][:len(models)]

    axes[0, 0].bar(models, df["Speedup (x Real-Time)"], color=colors)
    axes[0, 0].set_title("🚀 Speedup (x Real-Time - Higher is Better)", fontweight="bold")
    axes[0, 0].tick_params(axis='x', rotation=20)

    axes[0, 1].bar(models, df["RTF (Lower is Faster)"], color=colors)
    axes[0, 1].set_title("⏱️ Real-Time Factor (RTF - Lower is Better)", fontweight="bold")
    axes[0, 1].tick_params(axis='x', rotation=20)

    axes[1, 0].bar(models, df["Parameters (M)"], color=colors)
    axes[1, 0].set_title("📦 Parameters (Million - Lower is Lighter)", fontweight="bold")
    axes[1, 0].set_ylabel("Parameters (M)")
    axes[1, 0].tick_params(axis='x', rotation=20)

    axes[1, 1].bar(models, df["Peak VRAM (MB)"], color=colors)
    axes[1, 1].set_title("🔋 Peak GPU VRAM (MB - Lower is Lighter)", fontweight="bold")
    axes[1, 1].set_ylabel("VRAM (MB)")
    axes[1, 1].tick_params(axis='x', rotation=20)

    plt.tight_layout()
    chart_filename = "model_efficiency_benchmark.png"
    plt.savefig(chart_filename, dpi=300)
    print(f"📊 Saved Benchmark Chart to: {chart_filename}")
    ipd.display(ipd.Image(filename=chart_filename))

if __name__ == "__main__":
    main()

# ==============================================================================
# 🏁 END OF SCRIPT — MODEL BENCHMARK COMPARISON SUITE
# ==============================================================================
