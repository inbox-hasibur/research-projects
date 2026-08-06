# ============================================================
# MDXNet2 — Complete Evaluation + Demo Pipeline
# Architecture: TFC-TDF U-Net (MDX-Net Base)
# ============================================================

import os, sys, time, warnings
warnings.filterwarnings('ignore')

# Install required packages
import subprocess
for pkg in ["gradio", "pesq", "pystoi", "librosa", "soundfile", "nest_asyncio"]:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

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
from torch.amp import autocast

from pesq import pesq
from pystoi import stoi

# ─── Device Setup ───────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_GPUS = torch.cuda.device_count()
print(f"✅ Device: {DEVICE} | GPUs: {N_GPUS}")

# ─── Paths ──────────────────────────────────────────────────
MODEL_PATH  = Path("/kaggle/input/models/inboxhasibur/mdxnet2/pytorch/default/1/mdxnet2_best")
DATA_DIR    = Path("/kaggle/input/datasets/inboxhasibur/mecnature-audio-dataset")
OUTPUT_DIR  = Path("/kaggle/working/mdxnet2_eval")
DEMO_DIR    = OUTPUT_DIR / "demo_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DEMO_DIR.mkdir(parents=True, exist_ok=True)

# ─── Config (must match training exactly) ───────────────────
CONFIG = {
    "sr"              : 22050,
    "clip_length"     : 4,
    "n_fft"           : 1024,
    "hop_length"      : 256,
    "channels"        : 24,
    "num_blocks"      : 4,
}

SAMPLES = CONFIG["sr"] * CONFIG["clip_length"]


# ══════════════════════════════════════════════════════════════
# 1. MODEL DEFINITION (Must match training structure)
# ══════════════════════════════════════════════════════════════

class TFCBlock(nn.Module):
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
    def __init__(self, channels, freq_bins):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(freq_bins, freq_bins // 2, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Linear(freq_bins // 2, freq_bins, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = x.transpose(-1, -2)
        out = self.fc(out)
        out = out.transpose(-1, -2)
        return x * out

class MDXNet2(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_fft = CONFIG["n_fft"]
        self.hop_length = CONFIG["hop_length"]
        self.register_buffer("window", torch.hann_window(self.n_fft))
        
        freq_bins = self.n_fft // 2 + 1
        base_ch = CONFIG["channels"]
        self.num_blocks = CONFIG["num_blocks"]
        
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.tdfs     = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        self.upsamples   = nn.ModuleList()
        
        in_c = 2 
        
        # Encoder
        for i in range(self.num_blocks):
            out_c = base_ch * (2 ** i)
            self.encoders.append(TFCBlock(in_c, out_c))
            self.downsamples.append(nn.Conv2d(out_c, out_c, kernel_size=(2,2), stride=(2,2)))
            in_c = out_c
            freq_bins = freq_bins // 2
            
        # Bottleneck
        out_c = base_ch * (2 ** self.num_blocks)
        self.bottleneck_tfc = TFCBlock(in_c, out_c)
        self.bottleneck_tdf = TDFBlock(out_c, freq_bins)
        
        # Decoder
        for i in range(self.num_blocks - 1, -1, -1):
            dec_in_c = out_c + base_ch * (2 ** i)
            dec_out_c = base_ch * (2 ** i)
            
            self.upsamples.append(nn.ConvTranspose2d(out_c, base_ch * (2 ** i), kernel_size=(2,2), stride=(2,2)))
            self.tdfs.append(TDFBlock(dec_in_c, freq_bins * 2))
            self.decoders.append(TFCBlock(dec_in_c, dec_out_c))
            
            out_c = dec_out_c
            freq_bins = freq_bins * 2
            
        self.final_conv = nn.Conv2d(base_ch, 2, kernel_size=3, padding=1)
        
    def stft(self, x):
        X = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop_length, 
                       window=self.window, return_complex=True)
        X_ri = torch.stack([X.real, X.imag], dim=1)
        return X_ri
        
    def istft(self, X_ri, length):
        X = torch.complex(X_ri[:, 0], X_ri[:, 1])
        x = torch.istft(X, n_fft=self.n_fft, hop_length=self.hop_length, 
                        window=self.window, length=length)
        return x

    def forward(self, x):
        length = x.shape[-1]
        X_ri = self.stft(x)
        orig_shape = X_ri.shape
        
        pad_f = (2 ** self.num_blocks) - (orig_shape[2] % (2 ** self.num_blocks))
        pad_t = (2 ** self.num_blocks) - (orig_shape[3] % (2 ** self.num_blocks))
        pad_f = 0 if pad_f == (2 ** self.num_blocks) else pad_f
        pad_t = 0 if pad_t == (2 ** self.num_blocks) else pad_t
        
        if pad_f > 0 or pad_t > 0:
            X_ri = F.pad(X_ri, (0, pad_t, 0, pad_f))
            
        skips = []
        out = X_ri
        
        for i in range(self.num_blocks):
            out = self.encoders[i](out)
            skips.append(out)
            out = self.downsamples[i](out)
            
        out = self.bottleneck_tfc(out)
        out = self.bottleneck_tdf(out)
        
        for i in range(self.num_blocks):
            out = self.upsamples[i](out)
            skip = skips[-(i+1)]
            if out.shape[-2:] != skip.shape[-2:]:
                out = F.interpolate(out, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            out = torch.cat([out, skip], dim=1)
            out = self.tdfs[i](out)
            out = self.decoders[i](out)
            
        mask = self.final_conv(out)
        if pad_f > 0 or pad_t > 0:
            mask = mask[:, :, :orig_shape[2], :orig_shape[3]]
            
        Y_ri = X_ri[:, :, :orig_shape[2], :orig_shape[3]] * mask
        y = self.istft(Y_ri, length)
        return y

# ══════════════════════════════════════════════════════════════
# 2. MODEL LOADER
# ══════════════════════════════════════════════════════════════

def load_model(model_path):
    print(f"\n{'='*60}")
    print(f"📦 Loading MDXNet2 from:")
    print(f"   {model_path}")
    
    model = MDXNet2().to(DEVICE)
    
    pt_candidates = [
        model_path,
        model_path / "mdxnet2_best.pt",
        model_path.parent / "mdxnet2_best.pt",
    ]
    
    loaded = False
    for candidate in pt_candidates:
        if Path(str(candidate) + ".pt").exists():
            candidate = Path(str(candidate) + ".pt")
        
        if Path(candidate).exists():
            if Path(candidate).is_file():
                try:
                    state = torch.load(candidate, map_location=DEVICE, weights_only=False)
                    if isinstance(state, dict):
                        if 'model_state' in state: state = state['model_state']
                    
                    new_state = {}
                    for k, v in state.items():
                        new_state[k.replace('module.', '')] = v
                        
                    model.load_state_dict(new_state, strict=False)
                    loaded = True
                    print(f"   ✅ Successfully loaded: {candidate.name}")
                    break
                except Exception as e:
                    print(f"   ⚠️  Failed {candidate}: {e}")
                    continue
    
    if not loaded:
        # Handle Kaggle unpacked directories
        if Path(model_path).is_dir() and (Path(model_path) / "data.pkl").exists():
            print(f"   📦 Detected unpacked PyTorch checkpoint. Re-zipping on the fly...")
            import zipfile
            import tempfile
            
            tmp_zip = Path(tempfile.gettempdir()) / "temp_mdxnet_ckpt.zip"
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
                    if 'model_state' in state: state = state['model_state']
                
                new_state = {}
                for k, v in state.items():
                    new_state[k.replace('module.', '')] = v
                
                model.load_state_dict(new_state, strict=False)
                loaded = True
                print(f"   ✅ Successfully loaded unpacked model via temp zip!")
            except Exception as e:
                print(f"   ⚠️  Failed loading re-zipped model: {e}")
    
    if not loaded:
        print("   ⚠️  Could not load weights — using random init")
        print("   📌 Tip: Make sure MODEL_PATH is correct")
        
    n_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {n_params:,}")
    print(f"{'='*60}\n")
    
    if N_GPUS > 1:
        print(f"  ✅ DataParallel: {N_GPUS} GPUs\n")
        model = nn.DataParallel(model)
        
    model.eval()
    return model


# ══════════════════════════════════════════════════════════════
# 3. EVALUATION METRICS
# ══════════════════════════════════════════════════════════════

def calc_metrics(est, tgt, mix):
    est_np = est.cpu().numpy()
    tgt_np = tgt.cpu().numpy()
    mix_np = mix.cpu().numpy()

    # PESQ and STOI
    try:
        p_score = pesq(CONFIG["sr"], tgt_np, est_np, 'wb')
    except:
        p_score = np.nan
        
    try:
        s_score = stoi(tgt_np, est_np, CONFIG["sr"], extended=False)
    except:
        s_score = np.nan

    # SI-SDR
    def si_sdr(e, t, eps=1e-8):
        t_energy = np.sum(t ** 2) + eps
        scale = np.sum(t * e) / t_energy
        proj = scale * t
        noise = e - proj
        return 10 * np.log10((np.sum(proj ** 2) + eps) / (np.sum(noise ** 2) + eps))

    sdr_val = si_sdr(est_np, tgt_np)
    mix_sdr = si_sdr(mix_np, tgt_np)

    # SDR
    def sdr_standard(e, t, eps=1e-8):
        noise = t - e
        return 10 * np.log10((np.sum(t ** 2) + eps) / (np.sum(noise ** 2) + eps))
        
    std_sdr = sdr_standard(est_np, tgt_np)
    std_mix_sdr = sdr_standard(mix_np, tgt_np)

    return {
        "SI-SDR" : sdr_val,
        "SDR"    : std_sdr,
        "SNR"    : std_sdr,
        "SI-SDRi": sdr_val - mix_sdr,
        "SDRi"   : std_sdr - std_mix_sdr,
        "PESQ"   : p_score,
        "STOI"   : s_score
    }


def evaluate_test_set(model):
    print(f"{'='*60}")
    print("🧪 FULL TEST SET EVALUATION")
    print(f"{'='*60}")
    
    test_mix = sorted((DATA_DIR / "test/mixture").glob("*.wav"))
    test_tgt = sorted((DATA_DIR / "test/natural").glob("*.wav"))
    
    if len(test_mix) == 0:
        print("  ⚠️ No test files found!")
        return
        
    print(f"  [test] Found {len(test_mix)} samples")
    results = []
    
    for m_path, t_path in tqdm(zip(test_mix, test_tgt), total=len(test_mix), desc="Evaluating"):
        mix, _ = torchaudio.load(str(m_path))
        tgt, _ = torchaudio.load(str(t_path))
        
        mix = mix.mean(0)
        tgt = tgt.mean(0)
        
        if mix.shape[-1] < SAMPLES:
            mix = F.pad(mix, (0, SAMPLES - mix.shape[-1]))
            tgt = F.pad(tgt, (0, SAMPLES - tgt.shape[-1]))
        else:
            mix = mix[:SAMPLES]
            tgt = tgt[:SAMPLES]
            
        mix_t = mix.unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            with autocast('cuda'):
                est_t = model(mix_t)
        est = est_t.squeeze(0).cpu()
        
        m_dict = calc_metrics(est, tgt, mix)
        m_dict['file'] = m_path.name
        results.append(m_dict)
        
    df = pd.DataFrame(results)
    
    print(f"\n{'='*60}")
    print("📊 TEST RESULTS SUMMARY")
    print(f"{'='*60}")
    
    for k in ['SI-SDR', 'SDR', 'SNR', 'SI-SDRi', 'SDRi']:
        val = df[k].dropna().mean()
        std = df[k].dropna().std()
        print(f"  {k:<19}: {val:+7.3f} ± {std:6.3f} dB")
        
    pesq_val = df['PESQ'].dropna().mean()
    pesq_std = df['PESQ'].dropna().std()
    print(f"  PESQ                : {pesq_val:7.4f} ± {pesq_std:6.4f}")
    
    stoi_val = df['STOI'].dropna().mean()
    stoi_std = df['STOI'].dropna().std()
    print(f"  STOI                : {stoi_val:7.4f} ± {stoi_std:6.4f}")
    
    print(f"\n  Total samples evaluated: {len(df)}")
    
    csv_path = OUTPUT_DIR / "test_metrics.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n  💾 Saved: {csv_path}")
    
    plot_path = OUTPUT_DIR / "test_results.png"
    plot_results(df, plot_path)
    print(f"\n✅ Plot saved: {plot_path}")


def plot_results(df, save_path):
    fig = plt.figure(figsize=(15, 12))
    gs = gridspec.GridSpec(2, 3, height_ratios=[1, 1], figure=fig)
    fig.patch.set_facecolor('#1e1e2f')
    
    for ax in fig.axes:
        ax.set_facecolor('#1e1e2f')
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_color('#444')

    plt.suptitle("MDXNet2 Performance Summary", color="white", fontsize=18, y=0.98)
    
    ax1 = fig.add_subplot(gs[0, 0])
    vals = df['SI-SDRi'].dropna()
    ax1.hist(vals, bins=30, color='dodgerblue', edgecolor='white', alpha=0.8)
    ax1.set_title('SI-SDRi Distribution (dB)', color='white')
    ax1.tick_params(colors='white')
    
    plt.tight_layout()
    plt.savefig(save_path, facecolor=fig.get_facecolor(), dpi=150)
    plt.close()


# ══════════════════════════════════════════════════════════════
# 4. LIVE DEMO (Gradio Web UI)
# ══════════════════════════════════════════════════════════════

def build_gradio_ui(model: nn.Module):
    import gradio as gr
    
    def separate_audio(mix_audio, has_target, tgt_audio):
        if mix_audio is None:
            return None, None, "⚠️ Please upload a mixture audio file."
            
        sr, mix_np = mix_audio
        if mix_np.dtype == np.int16:
            mix_np = mix_np.astype(np.float32) / 32768.0
        
        if len(mix_np.shape) > 1:
            mix_np = mix_np.mean(axis=1) # stereo to mono
            
        original_length = len(mix_np)
        mix_t = torch.tensor(mix_np).unsqueeze(0).to(DEVICE)
        
        # Padding for STFT
        pad_len = 0
        if mix_t.shape[-1] < SAMPLES:
            pad_len = SAMPLES - mix_t.shape[-1]
            mix_t = F.pad(mix_t, (0, pad_len))
            
        with torch.no_grad():
            with autocast('cuda'):
                est_t = model(mix_t)
                
        est_np = est_t.squeeze(0).cpu().numpy()
        
        if pad_len > 0:
            est_np = est_np[:-pad_len]
            
        out_wav = (sr, np.clip(est_np, -1.0, 1.0))
        metrics_msg = "✅ Separated successfully!"
        
        if has_target and tgt_audio is not None:
            tgt_sr, tgt_np = tgt_audio
            if tgt_np.dtype == np.int16:
                tgt_np = tgt_np.astype(np.float32) / 32768.0
            if len(tgt_np.shape) > 1:
                tgt_np = tgt_np.mean(axis=1)
                
            tgt_np = tgt_np[:original_length]
            mix_np = mix_np[:original_length]
            est_np = est_np[:original_length]
            
            m = calc_metrics(torch.tensor(est_np), torch.tensor(tgt_np), torch.tensor(mix_np))
            metrics_msg = (
                f"**SI-SDR:** {m['SI-SDR']:.2f} dB\n"
                f"**SI-SDRi:** {m['SI-SDRi']:.2f} dB\n"
                f"**PESQ:** {m['PESQ']:.2f}\n"
                f"**STOI:** {m['STOI']:.2f}"
            )
            
        return out_wav, None, metrics_msg

    with gr.Blocks(theme=gr.themes.Base()) as demo:
        gr.Markdown("# 🎵 MDXNet2 — Natural & Mechanical Sound Separation")
        
        with gr.Row():
            with gr.Column():
                mix_in = gr.Audio(label="Upload Mixture", type="numpy")
                has_tgt = gr.Checkbox(label="I have Ground Truth Target")
                tgt_in = gr.Audio(label="Upload Target (Optional)", type="numpy", visible=False)
                sep_btn = gr.Button("Separate Audio", variant="primary")
            
            with gr.Column():
                audio_out = gr.Audio(label="Separated Output", type="numpy")
                metrics_out = gr.Markdown("Waiting for input...")
                
        def toggle_tgt(checked):
            return gr.update(visible=checked)
            
        has_tgt.change(toggle_tgt, inputs=has_tgt, outputs=tgt_in)
        sep_btn.click(separate_audio, inputs=[mix_in, has_tgt, tgt_in], outputs=[audio_out, None, metrics_out])
        
    return demo


# ══════════════════════════════════════════════════════════════
# MAIN ENTRY
# ══════════════════════════════════════════════════════════════

def main():
    print(f"{'='*60}")
    print("  MDXNet2 — Test & Demo Pipeline")
    print(f"{'='*60}")
    
    model = load_model(MODEL_PATH)
    
    evaluate_test_set(model)
    
    print(f"\n{'='*60}")
    print("🌐 DEMO TYPE 2 — Gradio Web UI")
    print(f"{'='*60}")
    
    demo = build_gradio_ui(model)
    demo.launch(share=True, show_error=True, quiet=False)

if __name__ == "__main__":
    main()
