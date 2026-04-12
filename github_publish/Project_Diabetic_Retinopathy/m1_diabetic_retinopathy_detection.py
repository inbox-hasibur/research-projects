# =============================================================================
# STEP 01: Install Dependencies 
# =============================================================================
# !pip install -q timm scikit-learn matplotlib seaborn pytorch-grad-cam

# =============================================================================
# STEP 02: Imports
# =============================================================================
import os
import sys
import gc
import time
import math
import random
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend → no GUI hang on Kaggle
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import timm
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    cohen_kappa_score,
    precision_recall_curve,
    average_precision_score,
)

warnings.filterwarnings("ignore")

# ── helper: always flush so Kaggle shows output immediately ──────────────────
def _print(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()

# Global plot style
sns.set_style("whitegrid")
plt.rcParams.update({
    "figure.dpi": 80,       # lower DPI → smaller memory footprint
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
})

_print("✓ Imports complete")

# =============================================================================
# STEP 03: Configuration
# =============================================================================
@dataclass
class CFG:
    """Master configuration for the entire pipeline."""

    # -- Reproducibility --
    seed: int = 42

    # -- Data & Model --
    img_size: int = 224
    num_classes: int = 5
    class_names: Tuple[str, ...] = ("No DR", "Mild", "Moderate", "Severe", "Proliferative")

    # -- Training --
    # SPEED-FIX: batch_size=64 → 115K/64 = 1,801 batches/epoch (was 14,405 at bs=8)
    # 2× T4 with 14.6GB each → plenty of VRAM for bs=64 without gradient checkpointing
    batch_size: int = 64
    accumulate_grad_batches: int = 1       # no accumulation needed; effective batch=64
    # Cap steps per epoch so 20 epochs fit in Kaggle 12hr session
    # 1800 steps × ~24s(avg) ≈ 43200s/20 = 2160s/ep → ~12hr total
    max_steps_per_epoch: Optional[int] = 1800
    epochs: int = 20
    warmup_epochs: int = 2                 # linear warmup before cosine decay
    lr_backbone: float = 5e-6             # lower backbone LR → better fine-tune
    lr_new: float = 5e-5                  # lower head LR → less divergence
    weight_decay: float = 1e-4
    label_smoothing: float = 0.1
    patience: int = 7                     # more patience → full training

    # -- Paths --
    num_workers: int = 4                  # 4 workers → GPU never starves on 115K dataset
    pin_memory: bool = True
    prefetch_factor: int = 2              # SPEED-FIX: 2 batches pre-fetched → better GPU feeding
    save_path: str = "best_m1_efficientnetv2_cbam_gem.pth"
    checkpoint_path: str = "checkpoint_m1_full.pth"

    # -- Kaggle Dataset Candidates (auto-resolved) --
    candidates: Tuple[Tuple[str, str, str], ...] = (
        (
            "/kaggle/input/datasets/ascanipek/eyepacs-aptos-messidor-diabetic-retinopathy/augmented_resized_V2/train",
            "/kaggle/input/datasets/ascanipek/eyepacs-aptos-messidor-diabetic-retinopathy/augmented_resized_V2/val",
            "/kaggle/input/datasets/ascanipek/eyepacs-aptos-messidor-diabetic-retinopathy/augmented_resized_V2/test",
        ),
        (
            "/kaggle/input/datasets/ascanipek/eyepacs-aptos-messidor-diabetic-retinopathy/dr_unified_v2/dr_unified_v2/train",
            "/kaggle/input/datasets/ascanipek/eyepacs-aptos-messidor-diabetic-retinopathy/dr_unified_v2/dr_unified_v2/val",
            "/kaggle/input/datasets/ascanipek/eyepacs-aptos-messidor-diabetic-retinopathy/dr_unified_v2/dr_unified_v2/test",
        ),
    )

    # -- Normalization (ImageNet) --
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)

    # -- Robustness Testing --
    noise_sigmas: Tuple[float, ...] = (0.0, 0.05, 0.10, 0.20)
    brightness_contrast_pairs: Tuple[Tuple[float, float], ...] = ((0.7, 0.8), (1.3, 1.2))
    fgsm_eps: float = 0.02

    # -- Device --
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


cfg = CFG()
CLASS_COLORS = ["#2ecc71", "#3498db", "#f39c12", "#e67e22", "#e74c3c"]

# AMP enabled flag
USE_AMP = cfg.device.type == "cuda"

_print("✓ Configuration set")
_print(f"  Classes    : {cfg.class_names}")
_print(f"  Image size : {cfg.img_size}x{cfg.img_size}")
effective_bs = cfg.batch_size * cfg.accumulate_grad_batches
_print(f"  Batch size : {cfg.batch_size}  (effective {effective_bs} | accum_steps={cfg.accumulate_grad_batches})")
_print(f"  Epochs     : {cfg.epochs}")

# =============================================================================
# STEP 04: Reproducibility & Runtime Info
# =============================================================================
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

set_seed(cfg.seed)

_print("\n" + "=" * 70)
_print("RUNTIME CONFIGURATION")
_print("=" * 70)
_print(f"  Seed              : {cfg.seed}")
_print(f"  Device            : {cfg.device}")
_print(f"  CUDA available    : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    _print(f"  GPU count         : {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        total_gb = props.total_memory / 1024**3
        _print(f"  GPU {i}             : {props.name} ({total_gb:.1f} GB)")
_print("=" * 70)

# =============================================================================
# STEP 05: Dataset Discovery & Loading
# =============================================================================
def is_valid_split(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    return all(os.path.isdir(os.path.join(path, str(i))) for i in range(cfg.num_classes))

def resolve_splits() -> Tuple[str, str, str]:
    for tr, va, te in cfg.candidates:
        if is_valid_split(tr) and is_valid_split(va) and is_valid_split(te):
            return tr, va, te

    _print("Warning: Candidates not found. Creating dummy data structure for testing.")
    for split in ["train", "val", "test"]:
        for c in range(cfg.num_classes):
            os.makedirs(f"./dummy_data/{split}/{c}", exist_ok=True)
            img = Image.new("RGB", (100, 100), color=(73, 109, 137))
            img.save(f"./dummy_data/{split}/{c}/dummy_1.jpg")
            img.save(f"./dummy_data/{split}/{c}/dummy_2.jpg")

    return "./dummy_data/train", "./dummy_data/val", "./dummy_data/test"

def load_paths_labels(split_dir: str) -> Tuple[List[str], List[int]]:
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    paths, labels = [], []
    for c in range(cfg.num_classes):
        c_dir = os.path.join(split_dir, str(c))
        for fn in sorted(os.listdir(c_dir)):
            if fn.lower().endswith(exts):
                paths.append(os.path.join(c_dir, fn))
                labels.append(c)
    return paths, labels

train_dir, val_dir, test_dir = resolve_splits()
X_train, y_train = load_paths_labels(train_dir)
X_val,   y_val   = load_paths_labels(val_dir)
X_test,  y_test  = load_paths_labels(test_dir)

_print("✓ Dataset paths resolved:")
_print(f"  Train : {train_dir}")
_print(f"  Val   : {val_dir}")
_print(f"  Test  : {test_dir}")
_print(f"\n  Train : {len(X_train):,} images")
_print(f"  Val   : {len(X_val):,} images")
_print(f"  Test  : {len(X_test):,} images")

# =============================================================================
# STEP 06: Data Analysis (Class Distribution & Statistics)
# =============================================================================
def class_counts(y: List[int]) -> np.ndarray:
    return np.bincount(np.array(y), minlength=cfg.num_classes)

cnt_tr = class_counts(y_train)
cnt_va = class_counts(y_val)
cnt_te = class_counts(y_test)

imbalance_ratio = cnt_tr.max() / max(cnt_tr.min(), 1)

_print("\n" + "=" * 70)
_print("CLASS DISTRIBUTION ANALYSIS")
_print("=" * 70)
_print(f"{'Class':<16} {'Train':>8} {'Val':>8} {'Test':>8}  {'Train%':>8}")
_print("-" * 70)
for i, name in enumerate(cfg.class_names):
    pct = (cnt_tr[i] / cnt_tr.sum() * 100) if cnt_tr.sum() > 0 else 0
    _print(f"  {name:<14} {cnt_tr[i]:>8,} {cnt_va[i]:>8,} {cnt_te[i]:>8,}  {pct:>7.2f}%")
_print("-" * 70)
_print(f"  {'TOTAL':<14} {cnt_tr.sum():>8,} {cnt_va.sum():>8,} {cnt_te.sum():>8,}")
_print(f"\n  Imbalance ratio (max/min train): {imbalance_ratio:.2f}x")
_print("=" * 70)

# =============================================================================
# STEP 07: Data Visualization: Distribution & Resolution
# =============================================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("M1 - Dataset Analysis", fontsize=14, fontweight="bold")

x = np.arange(cfg.num_classes)
w = 0.25
b1 = axes[0].bar(x - w, cnt_tr, width=w, label="Train", color="#3498db", alpha=0.87, edgecolor="white")
b2 = axes[0].bar(x,     cnt_va, width=w, label="Val",   color="#2ecc71", alpha=0.87, edgecolor="white")
b3 = axes[0].bar(x + w, cnt_te, width=w, label="Test",  color="#e74c3c", alpha=0.87, edgecolor="white")
axes[0].set_title("Class Distribution Across Splits")
axes[0].set_xticks(x)
axes[0].set_xticklabels(cfg.class_names, rotation=15, ha="right", fontsize=9)
axes[0].set_ylabel("Image Count")
axes[0].legend(fontsize=9)
axes[0].grid(alpha=0.3, axis="y")

for bar in list(b1) + list(b2) + list(b3):
    h = bar.get_height()
    if h > 0:
        axes[0].text(bar.get_x() + bar.get_width() / 2, h + (h * 0.02), f"{int(h):,}",
                     ha="center", va="bottom", fontsize=6.5)

if cnt_tr.sum() > 0:
    axes[1].pie(
        cnt_tr, labels=cfg.class_names, colors=CLASS_COLORS,
        autopct="%1.1f%%", startangle=90, pctdistance=0.78,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
    )
axes[1].set_title("Train Set Class Proportion")

size_samples = random.sample(X_train, k=min(100, len(X_train)))  # MEM-OPT: 100 is enough
if size_samples:
    wh = []
    for _sp in size_samples:
        try:
            with Image.open(_sp) as _im:
                wh.append(_im.size)
        except Exception:
            pass
    if wh:
        ws, hs = [s[0] for s in wh], [s[1] for s in wh]
        axes[2].scatter(ws, hs, s=10, alpha=0.5, color="steelblue", edgecolors="none")
axes[2].set_title("Raw Image Resolution Scatter")
axes[2].set_xlabel("Width (px)")
axes[2].set_ylabel("Height (px)")
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("01_data_distribution.png", dpi=80, bbox_inches="tight")
plt.close(fig)
_print("✓ Saved: 01_data_distribution.png")

# =============================================================================
# STEP 08: Data Visualization: Sample Grid per Class
# =============================================================================
def show_class_samples(paths: List[str], labels: List[int], n_per_class: int = 5) -> None:
    if not paths:
        return
    fig, axes = plt.subplots(
        cfg.num_classes, n_per_class,
        figsize=(2.5 * n_per_class, 2.2 * cfg.num_classes)
    )
    for c in range(cfg.num_classes):
        idxs = [i for i, y in enumerate(labels) if y == c]
        picks = random.sample(idxs, k=min(len(idxs), n_per_class))
        for j in range(n_per_class):
            ax = axes[c][j] if cfg.num_classes > 1 else axes[j]
            ax.axis("off")
            if j < len(picks):
                try:
                    with Image.open(paths[picks[j]]) as _raw:
                        img = _raw.convert("RGB")
                    ax.imshow(img)
                    del img
                except Exception:
                    pass
                if j == 0:
                    ax.set_ylabel(f"Grade {c}\n{cfg.class_names[c]}", fontsize=9,
                                  fontweight="bold", rotation=0, labelpad=50, va="center")
            if c == 0:
                ax.set_title(f"Sample {j+1}", fontsize=9)

    fig.suptitle("Train Samples by DR Grade (5-class)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("02_sample_grid.png", dpi=80, bbox_inches="tight")
    plt.close(fig)
    _print("✓ Saved: 02_sample_grid.png")

show_class_samples(X_train, y_train, n_per_class=5)

# =============================================================================
# STEP 09: Data Visualization: Augmentation Pipeline Preview
# =============================================================================
train_tfms = transforms.Compose([
    transforms.Resize((cfg.img_size + 24, cfg.img_size + 24)),
    transforms.RandomCrop(cfg.img_size),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.2)),
    transforms.ToTensor(),
    transforms.Normalize(cfg.mean, cfg.std),
    transforms.RandomErasing(p=0.15, scale=(0.02, 0.10)),
])

val_tfms = transforms.Compose([
    transforms.Resize((cfg.img_size, cfg.img_size)),
    transforms.ToTensor(),
    transforms.Normalize(cfg.mean, cfg.std),
])

def denorm(t: torch.Tensor) -> np.ndarray:
    img = t.permute(1, 2, 0).numpy()
    img = img * np.array(cfg.std) + np.array(cfg.mean)
    return np.clip(img, 0, 1)

if X_train:
    sample_pil = Image.open(X_train[0]).convert("RGB")
    n_aug = 5

    fig, axes = plt.subplots(1, n_aug + 1, figsize=(3.2 * (n_aug + 1), 3.2))
    fig.suptitle("Augmentation Pipeline Preview", fontsize=14, fontweight="bold")

    axes[0].imshow(sample_pil.resize((cfg.img_size, cfg.img_size)))
    axes[0].set_title("Original", fontsize=10, fontweight="bold")
    axes[0].axis("off")
    for i in range(1, n_aug + 1):
        aug = denorm(train_tfms(sample_pil))
        axes[i].imshow(aug)
        axes[i].set_title(f"Aug #{i}", fontsize=10)
        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig("03_augmentation_preview.png", dpi=80, bbox_inches="tight")
    plt.close(fig)
    _print("✓ Saved: 03_augmentation_preview.png")

# =============================================================================
# STEP 10: Dataset Class & DataLoaders
# =============================================================================
class DRDataset(Dataset):
    def __init__(self, paths: List[str], labels: List[int], transform):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        img = Image.open(self.paths[idx]).convert("RGB")
        x = self.transform(img)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y

# MEM-OPT: num_workers=2, persistent_workers on, prefetch_factor=1 to save RAM
_persistent = cfg.num_workers > 0
_prefetch   = cfg.prefetch_factor if cfg.num_workers > 0 else None
train_loader = DataLoader(
    DRDataset(X_train, y_train, train_tfms),
    batch_size=cfg.batch_size, shuffle=True,
    num_workers=cfg.num_workers, pin_memory=cfg.pin_memory,
    drop_last=True,
    persistent_workers=_persistent,
    prefetch_factor=_prefetch,
)
# Eval batch: 2x train batch, capped at 64 for memory safety
_eval_batch = min(cfg.batch_size * 2, 64)
val_loader = DataLoader(
    DRDataset(X_val, y_val, val_tfms),
    batch_size=_eval_batch,
    shuffle=False,
    num_workers=cfg.num_workers, pin_memory=cfg.pin_memory,
    persistent_workers=_persistent,
    prefetch_factor=_prefetch,
)
test_loader = DataLoader(
    DRDataset(X_test, y_test, val_tfms),
    batch_size=_eval_batch,
    shuffle=False,
    num_workers=cfg.num_workers, pin_memory=cfg.pin_memory,
    persistent_workers=_persistent,
    prefetch_factor=_prefetch,
)

_print("✓ DataLoaders ready")
_print(f"  Train batches : {len(train_loader)}")
_print(f"  Val   batches : {len(val_loader)}")
_print(f"  Test  batches : {len(test_loader)}")

# =============================================================================
# STEP 11: Model Blocks (ChannelAttention, SpatialAttention, CBAM, GeM, MSFA)
# =============================================================================
class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        avg_out = self.mlp(self.avg_pool(x).view(b, c))
        max_out = self.mlp(self.max_pool(x).view(b, c))
        w = torch.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        return x * w


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_pool = x.mean(dim=1, keepdim=True)
        max_pool = x.max(dim=1, keepdim=True).values
        cat = torch.cat([avg_pool, max_pool], dim=1)
        w = torch.sigmoid(self.conv(cat))
        return x * w


class CBAM(nn.Module):
    def __init__(self, channels: int, reduction: int = 16, spatial_k: int = 7):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention(spatial_k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ca(x)
        x = self.sa(x)
        return x


class GeMPooling(nn.Module):
    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(p, dtype=torch.float32))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_clamped = x.clamp(min=self.eps).pow(self.p)
        return F.adaptive_avg_pool2d(x_clamped, 1).pow(1.0 / self.p).flatten(1)


class MultiScaleFeatureAgg(nn.Module):
    def __init__(self, in_channels: List[int], out_channels: int = 256):
        super().__init__()
        # FIX: out_channels 384→256 to reduce VRAM
        self.proj = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(7),
                nn.Conv2d(c, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.GELU(),
            )
            for c in in_channels
        ])
        self.scale_w = nn.Parameter(torch.ones(len(in_channels)) / len(in_channels))

    def forward(self, feats: List[torch.Tensor]) -> torch.Tensor:
        projected = [p(f) for p, f in zip(self.proj, feats)]
        w = torch.softmax(self.scale_w, dim=0)
        return sum(w[i] * projected[i] for i in range(len(projected)))

# =============================================================================
# STEP 12: Full Model Architecture (EfficientNetV2-S + CBAM + GeM)
# =============================================================================
FEATURE_DIM = 256   # FIX: reduced from 384 → less VRAM

class EfficientNetV2CBAMGeM(nn.Module):
    def __init__(self, num_classes: int = 5):
        super().__init__()
        self.backbone = timm.create_model(
            "tf_efficientnetv2_s",
            pretrained=True,
            features_only=True,
            out_indices=[1, 2, 3, 4],
        )
        ch = self.backbone.feature_info.channels()

        fd = FEATURE_DIM
        self.msfa       = MultiScaleFeatureAgg(ch[:3], fd)
        self.cbam_fused = CBAM(fd)
        self.cbam_deep  = CBAM(ch[3])

        self.deep_proj = nn.Sequential(
            nn.Conv2d(ch[3], fd, 1, bias=False),
            nn.BatchNorm2d(fd),
            nn.GELU(),
        )

        self.gem = GeMPooling(p=3.0)

        # FIX: head 512→256 reduces param count & VRAM
        self.head = nn.Sequential(
            nn.Linear(fd * 2, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.40),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        fused     = self.cbam_fused(self.msfa(feats[:3]))
        pool_fused = self.gem(fused)

        deep      = self.deep_proj(self.cbam_deep(feats[3]))
        pool_deep = self.gem(deep)

        pooled = torch.cat([pool_fused, pool_deep], dim=1)
        return self.head(pooled)

# =============================================================================
# STEP 13: Build Model & Multi-GPU Setup
# =============================================================================
_print("\nBuilding model...")

# FIX: clear GPU cache before building model
torch.cuda.empty_cache()
gc.collect()

base_model = EfficientNetV2CBAMGeM(num_classes=cfg.num_classes).to(cfg.device)

# Gradient checkpointing disabled — 2× T4 (14.6GB each) has ample VRAM without it
# Enabling it saves ~30% VRAM but costs ~40% speed; at bs=64 VRAM is not the bottleneck
if False and hasattr(base_model.backbone, "set_grad_checkpointing"):
    base_model.backbone.set_grad_checkpointing(enable=True)
    _print("  ✓ Gradient checkpointing enabled")

if USE_AMP and torch.cuda.device_count() > 1:
    _print(f"  Using DataParallel on {torch.cuda.device_count()} GPUs")
    model = nn.DataParallel(base_model)
else:
    model = base_model

core = model.module if hasattr(model, "module") else model
total_params = sum(p.numel() for p in core.parameters())
train_params = sum(p.numel() for p in core.parameters() if p.requires_grad)

_print(f"  Total parameters : {total_params:,}")
_print(f"  Trainable params : {train_params:,}")
_print(f"  Initial GeM p    : {core.gem.p.item():.3f}")

if torch.cuda.is_available():
    alloc = torch.cuda.memory_allocated() / 1024**2
    _print(f"  GPU mem (model)  : {alloc:.0f} MB allocated")

# =============================================================================
# STEP 14: Loss, Optimizer, Scheduler & Early Stopping
# =============================================================================
weights_arr = cnt_tr.astype(np.float32)
if weights_arr.sum() > 0:
    weights_arr = weights_arr.sum() / (cfg.num_classes * np.maximum(weights_arr, 1.0))
    weights_arr = weights_arr / weights_arr.mean()
else:
    weights_arr = np.ones(cfg.num_classes, dtype=np.float32)

class_weights = torch.tensor(weights_arr, dtype=torch.float32, device=cfg.device)
criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=cfg.label_smoothing)

backbone_params = list(core.backbone.parameters())
new_params = (
    list(core.msfa.parameters()) + list(core.cbam_fused.parameters()) +
    list(core.cbam_deep.parameters()) + list(core.deep_proj.parameters()) +
    list(core.gem.parameters()) + list(core.head.parameters())
)

optimizer = torch.optim.AdamW([
    {"params": backbone_params, "lr": cfg.lr_backbone},
    {"params": new_params,      "lr": cfg.lr_new},
], weight_decay=cfg.weight_decay)

# FIX: OneCycleLR — handles warmup + cosine decay in one go, much easier to tune
steps_per_epoch = math.ceil(len(train_loader) / cfg.accumulate_grad_batches)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=[cfg.lr_backbone * 10, cfg.lr_new * 10],
    steps_per_epoch=steps_per_epoch,
    epochs=cfg.epochs,
    pct_start=cfg.warmup_epochs / cfg.epochs,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=100.0,
)

# GradScaler
try:
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)
except TypeError:
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)


class EarlyStopping:
    def __init__(self, patience: int = 7, min_delta: float = 1e-4, path: str = "best.pth"):
        self.patience   = patience
        self.min_delta  = min_delta
        self.path       = path
        self.best_loss  = float("inf")
        self.count      = 0
        self.best_epoch = 0
        self._improved  = False

    def step(self, val_loss: float, epoch: int, model_obj: nn.Module) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss  = val_loss
            self.count      = 0
            self.best_epoch = epoch
            self._improved  = True
            save_obj = model_obj.module if hasattr(model_obj, "module") else model_obj
            torch.save(save_obj.state_dict(), self.path)
            return False
        self._improved = False
        self.count += 1
        return self.count >= self.patience

    @property
    def improved(self) -> bool:
        return self._improved


stopper = EarlyStopping(patience=cfg.patience, path=cfg.save_path)

# =============================================================================
# STEP 15: Train & Evaluation Functions
# =============================================================================
def train_one_epoch(model_obj: nn.Module, loader: DataLoader, epoch: int) -> Tuple[float, float]:
    model_obj.train()
    total_loss    = 0.0
    batch_times   = []
    accum_steps   = cfg.accumulate_grad_batches
    # Cap steps per epoch if configured (useful for fast iteration / debugging)
    n_batches     = len(loader)
    if cfg.max_steps_per_epoch is not None:
        n_batches = min(n_batches, cfg.max_steps_per_epoch)
    print_every   = max(1, n_batches // 10)   # log every ~10% of epoch
    t_epoch       = time.time()

    optimizer.zero_grad(set_to_none=True)

    for i, (imgs, lbls) in enumerate(loader):
        # Honour epoch-length cap
        if i >= n_batches:
            break

        t_batch = time.time()
        imgs = imgs.to(cfg.device, non_blocking=True)
        lbls = lbls.to(cfg.device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=USE_AMP):
            logits = model_obj(imgs)
            loss   = criterion(logits, lbls) / accum_steps

        scaler.scale(loss).backward()

        if (i + 1) % accum_steps == 0 or (i + 1) == n_batches:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model_obj.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()   # once per optimizer step, not per micro-batch

        actual_loss = loss.item() * accum_steps
        total_loss += actual_loss
        batch_times.append(time.time() - t_batch)

        if (i + 1) % print_every == 0 or (i + 1) == n_batches:
            window  = batch_times[-print_every:] if len(batch_times) >= print_every else batch_times
            avg_bt  = np.mean(window)
            cur_lr  = optimizer.param_groups[1]["lr"]
            elapsed = time.time() - t_epoch
            eta_s   = avg_bt * (n_batches - i - 1)
            _print(
                f"    Ep {epoch:02d} | Step {i+1:>5}/{n_batches}"
                f" | {elapsed:>5.0f}s (ETA {eta_s:.0f}s) | lr={cur_lr:.2e}"
                f" | loss={actual_loss:.4f} | {1/avg_bt:.1f} it/s"
            )

    # Free GPU + CPU memory at end of every epoch
    torch.cuda.empty_cache()
    gc.collect()
    return total_loss / max(1, n_batches), time.time() - t_epoch


@torch.no_grad()
def evaluate(model_obj: nn.Module, loader: DataLoader) -> Tuple:
    model_obj.eval()
    total_loss = 0.0
    probs_all, preds_all, true_all = [], [], []

    for imgs, lbls in loader:
        imgs = imgs.to(cfg.device, non_blocking=True)
        lbls = lbls.to(cfg.device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=USE_AMP):
            logits = model_obj(imgs)
            loss   = criterion(logits, lbls)

        total_loss += loss.item()

        probs = F.softmax(logits.float(), dim=1).cpu().numpy()
        preds = probs.argmax(axis=1)

        probs_all.extend(probs.tolist())
        preds_all.extend(preds.tolist())
        true_all.extend(lbls.cpu().numpy().tolist())

    avg_loss = total_loss / max(1, len(loader))
    acc  = accuracy_score(true_all, preds_all) if true_all else 0.0
    f1m  = f1_score(true_all, preds_all, average="macro", zero_division=0) if true_all else 0.0
    return avg_loss, acc, f1m, probs_all, preds_all, true_all

# =============================================================================
# STEP 16: Training Loop
# =============================================================================
history = {k: [] for k in ("train_loss", "val_loss", "val_acc", "val_f1", "lr")}

_steps_shown = cfg.max_steps_per_epoch if cfg.max_steps_per_epoch else len(train_loader)
_print("\n" + "=" * 70)
_print("TRAINING STARTED")
_print("=" * 70)
_print(f"  Total epochs     : {cfg.epochs}")
_print(f"  Total batches    : {len(train_loader)} (batch_size={cfg.batch_size})")
_print(f"  Steps per epoch  : {_steps_shown}" + (" [CAPPED]" if cfg.max_steps_per_epoch else ""))
_print(f"  Accum steps      : {cfg.accumulate_grad_batches}")
_print(f"  Effective batch  : {cfg.batch_size * cfg.accumulate_grad_batches}")
_print(f"  Backbone LR      : {cfg.lr_backbone:.2e} → peak {cfg.lr_backbone*10:.2e}")
_print(f"  Head LR          : {cfg.lr_new:.2e} → peak {cfg.lr_new*10:.2e}")
_print("=" * 70)

for epoch in range(1, cfg.epochs + 1):
    if len(train_loader) == 0:
        _print("Skipping training loop due to empty train loader")
        break

    _print(f"\n──── Epoch {epoch:02d}/{cfg.epochs} ────")
    tr_loss, tr_time = train_one_epoch(model, train_loader, epoch)

    _print(f"  Evaluating on validation set...")
    vl_loss, vl_acc, vl_f1, _, _, _ = evaluate(model, val_loader)

    history["train_loss"].append(tr_loss)
    history["val_loss"].append(vl_loss)
    history["val_acc"].append(vl_acc)
    history["val_f1"].append(vl_f1)
    history["lr"].append(optimizer.param_groups[1]["lr"])

    done   = stopper.step(vl_loss, epoch, model)
    status = "★ BEST" if stopper.improved else f"patience {stopper.count}/{cfg.patience}"

    _print(
        f"  ► Ep {epoch:02d}/{cfg.epochs} | {tr_time:.0f}s | "
        f"TrLoss {tr_loss:.4f} | VlLoss {vl_loss:.4f} | "
        f"ValAcc {vl_acc*100:.2f}% | ValF1 {vl_f1*100:.2f}% | {status}"
    )

    if done:
        _print(f"\n[Early stopping at epoch {epoch} (best: epoch {stopper.best_epoch})]")
        break

_print("\n✓ Training complete")
torch.cuda.empty_cache()
gc.collect()

# =============================================================================
# STEP 17: Training Dashboard & Learning Curves
# =============================================================================
if history["train_loss"]:
    ep = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle("Training Dashboard", fontsize=14, fontweight="bold")

    axes[0, 0].plot(ep, history["train_loss"], "b-o", ms=4, lw=2, label="Train")
    axes[0, 0].plot(ep, history["val_loss"],   "r-s", ms=4, lw=2, label="Val")
    if stopper.best_epoch > 0:
        axes[0, 0].axvline(stopper.best_epoch, color="g", linestyle="--",
                           label=f"Best ep {stopper.best_epoch}")
    axes[0, 0].set_title("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(ep, [v * 100 for v in history["val_acc"]], "g-^", ms=4, lw=2)
    axes[0, 1].set_title("Val Accuracy (%)")
    axes[0, 1].grid(alpha=0.3)

    axes[1, 0].plot(ep, [v * 100 for v in history["val_f1"]], "m-D", ms=4, lw=2)
    axes[1, 0].set_title("Val Macro-F1 (%)")
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].plot(ep, history["lr"], color="darkorange", lw=2)
    axes[1, 1].set_title("Learning Rate (head group)")
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("04_training_dashboard.png", dpi=80, bbox_inches="tight")
    plt.close(fig)
    _print("✓ Saved: 04_training_dashboard.png")

# =============================================================================
# STEP 18: Test Evaluation & Classification Report
# =============================================================================
if os.path.exists(cfg.save_path):
    try:
        core.load_state_dict(torch.load(cfg.save_path, map_location=cfg.device, weights_only=True))
    except TypeError:
        core.load_state_dict(torch.load(cfg.save_path, map_location=cfg.device))
    _print(f"✓ Best model loaded from {cfg.save_path}")

_print("  Running test evaluation...")
test_loss, test_acc, test_f1m, probs, preds, true = evaluate(model, test_loader)
probs_np = np.array(probs) if probs else np.empty((0, cfg.num_classes))

auc_ovr = float("nan")
qwk     = float("nan")
auc_cls = [float("nan")] * cfg.num_classes
f1_cls   = np.zeros(cfg.num_classes)
prec_cls = np.zeros(cfg.num_classes)
rec_cls  = np.zeros(cfg.num_classes)

if true:
    prec = precision_score(true, preds, average="macro", zero_division=0)
    rec  = recall_score(true, preds,    average="macro", zero_division=0)
    qwk  = cohen_kappa_score(true, preds, weights="quadratic")
    try:
        auc_ovr = roc_auc_score(true, probs_np, multi_class="ovr", average="macro")
    except Exception:
        auc_ovr = float("nan")

    f1_cls   = f1_score(true, preds, average=None, zero_division=0)
    prec_cls = precision_score(true, preds, average=None, zero_division=0)
    rec_cls  = recall_score(true, preds, average=None, zero_division=0)

    for c in range(cfg.num_classes):
        try:
            auc_cls[c] = roc_auc_score([1 if t == c else 0 for t in true], probs_np[:, c])
        except Exception:
            auc_cls[c] = float("nan")

    _print("\n" + "=" * 70)
    _print("FINAL TEST RESULTS")
    _print("=" * 70)
    _print(f"  Loss              : {test_loss:.4f}")
    _print(f"  Accuracy          : {test_acc*100:.2f}%")
    _print(f"  Macro F1-Score    : {test_f1m*100:.2f}%")
    _print(f"  Quadratic Kappa   : {qwk:.4f}")
    _print(f"  AUC (OvR macro)   : {auc_ovr:.4f}")
    _print("\nFull Classification Report:")
    _print(classification_report(true, preds, target_names=list(cfg.class_names), zero_division=0))

# =============================================================================
# STEP 19: Performance Matrices & Evaluation Visualizations
# =============================================================================
if true:
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    fig.suptitle("Performance Matrices & Evaluation", fontsize=14, fontweight="bold")

    # Confusion matrix
    cm = confusion_matrix(true, preds)
    cm_norm = cm.astype(float) / np.clip(cm.sum(axis=1, keepdims=True), 1e-9, None)
    sns.heatmap(cm_norm, annot=cm, fmt="d", cmap="Blues", ax=axes[0, 0],
                xticklabels=cfg.class_names, yticklabels=cfg.class_names,
                cbar_kws={"label": "Rate"})
    axes[0, 0].set_title(f"Confusion Matrix (F1={test_f1m*100:.2f}%)")
    axes[0, 0].set_xlabel("Predicted")
    axes[0, 0].set_ylabel("True")

    # ROC curves
    for c in range(cfg.num_classes):
        yb = np.array([1 if t == c else 0 for t in true], dtype=np.int32)
        if yb.sum() > 0:
            fpr, tpr, _ = roc_curve(yb, probs_np[:, c])
            axes[0, 1].plot(fpr, tpr, color=CLASS_COLORS[c], lw=2,
                            label=f"G{c} (AUC={auc_cls[c]:.3f})")
    axes[0, 1].plot([0, 1], [0, 1], "k--", lw=1)
    axes[0, 1].set_title("ROC Curves per Class (OvR)")
    axes[0, 1].legend(fontsize=9)
    axes[0, 1].grid(alpha=0.3)

    # F1 / Precision / Recall per class
    x_cls = np.arange(cfg.num_classes)
    w3 = 0.25
    axes[1, 0].bar(x_cls - w3, f1_cls   * 100, width=w3, label="F1",        color="#3498db")
    axes[1, 0].bar(x_cls,      prec_cls  * 100, width=w3, label="Precision", color="#2ecc71")
    axes[1, 0].bar(x_cls + w3, rec_cls   * 100, width=w3, label="Recall",    color="#e74c3c")
    axes[1, 0].set_xticks(x_cls)
    axes[1, 0].set_xticklabels(cfg.class_names, rotation=15, ha="right")
    axes[1, 0].set_title("Metrics per Class")
    axes[1, 0].legend(fontsize=9)
    axes[1, 0].grid(axis="y", alpha=0.3)

    # Confidence distribution
    max_prob = probs_np.max(axis=1)
    corr = (np.array(preds) == np.array(true))
    axes[1, 1].hist(max_prob[corr],  bins=20, alpha=0.7, color="g", label="Correct", edgecolor="black")
    axes[1, 1].hist(max_prob[~corr], bins=20, alpha=0.7, color="r", label="Wrong",   edgecolor="black")
    axes[1, 1].set_title("Confidence Distribution")
    axes[1, 1].legend(fontsize=9)
    axes[1, 1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig("05_performance_matrices.png", dpi=80, bbox_inches="tight")
    plt.close(fig)
    _print("✓ Saved: 05_performance_matrices.png")

# =============================================================================
# STEP 20: XAI: Grad-CAM++ & EigenCAM Interpretability
# =============================================================================
def run_xai(model_obj: nn.Module, paths: List[str], labels: List[int], n_samples: int = 4):
    try:
        from pytorch_grad_cam import GradCAMPlusPlus, EigenCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image

        m = model_obj.module if hasattr(model_obj, "module") else model_obj
        m.eval()
        target_layer = [m.cbam_fused.sa.conv]

        cam_pp    = GradCAMPlusPlus(model=m, target_layers=target_layer)
        cam_eigen = EigenCAM(model=m, target_layers=target_layer)

        n    = min(n_samples, len(paths))
        idxs = np.random.choice(len(paths), n, replace=False)

        fig, axes = plt.subplots(3, n, figsize=(3.2 * n, 9))
        fig.suptitle("XAI: Grad-CAM++ & EigenCAM", fontsize=14, fontweight="bold")

        for i, idx in enumerate(idxs):
            with Image.open(paths[idx]) as _xai_raw:
                pil = _xai_raw.convert("RGB").resize((cfg.img_size, cfg.img_size))
            rgb = np.array(pil).astype(np.float32) / 255.0
            inp = val_tfms(pil).unsqueeze(0).to(cfg.device)

            g_pp = cam_pp(input_tensor=inp, targets=None)[0]
            g_ei = cam_eigen(input_tensor=inp, targets=None)[0]

            axes[0, i].imshow(rgb); axes[0, i].axis("off")
            axes[0, i].set_title(f"GT: {cfg.class_names[labels[idx]]}")
            axes[1, i].imshow(show_cam_on_image(rgb, g_pp, use_rgb=True)); axes[1, i].axis("off")
            axes[2, i].imshow(show_cam_on_image(rgb, g_ei, use_rgb=True)); axes[2, i].axis("off")

        axes[0, 0].set_ylabel("Original")
        axes[1, 0].set_ylabel("Grad-CAM++")
        axes[2, 0].set_ylabel("EigenCAM")
        plt.tight_layout()
        plt.savefig("06_xai_gradcam.png", dpi=80, bbox_inches="tight")
        plt.close(fig)
        _print("✓ Saved: 06_xai_gradcam.png")
        # Release CAM hooks & GPU memory
        del cam_pp, cam_eigen
    except Exception as e:
        _print(f"XAI skipped: {e}")
    finally:
        torch.cuda.empty_cache()
        gc.collect()

if X_test:
    torch.cuda.empty_cache()
    gc.collect()
    run_xai(model, X_test, y_test, 4)

# =============================================================================
# STEP 21: Robustness Testing (Gaussian Noise, Brightness/Contrast, FGSM)
# =============================================================================
class NoisyDataset(Dataset):
    def __init__(self, paths, labels, sigma):
        self.paths, self.labels, self.sigma = paths, labels, sigma

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_t = val_tfms(Image.open(self.paths[idx]).convert("RGB"))
        if self.sigma > 0:
            img_t = img_t + torch.randn_like(img_t) * self.sigma
        return img_t, torch.tensor(self.labels[idx], dtype=torch.long)


class BCorrDataset(Dataset):
    def __init__(self, paths, labels, b, c):
        self.paths, self.labels, self.b, self.c = paths, labels, b, c

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        img = transforms.functional.adjust_brightness(img, self.b)
        img = transforms.functional.adjust_contrast(img, self.c)
        return val_tfms(img), torch.tensor(self.labels[idx], dtype=torch.long)


@torch.enable_grad()
def fgsm_eval(model_obj: nn.Module, loader: DataLoader, epsilon: float = 0.02):
    """FGSM adversarial robustness evaluation on at most 500 samples (faster)."""
    m = model_obj.module if hasattr(model_obj, "module") else model_obj
    m.eval()
    ce = nn.CrossEntropyLoss()
    preds_adv, true_adv = [], []
    max_samples = 500     # FIX: cap FGSM at 500 images to avoid long runtime/OOM

    for imgs, lbls in loader:
        if len(true_adv) >= max_samples:
            break
        imgs = imgs.to(cfg.device).requires_grad_(True)
        lbls = lbls.to(cfg.device)

        with torch.amp.autocast("cuda", enabled=USE_AMP):
            logits = m(imgs)
            loss   = ce(logits.float(), lbls)

        m.zero_grad()
        loss.backward()

        if imgs.grad is None:        # safety guard — skip batch if grads didn't flow
            continue
        x_adv = torch.clamp(imgs.data + epsilon * imgs.grad.sign(), -3.5, 3.5)
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=USE_AMP):
                p = m(x_adv).float().argmax(dim=1).cpu().numpy()

        preds_adv.extend(p)
        true_adv.extend(lbls.cpu().numpy())

    return (
        accuracy_score(true_adv, preds_adv),
        f1_score(true_adv, preds_adv, average="macro", zero_division=0),
    )


# Flush memory before robustness section
torch.cuda.empty_cache()
gc.collect()

robustness_results = []
rob_batch = _eval_batch   # same capped eval batch → consistent memory use

if len(test_loader) > 0:
    _print("\nRunning robustness tests...")
    for sig in cfg.noise_sigmas:
        _, a, f, _, _, _ = evaluate(
            model, DataLoader(NoisyDataset(X_test, y_test, sig),
                              batch_size=rob_batch, num_workers=0))
        robustness_results.append({"type": f"Noise σ={sig}", "acc": a * 100, "f1": f * 100})
        _print(f"  Noise σ={sig}: Acc={a*100:.1f}%, F1={f*100:.1f}%")

    for b, c_ in cfg.brightness_contrast_pairs:
        _, a, f, _, _, _ = evaluate(
            model, DataLoader(BCorrDataset(X_test, y_test, b, c_),
                              batch_size=rob_batch, num_workers=0))
        robustness_results.append({"type": f"B={b} C={c_}", "acc": a * 100, "f1": f * 100})
        _print(f"  B={b} C={c_}: Acc={a*100:.1f}%, F1={f*100:.1f}%")

    _print("  Running FGSM (capped at 500 samples)...")
    fa, ff = fgsm_eval(model, test_loader, cfg.fgsm_eps)
    robustness_results.append({"type": f"FGSM ε={cfg.fgsm_eps}", "acc": fa * 100, "f1": ff * 100})
    _print(f"  FGSM ε={cfg.fgsm_eps}: Acc={fa*100:.1f}%, F1={ff*100:.1f}%")

# =============================================================================
# STEP 22: Robustness Visualization & Error Analysis
# =============================================================================
if robustness_results and true:
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle("Robustness & Error Analysis", fontsize=14, fontweight="bold")

    names_r = [r["type"] for r in robustness_results]
    x_r = np.arange(len(robustness_results))

    axes[0].bar(x_r, [r["acc"] for r in robustness_results], color="#3498db")
    axes[0].axhline(test_acc * 100, color="r", ls="--", label="Clean Acc")
    axes[0].set_xticks(x_r)
    axes[0].set_xticklabels(names_r, rotation=30, ha="right")
    axes[0].legend()
    axes[0].set_title("Robustness: Accuracy")

    axes[1].bar(x_r, [r["f1"] for r in robustness_results], color="#e74c3c")
    axes[1].axhline(test_f1m * 100, color="navy", ls="--", label="Clean F1")
    axes[1].set_xticks(x_r)
    axes[1].set_xticklabels(names_r, rotation=30, ha="right")
    axes[1].legend()
    axes[1].set_title("Robustness: Macro-F1")

    wrong_idx = [i for i, (t, p) in enumerate(zip(true, preds)) if t != p]
    if wrong_idx:
        miss_map = np.zeros((cfg.num_classes, cfg.num_classes), dtype=int)
        for i in wrong_idx:
            miss_map[true[i]][preds[i]] += 1
        sns.heatmap(miss_map, annot=True, fmt="d", cmap="Reds", ax=axes[2],
                    xticklabels=cfg.class_names, yticklabels=cfg.class_names)
        axes[2].set_title(f"Error Map ({len(wrong_idx)} mistakes)")
        axes[2].set_xlabel("Pred")
        axes[2].set_ylabel("True")

    plt.tight_layout()
    plt.savefig("07_robustness_error_analysis.png", dpi=80, bbox_inches="tight")
    plt.close(fig)
    _print("✓ Saved: 07_robustness_error_analysis.png")

# =============================================================================
# STEP 23: Model Persistence & Inference Pipeline
# =============================================================================
checkpoint = {
    "model_state":     core.state_dict(),
    "optimizer_state": optimizer.state_dict(),
    "history":         history,
    "best_epoch":      stopper.best_epoch,
    "cfg": {"img_size": cfg.img_size, "num_classes": cfg.num_classes},
}
torch.save(checkpoint, cfg.checkpoint_path)
_print(f"✓ Checkpoint saved: {cfg.checkpoint_path}")


def predict(img_path: str, model_obj: nn.Module) -> Tuple[int, np.ndarray]:
    """Return predicted class index and per-class probability vector."""
    model_obj.eval()
    x = val_tfms(Image.open(img_path).convert("RGB")).unsqueeze(0).to(cfg.device)
    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=USE_AMP):
            logits = model_obj(x)
    prob = F.softmax(logits.float(), dim=1).cpu().numpy()[0]
    return int(prob.argmax()), prob

# =============================================================================
# STEP 24: Final Summary Report
# =============================================================================
_print("\n" + "=" * 80)
_print("FINAL SUMMARY REPORT")
_print("=" * 80)
_print(f"Architecture : EfficientNetV2-S + CBAM + GeM (MSFA)")
_print(f"Total params : {total_params:,}")
_print(f"Feature dim  : {FEATURE_DIM} (MSFA out) × 2 streams")
_print(f"Batch size   : {cfg.batch_size} × {cfg.accumulate_grad_batches} accum = {cfg.batch_size*cfg.accumulate_grad_batches} effective")
if true:
    _print(f"Accuracy     : {test_acc*100:.2f}%")
    _print(f"Macro F1     : {test_f1m*100:.2f}%")
    _print(f"AUC (OvR)    : {auc_ovr:.4f}")
    _print(f"QWK          : {qwk:.4f}")
_print(f"Best epoch   : {stopper.best_epoch}")
_print("=" * 80)
