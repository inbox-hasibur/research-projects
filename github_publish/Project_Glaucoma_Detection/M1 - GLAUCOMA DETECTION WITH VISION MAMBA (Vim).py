# ============================================================
# 1 Install
# ============================================================
!pip install -q timm scikit-learn matplotlib seaborn einops
!pip install -q mamba-ssm causal-conv1d  
!pip install -q grad-cam

# ============================================================
# 2 Imports & Config
# ============================================================
import os, time, warnings
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             roc_curve, classification_report)
import timm
from einops import rearrange, repeat
warnings.filterwarnings('ignore')

try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
    print("✅ mamba-ssm loaded — true Mamba SSM")
except ImportError:
    MAMBA_AVAILABLE = False
    print("⚠️ mamba-ssm not found — using PyTorch SSM fallback")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Device: {DEVICE}")

# ── Hyperparameters ───────────────────────────────────────────
IMG_SIZE     = 224
BATCH_SIZE   = 16
EPOCHS       = 25
LR           = 1e-4
WEIGHT_DECAY = 1e-4
LABEL_SMOOTH = 0.1
PATIENCE     = 5        # Early stopping patience

DATA_DIR  = "/kaggle/input/datasets/hindsaud/datasets-higancnn-glaucoma-detection/Datasets/Acrima"
SAVE_PATH = "best_vim_glaucoma_v3.pth"

# ============================================================
# 3 SimpleSSM (PyTorch fallback)
# ============================================================
class SimpleSSM(nn.Module):
    """Pure-PyTorch S4-like SSM — dimension bug fixed."""
    def __init__(self, d_model, d_state=16, expand=2):
        super().__init__()
        d_inner = d_model * expand
        self.d_inner = d_inner
        self.d_state = d_state

        self.in_proj  = nn.Linear(d_model, d_inner * 2, bias=False)
        self.conv1d   = nn.Conv1d(d_inner, d_inner, 3, padding=1,
                                  groups=d_inner, bias=True)
        # Output: d_state (dt) + d_state (B) + d_inner (C) = d_state*2 + d_inner
        self.x_proj   = nn.Linear(d_inner, d_state * 2 + d_inner, bias=False)
        self.dt_proj  = nn.Linear(d_state, d_inner, bias=True)   # dt_proj: d_state → d_inner
        self.A_log    = nn.Parameter(
            torch.log(torch.arange(1, d_state + 1).float())
            .unsqueeze(0).repeat(d_inner, 1)
        )
        self.D        = nn.Parameter(torch.ones(d_inner))
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
        self.norm     = nn.LayerNorm(d_inner)

    def forward(self, x):                          # x: (B, L, d_model)
        B, L, _ = x.shape
        xz  = self.in_proj(x)
        xr, z = xz.chunk(2, dim=-1)               # each: (B, L, d_inner)

        # Depthwise conv along sequence dim
        xr = rearrange(self.conv1d(rearrange(xr, 'b l d -> b d l')),
                       'b d l -> b l d')
        xr = F.silu(xr)

        x_dbl = self.x_proj(xr)                   # (B, L, d_state*2 + d_inner)

        # correct split order — dt first (d_state), then B, then C
        dt, B_ssm, C = x_dbl.split([self.d_state, self.d_state, self.d_inner], dim=-1)
        dt = F.softplus(self.dt_proj(dt))          # dt_proj: d_state → d_inner

        # Simplified skip-connection output
        y = (xr * self.D) * F.silu(z)
        return self.out_proj(self.norm(y))

# ============================================================
# 4 Bidirectional VimBlock
# ============================================================
class VimBlock(nn.Module):
    def __init__(self, dim, d_state=16, expand=2, drop=0.1):
        super().__init__()
        self.norm  = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        if MAMBA_AVAILABLE:
            self.ssm_fwd = Mamba(d_model=dim, d_state=d_state, d_conv=4, expand=expand)
            self.ssm_bwd = Mamba(d_model=dim, d_state=d_state, d_conv=4, expand=expand)
        else:
            self.ssm_fwd = SimpleSSM(dim, d_state, expand)
            self.ssm_bwd = SimpleSSM(dim, d_state, expand)

        self.proj = nn.Linear(dim * 2, dim)
        self.drop = nn.Dropout(drop)
        self.mlp  = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(drop),
            nn.Linear(dim * 4, dim), nn.Dropout(drop),
        )

    def forward(self, x):                          # (B, L, d)
        res = x
        x   = self.norm(x)
        fwd = self.ssm_fwd(x)
        bwd = self.ssm_bwd(x.flip(1)).flip(1)
        x   = self.proj(torch.cat([fwd, bwd], dim=-1))
        x   = self.drop(x) + res
        x   = x + self.drop(self.mlp(self.norm2(x)))
        return x

# ============================================================
# 5 Hybrid Model: Swin Backbone + Vim Head
# ============================================================
def build_model():
    backbone = timm.create_model(
        'swin_tiny_patch4_window7_224',
        pretrained=True,
        features_only=True,
        out_indices=[3]          # stage-3 → (B, 7, 7, 768)
    )

    class HybridVim(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = backbone
            self.proj     = nn.Linear(768, 384)
            self.vim      = VimBlock(384, d_state=16, expand=2, drop=0.1)
            self.norm     = nn.LayerNorm(384)
            self.head     = nn.Sequential(
                nn.Linear(384, 128), nn.GELU(), nn.Dropout(0.3),
                nn.Linear(128, 1)
            )

        def forward(self, x):
            feat = self.backbone(x)[0]             # (B, 7, 7, 768)
            feat = rearrange(feat, 'b h w d -> b (h w) d')
            feat = self.proj(feat)                 # (B, 49, 384)
            feat = self.vim(feat)                  # (B, 49, 384)
            feat = self.norm(feat).mean(1)         # global avg pool
            return self.head(feat)

    return HybridVim()

# ============================================================
# 6 Dataset & Augmentation
# ============================================================
mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
])
val_test_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

class GlaucomaDataset(Dataset):
    def __init__(self, paths, labels, transform=None):
        self.paths, self.labels, self.transform = paths, labels, transform
    def __len__(self):  return len(self.paths)
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform: img = self.transform(img)
        return img, torch.tensor([self.labels[idx]], dtype=torch.float32)

# ============================================================
# 7 Label-Smoothing Loss
# ============================================================
class BCELabelSmoothing(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.s = smoothing
    def forward(self, logits, targets):
        targets_s = targets * (1 - self.s) + 0.5 * self.s
        return F.binary_cross_entropy_with_logits(logits, targets_s)

# ============================================================
# 8 Early Stopping Helper
# ============================================================
class EarlyStopping:
    """Stop training when val_loss doesn't improve for `patience` epochs."""
    def __init__(self, patience=5, min_delta=1e-4, save_path=SAVE_PATH):
        self.patience   = patience
        self.min_delta  = min_delta
        self.save_path  = save_path
        self.best_loss  = float('inf')
        self.counter    = 0
        self.best_epoch = 0

    def step(self, val_loss, epoch, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss  = val_loss
            self.counter    = 0
            self.best_epoch = epoch
            torch.save(model.state_dict(), self.save_path)
            return False          # don't stop
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True       # stop!
            return False
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 121.7/121.7 kB 3.4 MB/s eta 0:00:00
  Installing build dependencies ... done
  Getting requirements to build wheel ... done
  Preparing metadata (pyproject.toml) ... done
  Installing build dependencies ... done
  Getting requirements to build wheel ... done
  Preparing metadata (pyproject.toml) ... done
  Building wheel for mamba-ssm (pyproject.toml) ... done
  Building wheel for causal-conv1d (pyproject.toml) ... done
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 7.8/7.8 MB 51.4 MB/s eta 0:00:00
  Installing build dependencies ... done
  Getting requirements to build wheel ... done
  Preparing metadata (pyproject.toml) ... done
  Building wheel for grad-cam (pyproject.toml) ... done
⚠️ mamba-ssm not found — using PyTorch SSM fallback
🚀 Device: cuda
# ============================================================
# 9 Training & Eval Functions
# ============================================================
def train_one_epoch(model, loader, criterion, optimizer, scaler):
    model.train()
    total = 0.0
    for imgs, lbls in loader:
        imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            loss = criterion(model(imgs), lbls)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer); scaler.update()
        total += loss.item()
    return total / len(loader)

@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total, probs_all, preds_all, lbls_all = 0.0, [], [], []
    for imgs, lbls in loader:
        imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
        out  = model(imgs)
        total += criterion(out, lbls).item()
        p = torch.sigmoid(out).cpu().numpy()
        probs_all.extend(p.flatten())
        preds_all.extend((p > 0.5).astype(int).flatten())
        lbls_all.extend(lbls.cpu().numpy().flatten())
    return total / len(loader), accuracy_score(lbls_all, preds_all), probs_all, preds_all, lbls_all

# ============================================================
# 10 Load Data
# ============================================================
image_paths, labels = [], []
for cls_name, cls_label in {'Normal': 0, 'Glaucoma': 1}.items():
    cls_dir = os.path.join(DATA_DIR, cls_name)
    if not os.path.exists(cls_dir):
        print(f"❌ Not found: {cls_dir}"); continue
    for fn in os.listdir(cls_dir):
        if fn.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_paths.append(os.path.join(cls_dir, fn))
            labels.append(cls_label)

print(f"📂 Total: {len(image_paths)}  "
      f"(Normal={labels.count(0)}, Glaucoma={labels.count(1)})")

X_tv, X_test, y_tv, y_test = train_test_split(
    image_paths, labels, test_size=0.15, random_state=42, stratify=labels)
X_train, X_val, y_train, y_val = train_test_split(
    X_tv, y_tv, test_size=0.176, random_state=42, stratify=y_tv)

print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

train_loader = DataLoader(GlaucomaDataset(X_train, y_train, train_transforms),
                          BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
val_loader   = DataLoader(GlaucomaDataset(X_val,   y_val,   val_test_transforms),
                          BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
test_loader  = DataLoader(GlaucomaDataset(X_test,  y_test,  val_test_transforms),
                          BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

# ============================================================
# 11 Build Model & Optimizer
# ============================================================
model     = build_model().to(DEVICE)
params    = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"📐 Trainable Parameters: {params:,}")

criterion = BCELabelSmoothing(LABEL_SMOOTH)
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
scaler    = torch.cuda.amp.GradScaler()
stopper   = EarlyStopping(patience=PATIENCE, save_path=SAVE_PATH)

# ============================================================
# 12 Training Loop with Early Stopping
# ============================================================
train_losses, val_losses, val_accs, lrs = [], [], [], []

print("🔥 Training Vision Mamba (Vim) V3...")
print("─" * 70)

for epoch in range(1, EPOCHS + 1):
    t0      = time.time()
    tr_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler)
    vl_loss, vl_acc, _, _, _ = evaluate(model, val_loader, criterion)
    scheduler.step()

    train_losses.append(tr_loss)
    val_losses.append(vl_loss)
    val_accs.append(vl_acc)
    lrs.append(optimizer.param_groups[0]['lr'])

    stop = stopper.step(vl_loss, epoch, model)
    flag = "✅ BEST" if stopper.counter == 0 else f"  (patience {stopper.counter}/{PATIENCE})"

    print(f"Epoch {epoch:02d}/{EPOCHS} | {time.time()-t0:.0f}s | "
          f"Train: {tr_loss:.4f} | Val: {vl_loss:.4f} | "
          f"Acc: {vl_acc*100:.2f}% | LR: {lrs[-1]:.2e} {flag}")

    if stop:
        print(f"\n⏹️  Early stopping at epoch {epoch}. "
              f"Best was epoch {stopper.best_epoch} (val_loss={stopper.best_loss:.4f})")
        break

print(f"\n✅ Training done. Best model saved → {SAVE_PATH}")

# ============================================================
# 13 Training Curves
# ============================================================
ep = range(1, len(train_losses) + 1)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].plot(ep, train_losses, 'b-o', ms=4, label='Train')
axes[0].plot(ep, val_losses,   'r-o', ms=4, label='Val')
axes[0].axvline(stopper.best_epoch, color='g', linestyle='--', label=f'Best (ep {stopper.best_epoch})')
axes[0].set_title("Loss Curves", fontweight='bold'); axes[0].legend(); axes[0].grid(alpha=0.4)

axes[1].plot(ep, [a*100 for a in val_accs], 'g-s', ms=4)
axes[1].set_title("Validation Accuracy (%)", fontweight='bold'); axes[1].grid(alpha=0.4)

axes[2].plot(ep, lrs, 'm-', lw=2)
axes[2].set_title("Learning Rate", fontweight='bold'); axes[2].grid(alpha=0.4)

plt.suptitle("V3 Vision Mamba — Training Dashboard", fontsize=15, fontweight='bold')
plt.tight_layout(); plt.show()

# ============================================================
# 14 Final Test Evaluation
# ============================================================
model.load_state_dict(torch.load(SAVE_PATH))
_, _, test_probs, test_preds, test_true = evaluate(model, test_loader, criterion)

acc  = accuracy_score(test_true, test_preds)
prec = precision_score(test_true, test_preds, zero_division=0)
rec  = recall_score(test_true, test_preds, zero_division=0)
spec = recall_score(test_true, test_preds, pos_label=0, zero_division=0)
f1   = f1_score(test_true, test_preds, zero_division=0)
auc  = roc_auc_score(test_true, test_probs)

print("\n" + "═"*52)
print("✅ V3 VISION MAMBA — FINAL TEST RESULTS")
print("═"*52)
print(f"  Accuracy        : {acc*100:.2f}%")
print(f"  Precision       : {prec*100:.2f}%")
print(f"  Sensitivity     : {rec*100:.2f}%")
print(f"  Specificity     : {spec*100:.2f}%")
print(f"  F1-Score        : {f1*100:.2f}%")
print(f"  AUC-ROC         : {auc:.4f}")
print("═"*52)
print(classification_report(test_true, test_preds, target_names=['Normal', 'Glaucoma']))

# ============================================================
# 15 Confusion Matrix + ROC + Score Distribution
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

cm = confusion_matrix(test_true, test_preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            annot_kws={"size": 16, "weight": "bold"})
axes[0].set_title('Confusion Matrix', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Predicted'); axes[0].set_ylabel('True')
axes[0].xaxis.set_ticklabels(['Normal', 'Glaucoma'])
axes[0].yaxis.set_ticklabels(['Normal', 'Glaucoma'])

fpr, tpr, _ = roc_curve(test_true, test_probs)
axes[1].plot(fpr, tpr, 'darkorange', lw=2.5, label=f'AUC = {auc:.3f}')
axes[1].plot([0,1],[0,1],'navy', lw=1.5, linestyle='--')
axes[1].fill_between(fpr, tpr, alpha=0.15, color='darkorange')
axes[1].set_title('ROC Curve', fontsize=14, fontweight='bold')
axes[1].set_xlabel('FPR'); axes[1].set_ylabel('TPR')
axes[1].legend(); axes[1].grid(alpha=0.4)

pn = [p for p,l in zip(test_probs, test_true) if l==0]
pg = [p for p,l in zip(test_probs, test_true) if l==1]
axes[2].hist(pn, bins=20, alpha=0.6, color='#2ecc71', label='Normal')
axes[2].hist(pg, bins=20, alpha=0.6, color='#e74c3c', label='Glaucoma')
axes[2].axvline(0.5, color='k', linestyle='--', label='Threshold')
axes[2].set_title('Score Distribution', fontsize=14, fontweight='bold')
axes[2].legend(); axes[2].grid(alpha=0.4)

plt.suptitle("V3 Vision Mamba — Test Evaluation", fontsize=16, fontweight='bold')
plt.tight_layout(); plt.show()


# ============================================================
# CELL 16 — Grad-CAM XAI (FIXED for Vision Mamba)
# ============================================================
# ❌ আগের সমস্যা:
#    model.backbone.layers[-1]  → FeatureListNet এ .layers নেই
#    use_cuda= → নতুন pytorch_grad_cam এ নেই
#
# ✅ Fix:
#    model.vim.norm2 target করো (HybridVim এর নিজস্ব layer)
#    reshape_transform দাও — Transformer tokens → 2D map

try:
    from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
    from pytorch_grad_cam.utils.image import show_cam_on_image

    # ── Reshape transform: (B, 49, 384) → (B, 384, 7, 7) ──────
    def vim_reshape_transform(tensor):
        """VimBlock output is (B, L, D) — reshape to (B, D, H, W) for Grad-CAM."""
        B, L, D = tensor.shape
        H = W = int(L ** 0.5)          # 49 → 7×7
        result = tensor.reshape(B, H, W, D)
        result = result.permute(0, 3, 1, 2)   # (B, D, H, W)
        return result.contiguous()

    # ── Target: VimBlock norm2 — after SSM, before MLP ─────────
    # This is inside HybridVim and always accessible
    target_layers = [model.vim.norm2]

    cam    = GradCAM(model=model, target_layers=target_layers,
                     reshape_transform=vim_reshape_transform)
    cam_pp = GradCAMPlusPlus(model=model, target_layers=target_layers,
                              reshape_transform=vim_reshape_transform)

    sample_indices = np.random.choice(len(X_test), 5, replace=False)
    fig, axes = plt.subplots(3, 5, figsize=(22, 13))

    row_titles = ["Original Image", "Grad-CAM", "Grad-CAM++"]
    for row, title in enumerate(row_titles):
        axes[row][0].set_ylabel(title, fontsize=11, fontweight='bold')

    for i, idx in enumerate(sample_indices):
        raw_img = Image.open(X_test[idx]).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        rgb_img = np.array(raw_img) / 255.0
        inp     = val_test_transforms(raw_img).unsqueeze(0).to(DEVICE)

        gc    = cam(input_tensor=inp,    targets=None)
        gc_pp = cam_pp(input_tensor=inp, targets=None)

        cam_img    = show_cam_on_image(rgb_img.astype(np.float32), gc[0],    use_rgb=True)
        cam_pp_img = show_cam_on_image(rgb_img.astype(np.float32), gc_pp[0], use_rgb=True)

        true_lbl = "Glaucoma" if y_test[idx] == 1 else "Normal"
        color    = 'red' if y_test[idx] == 1 else 'green'

        # Row 0: original
        axes[0][i].imshow(raw_img)
        axes[0][i].set_title(f"GT: {true_lbl}", color=color, fontweight='bold')
        axes[0][i].axis('off')

        # Row 1: Grad-CAM
        axes[1][i].imshow(cam_img)
        axes[1][i].set_title("Vim SSM Focus", fontsize=10)
        axes[1][i].axis('off')

        # Row 2: Grad-CAM++
        axes[2][i].imshow(cam_pp_img)
        axes[2][i].set_title("Grad-CAM++ (Sharper)", fontsize=10)
        axes[2][i].axis('off')

    plt.suptitle(
        "XAI — Vision Mamba Grad-CAM | Grad-CAM++\n"
        "Bidirectional SSM Focus: Optic Disc & Cup Region",
        fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()
    print("✅ Grad-CAM visualization complete!")

except Exception as e:
    print(f"Grad-CAM error: {e}")

# ============================================================
# CELL 17 — Robustness Test (FIXED — no double evaluate call)
# ============================================================
print("\n🛡️  Robustness Evaluation...")
print("─" * 55)

class NoisyDataset(GlaucomaDataset):
    def __init__(self, paths, labels, transform, sigma):
        super().__init__(paths, labels, transform)
        self.sigma = sigma
    def __getitem__(self, idx):
        img, lbl = super().__getitem__(idx)
        if self.sigma > 0:
            img = img + torch.randn_like(img) * self.sigma
        return img, lbl

noise_levels = [0.0, 0.05, 0.10, 0.20]
rob_results  = []

for sigma in noise_levels:
    noisy_loader = DataLoader(
        NoisyDataset(X_test, y_test, val_test_transforms, sigma),
        BATCH_SIZE, shuffle=False, num_workers=2)

    # ✅ FIX: single evaluate call, use probs_n directly
    _, acc_n, probs_n, _, true_n = evaluate(model, noisy_loader, criterion)
    auc_n = roc_auc_score(true_n, probs_n)

    rob_results.append({'sigma': sigma, 'acc': acc_n*100, 'auc': auc_n})
    print(f"  Noise σ={sigma:.2f} → Acc: {acc_n*100:.2f}%  AUC: {auc_n:.4f}")

# ── Plot ─────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sigmas = [r['sigma'] for r in rob_results]
accs_r = [r['acc']   for r in rob_results]
aucs_r = [r['auc']   for r in rob_results]

axes[0].plot(sigmas, accs_r, 'b-o', ms=8, lw=2)
axes[0].fill_between(sigmas, accs_r, alpha=0.1, color='blue')
axes[0].set_title("Robustness — Accuracy vs Noise", fontweight='bold')
axes[0].set_xlabel("Gaussian Noise σ"); axes[0].set_ylabel("Accuracy (%)")
axes[0].grid(alpha=0.4); axes[0].set_ylim([50, 101])
for x, y in zip(sigmas, accs_r):
    axes[0].annotate(f"{y:.1f}%", (x, y), textcoords="offset points",
                     xytext=(0, 8), ha='center', fontsize=9)

axes[1].plot(sigmas, aucs_r, 'r-s', ms=8, lw=2)
axes[1].fill_between(sigmas, aucs_r, alpha=0.1, color='red')
axes[1].set_title("Robustness — AUC vs Noise", fontweight='bold')
axes[1].set_xlabel("Gaussian Noise σ"); axes[1].set_ylabel("AUC")
axes[1].grid(alpha=0.4); axes[1].set_ylim([0.5, 1.01])
for x, y in zip(sigmas, aucs_r):
    axes[1].annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                     xytext=(0, 8), ha='center', fontsize=9)

plt.suptitle("Vision Mamba (Vim) — Robustness Under Gaussian Noise",
             fontsize=14, fontweight='bold')
plt.tight_layout(); plt.show()

print("\n" + "═"*55)
print("📄  ROBUSTNESS SUMMARY — VISION MAMBA")
print("═"*55)
for r in rob_results:
    bar = "█" * int(r['acc'] / 5)
    print(f"  σ={r['sigma']:.2f}  Acc: {r['acc']:6.2f}%  AUC: {r['auc']:.4f}  {bar}")
print("═"*55)