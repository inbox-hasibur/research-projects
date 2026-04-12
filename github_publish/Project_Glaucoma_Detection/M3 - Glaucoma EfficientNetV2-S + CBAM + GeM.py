# =============================================================================
# 🔬 GLAUCOMA DETECTION — MODEL #3
# EfficientNetV2-S + CBAM + GeM Pooling (The Optimized Classic)
# ACRIMA Dataset | XAI (Grad-CAM) + Robustness Testing
# =============================================================================

# ============================================================
# CELL 1 — Install
# ============================================================
!pip install -q timm scikit-learn matplotlib seaborn einops
!pip install -q grad-cam

# ============================================================
# CELL 2 — Imports & Config
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
warnings.filterwarnings('ignore')

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Device: {DEVICE}")

# ── Hyperparameters ───────────────────────────────────────────
IMG_SIZE     = 224
BATCH_SIZE   = 32          # EfficientNetV2 is lightweight → bigger batch
EPOCHS       = 25
LR           = 1e-4
WEIGHT_DECAY = 1e-4
LABEL_SMOOTH = 0.1
PATIENCE     = 5

DATA_DIR  = "/kaggle/input/datasets/hindsaud/datasets-higancnn-glaucoma-detection/Datasets/Acrima"
SAVE_PATH = "best_efficientv2_glaucoma.pth"

# ============================================================
# CELL 3 — Architecture Diagram
# ============================================================
#
#  Input Image (224×224)
#       │
#  ┌────▼──────────────────────┐
#  │  EfficientNetV2-S          │  ← Fast, accurate CNN backbone
#  │  (pretrained ImageNet)     │    6 Fused-MBConv stages
#  │  Output: (B, 1280, 7, 7)  │    Much faster than EfficientNetV1
#  └────┬──────────────────────┘
#       │
#  ┌────▼──────────────────────┐
#  │  CBAM Attention           │  ← Channel attention: WHICH features?
#  │  Channel + Spatial        │    Spatial attention: WHERE to look?
#  └────┬──────────────────────┘
#       │
#  ┌────▼──────────────────────┐
#  │  Multi-Scale Feature      │  ← Extract features at 3 scales
#  │  Aggregation (MSFA)       │    (7×7, 14×14, 28×28) then fuse
#  └────┬──────────────────────┘
#       │
#  ┌────▼──────────────────────┐
#  │  GeM Pooling (p=3)        │  ← Better than AvgPool for medical
#  └────┬──────────────────────┘
#       │
#  ┌────▼──────────────────────┐
#  │  MLP Head (Dropout 0.4)   │
#  │  → Binary Output          │
#  └──────────────────────────┘

# ============================================================
# CELL 4 — CBAM Module
# ============================================================
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels, bias=False),
        )

    def forward(self, x):
        B, C, _, _ = x.shape
        avg = self.mlp(self.avg_pool(x).view(B, C))
        mx  = self.mlp(self.max_pool(x).view(B, C))
        return torch.sigmoid(avg + mx).view(B, C, 1, 1)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size,
                              padding=kernel_size // 2, bias=False)

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx  = x.max(dim=1, keepdim=True).values
        return torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))


class CBAM(nn.Module):
    """Full CBAM: Channel Attention → Spatial Attention."""
    def __init__(self, channels, reduction=16, spatial_kernel=7):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention(spatial_kernel)

    def forward(self, x):
        x = x * self.ca(x)   # channel-wise re-weighting
        x = x * self.sa(x)   # spatial re-weighting
        return x

# ============================================================
# CELL 5 — GeM Pooling
# ============================================================
class GeMPooling(nn.Module):
    """
    Generalized Mean Pooling.
    p=1  → Average Pooling
    p=∞  → Max Pooling
    p=3  → Best for fine-grained / medical (learned)
    """
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p   = nn.Parameter(torch.tensor(p))
        self.eps = eps

    def forward(self, x):
        return F.adaptive_avg_pool2d(
            x.clamp(min=self.eps).pow(self.p), 1
        ).pow(1.0 / self.p).flatten(1)

    def __repr__(self):
        return f"GeMPooling(p={self.p.item():.2f})"

# ============================================================
# CELL 6 — Multi-Scale Feature Aggregation (MSFA)
# ============================================================
class MultiScaleFeatureAgg(nn.Module):
    """
    Takes EfficientNetV2 intermediate features at 3 scales,
    projects them to the same dim, then fuses via learned weights.
    This lets the model use both fine (blood vessels) and
    coarse (overall disc shape) features simultaneously.
    """
    def __init__(self, in_channels_list, out_channels=512):
        super().__init__()
        self.projectors = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(7),
                nn.Conv2d(c, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.GELU(),
            )
            for c in in_channels_list
        ])
        # Learned scale weights
        self.scale_weights = nn.Parameter(
            torch.ones(len(in_channels_list)) / len(in_channels_list)
        )

    def forward(self, feature_list):
        projs = [proj(f) for proj, f in zip(self.projectors, feature_list)]
        w = torch.softmax(self.scale_weights, dim=0)
        fused = sum(w[i] * projs[i] for i in range(len(projs)))
        return fused                                # (B, out_channels, 7, 7)

# ============================================================
# CELL 7 — Full EfficientNetV2 + CBAM + GeM Model
# ============================================================
class EfficientNetV2CBAM(nn.Module):
    """
    EfficientNetV2-S backbone
    + Multi-Scale Feature Aggregation (3 stages)
    + CBAM Attention
    + GeM Pooling
    + Dropout MLP Head
    """
    def __init__(self, num_classes=1, drop=0.4):
        super().__init__()

        # ── EfficientNetV2-S feature extractor ───────────────────
        # stages 1,2,3 → channels: [48, 64, 160] (approx for EfficientNetV2-S)
        self.backbone = timm.create_model(
            'tf_efficientnetv2_s',
            pretrained=True,
            features_only=True,
            out_indices=[1, 2, 3, 4]   # 4 stages
        )
        feat_info = self.backbone.feature_info.channels()
        # feat_info typically: [48, 64, 160, 256] for EfficientNetV2-S
        print(f"  EfficientNetV2-S feature channels: {feat_info}")

        # ── Multi-Scale Feature Aggregation (stages 1,2,3) ───────
        FUSED_DIM = 512
        self.msfa = MultiScaleFeatureAgg(feat_info[1:4], out_channels=FUSED_DIM)

        # ── CBAM on fused feature map ─────────────────────────────
        self.cbam = CBAM(FUSED_DIM, reduction=16, spatial_kernel=7)

        # Also apply CBAM on deepest stage separately (stage 4)
        self.cbam_deep = CBAM(feat_info[3], reduction=16)
        self.deep_proj = nn.Sequential(
            nn.Conv2d(feat_info[3], FUSED_DIM, 1, bias=False),
            nn.BatchNorm2d(FUSED_DIM),
            nn.GELU(),
        )

        # ── GeM Pooling ───────────────────────────────────────────
        self.gem = GeMPooling(p=3.0)

        # Final feature = fused (512) + deep (512) = 1024
        FINAL_DIM = FUSED_DIM * 2

        # ── Classification Head ───────────────────────────────────
        self.head = nn.Sequential(
            nn.Linear(FINAL_DIM, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(drop / 2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        feats = self.backbone(x)          # list of 4 feature maps

        # Multi-scale aggregation on stages 1,2,3
        fused = self.msfa(feats[1:4])     # (B, 512, 7, 7)
        fused = self.cbam(fused)          # apply CBAM

        # Deep stage 4 with its own CBAM
        deep  = self.cbam_deep(feats[3])  # (B, deep_ch, 7, 7)
        deep  = self.deep_proj(deep)      # (B, 512, 7, 7)

        # GeM pool both branches
        pooled_fused = self.gem(fused)    # (B, 512)
        pooled_deep  = self.gem(deep)     # (B, 512)

        # Concat → (B, 1024)
        out = torch.cat([pooled_fused, pooled_deep], dim=1)
        return self.head(out)


def build_model():
    return EfficientNetV2CBAM(num_classes=1, drop=0.4)

# ============================================================
# CELL 8 — Dataset & Augmentation
# ============================================================
mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.3, contrast=0.3,
                           saturation=0.2, hue=0.05),
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
# CELL 9 — Label Smoothing Loss
# ============================================================
class BCELabelSmoothing(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.s = smoothing
    def forward(self, logits, targets):
        t = targets * (1 - self.s) + 0.5 * self.s
        return F.binary_cross_entropy_with_logits(logits, t)

# ============================================================
# CELL 10 — Early Stopping
# ============================================================
class EarlyStopping:
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
            return False
        self.counter += 1
        return self.counter >= self.patience

# ============================================================
# CELL 11 — Train / Eval Functions
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
        out   = model(imgs)
        total += criterion(out, lbls).item()
        p = torch.sigmoid(out).cpu().numpy()
        probs_all.extend(p.flatten())
        preds_all.extend((p > 0.5).astype(int).flatten())
        lbls_all.extend(lbls.cpu().numpy().flatten())
    return (total / len(loader),
            accuracy_score(lbls_all, preds_all),
            probs_all, preds_all, lbls_all)

# ============================================================
# CELL 12 — Load Data
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

print(f"📂 Total: {len(image_paths)} "
      f"(Normal={labels.count(0)}, Glaucoma={labels.count(1)})")

X_tv, X_test, y_tv, y_test = train_test_split(
    image_paths, labels, test_size=0.15, random_state=42, stratify=labels)
X_train, X_val, y_train, y_val = train_test_split(
    X_tv, y_tv, test_size=0.176, random_state=42, stratify=y_tv)

print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

train_loader = DataLoader(
    GlaucomaDataset(X_train, y_train, train_transforms),
    BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
val_loader   = DataLoader(
    GlaucomaDataset(X_val, y_val, val_test_transforms),
    BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
test_loader  = DataLoader(
    GlaucomaDataset(X_test, y_test, val_test_transforms),
    BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

# ============================================================
# CELL 13 — Build Model
# ============================================================
model  = build_model().to(DEVICE)
params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"📐 Trainable Parameters: {params:,}")
print(f"   GeM p value (learned): {model.gem.p.item():.2f}")

# Differential LR: backbone slow, new heads fast
backbone_params = list(model.backbone.parameters())
new_params      = (list(model.msfa.parameters())      +
                   list(model.cbam.parameters())       +
                   list(model.cbam_deep.parameters())  +
                   list(model.deep_proj.parameters())  +
                   list(model.gem.parameters())        +
                   list(model.head.parameters()))

optimizer = optim.AdamW([
    {'params': backbone_params, 'lr': LR * 0.1},  # 1e-5
    {'params': new_params,      'lr': LR},         # 1e-4
], weight_decay=WEIGHT_DECAY)

criterion = BCELabelSmoothing(LABEL_SMOOTH)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=10, T_mult=2)
scaler    = torch.cuda.amp.GradScaler()
stopper   = EarlyStopping(patience=PATIENCE, save_path=SAVE_PATH)

# ============================================================
# CELL 14 — Training Loop
# ============================================================
train_losses, val_losses, val_accs, lrs = [], [], [], []
print("🔥 Training EfficientNetV2 + CBAM + GeM...")
print("─" * 72)

for epoch in range(1, EPOCHS + 1):
    t0      = time.time()
    tr_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler)
    vl_loss, vl_acc, _, _, _ = evaluate(model, val_loader, criterion)
    scheduler.step()

    train_losses.append(tr_loss)
    val_losses.append(vl_loss)
    val_accs.append(vl_acc)
    lrs.append(optimizer.param_groups[1]['lr'])

    stop = stopper.step(vl_loss, epoch, model)
    flag = ("🏅 BEST" if stopper.counter == 0
            else f"  (patience {stopper.counter}/{PATIENCE})")

    print(f"Epoch {epoch:02d}/{EPOCHS} | {time.time()-t0:.0f}s | "
          f"Train: {tr_loss:.4f} | Val: {vl_loss:.4f} | "
          f"Acc: {vl_acc*100:.2f}% | LR: {lrs[-1]:.2e} {flag}")

    if stop:
        print(f"\n⏹️  Early stopping @ epoch {epoch}. "
              f"Best: epoch {stopper.best_epoch} "
              f"(val_loss={stopper.best_loss:.4f})")
        break

print(f"\n✅ Training done → {SAVE_PATH}")

# ============================================================
# CELL 15 — Training Curves
# ============================================================
ep = range(1, len(train_losses) + 1)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].plot(ep, train_losses, 'b-o', ms=4, label='Train')
axes[0].plot(ep, val_losses,   'r-o', ms=4, label='Val')
axes[0].axvline(stopper.best_epoch, color='g', linestyle='--',
                label=f'Best (ep {stopper.best_epoch})')
axes[0].set_title("Loss Curves", fontweight='bold')
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
axes[0].legend(); axes[0].grid(alpha=0.4)

axes[1].plot(ep, [a*100 for a in val_accs], 'g-s', ms=4)
axes[1].set_title("Validation Accuracy (%)", fontweight='bold')
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy (%)")
axes[1].grid(alpha=0.4)

axes[2].plot(ep, lrs, 'm-', lw=2)
axes[2].set_title("Learning Rate (head)", fontweight='bold')
axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("LR")
axes[2].grid(alpha=0.4)

plt.suptitle("EfficientNetV2 + CBAM + GeM — Training Dashboard",
             fontsize=15, fontweight='bold')
plt.tight_layout(); plt.show()

# ============================================================
# CELL 16 — Final Test Evaluation
# ============================================================
model.load_state_dict(torch.load(SAVE_PATH))
_, _, test_probs, test_preds, test_true = evaluate(model, test_loader, criterion)

acc  = accuracy_score(test_true, test_preds)
prec = precision_score(test_true, test_preds, zero_division=0)
rec  = recall_score(test_true, test_preds, zero_division=0)
spec = recall_score(test_true, test_preds, pos_label=0, zero_division=0)
f1   = f1_score(test_true, test_preds, zero_division=0)
auc  = roc_auc_score(test_true, test_probs)

print("\n" + "═"*54)
print("🏆  EfficientNetV2 + CBAM + GeM — FINAL TEST RESULTS")
print("═"*54)
print(f"  Accuracy        : {acc*100:.2f}%")
print(f"  Precision       : {prec*100:.2f}%")
print(f"  Sensitivity     : {rec*100:.2f}%")
print(f"  Specificity     : {spec*100:.2f}%")
print(f"  F1-Score        : {f1*100:.2f}%")
print(f"  AUC-ROC         : {auc:.4f}")
print("═"*54)
print(classification_report(test_true, test_preds,
                             target_names=['Normal', 'Glaucoma']))

# ============================================================
# CELL 17 — Confusion Matrix + ROC + Score Distribution
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

cm = confusion_matrix(test_true, test_preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', ax=axes[0],
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

plt.suptitle("EfficientNetV2 + CBAM + GeM — Test Evaluation",
             fontsize=16, fontweight='bold')
plt.tight_layout(); plt.show()

# ============================================================
# CELL 18 — XAI: Grad-CAM (3 variants for paper)
# ============================================================
try:
    from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, EigenCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image

    # Target: last conv block of EfficientNetV2
    # blocks[-1] is the final stage
    target_layers = [model.backbone.blocks[-1][-1]]

    cam_standard = GradCAM(model=model, target_layers=target_layers)
    cam_pp       = GradCAMPlusPlus(model=model, target_layers=target_layers)
    cam_eigen    = EigenCAM(model=model, target_layers=target_layers)

    sample_indices = np.random.choice(len(X_test), 4, replace=False)
    fig, axes = plt.subplots(4, 4, figsize=(20, 18))

    row_titles = ["Original", "Grad-CAM", "Grad-CAM++", "EigenCAM"]
    for ax, title in zip(axes[:, 0], row_titles):
        ax.set_ylabel(title, fontsize=12, fontweight='bold', rotation=90)

    for i, idx in enumerate(sample_indices):
        raw_img = Image.open(X_test[idx]).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        rgb_img = np.array(raw_img) / 255.0
        inp     = val_test_transforms(raw_img).unsqueeze(0).to(DEVICE)

        gc        = cam_standard(input_tensor=inp, targets=None)
        gc_pp     = cam_pp(input_tensor=inp, targets=None)
        gc_eigen  = cam_eigen(input_tensor=inp, targets=None)

        overlays = [
            rgb_img,
            show_cam_on_image(rgb_img.astype(np.float32), gc[0],       use_rgb=True),
            show_cam_on_image(rgb_img.astype(np.float32), gc_pp[0],    use_rgb=True),
            show_cam_on_image(rgb_img.astype(np.float32), gc_eigen[0], use_rgb=True),
        ]

        true_lbl = "Glaucoma" if y_test[idx] == 1 else "Normal"
        color    = 'red' if y_test[idx] == 1 else 'green'

        for row, overlay in enumerate(overlays):
            axes[row][i].imshow(overlay)
            axes[row][i].axis('off')
            if row == 0:
                axes[row][i].set_title(f"GT: {true_lbl}",
                                       color=color, fontweight='bold')

    plt.suptitle("XAI Comparison — Grad-CAM vs Grad-CAM++ vs EigenCAM\n"
                 "EfficientNetV2 + CBAM + GeM | Optic Disc Focus",
                 fontsize=13, fontweight='bold')
    plt.tight_layout(); plt.show()
    print("✅ XAI visualization done!")

except Exception as e:
    print(f"Grad-CAM skipped: {e}")

# ============================================================
# CELL 19 — XAI: CBAM Attention Visualization
# ============================================================
# Show what CBAM's spatial attention map looks like
# This is unique to Model #3 — great for paper!
try:
    def get_cbam_spatial_attention(model, img_tensor):
        """Hook to extract CBAM spatial attention map."""
        attention_maps = {}

        def hook_fn(module, input, output):
            attention_maps['spatial'] = output.detach()

        hook = model.cbam.sa.conv.register_forward_hook(hook_fn)
        with torch.no_grad():
            _ = model(img_tensor)
        hook.remove()
        return torch.sigmoid(attention_maps['spatial'])

    sample_indices3 = np.random.choice(len(X_test), 5, replace=False)
    fig, axes = plt.subplots(2, 5, figsize=(22, 9))

    for i, idx in enumerate(sample_indices3):
        raw_img    = Image.open(X_test[idx]).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        inp        = val_test_transforms(raw_img).unsqueeze(0).to(DEVICE)

        attn_map   = get_cbam_spatial_attention(model, inp)  # (1,1,H,W)
        attn_np    = attn_map.squeeze().cpu().numpy()
        attn_resized = np.array(
            Image.fromarray((attn_np * 255).astype(np.uint8)).resize((IMG_SIZE, IMG_SIZE))
        ) / 255.0

        rgb_img    = np.array(raw_img) / 255.0
        true_lbl   = "Glaucoma" if y_test[idx] == 1 else "Normal"
        color      = 'red' if y_test[idx] == 1 else 'green'

        axes[0][i].imshow(raw_img)
        axes[0][i].set_title(f"GT: {true_lbl}", color=color, fontweight='bold')
        axes[0][i].axis('off')

        axes[1][i].imshow(raw_img)
        axes[1][i].imshow(attn_resized, alpha=0.5, cmap='hot')
        axes[1][i].set_title("CBAM Spatial Attention", fontsize=10)
        axes[1][i].axis('off')

    plt.suptitle("CBAM Spatial Attention Maps — Where the Model Focuses\n"
                 "(Bright = High Attention | Optic disc area should be bright)",
                 fontsize=13, fontweight='bold')
    plt.tight_layout(); plt.show()
    print("✅ CBAM attention visualization done!")

except Exception as e:
    print(f"CBAM visualization skipped: {e}")

# ============================================================
# CELL 20 — Robustness Test
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
    _, acc_n, probs_n, _, true_n = evaluate(model, noisy_loader, criterion)
    auc_n = roc_auc_score(true_n, probs_n)
    rob_results.append({'sigma': sigma, 'acc': acc_n*100, 'auc': auc_n})
    print(f"  Noise σ={sigma:.2f} → Acc: {acc_n*100:.2f}%  AUC: {auc_n:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sigmas = [r['sigma'] for r in rob_results]
accs_r = [r['acc']   for r in rob_results]
aucs_r = [r['auc']   for r in rob_results]

axes[0].plot(sigmas, accs_r, 'darkorange', marker='o', ms=8, lw=2)
axes[0].set_title("Robustness — Accuracy vs Noise", fontweight='bold')
axes[0].set_xlabel("Gaussian Noise σ"); axes[0].set_ylabel("Accuracy (%)")
axes[0].grid(alpha=0.4); axes[0].set_ylim([50, 101])

axes[1].plot(sigmas, aucs_r, 'r-s', ms=8, lw=2)
axes[1].set_title("Robustness — AUC vs Noise", fontweight='bold')
axes[1].set_xlabel("Gaussian Noise σ"); axes[1].set_ylabel("AUC")
axes[1].grid(alpha=0.4); axes[1].set_ylim([0.5, 1.01])

plt.suptitle("EfficientNetV2 + CBAM + GeM — Robustness Under Gaussian Noise",
             fontsize=14, fontweight='bold')
plt.tight_layout(); plt.show()

# ============================================================
# CELL 21 — Paper Summary Table
# ============================================================
print("\n" + "═"*60)
print("📄  PAPER METRICS — EfficientNetV2 + CBAM + GeM")
print("═"*60)
print(f"  Architecture    : EfficientNetV2-S + MSFA + CBAM + GeM")
print(f"  Dataset         : ACRIMA ({len(image_paths)} images)")
print(f"  Parameters      : {params:,}")
print(f"  Epochs trained  : {stopper.best_epoch} (early stop)")
print(f"  Batch Size      : {BATCH_SIZE}")
print(f"  Optimizer       : AdamW (differential LR)")
print(f"  Scheduler       : CosineAnnealingWarmRestarts")
print("─"*60)
print(f"  Accuracy        : {acc*100:.2f}%")
print(f"  Sensitivity     : {rec*100:.2f}%")
print(f"  Specificity     : {spec*100:.2f}%")
print(f"  Precision       : {prec*100:.2f}%")
print(f"  F1-Score        : {f1*100:.2f}%")
print(f"  AUC-ROC         : {auc:.4f}")
print("─"*60)
print("  Robustness:")
for r in rob_results:
    print(f"    σ={r['sigma']:.2f} → Acc: {r['acc']:.2f}%  AUC: {r['auc']:.4f}")
print("═"*60)

# ============================================================
# CELL 22 — 3-Model Comparison Table (fill after all 3 done)
# ============================================================
print("\n" + "═"*68)
print("📊  COMPARISON TABLE — Fill after running all 3 models")
print("═"*68)
print(f"{'Model':<35} {'Acc':>7} {'F1':>7} {'AUC':>7} {'Params':>12}")
print("─"*68)
print(f"{'1. Vision Mamba (Swin+Vim)':<35} {'99.78%':>7} {'99.78%':>7} {'1.000':>7} {'32.4M':>12}")
print(f"{'2. Hybrid ConvNeXt+Swin':<35} {'  ?%':>7} {'  ?%':>7} {'?.???':>7} {'  ?M':>12}")
print(f"{'3. EfficientNetV2+CBAM+GeM':<35} {'  ?%':>7} {'  ?%':>7} {'?.???':>7} {'  ?M':>12}")
print("═"*68)
print("  ← Update after Model 2 & 3 finish running")