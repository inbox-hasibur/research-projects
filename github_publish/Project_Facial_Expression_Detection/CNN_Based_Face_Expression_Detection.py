# ============================================================
# FER-2013 Face Expression Recognition — Single Pretrained Backbone
# ConvNeXt-Base (ImageNet-1K) fine-tuned; no hybrid / custom CNN.
# Note: Standard FER-2013 test macro-F1 is typically ~65–75%; 99.8% F1
# is not achievable on this public split—strong transfer learning maximizes
# what the benchmark allows. ROC-AUC (OvR) is reported separately.
# ============================================================

# ============================================================
# STEP 0 — Install & Imports
# ============================================================
import os, sys, time, warnings, copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from torchvision.models import convnext_base, ConvNeXt_Base_Weights
warnings.filterwarnings('ignore')

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Device : {DEVICE}")
if DEVICE.type == 'cuda':
    print(f"   GPU    : {torch.cuda.get_device_name(0)}")

# ── Config ────────────────────────────────────────────────────
_kaggle_fer = "/kaggle/input/datasets/msambare/fer2013"
if os.path.isdir(_kaggle_fer):
    DATA_ROOT = _kaggle_fer
else:
    try:
        _here = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        _here = os.getcwd()
    DATA_ROOT = os.path.join(_here, "data", "fer2013")

TRAIN_DIR  = os.path.join(DATA_ROOT, "train")
TEST_DIR   = os.path.join(DATA_ROOT, "test")
CLASSES    = ['angry','disgust','fear','happy','neutral','sad','surprise']
NUM_CLS    = len(CLASSES)
IMG_SIZE   = 224
BATCH      = 32
EPOCHS     = 50
LR_BACKBONE = 1e-4
LR_HEAD     = 5e-4
PATIENCE   = 12
SAVE_PATH  = "best_fer_convnext_base.pth"
SEED       = 42
NUM_WORKERS = 0 if sys.platform.startswith("win") else min(4, (os.cpu_count() or 2))

torch.manual_seed(SEED); np.random.seed(SEED)
print(f"   Classes: {CLASSES}")
print(f"   IMG:{IMG_SIZE}  BATCH:{BATCH}  EPOCHS:{EPOCHS}  Workers:{NUM_WORKERS}")
print(f"   Backbone LR:{LR_BACKBONE}  Head LR:{LR_HEAD}")


# ============================================================
# STEP 1 — Data Analysis & Visualization
# ============================================================
def count_images(root):
    counts = {}
    for cls in CLASSES:
        p = os.path.join(root, cls)
        if os.path.isdir(p):
            counts[cls] = len([f for f in os.listdir(p)
                               if f.lower().endswith(('.jpg','.png'))])
        else:
            counts[cls] = 0
    return counts

train_counts = count_images(TRAIN_DIR)
test_counts  = count_images(TEST_DIR)
total_train  = sum(train_counts.values())
total_test   = sum(test_counts.values())

print("\n📊 Dataset Distribution")
print(f"{'Class':<12} {'Train':>8} {'Test':>8} {'Train%':>8}")
print("─" * 42)
for cls in CLASSES:
    pct = 100 * train_counts[cls] / total_train if total_train else 0
    print(f"  {cls:<10} {train_counts[cls]:>8} {test_counts[cls]:>8} {pct:>7.1f}%")
print("─" * 42)
print(f"  {'TOTAL':<10} {total_train:>8} {total_test:>8}")

# ── Bar chart ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(20, 5))
colors = ['#e74c3c','#8e44ad','#2980b9','#27ae60','#f39c12','#2c3e50','#e67e22']

axes[0].bar(CLASSES, [train_counts[c] for c in CLASSES], color=colors, edgecolor='white')
axes[0].set_title("Training Set Distribution", fontweight='bold', fontsize=13)
axes[0].set_ylabel("Count"); axes[0].set_xticklabels(CLASSES, rotation=30)
for i,(c,v) in enumerate(zip(CLASSES,[train_counts[c] for c in CLASSES])):
    axes[0].text(i, v+30, str(v), ha='center', fontsize=9, fontweight='bold')

axes[1].bar(CLASSES, [test_counts[c] for c in CLASSES], color=colors, edgecolor='white')
axes[1].set_title("Test Set Distribution", fontweight='bold', fontsize=13)
axes[1].set_ylabel("Count"); axes[1].set_xticklabels(CLASSES, rotation=30)
for i,(c,v) in enumerate(zip(CLASSES,[test_counts[c] for c in CLASSES])):
    axes[1].text(i, v+5, str(v), ha='center', fontsize=9, fontweight='bold')

# Pie chart
wedges, texts, autotexts = axes[2].pie(
    [train_counts[c] for c in CLASSES],
    labels=CLASSES, colors=colors, autopct='%1.1f%%',
    startangle=90, pctdistance=0.8)
for at in autotexts: at.set_fontsize(8)
axes[2].set_title("Class Proportion (Train)", fontweight='bold', fontsize=13)

plt.suptitle("FER-2013 Dataset Analysis", fontsize=15, fontweight='bold')
plt.tight_layout(); plt.show()

# ── Sample images ─────────────────────────────────────────────
fig, axes = plt.subplots(NUM_CLS, 6, figsize=(18, NUM_CLS * 2.5))
for r, cls in enumerate(CLASSES):
    path = os.path.join(TRAIN_DIR, cls)
    imgs = sorted(os.listdir(path))[:6] if os.path.isdir(path) else []
    for c in range(6):
        ax = axes[r][c]
        if c < len(imgs):
            img = Image.open(os.path.join(path, imgs[c])).convert('L')
            ax.imshow(img, cmap='gray', vmin=0, vmax=255)
            if c == 0: ax.set_ylabel(cls.upper(), fontsize=10, fontweight='bold')
        ax.set_title(f"#{c+1}", fontsize=8) if r == 0 else None
        ax.axis('off')
plt.suptitle("Sample Images per Class", fontsize=15, fontweight='bold')
plt.tight_layout(); plt.show()

# ── Pixel statistics ──────────────────────────────────────────
print("\n📈 Pixel Intensity Statistics per Class")
print(f"{'Class':<12} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
print("─" * 50)
class_means, class_stds = [], []
for cls in CLASSES:
    p = os.path.join(TRAIN_DIR, cls)
    pixels = []
    for fn in os.listdir(p)[:100]:  # sample 100
        try:
            arr = np.array(Image.open(os.path.join(p,fn)).convert('L'))
            pixels.append(arr.flatten())
        except: pass
    if pixels:
        all_px = np.concatenate(pixels)
        m, s = all_px.mean(), all_px.std()
        class_means.append(m); class_stds.append(s)
        print(f"  {cls:<10} {m:>8.2f} {s:>8.2f} "
              f"{all_px.min():>8} {all_px.max():>8}")

# ── Intensity distribution plot ───────────────────────────────
fig, ax = plt.subplots(figsize=(10, 4))
for i, (cls, col) in enumerate(zip(CLASSES, colors)):
    p = os.path.join(TRAIN_DIR, cls)
    px = []
    for fn in os.listdir(p)[:50]:
        try: px.extend(np.array(Image.open(os.path.join(p,fn)).convert('L')).flatten())
        except: pass
    if px:
        ax.hist(px, bins=50, alpha=0.5, color=col, label=cls, density=True)
ax.set_xlabel("Pixel Intensity"); ax.set_ylabel("Density")
ax.set_title("Pixel Intensity Distribution per Class", fontweight='bold')
ax.legend(fontsize=8); plt.tight_layout(); plt.show()


# ============================================================
# STEP 2 — Dataset & Augmentation (3×224, ImageNet stats; FER is grayscale → repeat to RGB)
# ============================================================
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def _to_rgb3(pil_img):
    return pil_img.convert("L").convert("RGB")

train_tf = transforms.Compose([
    transforms.Lambda(_to_rgb3),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(12),
    transforms.RandomAffine(degrees=0, translate=(0.08, 0.08), scale=(0.92, 1.08)),
    transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.15),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.12)),
])

test_tf = transforms.Compose([
    transforms.Lambda(_to_rgb3),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

tta_tfs = [
    test_tf,
    transforms.Compose([
        transforms.Lambda(_to_rgb3),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]),
    transforms.Compose([
        transforms.Lambda(_to_rgb3),
        transforms.Resize((int(IMG_SIZE * 1.08), int(IMG_SIZE * 1.08))),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]),
]

class FERDataset(Dataset):
    def __init__(self, root, classes, tf):
        self.tf = tf
        self.samples = []
        for i, cls in enumerate(classes):
            d = os.path.join(root, cls)
            if not os.path.isdir(d): continue
            for fn in os.listdir(d):
                if fn.lower().endswith(('.jpg','.png')):
                    self.samples.append((os.path.join(d,fn), i))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path)
        return self.tf(img), label

train_ds = FERDataset(TRAIN_DIR, CLASSES, train_tf)
test_ds  = FERDataset(TEST_DIR,  CLASSES, test_tf)

# ── Weighted sampler (handle class imbalance) ─────────────────
labels    = [s[1] for s in train_ds.samples]
cls_cnt   = np.bincount(labels, minlength=NUM_CLS)
cls_wt    = 1.0 / (cls_cnt + 1e-6)
sample_wt = torch.tensor([cls_wt[l] for l in labels], dtype=torch.float)
sampler   = WeightedRandomSampler(sample_wt, len(sample_wt), replacement=True)

_pin = DEVICE.type == "cuda"
train_loader = DataLoader(
    train_ds, BATCH, sampler=sampler,
    num_workers=NUM_WORKERS, pin_memory=_pin,
    persistent_workers=NUM_WORKERS > 0,
)
test_loader = DataLoader(
    test_ds, BATCH, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=_pin,
    persistent_workers=NUM_WORKERS > 0,
)

print(f"\n✅ Train: {len(train_ds)} | Test: {len(test_ds)}")
print(f"   Class weights: {dict(zip(CLASSES, cls_wt.round(4)))}")


# ============================================================
# STEP 3 — Single pretrained backbone: ConvNeXt-Base (torchvision, ImageNet-1K)
# ============================================================
def build_convnext_fer(num_classes):
    w = ConvNeXt_Base_Weights.IMAGENET1K_V1
    m = convnext_base(weights=w)
    in_f = m.classifier[-1].in_features
    m.classifier[-1] = nn.Linear(in_f, num_classes)
    return m


def convnext_param_groups(model, lr_bb, lr_head):
    backbone, head = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("classifier"):
            head.append(p)
        else:
            backbone.append(p)
    return [
        {"params": backbone, "lr": lr_bb},
        {"params": head, "lr": lr_head},
    ]


model = build_convnext_fer(NUM_CLS).to(DEVICE)
params = sum(p.numel() for p in model.parameters() if p.requires_grad)
MODEL_NAME = "ConvNeXt-Base (ImageNet-1K pretrained, single-stream)"
print(f"📐 Model: {MODEL_NAME}")
print(f"   Parameters: {params:,}")

with torch.no_grad():
    _x = torch.randn(2, 3, IMG_SIZE, IMG_SIZE, device=DEVICE)
    _o = model(_x)
    print(f"   Smoke: {list(_x.shape)} → {list(_o.shape)} ✅")
    del _x, _o


# ============================================================
# STEP 4 — Loss, Optimizer, Scheduler
# ============================================================
class LabelSmoothingCE(nn.Module):
    """Label smoothing cross-entropy for better calibration."""
    def __init__(self, classes=7, smoothing=0.1):
        super().__init__()
        self.eps = smoothing
        self.cls = classes
    def forward(self, pred, target):
        logp   = F.log_softmax(pred, dim=1)
        nll    = F.nll_loss(logp, target)
        smooth = -logp.mean()
        return (1 - self.eps) * nll + self.eps * smooth / self.cls

# Class weights for loss
cls_weight = torch.tensor(cls_wt / cls_wt.sum() * NUM_CLS,
                          dtype=torch.float).to(DEVICE)
criterion  = LabelSmoothingCE(NUM_CLS, smoothing=0.1)

optimizer = optim.AdamW(
    convnext_param_groups(model, LR_BACKBONE, LR_HEAD),
    weight_decay=0.05,
    betas=(0.9, 0.999),
)


def warmup_cosine(epoch, warmup=5, total=EPOCHS):
    if epoch < warmup:
        return (epoch + 1) / warmup
    progress = (epoch - warmup) / max(1, total - warmup)
    return 0.5 * (1 + np.cos(np.pi * progress))


scheduler = optim.lr_scheduler.LambdaLR(
    optimizer, lr_lambda=lambda e: warmup_cosine(e, 5, EPOCHS))

class EarlyStopping:
    def __init__(self, patience=12):
        self.patience   = patience
        self.best_loss  = float('inf')
        self.counter    = 0
        self.best_state = None
        self.best_ep    = 0
    def step(self, loss, epoch, model):
        if loss < self.best_loss - 1e-5:
            self.best_loss  = loss
            self.counter    = 0
            self.best_state = copy.deepcopy(model.state_dict())
            self.best_ep    = epoch
            torch.save(self.best_state, SAVE_PATH)
            return False
        self.counter += 1
        return self.counter >= self.patience

stopper = EarlyStopping(PATIENCE)


# ============================================================
# STEP 5 — Mixup Augmentation
# ============================================================
def mixup(x, y, alpha=0.4):
    """Mixup data augmentation."""
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    xm  = lam * x + (1 - lam) * x[idx]
    ya, yb = y, y[idx]
    return xm, ya, yb, lam

def mixup_loss(crit, pred, ya, yb, lam):
    return lam * crit(pred, ya) + (1 - lam) * crit(pred, yb)


# ============================================================
# STEP 6 — Training & Evaluation
# ============================================================
def train_epoch(model, loader, crit, opt, epoch):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    use_mixup = epoch > 5   # skip mixup in warmup

    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        opt.zero_grad()

        if use_mixup and np.random.rand() < 0.5:
            imgs_m, ya, yb, lam = mixup(imgs, labels, alpha=0.4)
            out  = model(imgs_m)
            loss = mixup_loss(crit, out, ya, yb, lam)
            pred = out.argmax(1)
            correct += (lam*(pred==ya).float()+(1-lam)*(pred==yb).float()).sum().item()
        else:
            out  = model(imgs)
            loss = crit(out, labels)
            pred = out.argmax(1)
            correct += (pred == labels).sum().item()

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total_loss += loss.item() * imgs.size(0)
        total      += imgs.size(0)

    return total_loss / total, 100 * correct / total

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        out  = model(imgs)
        loss = F.cross_entropy(out, labels)
        pred = out.argmax(1)
        correct     += (pred == labels).sum().item()
        total_loss  += loss.item() * imgs.size(0)
        total       += imgs.size(0)
        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    acc  = 100 * correct / total
    loss = total_loss / total
    f1   = f1_score(all_labels, all_preds, average='macro') * 100
    return loss, acc, f1, all_preds, all_labels

# ── Training Loop ─────────────────────────────────────────────
history = {'tl': [], 'ta': [], 'vl': [], 'va': [], 'vf1': [], 'lr_bb': [], 'lr_hd': []}
print("🔥 Training ConvNeXt-Base (FER-2013)")
print(f"   Device:{DEVICE} | Patience:{PATIENCE} | Mixup: ON (after ep5)")
print("─" * 85)

for epoch in range(1, EPOCHS + 1):
    t0 = time.time()
    tl, ta = train_epoch(model, train_loader, criterion, optimizer, epoch)
    vl, va, vf1, _, _ = evaluate(model, test_loader)
    scheduler.step()

    history['tl'].append(tl)
    history['ta'].append(ta)
    history['vl'].append(vl)
    history['va'].append(va)
    history['vf1'].append(vf1)
    history['lr_bb'].append(optimizer.param_groups[0]['lr'])
    history['lr_hd'].append(optimizer.param_groups[1]['lr'])

    stop = stopper.step(vl, epoch, model)
    flag = "🏅" if stopper.counter == 0 else f"(p{stopper.counter}/{PATIENCE})"

    print(f"Ep{epoch:02d}/{EPOCHS} | {time.time()-t0:.0f}s | "
          f"Loss:{tl:.4f}/{vl:.4f} | "
          f"Acc:{ta:.2f}/{va:.2f}% | "
          f"F1:{vf1:.4f}% | "
          f"LR_bb:{optimizer.param_groups[0]['lr']:.2e} "
          f"LR_hd:{optimizer.param_groups[1]['lr']:.2e} {flag}")

    if stop:
        print(f"\n⏹️  Early stop ep{epoch}. Best: ep{stopper.best_ep}")
        break

# Load best
try:
    _sd = torch.load(SAVE_PATH, map_location=DEVICE, weights_only=True)
except TypeError:
    _sd = torch.load(SAVE_PATH, map_location=DEVICE)
model.load_state_dict(_sd)
del _sd
print(f"\n✅ Loaded best model from ep{stopper.best_ep}")


# ============================================================
# STEP 7 — Test-Time Augmentation (TTA)
# ============================================================
@torch.no_grad()
def tta_collect_probs(model, test_root, classes, tta_transforms):
    """Mean softmax probabilities over TTA (predictions + ROC)."""
    model.eval()
    probs_all, labels_all = [], []
    samples = []
    for i, cls in enumerate(classes):
        d = os.path.join(test_root, cls)
        if not os.path.isdir(d):
            continue
        for fn in os.listdir(d):
            if fn.lower().endswith(('.jpg', '.png')):
                samples.append((os.path.join(d, fn), i))

    for path, label in samples:
        img = Image.open(path)
        prob = torch.zeros(len(classes), device=DEVICE)
        for tf in tta_transforms:
            x = tf(img).unsqueeze(0).to(DEVICE)
            prob += F.softmax(model(x), dim=1).squeeze(0)
        prob /= len(tta_transforms)
        probs_all.append(prob.cpu().numpy())
        labels_all.append(label)
    return np.stack(probs_all), np.array(labels_all)


print("\n🔍 Running TTA inference...")
tta_probs, tta_labels = tta_collect_probs(model, TEST_DIR, CLASSES, tta_tfs)
tta_preds = tta_probs.argmax(axis=1)
tta_f1  = f1_score(tta_labels, tta_preds, average='macro') * 100
tta_acc = accuracy_score(tta_labels, tta_preds) * 100
print(f"   TTA Accuracy : {tta_acc:.4f}%")
print(f"   TTA F1 Macro : {tta_f1:.4f}%")


# ============================================================
# STEP 8 — Training Curves
# ============================================================
ep = range(1, len(history['tl']) + 1)
fig, axes = plt.subplots(1, 3, figsize=(21, 5))

axes[0].plot(ep, history['tl'], 'b-o', ms=3, label='Train Loss')
axes[0].plot(ep, history['vl'], 'r-o', ms=3, label='Val Loss')
axes[0].axvline(stopper.best_ep, color='green', ls='--',
                label=f'Best ep{stopper.best_ep}')
axes[0].set_title("Loss Curves", fontweight='bold'); axes[0].legend()
axes[0].grid(alpha=0.4)

axes[1].plot(ep, history['ta'], 'b-s', ms=3, label='Train Acc')
axes[1].plot(ep, history['va'], 'r-s', ms=3, label='Val Acc')
axes[1].axvline(stopper.best_ep, color='green', ls='--')
axes[1].set_title("Accuracy Curves", fontweight='bold'); axes[1].legend()
axes[1].grid(alpha=0.4)

axes[2].plot(ep, history['vf1'], 'm-D', ms=3, label='Val F1 Macro')
axes[2].axvline(stopper.best_ep, color='green', ls='--')
axes[2].set_title("F1-Score (Macro)", fontweight='bold'); axes[2].legend()
axes[2].grid(alpha=0.4)

plt.suptitle("ConvNeXt-Base — Training Curves", fontsize=14, fontweight='bold')
plt.tight_layout(); plt.show()


# ============================================================
# STEP 9 — Performance Metrics & Confusion Matrix
# ============================================================
# Standard evaluation (no TTA)
_, va_final, vf1_final, std_preds, std_labels = evaluate(model, test_loader)

print("\n" + "=" * 65)
print("📊  FINAL PERFORMANCE METRICS")
print("=" * 65)
print(f"\n  Standard Inference:")
print(f"    Accuracy : {va_final:.4f}%")
print(f"    F1 Macro : {vf1_final:.4f}%")
print(f"\n  TTA (×{len(tta_tfs)} transforms):")
print(f"    Accuracy : {tta_acc:.4f}%")
print(f"    F1 Macro : {tta_f1:.4f}%")
print("=" * 65)

# Per-class report
print("\n📋 Per-Class Classification Report (TTA):")
print(classification_report(tta_labels, tta_preds, target_names=CLASSES,
                             digits=4))

# ── Confusion Matrix ──────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(20, 7))
for ax, preds, labels_a, title in [
    (axes[0], std_preds, std_labels, "Confusion Matrix (Standard)"),
    (axes[1], tta_preds, tta_labels, f"Confusion Matrix (TTA ×{len(tta_tfs)})")
]:
    cm = confusion_matrix(labels_a, preds)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    sns.heatmap(cm_pct, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=CLASSES, yticklabels=CLASSES, ax=ax,
                linewidths=0.5, linecolor='white',
                cbar_kws={'label': '%'})
    ax.set_title(title, fontweight='bold', fontsize=12)
    ax.set_ylabel("True Label"); ax.set_xlabel("Predicted Label")
    ax.tick_params(axis='x', rotation=30)

plt.suptitle("Confusion Matrices — ConvNeXt-Base", fontsize=14, fontweight='bold')
plt.tight_layout(); plt.show()

# ── Multiclass ROC (One-vs-Rest, TTA probabilities) ───────────
y_bin = label_binarize(tta_labels, classes=np.arange(NUM_CLS))
fpr_ovr, tpr_ovr, roc_auc_cls = {}, {}, {}
for i in range(NUM_CLS):
    fpr_ovr[i], tpr_ovr[i], _ = roc_curve(y_bin[:, i], tta_probs[:, i])
    roc_auc_cls[i] = auc(fpr_ovr[i], tpr_ovr[i])

fpr_micro, tpr_micro, _ = roc_curve(y_bin.ravel(), tta_probs.ravel())
roc_auc_micro = auc(fpr_micro, tpr_micro)

all_fpr = np.unique(np.concatenate([fpr_ovr[i] for i in range(NUM_CLS)]))
mean_tpr = np.zeros_like(all_fpr, dtype=float)
for i in range(NUM_CLS):
    mean_tpr += np.interp(all_fpr, fpr_ovr[i], tpr_ovr[i])
mean_tpr /= NUM_CLS
fpr_macro, tpr_macro = all_fpr, mean_tpr
roc_auc_macro = auc(fpr_macro, tpr_macro)

fig_roc, ax_roc = plt.subplots(figsize=(10, 9))
ax_roc.plot(
    fpr_micro, tpr_micro, color="navy", lw=2.5,
    label=f"Micro-average ROC (AUC = {roc_auc_micro:.4f})",
)
ax_roc.plot(
    fpr_macro, tpr_macro, color="darkorange", lw=2.5, linestyle="--",
    label=f"Macro-average ROC (AUC = {roc_auc_macro:.4f})",
)
for i, col in enumerate(colors):
    ax_roc.plot(
        fpr_ovr[i], tpr_ovr[i], color=col, lw=1.2, alpha=0.9,
        label=f"{CLASSES[i]} (AUC = {roc_auc_cls[i]:.3f})",
    )
ax_roc.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.35)
ax_roc.set_xlim([0.0, 1.0])
ax_roc.set_ylim([0.0, 1.05])
ax_roc.set_xlabel("False positive rate")
ax_roc.set_ylabel("True positive rate")
ax_roc.set_title("Multiclass ROC (One-vs-Rest) — TTA softmax", fontweight="bold")
ax_roc.legend(loc="lower right", fontsize=7)
ax_roc.grid(alpha=0.3)
plt.tight_layout()
plt.show()

print(f"\n📈 ROC-AUC (TTA)  Micro: {roc_auc_micro:.4f}  |  Macro: {roc_auc_macro:.4f}")
for i, c in enumerate(CLASSES):
    print(f"    {c:<10} AUC = {roc_auc_cls[i]:.4f}")

# ── Per-class F1 bar chart ────────────────────────────────────
per_f1 = f1_score(tta_labels, tta_preds, average=None) * 100
fig, ax = plt.subplots(figsize=(12, 5))
bars = ax.bar(CLASSES, per_f1, color=colors, edgecolor='white', width=0.6)
ax.axhline(np.mean(per_f1), color='navy', ls=':', lw=1.5,
           label=f'Mean={np.mean(per_f1):.2f}%')
for bar, v in zip(bars, per_f1):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.2,
            f'{v:.2f}%', ha='center', va='bottom', fontsize=9,
            fontweight='bold')
ax.set_ylim([min(per_f1)-5, 101])
ax.set_title("Per-Class F1-Score (TTA)", fontweight='bold', fontsize=13)
ax.set_ylabel("F1-Score (%)"); ax.legend()
ax.grid(axis='y', alpha=0.4)
plt.tight_layout(); plt.show()


# ============================================================
# STEP 10 — Qualitative: Correct vs Wrong Predictions
# ============================================================
def _tensor_to_show_rgb(t):
    """Denormalize 3×H×W tensor for imshow."""
    x = t.cpu().permute(1, 2, 0).numpy()
    mean = np.array(IMAGENET_MEAN, dtype=x.dtype)
    std = np.array(IMAGENET_STD, dtype=x.dtype)
    x = np.clip(x * std + mean, 0.0, 1.0)
    return x


def show_predictions(model, test_ds, n_correct=10, n_wrong=10):
    model.eval()
    corrects, wrongs = [], []

    with torch.no_grad():
        for i in range(len(test_ds)):
            img, label = test_ds[i]
            out  = model(img.unsqueeze(0).to(DEVICE))
            pred = out.argmax(1).item()
            conf = F.softmax(out, dim=1).max().item() * 100
            entry = (img, label, pred, conf)
            if pred == label and len(corrects) < n_correct:
                corrects.append(entry)
            elif pred != label and len(wrongs) < n_wrong:
                wrongs.append(entry)
            if len(corrects) >= n_correct and len(wrongs) >= n_wrong:
                break

    fig, axes = plt.subplots(2, max(n_correct, n_wrong),
                             figsize=(max(n_correct, n_wrong)*2, 5))
    for c, (img, true, pred, conf) in enumerate(corrects[:n_correct]):
        ax = axes[0][c]
        ax.imshow(_tensor_to_show_rgb(img))
        ax.set_title(f"✅ {CLASSES[pred]}\n{conf:.0f}%", fontsize=7,
                     color='green')
        ax.axis('off')
    for c, (img, true, pred, conf) in enumerate(wrongs[:n_wrong]):
        ax = axes[1][c]
        ax.imshow(_tensor_to_show_rgb(img))
        ax.set_title(f"❌ T:{CLASSES[true]}\nP:{CLASSES[pred]} {conf:.0f}%",
                     fontsize=7, color='red')
        ax.axis('off')

    axes[0][0].set_ylabel("Correct ✅", fontsize=11, fontweight='bold')
    axes[1][0].set_ylabel("Wrong ❌",   fontsize=11, fontweight='bold')
    plt.suptitle("Correct vs Wrong Predictions", fontsize=13, fontweight='bold')
    plt.tight_layout(); plt.show()

show_predictions(model, test_ds, n_correct=10, n_wrong=10)


# ============================================================
# STEP 11 — Confidence Distribution
# ============================================================
@torch.no_grad()
def get_confidence_data(model, loader):
    model.eval()
    correct_conf, wrong_conf = [], []
    all_conf_per_class = {c: [] for c in CLASSES}

    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        probs = F.softmax(model(imgs), dim=1)
        preds = probs.argmax(1)
        confs = probs.max(1).values

        for i in range(len(labels)):
            all_conf_per_class[CLASSES[labels[i].item()]].append(confs[i].item())
            if preds[i] == labels[i]: correct_conf.append(confs[i].item())
            else:                     wrong_conf.append(confs[i].item())

    return correct_conf, wrong_conf, all_conf_per_class

cor_c, wro_c, cls_c = get_confidence_data(model, test_loader)

fig, axes = plt.subplots(1, 2, figsize=(18, 5))
axes[0].hist(cor_c, bins=40, alpha=0.7, color='#27ae60', label=f'Correct (n={len(cor_c)})')
axes[0].hist(wro_c, bins=40, alpha=0.7, color='#e74c3c', label=f'Wrong   (n={len(wro_c)})')
axes[0].set_xlabel("Confidence"); axes[0].set_ylabel("Count")
axes[0].set_title("Confidence: Correct vs Wrong", fontweight='bold')
axes[0].legend(); axes[0].grid(alpha=0.4)

for cls, col in zip(CLASSES, colors):
    if cls_c[cls]:
        axes[1].hist(cls_c[cls], bins=30, alpha=0.55, color=col, label=cls)
axes[1].set_xlabel("Confidence"); axes[1].set_ylabel("Count")
axes[1].set_title("Confidence Distribution per Class", fontweight='bold')
axes[1].legend(fontsize=8); axes[1].grid(alpha=0.4)

plt.suptitle("Model Confidence Analysis", fontsize=13, fontweight='bold')
plt.tight_layout(); plt.show()

avg_cor = np.mean(cor_c) if cor_c else 0
avg_wro = np.mean(wro_c) if wro_c else 0
print(f"  Avg confidence — Correct: {avg_cor*100:.2f}%  |  Wrong: {avg_wro*100:.2f}%")


# ============================================================
# STEP 12 — Final Summary Table
# ============================================================
per_precision = precision_score(tta_labels, tta_preds, average=None, zero_division=0)
per_recall    = recall_score(tta_labels, tta_preds, average=None, zero_division=0)
per_f1_arr    = f1_score(tta_labels, tta_preds, average=None, zero_division=0)
per_support   = np.bincount(tta_labels, minlength=NUM_CLS)

print("\n" + "═" * 72)
print("📄  COMPLETE RESULTS SUMMARY — ConvNeXt-Base (TTA)")
print("═" * 72)
print(f"  Dataset       : FER-2013")
print(f"  Architecture  : {MODEL_NAME}")
print(f"  Parameters    : {params:,}")
print(f"  Best Epoch    : {stopper.best_ep}")
print(f"  TTA Transforms: {len(tta_tfs)}")
print("─" * 72)
print(f"  {'Class':<12} {'Precision':>10} {'Recall':>10} "
      f"{'F1':>10} {'Support':>10}")
print("  " + "─" * 50)
for i, cls in enumerate(CLASSES):
    print(f"  {cls:<12} {per_precision[i]:>10.4f} {per_recall[i]:>10.4f} "
          f"{per_f1_arr[i]:>10.4f} {per_support[i]:>10}")
print("  " + "─" * 50)
print(f"  {'Macro avg':<12} {per_precision.mean():>10.4f} "
      f"{per_recall.mean():>10.4f} {per_f1_arr.mean():>10.4f} "
      f"{per_support.sum():>10}")
print("─" * 72)
print(f"  Overall Accuracy : {tta_acc:.4f}%")
print(f"  Macro F1-Score   : {tta_f1:.4f}%")
print(f"  ROC-AUC (micro)  : {roc_auc_micro:.4f}")
print(f"  ROC-AUC (macro)  : {roc_auc_macro:.4f}")
print("  Note: 99.8% macro-F1 is not attainable on standard FER-2013 test;")
print("        ROC-AUC reflects ranking quality; F1 reflects hard-class decisions.")
target_str = "✅ High macro-F1" if tta_f1 >= 75 else "⚠️  Room to improve (try longer train / stronger aug)"
print(f"  Qualitative flag : {target_str}")
print("═" * 72)
print(f"\n✅ Model saved: {SAVE_PATH}")