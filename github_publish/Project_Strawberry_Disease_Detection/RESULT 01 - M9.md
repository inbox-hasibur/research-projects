# =============================================================================
# 🍓 STRAWBERRY 9 MODEL COMPARISON
# Correct Methodology: Train 9 Models → Pick Winner → 5-Fold CV Final Training
# Dataset: Afzaal et al. 2021 + PlantVillage | XAI: Grad-CAM++ + EigenCAM
# =============================================================================

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import time, json, warnings
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             roc_curve, classification_report)
from sklearn.preprocessing import label_binarize
import timm

warnings.filterwarnings('ignore')

def save_fig(fname, title=""):
    plt.savefig(f"{fname}.png", bbox_inches='tight', dpi=150)
    plt.show()
    print(f"  ✅ {title or fname} saved")

# ─────────────────────────────────────────────────────────────────────────────
# HARDWARE
# ─────────────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("\n" + "─" * 80)
print("HARDWARE CONFIGURATION")
print("─" * 80)
print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU   : {torch.cuda.get_device_name(0)}")
    print(f"Compute Capability: {torch.cuda.get_device_capability()}")
    try:
        _ = torch.randn(2, 2, device=DEVICE) @ torch.randn(2, 2, device=DEVICE)
        print("CUDA Status: ✓ Operational")
    except Exception as e:
        print(f"CUDA Status: ✗ Failed ({e}) — falling back to CPU")
        DEVICE = torch.device("cpu")
else:
    print("GPU: Not Available | Mode: CPU")
print("─" * 80 + "\n")

# ─────────────────────────────────────────────────────────────────────────────
# HYPERPARAMETERS & CONFIG
# ─────────────────────────────────────────────────────────────────────────────
IMG_SIZE     = 224
BATCH_SIZE   = 16
EPOCHS       = 50
ABL_PATIENCE = 5          # ← ablation study patience = 5
CV_PATIENCE  = 6
LR           = 5e-5
WEIGHT_DECAY = 1e-4
LABEL_SMOOTH = 0.1
NUM_CLASSES  = 8
CROP_PROB    = 0.6
N_FOLDS      = 5

AFZAAL_ROOT       = "/kaggle/input/datasets/usmanafzaal/strawberry-disease-detection-dataset"
PLANTVILLAGE_ROOT = "/kaggle/input/datasets/abdallahalidev/plantvillage-dataset"
TRAIN_DIR = os.path.join(AFZAAL_ROOT, "train")
VAL_DIR   = os.path.join(AFZAAL_ROOT, "val")
TEST_DIR  = os.path.join(AFZAAL_ROOT, "test")

LABEL_MAP = {
    "angular_leafspot": 0, "anthracnose": 1, "blossom_blight": 2,
    "gray_mold": 3,        "leaf_spot":   4, "powdery_mildew": 5,
    "leaf_scorch": 6,      "healthy":     7,
}
IDX_TO_CLASS = {v: k for k, v in LABEL_MAP.items()}

# ─────────────────────────────────────────────────────────────────────────────
# ANNOTATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def load_annotation_bbox(json_path):
    if not os.path.exists(json_path): return None
    try:
        with open(json_path, 'r') as f: data = json.load(f)
        all_x, all_y = [], []
        for val in data.values():
            regions = val.get('regions', [])
            if isinstance(regions, dict): regions = list(regions.values())
            for region in regions:
                shape = region.get('shape_attributes', {})
                stype = shape.get('name', '')
                if stype == 'polygon':
                    all_x.extend(shape.get('all_points_x', []))
                    all_y.extend(shape.get('all_points_y', []))
                elif stype == 'rect':
                    x, y, w, h = (shape.get('x', 0), shape.get('y', 0),
                                  shape.get('width', 0), shape.get('height', 0))
                    all_x += [x, x+w]; all_y += [y, y+h]
                elif stype == 'ellipse':
                    cx, cy, rx, ry = (shape.get('cx', 0), shape.get('cy', 0),
                                      shape.get('rx', 0), shape.get('ry', 0))
                    all_x += [cx-rx, cx+rx]; all_y += [cy-ry, cy+ry]
        return (min(all_x), min(all_y), max(all_x), max(all_y)) if all_x else None
    except Exception:
        return None

def annotation_crop(img, bbox, padding=0.20):
    if bbox is None: return img
    w, h = img.size
    x1, y1, x2, y2 = bbox
    bw, bh = x2-x1, y2-y1
    if bw <= 0 or bh <= 0: return img
    px, py = int(bw*padding), int(bh*padding)
    x1, y1 = max(0, x1-px), max(0, y1-py)
    x2, y2 = min(w, x2+px), min(h, y2+py)
    return img.crop((x1, y1, x2, y2)) if (x2-x1) >= 10 and (y2-y1) >= 10 else img

# ─────────────────────────────────────────────────────────────────────────────
# BUILDING BLOCKS
# ─────────────────────────────────────────────────────────────────────────────
class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.ch_avg = nn.AdaptiveAvgPool2d(1)
        self.ch_max = nn.AdaptiveMaxPool2d(1)
        self.ch_fc  = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels, bias=False),
        )
        self.sp_conv = nn.Conv2d(2, 1, kernel_size=kernel_size,
                                 padding=kernel_size // 2, bias=False)

    def forward(self, x):
        avg = self.ch_fc(self.ch_avg(x))
        mx  = self.ch_fc(self.ch_max(x))
        x   = x * torch.sigmoid(avg + mx).unsqueeze(-1).unsqueeze(-1)
        sp  = torch.cat([x.mean(dim=1, keepdim=True),
                         x.max(dim=1, keepdim=True)[0]], dim=1)
        return x * torch.sigmoid(self.sp_conv(sp))

class ECA(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        import math
        t = int(abs((math.log2(channels) + b) / gamma))
        k = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv     = nn.Conv1d(1, 1, kernel_size=k,
                                  padding=(k-1)//2, bias=False)

    def forward(self, x):
        y = self.avg_pool(x)
        y = (self.conv(y.squeeze(-1).transpose(-1, -2))
             .transpose(-1, -2).unsqueeze(-1))
        return x * torch.sigmoid(y)

class GeMPooling(nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p   = nn.Parameter(torch.tensor(p))
        self.eps = eps

    def forward(self, x):
        return (F.adaptive_avg_pool2d(x.clamp(min=self.eps).pow(self.p), 1)
                .pow(1.0 / self.p).flatten(1))

def _swin_pool(feat):
    if feat.dim() == 4:   feat = feat.mean(dim=[1, 2])
    elif feat.dim() == 3: feat = feat.mean(dim=1)
    return feat.contiguous()

class SmoothCE(nn.Module):
    def __init__(self, num_classes=8, smoothing=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing   = smoothing

    def forward(self, logits, targets):
        log_probs = F.log_softmax(logits, dim=-1)
        if targets.dim() == 1:
            st = torch.full_like(log_probs,
                                 self.smoothing / (self.num_classes - 1))
            st.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        else:
            st = (targets * (1.0 - self.smoothing)
                  + self.smoothing / self.num_classes)
        return -(st * log_probs).sum(dim=-1).mean()

class EarlyStopping:
    def __init__(self, patience=8, min_delta=1e-4, save_path="best.pth"):
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

# ─────────────────────────────────────────────────────────────────────────────
# TRANSFORMS
# ─────────────────────────────────────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE + 48, IMG_SIZE + 48)),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.4, contrast=0.4,
                           saturation=0.3, hue=0.08),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),
])

val_test_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# ─────────────────────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────────────────────
class StrawberryDataset(Dataset):
    def __init__(self, paths, labels, transform=None,
                 use_annotation_crop=False):
        self.paths               = paths
        self.labels              = labels
        self.transform           = transform
        self.use_annotation_crop = use_annotation_crop

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        img   = Image.open(self.paths[idx]).convert("RGB")
        label = self.labels[idx]
        if self.use_annotation_crop and np.random.random() < CROP_PROB:
            json_path = os.path.splitext(self.paths[idx])[0] + '.json'
            bbox = load_annotation_bbox(json_path)
            if bbox: img = annotation_crop(img, bbox)
        if self.transform: img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.long)

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
def label_from_filename(fname, label_map):
    base = os.path.splitext(os.path.basename(fname))[0].lower()
    best_cls, best_idx = None, None
    for cls_name, cls_idx in label_map.items():
        if base.startswith(cls_name):
            if best_cls is None or len(cls_name) > len(best_cls):
                best_cls, best_idx = cls_name, cls_idx
    return best_idx, best_cls

def scan_afzaal_split(split_dir, split_name="split"):
    paths, labels = [], []
    if not os.path.exists(split_dir):
        print(f"  ⚠️  Not found: {split_dir}"); return paths, labels
    for fn in sorted(os.listdir(split_dir)):
        if not fn.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')): continue
        cls_idx, _ = label_from_filename(fn, LABEL_MAP)
        if cls_idx is None: continue
        paths.append(os.path.join(split_dir, fn)); labels.append(cls_idx)
    cc    = {}
    for l in labels: cc[l] = cc.get(l, 0) + 1
    found = [IDX_TO_CLASS[k] for k in sorted(cc)]
    print(f"  [Afzaal-{split_name}] {len(paths)} imgs | {found}")
    return paths, labels

def scan_plantvillage_strawberry():
    paths, labels = [], []
    PV_COLOR_ROOT = os.path.join(PLANTVILLAGE_ROOT, "color")
    for folder, cls_idx in [
        (os.path.join(PV_COLOR_ROOT, "Strawberry___Leaf_scorch"),
         LABEL_MAP["leaf_scorch"]),
        (os.path.join(PV_COLOR_ROOT, "Strawberry___healthy"),
         LABEL_MAP["healthy"]),
    ]:
        if not os.path.exists(folder):
            print(f"  ⚠️  Not found: {folder}"); continue
        imgs = [f for f in os.listdir(folder)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        for fn in imgs:
            paths.append(os.path.join(folder, fn)); labels.append(cls_idx)
        print(f"  [PlantVillage] {IDX_TO_CLASS[cls_idx]}: {len(imgs)} imgs")
    return paths, labels

print("📂 Pooling all datasets...")
afl_paths, afl_labels = [], []
for sdir in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
    p, l = scan_afzaal_split(sdir, os.path.basename(sdir))
    afl_paths.extend(p); afl_labels.extend(l)
pv_paths, pv_labels = scan_plantvillage_strawberry()

all_paths  = afl_paths + pv_paths
all_labels = afl_labels + pv_labels
print(f"Total pooled: {len(all_paths)}\n")

# 10 % held-out test (never touched during Phase 1 or 2)
X_tv, test_paths, y_tv, test_labels = train_test_split(
    all_paths, all_labels, test_size=0.10, random_state=42, stratify=all_labels)

# Phase-1 comparison split: ~80 / 10 train / val from remaining 90 %
train_paths, val_paths, train_labels, val_labels = train_test_split(
    X_tv, y_tv, test_size=0.1111, random_state=42, stratify=y_tv)

print("─" * 80)
print("DATASET SUMMARY")
print("─" * 80)
print(f"Classes : {NUM_CLASSES} ({', '.join(LABEL_MAP.keys())})")
print(f"Total   : {len(train_paths)+len(val_paths)+len(test_paths)}")
print(f"  ├─ Train : {len(train_paths)}")
print(f"  ├─ Val   : {len(val_paths)}")
print(f"  └─ Test  : {len(test_paths)}  (held-out)")
print(f"  Test classes: {[IDX_TO_CLASS[i] for i in sorted(set(test_labels))]}")
print("─" * 80 + "\n")

n_ann    = sum(1 for p in train_paths
               if os.path.exists(os.path.splitext(p)[0] + '.json'))
USE_CROP = n_ann > 0
print(f"Annotations: {n_ann}/{len(train_paths)} "
      f"({'Enabled' if USE_CROP else 'Disabled'})\n")

# ── Weighted sampler ──────────────────────────────────────────────────────────
def make_sampler(labels):
    counts  = np.bincount(labels, minlength=NUM_CLASSES)
    wpc     = 1.0 / np.where(counts == 0, 1, counts)
    weights = torch.tensor([wpc[l] for l in labels], dtype=torch.double)
    return WeightedRandomSampler(weights, len(weights), replacement=True)

train_loader = DataLoader(
    StrawberryDataset(train_paths, train_labels, train_transforms, USE_CROP),
    batch_size=BATCH_SIZE, sampler=make_sampler(train_labels),
    num_workers=2, pin_memory=True)
val_loader = DataLoader(
    StrawberryDataset(val_paths, val_labels, val_test_transforms),
    batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
test_loader = DataLoader(
    StrawberryDataset(test_paths, test_labels, val_test_transforms),
    batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

criterion = SmoothCE(num_classes=NUM_CLASSES, smoothing=LABEL_SMOOTH)

# ─────────────────────────────────────────────────────────────────────────────
# TRAIN / EVALUATE
# ─────────────────────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, crit, optimizer):
    model.train(); total_loss = 0.0
    for imgs, lbls in loader:
        imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        loss = crit(model(imgs), lbls)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

@torch.no_grad()
def evaluate(model, loader, crit):
    model.eval(); total_loss = 0.0
    all_probs, all_preds, all_labels_out = [], [], []
    for imgs, lbls in loader:
        imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
        logits = model(imgs)
        total_loss += crit(logits, lbls).item()
        all_probs.extend(F.softmax(logits, dim=-1).cpu().numpy())
        all_preds.extend(logits.argmax(dim=-1).cpu().numpy())
        all_labels_out.extend(lbls.cpu().numpy())
    acc = accuracy_score(all_labels_out, all_preds)
    return total_loss / len(loader), acc, all_probs, all_preds, all_labels_out

def _compute_metrics(probs, preds, labels):
    """Return (acc, mac_f1, auc) from raw lists."""
    acc    = accuracy_score(labels, preds)
    mac_f1 = f1_score(labels, preds, average='macro', zero_division=0)
    try:
        auc = roc_auc_score(labels, np.array(probs),
                            multi_class='ovr', average='macro')
    except Exception:
        auc = float('nan')
    return acc, mac_f1, auc

# ─────────────────────────────────────────────────────────────────────────────
# CORE TRAINING LOOP
# patience      → early-stopping patience
# report_loader → loader used for the ✅ final summary line
#                 (val_loader for phase-1 internal; test_loader for ablation)
# ─────────────────────────────────────────────────────────────────────────────
def train_model(model, name, save_path, patience,
                tr_loader=None, vl_loader=None,
                report_loader=None,          # ← NEW: loader for final metrics
                epochs=EPOCHS,
                differential_lr=False):
    """
    Train with early-stopping on vl_loader loss.
    Final ✅ metrics are computed on report_loader
    (defaults to vl_loader if not given).
    Returns (acc, mac_f1, auc, best_epoch).
    """
    tr_loader     = tr_loader     or train_loader
    vl_loader     = vl_loader     or val_loader
    report_loader = report_loader or vl_loader   # default = val

    if differential_lr and hasattr(model, 'eff_backbone'):
        bb_params   = (list(model.eff_backbone.parameters()) +
                       list(model.swin_backbone.parameters()))
        head_params = [p for p in model.parameters()
                       if not any(p is b for b in bb_params)]
        optimizer = optim.AdamW([
            {'params': bb_params,   'lr': LR * 0.1},
            {'params': head_params, 'lr': LR},
        ], weight_decay=WEIGHT_DECAY)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=LR,
                                weight_decay=WEIGHT_DECAY)

    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2)
    stopper   = EarlyStopping(patience=patience, save_path=save_path)

    print(f"\n🔷  {name}")
    print("─" * 60)
    for ep in range(1, epochs + 1):
        t0      = time.time()
        tr_loss = train_one_epoch(model, tr_loader, criterion, optimizer)
        vl_loss, vl_acc, _, _, _ = evaluate(model, vl_loader, criterion)
        scheduler.step()
        stop  = stopper.step(vl_loss, ep, model)
        medal = ("🏅" if stopper.counter == 0
                 else f"({stopper.counter}/{patience})")
        print(f"  Ep{ep:02d} | {int(time.time()-t0):3d}s | "
              f"Tr:{tr_loss:.4f} Vl:{vl_loss:.4f} "
              f"Acc:{vl_acc*100:.2f}% {medal}")
        if stop:
            break

    # Load best weights → evaluate on report_loader
    model.load_state_dict(torch.load(save_path, map_location=DEVICE))
    _, acc_r, probs_r, preds_r, true_r = evaluate(model, report_loader, criterion)
    acc_r, mac_f1, auc = _compute_metrics(probs_r, preds_r, true_r)

    lbl = ("Test" if report_loader is test_loader else "Val")
    print(f"\n  ✅ {name} [{lbl}] | Acc:{acc_r*100:.3f}%  "
          f"MacF1:{mac_f1*100:.3f}%  AUC:{auc:.4f}")
    return acc_r, mac_f1, auc, stopper.best_epoch

# ─────────────────────────────────────────────────────────────────────────────
# MODEL DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────
class VGG19CBAM(nn.Module):
    def __init__(self, nc, drop=0.3):
        super().__init__()
        self.bb   = timm.create_model('vgg19_bn', pretrained=True,
                                       num_classes=0, global_pool='')
        self.cbam = CBAM(512); self.gem = GeMPooling()
        self.head = nn.Sequential(nn.Dropout(drop), nn.Linear(512, nc))
    def forward(self, x):
        return self.head(self.gem(self.cbam(self.bb.forward_features(x))))

class ResNet50CBAMGeM(nn.Module):
    def __init__(self, nc, drop=0.3):
        super().__init__()
        self.bb   = timm.create_model('resnet50', pretrained=True,
                                       num_classes=0, global_pool='')
        self.cbam = CBAM(2048); self.gem = GeMPooling()
        self.head = nn.Sequential(nn.Dropout(drop), nn.Linear(2048, nc))
    def forward(self, x):
        return self.head(self.gem(self.cbam(self.bb.forward_features(x))))

class DenseNetCBAM(nn.Module):
    def __init__(self, nc):
        super().__init__()
        self.bb   = timm.create_model('densenet121', pretrained=True,
                                       num_classes=0, global_pool='')
        self.cbam = CBAM(1024); self.gem = GeMPooling()
        self.head = nn.Linear(1024, nc)
    def forward(self, x):
        return self.head(self.gem(self.cbam(self.bb.forward_features(x))))

class SwinGeM(nn.Module):
    def __init__(self, nc):
        super().__init__()
        self.bb   = timm.create_model('swin_tiny_patch4_window7_224',
                                       pretrained=True, num_classes=0)
        self.head = nn.Linear(768, nc)
    def forward(self, x):
        return self.head(_swin_pool(self.bb.forward_features(x)))

class EfficientNetV2GeM(nn.Module):
    def __init__(self, nc):
        super().__init__()
        self.bb   = timm.create_model('tf_efficientnetv2_s', pretrained=True,
                                       num_classes=0, global_pool='')
        self.eca  = ECA(1280); self.gem = GeMPooling()
        self.head = nn.Linear(1280, nc)
    def forward(self, x):
        return self.head(self.gem(self.eca(self.bb.forward_features(x))))

class MobileViTGeM(nn.Module):
    def __init__(self, nc):
        super().__init__()
        self.bb   = timm.create_model('mobilevit_s', pretrained=True,
                                       num_classes=0, global_pool='')
        self.gem  = GeMPooling(); self.head = nn.Linear(640, nc)
    def forward(self, x):
        f = self.bb.forward_features(x)
        if f.dim() == 3:
            B, L, C = f.shape; H = W = int(L ** 0.5)
            f = f.view(B, C, H, W)
        return self.head(self.gem(f))

class ConvNeXtCBAM(nn.Module):
    def __init__(self, nc):
        super().__init__()
        self.bb   = timm.create_model('convnext_tiny', pretrained=True,
                                       num_classes=0, global_pool='')
        self.cbam = CBAM(768); self.gem = GeMPooling()
        self.head = nn.Linear(768, nc)
    def forward(self, x):
        f = self.bb.forward_features(x)
        if f.dim() == 4 and f.shape[-1] == 768:
            f = f.permute(0, 3, 1, 2).contiguous()
        return self.head(self.gem(self.cbam(f)))

class EffSwinConcat(nn.Module):
    def __init__(self, nc):
        super().__init__()
        self.eff  = timm.create_model('tf_efficientnetv2_s', pretrained=True,
                                       num_classes=0, global_pool='avg')
        self.swin = timm.create_model('swin_tiny_patch4_window7_224',
                                       pretrained=True, num_classes=0)
        self.head = nn.Linear(1280 + 768, nc)
    def forward(self, x):
        fe = self.eff(x)
        fs = _swin_pool(self.swin.forward_features(x))
        return self.head(torch.cat([fe, fs], dim=1))

class EffSwinHybrid(nn.Module):
    def __init__(self, num_classes=8, drop=0.4):
        super().__init__()
        self.eff_backbone = timm.create_model('tf_efficientnetv2_s',
                                               pretrained=True,
                                               num_classes=0, global_pool='')
        self.eca = ECA(1280); self.gem = GeMPooling(p=3.0)
        self.eff_proj = nn.Sequential(
            nn.Linear(1280, 512), nn.LayerNorm(512),
            nn.GELU(), nn.Dropout(drop * 0.5))
        self.swin_backbone = timm.create_model(
            'swin_tiny_patch4_window7_224', pretrained=True, num_classes=0)
        self.swin_norm = nn.LayerNorm(768)
        self.swin_proj = nn.Sequential(
            nn.Linear(768, 512), nn.LayerNorm(512),
            nn.GELU(), nn.Dropout(drop * 0.5))
        self.fusion_head = nn.Sequential(
            nn.Linear(1024, 512), nn.LayerNorm(512),
            nn.GELU(), nn.Dropout(drop),
            nn.Linear(512, 128), nn.GELU(),
            nn.Dropout(drop / 2), nn.Linear(128, num_classes))

    def forward_eff(self, x):
        return self.eff_proj(
            self.gem(self.eca(self.eff_backbone.forward_features(x))))

    def forward_swin(self, x):
        return self.swin_proj(
            self.swin_norm(_swin_pool(
                self.swin_backbone.forward_features(x))))

    def forward(self, x):
        return self.fusion_head(
            torch.cat([self.forward_eff(x), self.forward_swin(x)], dim=1))

COMPARISON_CONFIGS = [
    ("VGG19+CBAM",             lambda nc: VGG19CBAM(nc)),
    ("ResNet50+CBAM+GeM",      lambda nc: ResNet50CBAMGeM(nc)),
    ("DenseNet121+CBAM",       lambda nc: DenseNetCBAM(nc)),
    ("Swin-T+GeM",             lambda nc: SwinGeM(nc)),
    ("EfficientNetV2+ECA+GeM", lambda nc: EfficientNetV2GeM(nc)),
    ("MobileViT-S+GeM",        lambda nc: MobileViTGeM(nc)),
    ("ConvNeXt-T+CBAM",        lambda nc: ConvNeXtCBAM(nc)),
    ("EffSwin-Concat",          lambda nc: EffSwinConcat(nc)),
    ("EffSwin-Hybrid (Ours)",   lambda nc: EffSwinHybrid(nc)),
]

# ─────────────────────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════
# PHASE 1 — 9-MODEL ABLATION / COMPARISON
#   patience = ABL_PATIENCE (5)
#   early-stopping monitored on val_loader (loss)
#   final metrics reported on TEST set  ← key change
# ═══════════════════════════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "═" * 80)
print("🔬 PHASE 1 — 9-MODEL ABLATION STUDY")
print(f"   Patience : {ABL_PATIENCE} epochs")
print("   Early-stop: monitored on val loss")
print("   Metrics   : reported on HELD-OUT TEST SET")
print("═" * 80)

comparison_results = {}   # name → (test_acc, test_mac_f1, auc, best_epoch)

for name, factory in COMPARISON_CONFIGS:
    model    = factory(NUM_CLASSES).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  📐 {name} | Params: {n_params:,}")
    safe_name = (name.replace(' ', '_')
                     .replace('(', '').replace(')', '')
                     .replace('+', '_'))
    save_path   = f"best_cmp_{safe_name}.pth"
    use_diff_lr = "EffSwin" in name

    # ← report_loader = test_loader  (final line shows Test metrics)
    acc, f1, auc_v, best_ep = train_model(
        model, name, save_path,
        patience=ABL_PATIENCE,
        report_loader=test_loader,
        differential_lr=use_diff_lr)

    comparison_results[name] = (acc * 100, f1 * 100, auc_v, best_ep)
    del model
    torch.cuda.empty_cache()

# ── Leaderboard (ranked by Test MacF1) ───────────────────────────────────────
print("\n\n" + "═" * 80)
print("📊  PHASE 1 RESULTS — ABLATION LEADERBOARD  (Test Set)")
print("═" * 80)
print(f"  {'Rank':<5} {'Model':<30} {'Test Acc':>10} "
      f"{'Test MacF1':>12} {'AUC':>7} {'BestEp':>7}")
print("─" * 76)
sorted_results = sorted(comparison_results.items(),
                        key=lambda x: x[1][1], reverse=True)
for rank, (name, (acc, f1, auc_v, best_ep)) in enumerate(sorted_results, 1):
    star = " ◄ WINNER" if rank == 1 else ""
    print(f"  {rank:<5} {name:<30} {acc:>9.3f}% "
          f"{f1:>11.3f}% {auc_v:>7.4f} {best_ep:>7}{star}")
print("═" * 80)

best_model_name = sorted_results[0][0]
print(f"\n🏆 WINNER: {best_model_name}")
print(f"   → Phase 2: 5-Fold Cross-Validation on full trainval set\n")

# Phase-1 bar chart (Test MacF1)
names_sorted = [r[0] for r in sorted_results]
f1s_sorted   = [r[1][1] for r in sorted_results]
colors_bar   = ['#f39c12' if n == best_model_name else '#3498db'
                for n in names_sorted]

plt.figure(figsize=(11, 5))
bars = plt.barh(names_sorted, f1s_sorted,
                color=colors_bar, edgecolor='white', linewidth=1.5)
plt.xlabel("Test Macro F1 (%)", fontweight='bold')
plt.title("Phase 1 Ablation Study — Test Macro F1 (patience=5)",
          fontweight='bold')
plt.xlim([max(0, min(f1s_sorted) - 5), 101])
for bar, val in zip(bars, f1s_sorted):
    plt.text(val + 0.3, bar.get_y() + bar.get_height() / 2,
             f"{val:.2f}%", va='center', fontsize=8)
plt.tight_layout()
save_fig(f"fig_phase1_ablation_{int(time.time())}",
         "Phase 1 Ablation Chart")

# ─────────────────────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════
# PHASE 2 — 5-FOLD CV ON WINNER
#   patience = CV_PATIENCE (6)
#   OOF evaluated on each fold's validation split (standard CV)
# ═══════════════════════════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "═" * 80)
print(f"🎯 PHASE 2 — FINAL 5-FOLD CROSS-VALIDATION")
print(f"   Model   : {best_model_name}")
print(f"   Patience: {CV_PATIENCE}")
print(f"   Data    : Full trainval ({len(X_tv)}) + "
      f"held-out test ({len(test_paths)})")
print("═" * 80)

winner_factory_map = {n: f for n, f in COMPARISON_CONFIGS}
winner_factory     = winner_factory_map[best_model_name]

skf      = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
X_tv_arr = np.array(X_tv)
y_tv_arr = np.array(y_tv)

fold_results = []
oof_probs    = np.zeros((len(X_tv), NUM_CLASSES))
oof_preds    = np.zeros(len(X_tv), dtype=int)

for fold, (tr_idx, vl_idx) in enumerate(skf.split(X_tv_arr, y_tv_arr), 1):
    print(f"\n{'─'*60}")
    print(f"  FOLD {fold}/{N_FOLDS}")
    print(f"{'─'*60}")

    f_tr_paths  = X_tv_arr[tr_idx].tolist()
    f_tr_labels = y_tv_arr[tr_idx].tolist()
    f_vl_paths  = X_tv_arr[vl_idx].tolist()
    f_vl_labels = y_tv_arr[vl_idx].tolist()

    fold_tr_loader = DataLoader(
        StrawberryDataset(f_tr_paths, f_tr_labels,
                          train_transforms, USE_CROP),
        batch_size=BATCH_SIZE, sampler=make_sampler(f_tr_labels),
        num_workers=2, pin_memory=True)
    fold_vl_loader = DataLoader(
        StrawberryDataset(f_vl_paths, f_vl_labels, val_test_transforms),
        batch_size=BATCH_SIZE, shuffle=False,
        num_workers=2, pin_memory=True)

    fold_model    = winner_factory(NUM_CLASSES).to(DEVICE)
    fold_savepath = f"best_final_fold{fold}.pth"
    use_diff_lr   = "EffSwin" in best_model_name

    # Phase-2 uses val_loader as report_loader (OOF is fold-val)
    _, _, _, best_ep = train_model(
        fold_model,
        f"{best_model_name} — Fold {fold}",
        fold_savepath,
        patience=CV_PATIENCE,
        tr_loader=fold_tr_loader,
        vl_loader=fold_vl_loader,
        report_loader=fold_vl_loader,   # OOF = fold validation
        differential_lr=use_diff_lr)

    fold_model.load_state_dict(torch.load(fold_savepath, map_location=DEVICE))
    _, acc_f, probs_f, preds_f, true_f = evaluate(
        fold_model, fold_vl_loader, criterion)
    acc_f, mac_f1_f, auc_f = _compute_metrics(probs_f, preds_f, true_f)

    oof_probs[vl_idx] = np.array(probs_f)
    oof_preds[vl_idx] = np.array(preds_f)
    fold_results.append({'fold': fold, 'acc': acc_f * 100,
                         'mac_f1': mac_f1_f * 100,
                         'auc': auc_f, 'best_epoch': best_ep})
    print(f"\n  📌 Fold {fold} [OOF-Val] | "
          f"Acc:{acc_f*100:.3f}%  MacF1:{mac_f1_f*100:.3f}%  AUC:{auc_f:.4f}")
    del fold_model
    torch.cuda.empty_cache()

# ── 5-Fold Summary ────────────────────────────────────────────────────────────
print("\n\n" + "═" * 80)
print("📊  PHASE 2 — 5-FOLD CROSS-VALIDATION SUMMARY")
print("═" * 80)
print(f"  {'Fold':<8} {'Accuracy':>10} {'Macro F1':>10} "
      f"{'AUC':>8} {'BestEp':>8}")
print("─" * 52)
accs   = [r['acc']    for r in fold_results]
f1s_cv = [r['mac_f1'] for r in fold_results]
aucs   = [r['auc']    for r in fold_results]
for r in fold_results:
    print(f"  Fold {r['fold']:<4} {r['acc']:>9.3f}%  "
          f"{r['mac_f1']:>9.3f}%  {r['auc']:>8.4f}  {r['best_epoch']:>8}")
print("─" * 52)
print(f"  {'Mean':<8} {np.mean(accs):>9.3f}%  "
      f"{np.mean(f1s_cv):>9.3f}%  {np.mean(aucs):>8.4f}")
print(f"  {'±Std':<8} {np.std(accs):>9.3f}%  "
      f"{np.std(f1s_cv):>9.3f}%  {np.std(aucs):>8.4f}")
print("═" * 80)

oof_acc    = accuracy_score(y_tv_arr, oof_preds)
oof_mac_f1 = f1_score(y_tv_arr, oof_preds, average='macro', zero_division=0)
try:
    oof_auc = roc_auc_score(y_tv_arr, oof_probs,
                            multi_class='ovr', average='macro')
except Exception:
    oof_auc = float('nan')
print(f"\n  OOF Metrics (combined):")
print(f"  Accuracy : {oof_acc*100:.4f}%")
print(f"  Macro F1 : {oof_mac_f1*100:.4f}%")
print(f"  AUC-ROC  : {oof_auc:.4f}\n")

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 3 — FINAL HELD-OUT TEST EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
best_fold_idx  = int(np.argmax(f1s_cv)) + 1
best_fold_path = f"best_final_fold{best_fold_idx}.pth"
print(f"🏆 Best fold: Fold {best_fold_idx} — "
      f"loading for final test evaluation...\n")

final_model = winner_factory(NUM_CLASSES).to(DEVICE)
final_model.load_state_dict(torch.load(best_fold_path, map_location=DEVICE))

_, acc_test, probs_test, preds_test, true_test = evaluate(
    final_model, test_loader, criterion)
probs_test_np = np.array(probs_test)
mac_prec_t = precision_score(true_test, preds_test,
                             average='macro', zero_division=0)
mac_rec_t  = recall_score(true_test, preds_test,
                          average='macro', zero_division=0)
mac_f1_t   = f1_score(true_test, preds_test,
                      average='macro', zero_division=0)
wt_f1_t    = f1_score(true_test, preds_test,
                      average='weighted', zero_division=0)
try:
    auc_test = roc_auc_score(true_test, probs_test_np,
                             multi_class='ovr', average='macro')
except Exception:
    auc_test = float('nan')

print("\n" + "═" * 80)
print("🎯 FINAL HELD-OUT TEST RESULTS")
print("═" * 80)
print(f"  {'Metric':<22} {'Value':>15}")
print("─" * 45)
print(f"  {'Accuracy':<22} {acc_test*100:>14.4f}%")
print(f"  {'Macro Precision':<22} {mac_prec_t*100:>14.4f}%")
print(f"  {'Macro Recall':<22} {mac_rec_t*100:>14.4f}%")
print(f"  {'Macro F1':<22} {mac_f1_t*100:>14.4f}%")
print(f"  {'Weighted F1':<22} {wt_f1_t*100:>14.4f}%")
print(f"  {'Macro AUC-ROC':<22} {auc_test:>15.4f}")
print("─" * 45)
print(f"\nDETAILED CLASSIFICATION REPORT")
print(classification_report(
    true_test, preds_test,
    target_names=[IDX_TO_CLASS[i] for i in range(NUM_CLASSES)], digits=4))
print("═" * 80)

# ─────────────────────────────────────────────────────────────────────────────
# VISUALIZATIONS
# ─────────────────────────────────────────────────────────────────────────────
cls_short = [IDX_TO_CLASS[i].replace('_', '\n')[:12]
             for i in range(NUM_CLASSES)]
cls_label = [IDX_TO_CLASS[i] for i in range(NUM_CLASSES)]

# Confusion Matrix
plt.figure(figsize=(8, 7))
cm = confusion_matrix(true_test, preds_test,
                      labels=list(range(NUM_CLASSES)))
sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd',
            xticklabels=cls_short, yticklabels=cls_short,
            annot_kws={"size": 9, "weight": "bold"})
plt.title(f"Confusion Matrix — {best_model_name}", fontweight='bold')
plt.xticks(rotation=45); plt.tight_layout()
save_fig(f"fig_confusion_matrix_{int(time.time())}", "Confusion Matrix")

# Per-class F1
plt.figure(figsize=(9, 5))
per_f1     = f1_score(true_test, preds_test, average=None,
                      zero_division=0, labels=list(range(NUM_CLASSES)))
bar_colors = ['#e74c3c' if f < 0.97 else '#2ecc71' for f in per_f1]
bars       = plt.bar(cls_short, per_f1 * 100,
                     color=bar_colors, edgecolor='white')
plt.axhline(99, color='navy', ls='--', alpha=0.6, label='99% target')
plt.ylim([max(0, per_f1.min() * 100 - 5), 101])
plt.title("Per-Class F1 (%) — Test Set", fontweight='bold')
plt.legend(); plt.grid(axis='y', alpha=0.4)
for bar, val in zip(bars, per_f1):
    plt.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 0.2,
             f"{val*100:.1f}", ha='center', va='bottom', fontsize=8)
plt.tight_layout()
save_fig(f"fig_per_class_f1_{int(time.time())}", "Per-Class F1")

# ROC Curves
plt.figure(figsize=(8, 7))
y_bin  = label_binarize(true_test, classes=list(range(NUM_CLASSES)))
colors = plt.cm.tab10(np.linspace(0, 1, NUM_CLASSES))
for i in range(NUM_CLASSES):
    if y_bin[:, i].sum() == 0: continue
    fpr, tpr, _ = roc_curve(y_bin[:, i], probs_test_np[:, i])
    ai = roc_auc_score(y_bin[:, i], probs_test_np[:, i])
    plt.plot(fpr, tpr, color=colors[i], lw=1.5,
             label=f"{cls_label[i][:14]} ({ai:.3f})")
plt.plot([0, 1], [0, 1], 'navy', lw=1.2, ls='--')
plt.title("ROC Curves (One-vs-Rest) — Test Set", fontweight='bold')
plt.legend(fontsize=7, loc='lower right')
plt.grid(alpha=0.35); plt.tight_layout()
save_fig(f"fig_roc_curves_{int(time.time())}", "ROC Curves")

# 5-Fold accuracy bars
plt.figure(figsize=(7, 4))
fold_labels = [f"Fold {r['fold']}" for r in fold_results]
fold_accs   = [r['acc'] for r in fold_results]
fold_cols   = ['#f39c12' if r['fold'] == best_fold_idx else '#3498db'
               for r in fold_results]
plt.bar(fold_labels, fold_accs, color=fold_cols,
        edgecolor='white', linewidth=1.5)
plt.axhline(np.mean(fold_accs), color='red', ls='--', lw=1.5,
            label=f"Mean={np.mean(fold_accs):.2f}%")
plt.ylim([max(0, min(fold_accs) - 2), 101])
plt.title(f"5-Fold OOF Accuracy — {best_model_name}", fontweight='bold')
plt.ylabel("Accuracy (%)"); plt.legend(); plt.grid(axis='y', alpha=0.4)
for i, v in enumerate(fold_accs):
    plt.text(i, v + 0.2, f"{v:.2f}%", ha='center',
             fontsize=9, fontweight='bold')
plt.tight_layout()
save_fig(f"fig_5fold_accuracy_{int(time.time())}", "5-Fold Accuracy")

# ─────────────────────────────────────────────────────────────────────────────
# ROBUSTNESS TEST
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "═" * 80)
print("🛡️  ROBUSTNESS TEST — Gaussian Noise")
print("═" * 80)

class NoisyDataset(StrawberryDataset):
    def __init__(self, paths, labels, transform, sigma):
        super().__init__(paths, labels, transform, False)
        self.sigma = sigma
    def __getitem__(self, idx):
        img, lbl = super().__getitem__(idx)
        if self.sigma > 0:
            img = img + torch.randn_like(img) * self.sigma
        return img, lbl

rob_results = []
for sigma in [0.0, 0.05, 0.10, 0.20]:
    nl = DataLoader(
        NoisyDataset(test_paths, test_labels, val_test_transforms, sigma),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    _, acc_n, _, preds_n, true_n = evaluate(final_model, nl, criterion)
    f1_n = f1_score(true_n, preds_n, average='macro', zero_division=0)
    rob_results.append({'sigma': sigma, 'acc': acc_n * 100, 'f1': f1_n * 100})
    print(f"  σ={sigma:.2f} → Acc: {acc_n*100:.2f}%  "
          f"Macro F1: {f1_n*100:.2f}%")

sigs = [r['sigma'] for r in rob_results]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.plot(sigs, [r['acc'] for r in rob_results],
         'darkorange', marker='o', ms=8, lw=2.5)
ax1.set_title("Accuracy vs Noise", fontweight='bold')
ax1.set_xlabel("σ"); ax1.set_ylabel("Accuracy (%)")
ax1.set_ylim([50, 101]); ax1.grid(alpha=0.4)
ax2.plot(sigs, [r['f1'] for r in rob_results],
         '#e74c3c', marker='s', ms=8, lw=2.5)
ax2.set_title("Macro F1 vs Noise", fontweight='bold')
ax2.set_xlabel("σ"); ax2.set_ylabel("Macro F1 (%)")
ax2.set_ylim([50, 101]); ax2.grid(alpha=0.4)
plt.suptitle("Robustness Under Gaussian Noise", fontweight='bold')
plt.tight_layout()
save_fig(f"fig_robustness_{int(time.time())}", "Robustness Analysis")

# ─────────────────────────────────────────────────────────────────────────────
# XAI — GRAD-CAM++ + EIGENCAM
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "═" * 80)
print("📸  XAI — GRAD-CAM++ & EIGEN-CAM")
print("═" * 80)

try:
    try:
        from pytorch_grad_cam import GradCAMPlusPlus, EigenCAM
    except ImportError:
        print("📦 Installing grad-cam...")
        os.system("pip install grad-cam -q")
        from pytorch_grad_cam import GradCAMPlusPlus, EigenCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image

    def get_target_layer(model):
        if isinstance(model, EffSwinHybrid):
            return model.eff_backbone.blocks[-1]
        if isinstance(model, VGG19CBAM):
            return model.bb.features[-1]
        if isinstance(model, ResNet50CBAMGeM):
            return model.bb.layer4[-1]
        if isinstance(model, DenseNetCBAM):
            return model.bb.features.norm5
        if isinstance(model, SwinGeM):
            return model.bb.layers[-1].blocks[-1]
        if isinstance(model, EfficientNetV2GeM):
            return model.bb.blocks[-1]
        if isinstance(model, MobileViTGeM):
            return model.bb.stages[-1]
        if isinstance(model, ConvNeXtCBAM):
            return model.bb.stages[-1].blocks[-1]
        if isinstance(model, EffSwinConcat):
            return model.eff.blocks[-1]
        for m in reversed(list(model.modules())):
            if isinstance(m, (nn.Conv2d, nn.BatchNorm2d)):
                return m
        return None

    target_layer = get_target_layer(final_model)

    if target_layer is None:
        print("  ⚠️  No suitable CAM layer found — XAI skipped.")
    else:
        if isinstance(final_model, EffSwinHybrid):
            class SwinWrapper(nn.Module):
                def __init__(self, m): super().__init__(); self.m = m
                def forward(self, x):
                    feat = self.m.swin_backbone.forward_features(x)
                    if feat.dim() == 3:
                        B, L, C = feat.shape
                        H = W = int(L ** 0.5)
                        feat = feat.view(B, H, W, C).permute(0, 3, 1, 2)
                    elif feat.dim() == 4 and feat.shape[-1] == 768:
                        feat = feat.permute(0, 3, 1, 2)
                    return feat.contiguous()
            swin_wrapper = SwinWrapper(final_model)
            swin_layer   = final_model.swin_backbone.layers[-1].blocks[-1]
            cam_a  = GradCAMPlusPlus(model=final_model,
                                     target_layers=[target_layer])
            cam_b  = EigenCAM(model=swin_wrapper,
                              target_layers=[swin_layer])
            dual   = True
        else:
            cam_a = GradCAMPlusPlus(model=final_model,
                                    target_layers=[target_layer])
            cam_b = None
            dual  = False

        sample_indices = []
        for ci in range(NUM_CLASSES):
            cands = [i for i, l in enumerate(test_labels) if l == ci]
            if cands:
                sample_indices.append(int(np.random.choice(cands)))

        n_show     = min(6, len(sample_indices))
        n_cols     = 5 if dual else 4
        col_titles = (["Original", "EfficientNet\n(Grad-CAM++)",
                       "Swin\n(EigenCAM)", "Hybrid\n(Overlay)", "Contour"]
                      if dual else
                      ["Original", "Grad-CAM++", "Overlay", "Contour"])

        for row, idx in enumerate(sample_indices[:n_show]):
            fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 4.5))
            fig.patch.set_facecolor('#1a1a2e')
            raw = Image.open(test_paths[idx]).convert("RGB").resize(
                (IMG_SIZE, IMG_SIZE))
            rgb = np.array(raw) / 255.0
            inp = val_test_transforms(raw).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                logits = final_model(inp)
                pred   = logits.argmax(dim=-1).item()
                conf   = F.softmax(logits, dim=-1)[0, pred].item()

            try:
                gc_a = cam_a(input_tensor=inp)[0]
                gc_b = cam_b(input_tensor=inp)[0] if dual else None
            except Exception:
                gc_a = np.zeros((IMG_SIZE, IMG_SIZE))
                gc_b = np.zeros((IMG_SIZE, IMG_SIZE)) if dual else None

            true_cls = IDX_TO_CLASS[test_labels[idx]]
            pred_cls = IDX_TO_CLASS[pred]
            correct  = pred == test_labels[idx]
            clr      = '#2ecc71' if correct else '#e74c3c'

            axes[0].imshow(raw)
            axes[0].set_ylabel(
                f"GT:{true_cls}\nPred:{pred_cls}"
                f"{'✓' if correct else '✗'}({conf*100:.1f}%)",
                fontsize=8, color=clr, fontweight='bold')
            axes[0].axis('off')

            if dual:
                im1 = axes[1].imshow(gc_a, cmap='jet', vmin=0, vmax=1,
                                     interpolation='bilinear')
                plt.colorbar(im1, ax=axes[1], fraction=0.046)
                axes[1].axis('off')
                im2 = axes[2].imshow(gc_b, cmap='jet', vmin=0, vmax=1,
                                     interpolation='bilinear')
                plt.colorbar(im2, ax=axes[2], fraction=0.046)
                axes[2].axis('off')
                axes[3].imshow(show_cam_on_image(
                    rgb.astype(np.float32), gc_a, use_rgb=True))
                axes[3].axis('off')
                axes[4].imshow(raw)
                axes[4].contour(gc_a, levels=[0.50],
                                colors=['yellow'], linewidths=[1.5], alpha=0.85)
                axes[4].contour(gc_a, levels=[0.75],
                                colors=['red'], linewidths=[2.5], alpha=0.95)
                axes[4].axis('off')
            else:
                im1 = axes[1].imshow(gc_a, cmap='jet', vmin=0, vmax=1,
                                     interpolation='bilinear')
                plt.colorbar(im1, ax=axes[1], fraction=0.046)
                axes[1].axis('off')
                axes[2].imshow(show_cam_on_image(
                    rgb.astype(np.float32), gc_a, use_rgb=True))
                axes[2].axis('off')
                axes[3].imshow(raw)
                axes[3].contour(gc_a, levels=[0.50],
                                colors=['yellow'], linewidths=[1.5], alpha=0.85)
                axes[3].contour(gc_a, levels=[0.75],
                                colors=['red'], linewidths=[2.5], alpha=0.95)
                axes[3].axis('off')

            for ax, t in zip(axes, col_titles):
                ax.set_title(t, fontsize=10, fontweight='bold', color='white')
                ax.set_facecolor('#1a1a2e')
            plt.suptitle(
                f"XAI ({best_model_name}) | GT: {true_cls} | Sample {row+1}",
                fontsize=12, fontweight='bold', color='white', y=1.01)
            fig.tight_layout()
            save_fig(f"fig_xai_sample_{row}", f"XAI Sample {row+1}")

        print("✅  XAI complete!")

except Exception as e:
    import traceback
    print(f"⚠️  XAI Error: {e}"); traceback.print_exc()

# ─────────────────────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "🎉 " * 20)
print("\n" + "═" * 80)
print("📁 FINAL SUMMARY")
print("═" * 80)
print(f"  Best Architecture  : {best_model_name}")
print(f"  Phase-1 patience   : {ABL_PATIENCE}  (metrics on TEST set)")
print(f"  Phase-2 patience   : {CV_PATIENCE}   (5-fold OOF on val splits)")
print(f"  5-Fold OOF Acc     : {oof_acc*100:.4f}%")
print(f"  5-Fold OOF MacF1   : {oof_mac_f1*100:.4f}%")
print(f"  Held-out Test Acc  : {acc_test*100:.4f}%")
print(f"  Held-out MacF1     : {mac_f1_t*100:.4f}%")
print(f"  Held-out AUC-ROC   : {auc_test:.4f}")
print(f"  Noise Resilience   : "
      f"{rob_results[0]['acc']-rob_results[-1]['acc']:.2f}% drop at σ=0.20")
print("═" * 80)
────────────────────────────────────────────────────────────────────────────────
HARDWARE CONFIGURATION
────────────────────────────────────────────────────────────────────────────────
Device: cuda
GPU   : Tesla T4
Compute Capability: (7, 5)
CUDA Status: ✓ Operational
────────────────────────────────────────────────────────────────────────────────

📂 Pooling all datasets...
  [Afzaal-train] 1450 imgs | ['angular_leafspot', 'anthracnose', 'blossom_blight', 'gray_mold', 'leaf_spot', 'powdery_mildew']
  [Afzaal-val] 307 imgs | ['angular_leafspot', 'anthracnose', 'blossom_blight', 'gray_mold', 'leaf_spot', 'powdery_mildew']
  [Afzaal-test] 743 imgs | ['angular_leafspot', 'anthracnose', 'blossom_blight', 'gray_mold', 'leaf_spot', 'powdery_mildew']
  [PlantVillage] leaf_scorch: 1109 imgs
  [PlantVillage] healthy: 456 imgs
Total pooled: 4065

────────────────────────────────────────────────────────────────────────────────
DATASET SUMMARY
────────────────────────────────────────────────────────────────────────────────
Classes : 8 (angular_leafspot, anthracnose, blossom_blight, gray_mold, leaf_spot, powdery_mildew, leaf_scorch, healthy)
Total   : 4065
  ├─ Train : 3251
  ├─ Val   : 407
  └─ Test  : 407  (held-out)
  Test classes: ['angular_leafspot', 'anthracnose', 'blossom_blight', 'gray_mold', 'leaf_spot', 'powdery_mildew', 'leaf_scorch', 'healthy']
────────────────────────────────────────────────────────────────────────────────

Annotations: 1999/3251 (Enabled)


════════════════════════════════════════════════════════════════════════════════
🔬 PHASE 1 — 9-MODEL ABLATION STUDY
   Patience : 5 epochs
   Early-stop: monitored on val loss
   Metrics   : reported on HELD-OUT TEST SET
════════════════════════════════════════════════════════════════════════════════
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
  📐 VGG19+CBAM | Params: 139,618,219

🔷  VGG19+CBAM
────────────────────────────────────────────────────────────
  Ep01 |  68s | Tr:1.5347 Vl:0.7995 Acc:93.61% 🏅
  Ep02 |  70s | Tr:0.7985 Vl:0.6345 Acc:97.54% 🏅
  Ep03 |  70s | Tr:0.6753 Vl:0.5830 Acc:98.28% 🏅
  Ep04 |  70s | Tr:0.6282 Vl:0.5720 Acc:99.26% 🏅
  Ep05 |  70s | Tr:0.6103 Vl:0.5582 Acc:99.26% 🏅
  Ep06 |  70s | Tr:0.5960 Vl:0.5570 Acc:99.75% 🏅
  Ep07 |  70s | Tr:0.5800 Vl:0.5504 Acc:99.51% 🏅
  Ep08 |  70s | Tr:0.5711 Vl:0.5475 Acc:99.75% 🏅
  Ep09 |  70s | Tr:0.5636 Vl:0.5452 Acc:99.51% 🏅
  Ep10 |  69s | Tr:0.5744 Vl:0.5454 Acc:99.51% (1/5)
  Ep11 |  69s | Tr:0.5777 Vl:0.5518 Acc:99.26% (2/5)
  Ep12 |  69s | Tr:0.5768 Vl:0.5520 Acc:99.02% (3/5)
  Ep13 |  70s | Tr:0.5794 Vl:0.5447 Acc:99.51% 🏅
  Ep14 |  70s | Tr:0.5604 Vl:0.5444 Acc:99.26% 🏅
  Ep15 |  70s | Tr:0.5574 Vl:0.5392 Acc:99.26% 🏅
  Ep16 |  69s | Tr:0.5542 Vl:0.5427 Acc:99.51% (1/5)
  Ep17 |  70s | Tr:0.5526 Vl:0.5375 Acc:99.51% 🏅
  Ep18 |  69s | Tr:0.5486 Vl:0.5433 Acc:99.75% (1/5)
  Ep19 |  70s | Tr:0.5431 Vl:0.5365 Acc:99.75% 🏅
  Ep20 |  70s | Tr:0.5418 Vl:0.5360 Acc:99.51% 🏅
  Ep21 |  69s | Tr:0.5418 Vl:0.5362 Acc:99.51% (1/5)
  Ep22 |  70s | Tr:0.5389 Vl:0.5359 Acc:99.51% 🏅
  Ep23 |  70s | Tr:0.5345 Vl:0.5308 Acc:99.75% 🏅
  Ep24 |  70s | Tr:0.5351 Vl:0.5295 Acc:99.75% 🏅
  Ep25 |  69s | Tr:0.5376 Vl:0.5323 Acc:99.51% (1/5)
  Ep26 |  69s | Tr:0.5314 Vl:0.5317 Acc:99.75% (2/5)
  Ep27 |  70s | Tr:0.5341 Vl:0.5292 Acc:100.00% 🏅
  Ep28 |  69s | Tr:0.5319 Vl:0.5291 Acc:100.00% (1/5)
  Ep29 |  70s | Tr:0.5329 Vl:0.5281 Acc:99.75% 🏅
  Ep30 |  70s | Tr:0.5317 Vl:0.5277 Acc:100.00% 🏅
  Ep31 |  69s | Tr:0.5446 Vl:0.5376 Acc:99.51% (1/5)
  Ep32 |  69s | Tr:0.5529 Vl:0.5662 Acc:98.77% (2/5)
  Ep33 |  69s | Tr:0.5495 Vl:0.5389 Acc:99.26% (3/5)
  Ep34 |  69s | Tr:0.5486 Vl:0.5438 Acc:99.26% (4/5)
  Ep35 |  69s | Tr:0.5428 Vl:0.5360 Acc:99.26% (5/5)

  ✅ VGG19+CBAM [Test] | Acc:99.017%  MacF1:98.595%  AUC:0.9987
  📐 ResNet50+CBAM+GeM | Params: 24,048,811

🔷  ResNet50+CBAM+GeM
────────────────────────────────────────────────────────────
  Ep01 |  44s | Tr:1.9399 Vl:1.8861 Acc:46.93% 🏅
  Ep02 |  44s | Tr:1.4017 Vl:1.1838 Acc:86.49% 🏅
  Ep03 |  43s | Tr:0.9508 Vl:0.8225 Acc:92.14% 🏅
  Ep04 |  43s | Tr:0.8242 Vl:0.7362 Acc:93.61% 🏅
  Ep05 |  43s | Tr:0.7789 Vl:0.6825 Acc:95.09% 🏅
  Ep06 |  43s | Tr:0.7470 Vl:0.6643 Acc:95.58% 🏅
  Ep07 |  43s | Tr:0.7443 Vl:0.6487 Acc:95.58% 🏅
  Ep08 |  43s | Tr:0.7025 Vl:0.6572 Acc:95.58% (1/5)
  Ep09 |  43s | Tr:0.7210 Vl:0.6531 Acc:96.31% (2/5)
  Ep10 |  43s | Tr:0.7086 Vl:0.6621 Acc:95.09% (3/5)
  Ep11 |  43s | Tr:0.7113 Vl:0.6235 Acc:97.30% 🏅
  Ep12 |  43s | Tr:0.6841 Vl:0.6257 Acc:96.81% (1/5)
  Ep13 |  43s | Tr:0.6645 Vl:0.6029 Acc:98.03% 🏅
  Ep14 |  43s | Tr:0.6492 Vl:0.5980 Acc:97.79% 🏅
  Ep15 |  43s | Tr:0.6432 Vl:0.5915 Acc:98.28% 🏅
  Ep16 |  43s | Tr:0.6260 Vl:0.5882 Acc:98.53% 🏅
  Ep17 |  43s | Tr:0.6222 Vl:0.5874 Acc:98.53% 🏅
  Ep18 |  43s | Tr:0.6142 Vl:0.5824 Acc:98.28% 🏅
  Ep19 |  43s | Tr:0.6095 Vl:0.5748 Acc:98.53% 🏅
  Ep20 |  43s | Tr:0.6041 Vl:0.5750 Acc:98.77% (1/5)
  Ep21 |  43s | Tr:0.5978 Vl:0.5727 Acc:99.26% 🏅
  Ep22 |  43s | Tr:0.5997 Vl:0.5718 Acc:99.02% 🏅
  Ep23 |  43s | Tr:0.5969 Vl:0.5693 Acc:99.26% 🏅
  Ep24 |  43s | Tr:0.5945 Vl:0.5670 Acc:99.26% 🏅
  Ep25 |  43s | Tr:0.5851 Vl:0.5688 Acc:99.26% (1/5)
  Ep26 |  43s | Tr:0.5934 Vl:0.5738 Acc:98.77% (2/5)
  Ep27 |  43s | Tr:0.5841 Vl:0.5654 Acc:99.26% 🏅
  Ep28 |  43s | Tr:0.5924 Vl:0.5683 Acc:98.77% (1/5)
  Ep29 |  43s | Tr:0.5895 Vl:0.5667 Acc:99.26% (2/5)
  Ep30 |  43s | Tr:0.5873 Vl:0.5657 Acc:99.26% (3/5)
  Ep31 |  43s | Tr:0.5866 Vl:0.5683 Acc:99.02% (4/5)
  Ep32 |  43s | Tr:0.5867 Vl:0.5629 Acc:99.51% 🏅
  Ep33 |  43s | Tr:0.5807 Vl:0.5598 Acc:99.26% 🏅
  Ep34 |  43s | Tr:0.5772 Vl:0.5606 Acc:99.51% (1/5)
  Ep35 |  43s | Tr:0.5728 Vl:0.5614 Acc:99.51% (2/5)
  Ep36 |  43s | Tr:0.5683 Vl:0.5613 Acc:99.26% (3/5)
  Ep37 |  43s | Tr:0.5677 Vl:0.5617 Acc:99.02% (4/5)
  Ep38 |  43s | Tr:0.5659 Vl:0.5562 Acc:99.51% 🏅
  Ep39 |  43s | Tr:0.5693 Vl:0.5520 Acc:99.51% 🏅
  Ep40 |  43s | Tr:0.5641 Vl:0.5484 Acc:99.51% 🏅
  Ep41 |  43s | Tr:0.5626 Vl:0.5478 Acc:99.26% 🏅
  Ep42 |  43s | Tr:0.5565 Vl:0.5464 Acc:99.51% 🏅
  Ep43 |  43s | Tr:0.5522 Vl:0.5511 Acc:99.51% (1/5)
  Ep44 |  43s | Tr:0.5566 Vl:0.5458 Acc:99.51% 🏅
  Ep45 |  43s | Tr:0.5590 Vl:0.5471 Acc:99.26% (1/5)
  Ep46 |  43s | Tr:0.5542 Vl:0.5463 Acc:99.51% (2/5)
  Ep47 |  43s | Tr:0.5550 Vl:0.5414 Acc:99.51% 🏅
  Ep48 |  43s | Tr:0.5535 Vl:0.5420 Acc:99.51% (1/5)
  Ep49 |  43s | Tr:0.5516 Vl:0.5458 Acc:99.51% (2/5)
  Ep50 |  43s | Tr:0.5497 Vl:0.5426 Acc:99.51% (3/5)

  ✅ ResNet50+CBAM+GeM [Test] | Acc:99.263%  MacF1:99.331%  AUC:0.9980
  📐 DenseNet121+CBAM | Params: 7,093,227

🔷  DenseNet121+CBAM
────────────────────────────────────────────────────────────
  Ep01 |  53s | Tr:1.3812 Vl:0.8667 Acc:87.96% 🏅
  Ep02 |  52s | Tr:0.8026 Vl:0.6845 Acc:95.82% 🏅
  Ep03 |  52s | Tr:0.7154 Vl:0.6332 Acc:97.05% 🏅
  Ep04 |  52s | Tr:0.6634 Vl:0.6109 Acc:98.03% 🏅
  Ep05 |  52s | Tr:0.6363 Vl:0.5956 Acc:98.28% 🏅
  Ep06 |  52s | Tr:0.6250 Vl:0.5955 Acc:98.77% 🏅
  Ep07 |  53s | Tr:0.6127 Vl:0.5869 Acc:98.53% 🏅
  Ep08 |  52s | Tr:0.6026 Vl:0.5903 Acc:98.53% (1/5)
  Ep09 |  52s | Tr:0.6128 Vl:0.5880 Acc:98.28% (2/5)
  Ep10 |  52s | Tr:0.6069 Vl:0.5880 Acc:98.77% (3/5)
  Ep11 |  52s | Tr:0.6037 Vl:0.5958 Acc:98.77% (4/5)
  Ep12 |  52s | Tr:0.5890 Vl:0.5755 Acc:98.77% 🏅
  Ep13 |  52s | Tr:0.5847 Vl:0.5689 Acc:98.77% 🏅
  Ep14 |  53s | Tr:0.5769 Vl:0.5647 Acc:99.02% 🏅
  Ep15 |  52s | Tr:0.5734 Vl:0.5719 Acc:99.02% (1/5)
  Ep16 |  52s | Tr:0.5676 Vl:0.5630 Acc:99.02% 🏅
  Ep17 |  53s | Tr:0.5611 Vl:0.5578 Acc:99.26% 🏅
  Ep18 |  52s | Tr:0.5621 Vl:0.5546 Acc:99.26% 🏅
  Ep19 |  53s | Tr:0.5560 Vl:0.5592 Acc:99.26% (1/5)
  Ep20 |  53s | Tr:0.5539 Vl:0.5596 Acc:98.77% (2/5)
  Ep21 |  53s | Tr:0.5500 Vl:0.5512 Acc:99.26% 🏅
  Ep22 |  52s | Tr:0.5477 Vl:0.5548 Acc:99.02% (1/5)
  Ep23 |  52s | Tr:0.5441 Vl:0.5518 Acc:99.51% (2/5)
  Ep24 |  52s | Tr:0.5424 Vl:0.5545 Acc:99.02% (3/5)
  Ep25 |  52s | Tr:0.5400 Vl:0.5503 Acc:99.51% 🏅
  Ep26 |  52s | Tr:0.5399 Vl:0.5517 Acc:99.51% (1/5)
  Ep27 |  53s | Tr:0.5441 Vl:0.5506 Acc:99.26% (2/5)
  Ep28 |  52s | Tr:0.5433 Vl:0.5500 Acc:99.26% 🏅
  Ep29 |  53s | Tr:0.5407 Vl:0.5471 Acc:99.51% 🏅
  Ep30 |  52s | Tr:0.5423 Vl:0.5561 Acc:99.02% (1/5)
  Ep31 |  52s | Tr:0.5452 Vl:0.5541 Acc:99.26% (2/5)
  Ep32 |  53s | Tr:0.5416 Vl:0.5515 Acc:99.51% (3/5)
  Ep33 |  53s | Tr:0.5493 Vl:0.5618 Acc:99.26% (4/5)
  Ep34 |  52s | Tr:0.5386 Vl:0.5494 Acc:99.26% (5/5)

  ✅ DenseNet121+CBAM [Test] | Acc:98.771%  MacF1:98.358%  AUC:0.9991
  📐 Swin-T+GeM | Params: 27,525,506

🔷  Swin-T+GeM
────────────────────────────────────────────────────────────
  Ep01 |  52s | Tr:0.7570 Vl:0.5630 Acc:98.28% 🏅
  Ep02 |  52s | Tr:0.5747 Vl:0.5314 Acc:99.51% 🏅
  Ep03 |  52s | Tr:0.5472 Vl:0.5347 Acc:99.26% (1/5)
  Ep04 |  52s | Tr:0.5367 Vl:0.5326 Acc:99.51% (2/5)
  Ep05 |  52s | Tr:0.5325 Vl:0.5300 Acc:99.26% 🏅
  Ep06 |  52s | Tr:0.5316 Vl:0.5337 Acc:99.51% (1/5)
  Ep07 |  52s | Tr:0.5284 Vl:0.5269 Acc:99.75% 🏅
  Ep08 |  52s | Tr:0.5245 Vl:0.5305 Acc:99.51% (1/5)
  Ep09 |  52s | Tr:0.5243 Vl:0.5292 Acc:99.51% (2/5)
  Ep10 |  52s | Tr:0.5241 Vl:0.5297 Acc:99.51% (3/5)
  Ep11 |  52s | Tr:0.5380 Vl:0.5439 Acc:99.51% (4/5)
  Ep12 |  52s | Tr:0.5405 Vl:0.5243 Acc:100.00% 🏅
  Ep13 |  52s | Tr:0.5385 Vl:0.5209 Acc:100.00% 🏅
  Ep14 |  52s | Tr:0.5284 Vl:0.5204 Acc:100.00% 🏅
  Ep15 |  52s | Tr:0.5373 Vl:0.5298 Acc:99.75% (1/5)
  Ep16 |  52s | Tr:0.5317 Vl:0.5225 Acc:99.75% (2/5)
  Ep17 |  52s | Tr:0.5255 Vl:0.5308 Acc:99.51% (3/5)
  Ep18 |  52s | Tr:0.5279 Vl:0.5413 Acc:99.26% (4/5)
  Ep19 |  52s | Tr:0.5285 Vl:0.5293 Acc:99.51% (5/5)

  ✅ Swin-T+GeM [Test] | Acc:99.509%  MacF1:99.553%  AUC:0.9980
  📐 EfficientNetV2+ECA+GeM | Params: 20,187,742

🔷  EfficientNetV2+ECA+GeM
────────────────────────────────────────────────────────────
  Ep01 |  57s | Tr:1.0172 Vl:0.6390 Acc:97.54% 🏅
  Ep02 |  56s | Tr:0.6773 Vl:0.5890 Acc:99.02% 🏅
  Ep03 |  57s | Tr:0.6133 Vl:0.5636 Acc:99.75% 🏅
  Ep04 |  57s | Tr:0.5863 Vl:0.5506 Acc:99.51% 🏅
  Ep05 |  57s | Tr:0.5777 Vl:0.5447 Acc:99.75% 🏅
  Ep06 |  57s | Tr:0.5691 Vl:0.5419 Acc:99.75% 🏅
  Ep07 |  57s | Tr:0.5589 Vl:0.5425 Acc:99.75% (1/5)
  Ep08 |  57s | Tr:0.5569 Vl:0.5400 Acc:99.75% 🏅
  Ep09 |  57s | Tr:0.5543 Vl:0.5389 Acc:99.51% 🏅
  Ep10 |  57s | Tr:0.5500 Vl:0.5394 Acc:99.75% (1/5)
  Ep11 |  56s | Tr:0.5563 Vl:0.5385 Acc:99.75% 🏅
  Ep12 |  57s | Tr:0.5503 Vl:0.5445 Acc:99.02% (1/5)
  Ep13 |  57s | Tr:0.5451 Vl:0.5340 Acc:99.51% 🏅
  Ep14 |  57s | Tr:0.5405 Vl:0.5352 Acc:99.51% (1/5)
  Ep15 |  57s | Tr:0.5410 Vl:0.5276 Acc:100.00% 🏅
  Ep16 |  57s | Tr:0.5386 Vl:0.5271 Acc:100.00% 🏅
  Ep17 |  57s | Tr:0.5350 Vl:0.5307 Acc:99.51% (1/5)
  Ep18 |  57s | Tr:0.5345 Vl:0.5298 Acc:99.75% (2/5)
  Ep19 |  57s | Tr:0.5325 Vl:0.5338 Acc:99.51% (3/5)
  Ep20 |  57s | Tr:0.5357 Vl:0.5360 Acc:99.26% (4/5)
  Ep21 |  57s | Tr:0.5303 Vl:0.5307 Acc:99.51% (5/5)

  ✅ EfficientNetV2+ECA+GeM [Test] | Acc:99.509%  MacF1:99.550%  AUC:0.9993
  📐 MobileViT-S+GeM | Params: 4,942,761

🔷  MobileViT-S+GeM
────────────────────────────────────────────────────────────
  Ep01 |  39s | Tr:1.6027 Vl:1.0018 Acc:89.19% 🏅
  Ep02 |  39s | Tr:0.8610 Vl:0.6196 Acc:96.81% 🏅
  Ep03 |  39s | Tr:0.6826 Vl:0.5805 Acc:98.53% 🏅
  Ep04 |  39s | Tr:0.6455 Vl:0.5532 Acc:99.26% 🏅
  Ep05 |  38s | Tr:0.6072 Vl:0.5456 Acc:99.75% 🏅
  Ep06 |  39s | Tr:0.5965 Vl:0.5434 Acc:99.51% 🏅
  Ep07 |  38s | Tr:0.5938 Vl:0.5379 Acc:99.75% 🏅
  Ep08 |  39s | Tr:0.5947 Vl:0.5371 Acc:100.00% 🏅
  Ep09 |  38s | Tr:0.5919 Vl:0.5397 Acc:99.75% (1/5)
  Ep10 |  39s | Tr:0.5858 Vl:0.5394 Acc:99.75% (2/5)
  Ep11 |  38s | Tr:0.5827 Vl:0.5424 Acc:99.26% (3/5)
  Ep12 |  38s | Tr:0.5758 Vl:0.5302 Acc:100.00% 🏅
  Ep13 |  38s | Tr:0.5678 Vl:0.5317 Acc:100.00% (1/5)
  Ep14 |  39s | Tr:0.5645 Vl:0.5330 Acc:99.75% (2/5)
  Ep15 |  38s | Tr:0.5605 Vl:0.5356 Acc:99.75% (3/5)
  Ep16 |  39s | Tr:0.5610 Vl:0.5338 Acc:99.75% (4/5)
  Ep17 |  38s | Tr:0.5570 Vl:0.5314 Acc:99.75% (5/5)

  ✅ MobileViT-S+GeM [Test] | Acc:98.771%  MacF1:97.914%  AUC:0.9993
  📐 ConvNeXt-T+CBAM | Params: 27,900,107

🔷  ConvNeXt-T+CBAM
────────────────────────────────────────────────────────────
  Ep01 | 100s | Tr:0.8336 Vl:0.5637 Acc:98.53% 🏅
  Ep02 |  90s | Tr:0.5732 Vl:0.5305 Acc:99.51% 🏅
  Ep03 |  89s | Tr:0.5512 Vl:0.5328 Acc:99.26% (1/5)
  Ep04 |  90s | Tr:0.5308 Vl:0.5229 Acc:100.00% 🏅
  Ep05 |  89s | Tr:0.5261 Vl:0.5282 Acc:99.75% (1/5)
  Ep06 |  89s | Tr:0.5268 Vl:0.5213 Acc:100.00% 🏅
  Ep07 |  89s | Tr:0.5246 Vl:0.5243 Acc:99.75% (1/5)
  Ep08 |  89s | Tr:0.5228 Vl:0.5245 Acc:99.51% (2/5)
  Ep09 |  89s | Tr:0.5224 Vl:0.5216 Acc:100.00% (3/5)
  Ep10 |  89s | Tr:0.5210 Vl:0.5219 Acc:99.75% (4/5)
  Ep11 |  89s | Tr:0.5385 Vl:0.5315 Acc:99.75% (5/5)

  ✅ ConvNeXt-T+CBAM [Test] | Acc:99.263%  MacF1:98.774%  AUC:0.9997

  📐 EffSwin-Concat | Params: 47,713,234

🔷  EffSwin-Concat
────────────────────────────────────────────────────────────
  Ep01 | 107s | Tr:0.7552 Vl:0.5630 Acc:98.53% 🏅
  Ep02 | 108s | Tr:0.5632 Vl:0.5422 Acc:99.51% 🏅
  Ep03 | 107s | Tr:0.5507 Vl:0.5316 Acc:99.51% 🏅
  Ep04 | 107s | Tr:0.5349 Vl:0.5239 Acc:100.00% 🏅
  Ep05 | 107s | Tr:0.5305 Vl:0.5228 Acc:100.00% 🏅
  Ep06 | 107s | Tr:0.5291 Vl:0.5217 Acc:100.00% 🏅
  Ep07 | 107s | Tr:0.5276 Vl:0.5218 Acc:100.00% (1/5)
  Ep08 | 108s | Tr:0.5273 Vl:0.5210 Acc:100.00% 🏅
  Ep09 | 107s | Tr:0.5236 Vl:0.5209 Acc:100.00% 🏅
  Ep10 | 107s | Tr:0.5220 Vl:0.5207 Acc:100.00% 🏅
  Ep11 | 106s | Tr:0.5331 Vl:0.5307 Acc:99.75% (1/5)
  Ep12 | 107s | Tr:0.5318 Vl:0.5222 Acc:100.00% (2/5)
  Ep13 | 106s | Tr:0.5310 Vl:0.5232 Acc:100.00% (3/5)
  Ep14 | 107s | Tr:0.5323 Vl:0.5292 Acc:99.75% (4/5)
  Ep15 | 107s | Tr:0.5286 Vl:0.5392 Acc:99.26% (5/5)

  ✅ EffSwin-Concat [Test] | Acc:99.754%  MacF1:99.777%  AUC:1.0000

  📐 EffSwin-Hybrid (Ours) | Params: 49,342,552

🔷  EffSwin-Hybrid (Ours)
────────────────────────────────────────────────────────────
  Ep01 | 108s | Tr:1.2629 Vl:0.6639 Acc:94.59% 🏅
  Ep02 | 108s | Tr:0.6929 Vl:0.6094 Acc:96.07% 🏅
  Ep03 | 108s | Tr:0.6267 Vl:0.5863 Acc:97.05% 🏅
  Ep04 | 108s | Tr:0.6154 Vl:0.5774 Acc:98.03% 🏅
  Ep05 | 108s | Tr:0.5856 Vl:0.5557 Acc:99.02% 🏅
  Ep06 | 109s | Tr:0.5761 Vl:0.5550 Acc:99.26% 🏅
  Ep07 | 108s | Tr:0.5709 Vl:0.5529 Acc:99.26% 🏅
  Ep08 | 108s | Tr:0.5669 Vl:0.5519 Acc:99.26% 🏅
  Ep09 | 108s | Tr:0.5656 Vl:0.5500 Acc:99.26% 🏅
  Ep10 | 108s | Tr:0.5624 Vl:0.5492 Acc:99.26% 🏅
  Ep11 | 108s | Tr:0.5585 Vl:0.5470 Acc:99.26% 🏅
  Ep12 | 108s | Tr:0.5577 Vl:0.5410 Acc:99.26% 🏅
  Ep13 | 108s | Tr:0.5536 Vl:0.5369 Acc:99.26% 🏅
  Ep14 | 108s | Tr:0.5512 Vl:0.5278 Acc:99.75% 🏅
  Ep15 | 108s | Tr:0.5454 Vl:0.5274 Acc:100.00% 🏅
  Ep16 | 108s | Tr:0.5430 Vl:0.5315 Acc:99.75% (1/5)
  Ep17 | 108s | Tr:0.5436 Vl:0.5282 Acc:100.00% (2/5)
  Ep18 | 108s | Tr:0.5406 Vl:0.5248 Acc:100.00% 🏅
  Ep19 | 108s | Tr:0.5410 Vl:0.5236 Acc:100.00% 🏅
  Ep20 | 108s | Tr:0.5395 Vl:0.5242 Acc:100.00% (1/5)
  Ep21 | 108s | Tr:0.5411 Vl:0.5250 Acc:100.00% (2/5)
  Ep22 | 108s | Tr:0.5384 Vl:0.5249 Acc:99.75% (3/5)
  Ep23 | 108s | Tr:0.5364 Vl:0.5241 Acc:99.75% (4/5)
  Ep24 | 108s | Tr:0.5388 Vl:0.5239 Acc:100.00% (5/5)

  ✅ EffSwin-Hybrid (Ours) [Test] | Acc:99.509%  MacF1:99.553%  AUC:1.0000


════════════════════════════════════════════════════════════════════════════════
📊  PHASE 1 RESULTS — ABLATION LEADERBOARD  (Test Set)
════════════════════════════════════════════════════════════════════════════════
  Rank  Model                            Test Acc   Test MacF1     AUC  BestEp
────────────────────────────────────────────────────────────────────────────
  1     EffSwin-Concat                    99.754%      99.777%  1.0000      10 ◄ WINNER
  2     Swin-T+GeM                        99.509%      99.553%  0.9980      14
  3     EffSwin-Hybrid (Ours)             99.509%      99.553%  1.0000      19
  4     EfficientNetV2+ECA+GeM            99.509%      99.550%  0.9993      16
  5     ResNet50+CBAM+GeM                 99.263%      99.331%  0.9980      47
  6     ConvNeXt-T+CBAM                   99.263%      98.774%  0.9997       6
  7     VGG19+CBAM                        99.017%      98.595%  0.9987      30
  8     DenseNet121+CBAM                  98.771%      98.358%  0.9991      29
  9     MobileViT-S+GeM                   98.771%      97.914%  0.9993      12
════════════════════════════════════════════════════════════════════════════════

🏆 WINNER: EffSwin-Concat
   → Phase 2: 5-Fold Cross-Validation on full trainval set


  ✅ Phase 1 Ablation Chart saved

════════════════════════════════════════════════════════════════════════════════
🎯 PHASE 2 — FINAL 5-FOLD CROSS-VALIDATION
   Model   : EffSwin-Concat
   Patience: 6
   Data    : Full trainval (3658) + held-out test (407)
════════════════════════════════════════════════════════════════════════════════

────────────────────────────────────────────────────────────
  FOLD 1/5
────────────────────────────────────────────────────────────

🔷  EffSwin-Concat — Fold 1
────────────────────────────────────────────────────────────
  Ep01 | 100s | Tr:0.7723 Vl:0.5900 Acc:97.54% 🏅
  Ep02 | 100s | Tr:0.5715 Vl:0.5696 Acc:98.77% 🏅
  Ep03 | 100s | Tr:0.5493 Vl:0.5397 Acc:99.04% 🏅
  Ep04 | 100s | Tr:0.5377 Vl:0.5497 Acc:99.04% (1/6)
  Ep05 | 100s | Tr:0.5344 Vl:0.5388 Acc:99.32% 🏅
  Ep06 | 100s | Tr:0.5313 Vl:0.5411 Acc:99.32% (1/6)
  Ep07 | 100s | Tr:0.5296 Vl:0.5451 Acc:98.91% (2/6)
  Ep08 | 100s | Tr:0.5267 Vl:0.5425 Acc:99.18% (3/6)
  Ep09 | 100s | Tr:0.5257 Vl:0.5425 Acc:99.32% (4/6)
  Ep10 | 100s | Tr:0.5227 Vl:0.5427 Acc:99.32% (5/6)
  Ep11 | 100s | Tr:0.5370 Vl:0.5485 Acc:98.91% (6/6)

  ✅ EffSwin-Concat — Fold 1 [Val] | Acc:99.317%  MacF1:98.784%  AUC:0.9998

  📌 Fold 1 [OOF-Val] | Acc:99.317%  MacF1:98.784%  AUC:0.9998

────────────────────────────────────────────────────────────
  FOLD 2/5
────────────────────────────────────────────────────────────

🔷  EffSwin-Concat — Fold 2
────────────────────────────────────────────────────────────
  Ep01 | 100s | Tr:0.7910 Vl:0.6223 Acc:95.90% 🏅
  Ep02 | 100s | Tr:0.5713 Vl:0.5627 Acc:98.91% 🏅
  Ep03 | 100s | Tr:0.5540 Vl:0.5554 Acc:98.36% 🏅
  Ep04 | 100s | Tr:0.5351 Vl:0.5551 Acc:98.77% 🏅
  Ep05 |  99s | Tr:0.5319 Vl:0.5662 Acc:97.95% (1/6)
  Ep06 | 100s | Tr:0.5303 Vl:0.5454 Acc:98.63% 🏅
  Ep07 | 100s | Tr:0.5271 Vl:0.5356 Acc:99.32% 🏅
  Ep08 | 100s | Tr:0.5242 Vl:0.5317 Acc:99.59% 🏅
  Ep09 |  99s | Tr:0.5227 Vl:0.5317 Acc:99.45% (1/6)
  Ep10 |  99s | Tr:0.5244 Vl:0.5333 Acc:99.32% (2/6)
  Ep11 | 100s | Tr:0.5296 Vl:0.5562 Acc:98.63% (3/6)
  Ep12 |  99s | Tr:0.5359 Vl:0.5549 Acc:98.22% (4/6)
  Ep13 |  99s | Tr:0.5300 Vl:0.5400 Acc:98.91% (5/6)
  Ep14 | 100s | Tr:0.5316 Vl:0.5479 Acc:99.04% (6/6)

  ✅ EffSwin-Concat — Fold 2 [Val] | Acc:99.590%  MacF1:99.023%  AUC:1.0000

  📌 Fold 2 [OOF-Val] | Acc:99.590%  MacF1:99.023%  AUC:1.0000

────────────────────────────────────────────────────────────
  FOLD 3/5
────────────────────────────────────────────────────────────

🔷  EffSwin-Concat — Fold 3
────────────────────────────────────────────────────────────
  Ep01 | 100s | Tr:0.7753 Vl:0.5808 Acc:98.77% 🏅
  Ep02 | 100s | Tr:0.5765 Vl:0.5565 Acc:99.32% 🏅
  Ep03 | 100s | Tr:0.5476 Vl:0.5500 Acc:99.18% 🏅
  Ep04 | 100s | Tr:0.5362 Vl:0.5437 Acc:99.32% 🏅
  Ep05 | 100s | Tr:0.5363 Vl:0.5420 Acc:99.18% 🏅
  Ep06 | 100s | Tr:0.5285 Vl:0.5434 Acc:99.32% (1/6)
  Ep07 |  99s | Tr:0.5273 Vl:0.5422 Acc:99.32% (2/6)
  Ep08 | 100s | Tr:0.5285 Vl:0.5385 Acc:99.45% 🏅
  Ep09 | 100s | Tr:0.5241 Vl:0.5383 Acc:99.45% 🏅
  Ep10 | 100s | Tr:0.5247 Vl:0.5389 Acc:99.45% (1/6)
  Ep11 |  99s | Tr:0.5330 Vl:0.5476 Acc:98.91% (2/6)
  Ep12 |  99s | Tr:0.5351 Vl:0.5431 Acc:99.18% (3/6)
  Ep13 |  99s | Tr:0.5341 Vl:0.5394 Acc:99.32% (4/6)
  Ep14 | 100s | Tr:0.5300 Vl:0.5440 Acc:99.18% (5/6)
  Ep15 |  99s | Tr:0.5300 Vl:0.5454 Acc:99.04% (6/6)

  ✅ EffSwin-Concat — Fold 3 [Val] | Acc:99.454%  MacF1:98.586%  AUC:0.9928

  📌 Fold 3 [OOF-Val] | Acc:99.454%  MacF1:98.586%  AUC:0.9928

────────────────────────────────────────────────────────────
  FOLD 4/5
────────────────────────────────────────────────────────────

🔷  EffSwin-Concat — Fold 4
────────────────────────────────────────────────────────────
  Ep01 | 100s | Tr:0.7861 Vl:0.5742 Acc:98.50% 🏅
  Ep02 | 100s | Tr:0.5802 Vl:0.5624 Acc:98.91% 🏅
  Ep03 | 100s | Tr:0.5491 Vl:0.5586 Acc:98.77% 🏅
  Ep04 | 100s | Tr:0.5378 Vl:0.5502 Acc:98.91% 🏅
  Ep05 | 100s | Tr:0.5376 Vl:0.5436 Acc:99.18% 🏅
  Ep06 | 100s | Tr:0.5284 Vl:0.5480 Acc:99.18% (1/6)
  Ep07 | 100s | Tr:0.5280 Vl:0.5425 Acc:99.18% 🏅
  Ep08 | 100s | Tr:0.5244 Vl:0.5454 Acc:99.18% (1/6)
  Ep09 | 100s | Tr:0.5247 Vl:0.5452 Acc:99.18% (2/6)
  Ep10 | 100s | Tr:0.5232 Vl:0.5448 Acc:99.18% (3/6)
  Ep11 | 100s | Tr:0.5308 Vl:0.5617 Acc:98.63% (4/6)
  Ep12 | 100s | Tr:0.5409 Vl:0.5422 Acc:99.18% 🏅
  Ep13 | 100s | Tr:0.5311 Vl:0.5379 Acc:99.18% 🏅
  Ep14 | 100s | Tr:0.5340 Vl:0.5534 Acc:98.77% (1/6)
  Ep15 |  99s | Tr:0.5357 Vl:0.5436 Acc:99.32% (2/6)
  Ep16 | 100s | Tr:0.5279 Vl:0.5255 Acc:99.86% 🏅
  Ep17 | 100s | Tr:0.5271 Vl:0.5346 Acc:99.45% (1/6)
  Ep18 | 100s | Tr:0.5254 Vl:0.5299 Acc:99.59% (2/6)
  Ep19 |  99s | Tr:0.5250 Vl:0.5340 Acc:99.32% (3/6)
  Ep20 | 100s | Tr:0.5244 Vl:0.5289 Acc:99.45% (4/6)
  Ep21 | 100s | Tr:0.5214 Vl:0.5356 Acc:99.32% (5/6)
  Ep22 | 100s | Tr:0.5240 Vl:0.5345 Acc:99.45% (6/6)

  ✅ EffSwin-Concat — Fold 4 [Val] | Acc:99.863%  MacF1:99.916%  AUC:1.0000

  📌 Fold 4 [OOF-Val] | Acc:99.863%  MacF1:99.916%  AUC:1.0000

────────────────────────────────────────────────────────────
  FOLD 5/5
────────────────────────────────────────────────────────────

🔷  EffSwin-Concat — Fold 5
────────────────────────────────────────────────────────────
  Ep01 | 100s | Tr:0.7708 Vl:0.5895 Acc:98.08% 🏅
  Ep02 | 100s | Tr:0.5787 Vl:0.5564 Acc:98.91% 🏅
  Ep03 | 100s | Tr:0.5513 Vl:0.5534 Acc:98.77% 🏅
  Ep04 | 100s | Tr:0.5404 Vl:0.5498 Acc:98.91% 🏅
  Ep05 | 100s | Tr:0.5299 Vl:0.5555 Acc:98.77% (1/6)
  Ep06 | 100s | Tr:0.5266 Vl:0.5354 Acc:99.32% 🏅
  Ep07 | 100s | Tr:0.5245 Vl:0.5489 Acc:98.91% (1/6)
  Ep08 |  99s | Tr:0.5264 Vl:0.5437 Acc:99.04% (2/6)
  Ep09 | 100s | Tr:0.5240 Vl:0.5458 Acc:99.04% (3/6)
  Ep10 | 100s | Tr:0.5238 Vl:0.5447 Acc:99.18% (4/6)
  Ep11 |  99s | Tr:0.5336 Vl:0.5440 Acc:99.04% (5/6)
  Ep12 | 100s | Tr:0.5352 Vl:0.5393 Acc:99.32% (6/6)

  ✅ EffSwin-Concat — Fold 5 [Val] | Acc:99.316%  MacF1:99.414%  AUC:0.9998

  📌 Fold 5 [OOF-Val] | Acc:99.316%  MacF1:99.414%  AUC:0.9998


════════════════════════════════════════════════════════════════════════════════
📊  PHASE 2 — 5-FOLD CROSS-VALIDATION SUMMARY
════════════════════════════════════════════════════════════════════════════════
  Fold       Accuracy   Macro F1      AUC   BestEp
────────────────────────────────────────────────────
  Fold 1       99.317%     98.784%    0.9998         5
  Fold 2       99.590%     99.023%    1.0000         8
  Fold 3       99.454%     98.586%    0.9928         9
  Fold 4       99.863%     99.916%    1.0000        16
  Fold 5       99.316%     99.414%    0.9998         6
────────────────────────────────────────────────────
  Mean        99.508%     99.145%    0.9985
  ±Std         0.205%      0.474%    0.0028
════════════════════════════════════════════════════════════════════════════════

  OOF Metrics (combined):
  Accuracy : 99.5079%
  Macro F1 : 99.1425%
  AUC-ROC  : 0.9984

🏆 Best fold: Fold 4 — loading for final test evaluation...


════════════════════════════════════════════════════════════════════════════════
🎯 FINAL HELD-OUT TEST RESULTS
════════════════════════════════════════════════════════════════════════════════
  Metric                           Value
─────────────────────────────────────────────
  Accuracy                      99.7543%
  Macro Precision               99.8162%
  Macro Recall                  99.7396%
  Macro F1                      99.7758%
  Weighted F1                   99.7539%
  Macro AUC-ROC                   0.9980
─────────────────────────────────────────────

DETAILED CLASSIFICATION REPORT
                  precision    recall  f1-score   support

angular_leafspot     1.0000    1.0000    1.0000        43
     anthracnose     1.0000    1.0000    1.0000        10
  blossom_blight     1.0000    1.0000    1.0000        21
       gray_mold     1.0000    0.9792    0.9895        48
       leaf_spot     1.0000    1.0000    1.0000        61
  powdery_mildew     0.9853    1.0000    0.9926        67
     leaf_scorch     1.0000    1.0000    1.0000       111
         healthy     1.0000    1.0000    1.0000        46

        accuracy                         0.9975       407
       macro avg     0.9982    0.9974    0.9978       407
    weighted avg     0.9976    0.9975    0.9975       407

════════════════════════════════════════════════════════════════════════════════

  ✅ Confusion Matrix saved

  ✅ Per-Class F1 saved

  ✅ ROC Curves saved

  ✅ 5-Fold Accuracy saved

════════════════════════════════════════════════════════════════════════════════
🛡️  ROBUSTNESS TEST — Gaussian Noise
════════════════════════════════════════════════════════════════════════════════
  σ=0.00 → Acc: 99.75%  Macro F1: 99.78%
  σ=0.05 → Acc: 99.75%  Macro F1: 99.78%
  σ=0.10 → Acc: 99.75%  Macro F1: 99.78%
  σ=0.20 → Acc: 99.26%  Macro F1: 99.36%

  ✅ Robustness Analysis saved

════════════════════════════════════════════════════════════════════════════════
📸  XAI — GRAD-CAM++ & EIGEN-CAM
════════════════════════════════════════════════════════════════════════════════
📦 Installing grad-cam...
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 7.8/7.8 MB 42.0 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 12.2/12.2 MB 44.0 MB/s eta 0:00:00
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
dask-cuda 26.2.0 requires cuda-core==0.3.*, but you have cuda-core 1.0.1 which is incompatible.
dask-cuda 26.2.0 requires numba-cuda<0.23.0,>=0.22.1, but you have numba-cuda 0.30.2 which is incompatible.
distributed-ucxx-cu12 0.48.0 requires numba-cuda[cu12]<0.23.0,>=0.22.1, but you have numba-cuda 0.30.2 which is incompatible.
cuml-cu12 26.2.0 requires numba<0.62.0,>=0.60.0, but you have numba 0.65.1 which is incompatible.
cuml-cu12 26.2.0 requires numba-cuda[cu12]<0.23.0,>=0.22.1, but you have numba-cuda 0.30.2 which is incompatible.
ucxx-cu12 0.48.0 requires numba-cuda[cu12]<0.23.0,>=0.22.1, but you have numba-cuda 0.30.2 which is incompatible.
cudf-cu12 26.2.1 requires numba<0.62.0,>=0.60.0, but you have numba 0.65.1 which is incompatible.
cudf-cu12 26.2.1 requires numba-cuda[cu12]<0.23.0,>=0.22.2, but you have numba-cuda 0.30.2 which is incompatible.

  ✅ XAI Sample 1 saved

  ✅ XAI Sample 2 saved

  ✅ XAI Sample 3 saved

  ✅ XAI Sample 4 saved

  ✅ XAI Sample 5 saved

  ✅ XAI Sample 6 saved
✅  XAI complete!

🎉 🎉 🎉 🎉 🎉 🎉 🎉 🎉 🎉 🎉 🎉 🎉 🎉 🎉 🎉 🎉 🎉 🎉 🎉 🎉 

════════════════════════════════════════════════════════════════════════════════
📁 FINAL SUMMARY
════════════════════════════════════════════════════════════════════════════════
  Best Architecture  : EffSwin-Concat
  Phase-1 patience   : 5  (metrics on TEST set)
  Phase-2 patience   : 6   (5-fold OOF on val splits)
  5-Fold OOF Acc     : 99.5079%
  5-Fold OOF MacF1   : 99.1425%
  Held-out Test Acc  : 99.7543%
  Held-out MacF1     : 99.7758%
  Held-out AUC-ROC   : 0.9980
  Noise Resilience   : 0.49% drop at σ=0.20
════════════════════════════════════════════════════════════════════════════════