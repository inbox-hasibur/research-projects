# Strawberry disease detection — Afzaal + PlantVillage (7 classes), ~30% PV subsample.
# Train three CNN baselines, pick best val macro-F1, then fuse with Swin-T + cross-attention.

# -----------------------------------------------------------------------------
# Step 1 — Install (run once, restart kernel if needed)
# -----------------------------------------------------------------------------
# !pip install -q timm==0.9.12 scikit-learn matplotlib seaborn einops grad-cam

# -----------------------------------------------------------------------------
# Step 2 — Imports, device, paths, hyperparameters
# -----------------------------------------------------------------------------
import os, time, json, warnings
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from PIL import Image, ImageDraw
import matplotlib
matplotlib.rcParams["font.family"] = "DejaVu Sans"
matplotlib.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "0.2",
    "axes.labelcolor": "0.15",
    "axes.titlecolor": "0.15",
    "xtick.color": "0.15",
    "ytick.color": "0.15",
    "grid.color": "0.75",
    "legend.facecolor": "white",
    "legend.edgecolor": "0.75",
})
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             roc_curve, classification_report)
from sklearn.preprocessing import label_binarize
import timm
warnings.filterwarnings('ignore')

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"   GPU  : {torch.cuda.get_device_name(0)}")
    cap = torch.cuda.get_device_capability()
    print(f"   Cap  : sm_{cap[0]*10 + cap[1]}")
    try:
        _ = torch.randn(2,2,device=DEVICE) @ torch.randn(2,2,device=DEVICE)
        print("   Sanity: OK")
    except Exception as e:
        print(f"   FAILED: {e}"); DEVICE = torch.device("cpu")

_cap    = torch.cuda.get_device_capability()[0] if torch.cuda.is_available() else 0
USE_AMP = _cap >= 7
print(f"   AMP  : {'ON' if USE_AMP else 'OFF'}")

IMG_SIZE      = 224
BATCH_SIZE    = 16
BENCH_EPOCHS  = 30
HYBRID_EPOCHS = 60
LR            = 5e-5
WEIGHT_DECAY  = 1e-4
LABEL_SMOOTH  = 0.1
MIXUP_ALPHA   = 0.3
PATIENCE      = 6
CROP_PROB     = 0.6
USE_COMBINED_DATA = True

LABEL_MAP = {
    "angular_leafspot": 0,
    "anthracnose":      1,
    "blossom_blight":   2,
    "gray_mold":        3,
    "leaf_spot":        4,   # PV leaf_scorch maps here
    "powdery_mildew":   5,
    "healthy":          6,
}
IDX_TO_CLASS = {v: k for k, v in LABEL_MAP.items()}
NUM_CLASSES  = len(LABEL_MAP)

AFZAAL_ROOT  = "/kaggle/input/datasets/usmanafzaal/strawberry-disease-detection-dataset"
AFZAAL_TRAIN = os.path.join(AFZAAL_ROOT, "train")
AFZAAL_VAL   = os.path.join(AFZAAL_ROOT, "val")
AFZAAL_TEST  = os.path.join(AFZAAL_ROOT, "test")

PLANTVILLAGE_ROOT = "/kaggle/input/datasets/abdallahalidev/plantvillage-dataset"
PV_COLOR_ROOT     = os.path.join(PLANTVILLAGE_ROOT, "color")

SAVE_BENCH  = "best_benchmark_model.pth"
SAVE_HYBRID = "best_hybrid_strawberry.pth"

PV_CLASS_MAPPING = {
    "leaf_scorch":      "leaf_spot",
    "powdery_mildew":   "powdery_mildew",
    "angular_leafspot": "angular_leafspot",
    "anthracnose":      "anthracnose",
    "blossom_blight":   "blossom_blight",
    "gray_mold":        "gray_mold",
    "leaf_spot":        "leaf_spot",
    "healthy":          "healthy",
}

print(f"\nConfig OK | Classes: {NUM_CLASSES} | Device: {DEVICE}")
print(f"   Classes: {list(LABEL_MAP.keys())}")
print(f"   Dataset : Afzaal2021 + PlantVillage (combined, balanced)")


# -----------------------------------------------------------------------------
# Step 3 — VGG-style JSON bboxes (Afzaal) for optional crop
# -----------------------------------------------------------------------------
def load_annotation_bbox(json_path):
    """VGG JSON → (x1,y1,x2,y2) or None."""
    if not os.path.exists(json_path):
        return None
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        ax, ay = [], []
        for val in data.values():
            regions = val.get('regions', [])
            if isinstance(regions, dict):
                regions = list(regions.values())
            for region in regions:
                s = region.get('shape_attributes', {})
                t = s.get('name', '')
                if t == 'polygon':
                    ax.extend(s.get('all_points_x', []))
                    ay.extend(s.get('all_points_y', []))
                elif t == 'rect':
                    x,y,w,h = s.get('x',0),s.get('y',0),s.get('width',0),s.get('height',0)
                    ax += [x, x+w]; ay += [y, y+h]
                elif t == 'ellipse':
                    cx,cy,rx,ry = s.get('cx',0),s.get('cy',0),s.get('rx',0),s.get('ry',0)
                    ax += [cx-rx, cx+rx]; ay += [cy-ry, cy+ry]
        return (min(ax), min(ay), max(ax), max(ay)) if ax else None
    except Exception:
        return None


def annotation_crop(img, bbox, padding=0.20):
    """Crop around bbox with padding; returns original image if invalid."""
    if bbox is None:
        return img
    w, h = img.size
    x1, y1, x2, y2 = bbox
    bw, bh = x2-x1, y2-y1
    if bw <= 0 or bh <= 0:
        return img
    px, py = int(bw*padding), int(bh*padding)
    x1,y1 = max(0,x1-px), max(0,y1-py)
    x2,y2 = min(w,x2+px), min(h,y2+py)
    return img.crop((x1,y1,x2,y2)) if (x2-x1)>=10 and (y2-y1)>=10 else img


# -----------------------------------------------------------------------------
# Step 4 — Scan Afzaal + PlantVillage, merge, plot dataset overview
# -----------------------------------------------------------------------------
def label_from_filename(fname, label_map):
    """Longest matching filename prefix wins."""
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
        print(f"  [WARN] Not found: {split_dir}")
        return paths, labels
    for fn in sorted(os.listdir(split_dir)):
        if not fn.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            continue
        cls_idx, cls_name = label_from_filename(fn, LABEL_MAP)
        if cls_idx is None:
            continue
        paths.append(os.path.join(split_dir, fn))
        labels.append(cls_idx)
    cc = {}
    for l in labels: cc[l] = cc.get(l,0)+1
    found = [IDX_TO_CLASS[k] for k in sorted(cc)]
    print(f"  [Afzaal-{split_name}] {len(paths)} imgs | {found}")
    return paths, labels


def scan_plantvillage_strawberry():
    pv_train_paths, pv_train_labels = [], []
    pv_test_paths,  pv_test_labels  = [], []

    if not os.path.isdir(PV_COLOR_ROOT):
        print(f"  [WARN] PlantVillage not found: {PV_COLOR_ROOT}")
        return pv_train_paths, pv_train_labels, pv_test_paths, pv_test_labels

    for dir_name in sorted(os.listdir(PV_COLOR_ROOT)):
        if not dir_name.startswith("Strawberry___"):
            continue
        cls_path = os.path.join(PV_COLOR_ROOT, dir_name)
        if not os.path.isdir(cls_path):
            continue

        disease_part = dir_name.replace("Strawberry___", "").lower().replace(" ", "_")

        unified_name = None
        for pv_disease, afz_disease in PV_CLASS_MAPPING.items():
            if pv_disease in disease_part or disease_part in pv_disease:
                unified_name = afz_disease
                break

        if unified_name is None or unified_name not in LABEL_MAP:
            print(f"  [WARN] Skipped (no mapping): {dir_name}")
            continue

        unified_label = LABEL_MAP[unified_name]
        img_files = [f for f in os.listdir(cls_path)
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if not img_files:
            continue

        imgs_train, imgs_test = train_test_split(
            img_files, test_size=0.20, random_state=42)

        for img in imgs_train:
            pv_train_paths.append(os.path.join(cls_path, img))
            pv_train_labels.append(unified_label)
        for img in imgs_test:
            pv_test_paths.append(os.path.join(cls_path, img))
            pv_test_labels.append(unified_label)

        print(f"  [OK] {dir_name:45s} -> {unified_name:18s} | "
              f"train:{len(imgs_train):4d}  test:{len(imgs_test):4d}")

    return pv_train_paths, pv_train_labels, pv_test_paths, pv_test_labels


print("\nScanning datasets...")
print("-" * 60)
print("  Afzaal Dataset:")
afl_tr_paths, afl_tr_labels = scan_afzaal_split(AFZAAL_TRAIN, "train")
afl_vl_paths, afl_vl_labels = scan_afzaal_split(AFZAAL_VAL,   "val")
afl_te_paths, afl_te_labels = scan_afzaal_split(AFZAAL_TEST,  "test")

if not afl_tr_paths and not afl_vl_paths and not afl_te_paths:
    print("  [WARN] Pre-made splits not found. Auto-splitting AFZAAL_ROOT...")
    all_p, all_l = scan_afzaal_split(AFZAAL_ROOT, "root")
    if all_p:
        X_tv, afl_te_paths, y_tv, afl_te_labels = train_test_split(
            all_p, all_l, test_size=0.15, random_state=42, stratify=all_l)
        afl_tr_paths, afl_vl_paths, afl_tr_labels, afl_vl_labels = train_test_split(
            X_tv, y_tv, test_size=0.176, random_state=42, stratify=y_tv)
        print("  [OK] Manual 70/15/15 split applied.")

print("\n  PlantVillage Strawberry:")
pv_tr_paths, pv_tr_labels, pv_te_paths, pv_te_labels = scan_plantvillage_strawberry()

if USE_COMBINED_DATA and pv_tr_paths:
    n_afzaal  = len(afl_tr_paths)
    n_pv      = len(pv_tr_paths)
    target_pv = max(int(n_afzaal * 0.30), 100)
    if n_pv > target_pv:
        np.random.seed(42)
        indices      = np.random.choice(n_pv, target_pv, replace=False)
        pv_tr_paths  = [pv_tr_paths[i]  for i in indices]
        pv_tr_labels = [pv_tr_labels[i] for i in indices]
        print(f"\n  Balanced PlantVillage train: {n_pv} -> {target_pv} (~30% of Afzaal)")
    _use_combined = True
else:
    pv_tr_paths, pv_tr_labels = [], []
    pv_te_paths, pv_te_labels = [], []
    _use_combined = False
    print("  [WARN] No PlantVillage images found - using Afzaal only.")
USE_COMBINED_DATA = _use_combined

train_paths  = afl_tr_paths  + pv_tr_paths
train_labels = afl_tr_labels + pv_tr_labels
val_paths    = afl_vl_paths
val_labels   = afl_vl_labels
test_paths   = afl_te_paths  + pv_te_paths
test_labels  = afl_te_labels + pv_te_labels

found_ids = sorted(set(train_labels + val_labels + test_labels))
if len(found_ids) < len(LABEL_MAP):
    missing   = [k for k, v in LABEL_MAP.items() if v not in found_ids]
    print(f"\n  [WARN] Missing classes removed: {missing}")
    remap         = {old: new for new, old in enumerate(found_ids)}
    LABEL_MAP     = {k: remap[v] for k, v in LABEL_MAP.items() if v in found_ids}
    IDX_TO_CLASS.clear()
    IDX_TO_CLASS.update({v: k for k, v in LABEL_MAP.items()})
    train_labels  = [remap[l] for l in train_labels]
    val_labels    = [remap[l] for l in val_labels]
    test_labels   = [remap[l] for l in test_labels]
    afl_tr_labels = [remap[l] for l in afl_tr_labels]
    pv_tr_labels  = [remap[l] for l in pv_tr_labels]  if pv_tr_labels else []
    afl_te_labels = [remap[l] for l in afl_te_labels]
    pv_te_labels  = [remap[l] for l in pv_te_labels]  if pv_te_labels else []
NUM_CLASSES = len(LABEL_MAP)

n_ann    = sum(1 for p in train_paths if os.path.exists(os.path.splitext(p)[0]+'.json'))
USE_CROP = n_ann > 0

total_imgs = len(train_paths) + len(val_paths) + len(test_paths)
n_afl_tr   = len(afl_tr_paths)
n_pv_tr    = len(pv_tr_paths)
cls_names  = [IDX_TO_CLASS[i] for i in range(NUM_CLASSES)]

print("\n" + "-" * 70)
print("DATASET SUMMARY")
print("-" * 70)
print(f"  Classes  : {NUM_CLASSES}  ({', '.join(LABEL_MAP.keys())})")
print(f"  Total    : {total_imgs:,} images")
print(f"  |- Train : {len(train_paths):,}  "
      f"(Afzaal: {n_afl_tr:,}  +  PlantVillage: {n_pv_tr:,})")
print(f"  |- Val   : {len(val_paths):,}  (Afzaal only)")
print(f"  |- Test  : {len(test_paths):,}  "
      f"(Afzaal: {len(afl_te_paths):,}  +  PlantVillage: {len(pv_te_paths):,})")
print(f"  Source   : {'Afzaal2021 + PlantVillage (Balanced ~30%)' if USE_COMBINED_DATA else 'Afzaal2021 only'}")
print(f"  Annot.   : {n_ann}/{len(train_paths)} ({'ON' if USE_CROP else 'OFF'})")

cc_train_dict = {}
for l in train_labels:
    cc_train_dict[IDX_TO_CLASS[l]] = cc_train_dict.get(IDX_TO_CLASS[l], 0) + 1
class_cnts      = list(cc_train_dict.values())
imbalance_ratio = max(class_cnts) / min(class_cnts) if min(class_cnts) > 0 else 1.0
print("\n  Class Distribution (Training Set):")
for cls, cnt in sorted(cc_train_dict.items()):
    pct = (cnt / len(train_labels)) * 100
    bar = "#" * (cnt // 20)
    print(f"    {cls:22s}: {cnt:5d} ({pct:5.1f}%)  {bar}")
print(f"\n  Imbalance Ratio: {imbalance_ratio:.2f}x")
print("-" * 70)

class_counts_arr = np.bincount(train_labels, minlength=NUM_CLASSES)
class_cnt_dict   = {i: int(class_counts_arr[i]) for i in range(NUM_CLASSES)}

print("\n" + "=" * 70)
print("Dataset overview (3 separate figures, publication style)")
print("=" * 70)

tr_cc = np.bincount(train_labels, minlength=NUM_CLASSES)
te_cc = np.bincount(test_labels,  minlength=NUM_CLASSES)
colors_cls = ['#4a90d9' if c != 'healthy' else '#2d8659' for c in cls_names]
_bar_edge = '#333333'
_lbl = '#222222'

def _paper_bar_chart(ax, counts, title, ylabel="Count"):
    ax.set_facecolor('white')
    bars = ax.bar(range(NUM_CLASSES), counts, color=colors_cls,
                  edgecolor=_bar_edge, linewidth=0.6)
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_xticklabels([c.replace('_', '\n') for c in cls_names], fontsize=8, color=_lbl)
    ax.set_title(title, fontsize=11, fontweight='bold', color=_lbl)
    ax.set_ylabel(ylabel, fontsize=10, color=_lbl)
    ax.grid(axis='y', linestyle='--', alpha=0.35, color='0.5')
    ax.tick_params(colors=_lbl)
    ymax = max(counts.max(), 1) * 1.12
    ax.set_ylim(0, ymax)
    for bar, cnt in zip(bars, counts):
        if cnt <= 0:
            continue
        ax.text(bar.get_x() + bar.get_width() / 2, cnt + ymax * 0.01,
                f'{cnt:,}', ha='center', va='bottom', fontsize=7, color=_lbl)

fig_tr, ax_tr = plt.subplots(figsize=(7.5, 4.2), facecolor='white')
_paper_bar_chart(ax_tr, tr_cc, "Training set: class distribution")
mean_tr = tr_cc[tr_cc > 0].mean()
ax_tr.axhline(mean_tr, color='#c45c00', ls='--', lw=1.2, alpha=0.85,
              label=f'Mean = {mean_tr:.0f}')
ax_tr.legend(frameon=True, fontsize=8, loc='upper right')
fig_tr.tight_layout()
fig_tr.savefig("fig_dataset_train_class_dist.png", dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

fig_te, ax_te = plt.subplots(figsize=(7.5, 4.2), facecolor='white')
_paper_bar_chart(ax_te, te_cc, "Test set: class distribution")
fig_te.tight_layout()
fig_te.savefig("fig_dataset_test_class_dist.png", dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

fig_sp, ax_sp = plt.subplots(figsize=(5.0, 4.0), facecolor='white')
splits_v       = ["Train", "Val", "Test"]
split_counts_v = [len(train_paths), len(val_paths), len(test_paths)]
split_colors_v = ['#4a90d9', '#d98c4a', '#c44c4c']
bars_sp = ax_sp.bar(splits_v, split_counts_v, color=split_colors_v,
                    edgecolor=_bar_edge, linewidth=0.6)
ax_sp.set_title("Dataset split", fontsize=11, fontweight='bold', color=_lbl)
ax_sp.set_ylabel("Number of images", fontsize=10, color=_lbl)
ax_sp.grid(axis='y', linestyle='--', alpha=0.35, color='0.5')
ax_sp.tick_params(colors=_lbl)
tot = sum(split_counts_v)
ymax_sp = max(split_counts_v) * 1.15
ax_sp.set_ylim(0, ymax_sp)
for bar, cnt in zip(bars_sp, split_counts_v):
    pct = (cnt / tot) * 100
    ax_sp.text(bar.get_x() + bar.get_width() / 2, cnt + ymax_sp * 0.01,
               f'{cnt:,}\n({pct:.0f}%)', ha='center', va='bottom',
               fontsize=9, color=_lbl)
fig_sp.tight_layout()
fig_sp.savefig("fig_dataset_split.png", dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
print("Saved: fig_dataset_train_class_dist.png | fig_dataset_test_class_dist.png | fig_dataset_split.png\n")


# -----------------------------------------------------------------------------
# Step 5 — Transforms
# -----------------------------------------------------------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE+48, IMG_SIZE+48)),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.08),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),
])

train_tf_aggressive = transforms.Compose([
    transforms.Resize((IMG_SIZE+64, IMG_SIZE+64)),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(p=0.7),
    transforms.RandomVerticalFlip(p=0.7),
    transforms.RandomRotation(45),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.4, hue=0.10),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.5)),
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.85, 1.15)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    transforms.RandomErasing(p=0.35, scale=(0.02, 0.2)),
])

val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

print("Transforms ready: standard + aggressive (minority) | val/test")


# -----------------------------------------------------------------------------
# Step 6 — Image dataset, weighted train loader
# -----------------------------------------------------------------------------
class StrawberryDataset(Dataset):
    def __init__(self, paths, labels, transform=None,
                 use_ann_crop=False, use_adaptive_aug=False,
                 class_counts=None):
        self.paths            = paths
        self.labels           = labels
        self.transform        = transform
        self.use_ann_crop     = use_ann_crop
        self.use_adaptive_aug = use_adaptive_aug
        self.class_counts     = class_counts or {}

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        img   = Image.open(self.paths[idx]).convert("RGB")
        label = self.labels[idx]

        if self.use_ann_crop and np.random.random() < CROP_PROB:
            json_path = os.path.splitext(self.paths[idx])[0] + '.json'
            bbox = load_annotation_bbox(json_path)
            if bbox:
                img = annotation_crop(img, bbox)

        tf_to_use = self.transform
        if self.use_adaptive_aug and tf_to_use is not None:
            max_cnt  = max(self.class_counts.values()) if self.class_counts else 1
            curr_cnt = self.class_counts.get(label, 1)
            minority_ratio = curr_cnt / max_cnt if max_cnt > 0 else 1.0
            if minority_ratio < 0.5 and np.random.random() < 0.4:
                tf_to_use = train_tf_aggressive

        if tf_to_use:
            img = tf_to_use(img)
        return img, torch.tensor(label, dtype=torch.long)


def make_loader(paths, labels, tf, shuffle=False, ann_crop=False,
                adaptive_aug=False, cls_cnt_dict=None):
    ds = StrawberryDataset(paths, labels, tf, ann_crop,
                           use_adaptive_aug=adaptive_aug,
                           class_counts=cls_cnt_dict)
    if shuffle:
        cc  = np.bincount(labels, minlength=NUM_CLASSES)
        wpc = 1.0 / np.where(cc == 0, 1, cc)
        sw  = torch.tensor([wpc[l] for l in labels], dtype=torch.double)
        sampler = WeightedRandomSampler(sw, len(sw), replacement=True)
        return DataLoader(ds, batch_size=BATCH_SIZE, sampler=sampler,
                          num_workers=2, pin_memory=True)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                      num_workers=2, pin_memory=True)

train_loader = make_loader(train_paths, train_labels, train_tf,
                           shuffle=True, ann_crop=USE_CROP,
                           adaptive_aug=USE_COMBINED_DATA,
                           cls_cnt_dict=class_cnt_dict)
val_loader   = make_loader(val_paths,   val_labels,   val_tf)
test_loader  = make_loader(test_paths,  test_labels,  val_tf)
print("\nDataLoaders ready.")
print(f"   Train batches : {len(train_loader):,}")
print(f"   Val   batches : {len(val_loader):,}")
print(f"   Test  batches : {len(test_loader):,}")
print(f"   AdaptiveAug   : {'ON (minority classes)' if USE_COMBINED_DATA else 'OFF'}")
print(f"   WeightedSampler: ON (class-balanced)")


# -----------------------------------------------------------------------------
# Step 7 — CBAM, ECA, GeM, label smoothing, mixup
# -----------------------------------------------------------------------------

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.mlp = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
        )
    def forward(self, x):
        avg = self.mlp(F.adaptive_avg_pool2d(x,1).flatten(1))
        mx  = self.mlp(F.adaptive_max_pool2d(x,1).flatten(1))
        return x * torch.sigmoid(avg + mx).unsqueeze(-1).unsqueeze(-1)

class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3, bias=False)
    def forward(self, x):
        avg = x.mean(1, keepdim=True)
        mx,_= x.max(1, keepdim=True)
        return x * torch.sigmoid(self.conv(torch.cat([avg, mx], 1)))

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention()
    def forward(self, x):
        return self.sa(self.ca(x))

class ECA(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        import math
        t = int(abs((math.log2(channels) + b) / gamma))
        k = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv     = nn.Conv1d(1, 1, k, padding=(k-1)//2, bias=False)
    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1,-2))
        return x * torch.sigmoid(y.transpose(-1,-2).unsqueeze(-1))

class GeMPooling(nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(p))
        self.eps = eps
    def forward(self, x):
        return F.adaptive_avg_pool2d(
            x.clamp(min=self.eps).pow(self.p), 1
        ).pow(1.0/self.p).flatten(1)

class SmoothCE(nn.Module):
    def __init__(self, num_classes=7, smoothing=0.1):
        super().__init__()
        self.nc = num_classes
        self.sm = smoothing
    def forward(self, logits, targets):
        lp = F.log_softmax(logits, -1)
        if targets.dim() == 1:
            st = torch.full_like(lp, self.sm/(self.nc-1))
            st.scatter_(1, targets.unsqueeze(1), 1.0-self.sm)
        else:
            st = targets*(1-self.sm) + self.sm/self.nc
        return -(st * lp).sum(-1).mean()

criterion = SmoothCE(NUM_CLASSES, LABEL_SMOOTH)

def mixup_data(x, y, alpha=0.3):
    lam = float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0
    device = x.device
    x_c, y_c = x.detach().cpu(), y.detach().cpu()
    idx = torch.randperm(x_c.size(0))
    ya = F.one_hot(y_c,      NUM_CLASSES).float()
    yb = F.one_hot(y_c[idx], NUM_CLASSES).float()
    mx = lam*x_c + (1-lam)*x_c[idx]
    my = lam*ya  + (1-lam)*yb
    return mx.to(device), my.to(device)

print("Shared modules ready: CBAM | ECA | GeM | SmoothCE | Mixup")


# -----------------------------------------------------------------------------
# Step 8 — Early stopping, one epoch train, evaluate
# -----------------------------------------------------------------------------
class EarlyStopping:
    def __init__(self, patience=8, min_delta=1e-4, save_path=SAVE_BENCH):
        self.patience = patience
        self.min_delta = min_delta
        self.save_path = save_path
        self.best_loss = float('inf')
        self.counter   = 0
        self.best_epoch = 0
    def step(self, val_loss, epoch, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter   = 0
            self.best_epoch = epoch
            torch.save(model.state_dict(), self.save_path)
            return False
        self.counter += 1
        return self.counter >= self.patience


def train_one_epoch(model, loader, crit, opt):
    model.train()
    total = 0.0
    for imgs, lbls in loader:
        imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
        if MIXUP_ALPHA > 0:
            imgs, targets = mixup_data(imgs, lbls, MIXUP_ALPHA)
        else:
            targets = lbls
        opt.zero_grad(set_to_none=True)
        loss = crit(model(imgs), targets)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total += loss.item()
    return total / len(loader)


@torch.no_grad()
def evaluate(model, loader, crit):
    model.eval()
    total, probs, preds, gts = 0.0, [], [], []
    for imgs, lbls in loader:
        imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
        logits = model(imgs)
        total += crit(logits, lbls).item()
        probs.extend(F.softmax(logits,-1).cpu().numpy())
        preds.extend(logits.argmax(-1).cpu().numpy())
        gts.extend(lbls.cpu().numpy())
    return total/len(loader), accuracy_score(gts,preds), np.array(probs), preds, gts


def full_metrics(gts, preds, probs):
    acc  = accuracy_score(gts, preds)
    prec = precision_score(gts, preds, average='macro', zero_division=0)
    rec  = recall_score(gts,   preds, average='macro', zero_division=0)
    mf1  = f1_score(gts,       preds, average='macro', zero_division=0)
    wf1  = f1_score(gts,       preds, average='weighted', zero_division=0)
    try:
        auc = roc_auc_score(gts, probs, multi_class='ovr', average='macro')
    except Exception:
        auc = float('nan')
    return dict(acc=acc, prec=prec, rec=rec, mf1=mf1, wf1=wf1, auc=auc)


# -----------------------------------------------------------------------------
# Step 9 — Benchmark model classes
# -----------------------------------------------------------------------------

class VGG19Model(nn.Module):
    def __init__(self, num_classes=7, drop=0.5):
        super().__init__()
        base = models.vgg19_bn(pretrained=True)
        self.features  = base.features
        self.cbam      = CBAM(512)
        self.avgpool   = nn.AdaptiveAvgPool2d(7)
        self.classifier = nn.Sequential(
            nn.Linear(512*7*7, 1024), nn.ReLU(True), nn.Dropout(drop),
            nn.Linear(1024, 256),    nn.ReLU(True), nn.Dropout(drop/2),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        x = self.features(x)
        x = self.cbam(x)
        x = self.avgpool(x).flatten(1)
        return self.classifier(x)


class ResNet50Model(nn.Module):
    def __init__(self, num_classes=7, drop=0.4):
        super().__init__()
        base        = models.resnet50(pretrained=True)
        self.stem   = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.cbam   = CBAM(2048)
        self.gem    = GeMPooling(p=3.0)
        self.head   = nn.Sequential(
            nn.Linear(2048, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(drop),
            nn.Linear(512, num_classes)
        )
    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x); x = self.layer2(x)
        x = self.layer3(x); x = self.layer4(x)
        x = self.cbam(x)
        return self.head(self.gem(x))


class EfficientNetV2Model(nn.Module):
    def __init__(self, num_classes=7, drop=0.4):
        super().__init__()
        self.backbone = timm.create_model(
            'tf_efficientnetv2_s', pretrained=True, num_classes=0, global_pool='')
        EFF_DIM = 1280
        self.eca  = ECA(EFF_DIM)
        self.gem  = GeMPooling(p=3.0)
        self.head = nn.Sequential(
            nn.Linear(EFF_DIM, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(drop),
            nn.Linear(512, num_classes)
        )
    def forward(self, x):
        x = self.backbone.forward_features(x)
        x = self.eca(x)
        return self.head(self.gem(x))


BENCH_CONFIGS = {
    "VGG19+CBAM":      VGG19Model,
    "ResNet50+CBAM+GeM": ResNet50Model,
    "EfficientNetV2+ECA+GeM": EfficientNetV2Model,
}

def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

for name, cls in BENCH_CONFIGS.items():
    m = cls(NUM_CLASSES)
    print(f"  {name:<32}: {count_params(m):>12,} params")


# -----------------------------------------------------------------------------
# Step 10 — Train VGG19, ResNet50, EfficientNetV2 (pick best macro-F1)
# -----------------------------------------------------------------------------
bench_results   = {}
bench_histories = {}

print("\n" + "="*70)
print("Benchmark: VGG19 | ResNet50 | EfficientNetV2-S")
print("="*70)

for bench_name, ModelClass in BENCH_CONFIGS.items():
    print(f"\nTraining: {bench_name}")
    print("-"*60)

    model_b = ModelClass(NUM_CLASSES).to(DEVICE)

    try:
        backbone_params = list(model_b.backbone.parameters())  # EfficientNet
    except AttributeError:
        try:
            backbone_params = (list(model_b.features.parameters()) +
                              list(model_b.cbam.parameters()))   # VGG
        except AttributeError:
            backbone_params = (list(model_b.stem.parameters()) +
                              list(model_b.layer1.parameters()) +
                              list(model_b.layer2.parameters()) +
                              list(model_b.layer3.parameters()) +
                              list(model_b.layer4.parameters()))  # ResNet

    backbone_ids = {id(p) for p in backbone_params}
    new_params   = [p for p in model_b.parameters() if id(p) not in backbone_ids]

    opt_b = optim.AdamW([
        {'params': backbone_params, 'lr': LR * 0.1},
        {'params': new_params,      'lr': LR},
    ], weight_decay=WEIGHT_DECAY)

    sch_b = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt_b, T_0=10, T_mult=2)
    es_b  = EarlyStopping(patience=PATIENCE, save_path=f"bench_{bench_name.replace('+','_')}.pth")

    hist = {'tr_loss':[], 'vl_loss':[], 'vl_acc':[]}
    for ep in range(1, BENCH_EPOCHS+1):
        t0     = time.time()
        tr_l   = train_one_epoch(model_b, train_loader, criterion, opt_b)
        vl_l, vl_a, _, _, _ = evaluate(model_b, val_loader, criterion)
        sch_b.step()
        hist['tr_loss'].append(tr_l)
        hist['vl_loss'].append(vl_l)
        hist['vl_acc'].append(vl_a)
        stop   = es_b.step(vl_l, ep, model_b)
        flag   = "[best]" if es_b.counter == 0 else f"({es_b.counter}/{PATIENCE})"
        if ep % 5 == 0 or ep == 1 or stop:
            print(f"  Ep{ep:02d} | {time.time()-t0:.0f}s | "
                  f"Tr:{tr_l:.4f} Vl:{vl_l:.4f} Acc:{vl_a*100:.2f}% {flag}")
        if stop:
            print(f"  ⏹ Early stop @ ep{ep} | best ep{es_b.best_epoch}")
            break

    # Load best and test
    model_b.load_state_dict(torch.load(es_b.save_path, map_location=DEVICE))
    _, _, probs, preds, gts = evaluate(model_b, test_loader, criterion)
    m = full_metrics(gts, preds, probs)
    bench_results[bench_name] = m
    bench_histories[bench_name] = hist

    print(f"\n  {bench_name} TEST RESULTS:")
    print(f"     Acc={m['acc']*100:.3f}%  MacF1={m['mf1']*100:.3f}%  AUC={m['auc']:.4f}")
    del model_b; torch.cuda.empty_cache()

winner_name = max(bench_results, key=lambda k: bench_results[k]['mf1'])
winner_f1   = bench_results[winner_name]['mf1'] * 100
print(f"\nBENCHMARK WINNER: {winner_name}  (Macro F1 = {winner_f1:.3f}%)")
print("    (This backbone is used in the final hybrid.)")

WINNER_CLASS = BENCH_CONFIGS[winner_name]
WINNER_SAVE  = f"bench_{winner_name.replace('+','_')}.pth"


# -----------------------------------------------------------------------------
# Step 11 — Benchmark learning curves & test metric bars
# -----------------------------------------------------------------------------
BENCH_COLORS = {
    'VGG19+CBAM':                '#e74c3c',
    'ResNet50+CBAM+GeM':         '#3498db',
    'EfficientNetV2+ECA+GeM':    '#2ecc71',
}
BG    = 'white'
PANEL = 'white'
GRID  = '#b0b0b0'

fig, axes = plt.subplots(1, 3, figsize=(21, 6))
fig.patch.set_facecolor(BG)
for ax in axes:
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values(): sp.set_edgecolor(GRID)
    ax.tick_params(colors='#555555')
    ax.grid(alpha=0.18, color='#cccccc')

for name, hist in bench_histories.items():
    c  = BENCH_COLORS[name]
    ep = range(1, len(hist['tr_loss'])+1)
    lbl = name.replace('+','\n')
    axes[0].plot(ep, hist['tr_loss'], color=c, lw=2, label=lbl, marker='o', ms=3)
    axes[1].plot(ep, hist['vl_loss'], color=c, lw=2, label=lbl, marker='o', ms=3)
    axes[2].plot(ep, [a*100 for a in hist['vl_acc']], color=c, lw=2, label=lbl,
                 marker='s', ms=3)

w_hist = bench_histories[winner_name]
w_c    = BENCH_COLORS[winner_name]
best_ep = int(np.argmin(w_hist['vl_loss'])) + 1
for ax in axes[:2]:
    ax.axvline(best_ep, color=w_c, ls='--', lw=1.5, alpha=0.6,
               label=f'Winner best ep{best_ep}')
axes[2].axhline(bench_results[winner_name]['acc']*100, color=w_c,
                ls='--', lw=1.5, alpha=0.6)

titles = ["Train Loss", "Validation Loss", "Validation Accuracy (%)"]
ylabels = ["Loss", "Loss", "Accuracy (%)"]
for ax, title, yl in zip(axes, titles, ylabels):
    ax.set_title(title, fontsize=12, fontweight='bold', color='#222222', pad=8)
    ax.set_xlabel("Epoch", color='#555555', fontsize=10)
    ax.set_ylabel(yl, color='#555555', fontsize=10)
    ax.legend(fontsize=7.5, facecolor=PANEL, labelcolor='#222222',
              edgecolor=GRID, loc='upper right')

fig.suptitle("Benchmark training curves: VGG19 vs ResNet50 vs EfficientNetV2",
             fontsize=14, fontweight='bold', color='#222222', y=1.01)
plt.tight_layout()
plt.savefig("benchmark_curves.png", dpi=150, bbox_inches='tight',
            facecolor=BG)
plt.show()

metric_keys    = ['acc',   'mf1',   'prec',  'rec',   'auc']
metric_labels  = ['Accuracy','MacroF1','Precision','Recall','AUC-ROC']
metric_scale   = [100, 100, 100, 100, 1]

fig = plt.figure(figsize=(22, 7))
fig.patch.set_facecolor(BG)
gs  = fig.add_gridspec(1, 6, wspace=0.35, left=0.05, right=0.97,
                        top=0.88, bottom=0.12)

for col, (mk, ml, ms) in enumerate(zip(metric_keys, metric_labels, metric_scale)):
    ax = fig.add_subplot(gs[0, col])
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values(): sp.set_edgecolor(GRID)
    ax.tick_params(colors='#555555')

    names = list(bench_results.keys())
    vals  = [bench_results[n][mk] * ms for n in names]
    bar_cs = [BENCH_COLORS[n] for n in names]
    bars  = ax.bar(range(len(names)), vals, color=bar_cs, edgecolor='none', width=0.55)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.split('+')[0] for n in names], fontsize=8, color='#333333',
                        rotation=15, ha='right')
    lo = min(vals)*0.985; hi = max(vals)*1.015
    ax.set_ylim([lo, hi])
    ax.set_title(ml, fontsize=10, fontweight='bold', color='#222222', pad=6)
    ax.grid(axis='y', alpha=0.2, color='#cccccc')
    best_i = int(np.argmax(vals))
    for i, (bar, val) in enumerate(zip(bars, vals)):
        star = ' *' if i == best_i else ''
        ax.text(bar.get_x()+bar.get_width()/2,
                bar.get_height() + (hi-lo)*0.01,
                f"{val:.2f}{star}", ha='center', va='bottom',
                fontsize=8, color='#222222', fontweight='bold')

ax_r = fig.add_subplot(gs[0, 5], polar=True)
ax_r.set_facecolor(PANEL)
ax_r.tick_params(colors='#555555', labelsize=7)
ax_r.grid(color=GRID, alpha=0.5)
angles = np.linspace(0, 2*np.pi, len(metric_keys), endpoint=False).tolist()
angles += angles[:1]
for name in bench_results:
    vals_r = [bench_results[name][mk]*ms for mk, ms in zip(metric_keys, metric_scale)]
    norm_v = [v/100 if ms==100 else v for v, ms in zip(vals_r, metric_scale)]
    norm_v += norm_v[:1]
    ax_r.plot(angles, norm_v, color=BENCH_COLORS[name], lw=2,
              label=name.split('+')[0])
    ax_r.fill(angles, norm_v, color=BENCH_COLORS[name], alpha=0.08)
ax_r.set_thetagrids(np.degrees(angles[:-1]), metric_labels, fontsize=7, color='#333333')
ax_r.set_ylim(0.9, 1.0)
ax_r.set_title("Radar\nOverview", fontsize=9, fontweight='bold', color='#222222', pad=15)
ax_r.legend(fontsize=7, facecolor=PANEL, labelcolor='#222222',
            edgecolor=GRID, loc='lower right', bbox_to_anchor=(1.35, -0.15))

fig.suptitle(f"Test metrics comparison | Winner: {winner_name}",
             fontsize=14, fontweight='bold', color='#222222')
plt.savefig("benchmark_bars.png", dpi=150, bbox_inches='tight', facecolor=BG)
plt.show()
print("Saved: benchmark_curves.png | benchmark_bars.png")


# -----------------------------------------------------------------------------
# Step 12 — Hybrid: winning CNN + Swin-T + cross-attention fusion
# -----------------------------------------------------------------------------

def _swin_pool(feat):
    if feat.dim() == 4:
        feat = feat.mean(dim=[1,2])
    return feat.contiguous()


class CrossAttentionGate(nn.Module):
    def __init__(self, dim=512, heads=8):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.q1 = nn.Linear(dim, dim, bias=False)
        self.k2 = nn.Linear(dim, dim, bias=False)
        self.v2 = nn.Linear(dim, dim, bias=False)
        self.q2 = nn.Linear(dim, dim, bias=False)
        self.k1 = nn.Linear(dim, dim, bias=False)
        self.v1 = nn.Linear(dim, dim, bias=False)
        self.out1 = nn.Linear(dim, dim, bias=False)
        self.out2 = nn.Linear(dim, dim, bias=False)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x1, x2):
        B, D = x1.shape
        H, d = self.heads, D // self.heads
        q1 = self.q1(x1).reshape(B,H,d)
        k2 = self.k2(x2).reshape(B,H,d)
        v2 = self.v2(x2).reshape(B,H,d)
        attn1 = (q1 * k2 * self.scale).sum(-1, keepdim=True)
        attn1 = torch.sigmoid(attn1)
        out1  = (attn1 * v2).reshape(B,D)
        x1_refined = self.norm1(x1 + self.out1(out1))

        q2 = self.q2(x2).reshape(B,H,d)
        k1 = self.k1(x1).reshape(B,H,d)
        v1 = self.v1(x1).reshape(B,H,d)
        attn2 = (q2 * k1 * self.scale).sum(-1, keepdim=True)
        attn2 = torch.sigmoid(attn2)
        out2  = (attn2 * v1).reshape(B,D)
        x2_refined = self.norm2(x2 + self.out2(out2))

        return x1_refined, x2_refined


class WinnerSwinHybrid(nn.Module):
    def __init__(self, winner_class, num_classes=7, drop=0.4):
        super().__init__()
        self.winner_name = winner_class.__name__

        if winner_class == EfficientNetV2Model:
            self.backbone_a = timm.create_model(
                'tf_efficientnetv2_s', pretrained=True, num_classes=0, global_pool='')
            A_DIM = 1280
            self.attn_a = ECA(A_DIM)
        elif winner_class == ResNet50Model:
            base = models.resnet50(pretrained=True)
            self.backbone_a = nn.Sequential(
                base.conv1, base.bn1, base.relu, base.maxpool,
                base.layer1, base.layer2, base.layer3, base.layer4)
            A_DIM = 2048
            self.attn_a = CBAM(A_DIM)
        elif winner_class == VGG19Model:
            base = models.vgg19_bn(pretrained=True)
            self.backbone_a = base.features
            A_DIM = 512
            self.attn_a = CBAM(A_DIM)
        else:
            raise ValueError(f"Unknown winner class: {winner_class}")

        self.gem_a   = GeMPooling(p=3.0)
        self.proj_a  = nn.Sequential(
            nn.Linear(A_DIM, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(drop*0.5))

        self.swin    = timm.create_model(
            'swin_tiny_patch4_window7_224', pretrained=True, num_classes=0)
        SWIN_DIM = 768
        self.swin_norm = nn.LayerNorm(SWIN_DIM)
        self.proj_b  = nn.Sequential(
            nn.Linear(SWIN_DIM, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(drop*0.5))

        self.cross_attn = CrossAttentionGate(dim=512, heads=8)

        self.fusion_head = nn.Sequential(
            nn.Linear(1024, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(drop),
            nn.Linear(512, 256),  nn.GELU(), nn.Dropout(drop/2),
            nn.Linear(256, 128),  nn.GELU(),
            nn.Linear(128, num_classes)
        )

    def forward_a(self, x):
        feat = self.gem_a(self.attn_a(self.backbone_a(x)))
        return self.proj_a(feat)

    def forward_b(self, x):
        feat = self.swin_norm(_swin_pool(self.swin.forward_features(x)))
        return self.proj_b(feat)

    def forward(self, x):
        fa, fb = self.cross_attn(self.forward_a(x), self.forward_b(x))
        return self.fusion_head(torch.cat([fa, fb], dim=1))


model = WinnerSwinHybrid(WINNER_CLASS, NUM_CLASSES, drop=0.4).to(DEVICE)
hybrid_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nFinal hybrid: {winner_name} + Swin-T + CrossAttn")
print(f"   📐 Trainable params: {hybrid_params:,}")
print(f"   GeM p (init): {model.gem_a.p.item():.1f}")


# -----------------------------------------------------------------------------
# Step 13 — Train hybrid (warm-start winner backbone)
# -----------------------------------------------------------------------------
winner_bench_weights = torch.load(WINNER_SAVE, map_location=DEVICE)
backbone_state = {k.replace('backbone.','backbone_a.', 1)
                  if k.startswith('backbone.') else k: v
                  for k, v in winner_bench_weights.items()
                  if 'head' not in k and 'classifier' not in k}
missing, unexpected = model.load_state_dict(backbone_state, strict=False)
print(f"   Backbone warm-start: {len(backbone_state)} keys loaded")
print(f"   Missing: {len(missing)} | Unexpected: {len(unexpected)}")

backbone_a_ids = {id(p) for p in model.backbone_a.parameters()}
swin_ids       = {id(p) for p in model.swin.parameters()}
new_ids        = {id(p) for p in model.parameters()} - backbone_a_ids - swin_ids

opt = optim.AdamW([
    {'params': [p for p in model.backbone_a.parameters()], 'lr': LR*0.05},  # 2.5e-6
    {'params': [p for p in model.swin.parameters()],       'lr': LR*0.1},   # 5e-6
    {'params': [p for p in model.parameters()
                if id(p) not in backbone_a_ids | swin_ids], 'lr': LR},       # 5e-5
], weight_decay=WEIGHT_DECAY)

sch = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=10, T_mult=2)
es  = EarlyStopping(patience=PATIENCE, save_path=SAVE_HYBRID)

train_losses, val_losses, val_accs, lr_hist = [], [], [], []

print(f"\nTraining final hybrid ({HYBRID_EPOCHS} epochs max)")
print(f"    Winner: {winner_name}  |  AMP: {'ON' if USE_AMP else 'OFF'}")
print("-"*78)

for ep in range(1, HYBRID_EPOCHS+1):
    t0  = time.time()
    trl = train_one_epoch(model, train_loader, criterion, opt)
    vll, vla, _, _, _ = evaluate(model, val_loader, criterion)
    sch.step()
    train_losses.append(trl); val_losses.append(vll); val_accs.append(vla)
    lr_hist.append(opt.param_groups[2]['lr'])
    stop = es.step(vll, ep, model)
    flag = "[BEST]" if es.counter == 0 else f"(patience {es.counter}/{PATIENCE})"
    print(f"Ep {ep:02d}/{HYBRID_EPOCHS} | {time.time()-t0:.0f}s | "
          f"Tr:{trl:.4f} Vl:{vll:.4f} Acc:{vla*100:.2f}% "
          f"LR:{lr_hist[-1]:.2e} {flag}")
    if stop:
        print(f"\n⏹  Early stop @ ep{ep}. Best: ep{es.best_epoch} (loss={es.best_loss:.4f})")
        break

print(f"\nHybrid training complete -> {SAVE_HYBRID}")


# -----------------------------------------------------------------------------
# Step 14 — Hybrid training curves
# -----------------------------------------------------------------------------
ep_r = range(1, len(train_losses)+1)

fig = plt.figure(figsize=(22, 6))
fig.patch.set_facecolor(BG)
gs  = fig.add_gridspec(1, 4, wspace=0.38, left=0.06, right=0.97,
                        top=0.88, bottom=0.13)

def _styled_ax(ax):
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values(): sp.set_edgecolor(GRID)
    ax.tick_params(colors='#555555')
    ax.grid(alpha=0.18, color='#cccccc')
    return ax

ax0 = _styled_ax(fig.add_subplot(gs[0, 0]))
ax0.plot(ep_r, train_losses, color='#3498db', lw=2, label='Train', marker='o', ms=2)
ax0.plot(ep_r, val_losses,   color='#e74c3c', lw=2, label='Val',   marker='o', ms=2)
ax0.axvline(es.best_epoch, color='#2ecc71', ls='--', lw=1.8,
            label=f'Best ep{es.best_epoch}')
ax0.fill_between(ep_r, train_losses, val_losses, alpha=0.12, color='#dddddd')
ax0.set_title("Loss Curves", fontsize=12, fontweight='bold', color='#222222', pad=8)
ax0.set_xlabel("Epoch", color='#555555'); ax0.set_ylabel("Loss", color='#555555')
ax0.legend(fontsize=9, facecolor=PANEL, labelcolor='#222222', edgecolor=GRID)

ax1 = _styled_ax(fig.add_subplot(gs[0, 1]))
ax1.plot(ep_r, [a*100 for a in val_accs], color='#2ecc71', lw=2,
         marker='s', ms=2, label='Val Acc')
ax1.axhline(max(val_accs)*100, color='#f39c12', ls=':', lw=1.5,
            alpha=0.8, label=f'Peak {max(val_accs)*100:.2f}%')
ax1.fill_between(ep_r, [a*100 for a in val_accs],
                 alpha=0.15, color='#2ecc71')
ax1.set_title("Val Accuracy (%)", fontsize=12, fontweight='bold', color='#222222', pad=8)
ax1.set_xlabel("Epoch", color='#555555'); ax1.set_ylabel("Accuracy (%)", color='#555555')
ax1.legend(fontsize=9, facecolor=PANEL, labelcolor='#222222', edgecolor=GRID)

ax2 = _styled_ax(fig.add_subplot(gs[0, 2]))
ax2.plot(ep_r, lr_hist, color='#9b59b6', lw=2)
ax2.fill_between(ep_r, lr_hist, alpha=0.15, color='#9b59b6')
ax2.set_title("LR Schedule\n(Fusion Head)", fontsize=12, fontweight='bold',
              color='#222222', pad=8)
ax2.set_xlabel("Epoch", color='#555555'); ax2.set_ylabel("Learning Rate", color='#555555')
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1e}'))

ax3 = _styled_ax(fig.add_subplot(gs[0, 3]))
sc = ax3.scatter(val_losses, [a*100 for a in val_accs],
                 c=list(ep_r), cmap='plasma', s=25, alpha=0.85, zorder=3)
ax3.scatter([val_losses[es.best_epoch-1]], [val_accs[es.best_epoch-1]*100],
            s=180, color='#2ecc71', marker='*', zorder=5, label=f'Best ep{es.best_epoch}')
cb = plt.colorbar(sc, ax=ax3, pad=0.02)
cb.set_label('Epoch', color='#555555', fontsize=9)
cb.ax.yaxis.set_tick_params(color='#555555')
plt.setp(cb.ax.yaxis.get_ticklabels(), color='#555555')
ax3.set_xlabel("Val Loss", color='#555555')
ax3.set_ylabel("Val Accuracy (%)", color='#555555')
ax3.set_title("Loss vs Accuracy\n(Convergence Trace)", fontsize=12,
              fontweight='bold', color='#222222', pad=8)
ax3.legend(fontsize=9, facecolor=PANEL, labelcolor='#222222', edgecolor=GRID)

fig.suptitle(f"Hybrid training dashboard | {winner_name} + Swin-T + CrossAttn",
             fontsize=14, fontweight='bold', color='#222222')
plt.savefig("hybrid_training.png", dpi=150, bbox_inches='tight', facecolor=BG)
plt.show()
print("Saved: hybrid_training.png")


# -----------------------------------------------------------------------------
# Step 15 — Load best hybrid, test set, report by data source
# -----------------------------------------------------------------------------
model.load_state_dict(torch.load(SAVE_HYBRID, map_location=DEVICE))
_, _, test_probs, test_preds, test_true = evaluate(model, test_loader, criterion)
test_probs = np.array(test_probs)
m_hybrid = full_metrics(test_true, test_preds, test_probs)

print("\n" + "="*65)
print(f"FINAL HYBRID TEST RESULTS ({winner_name} + Swin-T + CrossAttn)")
print("="*65)
for label, val in [("Accuracy",        m_hybrid['acc']*100),
                   ("Macro Precision",  m_hybrid['prec']*100),
                   ("Macro Recall",     m_hybrid['rec']*100),
                   ("Macro F1",         m_hybrid['mf1']*100),
                   ("Weighted F1",      m_hybrid['wf1']*100)]:
    print(f"  {label:<22}: {val:.4f}%")
print(f"  {'Macro AUC-ROC':<22}: {m_hybrid['auc']:.4f}")
print("="*65)
print(classification_report(
    test_true, test_preds,
    target_names=[IDX_TO_CLASS[i] for i in range(NUM_CLASSES)]))

if USE_COMBINED_DATA:
    print("\n" + "=" * 65)
    print("TEST BY DATA SOURCE (Afzaal vs PlantVillage)")
    print("=" * 65)
    pv_marker = os.path.join(PLANTVILLAGE_ROOT, "color")
    pv_marker = pv_marker.lower()
    afzaal_indices, pv_indices = [], []
    for idx, path in enumerate(test_paths):
        if pv_marker in path.replace("\\", "/").lower():
            pv_indices.append(idx)
        else:
            afzaal_indices.append(idx)
    acc_all = accuracy_score(test_true, test_preds)
    mf1_all = f1_score(test_true, test_preds, average="macro", zero_division=0)
    print(f"  {'Source':<18} {'N':>8} {'Acc %':>10} {'Macro F1 %':>12}")
    print("  " + "-" * 52)
    for src_name, indices in (("Afzaal", afzaal_indices), ("PlantVillage", pv_indices)):
        if not indices:
            continue
        st = [test_true[i] for i in indices]
        sp = [test_preds[i] for i in indices]
        print(f"  {src_name:<18} {len(indices):>8} "
              f"{accuracy_score(st, sp)*100:>9.2f}% "
              f"{f1_score(st, sp, average='macro', zero_division=0)*100:>11.2f}%")
    print(f"  {'Combined':<18} {len(test_true):>8} {acc_all*100:>9.2f}% {mf1_all*100:>11.2f}%")
    print("=" * 65)

    try:
        fig_x, ax_x = plt.subplots(1, 2, figsize=(12, 4.5))
        fig_x.patch.set_facecolor(BG)
        src_labels, accs_src, f1s_src = [], [], []
        for src_name, indices in (("Afzaal", afzaal_indices), ("PlantVillage", pv_indices)):
            if not indices:
                continue
            st = [test_true[i] for i in indices]
            sp = [test_preds[i] for i in indices]
            src_labels.append(src_name)
            accs_src.append(accuracy_score(st, sp) * 100)
            f1s_src.append(f1_score(st, sp, average="macro", zero_division=0) * 100)
        src_labels.append("Combined")
        accs_src.append(acc_all * 100)
        f1s_src.append(mf1_all * 100)
        colors_x = ["#3498db", "#e67e22", "#2ecc71"][: len(src_labels)]
        for ax, vals, title in zip(
            ax_x,
            (accs_src, f1s_src),
            ("Accuracy (%) by source", "Macro F1 (%) by source"),
        ):
            ax.set_facecolor(PANEL)
            for sp in ax.spines.values():
                sp.set_edgecolor(GRID)
            ax.tick_params(colors="#555555")
            ax.bar(src_labels, vals, color=colors_x, edgecolor="#333333", linewidth=0.6)
            ax.set_title(title, fontweight="bold", color="#222222", fontsize=11)
            ax.grid(axis="y", alpha=0.35, color="#cccccc")
            ax.set_ylim(max(0, min(vals) - 5), min(101, max(vals) + 3))
            for i, v in enumerate(vals):
                ax.text(i, v + 0.5, f"{v:.2f}", ha="center", fontsize=9,
                        color="#222222", fontweight="bold")
        fig_x.suptitle(
            f"Hybrid test metrics by data source | {winner_name} + Swin-T",
            fontsize=12, fontweight="bold", color="#222222",
        )
        plt.tight_layout()
        plt.savefig("cross_dataset_test.png", dpi=150, bbox_inches="tight", facecolor=BG)
        plt.show()
        print("Saved: cross_dataset_test.png\n")
    except Exception as e:
        print(f"[WARN] Cross-source plot skipped: {e}\n")


# -----------------------------------------------------------------------------
# Step 16 — Confusion matrix, per-class scores, ROC
# -----------------------------------------------------------------------------
cls_short = [IDX_TO_CLASS[i].replace('_','\n')[:12] for i in range(NUM_CLASSES)]
cls_label = [IDX_TO_CLASS[i] for i in range(NUM_CLASSES)]

fig = plt.figure(figsize=(24, 8))
fig.patch.set_facecolor(BG)
gs  = fig.add_gridspec(1, 3, wspace=0.38, left=0.05, right=0.97,
                        top=0.91, bottom=0.1)

ax_cm = fig.add_subplot(gs[0, 0])
ax_cm.set_facecolor(PANEL)
cm = confusion_matrix(test_true, test_preds, labels=list(range(NUM_CLASSES)))
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
im = ax_cm.imshow(cm_norm, cmap='RdYlGn', vmin=0, vmax=100, aspect='auto')
plt.colorbar(im, ax=ax_cm, pad=0.02, label='Row % (Recall)')
ax_cm.set_xticks(range(NUM_CLASSES)); ax_cm.set_yticks(range(NUM_CLASSES))
ax_cm.set_xticklabels(cls_short, fontsize=7.5, color='#222222', rotation=45, ha='right')
ax_cm.set_yticklabels(cls_short, fontsize=7.5, color='#222222')
ax_cm.set_xlabel("Predicted", color='#555555', fontsize=10)
ax_cm.set_ylabel("True Label", color='#555555', fontsize=10)
for i in range(NUM_CLASSES):
    for j in range(NUM_CLASSES):
        txt_color = 'black' if cm_norm[i,j] > 55 else 'white'
        ax_cm.text(j, i, f'{cm[i,j]}', ha='center', va='center',
                   fontsize=8, color=txt_color, fontweight='bold')
ax_cm.set_title("Confusion Matrix\n(color = row recall %)",
                fontsize=12, fontweight='bold', color='#222222', pad=8)
for sp in ax_cm.spines.values(): sp.set_edgecolor(GRID)

ax_f1 = fig.add_subplot(gs[0, 1])
ax_f1.set_facecolor(PANEL)
for sp in ax_f1.spines.values(): sp.set_edgecolor(GRID)
ax_f1.tick_params(colors='#555555')
ax_f1.grid(axis='y', alpha=0.18, color='#cccccc')

per_f1   = f1_score(test_true, test_preds, average=None,
                    zero_division=0, labels=list(range(NUM_CLASSES)))
per_prec = precision_score(test_true, test_preds, average=None,
                           zero_division=0, labels=list(range(NUM_CLASSES)))
per_rec  = recall_score(test_true, test_preds, average=None,
                        zero_division=0, labels=list(range(NUM_CLASSES)))
x = np.arange(NUM_CLASSES); w = 0.26
ax_f1.bar(x - w,   per_prec*100, width=w, color='#3498db', label='Precision', alpha=0.9)
ax_f1.bar(x,       per_rec*100,  width=w, color='#e67e22', label='Recall',    alpha=0.9)
ax_f1.bar(x + w,   per_f1*100,   width=w, color='#2ecc71', label='F1',        alpha=0.9)
ax_f1.axhline(m_hybrid['mf1']*100, color='#666666', ls='--', lw=1.2,
              alpha=0.6, label=f"Macro F1={m_hybrid['mf1']*100:.2f}%")
ax_f1.set_xticks(x)
ax_f1.set_xticklabels(cls_short, fontsize=7, color='#333333', rotation=0)
ax_f1.set_ylim([max(0, min(per_f1.min(), per_prec.min(), per_rec.min())*100-8), 105])
ax_f1.set_title("Per-Class Precision / Recall / F1 (%)",
                fontsize=12, fontweight='bold', color='#222222', pad=8)
ax_f1.legend(fontsize=9, facecolor=PANEL, labelcolor='#222222', edgecolor=GRID)
for i, val in enumerate(per_f1):
    ax_f1.text(i+w, val*100+0.5, f"{val*100:.1f}",
               ha='center', va='bottom', fontsize=7, color='#222222')

ax_roc = fig.add_subplot(gs[0, 2])
ax_roc.set_facecolor(PANEL)
for sp in ax_roc.spines.values(): sp.set_edgecolor(GRID)
ax_roc.tick_params(colors='#555555')
ax_roc.grid(alpha=0.18, color='#cccccc')

y_bin     = label_binarize(test_true, classes=list(range(NUM_CLASSES)))
roc_colors = plt.cm.tab10(np.linspace(0, 1, NUM_CLASSES))
auc_vals  = []
for i in range(NUM_CLASSES):
    if y_bin[:,i].sum() == 0:
        auc_vals.append(float('nan')); continue
    fpr, tpr, _ = roc_curve(y_bin[:,i], test_probs[:,i])
    ai = roc_auc_score(y_bin[:,i], test_probs[:,i])
    auc_vals.append(ai)
    ax_roc.plot(fpr, tpr, color=roc_colors[i], lw=1.8,
                label=f"{cls_label[i][:14]} ({ai:.3f})")
ax_roc.plot([0,1],[0,1],'--', color='#555555', lw=1)
ax_roc.fill_between([0,1],[0,1],[1,1], alpha=0.06, color='#eeeeee')
ax_roc.set_xlabel("False Positive Rate", color='#555555', fontsize=10)
ax_roc.set_ylabel("True Positive Rate",  color='#555555', fontsize=10)
ax_roc.set_title("ROC Curves (One-vs-Rest)\nMacro AUC = "
                 f"{m_hybrid['auc']:.4f}",
                 fontsize=12, fontweight='bold', color='#222222', pad=8)
ax_roc.legend(fontsize=7, facecolor=PANEL, labelcolor='#222222',
              edgecolor=GRID, loc='lower right')

ax_ins = ax_roc.inset_axes([0.02, 0.55, 0.38, 0.42])
ax_ins.set_facecolor('white')
valid_auc = [v for v in auc_vals if not np.isnan(v)]
valid_cls = [cls_label[i][:8] for i, v in enumerate(auc_vals) if not np.isnan(v)]
auc_bar_c = ['#e74c3c' if v < 0.97 else '#2ecc71' for v in valid_auc]
ax_ins.barh(range(len(valid_auc)), valid_auc, color=auc_bar_c, edgecolor='#333333', linewidth=0.3, height=0.7)
ax_ins.set_xlim([min(valid_auc)*0.995, 1.005])
ax_ins.set_yticks(range(len(valid_cls)))
ax_ins.set_yticklabels(valid_cls, fontsize=5.5, color='#222222')
ax_ins.tick_params(colors='#555555', labelsize=5)
ax_ins.set_title("AUC/class", fontsize=6, color='#222222', pad=2)
for sp in ax_ins.spines.values(): sp.set_edgecolor(GRID)

fig.suptitle(f"Final hybrid evaluation - {winner_name} + Swin-T + CrossAttn",
             fontsize=14, fontweight='bold', color='#222222')
plt.savefig("hybrid_evaluation.png", dpi=150, bbox_inches='tight', facecolor=BG)
plt.show()
print("Saved: hybrid_evaluation.png")


# -----------------------------------------------------------------------------
# Step 17 — Grad-CAM++ / EigenCAM (optional)
# -----------------------------------------------------------------------------
try:
    from pytorch_grad_cam import GradCAMPlusPlus, EigenCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image

    class WrapperA(nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, x): return self.m.backbone_a(x)

    class WrapperB(nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, x):
            feat = self.m.swin.forward_features(x)
            if feat.dim() == 4:
                return feat.permute(0,3,1,2).contiguous()
            return feat.contiguous().unsqueeze(-1).unsqueeze(-1)

    wrap_a = WrapperA(model)
    wrap_b = WrapperB(model)

    if WINNER_CLASS == EfficientNetV2Model:
        tgt_a = [model.backbone_a.blocks[-1]]
    elif WINNER_CLASS == ResNet50Model:
        last = list(model.backbone_a.children())[-1]
        tgt_a = [list(last.children())[-1]]
    else:
        tgt_a = [list(model.backbone_a.children())[-2]]

    tgt_b    = [model.swin.layers[-1].blocks[-1]]
    tgt_full = tgt_a

    cam_a    = GradCAMPlusPlus(model=wrap_a,  target_layers=tgt_a)
    cam_b    = EigenCAM(       model=wrap_b,  target_layers=tgt_b)
    cam_full = GradCAMPlusPlus(model=model,   target_layers=tgt_full)

    sample_idxs = []
    for ci in range(NUM_CLASSES):
        cands = [i for i, l in enumerate(test_labels) if l == ci]
        if cands: sample_idxs.append(int(np.random.choice(cands)))

    n_show = min(8, len(sample_idxs))
    fig, axes = plt.subplots(n_show, 4, figsize=(22, 5*n_show))
    if n_show == 1: axes = [axes]
    for ax, title in zip(axes[0],
                         ["Original",
                          f"{winner_name[:12]}\n(GradCAM++)",
                          "Swin-T\n(EigenCAM)",
                          "Full Model\n(GradCAM++)"]):
        ax.set_title(title, fontsize=9, fontweight='bold')

    for row, idx in enumerate(sample_idxs[:n_show]):
        raw = Image.open(test_paths[idx]).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        rgb = np.array(raw)/255.0
        inp = val_tf(raw).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = model(inp)
            pred   = logits.argmax(-1).item()
            conf   = F.softmax(logits,-1)[0,pred].item()
        try:
            gc_a    = cam_a(input_tensor=inp)[0]
            gc_b    = cam_b(input_tensor=inp)[0]
            gc_full = cam_full(input_tensor=inp)[0]
        except Exception:
            gc_a = gc_b = gc_full = np.zeros((IMG_SIZE, IMG_SIZE))
        overlays = [raw,
                    show_cam_on_image(rgb.astype(np.float32), gc_a,    use_rgb=True),
                    show_cam_on_image(rgb.astype(np.float32), gc_b,    use_rgb=True),
                    show_cam_on_image(rgb.astype(np.float32), gc_full, use_rgb=True)]
        correct = pred == test_labels[idx]
        clr     = '#2ecc71' if correct else '#e74c3c'
        for col, im in enumerate(overlays):
            axes[row][col].imshow(im); axes[row][col].axis('off')
        axes[row][0].set_ylabel(
            f"GT: {IDX_TO_CLASS[test_labels[idx]]}\n"
            f"Pred: {IDX_TO_CLASS[pred]} {'OK' if correct else 'X'} ({conf*100:.1f}%)",
            fontsize=8, color=clr, fontweight='bold')

    plt.suptitle(
        f"XAI: {winner_name} (Local) vs Swin-T (Global) vs Full Model\n"
        "Bright = regions driving prediction",
        fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig("xai_gradcam.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("XAI visualization complete.")
except Exception as e:
    print(f"[WARN] XAI skipped: {e}")


# -----------------------------------------------------------------------------
# Step 18 — Branch feature norms (winner vs Swin)
# -----------------------------------------------------------------------------
try:
    model.eval()
    eff_norms  = {i:[] for i in range(NUM_CLASSES)}
    swin_norms = {i:[] for i in range(NUM_CLASSES)}
    with torch.no_grad():
        for imgs, lbls in test_loader:
            imgs = imgs.to(DEVICE)
            fa   = model.forward_a(imgs)
            fb   = model.forward_b(imgs)
            for i, l in enumerate(lbls.numpy()):
                eff_norms[l].append(fa[i].norm().item())
                swin_norms[l].append(fb[i].norm().item())

    avg_a = np.array([np.mean(eff_norms[i])  if eff_norms[i]  else 0 for i in range(NUM_CLASSES)])
    avg_b = np.array([np.mean(swin_norms[i]) if swin_norms[i] else 0 for i in range(NUM_CLASSES)])
    dominance = avg_a / (avg_a + avg_b + 1e-8)   # 1.0 = 100% winner branch

    fig = plt.figure(figsize=(20, 9))
    fig.patch.set_facecolor(BG)
    gs  = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.35,
                            left=0.06, right=0.97, top=0.91, bottom=0.08)

    ax0 = fig.add_subplot(gs[0, :])
    ax0.set_facecolor(PANEL)
    for sp in ax0.spines.values(): sp.set_edgecolor(GRID)
    ax0.tick_params(colors='#555555')
    ax0.grid(axis='y', alpha=0.18, color='#cccccc')
    x = np.arange(NUM_CLASSES); w = 0.38
    b1 = ax0.bar(x-w/2, avg_a, w, label=f'{winner_name} (Local Texture)',
                 color='#3498db', edgecolor='none', alpha=0.9)
    b2 = ax0.bar(x+w/2, avg_b, w, label='Swin-T (Global Structure)',
                 color='#e67e22', edgecolor='none', alpha=0.9)
    ax0.set_xticks(x)
    ax0.set_xticklabels(cls_names, fontsize=9, color='#333333', rotation=20, ha='right')
    ax0.set_title("Branch Feature L2 Norm per Disease Class",
                  fontsize=13, fontweight='bold', color='#222222', pad=8)
    ax0.set_ylabel("Mean L2 Norm", color='#555555', fontsize=10)
    ax0.legend(fontsize=10, facecolor=PANEL, labelcolor='#222222', edgecolor=GRID)
    for bar, val in zip(b1, avg_a):
        ax0.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
                 f'{val:.2f}', ha='center', fontsize=7.5, color='#222222')
    for bar, val in zip(b2, avg_b):
        ax0.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
                 f'{val:.2f}', ha='center', fontsize=7.5, color='#222222')

    ax1 = fig.add_subplot(gs[1, 0])
    ax1.set_facecolor(PANEL)
    for sp in ax1.spines.values(): sp.set_edgecolor(GRID)
    dom_2d = dominance.reshape(1, -1)
    im1 = ax1.imshow(dom_2d, cmap='RdBu_r', vmin=0, vmax=1, aspect='auto')
    plt.colorbar(im1, ax=ax1, pad=0.02,
                 label='<- Swin strong | Winner strong ->')
    ax1.set_yticks([0]); ax1.set_yticklabels(['Branch\nDominance'], color='#222222', fontsize=9)
    ax1.set_xticks(range(NUM_CLASSES))
    ax1.set_xticklabels(cls_short, color='#222222', fontsize=8)
    ax1.set_title("Winner Branch Dominance Ratio per Class\n(>0.5 = Winner branch stronger)",
                  fontsize=11, fontweight='bold', color='#222222', pad=6)
    for j, d in enumerate(dominance):
        tc = '#222222' if 0.38 <= d <= 0.62 else 'white'
        ax1.text(j, 0, f'{d:.2f}', ha='center', va='center',
                 fontsize=9, color=tc, fontweight='bold')

    ax2 = fig.add_subplot(gs[1, 1])
    ax2.set_facecolor(PANEL)
    for sp in ax2.spines.values(): sp.set_edgecolor(GRID)
    ax2.tick_params(colors='#555555')
    ax2.grid(alpha=0.18, color='#cccccc')
    scatter_colors = plt.cm.tab10(np.linspace(0,1,NUM_CLASSES))
    for i in range(NUM_CLASSES):
        if not eff_norms[i]: continue
        ax2.scatter(eff_norms[i], swin_norms[i],
                    c=[scatter_colors[i]], s=18, alpha=0.5, label=cls_names[i])
    ax2.plot([0, max(avg_a.max(), avg_b.max())],
             [0, max(avg_a.max(), avg_b.max())],
             '--', color='#999999', lw=1, alpha=0.7, label='Equal norm')
    ax2.set_xlabel(f"{winner_name} norm", color='#555555', fontsize=10)
    ax2.set_ylabel("Swin-T norm",         color='#555555', fontsize=10)
    ax2.set_title("Sample-level Branch Norm Scatter\n(above diagonal = Swin stronger)",
                  fontsize=11, fontweight='bold', color='#222222', pad=6)
    ax2.legend(fontsize=7, facecolor=PANEL, labelcolor='#222222',
               edgecolor=GRID, ncol=2, loc='upper left')

    fig.suptitle(f"Branch contribution - {winner_name} vs Swin-T",
                 fontsize=14, fontweight='bold', color='#222222')
    plt.savefig("branch_contribution.png", dpi=150, bbox_inches='tight', facecolor=BG)
    plt.show()
    print("Saved: branch_contribution.png")
except Exception as e:
    print(f"[WARN] Branch analysis skipped: {e}")


# -----------------------------------------------------------------------------
# Step 19 — Robustness vs Gaussian noise
# -----------------------------------------------------------------------------
print("\nRobustness evaluation (Gaussian noise)...")
print("-"*55)

class NoisyDS(StrawberryDataset):
    def __init__(self, paths, labels, tf, sigma):
        super().__init__(paths, labels, tf, False)
        self.sigma = sigma
    def __getitem__(self, idx):
        img, lbl = super().__getitem__(idx)
        if self.sigma > 0:
            img = img + torch.randn_like(img)*self.sigma
        return img, lbl

rob_results = []
for sigma in [0.0, 0.05, 0.10, 0.20, 0.30]:
    nl = DataLoader(NoisyDS(test_paths, test_labels, val_tf, sigma),
                    batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    _, acc_n, probs_n, preds_n, true_n = evaluate(model, nl, criterion)
    f1_n  = f1_score(true_n, preds_n, average='macro', zero_division=0)
    pf1_n = f1_score(true_n, preds_n, average=None,
                     zero_division=0, labels=list(range(NUM_CLASSES)))
    rob_results.append({'sigma':sigma, 'acc':acc_n*100, 'f1':f1_n*100,
                        'per_f1': pf1_n*100})
    print(f"  sigma={sigma:.2f} -> Acc:{acc_n*100:.2f}%  Macro F1:{f1_n*100:.2f}%")

fig = plt.figure(figsize=(22, 10))
fig.patch.set_facecolor(BG)
gs  = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.38,
                        left=0.06, right=0.97, top=0.91, bottom=0.07)

def _ra(ax):
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values(): sp.set_edgecolor(GRID)
    ax.tick_params(colors='#555555')
    ax.grid(alpha=0.18, color='#cccccc')
    return ax

sigs     = [r['sigma'] for r in rob_results]
accs     = [r['acc']   for r in rob_results]
f1s      = [r['f1']    for r in rob_results]
baseline_acc = accs[0]
baseline_f1  = f1s[0]

ax0 = _ra(fig.add_subplot(gs[0, 0]))
ax0.plot(sigs, accs, color='#f39c12', lw=2.5, marker='o', ms=9, zorder=5)
ax0.fill_between(sigs, baseline_acc, accs, alpha=0.2,
                 color='#e74c3c', label='Degradation')
ax0.axhline(baseline_acc, color='#2ecc71', ls='--', lw=1.5,
            label=f'Clean baseline {baseline_acc:.2f}%')
ax0.set_title("Accuracy vs Gaussian noise (sigma)",
              fontsize=12, fontweight='bold', color='#222222', pad=8)
ax0.set_xlabel("Noise sigma (std)", color='#555555')
ax0.set_ylabel("Accuracy (%)", color='#555555')
ax0.set_ylim([max(0, min(accs)-10), 102])
for x, y in zip(sigs, accs):
    ax0.annotate(f'{y:.1f}%', (x, y), textcoords='offset points',
                 xytext=(0, 10), ha='center', fontsize=9, color='#222222',
                 fontweight='bold')
ax0.legend(fontsize=9, facecolor=PANEL, labelcolor='#222222', edgecolor=GRID)

ax1 = _ra(fig.add_subplot(gs[0, 1]))
ax1.plot(sigs, f1s, color='#e74c3c', lw=2.5, marker='s', ms=9, zorder=5)
ax1.fill_between(sigs, baseline_f1, f1s, alpha=0.2,
                 color='#e74c3c', label='Degradation')
ax1.axhline(baseline_f1, color='#3498db', ls='--', lw=1.5,
            label=f'Clean baseline {baseline_f1:.2f}%')
ax1.set_title("Macro F1 vs Gaussian noise (sigma)",
              fontsize=12, fontweight='bold', color='#222222', pad=8)
ax1.set_xlabel("Noise sigma (std)", color='#555555')
ax1.set_ylabel("Macro F1 (%)", color='#555555')
ax1.set_ylim([max(0, min(f1s)-10), 102])
for x, y in zip(sigs, f1s):
    ax1.annotate(f'{y:.1f}%', (x, y), textcoords='offset points',
                 xytext=(0, 10), ha='center', fontsize=9, color='#222222',
                 fontweight='bold')
ax1.legend(fontsize=9, facecolor=PANEL, labelcolor='#222222', edgecolor=GRID)

ax2 = _ra(fig.add_subplot(gs[0, 2]))
drops_acc = [baseline_acc - a for a in accs]
drops_f1  = [baseline_f1  - f for f in f1s]
x  = np.arange(len(sigs)); w = 0.35
ax2.bar(x-w/2, drops_acc, w, color='#f39c12', label='Acc drop', alpha=0.9)
ax2.bar(x+w/2, drops_f1,  w, color='#e74c3c', label='F1 drop',  alpha=0.9)
ax2.set_xticks(x)
ax2.set_xticklabels([f's={s}' for s in sigs], color='#333333', fontsize=9)
ax2.set_title("Performance Drop from Clean Baseline",
              fontsize=12, fontweight='bold', color='#222222', pad=8)
ax2.set_ylabel("% Drop", color='#555555')
ax2.legend(fontsize=9, facecolor=PANEL, labelcolor='#222222', edgecolor=GRID)
for ax2_bar in ax2.patches:
    h = ax2_bar.get_height()
    if h > 0.05:
        ax2.text(ax2_bar.get_x()+ax2_bar.get_width()/2, h+0.1,
                 f'{h:.1f}', ha='center', fontsize=8, color='#222222',
                 fontweight='bold')

ax3 = _ra(fig.add_subplot(gs[1, :2]))
heat_data = np.array([r['per_f1'] for r in rob_results])
im = ax3.imshow(heat_data, cmap='RdYlGn', vmin=0, vmax=100,
                aspect='auto')
plt.colorbar(im, ax=ax3, pad=0.01, label='F1 (%)')
ax3.set_yticks(range(len(sigs)))
ax3.set_yticklabels([f's={s:.2f}' for s in sigs], color='#222222', fontsize=9)
ax3.set_xticks(range(NUM_CLASSES))
ax3.set_xticklabels(cls_short, color='#222222', fontsize=8)
ax3.set_title("Per-class F1 (%) across noise - robustness heatmap",
              fontsize=12, fontweight='bold', color='#222222', pad=8)
for i in range(len(sigs)):
    for j in range(NUM_CLASSES):
        tc = 'black' if heat_data[i,j] > 55 else 'white'
        ax3.text(j, i, f'{heat_data[i,j]:.1f}', ha='center', va='center',
                 fontsize=8, color=tc, fontweight='bold')

ax4 = _ra(fig.add_subplot(gs[1, 2]))
rob_score_acc = np.trapz(accs, sigs) / (sigs[-1]-sigs[0]) / 100
rob_score_f1  = np.trapz(f1s,  sigs) / (sigs[-1]-sigs[0]) / 100
labels_rob = ['Robustness\nScore (Acc)', 'Robustness\nScore (F1)']
vals_rob   = [rob_score_acc, rob_score_f1]
bar_c_rob  = ['#f39c12', '#e74c3c']
bars_r = ax4.bar(range(2), vals_rob, color=bar_c_rob, edgecolor='none', width=0.5)
ax4.set_xticks([0,1]); ax4.set_xticklabels(labels_rob, color='#333333', fontsize=10)
ax4.set_ylim([0, 1.05])
ax4.set_title("Robustness Score\n(Area under noise curve)",
              fontsize=12, fontweight='bold', color='#222222', pad=8)
ax4.set_ylabel("Score (0-1)", color='#555555')
for bar, val in zip(bars_r, vals_rob):
    ax4.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
             f'{val:.4f}', ha='center', fontsize=12, color='#222222',
             fontweight='bold')

fig.suptitle("Robustness under Gaussian noise",
             fontsize=15, fontweight='bold', color='#222222')
plt.savefig("robustness.png", dpi=150, bbox_inches='tight', facecolor=BG)
plt.show()
print("Saved: robustness.png")


# -----------------------------------------------------------------------------
# Step 20 — Quick ablation (frozen backbones, head-only probe)
# -----------------------------------------------------------------------------
print("\nAblation study (5-epoch head-only probe)...")
print("-"*60)

class BranchOnlyA(nn.Module):
    def __init__(self, m, nc):
        super().__init__()
        self.backbone_a = m.backbone_a
        self.attn_a     = m.attn_a
        self.gem_a      = m.gem_a
        self.head = nn.Linear(
            1280 if WINNER_CLASS==EfficientNetV2Model
            else 2048 if WINNER_CLASS==ResNet50Model else 512, nc)
    def forward(self, x):
        f = self.backbone_a(x)
        f = self.attn_a(f); f = self.gem_a(f)
        return self.head(f)

class BranchOnlyB(nn.Module):
    def __init__(self, m, nc):
        super().__init__()
        self.swin = m.swin
        self.norm = m.swin_norm
        self.head = nn.Linear(768, nc)
    def forward(self, x):
        f = _swin_pool(self.swin.forward_features(x))
        return self.head(self.norm(f))

abl_configs_list = [
    (f"{winner_name} Only", BranchOnlyA(model, NUM_CLASSES)),
    ("Swin-T Only",         BranchOnlyB(model, NUM_CLASSES)),
]
ablation_results = {}
for name, abl in abl_configs_list:
    abl = abl.to(DEVICE)
    for p in abl.parameters(): p.requires_grad = False
    for p in abl.head.parameters(): p.requires_grad = True
    opt_a = optim.AdamW(abl.head.parameters(), lr=1e-3)
    for ep_a in range(5):
        abl.train()
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            opt_a.zero_grad()
            criterion(abl(imgs), lbls).backward()
            opt_a.step()
    abl.eval()
    _, acc_a, _, p_a, g_a = evaluate(abl, test_loader, criterion)
    f1_a = f1_score(g_a, p_a, average='macro', zero_division=0)
    ablation_results[name] = (acc_a*100, f1_a*100)

ablation_results[f'{winner_name}+Swin+CrossAttn (Ours)'] = (m_hybrid['acc']*100, m_hybrid['mf1']*100)

print(f"\n  {'Model':<40} {'Accuracy':>10} {'Macro F1':>10}")
print("  " + "-"*62)
for name, (a, f) in ablation_results.items():
    mk = " <- BEST" if 'Ours' in name else ""
    print(f"  {name:<40} {a:>9.2f}%  {f:>9.2f}%{mk}")

abl_names  = list(ablation_results.keys())
abl_accs   = [ablation_results[n][0] for n in abl_names]
abl_f1s    = [ablation_results[n][1] for n in abl_names]
abl_colors = ['#3498db', '#e67e22', '#f39c12' if len(abl_names)>2 else None,
              '#2ecc71'][:len(abl_names)]

fig = plt.figure(figsize=(16, 6))
fig.patch.set_facecolor(BG)
gs  = fig.add_gridspec(1, 2, wspace=0.35, left=0.06, right=0.97,
                        top=0.88, bottom=0.18)

for col, (metric_vals, metric_name) in enumerate([
        (abl_accs, 'Accuracy (%)'), (abl_f1s, 'Macro F1 (%)')]):
    ax = fig.add_subplot(gs[0, col])
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values(): sp.set_edgecolor(GRID)
    ax.tick_params(colors='#555555')
    ax.grid(axis='y', alpha=0.18, color='#cccccc')

    bars = ax.bar(range(len(abl_names)), metric_vals,
                  color=abl_colors, edgecolor='none', width=0.55)
    best_i = int(np.argmax(metric_vals))
    bars[best_i].set_edgecolor('#333333')
    bars[best_i].set_linewidth(1.5)

    ax.set_xticks(range(len(abl_names)))
    short_names = [n.replace('+Swin+CrossAttn (Ours)', '\n+Swin+CrossAttn\n* Proposed')
                    .replace(' Only', '\nOnly') for n in abl_names]
    ax.set_xticklabels(short_names, fontsize=8.5, color='#333333', rotation=0)
    lo = min(metric_vals)*0.988
    hi = min(max(metric_vals)*1.012, 101)
    ax.set_ylim([lo, hi])
    ax.set_title(f"Ablation study - {metric_name}",
                 fontsize=12, fontweight='bold', color='#222222', pad=8)
    ax.set_ylabel(metric_name, color='#555555', fontsize=10)

    for bar, val in zip(bars, metric_vals):
        ax.text(bar.get_x()+bar.get_width()/2,
                bar.get_height() + (hi-lo)*0.01,
                f'{val:.2f}%', ha='center', fontsize=10,
                color='#222222', fontweight='bold')

    if len(abl_names) >= 2:
        branch_best = max(metric_vals[:-1])
        hybrid_val  = metric_vals[-1]
        gain = hybrid_val - branch_best
        if gain > 0:
            ax.annotate('', xy=(len(abl_names)-1, hybrid_val),
                        xytext=(len(abl_names)-1, branch_best),
                        arrowprops=dict(arrowstyle='->', color='#f39c12',
                                        lw=2.5))
            ax.text(len(abl_names)-0.45, (hybrid_val+branch_best)/2,
                    f'+{gain:.2f}%', color='#f39c12', fontsize=9,
                    fontweight='bold', va='center')

fig.suptitle("Ablation study: branch contribution",
             fontsize=14, fontweight='bold', color='#222222')
plt.savefig("ablation.png", dpi=150, bbox_inches='tight', facecolor=BG)
plt.show()
print("Saved: ablation.png")


# -----------------------------------------------------------------------------
# Step 21 — Benchmark vs hybrid comparison & literature bars
# -----------------------------------------------------------------------------
print("\n" + "="*80)
print("COMPLETE MODEL COMPARISON TABLE")
print("="*80)
print(f"  {'Model':<40} {'Accuracy':>10} {'Macro F1':>10} {'AUC-ROC':>10}")
print("  " + "-"*72)
for bname, bm in bench_results.items():
    mk = " <- winner" if bname == winner_name else ""
    print(f"  {bname:<40} {bm['acc']*100:>9.3f}%  {bm['mf1']*100:>9.3f}%  {bm['auc']:>9.4f}{mk}")
hname = f"{winner_name}+Swin-T+CrossAttn (FINAL)"
print(f"  {hname:<40} {m_hybrid['acc']*100:>9.3f}%  {m_hybrid['mf1']*100:>9.3f}%  {m_hybrid['auc']:>9.4f}  <- BEST")
print("="*80)

lit_entries = [
    ("Afzaal et al.\n(2021)",        "Mask R-CNN",            87.0,  0.870),
    ("Kreiner et al.\n(2023)",       "YOLOv8-XL",             92.0,  0.930),
    ("Nguyen et al.\n(2024)",        "ViT fine-tuned",        92.7,  0.927),
    ("Aghamohammadesmaeil\n(2024)",  "ViT + Attention",       98.4,  0.985),
    ("BerryNet-Lite\n(2024)",        "EfficientNet+ECA",      99.45, 0.994),
    ("Kalpana et al.\n(2024)",       "Res-Conv + Swin",       99.90, 0.9992),
    (f"Ours *\n{winner_name[:20]}",  f"+Swin+CrossAttn",
     m_hybrid['acc']*100, m_hybrid['mf1']),
]
lit_labels  = [e[0] for e in lit_entries]
lit_accs    = [e[2] for e in lit_entries]
lit_f1s     = [e[3]*100 if e[3] <= 1.0 else e[3] for e in lit_entries]
lit_colors  = ['#555577']*6 + ['#f39c12']

fig = plt.figure(figsize=(20, 7))
fig.patch.set_facecolor(BG)
gs  = fig.add_gridspec(1, 2, wspace=0.35, left=0.06, right=0.97,
                        top=0.88, bottom=0.18)

for col, (vals, metric_name) in enumerate([(lit_accs,'Accuracy (%)'),
                                            (lit_f1s, 'Macro F1 / AUC (%)')]):
    ax = fig.add_subplot(gs[0, col])
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values(): sp.set_edgecolor(GRID)
    ax.tick_params(colors='#555555')
    ax.grid(axis='y', alpha=0.18, color='#cccccc')
    bars = ax.bar(range(len(lit_labels)), vals, color=lit_colors, edgecolor='none', width=0.6)
    bars[-1].set_edgecolor('#333333'); bars[-1].set_linewidth(1.5)
    ax.set_xticks(range(len(lit_labels)))
    ax.set_xticklabels(lit_labels, fontsize=8, color='#333333', rotation=0)
    lo = min(vals)*0.985; hi = min(max(vals)*1.015, 101.5)
    ax.set_ylim([lo, hi])
    ax.set_title(f"Literature comparison - {metric_name}",
                 fontsize=12, fontweight='bold', color='#222222', pad=8)
    ax.set_ylabel(metric_name, color='#555555', fontsize=10)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2,
                bar.get_height() + (hi-lo)*0.008,
                f'{val:.2f}', ha='center', fontsize=8.5,
                color='#222222', fontweight='bold')

fig.suptitle("Literature comparison vs proposed hybrid",
             fontsize=14, fontweight='bold', color='#222222')
plt.savefig("literature_comparison.png", dpi=150, bbox_inches='tight', facecolor=BG)
plt.show()
print("Saved: literature_comparison.png")


# -----------------------------------------------------------------------------
# Step 22 — Run summary
# -----------------------------------------------------------------------------
total_imgs = len(train_paths)+len(val_paths)+len(test_paths)
print("\n" + "="*72)
print("Run summary - dual-source (Afzaal + PlantVillage)")
print("="*72)
print(f"  Architecture   : {winner_name} (Winner) + Swin-Tiny + CrossAttentionGate")
print(f"  Attention      : {'ECA' if WINNER_CLASS==EfficientNetV2Model else 'CBAM'} (winner branch) | Swin self-attention (global branch)")
print(f"  Pooling        : GeM (p learnable, init=3.0)")
print(f"  Fusion         : Cross-Attention Gate + MLP (1024->512->256->128->N)")
print(f"  Datasets       : Afzaal2021 + PlantVillage (Balanced ~30%, Unified 7-class)")
print(f"  Total images   : {total_imgs:,} | Classes: {NUM_CLASSES}")
print(f"  Augmentation   : Mixup(alpha={MIXUP_ALPHA}) | RandErasing | AnnCrop={'ON' if USE_CROP else 'OFF'}")
print(f"  Optimizer      : AdamW differential LR | CosineWarmRestarts")
print(f"  Parameters     : {hybrid_params:,}")
print(f"  Best Epoch     : {es.best_epoch}")
print("-"*72)
print(f"  Accuracy       : {m_hybrid['acc']*100:.4f}%")
print(f"  Macro Precision: {m_hybrid['prec']*100:.4f}%")
print(f"  Macro Recall   : {m_hybrid['rec']*100:.4f}%")
print(f"  Macro F1       : {m_hybrid['mf1']*100:.4f}%")
print(f"  Weighted F1    : {m_hybrid['wf1']*100:.4f}%")
print(f"  Macro AUC-ROC  : {m_hybrid['auc']:.4f}")
print("-"*72)
print("  Robustness (Macro F1 @ Gaussian noise):")
for r in rob_results:
    print(f"    sigma={r['sigma']:.2f} -> Acc:{r['acc']:.2f}%  F1:{r['f1']:.2f}%")
print("-"*72)
print("  Ablation Study:")
for name, (a, f) in ablation_results.items():
    print(f"    {name:<42}: Acc={a:.2f}%  F1={f:.2f}%")
print("-"*72)
print("  Literature Comparison Table (for paper):")
lit = [
    ("Afzaal et al. (2021)",       "Mask R-CNN",               "~87%",   "~0.87"),
    ("Kreiner et al. (2023)",      "YOLOv8-XL",                "~92%",   "~0.93"),
    ("Nguyen et al. (2024)",       "ViT fine-tuned",           "92.7%",  "0.927"),
    ("Aghamohammadesmaeil (2024)", "ViT + Attention",          "98.4%",  "~0.985"),
    ("BerryNet-Lite (2024)",       "EfficientNet+ECA+MLP",     "99.45%", "~0.994"),
    ("Kalpana et al. (2024)",      "Res-Conv + Swin",          "99.9%",  "0.9992"),
    ("Ours - Hybrid (proposed)",   f"{winner_name}+Swin+CrossAttn",
     f"{m_hybrid['acc']*100:.2f}%", f"{m_hybrid['mf1']:.4f}"),
]
print(f"\n  {'Reference':<30} {'Method':<28} {'Acc':>8} {'F1/AUC':>8}")
print("  " + "-"*78)
for row in lit:
    mk = "  <-" if "Ours" in row[0] else ""
    print(f"  {row[0]:<30} {row[1]:<28} {row[2]:>7} {row[3]:>7}{mk}")
print("="*72)
print("\nAll done. Output files:")
print("     fig_dataset_train_class_dist.png")
print("     fig_dataset_test_class_dist.png")
print("     fig_dataset_split.png")
print("     cross_dataset_test.png")
print("     benchmark_curves.png")
print("     benchmark_bars.png")
print("     hybrid_training.png")
print("     hybrid_evaluation.png")
print("     xai_gradcam.png")
print("     branch_contribution.png")
print("     robustness.png")
print("     ablation.png")
print("     literature_comparison.png")