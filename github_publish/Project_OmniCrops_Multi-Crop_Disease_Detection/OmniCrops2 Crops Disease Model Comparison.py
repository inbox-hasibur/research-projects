 =================
# 🌿 OmniCrops — Multi-Crop Disease Detection
# 20 classes across 6 crops → PlantDoc + PlantVillage → 1000/200/200 per class
# Models: OmniNet-CS | OmniNet-EV | ResNet-50 | ViT-Base
 =================

import sys
_stale = ['UNIFIED_CLASSES','LABEL_MAP','IDX_TO_CLASS','NUM_CLASSES',
          'PLANTDOC_MAP','PLANTVILLAGE_MAP']
for _v in _stale:
    if _v in vars(): exec(f'del {_v}')

 
# STEP 1 — Imports & Config
 
import os, time, math, warnings, random, copy
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import matplotlib
matplotlib.rcParams.update({
    'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':12,
    'axes.labelsize':10,'xtick.labelsize':9,'ytick.labelsize':9,
    'figure.dpi':150,'savefig.dpi':300,'savefig.bbox':'tight',
    'axes.spines.top':False,'axes.spines.right':False,
})
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report)
from sklearn.preprocessing import label_binarize
import timm
warnings.filterwarnings('ignore')

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_cap    = torch.cuda.get_device_capability()[0] if torch.cuda.is_available() else 0
USE_AMP = _cap >= 7

try:
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)
except TypeError:
    # older PyTorch or CPU-only build
    try:    scaler = torch.amp.GradScaler(enabled=USE_AMP)
    except: scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

print(f"🚀 Device: {DEVICE}  |  AMP: {'ON' if USE_AMP else 'OFF'}")
if torch.cuda.is_available():
    print(f"   GPU : {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

OUT_ROOT = (Path("/kaggle/working/OmniCrops") if Path("/kaggle/working").exists()
            else Path.cwd() / "OmniCrops")
FIG_DIR  = OUT_ROOT / "figures"; CKPT_DIR = OUT_ROOT / "checkpoints"
for d in [OUT_ROOT, FIG_DIR, CKPT_DIR]: d.mkdir(parents=True, exist_ok=True)

PLANTDOC_TRAIN = "/kaggle/input/datasets/abdulhasibuddin/plant-doc-dataset/PlantDoc-Dataset/train"
PLANTDOC_TEST  = "/kaggle/input/datasets/abdulhasibuddin/plant-doc-dataset/PlantDoc-Dataset/test"
PLANTVILLAGE   = "/kaggle/input/datasets/abdallahalidev/plantvillage-dataset/color"

IMG_SIZE      = 224
BATCH_SIZE    = 32
BASE_EPOCHS   = 40
HYBRID_EPOCHS = 60
LR            = 3e-4
WEIGHT_DECAY  = 1e-4
LABEL_SMOOTH  = 0.1
PATIENCE      = 5
IMG_EXTS      = {'.jpg','.jpeg','.png','.bmp','.tif','.tiff'}
N_TRAIN, N_VAL, N_TEST = 1000, 200, 200
N_TOTAL = N_TRAIN + N_VAL + N_TEST  # 1400

PAL = {'omnics':'#2166ac','omniev':'#d6604d','resnet':'#4dac26','vit':'#e08214',
       'pos':'#2ecc71','neg':'#e74c3c'}
print(f"✅ Config | Budget: {N_TRAIN}/{N_VAL}/{N_TEST}/class | OUT: {OUT_ROOT}")


 
# STEP 2 — Label Map (33 classes, 9 crops)
 
_CLASSES_RAW = [
    "Apple___Scab","Apple___Black_Rot","Apple___Cedar_Rust","Apple___Healthy",
    "BellPepper___Bacterial_Spot","BellPepper___Healthy",
    "Cherry___Powdery_Mildew","Cherry___Healthy",
    "Corn___Gray_Leaf_Spot","Corn___Common_Rust","Corn___Northern_Leaf_Blight","Corn___Healthy",
    "Grape___Black_Rot","Grape___Esca_Black_Measles","Grape___Leaf_Blight","Grape___Healthy",
    "Peach___Bacterial_Spot","Peach___Healthy",
    "Potato___Early_Blight","Potato___Late_Blight","Potato___Healthy",
    "Strawberry___Leaf_Scorch","Strawberry___Healthy",
    "Tomato___Bacterial_Spot","Tomato___Early_Blight","Tomato___Late_Blight",
    "Tomato___Leaf_Mold","Tomato___Septoria_Leaf_Spot","Tomato___Spider_Mites",
    "Tomato___Target_Spot","Tomato___Mosaic_Virus","Tomato___Yellow_Leaf_Curl_Virus",
    "Tomato___Healthy",
]
UNIFIED_CLASSES = list(dict.fromkeys(_CLASSES_RAW))
NUM_CLASSES     = len(UNIFIED_CLASSES)
LABEL_MAP       = {cls: idx for idx, cls in enumerate(UNIFIED_CLASSES)}
IDX_TO_CLASS    = {idx: cls for cls, idx in LABEL_MAP.items()}

print(f"\n✅ Unified label map: {NUM_CLASSES} classes")
_cc = {}
for cls in UNIFIED_CLASSES:
    crop = cls.split('___')[0]; _cc[crop] = _cc.get(crop,0)+1
for crop, cnt in _cc.items(): print(f"   {crop:<14}: {cnt} classes")

PLANTDOC_MAP = {
    "Apple Scab Leaf":"Apple___Scab","Apple leaf":"Apple___Healthy",
    "Apple rust leaf":"Apple___Cedar_Rust","Bell_pepper leaf":"BellPepper___Healthy",
    "Bell_pepper leaf spot":"BellPepper___Bacterial_Spot","Cherry leaf":"Cherry___Healthy",
    "Corn Gray leaf spot":"Corn___Gray_Leaf_Spot","Corn leaf blight":"Corn___Northern_Leaf_Blight",
    "Corn rust leaf":"Corn___Common_Rust","grape leaf":"Grape___Healthy",
    "grape leaf black rot":"Grape___Black_Rot","Peach leaf":"Peach___Healthy",
    "Potato leaf early blight":"Potato___Early_Blight","Potato leaf late blight":"Potato___Late_Blight",
    "Strawberry leaf":"Strawberry___Healthy","Tomato Early blight leaf":"Tomato___Early_Blight",
    "Tomato Septoria leaf spot":"Tomato___Septoria_Leaf_Spot","Tomato leaf":"Tomato___Healthy",
    "Tomato leaf bacterial spot":"Tomato___Bacterial_Spot","Tomato leaf late blight":"Tomato___Late_Blight",
    "Tomato leaf mosaic virus":"Tomato___Mosaic_Virus","Tomato leaf yellow virus":"Tomato___Yellow_Leaf_Curl_Virus",
    "Tomato mold leaf":"Tomato___Leaf_Mold",
}
PLANTVILLAGE_MAP = {
    "Apple___Apple_scab":"Apple___Scab","Apple___Black_rot":"Apple___Black_Rot",
    "Apple___Cedar_apple_rust":"Apple___Cedar_Rust","Apple___healthy":"Apple___Healthy",
    "Pepper,_bell___Bacterial_spot":"BellPepper___Bacterial_Spot",
    "Pepper,_bell___healthy":"BellPepper___Healthy",
    "Cherry_(including_sour)___Powdery_mildew":"Cherry___Powdery_Mildew",
    "Cherry_(including_sour)___healthy":"Cherry___Healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot":"Corn___Gray_Leaf_Spot",
    "Corn_(maize)___Common_rust_":"Corn___Common_Rust",
    "Corn_(maize)___Northern_Leaf_Blight":"Corn___Northern_Leaf_Blight",
    "Corn_(maize)___healthy":"Corn___Healthy",
    "Grape___Black_rot":"Grape___Black_Rot","Grape___Esca_(Black_Measles)":"Grape___Esca_Black_Measles",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)":"Grape___Leaf_Blight","Grape___healthy":"Grape___Healthy",
    "Peach___Bacterial_spot":"Peach___Bacterial_Spot","Peach___healthy":"Peach___Healthy",
    "Potato___Early_blight":"Potato___Early_Blight","Potato___Late_blight":"Potato___Late_Blight",
    "Potato___healthy":"Potato___Healthy",
    "Strawberry___Leaf_scorch":"Strawberry___Leaf_Scorch","Strawberry___healthy":"Strawberry___Healthy",
    "Tomato___Bacterial_spot":"Tomato___Bacterial_Spot","Tomato___Early_blight":"Tomato___Early_Blight",
    "Tomato___Late_blight":"Tomato___Late_Blight","Tomato___Leaf_Mold":"Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot":"Tomato___Septoria_Leaf_Spot",
    "Tomato___Spider_mites Two-spotted_spider_mite":"Tomato___Spider_Mites",
    "Tomato___Target_Spot":"Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus":"Tomato___Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus":"Tomato___Mosaic_Virus","Tomato___healthy":"Tomato___Healthy",
}

_valid = set(UNIFIED_CLASSES)
for _src, _sn in [(PLANTDOC_MAP,"PlantDoc"),(PLANTVILLAGE_MAP,"PlantVillage")]:
    for k, v in _src.items():
        assert v in _valid, f"[{_sn}] Unknown label: '{v}' (key='{k}')"
print("✅ All map entries validated")


 
# STEP 3 — Collect Source Images
 
def collect_imgs(root, folder_map):
    root = Path(root); out = defaultdict(list)
    if not root.exists():
        print(f"  ⚠️ Not found: {root}"); return out
    for folder, label in folder_map.items():
        fp = root / folder
        if not fp.exists(): continue
        out[label].extend(str(p) for p in fp.iterdir()
                          if p.suffix.lower() in IMG_EXTS)
    return out

print("\n📂 Collecting images…")
pd_tr = collect_imgs(PLANTDOC_TRAIN, PLANTDOC_MAP)
pd_te = collect_imgs(PLANTDOC_TEST,  PLANTDOC_MAP)
pv    = collect_imgs(PLANTVILLAGE,   PLANTVILLAGE_MAP)

pd = defaultdict(list)
for d in (pd_tr, pd_te):
    for cls, paths in d.items(): pd[cls].extend(paths)

# ══════════════════════════════════════════════════════════════════
# FILTER 1 — Common classes: present in BOTH PlantDoc AND PlantVillage
# ══════════════════════════════════════════════════════════════════
_pd_has = set(c for c in UNIFIED_CLASSES if len(pd[c]) > 0)
_pv_has = set(c for c in UNIFIED_CLASSES if len(pv[c]) > 0)
_common  = [c for c in UNIFIED_CLASSES if c in _pd_has and c in _pv_has]

# ══════════════════════════════════════════════════════════════════
# FILTER 2 — Drop crops that have NO disease class in the common set
#             (i.e., only a "Healthy" variant → not useful for disease detection)
# ══════════════════════════════════════════════════════════════════
from collections import defaultdict as _dd
_crop_classes = _dd(list)
for _c in _common:
    _crop_classes[_c.split('___')[0]].append(_c)

_kept_crops, _dropped_crops = [], []
for _crop, _cls_list in _crop_classes.items():
    _has_disease = any('Healthy' not in _c for _c in _cls_list)
    if _has_disease:
        _kept_crops.append(_crop)
    else:
        _dropped_crops.append(_crop)

UNIFIED_CLASSES = [c for c in _common if c.split('___')[0] in set(_kept_crops)]
NUM_CLASSES     = len(UNIFIED_CLASSES)
LABEL_MAP       = {cls: idx for idx, cls in enumerate(UNIFIED_CLASSES)}
IDX_TO_CLASS    = {idx: cls for cls, idx in LABEL_MAP.items()}

# ── Audit print ──────────────────────────────────────────────────
print(f"\n{'═'*68}")
print(f"  DATASET AUDIT — Common classes after both filters")
print(f"{'═'*68}")
print(f"  {'Class':<40} {'PD':>5} {'PV':>6} {'Total':>7}  Status")
print(f"  {'─'*68}")
for _crop in sorted(_crop_classes):
    _cls_list = _crop_classes[_crop]
    _is_kept  = _crop in set(_kept_crops)
    for _c in sorted(_cls_list):
        _in_common = _c in set(_common)
        if not _in_common:
            print(f"  {_c:<40} {'—':>5} {'—':>6} {'—':>7}  ✗ not in both datasets")
        elif not _is_kept:
            print(f"  {_c:<40} {len(pd[_c]):>5} {len(pv[_c]):>6} {len(pd[_c])+len(pv[_c]):>7}  ✗ crop-only-healthy → dropped")
        else:
            tag = '🌿 healthy' if 'Healthy' in _c else '🦠 disease'
            print(f"  {_c:<40} {len(pd[_c]):>5} {len(pv[_c]):>6} {len(pd[_c])+len(pv[_c]):>7}  ✅ {tag}")

if _dropped_crops:
    print(f"\n  ⚠️  Dropped crops (only-healthy, no disease class): {', '.join(sorted(_dropped_crops))}")
print(f"\n  ✅ Final: {NUM_CLASSES} classes across {len(_kept_crops)} crops: {', '.join(sorted(_kept_crops))}")
print(f"{'═'*68}")


 
# STEP 4 — Splits: aug-pad TRAIN only; val/test = real images only
 
def balanced_split_augmented(pd_imgs, pv_imgs, nt, nv, nte, seed=42):
    """
    Train split  → always exactly `nt` paths; padded with aug-repeats if needed.
    Val / Test   → REAL images only (up to nv / nte each); NO padding.
                   Fewer samples are acceptable — no repeated-image leakage.

    Returns tr, vl, te paths + is_aug bool lists
    """
    rng  = random.Random(seed)
    pool = list(set(pd_imgs + pv_imgs))   # deduplicate, combine both sources
    rng.shuffle(pool)
    n_real = len(pool)

    if n_real == 0:
        return [], [], [], [], [], []

    # ── proportional real-image allocation ───────────────────────────────
    n_need   = nt + nv + nte
    ratio_tr = nt / n_need
    ratio_vl = nv / n_need

    if n_real >= n_need:
        # enough real images → no augmentation needed anywhere
        tr_real = pool[:nt]
        vl_real = pool[nt:nt + nv]
        te_real = pool[nt + nv:nt + nv + nte]
        aug_tr  = [False] * nt
    else:
        # proportional split of available real images
        _nt  = max(1, min(n_real - 2, int(n_real * ratio_tr)))
        _nv  = max(1, min(n_real - _nt - 1, int(n_real * ratio_vl)))
        _nte = max(1, n_real - _nt - _nv)
        tr_real = pool[:_nt]
        vl_real = pool[_nt:_nt + _nv]
        te_real = pool[_nt + _nv:_nt + _nv + _nte]

        # ── TRAIN: pad to exact target with aug-repeats ───────────────
        out_tr = tr_real[:]
        aug_tr = [False] * len(tr_real)
        while len(out_tr) < nt:
            out_tr.append(rng.choice(tr_real))
            aug_tr.append(True)
        tr_real = out_tr[:nt]
        aug_tr  = aug_tr[:nt]

    # val / test: use only real images, no padding
    aug_vl = [False] * len(vl_real)
    aug_te = [False] * len(te_real)

    return tr_real, vl_real, te_real, aug_tr, aug_vl, aug_te


print("\n🔧 Building splits with augmentation balancing…")
train_paths, train_labels = [], []
val_paths,   val_labels   = [], []
test_paths,  test_labels  = [], []
train_is_aug = []; val_is_aug = []; test_is_aug = []
aug_report   = {}

for cls in UNIFIED_CLASSES:
    idx = LABEL_MAP[cls]
    tr, vl, te, a_tr, a_vl, a_te = balanced_split_augmented(
        pd[cls], pv[cls], N_TRAIN, N_VAL, N_TEST, seed=SEED)
    train_paths += tr;  train_labels += [idx]*len(tr);  train_is_aug += a_tr
    val_paths   += vl;  val_labels   += [idx]*len(vl);  val_is_aug   += a_vl
    test_paths  += te;  test_labels  += [idx]*len(te);  test_is_aug  += a_te
    aug_report[cls] = (sum(not x for x in a_tr), sum(a_tr),
                       sum(not x for x in a_vl), sum(a_vl),
                       sum(not x for x in a_te), sum(a_te))

total_aug_tr = sum(train_is_aug)
# val/test have NO aug-padding (all False)

# ── Detailed report ──────────────────────────────────────────────────────
print(f"\n  {'Class':<38} {'Tr':>5}/{'Aug':>5} {'Vl(real)':>9} {'Te(real)':>9}")
print("  " + "─"*72)
for cls in UNIFIED_CLASSES:
    nr_tr, na_tr, nr_vl, _, nr_te, _ = aug_report[cls]
    flag = " ←aug" if na_tr > 0 else ""
    print(f"  {cls:<38} {nr_tr:>4}/{na_tr:<5} {nr_vl:>9} {nr_te:>9}{flag}")

print(f"\n  Train : {len(train_paths):,}  "
      f"(real={len(train_paths)-total_aug_tr:,}  "
      f"aug-padded={total_aug_tr:,} = {total_aug_tr/max(len(train_paths),1)*100:.1f}%)")
print(f"  Val   : {len(val_paths):,}  (real images only — no padding)")
print(f"  Test  : {len(test_paths):,}  (real images only — no padding)")

# ── Sanity checks ────────────────────────────────────────────────────────
assert len(train_paths) == NUM_CLASSES * N_TRAIN, \
    f"Train size mismatch: {len(train_paths)} != {NUM_CLASSES*N_TRAIN}"
# Val/Test sizes are variable (real only), just verify every class is present
for sname, slbls in [("Val", val_labels), ("Test", test_labels)]:
    present = set(slbls)
    missing = set(range(NUM_CLASSES)) - present
    if missing:
        print(f"  ⚠️  {sname}: {len(missing)} classes have 0 real images! "
              f"(indices: {sorted(missing)[:5]}…)")
    else:
        print(f"  {sname}: ✅ all {NUM_CLASSES} classes present "
              f"({len(slbls):,} real images total)")
print(f"  Train: ✅ {NUM_CLASSES} classes × {N_TRAIN} = {len(train_paths):,} "
      f"(aug-padded where needed)")


 
# STEP 5 — Transforms
 
MEAN = [0.485, 0.456, 0.406]; STD = [0.229, 0.224, 0.225]

train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE+32, IMG_SIZE+32)),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(0.3, 0.3, 0.2, 0.05),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
    transforms.RandomErasing(p=0.15, scale=(0.02, 0.12)),
])
val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])
print("✅ Transforms ready (train: +RandomErasing for aug-padded diversity)")


 
# STEP 6 — Dataset & DataLoader
 
class CropDS(Dataset):
    def __init__(self, paths, labels, tf=None, is_aug=None):
        self.paths  = paths; self.labels = labels
        self.tf     = tf
        self.is_aug = is_aug or [False]*len(paths)
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        try:    img = Image.open(self.paths[i]).convert("RGB")
        except: img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (128,128,128))
        if self.tf: img = self.tf(img)
        return img, torch.tensor(self.labels[i], dtype=torch.long)

# Detect headless / Kaggle Save-Version to prevent multiprocessing hangs
_IS_INTERACTIVE = hasattr(sys, 'ps1') or (sys.stdout is not None and sys.stdout.isatty())
_NUM_WORKERS    = 0 if (not _IS_INTERACTIVE or not torch.cuda.is_available()) else 2
print(f"✅ DataLoader workers: {_NUM_WORKERS}")

def make_loader(paths, labels, tf, shuffle=False, is_aug=None):
    ds = CropDS(paths, labels, tf, is_aug)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle,
                      num_workers=_NUM_WORKERS, pin_memory=torch.cuda.is_available(),
                      drop_last=shuffle)

train_loader = make_loader(train_paths, train_labels, train_tf,
                           shuffle=True, is_aug=train_is_aug)
val_loader   = make_loader(val_paths,  val_labels,  val_tf, is_aug=val_is_aug)
test_loader  = make_loader(test_paths, test_labels, val_tf, is_aug=test_is_aug)
print(f"✅ Loaders | Train:{len(train_loader)} Val:{len(val_loader)} Test:{len(test_loader)} batches")


 
# Denormalize helper (used throughout visualizations)
 
def _denorm(t):
    """CHW tensor → HWC uint8 numpy array for display."""
    m = torch.tensor(MEAN).view(3,1,1); s = torch.tensor(STD).view(3,1,1)
    return ((t*s+m).clamp(0,1).permute(1,2,0).numpy()*255).astype(np.uint8)


# ════════════════════════════════════════════════════════════
# Dataset Samples — 1 image per class (5 cols × 4 rows)
# ════════════════════════════════════════════════════════════
print("\n🖼️  Dataset samples (1 per class) …")
_N_COLS_VIZ1 = 5
_N_ROWS_VIZ1 = math.ceil(NUM_CLASSES / _N_COLS_VIZ1)
fig, axes = plt.subplots(_N_ROWS_VIZ1, _N_COLS_VIZ1,
                         figsize=(_N_COLS_VIZ1 * 2.8, _N_ROWS_VIZ1 * 2.8),
                         squeeze=False)
fig.suptitle("Representative Samples per Disease Class",
             fontsize=13, fontweight='bold', y=1.01)
_ds_raw = CropDS(train_paths, train_labels, tf=val_tf)
_cls_done = {i: None for i in range(NUM_CLASSES)}
for _i in range(len(_ds_raw)):
    _lbl = train_labels[_i]
    if _cls_done[_lbl] is None:
        _img, _ = _ds_raw[_i]
        _cls_done[_lbl] = _img
    if all(v is not None for v in _cls_done.values()):
        break
for _idx in range(_N_ROWS_VIZ1 * _N_COLS_VIZ1):
    _r, _c = divmod(_idx, _N_COLS_VIZ1)
    ax = axes[_r][_c]
    if _idx < NUM_CLASSES:
        _img_t = _cls_done.get(_idx)
        if _img_t is not None:
            ax.imshow(_denorm(_img_t))
        else:
            ax.set_facecolor('#eee')
        _lbl_name = IDX_TO_CLASS[_idx].replace('___', '\n').replace('_', ' ')
        ax.set_title(_lbl_name, fontsize=6.5, pad=3)
    else:
        ax.set_visible(False)
    ax.axis('off')
plt.tight_layout()
save_fig(fig, "viz01_data_samples", "Dataset Samples per Class")
del _ds_raw, _cls_done


 
# STEP 7 — Loss & Mixup
 
class LabelSmoothCE(nn.Module):
    def __init__(self, nc, sm=0.1):
        super().__init__(); self.nc = nc; self.sm = sm
    def forward(self, logits, targets):
        lp = F.log_softmax(logits, -1)
        if targets.dim() == 1:
            st = torch.full_like(lp, self.sm/(self.nc-1))
            st.scatter_(1, targets.unsqueeze(1), 1.0-self.sm)
        else:
            st = targets*(1-self.sm) + self.sm/self.nc
        return -(st * lp).sum(-1).mean()

criterion = LabelSmoothCE(NUM_CLASSES, LABEL_SMOOTH)

def mixup(x, y, alpha=0.4):
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(x.size(0), device=x.device)
    ya  = F.one_hot(y, NUM_CLASSES).float()
    yb  = F.one_hot(y[idx], NUM_CLASSES).float()
    return lam*x + (1-lam)*x[idx], lam*ya + (1-lam)*yb

print("✅ Loss: LabelSmoothCE | Mixup α=0.4")


 
# STEP 8 — Training Helpers
 
class EarlyStopping:
    def __init__(self, patience=10, delta=1e-4, path="best.pth"):
        self.patience=patience; self.delta=delta; self.path=path
        self.best=float('inf'); self.counter=0; self.best_ep=0
    def step(self, vl, ep, model):
        if vl < self.best - self.delta:
            self.best=vl; self.counter=0; self.best_ep=ep
            torch.save(model.state_dict(), self.path); return False
        self.counter += 1
        return self.counter >= self.patience

def train_ep(model, loader, crit, opt):
    model.train(); total = 0.0
    _dev = DEVICE.type  # 'cuda' or 'cpu'
    for imgs, lbls in loader:
        imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
        imgs, tgts = mixup(imgs, lbls, alpha=0.4)
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=_dev, enabled=USE_AMP and _dev=='cuda'):
            loss = crit(model(imgs), tgts)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update()
        total += loss.item()
    if len(loader) == 0:
        return 0.0
    return total / len(loader)

@torch.no_grad()
def evaluate(model, loader, crit):
    model.eval(); tot, probs, preds, gts = 0.0, [], [], []
    _dev = DEVICE.type
    for imgs, lbls in loader:
        imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
        with torch.amp.autocast(device_type=_dev, enabled=USE_AMP and _dev=='cuda'):
            logits = model(imgs)
        tot  += crit(logits, lbls).item()
        probs.extend(F.softmax(logits,-1).cpu().float().numpy())
        preds.extend(logits.argmax(-1).cpu().numpy())
        gts.extend(lbls.cpu().numpy())
    if not gts:  # empty loader or dataset
        return 0.0, 0.0, np.zeros((0, NUM_CLASSES)), np.array([]), np.array([])
    return (tot/max(len(loader),1), accuracy_score(gts, preds),
            np.array(probs), np.array(preds), np.array(gts))

def full_metrics(gts, preds, probs):
    acc  = accuracy_score(gts, preds)
    prec = precision_score(gts, preds, average='macro', zero_division=0)
    rec  = recall_score(gts, preds, average='macro', zero_division=0)
    mf1  = f1_score(gts, preds, average='macro', zero_division=0)
    wf1  = f1_score(gts, preds, average='weighted', zero_division=0)
    try:
        cp = np.unique(gts)
        if len(cp) == NUM_CLASSES:
            auc = roc_auc_score(gts, probs, multi_class='ovr', average='macro')
        else:
            yb   = label_binarize(gts, classes=list(range(NUM_CLASSES)))
            aucs = [roc_auc_score(yb[:,i],probs[:,i])
                    for i in range(NUM_CLASSES) if i in cp and yb[:,i].sum()>0]
            auc = float(np.mean(aucs)) if aucs else float('nan')
    except: auc = float('nan')
    return dict(acc=acc,prec=prec,rec=rec,mf1=mf1,wf1=wf1,auc=auc)

@torch.no_grad()
def inf_time(model, n=200):
    model.eval(); d = torch.randn(1,3,IMG_SIZE,IMG_SIZE).to(DEVICE)
    for _ in range(20): model(d)
    if torch.cuda.is_available(): torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n): model(d)
    if torch.cuda.is_available(): torch.cuda.synchronize()
    return (time.perf_counter()-t0)/n*1000

def save_fig(fig, fname, title=""):
    p = FIG_DIR/fname
    fig.savefig(str(p)+".pdf", bbox_inches='tight')
    fig.savefig(str(p)+".png", bbox_inches='tight', dpi=150)
    plt.show(); plt.close(fig)
    print(f"  ✅ {title or fname}")

def safe_ylim(ax, vals, lo=0.97, hi=1.02):
    fv = [v for v in vals if np.isfinite(v)]
    if len(fv) >= 2: ax.set_ylim(min(fv)*lo, max(fv)*hi)

print("✅ Training helpers ready")


 
# STEP 9 — Attention Modules
 
class ChannelAttn(nn.Module):
    def __init__(self, c, r=16):
        super().__init__()
        mid = max(c//r, 8)
        self.mlp = nn.Sequential(nn.Linear(c,mid,bias=False),nn.ReLU(inplace=True),nn.Linear(mid,c,bias=False))
    def forward(self, x):
        a = self.mlp(F.adaptive_avg_pool2d(x,1).flatten(1))
        m = self.mlp(F.adaptive_max_pool2d(x,1).flatten(1))
        return x * torch.sigmoid(a+m).unsqueeze(-1).unsqueeze(-1)

class SpatialAttn(nn.Module):
    def __init__(self): super().__init__(); self.conv = nn.Conv2d(2,1,7,padding=3,bias=False)
    def forward(self, x):
        a = x.mean(1,keepdim=True); m,_ = x.max(1,keepdim=True)
        return x * torch.sigmoid(self.conv(torch.cat([a,m],1)))

class CBAM(nn.Module):
    def __init__(self,c,r=16): super().__init__(); self.ca=ChannelAttn(c,r); self.sa=SpatialAttn()
    def forward(self,x): return self.sa(self.ca(x))

class CoordAtt(nn.Module):
    def __init__(self,c,r=32):
        super().__init__(); mid=max(8,c//r)
        self.c1=nn.Conv2d(c,mid,1,bias=False); self.bn=nn.BatchNorm2d(mid); self.act=nn.Hardswish(inplace=True)
        self.ch=nn.Conv2d(mid,c,1,bias=False); self.cw=nn.Conv2d(mid,c,1,bias=False)
    def forward(self,x):
        B,C,H,W=x.shape
        xh=F.adaptive_avg_pool2d(x,(H,1)); xw=F.adaptive_avg_pool2d(x,(1,W)).permute(0,1,3,2)
        y=self.act(self.bn(self.c1(torch.cat([xh,xw],2))))
        xh,xw=torch.split(y,[H,W],2)
        return x*torch.sigmoid(self.ch(xh))*torch.sigmoid(self.cw(xw.permute(0,1,3,2)))

class GeMPool(nn.Module):
    def __init__(self,p=3.0,eps=1e-6): super().__init__(); self.p=nn.Parameter(torch.tensor(p)); self.eps=eps
    def forward(self,x): return F.adaptive_avg_pool2d(x.clamp(min=self.eps).pow(self.p),1).pow(1/self.p).flatten(1)

print("✅ Attention: CBAM | CoordAtt | GeMPool")


 
# STEP 10 — Baseline Models 
 
class ResNet50Base(nn.Module):
    def __init__(self,nc=NUM_CLASSES,drop=0.4):
        super().__init__()
        b = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(b.conv1,b.bn1,b.relu,b.maxpool,b.layer1,b.layer2,b.layer3,b.layer4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Dropout(drop), nn.Linear(2048,nc))
    def forward(self,x): return self.head(self.pool(self.backbone(x)).flatten(1))

class ViTBase(nn.Module):
    def __init__(self,nc=NUM_CLASSES,drop=0.1):
        super().__init__()
        self.vit  = timm.create_model('vit_base_patch16_224',pretrained=True,num_classes=0,drop_rate=drop)
        self.head = nn.Linear(768,nc)
    def forward(self,x): return self.head(self.vit(x))

BASELINE_CONFIGS = {"ResNet-50": ResNet50Base, "ViT-Base": ViTBase}
BPAL = {"ResNet-50": PAL['resnet'], "ViT-Base": PAL['vit']}

print("\n📐 Baseline params:")
for n,c in BASELINE_CONFIGS.items():
    m=c(); print(f"  {n:<16}: {sum(p.numel() for p in m.parameters() if p.requires_grad)/1e6:.1f}M"); del m


 
# STEP 11 — Hybrid Models  
 
class CrossAttnGate(nn.Module):
    def __init__(self,dim=512,heads=8):
        super().__init__(); assert dim%heads==0
        self.h=heads; self.sc=(dim//heads)**-0.5
        self.q1=nn.Linear(dim,dim,bias=False); self.k2=nn.Linear(dim,dim,bias=False)
        self.v2=nn.Linear(dim,dim,bias=False); self.o1=nn.Linear(dim,dim,bias=False)
        self.q2=nn.Linear(dim,dim,bias=False); self.k1=nn.Linear(dim,dim,bias=False)
        self.v1=nn.Linear(dim,dim,bias=False); self.o2=nn.Linear(dim,dim,bias=False)
        self.n1=nn.LayerNorm(dim); self.n2=nn.LayerNorm(dim)
    def forward(self,a,b):
        B,D=a.shape; H,d=self.h,D//self.h
        q1=self.q1(a).reshape(B,H,d); k2=self.k2(b).reshape(B,H,d); v2=self.v2(b).reshape(B,H,d)
        w1=torch.sigmoid((q1*k2*self.sc).sum(-1,keepdim=True))
        ar=self.n1(a+self.o1((w1*v2).reshape(B,D)))
        q2=self.q2(b).reshape(B,H,d); k1=self.k1(a).reshape(B,H,d); v1=self.v1(a).reshape(B,H,d)
        w2=torch.sigmoid((q2*k1*self.sc).sum(-1,keepdim=True))
        br=self.n2(b+self.o2((w2*v1).reshape(B,D)))
        return ar,br

def _vec(f):
    """Collapse spatial/token dims → [B, C] feature vector.

    Handles three layouts produced by timm backbones:
      • NCHW  [B, C, H, W]  – standard ConvNets (ConvNeXt, EfficientNet)
      • NHWC  [B, H, W, C]  – Swin Transformer (channel-last)
      • NLC   [B, L, C]     – ViT token sequence
    """
    if f.dim() == 4:
        _, d1, d2, d3 = f.shape
        # NHWC (Swin): last dim is the large channel dim  e.g. [B,7,7,768]
        # NCHW (ConvNet): dim-1 is the large channel dim  e.g. [B,768,7,7]
        if d3 >= d1:          # channel-last → average over H,W
            return f.mean(dim=[1, 2])
        else:                  # channel-first → average over H,W
            return f.mean(dim=[2, 3])
    if f.dim() == 3:           # [B, L, C] transformer tokens → average over L
        return f.mean(dim=1)
    return f                   # already [B, C]

class OmniNetCS(nn.Module):
    def __init__(self,nc=NUM_CLASSES,drop=0.4,proj=512):
        super().__init__()
        self.convnext = timm.create_model('convnext_tiny',pretrained=True,num_classes=0,global_pool='')
        self.cbam_a=CBAM(768); self.gem_a=GeMPool()
        self.proj_a=nn.Sequential(nn.Linear(768,proj),nn.LayerNorm(proj),nn.GELU(),nn.Dropout(drop*0.5))
        self.swin   = timm.create_model('swin_tiny_patch4_window7_224',pretrained=True,num_classes=0)
        self.norm_b = nn.LayerNorm(768)
        self.proj_b = nn.Sequential(nn.Linear(768,proj),nn.LayerNorm(proj),nn.GELU(),nn.Dropout(drop*0.5))
        self.cross  = CrossAttnGate(proj,heads=8)
        self.head   = nn.Sequential(nn.Linear(proj*2,512),nn.LayerNorm(512),nn.GELU(),nn.Dropout(drop),
                                     nn.Linear(512,256),nn.GELU(),nn.Dropout(drop/2),nn.Linear(256,nc))
    def forward_a(self,x): return self.proj_a(self.gem_a(self.cbam_a(self.convnext.forward_features(x))))
    def forward_b(self,x): return self.proj_b(self.norm_b(_vec(self.swin.forward_features(x))))
    def forward(self,x):
        a,b=self.cross(self.forward_a(x),self.forward_b(x)); return self.head(torch.cat([a,b],1))

class OmniNetEV(nn.Module):
    def __init__(self,nc=NUM_CLASSES,drop=0.35,proj=512):
        super().__init__()
        self.effnet = timm.create_model('tf_efficientnetv2_s',pretrained=True,num_classes=0,global_pool='')
        self.ca_a=CoordAtt(1280); self.gem_a=GeMPool()
        self.proj_a=nn.Sequential(nn.Linear(1280,proj),nn.LayerNorm(proj),nn.GELU(),nn.Dropout(drop*0.5))
        # vit_tiny: forward_features → [B, L, 192]; pool tokens first, then norm+proj
        self.vit    = timm.create_model('vit_tiny_patch16_224',pretrained=True,num_classes=0)
        self.norm_b = nn.LayerNorm(192)
        self.proj_b = nn.Sequential(nn.Linear(192,proj),nn.LayerNorm(proj),nn.GELU(),nn.Dropout(drop*0.5))
        self.cross  = CrossAttnGate(proj,heads=8)
        self.head   = nn.Sequential(nn.Linear(proj*2,512),nn.LayerNorm(512),nn.GELU(),nn.Dropout(drop),
                                     nn.Linear(512,256),nn.GELU(),nn.Dropout(drop/2),nn.Linear(256,nc))
    def forward_a(self,x): return self.proj_a(self.gem_a(self.ca_a(self.effnet.forward_features(x))))
    def forward_b(self,x):
        # Pool tokens → [B, 192], then normalise and project → [B, proj]
        tok = self.vit.forward_features(x)   # [B, L, 192]  (L=197 for 224px)
        vec = tok.mean(dim=1)                # [B, 192]
        return self.proj_b(self.norm_b(vec))
    def forward(self,x):
        a,b=self.cross(self.forward_a(x),self.forward_b(x)); return self.head(torch.cat([a,b],1))

HYBRID_CONFIGS = {"OmniNet-CS": OmniNetCS, "OmniNet-EV": OmniNetEV}
HPAL = {"OmniNet-CS": PAL['omnics'], "OmniNet-EV": PAL['omniev']}

print("\n📐 All model params (4 total):")
for n,c in {**BASELINE_CONFIGS,**HYBRID_CONFIGS}.items():
    m=c(); print(f"  {n:<20}: {sum(p.numel() for p in m.parameters() if p.requires_grad)/1e6:.1f}M"); del m


 
# STEP 12 — Training Runner 
 
def run_training(name, model, n_epochs, save_path, bb_attrs=None, lr_scales=None):
    model=model.to(DEVICE); save_path=str(save_path)
    if bb_attrs:
        seen=set(); groups=[]
        for attr,scale in zip(bb_attrs,lr_scales):
            ps=list(getattr(model,attr).parameters())
            groups.append({'params':ps,'lr':LR*scale}); seen|={id(p) for p in ps}
        rest=[p for p in model.parameters() if id(p) not in seen]
        groups.append({'params':rest,'lr':LR})
    else:
        groups=[{'params':model.parameters(),'lr':LR}]
    opt = optim.AdamW(groups, weight_decay=WEIGHT_DECAY)
    sch = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=10, T_mult=2)
    es  = EarlyStopping(patience=PATIENCE, path=save_path)
    hist = {'tr':[],'vl':[],'va':[],'vf1':[]}
    print(f"\n{'─'*65}")
    print(f"  {name}  ({n_epochs} ep | patience={PATIENCE})")
    print(f"  {'Ep':>4} {'Time':>5} {'TrainLoss':>10} {'ValLoss':>9} {'ValAcc%':>8} {'MacF1%':>8} {'':>8}")
    print(f"{'─'*65}")
    for ep in range(1, n_epochs+1):
        t0=time.time(); trl=train_ep(model,train_loader,criterion,opt)
        vll,vla,_vp,_vpr,_vgt=evaluate(model,val_loader,criterion); sch.step()
        vf1=f1_score(_vgt,_vpr,average='macro',zero_division=0)
        hist['tr'].append(trl); hist['vl'].append(vll)
        hist['va'].append(vla); hist['vf1'].append(vf1)
        stop=es.step(vll,ep,model)
        flag="🏅" if es.counter==0 else f"({es.counter}/{PATIENCE})"
        print(f"  {ep:4d} {time.time()-t0:4.0f}s {trl:10.4f} {vll:9.4f} {vla*100:8.2f} {vf1*100:8.2f}  {flag}")
        if stop:
            print(f"  ⏹ Early stop ep{ep} | best ep{es.best_ep}"); break
    print(f"{'─'*65}")
    model.load_state_dict(torch.load(save_path, map_location=DEVICE, weights_only=True))
    print(f"  ✅ Best weights loaded (ep{es.best_ep})")
    return model, hist, es.best_ep


 
# STEP 13 — Train Hybrids FIRST (proposed models)
 
hybrid_results={}; hybrid_hists={}; hybrid_inf={}; hybrid_models={}
print("\n"+"="*65+"\n🧬  HYBRIDS (2 models) — training first\n"+"="*65)
for h_name, MCls in HYBRID_CONFIGS.items():
    m=MCls(); sp=CKPT_DIR/f"hybrid_{h_name.replace('-','').replace(' ','_')}.pth"
    bb=['convnext','swin'] if h_name=="OmniNet-CS" else ['effnet','vit']
    sc=[0.05,0.10]
    m,hist,_=run_training(h_name,m,HYBRID_EPOCHS,sp,bb_attrs=bb,lr_scales=sc)
    _,_,probs,preds,gts=evaluate(m,test_loader,criterion)
    met=full_metrics(gts,preds,probs); it=inf_time(m)
    hybrid_results[h_name]=met; hybrid_hists[h_name]=hist
    hybrid_inf[h_name]=it; hybrid_models[h_name]=(m,str(sp))
    _auc_str = f"{met['auc']:.4f}" if np.isfinite(met['auc']) else "nan"
    print(f"\n  📌 {h_name} Acc:{met['acc']*100:.2f}% MacF1:{met['mf1']*100:.2f}% AUC:{_auc_str} Inf:{it:.1f}ms")

winner_name=max(hybrid_results, key=lambda k: hybrid_results[k]['mf1'])
winner_model,winner_save=hybrid_models[winner_name]
print(f"\n🏆  HYBRID WINNER: {winner_name}  MacF1={hybrid_results[winner_name]['mf1']*100:.3f}%")


 
# STEP 14 — Train Baselines (comparison)
 
baseline_results={}; baseline_hists={}; baseline_inf={}
print("\n"+"="*65+"\n📊  BASELINES (2 models) — comparison\n"+"="*65)
for b_name, MCls in BASELINE_CONFIGS.items():
    m=MCls(); sp=CKPT_DIR/f"base_{b_name.replace('-','').replace(' ','_')}.pth"
    bb=['backbone'] if b_name=="ResNet-50" else ['vit']
    sc=[0.1]       if b_name=="ResNet-50" else [0.05]
    m,hist,_=run_training(b_name,m,BASE_EPOCHS,sp,bb_attrs=bb,lr_scales=sc)
    _,_,probs,preds,gts=evaluate(m,test_loader,criterion)
    met=full_metrics(gts,preds,probs); it=inf_time(m)
    baseline_results[b_name]=met; baseline_hists[b_name]=hist; baseline_inf[b_name]=it
    _auc_str = f"{met['auc']:.4f}" if np.isfinite(met['auc']) else "nan"
    print(f"\n  📌 {b_name} Acc:{met['acc']*100:.2f}% MacF1:{met['mf1']*100:.2f}% AUC:{_auc_str} Inf:{it:.1f}ms")
    del m; torch.cuda.empty_cache()


 
# STEP 15 — Results Table 
 
all_results={**baseline_results,**hybrid_results}
all_inf={**baseline_inf,**hybrid_inf}
all_names=list(all_results.keys())
short_nms=all_names

print("\n"+"═"*78)
print(f"{'Model':<20} {'Acc%':>6} {'Prec%':>6} {'Rec%':>6} {'MacF1%':>7} {'WtF1%':>6} {'AUC':>6} {'ms':>6}")
print("═"*78)
for name,met in all_results.items():
    mk=" 🏆" if name==winner_name else ""
    _auc_disp = f"{met['auc']:6.4f}" if np.isfinite(met['auc']) else "   nan"
    print(f"{name:<20} {met['acc']*100:>6.2f} {met['prec']*100:>6.2f} "
          f"{met['rec']*100:>6.2f} {met['mf1']*100:>7.2f} "
          f"{met['wf1']*100:>6.2f} {_auc_disp} "
          f"{all_inf[name]:>6.1f}{mk}")
print("═"*78)

_,_,test_probs,test_preds,test_true=evaluate(winner_model,test_loader,criterion)
test_probs=np.array(test_probs)
m_win=full_metrics(test_true,test_preds,test_probs)
print(f"\nPer-class report — {winner_name}")
print(classification_report(test_true,test_preds,
      target_names=[IDX_TO_CLASS[i] for i in range(NUM_CLASSES)],zero_division=0))


 
# STEP 16 — Visualizations (each printed separately)
 

from sklearn.metrics import roc_curve, auc as sk_auc

# ════════════════════════════════════════════════════════════
# Class Distribution BEFORE augmentation (raw counts)
# ════════════════════════════════════════════════════════════
print("\n📊 Class distribution (before augmentation) …")
raw_counts = [aug_report[UNIFIED_CLASSES[i]][0] +      # real train
              aug_report[UNIFIED_CLASSES[i]][2] +      # real val
              aug_report[UNIFIED_CLASSES[i]][4]        # real test
              for i in range(NUM_CLASSES)]
cls_fl = [IDX_TO_CLASS[i].replace('___',' ').replace('_',' ')
          for i in range(NUM_CLASSES)]
fig, ax = plt.subplots(figsize=(14, max(6, NUM_CLASSES*0.32)))
bars = ax.barh(cls_fl, raw_counts,
               color='#5b8dd9', edgecolor='white', linewidth=0.4, height=0.72)
ax.axvline(N_TOTAL, color='crimson', lw=1.6, ls='--',
           label=f'Target = {N_TOTAL}')
for bar, val in zip(bars, raw_counts):
    ax.text(val + 6, bar.get_y()+bar.get_height()/2,
            str(val), va='center', fontsize=6.5)
ax.set_xlabel("Number of real images available")
ax.set_title("Dataset Class Distribution Before Augmentation",
             fontweight='bold')
ax.legend(fontsize=9); ax.grid(axis='x', alpha=0.3, ls='--')
plt.tight_layout()
save_fig(fig, "viz02a_dist_before_aug", "VIZ-2a — Distribution Before Aug")

# ════════════════════════════════════════════════════════════
# VIZ-2b  Class Distribution AFTER augmentation (train split)
# ════════════════════════════════════════════════════════════
print("\n📊 VIZ-2b: Class distribution after augmentation (train split) …")
real_c = [aug_report[UNIFIED_CLASSES[i]][0] for i in range(NUM_CLASSES)]
aug_c  = [aug_report[UNIFIED_CLASSES[i]][1] for i in range(NUM_CLASSES)]
y = np.arange(NUM_CLASSES)
fig, ax = plt.subplots(figsize=(14, max(6, NUM_CLASSES*0.32)))
ax.barh(y, real_c, color=PAL['omnics'], edgecolor='white',
        lw=0.4, height=0.72, label='Real images')
ax.barh(y, aug_c, left=real_c, color=PAL['neg'], edgecolor='white',
        lw=0.4, height=0.72, alpha=0.75,
        label='Aug-padded (repeat + random transform)')
ax.set_yticks(y); ax.set_yticklabels(cls_fl, fontsize=7.5)
ax.axvline(N_TRAIN, color='navy', lw=1.5, ls='--', label=f'Target={N_TRAIN}')
for i, (r, a) in enumerate(zip(real_c, aug_c)):
    lbl = f"{r+a}" if a == 0 else f"{r}+{a}aug"
    ax.text(r+a+3, i, lbl, va='center', fontsize=6)
ax.set_xlabel("Training images (Real + Aug-padded)")
ax.set_title(f"Class Distribution After Augmentation (train, target={N_TRAIN}/class)",
             fontweight='bold')
ax.legend(fontsize=9); ax.grid(axis='x', alpha=0.3, ls='--')
plt.tight_layout()
save_fig(fig, "viz02b_dist_after_aug", "VIZ-2b — Distribution After Aug")

# ════════════════════════════════════════════════════════════
# VIZ-3  Learning curves — all 4 models in one figure
# ════════════════════════════════════════════════════════════
print("\n📈 VIZ-3: Per-model learning curves …")
# Order: Hybrids first, then Baselines (matches training order)
_all_hists = {**hybrid_hists, **baseline_hists}
_all_pals  = {**HPAL, **BPAL}
_model_order = list(_all_hists.keys())   # 4 models
_n_models = len(_model_order)

fig, axes = plt.subplots(_n_models, 3,
                         figsize=(15, _n_models * 3.5),
                         squeeze=False)
fig.suptitle("Learning Curves — All Models",
             fontsize=13, fontweight='bold', y=1.01)

for _row, m_name in enumerate(_model_order):
    h   = _all_hists[m_name]
    col = _all_pals.get(m_name, '#555555')
    ep  = list(range(1, len(h['tr'])+1))

    # (a) Train Loss
    axes[_row][0].plot(ep, h['tr'], color=col, lw=2.0)
    axes[_row][0].set_ylabel("Loss"); axes[_row][0].grid(alpha=0.3, ls='--')
    if _row == 0:
        axes[_row][0].set_title("(a) Train Loss", fontweight='bold')
    axes[_row][0].set_ylabel(f"{m_name}\nLoss", fontsize=8)

    # (b) Val Loss
    axes[_row][1].plot(ep, h['vl'], color=col, lw=2.0)
    axes[_row][1].grid(alpha=0.3, ls='--')
    if _row == 0:
        axes[_row][1].set_title("(b) Val Loss", fontweight='bold')

    # (c) Val Acc + MacF1
    axes[_row][2].plot(ep, [a*100 for a in h['va']], color=col,
                       lw=2.0, label='Val Acc %')
    if h.get('vf1'):
        axes[_row][2].plot(ep, [f*100 for f in h['vf1']], color=col,
                           lw=1.5, ls='--', alpha=0.75, label='MacF1 %')
    axes[_row][2].legend(fontsize=7, loc='lower right')
    axes[_row][2].grid(alpha=0.3, ls='--')
    if _row == 0:
        axes[_row][2].set_title("(c) Val Acc & Macro-F1 (%)", fontweight='bold')

    # x-label on last row only
    if _row == _n_models - 1:
        for _ax in axes[_row]:
            _ax.set_xlabel("Epoch")

plt.tight_layout()
save_fig(fig, "viz03_learning_curves", "Learning Curves — All Models")

# ════════════════════════════════════════════════════════════
# VIZ-4  Confusion Matrix — Winner model
# ════════════════════════════════════════════════════════════
print("\n🔲 VIZ-4: Confusion matrix …")
cs = [IDX_TO_CLASS[i].replace('___','\n').replace('_',' ')
      for i in range(NUM_CLASSES)]
cm   = confusion_matrix(test_true, test_preds, labels=list(range(NUM_CLASSES)))
cmp  = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1) * 100
fig, ax = plt.subplots(figsize=(max(14, NUM_CLASSES*0.55),
                                 max(12, NUM_CLASSES*0.48)))
sns.heatmap(cmp, annot=True, fmt='.1f', cmap='Blues', ax=ax,
            xticklabels=cs, yticklabels=cs,
            linewidths=0.2, linecolor='lightgrey',
            cbar_kws={'label':'Row-norm (%)','shrink':0.8},
            annot_kws={'size':5.5})
ax.set_title(
    f"Confusion Matrix: {winner_name}  "
    f"Acc={m_win['acc']*100:.2f}%  MacF1={m_win['mf1']*100:.2f}%",
    fontweight='bold', pad=10, fontsize=10)
ax.set_ylabel("True"); ax.set_xlabel("Predicted")
ax.tick_params(axis='x', rotation=45, labelsize=5.5)
ax.tick_params(axis='y', rotation=0,  labelsize=5.5)
plt.tight_layout()
save_fig(fig, "viz04_confusion", "VIZ-4 — Confusion Matrix")

# ════════════════════════════════════════════════════════════
# VIZ-5  Per-Class F1 Bar Chart
# ════════════════════════════════════════════════════════════
print("\n📊 VIZ-5: Per-class F1 …")
per_f1 = f1_score(test_true, test_preds, average=None,
                  zero_division=0, labels=list(range(NUM_CLASSES)))
cls_fl2 = [IDX_TO_CLASS[i].replace('___',' ').replace('_',' ')
           for i in range(NUM_CLASSES)]
bc2 = [PAL['pos'] if f >= 0.85 else PAL['neg'] for f in per_f1]
fig, ax = plt.subplots(figsize=(12, max(8, NUM_CLASSES*0.32)))
bars = ax.barh(cls_fl2, per_f1*100, color=bc2,
               edgecolor='white', linewidth=0.3)
ax.axvline(85, color='navy', lw=1.3, ls='--', label='85% threshold')
ax.set_xlim([0, 110]); ax.set_xlabel("F1-Score (%)")
ax.set_title(f"Per-Class F1-Score ({winner_name})", fontweight='bold')
ax.legend(fontsize=9); ax.grid(axis='x', alpha=0.3, ls='--')
for bar, val in zip(bars, per_f1):
    ax.text(val*100+0.5, bar.get_y()+bar.get_height()/2,
            f"{val*100:.1f}", va='center', fontsize=6.5)
plt.tight_layout()
save_fig(fig, "viz05_per_class_f1", "VIZ-5 — Per-Class F1")

# ════════════════════════════════════════════════════════════
# VIZ-6  ROC Curves — per-class (left) + Macro/Micro avg (right)
# ════════════════════════════════════════════════════════════
print("\n📈 VIZ-6: Per-class ROC curves …")
yb = label_binarize(test_true, classes=list(range(NUM_CLASSES)))
try:
    cmap_roc = matplotlib.colormaps.get_cmap('tab20').resampled(NUM_CLASSES)
except AttributeError:
    cmap_roc = plt.cm.get_cmap('tab20', NUM_CLASSES)
roc_aucs = {}
for i in range(NUM_CLASSES):
    if yb[:, i].sum() == 0:
        continue
    try:
        fpr_i, tpr_i, _ = roc_curve(yb[:, i], test_probs[:, i])
        roc_aucs[i] = (fpr_i, tpr_i, sk_auc(fpr_i, tpr_i))
    except Exception:
        pass

# micro-average
try:
    fpr_micro, tpr_micro, _ = roc_curve(yb.ravel(), test_probs.ravel())
    auc_micro = sk_auc(fpr_micro, tpr_micro)
except Exception as _e:
    print(f"  ⚠️ micro-ROC failed: {_e}")
    fpr_micro, tpr_micro, auc_micro = np.array([0,1]), np.array([0,1]), 0.5

# macro-average
if roc_aucs:
    _all_fpr = np.unique(np.concatenate([v[0] for v in roc_aucs.values()]))
    _mean_tpr = np.zeros_like(_all_fpr)
    for _fpr_i, _tpr_i, _ in roc_aucs.values():
        _mean_tpr += np.interp(_all_fpr, _fpr_i, _tpr_i)
    _mean_tpr /= len(roc_aucs)
    auc_macro = sk_auc(_all_fpr, _mean_tpr)
else:
    _all_fpr = np.array([0, 1]); _mean_tpr = np.array([0, 1]); auc_macro = 0.5

print("\n📈 VIZ-7: Macro/Micro-average ROC …")
fig, (ax_pc, ax_avg) = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle(f"ROC Curves — {winner_name}",
             fontsize=13, fontweight='bold')

# ── left: per-class ──────────────────────────────────────────
for i, (fpr_i, tpr_i, auc_i) in roc_aucs.items():
    _short = IDX_TO_CLASS[i].split('___')[1].replace('_', ' ')
    ax_pc.plot(fpr_i, tpr_i, lw=1.0, color=cmap_roc(i),
               label=f"{_short} ({auc_i:.3f})")
ax_pc.plot([0,1],[0,1],'k--',lw=0.8)
ax_pc.set_xlabel("False Positive Rate")
ax_pc.set_ylabel("True Positive Rate")
ax_pc.set_title("Per-Class ROC Curves", fontweight='bold')
ax_pc.legend(fontsize=5.5, ncol=2, loc='lower right')
ax_pc.grid(alpha=0.25, ls='--')

# ── right: macro / micro average ─────────────────────────────
ax_avg.plot(fpr_micro, tpr_micro, color='#e74c3c', lw=2.2,
            label=f'Micro-avg (AUC = {auc_micro:.4f})')
ax_avg.plot(_all_fpr, _mean_tpr, color='#2166ac', lw=2.2, ls='--',
            label=f'Macro-avg (AUC = {auc_macro:.4f})')
ax_avg.plot([0,1],[0,1],'k--',lw=0.8, label='Random')
ax_avg.fill_between(fpr_micro, tpr_micro, alpha=0.08, color='#e74c3c')
ax_avg.fill_between(_all_fpr,  _mean_tpr, alpha=0.08, color='#2166ac')
ax_avg.set_xlabel("False Positive Rate")
ax_avg.set_ylabel("True Positive Rate")
ax_avg.set_title("Macro / Micro-Average ROC", fontweight='bold')
ax_avg.legend(fontsize=9); ax_avg.grid(alpha=0.25, ls='--')

plt.tight_layout()
save_fig(fig, "viz06_roc_combined", "ROC Curves (Per-Class & Average)")

# ════════════════════════════════════════════════════════════
# VIZ-8  Crop-Level Mean F1
# ════════════════════════════════════════════════════════════
# VIZ-8 — use only kept crops (dropped if only-healthy)
_viz8_crops = sorted(set(c.split('___')[0] for c in UNIFIED_CLASSES))
crop_f1 = {}
for crop in _viz8_crops:
    idxs = [LABEL_MAP[c] for c in UNIFIED_CLASSES if c.startswith(crop)]
    vals = [per_f1[i] for i in idxs]
    crop_f1[crop] = np.mean(vals)*100 if vals else 0.0
fig, ax = plt.subplots(figsize=(max(8, len(_viz8_crops)*1.3), 4.5))
f1v = list(crop_f1.values())
bc3 = [PAL['pos'] if v >= 80 else PAL['neg'] for v in f1v]
bars = ax.bar(range(len(_viz8_crops)), f1v, color=bc3,
              edgecolor='white', linewidth=0.8)
ax.set_xticks(range(len(_viz8_crops)))
ax.set_xticklabels(_viz8_crops, fontsize=10)
ax.axhline(80, color='navy', lw=1.5, ls='--', label='80% target')
ax.set_ylim([0, 110]); ax.set_ylabel("Mean F1 (%)")
ax.set_title(f"Crop-Level Mean F1 ({len(_viz8_crops)} Crops)", fontweight='bold')
ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.3, ls='--')
for bar, val in zip(bars, f1v):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
            f"{val:.1f}%", ha='center', va='bottom',
            fontsize=9, fontweight='bold')
plt.tight_layout()
save_fig(fig, "viz08_crop_f1", "VIZ-8 — Crop F1")

# ════════════════════════════════════════════════════════════
# VIZ-9  XAI — Grad-CAM on winner model (16 random test imgs)
# ════════════════════════════════════════════════════════════
print("\n🔬 VIZ-9: XAI Grad-CAM …")

def _get_gradcam_target_layer(model):
    """Return the last conv-like layer for Grad-CAM."""
    # OmniNet-CS / OmniNet-EV  → convnext / effnet last stage
    if hasattr(model, 'convnext'):
        stages = list(model.convnext.stages)
        return stages[-1]
    if hasattr(model, 'effnet'):
        blocks = list(model.effnet.blocks)
        return blocks[-1]
    if hasattr(model, 'backbone'):          # ResNet-50
        return model.backbone[-1]
    return None

def gradcam_heatmap(model, img_tensor, target_class, target_layer):
    """Returns numpy heatmap [H,W] ∈ [0,1]."""
    model.eval()
    feat_maps, grads = [], []

    def fwd_hook(_, __, out):
        feat_maps.append(out.detach())
    def bwd_hook(_, __, grad_out):
        grads.append(grad_out[0].detach())

    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)

    inp = img_tensor.unsqueeze(0).to(DEVICE)
    inp.requires_grad_(True)   # grad needed for backward pass
    out = model(inp)
    model.zero_grad()
    score = out[0, target_class]
    score.backward()

    h1.remove(); h2.remove()

    if not feat_maps or not grads:
        return np.zeros((IMG_SIZE, IMG_SIZE))

    fmap = feat_maps[0]       # (1, C, H, W) or (1, L, C)
    grad = grads[0]

    # handle transformer outputs (3-D)
    if fmap.dim() == 3:
        weights = grad.mean(dim=1, keepdim=True)   # (1,1,C)
        cam = (weights * fmap).sum(-1).squeeze()   # (L,)
        side = int(cam.numel() ** 0.5)
        cam = cam[:side*side].reshape(side, side)
    else:
        weights = grad.mean(dim=(2, 3), keepdim=True)
        cam = (weights * fmap).sum(dim=1).squeeze(0)  # keep at least 2D: (H,W)
        if cam.dim() == 0:   # scalar edge-case (1x1 spatial)
            cam = cam.unsqueeze(0).unsqueeze(0)
        elif cam.dim() == 1:  # (H,) edge-case
            cam = cam.unsqueeze(0)

    cam = cam.cpu().float().numpy()
    cam = np.maximum(cam, 0)
    if cam.max() > 0:
        cam /= cam.max()
    import cv2 as _cv2
    cam = _cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
    return cam

# Sample 16 test images
_target_layer = _get_gradcam_target_layer(winner_model)
_xai_ok = _target_layer is not None

if _xai_ok:
    _n_xai = 16
    _xai_idxs = random.sample(range(len(test_paths)), min(_n_xai, len(test_paths)))
    _ds_xai   = CropDS(test_paths, test_labels, tf=val_tf)

    fig, axes = plt.subplots(4, _n_xai//4, figsize=((_n_xai//4)*3.2, 4*3.2),
                             squeeze=False)
    fig.suptitle(f"Grad-CAM Explanations: {winner_name}", fontsize=12,
                 fontweight='bold')

    for _k, _idx in enumerate(_xai_idxs):
        _img_t, _true_lbl = _ds_xai[_idx]
        _pred_lbl = int(winner_model(
            _img_t.unsqueeze(0).to(DEVICE)).argmax(-1).item())
        try:
            _hm = gradcam_heatmap(winner_model, _img_t, _pred_lbl, _target_layer)
        except Exception as _e:
            print(f"   ⚠️ GradCAM failed for sample {_idx}: {_e}")
            _hm = np.zeros((IMG_SIZE, IMG_SIZE))

        _row, _col = _k // (_n_xai//4), _k % (_n_xai//4)
        ax = axes[_row][_col]

        _rgb = _denorm(_img_t)
        import matplotlib.cm as _mcm
        _overlay = (_mcm.jet(_hm)[..., :3] * 255).astype(np.uint8)
        _blended = (_rgb * 0.55 + _overlay * 0.45).astype(np.uint8)
        ax.imshow(_blended)

        _true_nm  = IDX_TO_CLASS[int(_true_lbl)].split('___')[1].replace('_',' ')
        _pred_nm  = IDX_TO_CLASS[_pred_lbl].split('___')[1].replace('_',' ')
        _color    = 'lime' if _true_lbl == _pred_lbl else 'red'
        ax.set_title(f"T:{_true_nm}\nP:{_pred_nm}",
                     fontsize=5.5, color=_color, pad=2)
        ax.axis('off')

    plt.tight_layout()
    save_fig(fig, "viz09_gradcam_xai", "VIZ-9 — Grad-CAM XAI")
else:
    print("  ⚠️  Grad-CAM skipped — could not resolve target layer for this model.")

# ════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ════════════════════════════════════════════════════════════
print("\n" + "═"*65)
print("📋  COMPLETE — OmniCrops (4 models)")
print("═"*65)
print(f"  Classes : {NUM_CLASSES} | Crops: {len(_kept_crops)}")
print(f"  Models  : ResNet-50 | ViT-Base | OmniNet-CS | OmniNet-EV")
print(f"  Train   : exact {N_TRAIN}/class (aug-padded where needed)")
print(f"  Val/Test: real images only ({N_VAL}/{N_TEST} target, actual varies)")
print(f"  Aug-padded train: {total_aug_tr:,}/{NUM_CLASSES*N_TRAIN:,} "
      f"= {total_aug_tr/max(NUM_CLASSES*N_TRAIN,1)*100:.1f}%")

print(f"\n🏆  Winner : {winner_name}")
print(f"    Accuracy : {m_win['acc']*100:.3f}%")
print(f"    Macro F1 : {m_win['mf1']*100:.3f}%")
print(f"    AUC-ROC  : {m_win['auc']:.4f}")
print(f"    Inf time : {hybrid_inf[winner_name]:.1f} ms/img")
print("═"*65)