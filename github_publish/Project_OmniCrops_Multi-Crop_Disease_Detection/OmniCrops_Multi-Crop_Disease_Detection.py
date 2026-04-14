# OmniCrops: A Vision Transformer Based Robust Multi-Crop Disease Detection in Real-World Agricultural Environments

# =============================================================================
# OmniCrops: combine 3 sources → balanced dataset → SwinV2-B+FPN → test + metrics
# Flow: A) build dataset  B) train  C) TTA test, confusion matrix, ROC, report
# =============================================================================

# --- Imports ---
import os, sys, shutil, random, json, math, warnings, time, copy
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from torchvision.models import swin_v2_b, Swin_V2_B_Weights

import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

warnings.filterwarnings("ignore")
random.seed(42)
np.random.seed(42)

print("✅ Imports OK")
print(f"   Python  : {sys.version.split()[0]}")
print(f"   Pillow  : {Image.__version__}")
print(f"   PyTorch : {torch.__version__}")


# === A. Build dataset (paths → maps → scan → augment → split → save) ===

# Paths (Kaggle layout; set OMNICROPS_OUT locally)

# ── PlantDoc ──────────────────────────────────────────────────
PLANTDOC_TRAIN = "/kaggle/input/datasets/abdulhasibuddin/plant-doc-dataset/PlantDoc-Dataset/train"
PLANTDOC_TEST  = "/kaggle/input/datasets/abdulhasibuddin/plant-doc-dataset/PlantDoc-Dataset/test"

# ── PlantVillage (colour only — highest quality variant) ─────
PLANTVILLAGE   = "/kaggle/input/datasets/abdallahalidev/plantvillage-dataset/color"

# ── Rice Leaf Diseases ────────────────────────────────────────
RICE_AUG  = "/kaggle/input/datasets/loki4514/rice-leaf-diseases-detection/Rice_Leaf_AUG/Rice_Leaf_AUG"
RICE_TRAIN = "/kaggle/input/datasets/loki4514/rice-leaf-diseases-detection/Rice_Leaf_Diease/Rice_Leaf_Diease/train"
RICE_TEST  = "/kaggle/input/datasets/loki4514/rice-leaf-diseases-detection/Rice_Leaf_Diease/Rice_Leaf_Diease/test"

# ── Output ────────────────────────────────────────────────────
if Path("/kaggle/working").exists():
    _default_out = Path("/kaggle/working/OmniCrops")
else:
    try:
        _here = Path(__file__).resolve().parent
    except NameError:
        _here = Path.cwd()
    _default_out = _here / "output" / "OmniCrops"
OUT_ROOT = Path(os.environ.get("OMNICROPS_OUT", str(_default_out)))
OUT_ROOT.mkdir(parents=True, exist_ok=True)
FIG_DIR = OUT_ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def save_fig_notebook(fig, path, dpi=120, **kwargs):
    """
    Save to disk (Kaggle Output / local) and show in notebook cell output in order.
    Set OMNICROPS_NO_INLINE_FIGS=1 for headless runs (save only, no plt.show).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    kwargs.setdefault("bbox_inches", "tight")
    fig.savefig(path, dpi=dpi, **kwargs)
    if os.environ.get("OMNICROPS_NO_INLINE_FIGS", "").lower() not in ("1", "true", "yes"):
        plt.show()
    plt.close(fig)


N_TRAIN = 1000
N_VAL = 100
N_TEST = 200
TARGET_PER_CLASS = N_TRAIN + N_VAL + N_TEST  # 1300
IMG_SIZE = 224
JPEG_QUALITY = 92
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

print("✅ Paths configured")
print(f"   OUT_ROOT : {OUT_ROOT}")


# Class name normaliser (raw folders → unified labels)
# Raw folder names → clean unified class names used across all datasets

PLANTDOC_MAP = {
    "Apple Scab Leaf"           : "Apple___Scab",
    "Apple leaf"                : "Apple___Healthy",
    "Apple rust leaf"           : "Apple___Cedar_Rust",
    "Bell_pepper leaf"          : "BellPepper___Healthy",
    "Bell_pepper leaf spot"     : "BellPepper___Bacterial_Spot",
    "Blueberry leaf"            : "Blueberry___Healthy",
    "Cherry leaf"               : "Cherry___Healthy",
    "Corn Gray leaf spot"       : "Corn___Gray_Leaf_Spot",
    "Corn leaf blight"          : "Corn___Northern_Leaf_Blight",
    "Corn rust leaf"            : "Corn___Common_Rust",
    "Peach leaf"                : "Peach___Healthy",
    "Potato leaf early blight"  : "Potato___Early_Blight",
    "Potato leaf late blight"   : "Potato___Late_Blight",
    "Raspberry leaf"            : "Raspberry___Healthy",
    "Soyabean leaf"             : "Soybean___Healthy",
    "Squash Powdery mildew leaf": "Squash___Powdery_Mildew",
    "Strawberry leaf"           : "Strawberry___Healthy",
    "Tomato Early blight leaf"  : "Tomato___Early_Blight",
    "Tomato Septoria leaf spot" : "Tomato___Septoria_Leaf_Spot",
    "Tomato leaf"               : "Tomato___Healthy",
    "Tomato leaf bacterial spot": "Tomato___Bacterial_Spot",
    "Tomato leaf late blight"   : "Tomato___Late_Blight",
    "Tomato leaf mosaic virus"  : "Tomato___Mosaic_Virus",
    "Tomato leaf yellow virus"  : "Tomato___Yellow_Leaf_Curl_Virus",
    "Tomato mold leaf"          : "Tomato___Leaf_Mold",
    "grape leaf"                : "Grape___Healthy",
    "grape leaf black rot"      : "Grape___Black_Rot",
}

PLANTVILLAGE_MAP = {
    "Apple___Apple_scab"                               : "Apple___Scab",
    "Apple___Black_rot"                                : "Apple___Black_Rot",
    "Apple___Cedar_apple_rust"                         : "Apple___Cedar_Rust",
    "Apple___healthy"                                  : "Apple___Healthy",
    "Blueberry___healthy"                              : "Blueberry___Healthy",
    "Cherry_(including_sour)___Powdery_mildew"         : "Cherry___Powdery_Mildew",
    "Cherry_(including_sour)___healthy"                : "Cherry___Healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "Corn___Gray_Leaf_Spot",
    "Corn_(maize)___Common_rust_"                      : "Corn___Common_Rust",
    "Corn_(maize)___Northern_Leaf_Blight"              : "Corn___Northern_Leaf_Blight",
    "Corn_(maize)___healthy"                           : "Corn___Healthy",
    "Grape___Black_rot"                                : "Grape___Black_Rot",
    "Grape___Esca_(Black_Measles)"                     : "Grape___Esca_Black_Measles",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)"       : "Grape___Leaf_Blight",
    "Grape___healthy"                                  : "Grape___Healthy",
    "Orange___Haunglongbing_(Citrus_greening)"         : "Orange___Citrus_Greening",
    "Peach___Bacterial_spot"                           : "Peach___Bacterial_Spot",
    "Peach___healthy"                                  : "Peach___Healthy",
    "Pepper,_bell___Bacterial_spot"                    : "BellPepper___Bacterial_Spot",
    "Pepper,_bell___healthy"                           : "BellPepper___Healthy",
    "Potato___Early_blight"                            : "Potato___Early_Blight",
    "Potato___Late_blight"                             : "Potato___Late_Blight",
    "Potato___healthy"                                 : "Potato___Healthy",
    "Raspberry___healthy"                              : "Raspberry___Healthy",
    "Soybean___healthy"                                : "Soybean___Healthy",
    "Squash___Powdery_mildew"                          : "Squash___Powdery_Mildew",
    "Strawberry___Leaf_scorch"                         : "Strawberry___Leaf_Scorch",
    "Strawberry___healthy"                             : "Strawberry___Healthy",
    "Tomato___Bacterial_spot"                          : "Tomato___Bacterial_Spot",
    "Tomato___Early_blight"                            : "Tomato___Early_Blight",
    "Tomato___Late_blight"                             : "Tomato___Late_Blight",
    "Tomato___Leaf_Mold"                               : "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot"                      : "Tomato___Septoria_Leaf_Spot",
    "Tomato___Spider_mites Two-spotted_spider_mite"    : "Tomato___Spider_Mites",
    "Tomato___Target_Spot"                             : "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus"           : "Tomato___Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus"                     : "Tomato___Mosaic_Virus",
    "Tomato___healthy"                                 : "Tomato___Healthy",
}

RICE_MAP = {
    # AUG folder names
    "Bacterial Leaf Blight"  : "Rice___Bacterial_Leaf_Blight",
    "Brown Spot"             : "Rice___Brown_Spot",
    "Healthy Rice Leaf"      : "Rice___Healthy",
    "Leaf Blast"             : "Rice___Leaf_Blast",
    "Leaf scald"             : "Rice___Leaf_Scald",
    "Narrow Brown Leaf Spot" : "Rice___Narrow_Brown_Leaf_Spot",
    "Neck_Blast"             : "Rice___Neck_Blast",
    "Rice Hispa"             : "Rice___Hispa",
    "Sheath Blight"          : "Rice___Sheath_Blight",
    # train/test subfolder names (lowercase variants)
    "bacterial_leaf_blight"  : "Rice___Bacterial_Leaf_Blight",
    "brown_spot"             : "Rice___Brown_Spot",
    "healthy"                : "Rice___Healthy",
    "leaf_blast"             : "Rice___Leaf_Blast",
    "leaf_scald"             : "Rice___Leaf_Scald",
    "narrow_brown_spot"      : "Rice___Narrow_Brown_Leaf_Spot",
    "neck_blast"             : "Rice___Neck_Blast",
    "rice_hispa"             : "Rice___Hispa",
    "sheath_blight"          : "Rice___Sheath_Blight",
    "Tungro"                 : "Rice___Tungro",
    "tungro"                 : "Rice___Tungro",
}

print(f"✅ Class maps ready")
print(f"   PlantDoc    : {len(PLANTDOC_MAP)} classes")
print(f"   PlantVillage: {len(PLANTVILLAGE_MAP)} classes")
print(f"   Rice        : {len(set(RICE_MAP.values()))} classes")


# ============================================================
# Scan three sources → raw inventory
# ============================================================

def scan_folder(root, name_map, source_tag):
    """
    Walk a folder where each subdirectory = one class.
    Returns dict: unified_class_name → list of absolute file paths
    """
    inventory = defaultdict(list)
    root = Path(root)
    if not root.exists():
        print(f"  ⚠️  Not found: {root}"); return inventory
    for cls_dir in sorted(root.iterdir()):
        if not cls_dir.is_dir(): continue
        raw = cls_dir.name
        unified = name_map.get(raw)
        if unified is None:
            # Try case-insensitive match
            for k, v in name_map.items():
                if k.lower() == raw.lower():
                    unified = v; break
        if unified is None:
            print(f"  ⚠️  [{source_tag}] unmapped class: '{raw}'")
            continue
        for fp in sorted(cls_dir.iterdir()):
            if fp.suffix.lower() in IMG_EXTS:
                inventory[unified].append(str(fp))
    return inventory

def merge_inventories(*inventories):
    merged = defaultdict(list)
    for inv in inventories:
        for cls, paths in inv.items():
            merged[cls].extend(paths)
    return merged

print("📂 Scanning datasets...")

# PlantDoc (train + test combined into raw pool)
inv_pd_train = scan_folder(PLANTDOC_TRAIN, PLANTDOC_MAP, "PlantDoc-train")
inv_pd_test  = scan_folder(PLANTDOC_TEST,  PLANTDOC_MAP, "PlantDoc-test")

# PlantVillage (colour only)
inv_pv       = scan_folder(PLANTVILLAGE,   PLANTVILLAGE_MAP, "PlantVillage")

# Rice (AUG + train + test all merged as raw pool)
inv_rice_aug   = scan_folder(RICE_AUG,   RICE_MAP, "Rice-AUG")
inv_rice_train = scan_folder(RICE_TRAIN, RICE_MAP, "Rice-train")
inv_rice_test  = scan_folder(RICE_TEST,  RICE_MAP, "Rice-test")

# Merge everything into one raw pool
raw_pool = merge_inventories(
    inv_pd_train, inv_pd_test,
    inv_pv,
    inv_rice_aug, inv_rice_train, inv_rice_test
)

# Shuffle each class list for reproducibility
for cls in raw_pool:
    random.shuffle(raw_pool[cls])

print(f"\n✅ Raw pool assembled: {len(raw_pool)} unique classes")
print(f"   Total images available: {sum(len(v) for v in raw_pool.values()):,}")


# ============================================================
# Raw distribution (before balance)
# ============================================================

raw_counts = {cls: len(paths) for cls, paths in sorted(raw_pool.items())}

print(f"\n{'Class':<45} {'Raw Count':>10} {'Source?':>8}")
print("─" * 65)
for cls, cnt in sorted(raw_counts.items(), key=lambda x: -x[1]):
    tag = "RICE" if cls.startswith("Rice") else "PLANT"
    print(f"  {cls:<43} {cnt:>8,}  {tag}")

print(f"\n  Total classes : {len(raw_counts)}")
print(f"  Max class     : {max(raw_counts.values()):,}")
print(f"  Min class     : {min(raw_counts.values()):,}")
print(f"  Mean          : {np.mean(list(raw_counts.values())):.0f}")
print(f"  Median        : {np.median(list(raw_counts.values())):.0f}")
print(f"  Classes < {TARGET_PER_CLASS}: {sum(1 for v in raw_counts.values() if v < TARGET_PER_CLASS)}")
print(f"  Classes ≥ {TARGET_PER_CLASS}: {sum(1 for v in raw_counts.values() if v >= TARGET_PER_CLASS)}")


# ============================================================
# Augmentation helpers
# ============================================================

def _u01(i: int) -> float:
    """Deterministic pseudo-random in [0, 1) from integer index."""
    x = math.sin(i * 12.9898 + 78.233) * 43758.5453
    return x - math.floor(x)


def augment_image(img: Image.Image, aug_id: int) -> Image.Image:
    """Indexed transforms; color/blur strengths depend on aug_id (reproducible)."""
    b = 0.6 + 0.8 * _u01(aug_id * 3 + 1)
    c = 0.6 + 0.8 * _u01(aug_id * 5 + 2)
    s = 0.5 + 1.0 * _u01(aug_id * 7 + 3)
    sh = 0.0 + 2.0 * _u01(aug_id * 11 + 4)
    blur_r = 0.5 + 1.0 * _u01(aug_id * 13 + 5)

    ops = [
        lambda i: i.transpose(Image.FLIP_LEFT_RIGHT),
        lambda i: i.transpose(Image.FLIP_TOP_BOTTOM),
        lambda i: i.rotate(15, expand=False, fillcolor=(0, 0, 0)),
        lambda i: i.rotate(-15, expand=False, fillcolor=(0, 0, 0)),
        lambda i: i.rotate(30, expand=False, fillcolor=(0, 0, 0)),
        lambda i: i.rotate(-30, expand=False, fillcolor=(0, 0, 0)),
        lambda i: ImageEnhance.Brightness(i).enhance(b),
        lambda i: ImageEnhance.Contrast(i).enhance(c),
        lambda i: ImageEnhance.Color(i).enhance(s),
        lambda i: ImageEnhance.Sharpness(i).enhance(sh),
        lambda i: i.filter(ImageFilter.GaussianBlur(radius=blur_r)),
        lambda i: ImageOps.autocontrast(i),
    ]
    try:
        return ops[aug_id % len(ops)](img)
    except Exception:
        return img


def load_and_resize(path: str, size: int = None) -> Image.Image:
    sz = size if size is not None else IMG_SIZE
    return Image.open(path).convert("RGB").resize((sz, sz), Image.LANCZOS)


def save_uniform_jpeg(src_path: str, dst_path: Path) -> bool:
    try:
        load_and_resize(src_path).save(dst_path, "JPEG", quality=JPEG_QUALITY)
        return True
    except Exception:
        return False


print("✅ Augmentation functions ready (12 transforms, deterministic per index)")



# ============================================================
# Build balanced JPEG dataset (TARGET_PER_CLASS / class)
# ============================================================

print(f"\n🔧 Building balanced dataset — target: {TARGET_PER_CLASS}/class "
      f"({N_TRAIN} train + {N_VAL} val + {N_TEST} test)")
print("=" * 65)

aug_report = {}
BALANCED_ROOT = OUT_ROOT / "balanced_raw"

for cls in tqdm(sorted(raw_pool.keys()), desc="Balancing classes"):
    paths = raw_pool[cls].copy()
    random.shuffle(paths)
    n_avail = len(paths)
    cls_dir = BALANCED_ROOT / cls
    cls_dir.mkdir(parents=True, exist_ok=True)
    written = 0

    if n_avail >= TARGET_PER_CLASS:
        for src in paths[:TARGET_PER_CLASS]:
            if save_uniform_jpeg(src, cls_dir / f"img_{written:05d}.jpg"):
                written += 1
        k = 0
        while written < TARGET_PER_CLASS and k < n_avail * 20:
            src = paths[k % n_avail]
            if save_uniform_jpeg(src, cls_dir / f"img_{written:05d}.jpg"):
                written += 1
            k += 1
        aug_report[cls] = {
            "raw_available": n_avail,
            "augmented": 0,
            "total": written,
            "status": "capped",
        }
    else:
        for src in paths:
            if save_uniform_jpeg(src, cls_dir / f"img_{written:05d}.jpg"):
                written += 1
        n_saved_orig = written
        aug_id = 0
        src_idx = 0
        while written < TARGET_PER_CLASS:
            src = paths[src_idx % len(paths)]
            try:
                img = load_and_resize(src)
                aug_img = augment_image(img, aug_id)
                aug_img.save(cls_dir / f"img_{written:05d}.jpg", "JPEG", quality=JPEG_QUALITY)
                written += 1
            except Exception:
                pass
            aug_id += 1
            src_idx += 1
            if aug_id > (TARGET_PER_CLASS - n_saved_orig) * 30 + 500:
                break
        added = max(0, written - n_saved_orig)
        aug_report[cls] = {
            "raw_available": n_avail,
            "augmented": added,
            "total": written,
            "status": f"augmented (+{added})",
        }

    print(f"  {cls:<45} raw={n_avail:>5,} → {aug_report[cls]['status']}")

print(f"\n✅ Balanced dataset built → {BALANCED_ROOT}")
print(f"   Classes   : {len(aug_report)}")
print(f"   Total imgs: {sum(r['total'] for r in aug_report.values()):,}")


# ============================================================
# Train / val / test split per class
# ============================================================

print(f"\n📂 Splitting: {N_TRAIN} train / {N_VAL} val / {N_TEST} test per class")

for split in ["train", "val", "test"]:
    (OUT_ROOT / split).mkdir(parents=True, exist_ok=True)

split_report = {}

for cls in tqdm(sorted(aug_report.keys()), desc="Splitting"):
    src_dir = BALANCED_ROOT / cls
    files = [f for f in src_dir.iterdir() if f.suffix.lower() in IMG_EXTS]
    random.shuffle(files)
    files = files[:TARGET_PER_CLASS]
    n = len(files)
    if n >= N_TRAIN + N_VAL + N_TEST:
        n_train, n_val, n_test = N_TRAIN, N_VAL, N_TEST
    else:
        n_train = min(N_TRAIN, max(0, n - N_VAL - N_TEST))
        n_val = min(N_VAL, max(0, n - n_train - N_TEST))
        n_test = n - n_train - n_val

    splits_files = {
        "train": files[:n_train],
        "val": files[n_train:n_train + n_val],
        "test": files[n_train + n_val:n_train + n_val + n_test],
    }

    for split, split_files in splits_files.items():
        cls_out = OUT_ROOT / split / cls
        cls_out.mkdir(parents=True, exist_ok=True)
        for fp in split_files:
            shutil.copy2(fp, cls_out / fp.name)

    split_report[cls] = {"train": n_train, "val": n_val, "test": n_test, "total": n}

print(f"\n✅ Splits complete")
total_train = sum(r["train"] for r in split_report.values())
total_val = sum(r["val"] for r in split_report.values())
total_test = sum(r["test"] for r in split_report.values())
print(f"   Train: {total_train:,} | Val: {total_val:,} | Test: {total_test:,}")
print(f"   Total: {total_train + total_val + total_test:,}")


# metadata.json + dataset.yaml
metadata = {
    "dataset_name": "OmniCrops",
    "sources": ["PlantDoc", "PlantVillage", "Rice Leaf Diseases"],
    "image_size": IMG_SIZE,
    "target_per_class": TARGET_PER_CLASS,
    "split_counts_per_class": {"train": N_TRAIN, "val": N_VAL, "test": N_TEST},
    "num_classes": len(aug_report),
    "total_images": total_train + total_val + total_test,
    "train_images": total_train,
    "val_images": total_val,
    "test_images": total_test,
    "classes": sorted(aug_report.keys()),
    "augmentation_report": aug_report,
    "split_report": split_report,
}

with open(OUT_ROOT / "metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print(f"✅ Metadata saved → {OUT_ROOT / 'metadata.json'}")

class_list = sorted(aug_report.keys())
yaml_content = f"""# OmniCrops — Auto-generated dataset config
path  : {str(OUT_ROOT)}
train : train
val   : val
test  : test

nc    : {len(class_list)}
names : {json.dumps({i: c for i, c in enumerate(class_list)}, indent=2)}
"""

with open(OUT_ROOT / "dataset.yaml", "w") as f:
    f.write(yaml_content)
print(f"✅ dataset.yaml saved → {OUT_ROOT / 'dataset.yaml'}")


# One dataset figure: raw vs balanced (before | after)
classes_sorted_orig = sorted(aug_report.keys(), key=lambda c: raw_counts[c])
orig_before = [raw_counts[c] for c in classes_sorted_orig]
total_after = [aug_report[c]["total"] for c in classes_sorted_orig]
x = np.arange(len(classes_sorted_orig))
_h = max(8.0, len(classes_sorted_orig) * 0.11)
fig_ds, (axb, axa) = plt.subplots(1, 2, figsize=(18, _h))
colors_before = ["#e74c3c" if v < TARGET_PER_CLASS else "#27ae60" for v in orig_before]
axb.barh(x, orig_before, color=colors_before, edgecolor="white", linewidth=0.5)
axb.axvline(TARGET_PER_CLASS, color="navy", ls="--", lw=2, label=f"target {TARGET_PER_CLASS}/class")
axb.set_yticks(x)
axb.set_yticklabels(classes_sorted_orig, fontsize=6)
axb.set_title("Before — raw counts / class", fontsize=11, fontweight="bold")
axb.set_xlabel("Count")
axb.legend()
axb.grid(axis="x", alpha=0.3)
axa.barh(x, total_after, color="#2ecc71", edgecolor="white", linewidth=0.5)
axa.axvline(TARGET_PER_CLASS, color="navy", ls="--", lw=2, label=f"target {TARGET_PER_CLASS}/class")
axa.set_yticks(x)
axa.set_yticklabels(classes_sorted_orig, fontsize=6)
axa.set_title("After — balanced pool", fontsize=11, fontweight="bold")
axa.set_xlabel("Count")
axa.legend()
axa.grid(axis="x", alpha=0.3)
plt.suptitle("OmniCrops — dataset balance", fontsize=12, fontweight="bold")
plt.tight_layout()
p_dataset = FIG_DIR / "dataset_overview.png"
print("\n▶ Figure (dataset): dataset_overview — before | after")
save_fig_notebook(fig_ds, p_dataset, dpi=120)
print(f"✅ Saved → {p_dataset}")

# Dataset build summary
print("\n" + "=" * 70)
print("🌿  OmniCrops dataset build — COMPLETE")
print("=" * 70)
print(f"  Location     : {OUT_ROOT}")
print(f"  Classes      : {len(class_list)}")
print(f"  Per class    : {TARGET_PER_CLASS} (pool) → {N_TRAIN}/{N_VAL}/{N_TEST} train/val/test")
print(f"  Total images : {total_train + total_val + total_test:,}")
print(f"  Train        : {total_train:,} | Val: {total_val:,} | Test: {total_test:,}")
print("─" * 70)
print(f"  Augmented classes  : {sum(1 for d in aug_report.values() if d['augmented'] > 0)}")
print(f"  Capped classes     : {sum(1 for d in aug_report.values() if d['status'] == 'capped')}")
print("─" * 70)
print("  Artifacts:")
for p in [p_dataset, OUT_ROOT / "metadata.json", OUT_ROOT / "dataset.yaml"]:
    ok = "✓" if Path(p).exists() else "✗"
    print(f"    {ok} {p}")
print("=" * 70)


# === B. Train SwinV2-B + FPN ===

# Device, hyperparameters, reproducibility
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n🚀 Device : {DEVICE}")
if DEVICE.type == "cuda":
    print(f"   GPU    : {torch.cuda.get_device_name(0)}")
    print(f"   VRAM   : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

# ── Hyperparameters ───────────────────────────────────────────
BATCH_SIZE    = 32
# Kaggle: many workers + heavy aug can make the *first* batches feel "stuck"; 2 is a sane default.
_kw = 0 if sys.platform.startswith("win") else min(8, (os.cpu_count() or 4))
if Path("/kaggle/working").exists():
    _kw = min(2, _kw)
NUM_WORKERS   = int(os.environ.get("OMNICROPS_NUM_WORKERS", str(_kw)))
EPOCHS        = 50
LR_BACKBONE   = 5e-6    # very low — protect pretrained weights
LR_FPN        = 1e-4    # FPN layers
LR_HEAD       = 5e-4    # classifier head
WARMUP_EP     = 3
PATIENCE      = 10
LABEL_SMOOTH  = 0.05
MIXUP_ALPHA   = 0.2
CUTMIX_ALPHA  = 1.0
DROPOUT_HEAD  = 0.3
WEIGHT_DECAY  = 1e-4
SEED          = 42
SAVE_PATH     = str(OUT_ROOT / "best_omnicrops_swinv2.pth")

torch.manual_seed(SEED)
if DEVICE.type == "cuda":
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True

USE_AMP = DEVICE.type == "cuda" and os.environ.get("OMNICROPS_NO_AMP", "").lower() not in ("1", "true", "yes")
_grad_scaler = GradScaler(enabled=USE_AMP)
MAX_TRAIN_BATCHES = int(os.environ.get("OMNICROPS_MAX_TRAIN_BATCHES", "0"))

# Class list from dataset prep (already defined above)
NUM_CLASSES = len(class_list)
print(f"   Classes : {NUM_CLASSES}")
print(f"   Batch   : {BATCH_SIZE} | Workers: {NUM_WORKERS} | Epochs: {EPOCHS} | Patience: {PATIENCE}")
print(f"   AMP     : {'ON (fp16/bf16 mix — faster)' if USE_AMP else 'OFF'}")
if MAX_TRAIN_BATCHES > 0:
    print(f"   ⚠️  OMNICROPS_MAX_TRAIN_BATCHES={MAX_TRAIN_BATCHES} (debug smoke run; not full epoch)")


# Transforms (train aug + val + TTA)
MEAN = [0.485, 0.456, 0.406]   # ImageNet stats
STD  = [0.229, 0.224, 0.225]

train_tf = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3,
                           saturation=0.3, hue=0.1),
    transforms.RandAugment(num_ops=2, magnitude=9),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),
])

val_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

# TTA transforms (×5)
tta_tfs = [
    val_tf,
    transforms.Compose([transforms.Resize((224,224)),
                        transforms.RandomHorizontalFlip(p=1.0),
                        transforms.ToTensor(), transforms.Normalize(MEAN,STD)]),
    transforms.Compose([transforms.Resize((256,256)),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(), transforms.Normalize(MEAN,STD)]),
    transforms.Compose([transforms.Resize((224,224)),
                        transforms.RandomRotation((90,90)),
                        transforms.ToTensor(), transforms.Normalize(MEAN,STD)]),
    transforms.Compose([transforms.Resize((224,224)),
                        transforms.RandomRotation((-90,-90)),
                        transforms.ToTensor(), transforms.Normalize(MEAN,STD)]),
]
if os.environ.get("OMNICROPS_FAST_TTA", "").lower() in ("1", "true", "yes"):
    tta_tfs = [val_tf, tta_tfs[1]]
    print(f"✅ Transforms ready | TTA: {len(tta_tfs)}× (OMNICROPS_FAST_TTA)")
else:
    print(f"✅ Transforms ready | TTA: {len(tta_tfs)}×")
USE_TTA = os.environ.get("OMNICROPS_USE_TTA", "1").lower() in ("1", "true", "yes")
print(f"   TTA enabled: {'YES' if USE_TTA else 'NO (standard test eval only)'}")


# ImageFolder-style dataset
class OmniCropsDataset(Dataset):
    def __init__(self, root, classes, tf):
        self.tf      = tf
        self.classes = classes
        self.cls2idx = {c: i for i, c in enumerate(classes)}
        self.samples = []
        for cls in classes:
            d = Path(root) / cls
            if not d.is_dir(): continue
            for fp in d.iterdir():
                if fp.suffix.lower() in {".jpg",".jpeg",".png"}:
                    self.samples.append((str(fp), self.cls2idx[cls]))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224), 0)
        return self.tf(img), label

train_ds = OmniCropsDataset(OUT_ROOT/"train", class_list, train_tf)
val_ds   = OmniCropsDataset(OUT_ROOT/"val",   class_list, val_tf)
test_ds  = OmniCropsDataset(OUT_ROOT/"test",  class_list, val_tf)

# Weighted sampler for class balance
labels_all  = [s[1] for s in train_ds.samples]
cls_cnt     = np.bincount(labels_all, minlength=NUM_CLASSES).astype(float)
cls_wt      = 1.0 / (cls_cnt + 1e-6)
sample_wt   = torch.tensor([cls_wt[l] for l in labels_all], dtype=torch.float)
sampler     = WeightedRandomSampler(sample_wt, len(sample_wt), replacement=True)

_pin = DEVICE.type == "cuda"
train_loader = DataLoader(
    train_ds, BATCH_SIZE, sampler=sampler,
    num_workers=NUM_WORKERS, pin_memory=_pin, drop_last=True,
    persistent_workers=NUM_WORKERS > 0,
)
val_loader = DataLoader(
    val_ds, BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=_pin,
    persistent_workers=NUM_WORKERS > 0,
)
test_loader = DataLoader(
    test_ds, BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=_pin,
    persistent_workers=NUM_WORKERS > 0,
)

_n_bt = len(train_loader)
print(f"✅ Datasets | Train:{len(train_ds):,} Val:{len(val_ds):,} Test:{len(test_ds):,}")
print(f"   Train steps/epoch: {_n_bt} batches (SwinV2-B + RandAugment → often ~0.5–2 s/batch on T4-class GPU)")
print(f"   Expect ~{max(1, _n_bt // 2)}–{_n_bt * 2} s for first epoch wall-clock if GPU is busy; not stuck until tqdm stops moving.")


# Model: SwinV2-B backbone + FPN head
class FPNFusion(nn.Module):
    """
    Lightweight FPN that fuses SwinV2-B stage outputs.
    SwinV2-B feature dims: S1=128, S2=256, S3=512, S4=1024
    Output: fused 512-d global descriptor
    """
    def __init__(self):
        super().__init__()
        # Lateral projections → common 256-d
        self.lat4 = nn.Sequential(nn.Linear(1024, 256), nn.GELU())
        self.lat3 = nn.Sequential(nn.Linear(512,  256), nn.GELU())
        self.lat2 = nn.Sequential(nn.Linear(256,  256), nn.GELU())

        # Top-down fusion refinement
        self.fuse43 = nn.Sequential(nn.Linear(256, 256), nn.GELU())
        self.fuse32 = nn.Sequential(nn.Linear(256, 256), nn.GELU())

        # Final aggregation: [p4‖p3‖p2] → 512
        self.agg = nn.Sequential(
            nn.Linear(768, 512), nn.GELU(),
            nn.Dropout(0.1))

    def forward(self, f2, f3, f4):
        # f2: (B, H2*W2, 256)  f3: (B, H3*W3, 512)  f4: (B, H4*W4, 1024)
        p4 = self.lat4(f4.mean(1))   # (B, 256) — global avg over tokens
        p3 = self.lat3(f3.mean(1))   # (B, 256)
        p2 = self.lat2(f2.mean(1))   # (B, 256)

        # Top-down fusion
        p3 = self.fuse43(p3 + p4)
        p2 = self.fuse32(p2 + p3)

        return self.agg(torch.cat([p4, p3, p2], dim=1))  # (B, 512)


class OmniCropsSwinFPN(nn.Module):
    """
    SwinV2-B backbone + FPN multi-scale fusion + classification head.

    Architecture:
      Input(224×224×3)
        → SwinV2-B (ImageNet-22K pretrained)
            Stage1 → 56×56, 128-d   [not used in FPN]
            Stage2 → 28×28, 256-d   ← FPN P2
            Stage3 → 14×14, 512-d   ← FPN P3
            Stage4 →  7×7,  1024-d  ← FPN P4
        → FPN fusion → 512-d
        → Head: FC(512→256)→GELU→Drop→FC(256→N)
    """
    def __init__(self, num_classes, dropout=DROPOUT_HEAD):
        super().__init__()
        # torchvision ships ImageNet-1K weights; IN-22K init is not exposed here.
        weights = Swin_V2_B_Weights.IMAGENET1K_V1
        backbone = swin_v2_b(weights=weights)

        # Extract stage-wise feature layers
        # SwinV2-B layers: [patch_embed, layers.0, layers.1, layers.2, layers.3, norm, head]
        self.patch_embed = backbone.features[0]   # patch partition
        self.stage1      = backbone.features[1]   # 128-d
        self.downsample1 = backbone.features[2]   # → 256-d
        self.stage2      = backbone.features[3]   # 256-d  ← P2
        self.downsample2 = backbone.features[4]   # → 512-d
        self.stage3      = backbone.features[5]   # 512-d  ← P3
        self.downsample3 = backbone.features[6]   # → 1024-d
        self.stage4      = backbone.features[7]   # 1024-d ← P4
        self.norm        = backbone.norm

        self.fpn  = FPNFusion()
        self.head = nn.Sequential(
            nn.Linear(512, 256), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes))

        self._init_head()

    def _init_head(self):
        for m in [self.fpn, self.head]:
            for layer in m.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.trunc_normal_(layer.weight, std=0.02)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

    def forward(self, x):
        x  = self.patch_embed(x)   # (B, 56, 56, 128)
        x  = self.stage1(x)
        x  = self.downsample1(x)   # (B, 28, 28, 256)
        f2 = self.stage2(x)        # (B, 28, 28, 256) ← P2
        x  = self.downsample2(f2)  # (B, 14, 14, 512)
        f3 = self.stage3(x)        # (B, 14, 14, 512) ← P3
        x  = self.downsample3(f3)  # (B,  7,  7, 1024)
        f4 = self.stage4(x)        # (B,  7,  7, 1024) ← P4

        # Flatten spatial → token sequence for FPN
        f2f = f2.flatten(1, 2)   # (B, 784,  256)
        f3f = f3.flatten(1, 2)   # (B, 196,  512)
        f4f = f4.flatten(1, 2)   # (B,  49, 1024)

        feat = self.fpn(f2f, f3f, f4f)   # (B, 512)
        return self.head(feat)            # (B, N_classes)

    def get_param_groups(self, lr_bb, lr_fpn, lr_head):
        """Separate LRs for backbone / FPN / head."""
        bb_params   = (list(self.patch_embed.parameters()) +
                       list(self.stage1.parameters())      +
                       list(self.downsample1.parameters()) +
                       list(self.stage2.parameters())      +
                       list(self.downsample2.parameters()) +
                       list(self.stage3.parameters())      +
                       list(self.downsample3.parameters()) +
                       list(self.stage4.parameters())      +
                       list(self.norm.parameters()))
        return [
            {"params": bb_params,              "lr": lr_bb,   "name": "backbone"},
            {"params": self.fpn.parameters(),  "lr": lr_fpn,  "name": "fpn"},
            {"params": self.head.parameters(), "lr": lr_head, "name": "head"},
        ]


model  = OmniCropsSwinFPN(NUM_CLASSES).to(DEVICE)
params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"📐 OmniCrops-SwinV2+FPN | Parameters: {params:,}")

# Smoke test
with torch.no_grad():
    _x = torch.randn(2, 3, 224, 224, device=DEVICE)
    _o = model(_x)
    print(f"   Smoke: {list(_x.shape)} → {list(_o.shape)} ✅")
    del _x, _o
    torch.cuda.empty_cache() if DEVICE.type == "cuda" else None


# Loss, optimizer, scheduler
class LabelSmoothCE(nn.Module):
    def __init__(self, classes, smooth=0.05):
        super().__init__()
        self.eps = smooth; self.n = classes
    def forward(self, pred, target):
        lp  = F.log_softmax(pred, dim=1)
        nll = F.nll_loss(lp, target)
        return (1 - self.eps) * nll + self.eps * (-lp.mean())

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__(); self.gamma = gamma
    def forward(self, pred, target):
        ce  = F.cross_entropy(pred, target, reduction='none')
        pt  = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()

class CombinedLoss(nn.Module):
    """LabelSmoothing + Focal — best of both worlds."""
    def __init__(self, classes, smooth=0.05, gamma=2.0, w_focal=0.3):
        super().__init__()
        self.ls    = LabelSmoothCE(classes, smooth)
        self.focal = FocalLoss(gamma)
        self.wf    = w_focal
    def forward(self, pred, target):
        return (1 - self.wf) * self.ls(pred, target) + \
                      self.wf * self.focal(pred, target)

criterion = CombinedLoss(NUM_CLASSES, LABEL_SMOOTH, gamma=2.0, w_focal=0.3)

param_groups = model.get_param_groups(LR_BACKBONE, LR_FPN, LR_HEAD)
optimizer    = optim.AdamW(param_groups, weight_decay=WEIGHT_DECAY,
                           betas=(0.9, 0.999))

def warmup_cosine(ep, warmup=WARMUP_EP, total=EPOCHS):
    if ep < warmup: return (ep + 1) / warmup
    p = (ep - warmup) / max(1, total - warmup)
    return 0.5 * (1 + np.cos(np.pi * p))

scheduler = optim.lr_scheduler.LambdaLR(
    optimizer, lr_lambda=lambda e: warmup_cosine(e))

class EarlyStopping:
    def __init__(self, patience=PATIENCE):
        self.patience   = patience
        self.best_loss  = float('inf')
        self.best_f1    = 0.0
        self.counter    = 0
        self.best_ep    = 0
        self.best_state = None
    def step(self, val_loss, val_f1, epoch, model):
        improved = val_f1 > self.best_f1 + 1e-5
        if improved:
            self.best_loss  = val_loss
            self.best_f1    = val_f1
            self.counter    = 0
            self.best_ep    = epoch
            self.best_state = copy.deepcopy(model.state_dict())
            torch.save(self.best_state, SAVE_PATH)
        else:
            self.counter += 1
        return self.counter >= self.patience

stopper = EarlyStopping(PATIENCE)
print(f"✅ Loss: LabelSmooth+Focal | Optimizer: AdamW | LR: bb={LR_BACKBONE} fpn={LR_FPN} head={LR_HEAD}")


# Mixup / CutMix
def mixup(x, y, alpha=MIXUP_ALPHA):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    return lam*x + (1-lam)*x[idx], y, y[idx], lam

def cutmix(x, y, alpha=CUTMIX_ALPHA):
    lam  = np.random.beta(alpha, alpha)
    idx  = torch.randperm(x.size(0), device=x.device)
    B,C,H,W = x.shape
    cut_rat = np.sqrt(1 - lam)
    cut_h, cut_w = int(H * cut_rat), int(W * cut_rat)
    cx = np.random.randint(W); cy = np.random.randint(H)
    x1 = max(cx - cut_w//2, 0); x2 = min(cx + cut_w//2, W)
    y1 = max(cy - cut_h//2, 0); y2 = min(cy + cut_h//2, H)
    xm = x.clone()
    xm[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
    lam = 1 - (x2-x1)*(y2-y1)/(H*W)
    return xm, y, y[idx], lam

def mixed_loss(crit, pred, ya, yb, lam):
    return lam * crit(pred, ya) + (1-lam) * crit(pred, yb)


# train_epoch / evaluate
def train_epoch(model, loader, crit, opt, epoch, scaler, use_amp):
    """Mixup/CutMix batches skip accuracy (labels no longer match single-class logits)."""
    model.train()
    total_loss = 0.0
    loss_n = 0
    correct = 0
    acc_n = 0

    pbar = tqdm(loader, desc=f"Ep{epoch:02d} train", leave=True)
    for batch_i, (imgs, labels) in enumerate(pbar):
        if MAX_TRAIN_BATCHES and batch_i >= MAX_TRAIN_BATCHES:
            break

        imgs = imgs.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        bs = imgs.size(0)
        r = np.random.rand()

        with autocast(enabled=use_amp):
            if r < 0.33 and epoch > WARMUP_EP:
                imgs_m, ya, yb, lam = mixup(imgs, labels)
                out = model(imgs_m)
                loss = mixed_loss(crit, out, ya, yb, lam)
                is_mixed = True
            elif r < 0.66 and epoch > WARMUP_EP:
                imgs_m, ya, yb, lam = cutmix(imgs, labels)
                out = model(imgs_m)
                loss = mixed_loss(crit, out, ya, yb, lam)
                is_mixed = True
            else:
                out = model(imgs)
                loss = crit(out, labels)
                is_mixed = False

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        total_loss += loss.item() * bs
        loss_n += bs
        if not is_mixed:
            pred = out.argmax(1)
            correct += (pred == labels).sum().item()
            acc_n += bs
        pbar.set_postfix(loss=f"{loss.item():.3f}", acc_batches=str(batch_i + 1))

    return total_loss / max(1, loss_n), (100.0 * correct / acc_n) if acc_n else 0.0

@torch.no_grad()
def evaluate(model, loader, use_amp=False):
    model.eval()
    total_loss = correct = total = 0
    all_preds, all_labels = [], []

    for imgs, labels in tqdm(loader, desc="Eval", leave=False):
        imgs = imgs.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        with autocast(enabled=use_amp):
            out = model(imgs)
        loss = F.cross_entropy(out.float(), labels)
        pred = out.argmax(1)
        correct += (pred == labels).sum().item()
        total_loss += loss.item() * imgs.size(0)
        total += imgs.size(0)
        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    acc = 100 * correct / total
    f1 = f1_score(all_labels, all_preds, average="macro") * 100
    return total_loss / total, acc, f1, all_preds, all_labels


# Training loop
history = {'tl': [], 'ta': [], 'vl': [], 'va': [], 'vf1': [], 'lr_head': []}

print("🔥 Training OmniCrops-SwinV2+FPN")
print(f"   Device:{DEVICE} | Classes:{NUM_CLASSES} | Patience:{PATIENCE}")
print(f"   Mixup:{MIXUP_ALPHA} | CutMix:{CUTMIX_ALPHA} | LabelSmooth:{LABEL_SMOOTH}")
print("─" * 95)

for epoch in range(1, EPOCHS + 1):
    t0 = time.time()
    tl, ta = train_epoch(
        model, train_loader, criterion, optimizer, epoch, _grad_scaler, USE_AMP
    )
    vl, va, vf1, _, _ = evaluate(model, val_loader, use_amp=USE_AMP)
    scheduler.step()

    history['tl'].append(tl)
    history['ta'].append(ta)
    history['vl'].append(vl)
    history['va'].append(va)
    history['vf1'].append(vf1)
    history['lr_head'].append(optimizer.param_groups[2]['lr'])

    stop = stopper.step(vl, vf1, epoch, model)
    flag = "🏅 BEST" if stopper.counter == 0 else f"(pat {stopper.counter}/{PATIENCE})"

    print(f"Ep{epoch:02d}/{EPOCHS} | {time.time()-t0:.0f}s | "
          f"Loss:{tl:.4f}/{vl:.4f} | Acc:{ta:.2f}/{va:.2f}% | "
          f"F1:{vf1:.4f}% | "
          f"LR:{optimizer.param_groups[2]['lr']:.2e} {flag}")

    if stop:
        print(f"\n⏹️  Early stop ep{epoch}. Best: ep{stopper.best_ep} "
              f"F1={stopper.best_f1:.4f}%")
        break

try:
    _ckpt = torch.load(SAVE_PATH, map_location=DEVICE, weights_only=True)
except TypeError:
    _ckpt = torch.load(SAVE_PATH, map_location=DEVICE)
model.load_state_dict(_ckpt)
del _ckpt
print(f"\n✅ Best model loaded from ep{stopper.best_ep}")


# === C. Evaluate: curves → TTA test → confusion matrix → ROC → report ===

# Training curves
ep = range(1, len(history['tl']) + 1)
fig, axes = plt.subplots(1, 4, figsize=(24, 5))

axes[0].plot(ep, history['tl'], 'b-o', ms=3, label='Train')
axes[0].plot(ep, history['vl'], 'r-o', ms=3, label='Val')
axes[0].axvline(stopper.best_ep, color='green', ls='--',
                label=f'Best ep{stopper.best_ep}')
axes[0].set_title("Loss", fontweight='bold'); axes[0].legend()
axes[0].grid(alpha=0.4)

axes[1].plot(ep, history['ta'], 'b-s', ms=3, label='Train Acc')
axes[1].plot(ep, history['va'], 'r-s', ms=3, label='Val Acc')
axes[1].axvline(stopper.best_ep, color='green', ls='--')
axes[1].set_title("Accuracy", fontweight='bold'); axes[1].legend()
axes[1].grid(alpha=0.4)

axes[2].plot(ep, history['vf1'], 'm-D', ms=3, label='Val F1 Macro')
axes[2].axvline(stopper.best_ep, color='green', ls='--')
axes[2].set_title("F1-Score (Macro) ↑", fontweight='bold'); axes[2].legend()
axes[2].grid(alpha=0.4)

axes[3].plot(ep, history['lr_head'], 'k-', lw=2, label='Head LR')
axes[3].set_title("Learning Rate (head)", fontweight='bold'); axes[3].legend()
axes[3].grid(alpha=0.4)

plt.suptitle("OmniCrops-SwinV2+FPN Training Curves", fontsize=14, fontweight='bold')
plt.tight_layout()
print("\n▶ Figure (train): training_curves — loss / acc / val F1 / LR")
save_fig_notebook(fig, FIG_DIR / "training_curves.png", dpi=120)


@torch.no_grad()
def collect_probs_loader(model, loader, use_amp=False):
    """Collect class probabilities and labels from a dataloader."""
    model.eval()
    probs_all, labels_all = [], []
    for imgs, labels in tqdm(loader, desc="Collect probs", leave=False):
        imgs = imgs.to(DEVICE, non_blocking=True)
        with autocast(enabled=use_amp):
            out = model(imgs)
        probs_all.append(F.softmax(out, dim=1).cpu().numpy())
        labels_all.append(labels.numpy())
    return np.concatenate(probs_all, axis=0), np.concatenate(labels_all, axis=0)


@torch.no_grad()
def tta_collect_probs(model, ds, tta_transforms, use_amp=False):
    """Average softmax probs over TTA views (one forward pass per view)."""
    model.eval()
    probs_all, labels_all = [], []
    for path, label in tqdm(ds.samples, desc=f"TTA ×{len(tta_transforms)}", leave=True):
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), 0)
        prob = torch.zeros(NUM_CLASSES, device=DEVICE)
        for tf in tta_transforms:
            x = tf(img).unsqueeze(0).to(DEVICE, non_blocking=True)
            with autocast(enabled=use_amp):
                prob += F.softmax(model(x), dim=1).squeeze(0)
        prob /= len(tta_transforms)
        probs_all.append(prob.cpu().numpy())
        labels_all.append(label)
    return np.stack(probs_all), np.array(labels_all)


std_probs, std_labels = collect_probs_loader(model, test_loader, use_amp=USE_AMP)
std_preds = std_probs.argmax(axis=1)
std_acc = accuracy_score(std_labels, std_preds) * 100
std_f1 = f1_score(std_labels, std_preds, average="macro") * 100
print(f"   Std Accuracy : {std_acc:.4f}%")
print(f"   Std F1 Macro : {std_f1:.4f}%")

tta_acc, tta_f1 = std_acc, std_f1
if USE_TTA:
    print(f"\n🔍 TTA (×{len(tta_tfs)}) on test set...")
    tta_probs, tta_labels = tta_collect_probs(model, test_ds, tta_tfs, use_amp=USE_AMP)
    tta_preds = tta_probs.argmax(axis=1)
    tta_acc = accuracy_score(tta_labels, tta_preds) * 100
    tta_f1 = f1_score(tta_labels, tta_preds, average="macro") * 100
    print(f"   TTA Accuracy : {tta_acc:.4f}%")
    print(f"   TTA F1 Macro : {tta_f1:.4f}%")
    eval_probs, eval_labels, eval_preds = tta_probs, tta_labels, tta_preds
    eval_tag = f"TTA ×{len(tta_tfs)}"
else:
    eval_probs, eval_labels, eval_preds = std_probs, std_labels, std_preds
    eval_tag = "Standard test"

per_f1 = f1_score(eval_labels, eval_preds, average=None, zero_division=0) * 100

# Confusion matrix (selected eval mode)
short_names = [c.replace("___", "\n") for c in class_list]
fig_cm, ax_cm = plt.subplots(figsize=(14, 12))
cm = confusion_matrix(eval_labels, eval_preds)
cm_pct = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1) * 100
sns.heatmap(
    cm_pct,
    annot=(NUM_CLASSES <= 24),
    fmt=".0f",
    cmap="Blues",
    xticklabels=short_names,
    yticklabels=short_names,
    ax=ax_cm,
    linewidths=0.2,
    cbar_kws={"label": "% row"},
)
ax_cm.set_title(f"Confusion matrix ({eval_tag})", fontweight="bold", fontsize=12)
ax_cm.set_ylabel("True")
ax_cm.set_xlabel("Predicted")
ax_cm.tick_params(axis="both", labelsize=6)
plt.tight_layout()
print("\n▶ Figure (eval): confusion_matrix_eval")
save_fig_notebook(fig_cm, FIG_DIR / "confusion_matrix_eval.png", dpi=100)

# Multiclass ROC (one-vs-rest): macro + micro (readable for many classes)
y_bin = label_binarize(eval_labels, classes=np.arange(NUM_CLASSES))
fpr_m, tpr_m, _ = roc_curve(y_bin.ravel(), eval_probs.ravel())
auc_micro = auc(fpr_m, tpr_m)
fpr_i, tpr_i, roc_auc_i = {}, {}, {}
for i in range(NUM_CLASSES):
    fpr_i[i], tpr_i[i], _ = roc_curve(y_bin[:, i], eval_probs[:, i])
    roc_auc_i[i] = auc(fpr_i[i], tpr_i[i])
all_fpr = np.unique(np.concatenate([fpr_i[i] for i in range(NUM_CLASSES)]))
mean_tpr = np.zeros_like(all_fpr, dtype=float)
for i in range(NUM_CLASSES):
    mean_tpr += np.interp(all_fpr, fpr_i[i], tpr_i[i])
mean_tpr /= NUM_CLASSES
auc_macro = auc(all_fpr, mean_tpr)

fig_roc, ax_r = plt.subplots(figsize=(8, 7))
# Plot all per-class ROC curves (thin) + averages (dotted, bold).
for i in range(NUM_CLASSES):
    ax_r.plot(fpr_i[i], tpr_i[i], color="#5dade2", lw=0.9, alpha=0.25)
ax_r.plot(
    fpr_m, tpr_m, color="#e74c3c", lw=2.6, linestyle=":",
    label=f"Micro-average (dotted) AUC = {auc_micro:.4f}"
)
ax_r.plot(
    all_fpr, mean_tpr, color="#2e86c1", lw=2.6, linestyle=":",
    label=f"Macro-average (dotted) AUC = {auc_macro:.4f}"
)
ax_r.plot([0, 1], [0, 1], "k:", lw=1, alpha=0.5)
ax_r.set_xlim([0.0, 1.0])
ax_r.set_ylim([0.0, 1.05])
ax_r.set_xlabel("False positive rate")
ax_r.set_ylabel("True positive rate")
ax_r.set_title(f"Class-wise ROC + dotted averages — {eval_tag}", fontweight="bold")
ax_r.legend(loc="lower right")
ax_r.grid(alpha=0.3)
plt.tight_layout()
print("\n▶ Figure (eval): roc_multiclass_eval")
save_fig_notebook(fig_roc, FIG_DIR / "roc_multiclass_eval.png", dpi=120)

# Prediction confidence distribution (M3-style score visualization)
eval_conf = eval_probs.max(axis=1)
is_correct = (eval_preds == eval_labels)
fig_conf, ax_conf = plt.subplots(figsize=(10, 5))
ax_conf.hist(eval_conf[is_correct], bins=25, alpha=0.7, color="#27ae60", label="Correct")
if np.any(~is_correct):
    ax_conf.hist(eval_conf[~is_correct], bins=25, alpha=0.7, color="#e74c3c", label="Wrong")
ax_conf.axvline(0.5, color="k", linestyle="--", lw=1.5, label="0.5 reference")
ax_conf.set_title(f"Prediction Confidence Distribution ({eval_tag})", fontweight="bold")
ax_conf.set_xlabel("Max softmax probability")
ax_conf.set_ylabel("Number of samples")
ax_conf.grid(alpha=0.3)
ax_conf.legend()
plt.tight_layout()
print("▶ Figure (eval): confidence_distribution")
save_fig_notebook(fig_conf, FIG_DIR / "confidence_distribution.png", dpi=120)

# Robustness under Gaussian noise (M3-style robustness figure)
@torch.no_grad()
def eval_with_noise(model, loader, sigma, use_amp=False):
    model.eval()
    preds_all, labels_all = [], []
    for imgs, labels in tqdm(loader, desc=f"Noise σ={sigma:.2f}", leave=False):
        imgs = imgs.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        if sigma > 0:
            imgs = torch.clamp(imgs + torch.randn_like(imgs) * sigma, -3.0, 3.0)
        with autocast(enabled=use_amp):
            out = model(imgs)
        preds_all.extend(out.argmax(1).cpu().numpy())
        labels_all.extend(labels.cpu().numpy())
    acc_n = accuracy_score(labels_all, preds_all) * 100
    f1_n = f1_score(labels_all, preds_all, average="macro", zero_division=0) * 100
    return acc_n, f1_n

print("\n🛡️ Robustness evaluation (Gaussian noise)")
noise_levels = [0.0, 0.03, 0.06, 0.10, 0.15]
rob_accs, rob_f1s = [], []
for sigma in noise_levels:
    acc_n, f1_n = eval_with_noise(model, test_loader, sigma, use_amp=USE_AMP)
    rob_accs.append(acc_n)
    rob_f1s.append(f1_n)
    print(f"   σ={sigma:.2f} | Acc={acc_n:.3f}% | F1={f1_n:.3f}%")

fig_rob, axes_rob = plt.subplots(1, 2, figsize=(12, 5))
axes_rob[0].plot(noise_levels, rob_accs, 'o-', color="#e67e22", lw=2, ms=6)
axes_rob[0].set_title("Robustness — Accuracy vs Noise", fontweight="bold")
axes_rob[0].set_xlabel("Gaussian noise σ")
axes_rob[0].set_ylabel("Accuracy (%)")
axes_rob[0].grid(alpha=0.3)

axes_rob[1].plot(noise_levels, rob_f1s, 's-', color="#8e44ad", lw=2, ms=6)
axes_rob[1].set_title("Robustness — Macro F1 vs Noise", fontweight="bold")
axes_rob[1].set_xlabel("Gaussian noise σ")
axes_rob[1].set_ylabel("F1 Macro (%)")
axes_rob[1].grid(alpha=0.3)
plt.tight_layout()
print("▶ Figure (eval): robustness_noise")
save_fig_notebook(fig_rob, FIG_DIR / "robustness_noise_eval.png", dpi=120)

print("\n" + "═" * 72)
print("📊  OmniCrops — SwinV2-B+FPN | test summary")
print("═" * 72)
print(f"  Classes: {NUM_CLASSES} | Best epoch: {stopper.best_ep} | Eval mode: {eval_tag}")
print(f"  {'':22} {'Standard':>12} {'Selected':>12}")
print("  " + "─" * 48)
print(f"  {'Accuracy %':<22} {std_acc:>11.4f} {accuracy_score(eval_labels, eval_preds) * 100:>12.4f}")
print(f"  {'F1 macro %':<22} {std_f1:>11.4f} {f1_score(eval_labels, eval_preds, average='macro') * 100:>12.4f}")
if USE_TTA:
    print(f"  {'TTA gain (F1)':<22} {0.0:>11.4f} {(tta_f1 - std_f1):>12.4f}")
print(f"  ROC AUC (micro/macro): {auc_micro:.4f} / {auc_macro:.4f}")
print("─" * 72)
print(
    f"  Per-class F1 ({eval_tag})  mean {per_f1.mean():.2f}% | "
    f"min {per_f1.min():.2f}% | max {per_f1.max():.2f}%"
)
print("\n" + classification_report(eval_labels, eval_preds, target_names=class_list, digits=4, zero_division=0))
print("═" * 72)
print(f"✅ Model: {SAVE_PATH}")
print(f"✅ Figures: {FIG_DIR} ({len(list(FIG_DIR.glob('*.png')))} PNG)")
print("═" * 72)