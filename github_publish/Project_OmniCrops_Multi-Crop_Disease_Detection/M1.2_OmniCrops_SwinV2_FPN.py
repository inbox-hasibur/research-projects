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
from torch.utils.data import Dataset, DataLoader
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
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

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


# Fixed per-class split sizes (requested): 1100 train / 200 val / 200 test.
# NOTE: We intentionally build a fixed-size balanced pool (TARGET_PER_CLASS) per class
# so the final dataset composition is stable and comparable across runs.
N_TRAIN = int(os.environ.get("OMNICROPS_N_TRAIN", "1100"))
N_VAL = int(os.environ.get("OMNICROPS_N_VAL", "200"))
N_TEST = int(os.environ.get("OMNICROPS_N_TEST", "200"))
TARGET_PER_CLASS = N_TRAIN + N_VAL + N_TEST  # 1500 by default

# Dataset build knobs
MAX_AUG_RATIO = float(os.environ.get("OMNICROPS_MAX_AUG_RATIO", "1.2"))
IMG_SIZE = 224
JPEG_QUALITY = 92
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

print("✅ Paths configured")
print(f"   OUT_ROOT : {OUT_ROOT}")
print(f"   Split cfg: train={N_TRAIN} val={N_VAL} test={N_TEST} | target/class={TARGET_PER_CLASS}")
print(f"   Balancing : fixed target/class={TARGET_PER_CLASS} | max_aug_ratio={MAX_AUG_RATIO:.2f}")


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

shape_targets = {cls: int(TARGET_PER_CLASS) for cls in raw_counts.keys()}

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
print(f"  Target/class (fixed): {TARGET_PER_CLASS}")

crop_type_counts = defaultdict(int)
for cls_name in raw_counts.keys():
    crop_name = cls_name.split("___")[0]
    crop_type_counts[crop_name] += 1
print(f"  Crop types: {len(crop_type_counts)}")


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
        lambda i: ImageEnhance.Saturation(i).enhance(s),
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
      f"({N_TRAIN} train + {N_VAL} val + {N_TEST} test, fixed per-class totals)")
print("=" * 65)

aug_report = {}
BALANCED_ROOT = OUT_ROOT / "balanced_raw"

for cls in tqdm(sorted(raw_pool.keys()), desc="Balancing classes"):
    paths = raw_pool[cls].copy()
    random.shuffle(paths)
    n_avail = len(paths)
    cls_target = shape_targets[cls]
    cls_dir = BALANCED_ROOT / cls
    cls_dir.mkdir(parents=True, exist_ok=True)
    written = 0

    if n_avail >= cls_target:
        for src in paths[:cls_target]:
            if save_uniform_jpeg(src, cls_dir / f"img_{written:05d}.jpg"):
                written += 1
        k = 0
        while written < cls_target and k < n_avail * 20:
            src = paths[k % n_avail]
            if save_uniform_jpeg(src, cls_dir / f"img_{written:05d}.jpg"):
                written += 1
            k += 1
        aug_report[cls] = {
            "raw_available": n_avail,
            "augmented": 0,
            "target": cls_target,
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
        aug_cap = int(max(0, round(n_saved_orig * MAX_AUG_RATIO)))
        while written < cls_target and (written - n_saved_orig) < aug_cap:
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
            if aug_id > (cls_target - n_saved_orig) * 30 + 500:
                break
        while written < cls_target:
            src = paths[written % max(1, len(paths))]
            if save_uniform_jpeg(src, cls_dir / f"img_{written:05d}.jpg"):
                written += 1
        added = max(0, written - n_saved_orig)
        aug_report[cls] = {
            "raw_available": n_avail,
            "augmented": added,
            "target": cls_target,
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

print(f"\n📂 Splitting with fixed counts per class: {N_TRAIN} train / {N_VAL} val / {N_TEST} test")

for split in ["train", "val", "test"]:
    (OUT_ROOT / split).mkdir(parents=True, exist_ok=True)

split_report = {}

for cls in tqdm(sorted(aug_report.keys()), desc="Splitting"):
    src_dir = BALANCED_ROOT / cls
    files = [f for f in src_dir.iterdir() if f.suffix.lower() in IMG_EXTS]
    random.shuffle(files)
    files = files[:aug_report[cls]["total"]]
    n = len(files)
    # Enforce exact per-class split sizes (or as close as possible if class is short).
    n_train = min(N_TRAIN, n)
    n_val = min(N_VAL, max(0, n - n_train))
    n_test = min(N_TEST, max(0, n - n_train - n_val))
    # If for some reason we have extra images (shouldn't happen with fixed build),
    # keep them in train to avoid changing val/test sizes.
    if n_train + n_val + n_test < n:
        n_train = min(n, n_train + (n - (n_train + n_val + n_test)))

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
    "max_aug_ratio": MAX_AUG_RATIO,
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
target_shaped = [shape_targets[c] for c in classes_sorted_orig]
total_after = [aug_report[c]["total"] for c in classes_sorted_orig]
x = np.arange(len(classes_sorted_orig))
_h = max(8.0, len(classes_sorted_orig) * 0.11)
fig_ds, (axb, axa) = plt.subplots(1, 2, figsize=(18, _h))
colors_before = ["#e74c3c" if v < TARGET_PER_CLASS else "#27ae60" for v in orig_before]
axb.barh(x, orig_before, color=colors_before, edgecolor="white", linewidth=0.5)
axb.axvline(TARGET_PER_CLASS, color="navy", ls="--", lw=2, label=f"peak {TARGET_PER_CLASS}/class")
axb.set_yticks(x)
axb.set_yticklabels(classes_sorted_orig, fontsize=6)
axb.set_title("Before — raw counts / class", fontsize=11, fontweight="bold")
axb.set_xlabel("Count")
axb.legend()
axb.grid(axis="x", alpha=0.3)
axa.barh(x, total_after, color="#2ecc71", edgecolor="white", linewidth=0.5, label="actual shaped total")
axa.plot(target_shaped, x, color="#8e44ad", lw=1.6, label="target/class (fixed)")
axa.axvline(TARGET_PER_CLASS, color="navy", ls="--", lw=2, label=f"peak {TARGET_PER_CLASS}/class")
axa.set_yticks(x)
axa.set_yticklabels(classes_sorted_orig, fontsize=6)
axa.set_title("After — balanced pool", fontsize=11, fontweight="bold")
axa.set_xlabel("Count")
axa.legend()
axa.grid(axis="x", alpha=0.3)
plt.suptitle("OmniCrops — dataset shaping (raw vs shaped)", fontsize=12, fontweight="bold")
plt.tight_layout()
p_dataset = FIG_DIR / "dataset_overview.png"
print("\n▶ Figure (dataset): dataset_overview — before | after")
save_fig_notebook(fig_ds, p_dataset, dpi=120)
print(f"✅ Saved → {p_dataset}")

# Paper Figure 1: class count histogram (before vs after balancing)
fig_hist, axh = plt.subplots(figsize=(11, 5))
bins = np.linspace(0, max(max(orig_before), TARGET_PER_CLASS), 25)
axh.hist(orig_before, bins=bins, alpha=0.65, label="Raw class counts", color="#e67e22")
axh.hist(total_after, bins=bins, alpha=0.65, label="Balanced class counts", color="#2ecc71")
axh.axvline(TARGET_PER_CLASS, color="navy", ls="--", lw=2, label=f"Target={TARGET_PER_CLASS}")
axh.set_title("Class-count distribution: raw vs balanced", fontweight="bold")
axh.set_xlabel("Images per class")
axh.set_ylabel("Number of classes")
axh.grid(alpha=0.25)
axh.legend()
plt.tight_layout()
p_hist = FIG_DIR / "class_count_hist_before_after.png"
print("\n▶ Figure (dataset): class_count_hist_before_after")
save_fig_notebook(fig_hist, p_hist, dpi=120)

# Small figure: classes per crop type
crop_items = sorted(crop_type_counts.items(), key=lambda x: x[1], reverse=True)
crop_labels = [k for k, _ in crop_items]
crop_vals = [v for _, v in crop_items]
fig_crop, ax_crop = plt.subplots(figsize=(6.2, 3.8))
bars = ax_crop.barh(np.arange(len(crop_labels)), crop_vals, color="#4c78a8")
ax_crop.set_yticks(np.arange(len(crop_labels)))
ax_crop.set_yticklabels(crop_labels, fontsize=8)
ax_crop.invert_yaxis()
ax_crop.set_xlabel("Classes", fontsize=9)
ax_crop.set_title("Classes per crop type", fontsize=10, fontweight="bold")
ax_crop.grid(axis="x", alpha=0.25)
for i, b in enumerate(bars):
    ax_crop.text(b.get_width() + 0.05, b.get_y() + b.get_height()/2, str(crop_vals[i]), va="center", fontsize=8)
plt.tight_layout()
p_crop = FIG_DIR / "classes_per_crop_type_small.png"
print("▶ Figure (dataset): classes_per_crop_type_small")
save_fig_notebook(fig_crop, p_crop, dpi=130)

# Paper Figure 2: qualitative sample grid from the train split
sample_classes = sorted(split_report.keys())[:min(12, len(split_report))]
fig_samp, axes = plt.subplots(3, 4, figsize=(12, 9))
axes = axes.flatten()
for i, cls in enumerate(sample_classes):
    ax = axes[i]
    cls_dir = OUT_ROOT / "train" / cls
    imgs = [p for p in cls_dir.iterdir() if p.suffix.lower() in IMG_EXTS]
    if imgs:
        pick = random.choice(imgs)
        try:
            im = Image.open(pick).convert("RGB")
            ax.imshow(im)
        except Exception:
            ax.imshow(np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8))
    else:
        ax.imshow(np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8))
    ax.set_title(cls.replace("___", "\n"), fontsize=8)
    ax.axis("off")
for j in range(len(sample_classes), len(axes)):
    axes[j].axis("off")
plt.suptitle("Training samples (representative classes)", fontsize=12, fontweight="bold")
plt.tight_layout()
p_samples = FIG_DIR / "dataset_samples_grid.png"
print("▶ Figure (dataset): dataset_samples_grid")
save_fig_notebook(fig_samp, p_samples, dpi=120)

# Paper Figure 3: split consistency per class (train/val/test)
split_mat = np.array([[split_report[c]["train"], split_report[c]["val"], split_report[c]["test"]]
                      for c in classes_sorted_orig], dtype=float)
fig_split, axsp = plt.subplots(figsize=(10, max(8, len(classes_sorted_orig) * 0.22)))
sns.heatmap(
    split_mat,
    cmap="YlGnBu",
    ax=axsp,
    cbar_kws={"label": "Images"},
    yticklabels=[c.replace("___", " | ") for c in classes_sorted_orig],
    xticklabels=["Train", "Val", "Test"],
)
axsp.set_title("Per-class split distribution", fontweight="bold")
axsp.tick_params(axis="y", labelsize=6)
plt.tight_layout()
p_split = FIG_DIR / "split_distribution_heatmap.png"
print("▶ Figure (dataset): split_distribution_heatmap")
save_fig_notebook(fig_split, p_split, dpi=120)

# Dataset build summary
print("\n" + "=" * 70)
print("🌿  OmniCrops dataset build — COMPLETE")
print("=" * 70)
print(f"  Location     : {OUT_ROOT}")
print(f"  Classes      : {len(class_list)}")
print(f"  Per class    : fixed total={TARGET_PER_CLASS} (train/val/test = {N_TRAIN}/{N_VAL}/{N_TEST})")
print(f"  Total images : {total_train + total_val + total_test:,}")
print(f"  Train        : {total_train:,} | Val: {total_val:,} | Test: {total_test:,}")
print("─" * 70)
print(f"  Augmented classes  : {sum(1 for d in aug_report.values() if d['augmented'] > 0)}")
print(f"  Capped classes     : {sum(1 for d in aug_report.values() if d['status'] == 'capped')}")
print("─" * 70)
print("  Artifacts:")
for p in [p_dataset, p_hist, p_crop, p_samples, p_split, OUT_ROOT / "metadata.json", OUT_ROOT / "dataset.yaml"]:
    ok = "✓" if Path(p).exists() else "✗"
    print(f"    {ok} {p}")
print("=" * 70)


# === B. Train SwinV2-B + FPN ===

# Device, hyperparameters, reproducibility
USE_TPU_REQ = os.environ.get("OMNICROPS_USE_TPU", "0").lower() in ("1", "true", "yes")
XLA_AVAILABLE = False
xm = None
if USE_TPU_REQ:
    try:
        import torch_xla.core.xla_model as xm  # type: ignore
        XLA_AVAILABLE = True
    except Exception:
        XLA_AVAILABLE = False

if USE_TPU_REQ and XLA_AVAILABLE:
    DEVICE = xm.xla_device()
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
print(f"\n🚀 Device : {DEVICE}")
if DEVICE.type == "cuda":
    print(f"   GPU    : {torch.cuda.get_device_name(0)}")
    print(f"   VRAM   : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
elif USE_TPU_REQ and XLA_AVAILABLE:
    print("   TPU    : XLA runtime enabled")
elif USE_TPU_REQ and not XLA_AVAILABLE:
    print("   ⚠️ TPU requested but torch_xla not found; fallback to CUDA/CPU.")

# ── Hyperparameters ───────────────────────────────────────────
# T4/T4x2-friendly defaults:
# keep per-step batch moderate and recover large effective batch via accumulation.
BATCH_SIZE    = int(os.environ.get("OMNICROPS_BATCH_SIZE", "32"))
EFFECTIVE_BATCH_TARGET = int(os.environ.get("OMNICROPS_EFFECTIVE_BATCH", "512"))
ACCUM_STEPS = max(1, int(math.ceil(EFFECTIVE_BATCH_TARGET / max(1, BATCH_SIZE))))
HIGH_F1_PRESET = os.environ.get("OMNICROPS_HIGH_F1_PRESET", "1").lower() in ("1", "true", "yes")
# Kaggle: many workers + heavy aug can make the *first* batches feel "stuck"; 2 is a sane default.
_kw = 0 if sys.platform.startswith("win") else min(8, (os.cpu_count() or 4))
if Path("/kaggle/working").exists():
    _kw = min(2, _kw)
NUM_WORKERS   = int(os.environ.get("OMNICROPS_NUM_WORKERS", str(_kw)))
EPOCHS_STAGE1 = int(os.environ.get("OMNICROPS_STAGE1_EPOCHS", "40"))
EPOCHS_STAGE2 = int(os.environ.get("OMNICROPS_STAGE2_EPOCHS", "10"))
EPOCHS        = EPOCHS_STAGE1 + EPOCHS_STAGE2
_effective_batch = BATCH_SIZE * ACCUM_STEPS
_bs_scale = max(1.0, _effective_batch / 256.0)
LR_BACKBONE   = float(os.environ.get("OMNICROPS_LR_BACKBONE", str(1.0e-5 * _bs_scale)))
LR_FPN        = float(os.environ.get("OMNICROPS_LR_FPN", str(8.0e-5 * _bs_scale)))
LR_HEAD       = float(os.environ.get("OMNICROPS_LR_HEAD", str(4.0e-4 * _bs_scale)))
WARMUP_EP     = 3
PATIENCE      = int(os.environ.get("OMNICROPS_PATIENCE", "8"))
LABEL_SMOOTH  = float(os.environ.get("OMNICROPS_LABEL_SMOOTH", "0.02"))
MIXUP_ALPHA   = float(os.environ.get("OMNICROPS_MIXUP_ALPHA", "0.15"))
CUTMIX_ALPHA  = float(os.environ.get("OMNICROPS_CUTMIX_ALPHA", "0.8"))
DROPOUT_HEAD  = float(os.environ.get("OMNICROPS_DROPOUT_HEAD", "0.2"))
WEIGHT_DECAY  = 1e-4
SEED          = 42
SAVE_PATH     = str(OUT_ROOT / "best_omnicrops_swinv2.pth")
USE_TTA = os.environ.get("OMNICROPS_USE_TTA", "1").lower() in ("1", "true", "yes")

if HIGH_F1_PRESET:
    # F1-first defaults: reduce destructive augmentation noise, keep useful regularization.
    MIXUP_ALPHA = float(os.environ.get("OMNICROPS_MIXUP_ALPHA", "0.05"))
    CUTMIX_ALPHA = float(os.environ.get("OMNICROPS_CUTMIX_ALPHA", "0.20"))
    LABEL_SMOOTH = float(os.environ.get("OMNICROPS_LABEL_SMOOTH", "0.010"))
    WEIGHT_DECAY = float(os.environ.get("OMNICROPS_WEIGHT_DECAY", "8e-5"))

MIXUP_PROB = float(os.environ.get("OMNICROPS_MIXUP_PROB", "0.10" if HIGH_F1_PRESET else "0.33"))
CUTMIX_PROB = float(os.environ.get("OMNICROPS_CUTMIX_PROB", "0.15" if HIGH_F1_PRESET else "0.33"))
FINETUNE_LABEL_SMOOTH = float(os.environ.get("OMNICROPS_FINETUNE_LABEL_SMOOTH", "0.005"))
STAGE2_LR_MULT = float(os.environ.get("OMNICROPS_STAGE2_LR_MULT", "0.20"))
IMG_SIZE_STAGE1 = int(os.environ.get("OMNICROPS_TRAIN_SIZE_STAGE1", "224"))
IMG_SIZE_STAGE2 = int(os.environ.get("OMNICROPS_TRAIN_SIZE_STAGE2", "320"))
EMA_DECAY = float(os.environ.get("OMNICROPS_EMA_DECAY", "0.9995"))
USE_EMA = os.environ.get("OMNICROPS_USE_EMA", "1").lower() in ("1", "true", "yes")

torch.manual_seed(SEED)
if DEVICE.type == "cuda":
    torch.cuda.manual_seed_all(SEED)
    # Prefer stability/reproducibility over small speed gains.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

USE_AMP = DEVICE.type == "cuda" and os.environ.get("OMNICROPS_NO_AMP", "").lower() not in ("1", "true", "yes")
_grad_scaler = GradScaler(enabled=USE_AMP)
MAX_TRAIN_BATCHES = int(os.environ.get("OMNICROPS_MAX_TRAIN_BATCHES", "0"))

# Class list from dataset prep (already defined above)
NUM_CLASSES = len(class_list)
GPU_COUNT = torch.cuda.device_count() if torch.cuda.is_available() else 0
print(f"   Classes : {NUM_CLASSES}")
print(f"   Batch   : {BATCH_SIZE} | Workers: {NUM_WORKERS} | Epochs: {EPOCHS} ({EPOCHS_STAGE1}+{EPOCHS_STAGE2}) | Patience: {PATIENCE}")
print(f"   Effective batch target: {EFFECTIVE_BATCH_TARGET} | Accum steps: {ACCUM_STEPS} | Effective used: {_effective_batch}")
print(f"   AMP     : {'ON (fp16/bf16 mix — faster)' if USE_AMP else 'OFF'}")
print(f"   High-F1 preset: {'ON' if HIGH_F1_PRESET else 'OFF'} | mixup_prob={MIXUP_PROB:.2f} cutmix_prob={CUTMIX_PROB:.2f}")
print(f"   Progressive resize: {IMG_SIZE_STAGE1} → {IMG_SIZE_STAGE2} | EMA: {'ON' if USE_EMA else 'OFF'} (decay={EMA_DECAY})")
if GPU_COUNT > 1 and os.environ.get("OMNICROPS_USE_DDP", "0").lower() not in ("1", "true", "yes"):
    print(f"   ⚠️  {GPU_COUNT} GPUs detected, but single-process training is active (no DDP).")
if DEVICE.type == "cuda" and BATCH_SIZE > 64:
    print("   ⚠️  Large per-step batch on T4 may cause OOM; prefer 24-64 + accumulation.")
if MAX_TRAIN_BATCHES > 0:
    print(f"   ⚠️  OMNICROPS_MAX_TRAIN_BATCHES={MAX_TRAIN_BATCHES} (debug smoke run; not full epoch)")


# Transforms (train aug + val + TTA)
MEAN = [0.485, 0.456, 0.406]   # ImageNet stats
STD  = [0.229, 0.224, 0.225]

def build_transforms(train_size: int, eval_size: int, no_mix_phase: bool = False):
    aug_rotation = 8 if no_mix_phase else (12 if HIGH_F1_PRESET else 20)
    train_tf = transforms.Compose([
        transforms.Resize((int(train_size * 1.14), int(train_size * 1.14))),
        transforms.RandomCrop(train_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.20 if no_mix_phase else 0.25),
        transforms.RandomRotation(aug_rotation),
        transforms.ColorJitter(
            brightness=0.15 if no_mix_phase else (0.2 if HIGH_F1_PRESET else 0.3),
            contrast=0.15 if no_mix_phase else (0.2 if HIGH_F1_PRESET else 0.3),
            saturation=0.15 if no_mix_phase else (0.2 if HIGH_F1_PRESET else 0.3),
            hue=0.04 if no_mix_phase else (0.06 if HIGH_F1_PRESET else 0.1),
        ),
        transforms.RandAugment(num_ops=2, magnitude=5 if HIGH_F1_PRESET else 7)
        if (not no_mix_phase and os.environ.get("OMNICROPS_USE_RANDAUG", "0").lower() in ("1", "true", "yes"))
        else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
        transforms.RandomErasing(
            p=0.05 if no_mix_phase else (0.08 if HIGH_F1_PRESET else 0.12),
            scale=(0.02, 0.05) if no_mix_phase else ((0.02, 0.06) if HIGH_F1_PRESET else (0.02, 0.10))
        ),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((eval_size, eval_size)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    tta_tfs = [
        val_tf,
        transforms.Compose([transforms.Resize((eval_size, eval_size)),
                            transforms.RandomHorizontalFlip(p=1.0),
                            transforms.ToTensor(), transforms.Normalize(MEAN, STD)]),
        transforms.Compose([transforms.Resize((int(eval_size * 1.12), int(eval_size * 1.12))),
                            transforms.CenterCrop(eval_size),
                            transforms.ToTensor(), transforms.Normalize(MEAN, STD)]),
        transforms.Compose([transforms.Resize((eval_size, eval_size)),
                            transforms.RandomRotation((90, 90)),
                            transforms.ToTensor(), transforms.Normalize(MEAN, STD)]),
        transforms.Compose([transforms.Resize((eval_size, eval_size)),
                            transforms.RandomRotation((-90, -90)),
                            transforms.ToTensor(), transforms.Normalize(MEAN, STD)]),
    ]
    return train_tf, val_tf, tta_tfs


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

def build_dataloaders(train_size: int, eval_size: int, no_mix_phase: bool = False):
    train_tf_local, val_tf_local, tta_tfs_local = build_transforms(train_size, eval_size, no_mix_phase=no_mix_phase)
    train_ds_local = OmniCropsDataset(OUT_ROOT/"train", class_list, train_tf_local)
    val_ds_local = OmniCropsDataset(OUT_ROOT/"val", class_list, val_tf_local)
    test_ds_local = OmniCropsDataset(OUT_ROOT/"test", class_list, val_tf_local)
    pin_local = DEVICE.type == "cuda"
    train_loader_local = DataLoader(
        train_ds_local, BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=pin_local, drop_last=True,
        persistent_workers=NUM_WORKERS > 0,
    )
    val_loader_local = DataLoader(
        val_ds_local, BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=pin_local,
        persistent_workers=NUM_WORKERS > 0,
    )
    test_loader_local = DataLoader(
        test_ds_local, BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=pin_local,
        persistent_workers=NUM_WORKERS > 0,
    )
    return (
        train_tf_local, val_tf_local, tta_tfs_local,
        train_ds_local, val_ds_local, test_ds_local,
        train_loader_local, val_loader_local, test_loader_local,
    )

(
    train_tf, val_tf, tta_tfs,
    train_ds, val_ds, test_ds,
    train_loader, val_loader, test_loader
) = build_dataloaders(IMG_SIZE_STAGE1, IMG_SIZE_STAGE1, no_mix_phase=False)
if os.environ.get("OMNICROPS_FAST_TTA", "").lower() in ("1", "true", "yes"):
    tta_tfs = [val_tf, tta_tfs[1]]
    print(f"✅ Transforms ready | TTA views: {len(tta_tfs)}× (OMNICROPS_FAST_TTA)")
else:
    print(f"✅ Transforms ready | TTA views: {len(tta_tfs)}×")
print(f"   TTA enabled: {'YES' if USE_TTA else 'NO (standard test metrics)'}")

labels_all  = [s[1] for s in train_ds.samples]
cls_cnt     = np.bincount(labels_all, minlength=NUM_CLASSES).astype(float)
cls_wt      = (cls_cnt.max() / np.maximum(cls_cnt, 1.0)) ** 0.5
cls_wt      = cls_wt / np.mean(cls_wt)
cls_wt_torch = torch.tensor(cls_wt, dtype=torch.float32, device=DEVICE)

_n_bt = len(train_loader)
print(f"✅ Datasets | Train:{len(train_ds):,} Val:{len(val_ds):,} Test:{len(test_ds):,}")
print(f"   Train steps/epoch: {_n_bt} batches (SwinV2-B + RandAugment → often ~0.5–2 s/batch on T4-class GPU)")
print(f"   Expect ~{max(1, _n_bt // 2)}–{_n_bt * 2} s for first epoch wall-clock if GPU is busy; not stuck until tqdm stops moving.")
print(f"   Class weight range (train): {cls_wt.min():.3f} .. {cls_wt.max():.3f}")


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
    _x = torch.randn(2, 3, IMG_SIZE_STAGE1, IMG_SIZE_STAGE1, device=DEVICE)
    _o = model(_x)
    print(f"   Smoke: {list(_x.shape)} → {list(_o.shape)} ✅")
    del _x, _o
    torch.cuda.empty_cache() if DEVICE.type == "cuda" else None


# Loss, optimizer, scheduler
class LabelSmoothCE(nn.Module):
    def __init__(self, classes, smooth=0.05, class_weights=None):
        super().__init__()
        self.eps = smooth; self.n = classes
        self.class_weights = class_weights
    def forward(self, pred, target):
        lp  = F.log_softmax(pred, dim=1)
        nll = F.nll_loss(lp, target, weight=self.class_weights)
        return (1 - self.eps) * nll + self.eps * (-lp.mean())

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, class_weights=None):
        super().__init__()
        self.gamma = gamma
        self.class_weights = class_weights
    def forward(self, pred, target):
        ce  = F.cross_entropy(pred, target, reduction='none', weight=self.class_weights)
        pt  = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()

class CombinedLoss(nn.Module):
    """LabelSmoothing + Focal — best of both worlds."""
    def __init__(self, classes, smooth=0.05, gamma=2.0, w_focal=0.3, class_weights=None):
        super().__init__()
        self.ls    = LabelSmoothCE(classes, smooth, class_weights=class_weights)
        self.focal = FocalLoss(gamma, class_weights=class_weights)
        self.wf    = w_focal
    def forward(self, pred, target):
        return (1 - self.wf) * self.ls(pred, target) + \
                      self.wf * self.focal(pred, target)

criterion = CombinedLoss(NUM_CLASSES, LABEL_SMOOTH, gamma=2.0, w_focal=0.3, class_weights=cls_wt_torch)
criterion_ft = CombinedLoss(NUM_CLASSES, FINETUNE_LABEL_SMOOTH, gamma=2.0, w_focal=0.2, class_weights=cls_wt_torch)

param_groups = model.get_param_groups(LR_BACKBONE, LR_FPN, LR_HEAD)
optimizer    = optim.AdamW(param_groups, weight_decay=WEIGHT_DECAY,
                           betas=(0.9, 0.999))

def warmup_cosine(ep, warmup=WARMUP_EP, total=EPOCHS_STAGE1):
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

class ModelEMA:
    """Exponential moving average of model weights for stabler validation."""
    def __init__(self, model, decay=0.9995):
        self.decay = decay
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if not v.dtype.is_floating_point:
                v.copy_(msd[k])
                continue
            v.mul_(self.decay).add_(msd[k].detach(), alpha=(1.0 - self.decay))


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
def train_epoch(model, loader, crit, opt, epoch, scaler, use_amp, mixup_prob, cutmix_prob, ema=None):
    """Mixup/CutMix batches skip accuracy (labels no longer match single-class logits)."""
    model.train()
    total_loss = 0.0
    loss_n = 0
    correct = 0
    acc_n = 0

    opt.zero_grad(set_to_none=True)
    pbar = tqdm(loader, desc=f"Ep{epoch:02d} train", leave=True)
    for batch_i, (imgs, labels) in enumerate(pbar):
        if MAX_TRAIN_BATCHES and batch_i >= MAX_TRAIN_BATCHES:
            break

        imgs = imgs.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        bs = imgs.size(0)
        r = np.random.rand()

        with autocast(enabled=use_amp):
            if r < mixup_prob and epoch > WARMUP_EP:
                imgs_m, ya, yb, lam = mixup(imgs, labels)
                out = model(imgs_m)
                loss = mixed_loss(crit, out, ya, yb, lam)
                is_mixed = True
            elif r < (mixup_prob + cutmix_prob) and epoch > WARMUP_EP:
                imgs_m, ya, yb, lam = cutmix(imgs, labels)
                out = model(imgs_m)
                loss = mixed_loss(crit, out, ya, yb, lam)
                is_mixed = True
            else:
                out = model(imgs)
                loss = crit(out, labels)
                is_mixed = False

        loss = loss / ACCUM_STEPS
        do_step = ((batch_i + 1) % ACCUM_STEPS == 0) or ((batch_i + 1) == len(loader))

        if use_amp:
            scaler.scale(loss).backward()
            if do_step:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if DEVICE.type == "xla" and xm is not None:
                    scaler.step(opt)
                    xm.mark_step()
                else:
                    scaler.step(opt)
                scaler.update()
                if ema is not None:
                    ema.update(model)
                opt.zero_grad(set_to_none=True)
        else:
            loss.backward()
            if do_step:
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if DEVICE.type == "xla" and xm is not None:
                    xm.optimizer_step(opt, barrier=True)
                else:
                    opt.step()
                if ema is not None:
                    ema.update(model)
                opt.zero_grad(set_to_none=True)

        total_loss += loss.item() * bs * ACCUM_STEPS
        loss_n += bs
        if not is_mixed:
            pred = out.argmax(1)
            correct += (pred == labels).sum().item()
            acc_n += bs
        pbar.set_postfix(loss=f"{(loss.item() * ACCUM_STEPS):.3f}", acc_batches=str(batch_i + 1), accum=f"{ACCUM_STEPS}")

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


@torch.no_grad()
def collect_probs_loader(model, loader, use_amp=False):
    """Collect softmax probabilities and labels from a dataloader."""
    model.eval()
    probs_all, labels_all = [], []
    for imgs, labels in tqdm(loader, desc="Collect probs", leave=False):
        imgs = imgs.to(DEVICE, non_blocking=True)
        with autocast(enabled=use_amp):
            out = model(imgs)
        probs_all.append(F.softmax(out, dim=1).cpu().numpy())
        labels_all.append(labels.numpy())
    return np.concatenate(probs_all, axis=0), np.concatenate(labels_all, axis=0)


# Training loop
history = {'tl': [], 'ta': [], 'vl': [], 'va': [], 'vf1': [], 'lr_head': []}

print("🔥 Training OmniCrops-SwinV2+FPN")
print(f"   Device:{DEVICE} | Classes:{NUM_CLASSES} | Patience:{PATIENCE}")
print(f"   Mixup:{MIXUP_ALPHA} | CutMix:{CUTMIX_ALPHA} | LabelSmooth:{LABEL_SMOOTH} -> fine-tune:{FINETUNE_LABEL_SMOOTH}")
print("─" * 95)
ema_model = ModelEMA(model, decay=EMA_DECAY) if USE_EMA else None
base_lrs = [pg["lr"] for pg in optimizer.param_groups]
stage2_started = False

for epoch in range(1, EPOCHS + 1):
    t0 = time.time()
    in_stage2 = epoch > EPOCHS_STAGE1
    if in_stage2 and not stage2_started:
        stage2_started = True
        for i, pg in enumerate(optimizer.param_groups):
            pg["lr"] = base_lrs[i] * STAGE2_LR_MULT
        (
            train_tf, val_tf, tta_tfs,
            train_ds, val_ds, test_ds,
            train_loader, val_loader, test_loader
        ) = build_dataloaders(IMG_SIZE_STAGE2, IMG_SIZE_STAGE2, no_mix_phase=True)
        if os.environ.get("OMNICROPS_FAST_TTA", "").lower() in ("1", "true", "yes"):
            tta_tfs = [val_tf, tta_tfs[1]]
        print(f"   🔁 Stage-2 fine-tune start @ep{epoch}: resize {IMG_SIZE_STAGE1}→{IMG_SIZE_STAGE2}, no mixup/cutmix")
        print(f"   ✅ Reloaded dataloaders | Train:{len(train_ds):,} Val:{len(val_ds):,} Test:{len(test_ds):,}")

    if in_stage2:
        prog = (epoch - EPOCHS_STAGE1 - 1) / max(1, EPOCHS_STAGE2 - 1)
        lr_mult = STAGE2_LR_MULT * 0.5 * (1 + np.cos(np.pi * prog))
        for i, pg in enumerate(optimizer.param_groups):
            pg["lr"] = base_lrs[i] * max(0.05 * STAGE2_LR_MULT, lr_mult)

    mixup_prob_ep = 0.0 if in_stage2 else MIXUP_PROB
    cutmix_prob_ep = 0.0 if in_stage2 else CUTMIX_PROB
    criterion_ep = criterion_ft if in_stage2 else criterion
    tl, ta = train_epoch(
        model, train_loader, criterion_ep, optimizer, epoch, _grad_scaler, USE_AMP,
        mixup_prob_ep, cutmix_prob_ep, ema=ema_model
    )
    eval_model = ema_model.ema if ema_model is not None else model
    vl, va, vf1, _, _ = evaluate(eval_model, val_loader, use_amp=USE_AMP)
    if not in_stage2:
        scheduler.step()

    history['tl'].append(tl)
    history['ta'].append(ta)
    history['vl'].append(vl)
    history['va'].append(va)
    history['vf1'].append(vf1)
    history['lr_head'].append(optimizer.param_groups[2]['lr'])

    stop = stopper.step(vl, vf1, epoch, eval_model)
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
fig, axes = plt.subplots(1, 3, figsize=(21, 5))

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

plt.suptitle("OmniCrops-SwinV2+FPN Training Curves", fontsize=14, fontweight='bold')
plt.tight_layout()
print("\n▶ Figure (train): training_curves — loss / acc / val F1")
save_fig_notebook(fig, FIG_DIR / "training_curves.png", dpi=120)


@torch.no_grad()
def tta_collect_probs(model, ds, tta_transforms, use_amp=False):
    """Average softmax probs over TTA views (one forward pass per view)."""
    model.eval()
    probs_all, labels_all = [], []
    for path, label in tqdm(ds.samples, desc=f"TTA ×{len(tta_transforms)}", leave=True):
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (IMG_SIZE_STAGE2, IMG_SIZE_STAGE2), 0)
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
print(f"\n🧪 Standard test eval")
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

# Confusion matrix (paper primary test view)
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
ax_r.plot(fpr_m, tpr_m, lw=2, label=f"Micro-average ROC (AUC = {auc_micro:.4f})")
ax_r.plot(all_fpr, mean_tpr, lw=2, linestyle="--", label=f"Macro-average ROC (AUC = {auc_macro:.4f})")
ax_r.plot([0, 1], [0, 1], "k:", lw=1, alpha=0.5)
ax_r.set_xlim([0.0, 1.0])
ax_r.set_ylim([0.0, 1.05])
ax_r.set_xlabel("False positive rate")
ax_r.set_ylabel("True positive rate")
ax_r.set_title(f"Multiclass ROC (OvR) — {eval_tag}", fontweight="bold")
ax_r.legend(loc="lower right")
ax_r.grid(alpha=0.3)
plt.tight_layout()
print("\n▶ Figure (eval): roc_multiclass_eval")
save_fig_notebook(fig_roc, FIG_DIR / "roc_multiclass_eval.png", dpi=120)

# Paper Figure: per-class F1 sorted (best/worst classes)
order = np.argsort(per_f1)
fig_f1, ax_f1 = plt.subplots(figsize=(10, max(8, NUM_CLASSES * 0.22)))
ax_f1.barh(np.arange(NUM_CLASSES), per_f1[order], color="#8e44ad")
ax_f1.set_yticks(np.arange(NUM_CLASSES))
ax_f1.set_yticklabels([class_list[i].replace("___", " | ") for i in order], fontsize=6)
ax_f1.set_xlabel("F1-score (%)")
ax_f1.set_title(f"Per-class F1 ({eval_tag})", fontweight="bold")
ax_f1.grid(axis="x", alpha=0.3)
plt.tight_layout()
print("▶ Figure (eval): per_class_f1")
save_fig_notebook(fig_f1, FIG_DIR / "per_class_f1.png", dpi=120)

# Paper Figure: top confusion pairs (off-diagonal counts)
cm_off = cm.copy()
np.fill_diagonal(cm_off, 0)
pairs = []
for i in range(NUM_CLASSES):
    for j in range(NUM_CLASSES):
        if cm_off[i, j] > 0:
            pairs.append((cm_off[i, j], i, j))
pairs = sorted(pairs, reverse=True)[:12]
if pairs:
    labels_pairs = [f"{class_list[i]} → {class_list[j]}" for _, i, j in pairs]
    values_pairs = [v for v, _, _ in pairs]
    fig_conf, ax_conf = plt.subplots(figsize=(12, 6))
    ax_conf.barh(np.arange(len(pairs)), values_pairs[::-1], color="#c0392b")
    ax_conf.set_yticks(np.arange(len(pairs)))
    ax_conf.set_yticklabels(labels_pairs[::-1], fontsize=7)
    ax_conf.set_xlabel("Misclassified images")
    ax_conf.set_title(f"Top confusion pairs ({eval_tag})", fontweight="bold")
    ax_conf.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    print("▶ Figure (eval): top_confusion_pairs")
    save_fig_notebook(fig_conf, FIG_DIR / "top_confusion_pairs.png", dpi=120)

# Robustness: Gaussian noise stress test
@torch.no_grad()
def evaluate_under_noise(model, loader, sigmas=(0.0, 0.05, 0.10, 0.20), use_amp=False):
    out = []
    model.eval()
    for sigma in sigmas:
        probs_all, labels_all = [], []
        for imgs, labels in tqdm(loader, desc=f"Noise sigma={sigma:.2f}", leave=False):
            imgs = imgs.to(DEVICE, non_blocking=True)
            if sigma > 0:
                imgs = torch.clamp(imgs + torch.randn_like(imgs) * sigma, -3.0, 3.0)
            with autocast(enabled=use_amp):
                logits = model(imgs)
            probs_all.append(F.softmax(logits, dim=1).cpu().numpy())
            labels_all.append(labels.numpy())
        probs = np.concatenate(probs_all, axis=0)
        labels_np = np.concatenate(labels_all, axis=0)
        preds = probs.argmax(axis=1)
        acc = accuracy_score(labels_np, preds) * 100
        y_bin = label_binarize(labels_np, classes=np.arange(NUM_CLASSES))
        fpr_n, tpr_n, _ = roc_curve(y_bin.ravel(), probs.ravel())
        auc_n = auc(fpr_n, tpr_n)
        out.append((sigma, acc, auc_n))
    return out

noise_results = evaluate_under_noise(model, test_loader, use_amp=USE_AMP)
print("\n🛡️ Robustness Evaluation (Gaussian noise)")
for sigma, acc_n, auc_n in noise_results:
    print(f"   Noise σ={sigma:.2f} -> Acc: {acc_n:.2f}%  AUC: {auc_n:.4f}")

fig_rob, (axr1, axr2) = plt.subplots(1, 2, figsize=(10.5, 4.2))
sigmas = [x[0] for x in noise_results]
accs = [x[1] for x in noise_results]
aucs = [x[2] for x in noise_results]
axr1.plot(sigmas, accs, marker="o", color="#f39c12")
axr2.plot(sigmas, aucs, marker="s", color="#e74c3c")
axr1.set_title("Robustness — Accuracy vs Noise", fontsize=10, fontweight="bold")
axr2.set_title("Robustness — AUC vs Noise", fontsize=10, fontweight="bold")
axr1.set_xlabel("Gaussian noise σ")
axr2.set_xlabel("Gaussian noise σ")
axr1.set_ylabel("Accuracy (%)")
axr2.set_ylabel("AUC")
axr1.grid(alpha=0.3); axr2.grid(alpha=0.3)
plt.suptitle("OmniCrops — Robustness Under Gaussian Noise", fontsize=12, fontweight="bold")
plt.tight_layout()
print("▶ Figure (eval): robustness_noise_curve")
save_fig_notebook(fig_rob, FIG_DIR / "robustness_noise_curve.png", dpi=120)

# XAI: gradient saliency maps (lightweight, architecture-agnostic)
def xai_saliency_grid(model, dataset, class_names, tf_eval, n_samples=6):
    model.eval()
    picks = random.sample(dataset.samples, k=min(n_samples, len(dataset.samples)))
    fig, axes = plt.subplots(len(picks), 2, figsize=(7, 3 * len(picks)))
    if len(picks) == 1:
        axes = np.array([axes])
    for i, (path, true_label) in enumerate(picks):
        try:
            pil_img = Image.open(path).convert("RGB")
        except Exception:
            pil_img = Image.new("RGB", (IMG_SIZE_STAGE2, IMG_SIZE_STAGE2), 0)
        x = tf_eval(pil_img).unsqueeze(0).to(DEVICE)
        x.requires_grad_(True)
        logits = model(x)
        pred = int(torch.argmax(logits, dim=1).item())
        score = logits[0, pred]
        model.zero_grad(set_to_none=True)
        score.backward()
        sal = x.grad.detach().abs().max(dim=1)[0].squeeze(0).cpu().numpy()
        sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-8)
        axes[i, 0].imshow(pil_img.resize((224, 224)))
        axes[i, 0].set_title(f"T:{class_names[true_label].split('___')[-1]} | P:{class_names[pred].split('___')[-1]}", fontsize=8)
        axes[i, 0].axis("off")
        axes[i, 1].imshow(pil_img.resize((224, 224)))
        axes[i, 1].imshow(sal, cmap="jet", alpha=0.45)
        axes[i, 1].set_title("Gradient saliency overlay", fontsize=8)
        axes[i, 1].axis("off")
    plt.suptitle("XAI — Saliency maps on test samples", fontsize=12, fontweight="bold")
    plt.tight_layout()
    return fig

if len(test_ds.samples) > 0:
    fig_xai = xai_saliency_grid(model, test_ds, class_list, val_tf, n_samples=6)
    print("▶ Figure (xai): saliency_grid")
    save_fig_notebook(fig_xai, FIG_DIR / "xai_saliency_grid.png", dpi=120)

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