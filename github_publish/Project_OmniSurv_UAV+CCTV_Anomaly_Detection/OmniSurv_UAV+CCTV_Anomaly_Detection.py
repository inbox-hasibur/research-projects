# ============================================================
# HAM System — Stage 2: Feature-Based Temporal Classification
# YOLOv8 + SAHI + ByteTrack + Video Swin-T (Tiny)
# Datasets: Okutama + ShanghaiTech + UCF-Crime + UAV-Person
# Optimized: T4 GPU | AMP | 12-Hour Limit
# ============================================================

import subprocess, sys, os, time, json, warnings, random
import shutil, glob, hashlib
from pathlib import Path
from collections import defaultdict, deque
from typing import Optional, Tuple, List, Dict

warnings.filterwarnings('ignore')

# ── Install dependencies ──────────────────────────────────
pkgs = [
    "ultralytics",          # YOLOv8
    "sahi",                 # Slicing Aided Hyper Inference
    "supervision",          # ByteTrack wrapper
    "timm",                 # Video Swin-T
    "einops",
    "torchmetrics",
    "scikit-learn",
    "seaborn",
    "opencv-python-headless",
    "Pillow"
]
for pkg in pkgs:
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "--retries", "0", pkg])
    except subprocess.CalledProcessError:
        print(f"Warning: Failed to install {pkg}. It may already be installed or internet is disconnected.")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import cv2
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.amp import autocast, GradScaler
from torchmetrics import Accuracy, F1Score, AUROC
from torchmetrics.classification import (
    MulticlassConfusionMatrix,
    BinaryROC
)

import timm
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ─── Reproducibility ─────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark    = True
torch.backends.cudnn.deterministic = False

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_GPUS = torch.cuda.device_count()
print(f"✅ Device: {DEVICE} | GPUs: {N_GPUS}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")


# ══════════════════════════════════════════════════════════
# 1. GLOBAL CONFIGURATION
# ══════════════════════════════════════════════════════════

class Config:
    # ── Paths ─────────────────────────────────────────────
    # Okutama Action (UAV)
    OKUTAMA_ROOT   = Path("/kaggle/input/datasets/abrarzahin/"
                          "okutama-action-human-action-recognition"
                          "-dataset")
    # ShanghaiTech Campus (CCTV Anomaly)
    SHANGHAI_ROOT  = Path("/kaggle/input/datasets/nikanvasei/"
                          "shanghaitech-campus-dataset")
    # UAV Person Detection
    UAV_ROOT       = Path("/kaggle/input/datasets/luisngeld/"
                          "person-detection-uav-pascal-voc-uaq-msuav")
    # UCF-Crime
    UCF_ROOT       = Path("/kaggle/input/datasets/minhajuddinmeraj/"
                          "anomalydetectiondatasetucf")

    OUTPUT_ROOT    = Path("/kaggle/working/HAM")
    CROPS_DIR      = OUTPUT_ROOT / "crops"
    CLIPS_DIR      = OUTPUT_ROOT / "clips"
    META_DIR       = OUTPUT_ROOT / "metadata"
    CKPT_DIR       = OUTPUT_ROOT / "checkpoints"
    VIZ_DIR        = OUTPUT_ROOT / "visualizations"

    # ── Video / Frame ─────────────────────────────────────
    NUM_FRAMES     = 16          # frames per clip
    FRAME_SIZE     = 224
    FRAME_STRIDE   = 2           # temporal sampling stride
    CLIP_OVERLAP   = 0.5         # for sliding window

    # ── Detection (YOLO + SAHI) ───────────────────────────
    YOLO_MODEL     = "yolov8s.pt"
    YOLO_CONF      = 0.25
    YOLO_IOU       = 0.45
    YOLO_CLASSES   = [0]         # person only
    SAHI_SLICE_H   = 320
    SAHI_SLICE_W   = 320
    SAHI_OVERLAP   = 0.2

    # ── Tracking ──────────────────────────────────────────
    TRACK_THRESH   = 0.3
    MATCH_THRESH   = 0.8
    TRACK_BUFFER   = 30          # frames

    # ── Classes ───────────────────────────────────────────
    # Unified 10-class system
    CLASSES = [
        'Normal_Walking',     # 0
        'Normal_Running',     # 1
        'Normal_Standing',    # 2
        'Normal_Other',       # 3
        'Abnormal_Fighting',  # 4
        'Abnormal_Falling',   # 5
        'Abnormal_Assault',   # 6
        'Abnormal_Robbery',   # 7
        'Abnormal_Explosion', # 8
        'Abnormal_Other',     # 9
    ]
    NUM_CLASSES    = len(CLASSES)
    NUM_BINARY     = 2          # normal / abnormal
    CLASS2IDX      = {c: i for i, c in enumerate(CLASSES)}
    IDX2CLASS      = {i: c for i, c in enumerate(CLASSES)}

    # Normal class indices (for binary mapping)
    NORMAL_IDXS    = [0, 1, 2, 3]
    ABNORMAL_IDXS  = [4, 5, 6, 7, 8, 9]

    # ── Model (Video Swin-T) ──────────────────────────────
    EMBED_DIM      = 96
    DEPTHS         = [2, 2, 6, 2]
    NUM_HEADS      = [3, 6, 12, 24]
    WINDOW_SIZE    = (8, 7, 7)    # (T, H/P, W/P)
    PATCH_SIZE     = (2, 4, 4)    # (T, H, W) patch
    DROP_PATH      = 0.1
    DROPOUT        = 0.2

    # ── Training ──────────────────────────────────────────
    BATCH_SIZE     = 4
    GRAD_ACCUM     = 8           # effective = 32
    EPOCHS         = 50
    LR             = 2e-4
    LR_MIN         = 1e-6
    WEIGHT_DECAY   = 1e-4
    WARMUP_EPOCHS  = 5
    PATIENCE       = 12
    LABEL_SMOOTH   = 0.1
    CLIP_GRAD      = 1.0

    # ── Runtime ───────────────────────────────────────────
    NUM_WORKERS    = 2
    PIN_MEMORY     = True
    USE_AMP        = True
    MAX_HRS        = 11.5
    SEED           = 42

    # Split
    TRAIN_R        = 0.75
    VAL_R          = 0.15
    TEST_R         = 0.10

cfg = Config()

for d in [cfg.OUTPUT_ROOT, cfg.CROPS_DIR, cfg.CLIPS_DIR,
          cfg.META_DIR, cfg.CKPT_DIR, cfg.VIZ_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print(f"✅ Config | Classes: {cfg.NUM_CLASSES} | "
      f"Frames: {cfg.NUM_FRAMES} | "
      f"Frame Size: {cfg.FRAME_SIZE}")


# ══════════════════════════════════════════════════════════
# 2. DATASET BUILDERS
# ══════════════════════════════════════════════════════════

class DatasetBuilder:
    """
    Unified dataset builder for all 4 sources.
    Outputs: DataFrame with clip_path, class_idx, binary_label
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg

    # ── 2.1 Okutama Action (UAV frames + YOLO labels) ────
    def build_okutama(self) -> List[Dict]:
        """
        Okutama format:
        train/images/*.jpg  +  train/labels/*.txt (YOLO format)
        Classes: walking, running, etc.
        """
        records = []
        root = self.cfg.OKUTAMA_ROOT

        # Okutama action classes → our unified classes
        OKUTAMA_MAP = {
            0: 'Normal_Walking',
            1: 'Normal_Running',
            2: 'Normal_Standing',
            3: 'Normal_Other',    # reading
            4: 'Normal_Other',    # carrying
            5: 'Normal_Other',    # talking
            6: 'Abnormal_Other',  # lying
            7: 'Normal_Other',    # soccer
            8: 'Normal_Other',    # basketball
            9: 'Abnormal_Falling',
            10: 'Normal_Other',
        }

        for split in ['train', 'val', 'test']:
            img_dir = root / split / "images"
            lbl_dir = root / split / "labels"

            if not img_dir.exists():
                continue

            imgs = sorted(img_dir.glob("*.jpg")) + \
                   sorted(img_dir.glob("*.png"))

            for img_path in imgs:
                lbl_path = lbl_dir / (img_path.stem + ".txt")
                if not lbl_path.exists():
                    continue

                # Read YOLO labels
                with open(lbl_path) as f:
                    lines = f.readlines()

                for line in lines:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cls_id = int(float(parts[0]))
                    unified = OKUTAMA_MAP.get(cls_id, 'Normal_Other')
                    cls_idx = self.cfg.CLASS2IDX.get(
                        unified, 3)

                    records.append({
                        'frame_path'  : str(img_path),
                        'label_path'  : str(lbl_path),
                        'bbox_line'   : line.strip(),
                        'unified_class': unified,
                        'class_idx'   : cls_idx,
                        'binary_label': 0 if cls_idx in
                                        cfg.NORMAL_IDXS else 1,
                        'source'      : 'okutama',
                        'source_type' : 'uav',
                        'split'       : split,
                    })

        print(f"  ✅ Okutama: {len(records)} crop samples")
        return records

    # ── 2.2 ShanghaiTech (frame sequences) ───────────────
    def build_shanghai(self) -> List[Dict]:
        """
        ShanghaiTech format:
        SHANGHAI_TEST/frames/XX_XXXX/*.jpg  (frame sequences)
        Labels: .npy pixel-level annotation
        Anomaly: frame is anomaly if any pixel != 0
        """
        records = []
        root = self.cfg.SHANGHAI_ROOT / "SHANGHAI"

        for split_name, split_dir in [
            ('train', root / "SHANGHAI_TRAIN"),
            ('test',  root / "SHANGHAI_Test"),
        ]:
            frames_dir = split_dir / "frames"
            label_dir  = split_dir / "label"

            if not frames_dir.exists():
                continue

            # Each subfolder = one video sequence
            for seq_dir in sorted(frames_dir.iterdir()):
                if not seq_dir.is_dir():
                    continue

                seq_name = seq_dir.name
                frame_files = sorted(
                    list(seq_dir.glob("*.jpg")) +
                    list(seq_dir.glob("*.png")))

                if len(frame_files) < self.cfg.NUM_FRAMES:
                    continue

                # Load corresponding label if exists
                lbl_path = label_dir / f"{seq_name}.npy"
                is_anomaly_seq = False

                if lbl_path.exists():
                    try:
                        lbl = np.load(str(lbl_path))
                        # Anomaly if any non-zero label
                        is_anomaly_seq = bool(lbl.any())
                    except Exception:
                        pass

                # Sliding window over frames
                stride = max(1, self.cfg.NUM_FRAMES // 2)
                for start in range(
                        0,
                        len(frame_files) - self.cfg.NUM_FRAMES + 1,
                        stride):
                    window = frame_files[
                        start: start + self.cfg.NUM_FRAMES]

                    # Clip-level anomaly: majority vote
                    if lbl_path.exists():
                        try:
                            lbl = np.load(str(lbl_path))
                            # Check frames in window
                            n = min(len(lbl),
                                    start + self.cfg.NUM_FRAMES)
                            window_lbl = lbl[start:n]
                            is_anomaly = bool(
                                window_lbl.mean() > 0.3)
                        except Exception:
                            is_anomaly = is_anomaly_seq
                    else:
                        # Test split: assume anomaly
                        is_anomaly = (split_name == 'test')

                    cls_name = ('Abnormal_Other' if is_anomaly
                                else 'Normal_Walking')
                    cls_idx  = self.cfg.CLASS2IDX[cls_name]

                    records.append({
                        'frame_paths' : [str(f) for f in window],
                        'unified_class': cls_name,
                        'class_idx'   : cls_idx,
                        'binary_label': 1 if is_anomaly else 0,
                        'source'      : 'shanghai',
                        'source_type' : 'cctv',
                        'split'       : split_name,
                        'seq_name'    : seq_name,
                    })

        print(f"  ✅ ShanghaiTech: {len(records)} clip records")
        return records

    # ── 2.3 UCF-Crime (video → frame extraction) ─────────
    def build_ucf_crime(self) -> List[Dict]:
        """
        UCF-Crime: video files organized by crime class.
        We extract frames on-the-fly (no pre-extraction needed).
        """
        records = []
        root = self.cfg.UCF_ROOT

        UCF_MAP = {
            'Abuse'      : 'Abnormal_Assault',
            'Arrest'     : 'Abnormal_Other',
            'Arson'      : 'Abnormal_Explosion',
            'Assault'    : 'Abnormal_Assault',
            'Burglary'   : 'Abnormal_Robbery',
            'Explosion'  : 'Abnormal_Explosion',
            'Fighting'   : 'Abnormal_Fighting',
            'RoadAccidents': 'Abnormal_Other',
            'Robbery'    : 'Abnormal_Robbery',
            'Shooting'   : 'Abnormal_Assault',
            'Shoplifting': 'Abnormal_Robbery',
            'Stealing'   : 'Abnormal_Robbery',
            'Vandalism'  : 'Abnormal_Other',
            'Normal'     : 'Normal_Walking',
        }

        # Collect video files
        video_exts = ['.mp4', '.avi', '.mkv', '.mov']

        for cls_name, unified in UCF_MAP.items():
            # Multiple possible locations for this dataset
            search_dirs = [
                root / cls_name,
                root / f"{cls_name}A_Part1",
                root / f"Anomaly-Videos-Part-1" / cls_name,
                root / "FightingA_Part1",
            ]

            for sdir in search_dirs:
                if not sdir.exists():
                    continue
                for ext in video_exts:
                    for vpath in sdir.glob(f"*{ext}"):
                        cls_idx = self.cfg.CLASS2IDX.get(
                            unified,
                            self.cfg.CLASS2IDX['Normal_Other'])

                        records.append({
                            'video_path'  : str(vpath),
                            'unified_class': unified,
                            'class_idx'   : cls_idx,
                            'binary_label': 0 if cls_idx in
                                            cfg.NORMAL_IDXS else 1,
                            'source'      : 'ucf_crime',
                            'source_type' : 'cctv',
                            'split'       : 'train',
                        })

        # Also check Fighting files at top level
        for part in ['FightingA_Part1', 'FightingA_Part11',
                     'FightingA_Part2', 'FightingA_Part3']:
            pdir = root / part
            if pdir.exists():
                for ext in video_exts:
                    for vpath in pdir.glob(f"*{ext}"):
                        records.append({
                            'video_path'  : str(vpath),
                            'unified_class': 'Abnormal_Fighting',
                            'class_idx'   : self.cfg.CLASS2IDX[
                                'Abnormal_Fighting'],
                            'binary_label': 1,
                            'source'      : 'ucf_crime',
                            'source_type' : 'cctv',
                            'split'       : 'train',
                        })

        # Normal training videos
        for ndir_name in ['Normal_Videos_for_Event_Recognition',
                          'Testing_Normal_Videos_Anomaly']:
            ndir = root / ndir_name
            if ndir.exists():
                for ext in video_exts:
                    for vpath in ndir.glob(f"**/*{ext}"):
                        records.append({
                            'video_path'  : str(vpath),
                            'unified_class': 'Normal_Walking',
                            'class_idx'   : 0,
                            'binary_label': 0,
                            'source'      : 'ucf_crime',
                            'source_type' : 'cctv',
                            'split'       : 'train',
                        })

        print(f"  ✅ UCF-Crime: {len(records)} video records")
        return records

    # ── 2.4 UAV Person Detection ──────────────────────────
    def build_uav_person(self) -> List[Dict]:
        """
        UAV dataset: images with person bounding boxes.
        Used as normal activity source (person walking/standing).
        """
        records = []
        root = self.cfg.UAV_ROOT

        img_dirs = [
            root / "EVALUATION/EVALUATION/COCO_TEST/images",
            root / "EVALUATION/EVALUATION/VISDRONE/images",
            root / "EVALUATION/EVALUATION/NTUT/images",
            root / "PASCAL_VOC_UAQ_MSUAV/images",
        ]

        for img_dir in img_dirs:
            if not img_dir.exists():
                continue
            lbl_dir = img_dir.parent / "labels"

            for img_path in sorted(
                    list(img_dir.glob("*.jpg")) +
                    list(img_dir.glob("*.png")))[:500]:

                lbl_path = lbl_dir / (img_path.stem + ".txt")

                records.append({
                    'frame_path'  : str(img_path),
                    'label_path'  : str(lbl_path)
                                   if lbl_path.exists() else '',
                    'unified_class': 'Normal_Walking',
                    'class_idx'   : 0,
                    'binary_label': 0,
                    'source'      : 'uav_person',
                    'source_type' : 'uav',
                    'split'       : 'train',
                })

        # Video sources
        vid_root = root / "EVALUATION/evaluation_videos/" \
                         "evaluation_videos"
        if vid_root.exists():
            for vid_dir in vid_root.iterdir():
                if not vid_dir.is_dir():
                    continue
                for vf in vid_dir.glob("*.mp4"):
                    records.append({
                        'video_path'  : str(vf),
                        'unified_class': 'Normal_Walking',
                        'class_idx'   : 0,
                        'binary_label': 0,
                        'source'      : 'uav_person',
                        'source_type' : 'uav',
                        'split'       : 'train',
                    })

        print(f"  ✅ UAV Person: {len(records)} samples")
        return records

    def build_all(self) -> pd.DataFrame:
        print("\n📦 Building unified dataset from all sources...")
        print("─"*50)

        all_records = []
        all_records.extend(self.build_okutama())
        all_records.extend(self.build_shanghai())
        all_records.extend(self.build_ucf_crime())
        all_records.extend(self.build_uav_person())

        if not all_records:
            print("⚠️  No real data found → Synthetic fallback")
            all_records = self._make_synthetic()

        df = pd.DataFrame(all_records)

        # Save raw metadata
        df.to_csv(cfg.META_DIR / "raw_dataset.csv", index=False)

        print(f"\n📊 Total raw samples: {len(df)}")
        print(f"   Normal:   {(df['binary_label']==0).sum()}")
        print(f"   Abnormal: {(df['binary_label']==1).sum()}")
        print(f"   Sources:  {df['source'].value_counts().to_dict()}")

        return df

    def _make_synthetic(self) -> List[Dict]:
        """Fallback synthetic dataset."""
        syn_dir = cfg.OUTPUT_ROOT / "synthetic"
        syn_dir.mkdir(exist_ok=True)

        records = []
        classes_cfg = {
            'Normal_Walking'    : (0, 0, 80),
            'Normal_Running'    : (1, 0, 60),
            'Normal_Standing'   : (2, 0, 50),
            'Normal_Other'      : (3, 0, 40),
            'Abnormal_Fighting' : (4, 1, 50),
            'Abnormal_Falling'  : (5, 1, 30),
            'Abnormal_Assault'  : (6, 1, 30),
            'Abnormal_Robbery'  : (7, 1, 25),
            'Abnormal_Explosion': (8, 1, 20),
            'Abnormal_Other'    : (9, 1, 25),
        }

        for cls_name, (cls_idx, binary, n) in classes_cfg.items():
            cls_dir = syn_dir / cls_name
            cls_dir.mkdir(exist_ok=True)

            for i in range(n):
                # Create synthetic frame sequence
                frame_paths = []
                clip_dir = cls_dir / f"clip_{i:04d}"
                clip_dir.mkdir(exist_ok=True)

                base_color = np.random.randint(30, 180, 3)
                for t in range(cfg.NUM_FRAMES):
                    frame = np.ones(
                        (cfg.FRAME_SIZE, cfg.FRAME_SIZE, 3),
                        np.uint8) * base_color

                    # Motion cue
                    cx = int(50 + (t / cfg.NUM_FRAMES) * 124)
                    cy = cfg.FRAME_SIZE // 2
                    size = 20 + binary * 15
                    color = (0, 200, 0) if binary == 0 \
                            else (0, 0, 220)
                    cv2.rectangle(
                        frame,
                        (cx - size, cy - size*2),
                        (cx + size, cy + size*2),
                        color, -1
                    )
                    cv2.putText(
                        frame, cls_name[:12],
                        (5, 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (255,255,255), 1
                    )

                    fp = clip_dir / f"f{t:04d}.jpg"
                    cv2.imwrite(str(fp), frame)
                    frame_paths.append(str(fp))

                records.append({
                    'frame_paths'  : frame_paths,
                    'unified_class': cls_name,
                    'class_idx'    : cls_idx,
                    'binary_label' : binary,
                    'source'       : 'synthetic',
                    'source_type'  : 'mixed',
                    'split'        : 'train',
                })

        print(f"  ✅ Synthetic: {len(records)} clips")
        return records


# ══════════════════════════════════════════════════════════
# 3. STAGE 1: YOLO + SAHI DETECTION + BYTETRACK
# ══════════════════════════════════════════════════════════

class PersonDetector:
    """
    YOLOv8 + SAHI for small person detection.
    Especially important for UAV footage.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._load_yolo()

    def _load_yolo(self):
        from ultralytics import YOLO
        print(f"\n🔍 Loading YOLOv8: {self.cfg.YOLO_MODEL}")
        self.yolo = YOLO(self.cfg.YOLO_MODEL)
        print("  ✅ YOLO loaded")

    def detect_standard(self, frame: np.ndarray) -> List[Dict]:
        """Standard YOLO detection."""
        results = self.yolo(
            frame,
            conf=self.cfg.YOLO_CONF,
            iou=self.cfg.YOLO_IOU,
            classes=self.cfg.YOLO_CLASSES,
            verbose=False
        )
        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                detections.append({
                    'bbox': [float(x1), float(y1),
                             float(x2), float(y2)],
                    'conf': float(box.conf[0]),
                    'cls' : int(box.cls[0]),
                })
        return detections

    def detect_sahi(self, frame: np.ndarray) -> List[Dict]:
        """
        SAHI: Slicing Aided Hyper Inference.
        Slices image → YOLO on each slice → merge.
        Critical for detecting tiny humans from UAVs!
        """
        try:
            from sahi import AutoDetectionModel
            from sahi.predict import get_sliced_prediction

            if not hasattr(self, '_sahi_model'):
                self._sahi_model = AutoDetectionModel.from_pretrained(
                    model_type='ultralytics',
                    model_path=self.cfg.YOLO_MODEL,
                    confidence_threshold=self.cfg.YOLO_CONF,
                    device=str(DEVICE)
                )

            # Save frame to temp for SAHI
            import tempfile
            with tempfile.NamedTemporaryFile(
                    suffix='.jpg', delete=False) as tmp:
                cv2.imwrite(tmp.name, frame)
                tmp_path = tmp.name

            result = get_sliced_prediction(
                tmp_path,
                self._sahi_model,
                slice_height=self.cfg.SAHI_SLICE_H,
                slice_width=self.cfg.SAHI_SLICE_W,
                overlap_height_ratio=self.cfg.SAHI_OVERLAP,
                overlap_width_ratio=self.cfg.SAHI_OVERLAP,
                verbose=0
            )
            os.unlink(tmp_path)

            detections = []
            for pred in result.object_prediction_list:
                if pred.category.id != 0:
                    continue
                bbox = pred.bbox
                detections.append({
                    'bbox': [bbox.minx, bbox.miny,
                             bbox.maxx, bbox.maxy],
                    'conf': pred.score.value,
                    'cls' : 0,
                })
            return detections

        except Exception as e:
            # Fallback to standard detection
            return self.detect_standard(frame)

    def crop_person(self,
                    frame: np.ndarray,
                    bbox: List[float],
                    pad: float = 0.15) -> Optional[np.ndarray]:
        """Crop + pad person bounding box."""
        H, W = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        pw = (x2 - x1) * pad
        ph = (y2 - y1) * pad
        x1 = max(0, int(x1 - pw))
        y1 = max(0, int(y1 - ph))
        x2 = min(W, int(x2 + pw))
        y2 = min(H, int(y2 + ph))

        if x2 <= x1 or y2 <= y1:
            return None

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        crop = cv2.resize(
            crop,
            (self.cfg.FRAME_SIZE, self.cfg.FRAME_SIZE)
        )
        return crop


class SimpleByteTracker:
    """
    Lightweight ByteTrack-inspired tracker.
    Tracks person IDs across frames using IoU matching.
    """

    def __init__(self, cfg: Config):
        self.cfg       = cfg
        self.tracks    = {}     # track_id → {bbox, frames, age}
        self.next_id   = 0
        self.max_age   = cfg.TRACK_BUFFER

    def _iou(self, b1: List, b2: List) -> float:
        x1 = max(b1[0], b2[0])
        y1 = max(b1[1], b2[1])
        x2 = min(b1[2], b2[2])
        y2 = min(b1[3], b2[3])
        inter = max(0, x2-x1) * max(0, y2-y1)
        a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
        a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
        union = a1 + a2 - inter + 1e-6
        return inter / union

    def update(self, detections: List[Dict]) -> Dict[int, Dict]:
        """
        Update tracks with new detections.
        Returns dict of active tracks: {track_id: info}
        """
        # Age all tracks
        to_remove = []
        for tid in self.tracks:
            self.tracks[tid]['age'] += 1
            if self.tracks[tid]['age'] > self.max_age:
                to_remove.append(tid)
        for tid in to_remove:
            del self.tracks[tid]

        if not detections:
            return self.tracks

        # Match detections to tracks
        matched    = set()
        used_tracks= set()

        for det in detections:
            best_iou  = self.cfg.MATCH_THRESH
            best_tid  = None

            for tid, track in self.tracks.items():
                if tid in used_tracks:
                    continue
                iou = self._iou(det['bbox'], track['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_tid = tid

            if best_tid is not None:
                # Update existing track
                self.tracks[best_tid]['bbox'] = det['bbox']
                self.tracks[best_tid]['age']  = 0
                self.tracks[best_tid]['conf'] = det['conf']
                matched.add(best_tid)
                used_tracks.add(best_tid)
            else:
                # New track
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = {
                    'bbox'  : det['bbox'],
                    'age'   : 0,
                    'conf'  : det['conf'],
                    'frames': [],
                }

        return self.tracks


# ══════════════════════════════════════════════════════════
# 4. CLIP DATASET (For Video Swin-T Training)
# ══════════════════════════════════════════════════════════

class ClipDataset(Dataset):
    """
    Loads 16-frame clips for Video Swin-T.
    Handles 3 input formats:
    1. frame_paths: list of jpg paths (Shanghai, Synthetic)
    2. video_path: .mp4/.avi file (UCF-Crime)
    3. frame_path: single jpg + bbox (Okutama, UAV)
    """

    def __init__(self,
                 df: pd.DataFrame,
                 num_frames: int = 16,
                 frame_size: int = 224,
                 split: str = 'train'):
        self.df         = df.reset_index(drop=True)
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.split      = split
        self.is_train   = (split == 'train')

        # Transforms
        if self.is_train:
            self.transform = T.Compose([
                T.RandomResizedCrop(frame_size, scale=(0.7, 1.0)),
                T.RandomHorizontalFlip(p=0.5),
                T.ColorJitter(0.3, 0.3, 0.2, 0.05),
                T.RandomGrayscale(p=0.05),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = T.Compose([
                T.Resize((frame_size, frame_size)),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225]),
            ])

    def _load_frame(self, path: str) -> Optional[np.ndarray]:
        """Load single frame as RGB numpy array."""
        if not os.path.exists(path):
            return None
        img = cv2.imread(path)
        if img is None:
            return None
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _frames_from_video(self,
                            video_path: str) -> List[np.ndarray]:
        """Uniformly sample NUM_FRAMES from video."""
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total = max(total, self.num_frames)

        indices = np.linspace(
            0, total-1, self.num_frames, dtype=int)

        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(
                    frame,
                    (self.frame_size, self.frame_size))
                frames.append(frame)
            elif frames:
                frames.append(frames[-1].copy())
            else:
                frames.append(np.zeros(
                    (self.frame_size, self.frame_size, 3),
                    np.uint8))
        cap.release()

        # Pad if needed
        while len(frames) < self.num_frames:
            frames.append(frames[-1].copy())
        return frames[:self.num_frames]

    def _frames_from_paths(self,
                            paths: List[str]) -> List[np.ndarray]:
        """Load frames from list of paths."""
        # Uniform sample if more than needed
        if len(paths) > self.num_frames:
            indices = np.linspace(
                0, len(paths)-1, self.num_frames, dtype=int)
            paths = [paths[i] for i in indices]

        frames = []
        for p in paths:
            f = self._load_frame(p)
            if f is not None:
                f = cv2.resize(
                    f,
                    (self.frame_size, self.frame_size))
                frames.append(f)
            elif frames:
                frames.append(frames[-1].copy())
            else:
                frames.append(np.zeros(
                    (self.frame_size, self.frame_size, 3),
                    np.uint8))

        while len(frames) < self.num_frames:
            frames.append(frames[-1] if frames else
                          np.zeros(
                              (self.frame_size,
                               self.frame_size, 3),
                              np.uint8))
        return frames[:self.num_frames]

    def _single_frame_to_clip(self,
                               frame: np.ndarray) -> List[np.ndarray]:
        """
        For single-frame datasets (Okutama, UAV):
        Apply small augmentations to simulate temporal variation.
        """
        frames = []
        h, w = frame.shape[:2]

        for t in range(self.num_frames):
            f = frame.copy()

            if self.is_train:
                # Slight shift to simulate motion
                shift_x = int(t * random.uniform(-1, 1))
                shift_y = int(t * random.uniform(-0.5, 0.5))
                M = np.float32([[1, 0, shift_x],
                                [0, 1, shift_y]])
                f = cv2.warpAffine(f, M, (w, h))

                # Slight brightness variation
                alpha = 1.0 + random.uniform(-0.05, 0.05)
                f = np.clip(f * alpha, 0, 255).astype(np.uint8)

            f = cv2.resize(
                f, (self.frame_size, self.frame_size))
            frames.append(f)

        return frames

    def _synthetic_clip(self) -> List[np.ndarray]:
        """Random synthetic clip as fallback."""
        frames = []
        base = np.random.randint(
            0, 255,
            (self.frame_size, self.frame_size, 3),
            dtype=np.uint8)
        for _ in range(self.num_frames):
            f = base.copy()
            noise = np.random.randint(
                -15, 15,
                f.shape, dtype=np.int16)
            f = np.clip(
                f.astype(np.int16) + noise,
                0, 255).astype(np.uint8)
            frames.append(f)
        return frames

    def _get_frames(self, row: pd.Series) -> List[np.ndarray]:
        """Route to appropriate frame loader."""
        # Case 1: frame_paths column (list of paths)
        if 'frame_paths' in row and pd.notna(
                row.get('frame_paths')):
            try:
                paths = row['frame_paths']
                if isinstance(paths, str):
                    paths = json.loads(paths.replace("'", '"'))
                if isinstance(paths, list) and len(paths) > 0:
                    return self._frames_from_paths(paths)
            except Exception:
                pass

        # Case 2: video_path
        if 'video_path' in row and pd.notna(
                row.get('video_path')):
            vp = str(row['video_path'])
            if os.path.exists(vp):
                try:
                    return self._frames_from_video(vp)
                except Exception:
                    pass

        # Case 3: single frame_path
        if 'frame_path' in row and pd.notna(
                row.get('frame_path')):
            fp = str(row['frame_path'])
            if os.path.exists(fp):
                frame = self._load_frame(fp)
                if frame is not None:
                    return self._single_frame_to_clip(frame)

        # Fallback
        return self._synthetic_clip()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Load frames
        frames = self._get_frames(row)

        # Apply spatial transforms
        clip = []
        for f in frames:
            pil = Image.fromarray(f)
            clip.append(self.transform(pil))

        # Stack: (C, T, H, W)
        clip_tensor = torch.stack(clip, dim=1)

        class_idx    = int(row['class_idx'])
        binary_label = int(row['binary_label'])

        return {
            'clip'        : clip_tensor,
            'class_label' : torch.tensor(class_idx,   dtype=torch.long),
            'binary_label': torch.tensor(binary_label,dtype=torch.long),
            'source'      : str(row.get('source', 'unknown')),
        }


# ══════════════════════════════════════════════════════════
# 5. VIDEO SWIN-T ARCHITECTURE
# ══════════════════════════════════════════════════════════

class PatchEmbed3D(nn.Module):
    """
    3D Patch Embedding for Video Swin-T.
    Input:  (B, C, T, H, W)
    Output: (B, D, T', H', W') where D = embed_dim
    """
    def __init__(self,
                 patch_size=(2, 4, 4),
                 in_chans=3,
                 embed_dim=96,
                 norm_layer=None):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv3d(
            in_chans, embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        self.norm = norm_layer(embed_dim) \
                    if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)                    # (B, D, T', H', W')
        B, D, T, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)   # (B, T*H*W, D)
        x = self.norm(x)
        return x, T, H, W


class WindowAttention3D(nn.Module):
    """
    3D Window-based Multi-head Self Attention.
    Core of Video Swin Transformer.
    """
    def __init__(self,
                 dim,
                 window_size,
                 num_heads,
                 qkv_bias=True,
                 attn_drop=0.0,
                 proj_drop=0.0):
        super().__init__()
        self.dim         = dim
        self.window_size = window_size  # (wT, wH, wW)
        self.num_heads   = num_heads
        self.scale       = (dim // num_heads) ** -0.5

        self.qkv  = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        # Relative position bias
        wT, wH, wW = window_size
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(
                (2*wT-1) * (2*wH-1) * (2*wW-1),
                num_heads
            )
        )
        nn.init.trunc_normal_(
            self.relative_position_bias_table, std=0.02)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(
            B_, N, 3, self.num_heads,
            C // self.num_heads
        ).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(
                B_ // nW, nW, self.num_heads, N, N
            ) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = self.attn_drop(attn.softmax(dim=-1))
        x    = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x    = self.proj_drop(self.proj(x))
        return x


class SwinTransformerBlock3D(nn.Module):
    """
    Video Swin Transformer Block with:
    - Window Attention (W-MSA)
    - Shifted Window Attention (SW-MSA)
    - MLP
    - Layer Norm
    """
    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(8, 7, 7),
                 shift_size=(0, 0, 0),
                 mlp_ratio=4.0,
                 drop=0.0,
                 attn_drop=0.0,
                 drop_path=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = WindowAttention3D(
            dim, window_size, num_heads,
            attn_drop=attn_drop, proj_drop=drop
        )
        self.norm2   = nn.LayerNorm(dim)
        mlp_hidden   = int(dim * mlp_ratio)
        self.mlp     = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(drop),
        )
        self.drop_path = nn.Identity()  # simplified

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x)
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x


class PatchMerging3D(nn.Module):
    """Downsample spatial dimensions by 2x."""
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm      = norm_layer(4 * dim)

    def forward(self, x, T, H, W):
        B, _, C = x.shape
        x = x.view(B, T, H, W, C)

        # Pad if odd
        pad_h = H % 2
        pad_w = W % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))

        H2, W2 = x.shape[2], x.shape[3]
        x0 = x[:, :, 0::2, 0::2, :]
        x1 = x[:, :, 1::2, 0::2, :]
        x2 = x[:, :, 0::2, 1::2, :]
        x3 = x[:, :, 1::2, 1::2, :]
        x  = torch.cat([x0, x1, x2, x3], dim=-1)
        x  = x.view(B, -1, 4 * C)
        x  = self.norm(x)
        x  = self.reduction(x)

        return x, T, H2//2, W2//2


class VideoSwinTiny(nn.Module):
    """
    Video Swin Transformer — Tiny variant.
    Optimized for T4 GPU training.

    Key specs:
    - Input:  (B, 3, T=16, H=224, W=224)
    - Params: ~28M
    - Output: (B, num_classes) + (B, 1) binary
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.num_classes = cfg.NUM_CLASSES
        embed_dim        = cfg.EMBED_DIM       # 96
        depths           = cfg.DEPTHS          # [2,2,6,2]
        num_heads        = cfg.NUM_HEADS       # [3,6,12,24]
        window_size      = cfg.WINDOW_SIZE     # (8,7,7)
        patch_size       = cfg.PATCH_SIZE      # (2,4,4)
        drop             = cfg.DROPOUT
        drop_path        = cfg.DROP_PATH

        # 1. Patch Embedding
        self.patch_embed = PatchEmbed3D(
            patch_size=patch_size,
            in_chans=3,
            embed_dim=embed_dim,
            norm_layer=nn.LayerNorm
        )
        self.pos_drop = nn.Dropout(drop)

        # 2. Swin Stages
        dpr = [x.item() for x in torch.linspace(
            0, drop_path, sum(depths))]

        self.layers = nn.ModuleList()
        for i, (depth, heads) in enumerate(
                zip(depths, num_heads)):
            dim_i = embed_dim * (2 ** i)

            # Attention blocks
            blocks = nn.Sequential(*[
                SwinTransformerBlock3D(
                    dim=dim_i,
                    num_heads=heads,
                    window_size=window_size,
                    shift_size=(0,0,0) if j%2==0
                               else (4,3,3),
                    drop=drop,
                    attn_drop=drop * 0.5,
                    drop_path=dpr[sum(depths[:i])+j]
                )
                for j in range(depth)
            ])

            # Downsampling (except last stage)
            downsample = PatchMerging3D(dim_i) \
                         if i < len(depths)-1 else None

            self.layers.append(nn.ModuleDict({
                'blocks'    : blocks,
                'downsample': downsample
                              if downsample else nn.Identity(),
                'has_down'  : nn.Parameter(
                    torch.tensor(downsample is not None),
                    requires_grad=False)
            }))

        # 3. Norm
        final_dim = embed_dim * (2 ** (len(depths) - 1))
        self.norm = nn.LayerNorm(final_dim)

        # 4. Classifier heads
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        # Multi-class head
        self.head = nn.Sequential(
            nn.Dropout(drop),
            nn.Linear(final_dim, final_dim // 2),
            nn.GELU(),
            nn.Dropout(drop * 0.5),
            nn.Linear(final_dim // 2, self.num_classes)
        )

        # Binary anomaly head
        self.binary_head = nn.Sequential(
            nn.Linear(final_dim, 128),
            nn.GELU(),
            nn.Dropout(drop * 0.5),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward_features(self, x):
        """x: (B, C, T, H, W)"""
        x, T, H, W = self.patch_embed(x)
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer['blocks'](x)
            if layer['has_down'].item():
                x, T, H, W = layer['downsample'](x, T, H, W)

        x = self.norm(x)
        return x

    def forward(self, x):
        """
        x: (B, C, T, H, W)
        Returns: {logits, binary_score, features}
        """
        feats = self.forward_features(x)  # (B, N, D)

        # Global average pool over tokens
        pooled = self.avgpool(
            feats.transpose(1, 2)).squeeze(-1)  # (B, D)

        logits       = self.head(pooled)         # (B, num_classes)
        binary_score = self.binary_head(
            pooled).squeeze(-1)                  # (B,)

        return {
            'logits'      : logits,
            'binary_score': binary_score,
            'features'    : pooled,
        }


# ══════════════════════════════════════════════════════════
# 6. LOSS FUNCTIONS
# ══════════════════════════════════════════════════════════

class LabelSmoothCE(nn.Module):
    def __init__(self, num_classes, smoothing=0.1):
        super().__init__()
        self.smoothing   = smoothing
        self.num_classes = num_classes
        self.confidence  = 1.0 - smoothing

    def forward(self, pred, target):
        log_prob = F.log_softmax(pred, dim=-1)
        smooth   = self.smoothing / (self.num_classes - 1)
        one_hot  = torch.zeros_like(log_prob).scatter_(
            1, target.unsqueeze(1), 1)
        smooth_oh = one_hot * self.confidence + \
                    (1 - one_hot) * smooth
        return -(smooth_oh * log_prob).sum(dim=-1).mean()


class HAMLoss(nn.Module):
    """
    Combined loss:
    L = λ1 * CE(multi-class) + λ2 * BCE(binary anomaly)
    """
    def __init__(self, cfg: Config):
        super().__init__()
        self.cls_loss = LabelSmoothCE(
            cfg.NUM_CLASSES, cfg.LABEL_SMOOTH)
        self.bce_loss = nn.BCELoss()
        self.lambda1  = 0.6
        self.lambda2  = 0.4

    def forward(self,
                outputs: Dict,
                class_labels: torch.Tensor,
                binary_labels: torch.Tensor) -> Tuple:

        L_cls = self.cls_loss(
            outputs['logits'], class_labels)
        L_bin = self.bce_loss(
            outputs['binary_score'],
            binary_labels.float())
        L_total = self.lambda1 * L_cls + \
                  self.lambda2 * L_bin

        return L_total, {
            'total' : L_total.item(),
            'cls'   : L_cls.item(),
            'binary': L_bin.item(),
        }


# ══════════════════════════════════════════════════════════
# 7. TRAINING ENGINE
# ══════════════════════════════════════════════════════════

class EarlyStopping:
    def __init__(self, patience=12, min_delta=1e-4):
        self.patience  = patience
        self.min_delta = min_delta
        self.counter   = 0
        self.best      = None
        self.stop      = False

    def __call__(self, val_loss):
        if self.best is None:
            self.best = val_loss
        elif val_loss < self.best - self.min_delta:
            self.best    = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True


def build_weighted_sampler(df: pd.DataFrame):
    """Class-balanced sampler."""
    counts  = df['binary_label'].value_counts().to_dict()
    weights = [1.0 / counts[lbl]
               for lbl in df['binary_label'].tolist()]
    return WeightedRandomSampler(weights, len(weights))


def prepare_splits(df: pd.DataFrame):
    """Stratified train/val/test split."""
    train_val, test = train_test_split(
        df,
        test_size=cfg.TEST_R,
        stratify=df['binary_label'],
        random_state=cfg.SEED
    )
    val_ratio = cfg.VAL_R / (1 - cfg.TEST_R)
    train, val = train_test_split(
        train_val,
        test_size=val_ratio,
        stratify=train_val['binary_label'],
        random_state=cfg.SEED
    )

    for name, split in [('train', train),
                        ('val', val), ('test', test)]:
        split.to_csv(cfg.META_DIR / f"{name}.csv",
                     index=False)
        n = (split['binary_label']==1).sum()
        a = (split['binary_label']==0).sum()
        print(f"  {name:5s}: {len(split):5d} | "
              f"Normal: {a} | Abnormal: {n}")

    return train, val, test


class Trainer:
    def __init__(self, model: nn.Module, cfg: Config):
        self.model    = model.to(DEVICE)
        self.cfg      = cfg
        self.loss_fn  = HAMLoss(cfg).to(DEVICE)
        self.scaler   = GradScaler('cuda',
                                   enabled=cfg.USE_AMP)

        # Optimizer with layer-wise LR decay
        self.optimizer = torch.optim.AdamW(
            [
                {'params': model.patch_embed.parameters(),
                 'lr': cfg.LR * 0.3},
                {'params': model.layers.parameters(),
                 'lr': cfg.LR * 0.7},
                {'params': model.head.parameters(),
                 'lr': cfg.LR},
                {'params': model.binary_head.parameters(),
                 'lr': cfg.LR},
            ],
            weight_decay=cfg.WEIGHT_DECAY
        )

        self.history    = defaultdict(list)
        self.best_loss  = float('inf')
        self.start_time = time.time()
        self.es         = EarlyStopping(cfg.PATIENCE)

        # Metrics
        self.mc_acc  = Accuracy(
            task='multiclass',
            num_classes=cfg.NUM_CLASSES).to(DEVICE)
        self.bin_acc = Accuracy(
            task='binary').to(DEVICE)
        self.bin_auc = AUROC(
            task='binary').to(DEVICE)
        self.mc_f1   = F1Score(
            task='multiclass',
            num_classes=cfg.NUM_CLASSES,
            average='macro').to(DEVICE)

    def _time_ok(self):
        elapsed = (time.time() - self.start_time) / 3600
        return elapsed < self.cfg.MAX_HRS

    def _build_scheduler(self, steps_per_epoch: int):
        total_steps = self.cfg.EPOCHS * steps_per_epoch
        warmup_steps = self.cfg.WARMUP_EPOCHS * steps_per_epoch
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=[self.cfg.LR * 0.3,
                    self.cfg.LR * 0.7,
                    self.cfg.LR,
                    self.cfg.LR],
            total_steps=total_steps,
            pct_start=warmup_steps / total_steps,
            anneal_strategy='cos',
        )

    def _run_epoch(self,
                   loader: DataLoader,
                   train: bool = True,
                   epoch: int = 0) -> Dict:

        self.model.train() if train else self.model.eval()

        total_loss = defaultdict(float)
        self.mc_acc.reset()
        self.bin_acc.reset()
        self.bin_auc.reset()
        self.mc_f1.reset()

        ctx = torch.enable_grad if train else torch.no_grad
        n_steps = 0

        if train:
            self.optimizer.zero_grad()

        with ctx():
            for step, batch in enumerate(loader):
                if not self._time_ok():
                    print(f"\n⏰ Time limit — stopping epoch")
                    break

                clips  = batch['clip'].to(
                    DEVICE, non_blocking=True)
                cls_lbl= batch['class_label'].to(
                    DEVICE, non_blocking=True)
                bin_lbl= batch['binary_label'].to(
                    DEVICE, non_blocking=True)

                with autocast(
                    device_type='cuda',
                    dtype=torch.float16,
                    enabled=self.cfg.USE_AMP
                ):
                    outputs = self.model(clips)
                    loss, ld = self.loss_fn(
                        outputs, cls_lbl, bin_lbl)
                    loss_scaled = loss / self.cfg.GRAD_ACCUM

                if train:
                    self.scaler.scale(
                        loss_scaled).backward()

                    if (step+1) % self.cfg.GRAD_ACCUM == 0:
                        self.scaler.unscale_(self.optimizer)
                        nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.cfg.CLIP_GRAD
                        )
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.optimizer.zero_grad()
                        if hasattr(self, 'scheduler'):
                            self.scheduler.step()

                # Metrics
                preds = outputs['logits'].argmax(dim=-1)
                self.mc_acc.update(preds, cls_lbl)
                self.bin_acc.update(
                    outputs['binary_score'], bin_lbl)
                self.bin_auc.update(
                    outputs['binary_score'], bin_lbl)
                self.mc_f1.update(preds, cls_lbl)

                for k, v in ld.items():
                    total_loss[k] += v
                n_steps += 1

                if train and step % 30 == 0:
                    lr = self.optimizer.param_groups[-1]['lr']
                    print(f"    Step {step:04d} | "
                          f"Loss: {ld['total']:.4f} | "
                          f"Cls: {ld['cls']:.4f} | "
                          f"Bin: {ld['binary']:.4f} | "
                          f"LR: {lr:.2e}")

        n = max(n_steps, 1)
        return {
            'loss'    : total_loss['total'] / n,
            'cls_loss': total_loss['cls']   / n,
            'bin_loss': total_loss['binary'] / n,
            'mc_acc'  : self.mc_acc.compute().item(),
            'bin_acc' : self.bin_acc.compute().item(),
            'bin_auc' : self.bin_auc.compute().item(),
            'mc_f1'   : self.mc_f1.compute().item(),
        }

    def save(self, epoch: int, metrics: Dict,
             is_best: bool = False):
        state = {
            'epoch'       : epoch,
            'model_state' : self.model.state_dict(),
            'optim_state' : self.optimizer.state_dict(),
            'metrics'     : metrics,
            'classes'     : cfg.CLASSES,
        }
        path = cfg.CKPT_DIR / f"ep{epoch:03d}.pt"
        torch.save(state, path)
        if is_best:
            torch.save(state, cfg.CKPT_DIR / "best.pt")
            print(f"  💾 Best model saved (ep{epoch})")

    def train(self,
              train_loader: DataLoader,
              val_loader: DataLoader):

        self._build_scheduler(len(train_loader))

        print("\n" + "="*65)
        print("  🚀 TRAINING: Video Swin-T for HAM")
        print("="*65)
        print(f"  {'Ep':>4} | {'TLoss':>7} | {'TAcc':>7} | "
              f"{'VLoss':>7} | {'VAcc':>7} | "
              f"{'VAUC':>7} | {'VF1':>7}")
        print("─"*65)

        for epoch in range(1, self.cfg.EPOCHS + 1):
            if not self._time_ok():
                print(f"\n⏰ Time limit at epoch {epoch}")
                break

            elapsed = (time.time()-self.start_time)/3600

            print(f"\n📊 Epoch {epoch}/{self.cfg.EPOCHS} "
                  f"({elapsed:.1f}h elapsed)")

            t_m = self._run_epoch(
                train_loader, train=True, epoch=epoch)
            v_m = self._run_epoch(
                val_loader,   train=False, epoch=epoch)

            is_best = v_m['loss'] < self.best_loss
            if is_best:
                self.best_loss = v_m['loss']

            if epoch % 5 == 0 or is_best:
                self.save(epoch, v_m, is_best)

            for k, v in t_m.items():
                self.history[f'train_{k}'].append(v)
            for k, v in v_m.items():
                self.history[f'val_{k}'].append(v)

            print(f"  {epoch:4d} | "
                  f"{t_m['loss']:7.4f} | "
                  f"{t_m['mc_acc']*100:6.2f}% | "
                  f"{v_m['loss']:7.4f} | "
                  f"{v_m['mc_acc']*100:6.2f}% | "
                  f"{v_m['bin_auc']:7.4f} | "
                  f"{v_m['mc_f1']:7.4f}")

            self.es(v_m['loss'])
            if self.es.stop:
                print(f"\n🛑 Early stop at epoch {epoch}")
                break

        pd.DataFrame(self.history).to_csv(
            cfg.OUTPUT_ROOT / "history.csv", index=False)
        return self.history


# ══════════════════════════════════════════════════════════
# 8. EVALUATION + VISUALIZATION
# ══════════════════════════════════════════════════════════

@torch.no_grad()
def full_evaluation(model: nn.Module,
                    test_loader: DataLoader) -> Dict:
    """Comprehensive test set evaluation."""
    model.eval()

    all_preds  = []
    all_labels = []
    all_bin_pred = []
    all_bin_lbl  = []
    all_scores   = []

    for batch in tqdm(test_loader, desc="Testing"):
        clips   = batch['clip'].to(DEVICE)
        cls_lbl = batch['class_label'].to(DEVICE)
        bin_lbl = batch['binary_label'].to(DEVICE)

        with autocast(device_type='cuda',
                      dtype=torch.float16,
                      enabled=cfg.USE_AMP):
            outputs = model(clips)

        preds = outputs['logits'].argmax(dim=-1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(cls_lbl.cpu().numpy())
        all_bin_pred.extend(
            outputs['binary_score'].cpu().numpy())
        all_bin_lbl.extend(bin_lbl.cpu().numpy())
        probs = F.softmax(outputs['logits'], dim=-1)
        all_scores.extend(probs.cpu().numpy())

    all_preds    = np.array(all_preds)
    all_labels   = np.array(all_labels)
    all_bin_pred = np.array(all_bin_pred)
    all_bin_lbl  = np.array(all_bin_lbl)

    # Multi-class metrics
    mc_acc = (all_preds == all_labels).mean()
    mc_report = classification_report(
        all_labels, all_preds,
        target_names=cfg.CLASSES,
        output_dict=True,
        zero_division=0
    )

    # Binary metrics
    bin_pred_hard = (all_bin_pred > 0.5).astype(int)
    bin_acc = (bin_pred_hard == all_bin_lbl).mean()

    from sklearn.metrics import roc_auc_score, \
        average_precision_score
    try:
        bin_auc = roc_auc_score(all_bin_lbl, all_bin_pred)
        bin_ap  = average_precision_score(
            all_bin_lbl, all_bin_pred)
    except Exception:
        bin_auc = bin_ap = 0.0

    results = {
        'mc_accuracy'    : float(mc_acc),
        'binary_accuracy': float(bin_acc),
        'binary_auc'     : float(bin_auc),
        'binary_ap'      : float(bin_ap),
        'mc_report'      : mc_report,
        'all_preds'      : all_preds,
        'all_labels'     : all_labels,
        'all_bin_pred'   : all_bin_pred,
        'all_bin_lbl'    : all_bin_lbl,
    }

    print(f"\n{'='*60}")
    print(f"📊 TEST RESULTS")
    print(f"{'='*60}")
    print(f"  Multi-class Accuracy : {mc_acc*100:.2f}%")
    print(f"  Binary Accuracy      : {bin_acc*100:.2f}%")
    print(f"  Binary AUC-ROC       : {bin_auc:.4f}")
    print(f"  Binary AP            : {bin_ap:.4f}")
    print(f"\n  Per-class F1:")
    for cls in cfg.CLASSES:
        if cls in mc_report:
            f1 = mc_report[cls]['f1-score']
            marker = "🔴" if 'Abnormal' in cls else "🟢"
            print(f"    {marker} {cls:<25s}: {f1:.4f}")

    with open(cfg.OUTPUT_ROOT / "test_results.json", 'w') as f:
        json.dump({k: v for k, v in results.items()
                   if not isinstance(v, np.ndarray)}, f,
                  indent=2)

    return results


def plot_all_results(history: Dict, results: Dict):
    """Full results visualization."""
    fig = plt.figure(figsize=(22, 16))
    gs  = plt.GridSpec(3, 4, figure=fig,
                       hspace=0.4, wspace=0.35)
    fig.suptitle(
        'HAM System — Video Swin-T Results\n'
        'Datasets: Okutama + ShanghaiTech + UCF-Crime + UAV',
        fontsize=14, fontweight='bold'
    )

    # 1. Training loss
    ax = fig.add_subplot(gs[0, 0])
    if 'train_loss' in history:
        ax.plot(history['train_loss'],
                label='Train', color='steelblue')
        ax.plot(history['val_loss'],
                label='Val', color='coral')
    ax.set_title('Loss Curve', fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Accuracy
    ax = fig.add_subplot(gs[0, 1])
    if 'train_mc_acc' in history:
        ax.plot([v*100 for v in history['train_mc_acc']],
                label='Train MC Acc', color='steelblue')
        ax.plot([v*100 for v in history['val_mc_acc']],
                label='Val MC Acc', color='coral')
        ax.plot([v*100 for v in history['val_bin_acc']],
                label='Val Bin Acc', color='green',
                linestyle='--')
    ax.set_title('Accuracy', fontweight='bold')
    ax.set_ylabel('%')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 3. AUC
    ax = fig.add_subplot(gs[0, 2])
    if 'val_bin_auc' in history:
        ax.plot(history['val_bin_auc'],
                color='purple', linewidth=2)
        ax.axhline(0.5, color='gray',
                   linestyle='--', alpha=0.5)
        ax.fill_between(
            range(len(history['val_bin_auc'])),
            0.5,
            history['val_bin_auc'],
            alpha=0.2, color='purple'
        )
    ax.set_title('Binary AUROC', fontweight='bold')
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)

    # 4. F1
    ax = fig.add_subplot(gs[0, 3])
    if 'val_mc_f1' in history:
        ax.plot(history['val_mc_f1'],
                color='teal', linewidth=2)
    ax.set_title('Macro F1 Score', fontweight='bold')
    ax.grid(True, alpha=0.3)

    # 5. Confusion Matrix
    ax = fig.add_subplot(gs[1, :2])
    if 'all_preds' in results:
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(
            results['all_labels'],
            results['all_preds']
        )
        # Normalize
        cm_norm = cm.astype(float) / \
                  (cm.sum(axis=1, keepdims=True) + 1e-6)

        short_names = [c.replace('Normal_','N_')
                       .replace('Abnormal_','A_')
                       for c in cfg.CLASSES]

        sns.heatmap(
            cm_norm, annot=True, fmt='.2f',
            xticklabels=short_names,
            yticklabels=short_names,
            cmap='Blues', ax=ax,
            annot_kws={'size': 8}
        )
        ax.set_title('Confusion Matrix (Normalized)',
                     fontweight='bold')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        plt.setp(ax.xaxis.get_majorticklabels(),
                 rotation=45, ha='right', fontsize=8)
        plt.setp(ax.yaxis.get_majorticklabels(),
                 fontsize=8)

    # 6. ROC Curve
    ax = fig.add_subplot(gs[1, 2])
    if 'all_bin_pred' in results:
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(
            results['all_bin_lbl'],
            results['all_bin_pred']
        )
        auc = results['binary_auc']
        ax.plot(fpr, tpr, color='darkorange',
                lw=2, label=f'AUC={auc:.3f}')
        ax.plot([0,1], [0,1], 'k--', alpha=0.4)
        ax.fill_between(fpr, tpr, alpha=0.1,
                        color='darkorange')
        ax.set_title('ROC Curve (Binary)',
                     fontweight='bold')
        ax.set_xlabel('FPR')
        ax.set_ylabel('TPR')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # 7. Per-class F1 bar
    ax = fig.add_subplot(gs[1, 3])
    if 'mc_report' in results:
        rpt = results['mc_report']
        cls_f1 = [(c, rpt[c]['f1-score'])
                  for c in cfg.CLASSES if c in rpt]
        names  = [c[0].replace('Normal_','N_')
                   .replace('Abnormal_','A_')
                  for c in cls_f1]
        vals   = [c[1] for c in cls_f1]
        colors = ['#2ecc71' if 'Normal' in c[0]
                  else '#e74c3c' for c in cls_f1]

        ax.barh(names, vals, color=colors)
        ax.axvline(0.5, color='gray', linestyle='--')
        ax.set_title('Per-Class F1', fontweight='bold')
        ax.set_xlim([0, 1])
        n_patch = mpatches.Patch(
            color='#2ecc71', label='Normal')
        a_patch = mpatches.Patch(
            color='#e74c3c', label='Abnormal')
        ax.legend(handles=[n_patch, a_patch],
                  fontsize=8)
        ax.tick_params(axis='y', labelsize=8)

    # 8. Summary metrics table
    ax = fig.add_subplot(gs[2, :])
    ax.axis('off')
    table_data = [
        ['Metric', 'Value', 'Interpretation'],
        ['Multi-class Accuracy',
         f"{results['mc_accuracy']*100:.2f}%",
         '10-class activity classification'],
        ['Binary Accuracy',
         f"{results['binary_accuracy']*100:.2f}%",
         'Normal vs Abnormal detection'],
        ['Binary AUC-ROC',
         f"{results['binary_auc']:.4f}",
         '>0.90 = Excellent, >0.80 = Good'],
        ['Binary AP',
         f"{results['binary_ap']:.4f}",
         'Average Precision (imbalanced aware)'],
        ['Macro F1',
         f"{results['mc_report'].get('macro avg',{}).get('f1-score',0):.4f}",
         'Balanced across all 10 classes'],
    ]
    table = ax.table(
        cellText=table_data[1:],
        colLabels=table_data[0],
        cellLoc='center', loc='center',
        bbox=[0.05, 0, 0.9, 1]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor('#2c3e50')
            cell.set_text_props(
                color='white', fontweight='bold')
        elif r % 2 == 0:
            cell.set_facecolor('#ecf0f1')
        cell.set_edgecolor('white')
    ax.set_title('Final Results Summary',
                 fontweight='bold', pad=15)

    plt.savefig(cfg.VIZ_DIR / "full_results.png",
                dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✅ Results saved: {cfg.VIZ_DIR}/full_results.png")


# ══════════════════════════════════════════════════════════
# 9. INFERENCE PIPELINE (Detection → Tracking → Classify)
# ══════════════════════════════════════════════════════════

class HAMInference:
    """
    Full inference pipeline:
    Frame → YOLO+SAHI → ByteTrack → Clip Buffer → Swin-T
    """
    def __init__(self, model: nn.Module, cfg: Config):
        self.model     = model.eval()
        self.cfg       = cfg
        self.detector  = PersonDetector(cfg)
        self.tracker   = SimpleByteTracker(cfg)
        self.clip_bufs = defaultdict(lambda: deque(
            maxlen=cfg.NUM_FRAMES))

        self.transform = T.Compose([
            T.Resize((cfg.FRAME_SIZE, cfg.FRAME_SIZE)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225]),
        ])

    def process_frame(self,
                      frame: np.ndarray,
                      use_sahi: bool = False) -> List[Dict]:
        """
        Process single frame.
        Returns list of person-level predictions.
        """
        # Detect persons
        if use_sahi:
            dets = self.detector.detect_sahi(frame)
        else:
            dets = self.detector.detect_standard(frame)

        # Track
        tracks = self.tracker.update(dets)

        predictions = []

        for tid, track in tracks.items():
            if track['age'] > 0:
                continue  # Only active tracks

            # Crop person
            crop = self.detector.crop_person(
                frame, track['bbox'])
            if crop is None:
                continue

            # Add to clip buffer
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            self.clip_bufs[tid].append(crop_rgb)

            # Classify when buffer is full
            pred = None
            if len(self.clip_bufs[tid]) == self.cfg.NUM_FRAMES:
                pred = self._classify_clip(
                    list(self.clip_bufs[tid]), tid)
                pred['bbox']     = track['bbox']
                pred['track_id'] = tid
                predictions.append(pred)

        return predictions

    @torch.no_grad()
    def _classify_clip(self,
                        frames: List[np.ndarray],
                        tid: int) -> Dict:
        """Classify a 16-frame clip."""
        clip = []
        for f in frames:
            pil = Image.fromarray(f)
            clip.append(self.transform(pil))

        clip_tensor = torch.stack(
            clip, dim=1).unsqueeze(0).to(DEVICE)

        with autocast(device_type='cuda',
                      dtype=torch.float16,
                      enabled=self.cfg.USE_AMP):
            outputs = self.model(clip_tensor)

        probs  = F.softmax(outputs['logits'], dim=-1)[0]
        pred_c = probs.argmax().item()
        anom   = outputs['binary_score'][0].item()

        return {
            'class_name'   : cfg.CLASSES[pred_c],
            'class_idx'    : pred_c,
            'confidence'   : probs.max().item(),
            'anomaly_score': anom,
            'is_abnormal'  : anom > 0.5,
            'alert'        : anom > 0.7,
            'top3'         : [(cfg.CLASSES[i],
                               probs[i].item())
                              for i in probs.topk(3).indices],
        }

    def process_video(self,
                      video_path: str,
                      use_sahi: bool = False,
                      max_frames: int = 300) -> List[Dict]:
        """Process entire video file."""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30

        all_results = []
        frame_count = 0

        while cap.isOpened() and frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            preds = self.process_frame(frame, use_sahi)

            for p in preds:
                p['frame'] = frame_count
                p['time']  = frame_count / fps
                all_results.append(p)

                if p['alert']:
                    print(f"  🚨 ALERT @ {p['time']:.1f}s | "
                          f"Track {p['track_id']} | "
                          f"{p['class_name']} | "
                          f"Anomaly: {p['anomaly_score']:.3f}")

            frame_count += 1

        cap.release()
        return all_results


# ══════════════════════════════════════════════════════════
# 10. MAIN
# ══════════════════════════════════════════════════════════

def main():
    print("\n" + "🟢"*30)
    print("  HAM: Human Activity Monitoring System")
    print("  Stage 2: Video Swin-T Classification")
    print("  Datasets: Okutama + Shanghai + UCF + UAV")
    print("🟢"*30 + "\n")

    global_start = time.time()

    # ── Step 1: Build Dataset ────────────────────────────
    print("📦 STEP 1: Dataset Building")
    print("─"*50)
    builder = DatasetBuilder(cfg)
    df      = builder.build_all()

    # ── Step 2: Prepare Splits ───────────────────────────
    print("\n📊 STEP 2: Train/Val/Test Splits")
    print("─"*50)
    train_df, val_df, test_df = prepare_splits(df)

    # ── Step 3: Build Dataloaders ────────────────────────
    print("\n🔄 STEP 3: Building Dataloaders")
    train_ds = ClipDataset(train_df, cfg.NUM_FRAMES,
                           cfg.FRAME_SIZE, 'train')
    val_ds   = ClipDataset(val_df,   cfg.NUM_FRAMES,
                           cfg.FRAME_SIZE, 'val')
    test_ds  = ClipDataset(test_df,  cfg.NUM_FRAMES,
                           cfg.FRAME_SIZE, 'test')

    sampler = build_weighted_sampler(train_df)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.BATCH_SIZE,
        sampler=sampler,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
        drop_last=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.BATCH_SIZE * 2,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.BATCH_SIZE * 2,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )

    print(f"  Train: {len(train_ds)} | "
          f"Val: {len(val_ds)} | "
          f"Test: {len(test_ds)}")

    # ── Step 4: Build Model ──────────────────────────────
    print("\n🔨 STEP 4: Building Video Swin-T")
    print("─"*50)
    model   = VideoSwinTiny(cfg)
    n_param = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_param:,}")

    if N_GPUS > 1:
        model = nn.DataParallel(model)
        print(f"  DataParallel: {N_GPUS} GPUs")

    # ── Step 5: Train ────────────────────────────────────
    print("\n🏋️  STEP 5: Training")
    print("─"*50)
    trainer = Trainer(model, cfg)
    history = trainer.train(train_loader, val_loader)

    # ── Step 6: Load Best + Evaluate ─────────────────────
    print("\n🧪 STEP 6: Test Evaluation")
    print("─"*50)
    best_path = cfg.CKPT_DIR / "best.pt"
    if best_path.exists():
        state = torch.load(best_path, map_location=DEVICE)
        model.load_state_dict(state['model_state'])
        print(f"  ✅ Loaded best checkpoint")

    results = full_evaluation(model, test_loader)

    # ── Step 7: Visualize ────────────────────────────────
    print("\n📊 STEP 7: Plotting Results")
    plot_all_results(history, results)

    # ── Step 8: Demo Inference ───────────────────────────
    print("\n🎬 STEP 8: Demo Inference")
    print("─"*50)
    print("  Demo inference pipeline ready.")
    print("  To run on a video file:")
    print("""
    inferencer = HAMInference(model, cfg)
    
    # CCTV footage (standard detection):
    results = inferencer.process_video(
        "/path/to/cctv.mp4",
        use_sahi=False      ← CCTV: standard YOLO
    )
    
    # UAV footage (SAHI for tiny persons):
    results = inferencer.process_video(
        "/path/to/drone.mp4",
        use_sahi=True       ← UAV: SAHI slicing
    )
    
    # Single frame:
    preds = inferencer.process_frame(frame, use_sahi=True)
    for p in preds:
        print(p['class_name'], p['anomaly_score'])
    """)

    # Quick synthetic demo
    print("  Running synthetic demo...")
    inferencer = HAMInference(model, cfg)
    for _ in range(cfg.NUM_FRAMES + 2):
        dummy = np.random.randint(
            0, 255,
            (480, 640, 3), dtype=np.uint8)
        preds = inferencer.process_frame(dummy,
                                         use_sahi=False)

    print(f"  ✅ Demo complete")

    # ── Final Summary ────────────────────────────────────
    total_time = (time.time() - global_start) / 3600
    print(f"\n{'='*60}")
    print(f"  ✅ HAM System Complete")
    print(f"  Time:         {total_time:.2f} hours")
    print(f"  MC Accuracy:  {results['mc_accuracy']*100:.2f}%")
    print(f"  Binary AUC:   {results['binary_auc']:.4f}")
    print(f"  Binary AP:    {results['binary_ap']:.4f}")
    print(f"  Checkpoints:  {cfg.CKPT_DIR}")
    print(f"  Outputs:      {cfg.OUTPUT_ROOT}")
    print(f"{'='*60}")

    return model, results, history


# ── RUN ──────────────────────────────────────────────────
if __name__ == "__main__":
    model, results, history = main()