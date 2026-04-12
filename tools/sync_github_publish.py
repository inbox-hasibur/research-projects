#!/usr/bin/env python3
"""
Copy selected project sources into github_publish/ for safe Git push.
Does not copy .env, data/, outputs — only listed .py files.

Run from anywhere:  python tools/sync_github_publish.py
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PUBLISH_ROOT = REPO_ROOT / "github_publish"

# project_dir -> list of .py filenames (under that folder)
PROJECT_FILES: dict[str, list[str]] = {
    "Project_OmniCrops_Multi-Crop_Disease_Detection": [
        "OmniCrops_Multi-Crop_Disease_Detection.py",
    ],
    "Project_Facial_Expression_Detection": [
        "CNN_Based_Face_Expression_Detection.py",
    ],
    "Project_Strawberry_Disease_Detection": [
        "Strawberry Disease Detection Model Comparison.py",
        "M1 - Strawberry disease detection Eff-Swin Hybrid.py",
        "dataset_analysis_dual_source.py",
    ],
    "Project_Glaucoma_Detection": [
        "M1 - GLAUCOMA DETECTION WITH VISION MAMBA (Vim).py",
        "M2 - Glaucoma detection with Hybrid ConvNext Swin.py",
        "M3 - Glaucoma EfficientNetV2-S + CBAM + GeM.py",
    ],
    "Project_CatNMF_Assets": [
        "CatNMF-Net Deep Learning Based Image Watermarking.py",
    ],
    "Project_Diabetic_Retinopathy": [
        "m1_diabetic_retinopathy_detection.py",
    ],
    "Project_Traffic_Detection": [
        "m1_yolov11m_dcnv3_cbam_ms_traffic_detection.py",
    ],
}


def main() -> int:
    copied = 0
    missing: list[str] = []
    for proj, files in PROJECT_FILES.items():
        src_dir = REPO_ROOT / proj
        dst_dir = PUBLISH_ROOT / proj
        if not src_dir.is_dir():
            missing.append(f"(missing dir) {proj}")
            continue
        dst_dir.mkdir(parents=True, exist_ok=True)
        for name in files:
            s = src_dir / name
            d = dst_dir / name
            if not s.is_file():
                missing.append(str(s.relative_to(REPO_ROOT)))
                continue
            shutil.copy2(s, d)
            copied += 1
            print(f"OK  {d.relative_to(REPO_ROOT)}")

    print("-" * 60)
    print(f"Copied {copied} file(s) -> {PUBLISH_ROOT.name}/")
    if missing:
        print("Skipped / missing:")
        for m in missing:
            print(f"  - {m}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
