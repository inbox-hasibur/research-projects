#!/usr/bin/env python3
# =============================================================================
# Strawberry dual-source dataset EDA (Usman Afzaal + PlantVillage Strawberry)
# =============================================================================
#
# What it does
#   - Inventory: splits, class counts (filename / folder mapping), JSON sidecars
#   - Merge view: unified class names, overlap (both / Afzaal-only / PV-only)
#   - Sampled image stats: resolution, file size, BG-separation proxy, Laplacian (optional)
#   - CSV exports + PNG plots under --out-dir
#
# Workflow (typical)
#   1. Edit this file locally (e.g. Cursor), upload/sync to Kaggle with your other code.
#   2. On Kaggle: add the Afzaal + PlantVillage datasets; defaults already use
#      /kaggle/input/... paths — run as a script cell or terminal:
#        !python dataset_analysis_dual_source.py
#   3. Locally (smoke test): point to copied data:
#        python dataset_analysis_dual_source.py --afzaal-root D:\data\afzaal --pv-root D:\data\pv\color
#
# Optional: from dataset_analysis_dual_source import run_analysis
#           run_analysis(out_dir="analysis_output")
#
# Note: CLI uses parse_known_args() so stray flags (e.g. some notebook runners)
#       do not crash the script; you can ignore that if you only use plain python.
#
# Deps:  pip install pillow matplotlib numpy pandas
# Opt:   pip install opencv-python   (Laplacian / blur proxy)
# =============================================================================

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

# -----------------------------------------------------------------------------
# 0. Optional third-party imports (fail fast with install hint)
# -----------------------------------------------------------------------------


def _import_plotting():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _import_pil():
    from PIL import Image

    return Image


def _import_pandas():
    import pandas as pd

    return pd


# -----------------------------------------------------------------------------
# 1. Constants & label mapping (aligned with training notebook LABEL_MAP)
# -----------------------------------------------------------------------------

IMAGE_EXT = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})

PV_FOLDER_TO_UNIFIED = {
    "Strawberry___Leaf_scorch": "leaf_scorch",
    "Strawberry___healthy": "healthy",
}

AFZAAL_CANONICAL = [
    "angular_leafspot",
    "anthracnose",
    "blossom_blight",
    "gray_mold",
    "leaf_spot",
    "powdery_mildew",
    "healthy",
    "leaf_scorch",
]


# -----------------------------------------------------------------------------
# 2. Afzaal: path scan & class-from-filename
# -----------------------------------------------------------------------------


def parse_afzaal_class_from_stem(stem: str) -> Optional[str]:
    """e.g. angular_leafspot359 → angular_leafspot (longest prefix match)."""
    s = stem.lower()
    best: Optional[str] = None
    best_len = 0
    for name in sorted(AFZAAL_CANONICAL, key=len, reverse=True):
        if not s.startswith(name):
            continue
        if len(s) == len(name):
            ok = True
        else:
            nxt = s[len(name)]
            ok = nxt == "_" or nxt.isdigit()
        if ok and len(name) > best_len:
            best, best_len = name, len(name)
    return best


def iter_images(root: Path) -> Iterable[Path]:
    if not root.is_dir():
        return
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXT:
            yield p


def collect_afzaal_by_split(afzaal_root: Path) -> Dict[str, List[Path]]:
    out: Dict[str, List[Path]] = defaultdict(list)
    for split in ("train", "val", "test"):
        d = afzaal_root / split
        for p in iter_images(d):
            out[split].append(p)
    sev_root = afzaal_root / "Test Disease Severity Level"
    if sev_root.is_dir():
        for lev in sorted(sev_root.iterdir()):
            if lev.is_dir() and lev.name.lower().startswith("level"):
                key = f"severity_{lev.name.replace(' ', '_')}"
                for p in iter_images(lev):
                    out[key].append(p)
    return out


def count_afzaal_classes(paths: List[Path]) -> Counter:
    c: Counter = Counter()
    unk = 0
    for p in paths:
        cls = parse_afzaal_class_from_stem(p.stem)
        if cls:
            c[cls] += 1
        else:
            unk += 1
    if unk:
        c["__unparsed_filename__"] = unk
    return c


# -----------------------------------------------------------------------------
# 3. PlantVillage: Strawberry_* under color/
# -----------------------------------------------------------------------------


def collect_plantvillage_strawberry(pv_color_root: Path) -> List[Tuple[str, Path]]:
    rows: List[Tuple[str, Path]] = []
    if not pv_color_root.is_dir():
        return rows
    for folder in sorted(pv_color_root.iterdir()):
        if not folder.is_dir() or not folder.name.startswith("Strawberry___"):
            continue
        unified = PV_FOLDER_TO_UNIFIED.get(folder.name)
        if unified is None:
            unified = folder.name.replace("Strawberry___", "").lower()
        for p in iter_images(folder):
            rows.append((unified, p))
    return rows


# -----------------------------------------------------------------------------
# 4. Image metrics (sampled)
# -----------------------------------------------------------------------------


@dataclass
class ImgMetrics:
    path: str
    w: int
    h: int
    mp: float
    aspect: float
    file_kb: float
    bg_sep: float
    lap_var: float


def compute_bg_separation_proxy(rgb: np.ndarray) -> float:
    """Border-ring mean RGB vs center crop — higher ≈ leaf separated from BG."""
    h, w = rgb.shape[:2]
    if h < 8 or w < 8:
        return 0.0
    br, bc = max(1, h // 20), max(1, w // 20)
    border = np.concatenate(
        [
            rgb[:br, :, :].reshape(-1, 3),
            rgb[-br:, :, :].reshape(-1, 3),
            rgb[:, :bc, :].reshape(-1, 3),
            rgb[:, -bc:, :].reshape(-1, 3),
        ],
        axis=0,
    )
    ch, cw = h // 4, w // 4
    cy1, cy2 = h // 2 - ch // 2, h // 2 + ch // 2
    cx1, cx2 = w // 2 - cw // 2, w // 2 + cw // 2
    center = rgb[cy1:cy2, cx1:cx2, :].reshape(-1, 3)
    return float(np.linalg.norm(border.mean(axis=0) - center.mean(axis=0)))


def laplacian_variance(rgb_uint8: np.ndarray) -> float:
    try:
        import cv2  # type: ignore
    except ImportError:
        return 0.0
    gray = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def measure_image(path: Path, Image_mod) -> Optional[ImgMetrics]:
    try:
        im = Image_mod.open(path).convert("RGB")
    except OSError:
        return None
    w, h = im.size
    arr = np.asarray(im, dtype=np.uint8)
    arr_f = arr.astype(np.float32) / 255.0
    kb = path.stat().st_size / 1024.0
    return ImgMetrics(
        path=str(path),
        w=w,
        h=h,
        mp=(w * h) / 1e6,
        aspect=w / max(h, 1),
        file_kb=kb,
        bg_sep=compute_bg_separation_proxy(arr_f),
        lap_var=laplacian_variance(arr),
    )


def sample_paths(paths: List[Path], k: int, seed: int) -> List[Path]:
    if len(paths) <= k:
        return list(paths)
    rng = random.Random(seed)
    return rng.sample(paths, k)


# -----------------------------------------------------------------------------
# 5. I/O helpers
# -----------------------------------------------------------------------------


def write_csv(path: Path, header: List[str], rows: List[List[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def imbalance_ratio(cnt: Counter) -> str:
    vals = [v for k, v in cnt.items() if not str(k).startswith("__")]
    if not vals:
        return "n/a"
    mx, mn = max(vals), min(vals)
    return f"max/min = {mx}/{mn} = {mx / max(mn, 1):.2f}x"


# -----------------------------------------------------------------------------
# 6. CLI (unknown extra argv tokens are ignored — robust on some hosted runners)
# -----------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Strawberry dual-dataset EDA (Afzaal + PlantVillage).",
        add_help=True,
    )
    p.add_argument(
        "--afzaal-root",
        default="/kaggle/input/datasets/usmanafzaal/strawberry-disease-detection-dataset",
        help="Usman Afzaal dataset root",
    )
    p.add_argument(
        "--pv-root",
        default="/kaggle/input/datasets/abdallahalidev/plantvillage-dataset/color",
        help="PlantVillage color/ root (Strawberry___* folders)",
    )
    p.add_argument("--out-dir", default="analysis_output", help="Output directory")
    p.add_argument(
        "--max-images-stats",
        type=int,
        default=400,
        help="Max images per source for pixel/file metrics",
    )
    p.add_argument("--seed", type=int, default=42)
    return p


def parse_cli_args(argv: Optional[List[str]] = None):
    """Parse our flags; ignore anything else (avoids SystemExit on extra argv)."""
    parser = build_arg_parser()
    args, unknown = parser.parse_known_args(argv)
    if unknown:
        # Typical: ['-f', '/path/to/kernel-....json']
        print(f"[EDA] Ignoring unrecognized argv tokens: {unknown}", file=sys.stderr)
    return args


# -----------------------------------------------------------------------------
# 7. Plots
# -----------------------------------------------------------------------------


def save_scatter_mp_vs_filekb(plt, out_dir: Path, af_m: List[ImgMetrics], pv_m: List[ImgMetrics]) -> None:
    if not (af_m and pv_m):
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter([m.mp for m in af_m], [m.file_kb for m in af_m], alpha=0.35, s=12, label="Afzaal")
    ax.scatter([m.mp for m in pv_m], [m.file_kb for m in pv_m], alpha=0.35, s=12, label="PlantVillage")
    ax.set_xlabel("Megapixels")
    ax.set_ylabel("File size (KB)")
    ax.legend()
    ax.set_title("Resolution vs file size (sampled)")
    fig.tight_layout()
    fig.savefig(out_dir / "scatter_mp_vs_filekb.png", dpi=150)
    plt.close(fig)


def save_boxplot_bg_sep(plt, out_dir: Path, af_m: List[ImgMetrics], pv_m: List[ImgMetrics]) -> None:
    if not (af_m and pv_m):
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.boxplot([[m.bg_sep for m in af_m], [m.bg_sep for m in pv_m]])
    ax.set_xticklabels(["Afzaal", "PlantVillage"])
    ax.set_ylabel("BG separation proxy (border vs center RGB L2)")
    ax.set_title("Background separation proxy")
    fig.tight_layout()
    fig.savefig(out_dir / "boxplot_bg_separation_proxy.png", dpi=150)
    plt.close(fig)


def save_boxplot_laplacian(plt, out_dir: Path, af_m: List[ImgMetrics], pv_m: List[ImgMetrics]) -> None:
    la = [m.lap_var for m in af_m if m.lap_var > 0]
    lp = [m.lap_var for m in pv_m if m.lap_var > 0]
    if not (la and lp):
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.boxplot([la, lp])
    ax.set_xticklabels(["Afzaal", "PlantVillage"])
    ax.set_ylabel("Laplacian variance (sharpness)")
    ax.set_title("Blur proxy (opencv-python)")
    fig.tight_layout()
    fig.savefig(out_dir / "boxplot_laplacian_var.png", dpi=150)
    plt.close(fig)


def save_bar_class_counts(
    plt,
    out_dir: Path,
    all_classes: List[str],
    unified_af: Counter,
    unified_pv: Counter,
) -> None:
    classes_plot = [c for c in all_classes if c != "__unparsed_filename__"]
    if not classes_plot:
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(classes_plot))
    width = 0.35
    ax.bar(x - width / 2, [unified_af.get(c, 0) for c in classes_plot], width=width, label="Afzaal")
    ax.bar(x + width / 2, [unified_pv.get(c, 0) for c in classes_plot], width=width, label="PlantVillage")
    ax.set_xticks(x)
    ax.set_xticklabels(classes_plot, rotation=35, ha="right")
    ax.legend()
    ax.set_title("Per-class image counts (unified names)")
    fig.tight_layout()
    fig.savefig(out_dir / "bar_class_counts_unified.png", dpi=150)
    plt.close(fig)


# -----------------------------------------------------------------------------
# 8. Core pipeline
# -----------------------------------------------------------------------------


def run_analysis(
    afzaal_root: str | Path = "/kaggle/input/datasets/usmanafzaal/strawberry-disease-detection-dataset",
    pv_root: str | Path = "/kaggle/input/datasets/abdallahalidev/plantvillage-dataset/color",
    out_dir: str | Path = "analysis_output",
    max_images_stats: int = 400,
    seed: int = 42,
) -> Path:
    """
    Run full EDA. Call from code if you do not want CLI (same defaults as Kaggle paths).

    Returns:
        Resolved output directory path.
    """
    plt = _import_plotting()
    Image_mod = _import_pil()
    pd = _import_pandas()

    afzaal_root = Path(afzaal_root)
    pv_root = Path(pv_root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    random.seed(seed)
    np.random.seed(seed)

    lines: List[str] = []

    def log(msg: str) -> None:
        print(msg)
        lines.append(msg)

    # --- 8a. Path sanity ----------------------------------------------------
    log("=== Paths ===")
    log(f"Afzaal root exists: {afzaal_root.is_dir()}  -> {afzaal_root}")
    log(f"PlantVillage color exists: {pv_root.is_dir()}  -> {pv_root}")

    # --- 8b. Afzaal inventory -------------------------------------------------
    af_by_split = collect_afzaal_by_split(afzaal_root)
    split_counters: Dict[str, Counter] = {sk: count_afzaal_classes(paths) for sk, paths in af_by_split.items()}

    log("\n=== Afzaal — image counts by split ===")
    total_af = 0
    for sk in sorted(af_by_split.keys()):
        n = len(af_by_split[sk])
        total_af += n
        log(f"  {sk}: {n}")
    log(f"  TOTAL images: {total_af}")

    all_af_paths: List[Path] = []
    for sk, paths in af_by_split.items():
        if sk.startswith("severity_"):
            continue
        all_af_paths.extend(paths)

    af_class_total = count_afzaal_classes(all_af_paths)
    log("\n=== Afzaal — class from filename (train+val+test) ===")
    for cls, n in af_class_total.most_common():
        log(f"  {cls}: {n}")

    for sk in sorted(af_by_split.keys()):
        if not sk.startswith("severity_"):
            continue
        log(f"\n=== Afzaal — {sk} ===")
        for cls, n in split_counters[sk].most_common():
            log(f"  {cls}: {n}")

    json_paths = list(afzaal_root.rglob("*.json"))
    json_files = [p for p in json_paths if p.is_file()]
    log(f"\n=== Afzaal — JSON files (recursive): {len(json_files)} ===")
    if json_files:
        sample_json = json_files[0]
        try:
            with sample_json.open(encoding="utf-8") as f:
                j = json.load(f)
            log(f"Sample JSON keys ({sample_json.name}): {list(j.keys())[:20]}")
        except json.JSONDecodeError as e:
            log(f"Sample JSON read failed: {e}")

    # --- 8c. PlantVillage inventory -----------------------------------------
    pv_rows = collect_plantvillage_strawberry(pv_root)
    pv_by_class: Counter = Counter(c for c, _ in pv_rows)
    log("\n=== PlantVillage (color) — Strawberry folders ===")
    for cls, n in pv_by_class.most_common():
        log(f"  {cls}: {n}")
    log(f"  TOTAL: {len(pv_rows)}")

    # --- 8d. Unified comparison & CSV ---------------------------------------
    unified_af = Counter({k: v for k, v in af_class_total.items() if not k.startswith("__")})
    unified_pv = Counter(pv_by_class)
    all_classes = sorted(set(unified_af) | set(unified_pv) | set(AFZAAL_CANONICAL))

    comp_rows = [
        [c, unified_af.get(c, 0), unified_pv.get(c, 0), unified_af.get(c, 0) + unified_pv.get(c, 0)]
        for c in all_classes
    ]
    write_csv(
        out_dir / "unified_class_comparison.csv",
        ["unified_class", "afzaal_train_val_test", "plantvillage_color", "combined"],
        comp_rows,
    )

    af_count_rows: List[List[object]] = []
    for sk in sorted(af_by_split.keys()):
        for cls, n in split_counters[sk].items():
            af_count_rows.append([sk, cls, n])
    write_csv(out_dir / "afzaal_counts.csv", ["split", "class", "count"], af_count_rows)

    write_csv(
        out_dir / "plantvillage_counts.csv",
        ["unified_class", "count"],
        [[c, n] for c, n in pv_by_class.most_common()],
    )

    log("\n=== Merge insight (unified_class) ===")
    only_af = [c for c in unified_af if unified_pv.get(c, 0) == 0 and not str(c).startswith("__")]
    only_pv = [c for c in unified_pv if unified_af.get(c, 0) == 0]
    both = [c for c in all_classes if unified_af.get(c, 0) > 0 and unified_pv.get(c, 0) > 0]
    log(f"  Classes in BOTH: {both}")
    log(f"  Only Afzaal: {only_af}")
    log(f"  Only PlantVillage: {only_pv}")

    log("\n=== Class imbalance ===")
    log(f"  Afzaal: {imbalance_ratio(unified_af)}")
    log(f"  PlantVillage: {imbalance_ratio(unified_pv)}")

    # --- 8e. Sampled image metrics ------------------------------------------
    af_sample = sample_paths(all_af_paths, max_images_stats, seed)
    pv_sample = sample_paths([p for _, p in pv_rows], max_images_stats, seed + 1)

    af_m = [m for p in af_sample if (m := measure_image(p, Image_mod)) is not None]
    pv_m = [m for p in pv_sample if (m := measure_image(p, Image_mod)) is not None]

    log(f"\n=== Image metrics (up to {max_images_stats} / source) ===")
    log(f"  Afzaal: {len(af_m)}  |  PlantVillage: {len(pv_m)}")

    def metrics_rows(ms: List[ImgMetrics], source: str) -> List[List[object]]:
        return [
            [source, m.path, m.w, m.h, m.mp, m.aspect, m.file_kb, m.bg_sep, m.lap_var]
            for m in ms
        ]

    write_csv(
        out_dir / "image_stats_sample_afzaal.csv",
        ["source", "path", "width", "height", "megapixels", "aspect", "file_kb", "bg_separation_proxy", "laplacian_var"],
        metrics_rows(af_m, "afzaal"),
    )
    write_csv(
        out_dir / "image_stats_sample_pv.csv",
        ["source", "path", "width", "height", "megapixels", "aspect", "file_kb", "bg_separation_proxy", "laplacian_var"],
        metrics_rows(pv_m, "plantvillage"),
    )

    def summarize(ms: List[ImgMetrics]) -> Dict[str, float]:
        if not ms:
            return {}
        df = pd.DataFrame(
            {
                "mp": [m.mp for m in ms],
                "file_kb": [m.file_kb for m in ms],
                "bg_sep": [m.bg_sep for m in ms],
                "lap_var": [m.lap_var for m in ms],
            }
        )
        return {
            "mp_mean": float(df["mp"].mean()),
            "mp_std": float(df["mp"].std()),
            "file_kb_mean": float(df["file_kb"].mean()),
            "bg_sep_mean": float(df["bg_sep"].mean()),
            "bg_sep_std": float(df["bg_sep"].std()),
            "lap_var_mean": float(df["lap_var"].mean()) if df["lap_var"].sum() > 0 else 0.0,
        }

    log("\n=== Domain proxy (higher bg_sep ≈ more studio-like BG contrast) ===")
    log(f"  Afzaal:       {summarize(af_m)}")
    log(f"  PlantVillage: {summarize(pv_m)}")

    # --- 8f. Figures ---------------------------------------------------------
    save_scatter_mp_vs_filekb(plt, out_dir, af_m, pv_m)
    save_boxplot_bg_sep(plt, out_dir, af_m, pv_m)
    save_boxplot_laplacian(plt, out_dir, af_m, pv_m)
    save_bar_class_counts(plt, out_dir, all_classes, unified_af, unified_pv)

    log(f"\n=== Saved ===\n  {out_dir.resolve()}")
    log("  CSV + PNG + summary.txt")

    summary_path = out_dir / "summary.txt"
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nDone. Summary: {summary_path}")
    return out_dir.resolve()


def main() -> None:
    args = parse_cli_args()
    run_analysis(
        afzaal_root=args.afzaal_root,
        pv_root=args.pv_root,
        out_dir=args.out_dir,
        max_images_stats=args.max_images_stats,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
