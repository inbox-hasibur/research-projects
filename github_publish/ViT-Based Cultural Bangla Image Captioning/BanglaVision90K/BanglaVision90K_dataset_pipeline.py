# BanglaVision90K pipeline with Qwen2-VL-7B-Instruct + MADLAD-400

from __future__ import annotations

import csv
import json
import os
import random
import re
import shutil
import gc
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import torch
from PIL import Image
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
)

try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    XLA_AVAILABLE = True
except Exception:
    XLA_AVAILABLE = False


# ---------- Config ----------
TARGET_IMAGES    = 90_000
CHECKPOINT_EVERY = 1_000
SEED             = 42

CAPTION_MODEL_CANDIDATES = [
    "Qwen/Qwen2-VL-7B-Instruct",
    "Qwen/Qwen2-VL-2B-Instruct",
]
TRANSLATE_MODEL_CANDIDATES = [
    "google/madlad400-3b-mt",
    "facebook/nllb-200-distilled-600M",
]

PROMPT_EN = (
    "Write one accurate, detailed, culturally aware English caption for this image. "
    "Mention key objects, actions, scene context, and visible details. "
    "Be specific and descriptive."
)

SOURCE_PREFIX = {
    "COCO-train2017"          : "cc",
    "COCO-val2017"            : "cc",
    "COCO-test2017"           : "cc",
    "Flickr30k"               : "flk",
    "BNATURE"                 : "bnat",
    "BanglaLekhaImageCaptions": "blk",
}


# ═══════════════════════════════════════════════════════════
#  Logging helper  — flush=True so Kaggle shows lines live
# ═══════════════════════════════════════════════════════════
_run_start = time.time()

def log(msg: str) -> None:
    elapsed = time.time() - _run_start
    h  = int(elapsed // 3600)
    m  = int((elapsed % 3600) // 60)
    s  = int(elapsed % 60)
    prefix = f"[{h:02d}:{m:02d}:{s:02d}]"
    print(f"{prefix} {msg}", flush=True)

def log_sep(title: str = "") -> None:
    line = "═" * 60
    if title:
        log(f"{line}")
        log(f"  {title}")
        log(f"{line}")
    else:
        log(line)


# ═══════════════════════════════════════════════════════════
#  Data classes
# ═══════════════════════════════════════════════════════════
@dataclass
class CaptionSample:
    image_path        : str
    source            : str
    output_image_name : str = ""
    en_caption        : str = ""
    bn_caption        : str = ""


# ═══════════════════════════════════════════════════════════
#  Device
# ═══════════════════════════════════════════════════════════
def get_device() -> torch.device:
    if XLA_AVAILABLE:
        return xm.xla_device()
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ═══════════════════════════════════════════════════════════
#  Text cleaning
# ═══════════════════════════════════════════════════════════
def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

def dedupe_repeated_phrases(text: str) -> str:
    text  = normalize_spaces(text)
    parts = re.split(r"(?<=[.!?।])\s+", text)
    seen: set = set()
    out  = []
    for p in parts:
        key = p.lower().strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(p.strip())
    return " ".join(out).strip()

def clean_caption(text: str) -> str:
    text = text.replace("\n", " ")
    text = normalize_spaces(text)
    text = dedupe_repeated_phrases(text)
    return text


# ═══════════════════════════════════════════════════════════
#  Dataset loaders
# ═══════════════════════════════════════════════════════════
def load_banglalekha(root: Path) -> List[CaptionSample]:
    image_dir = root / "images"
    if not image_dir.exists():
        log(f"  [WARN] BanglaLekha not found at {image_dir}")
        return []
    samples = [
        CaptionSample(image_path=str(p), source="BanglaLekhaImageCaptions")
        for p in image_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    ]
    log(f"  BanglaLekha   → {len(samples):>6} images")
    return samples

def load_flickr30k(root: Path) -> List[CaptionSample]:
    image_dir = root / "Images" / "flickr30k_images"
    if not image_dir.exists():
        log(f"  [WARN] Flickr30k not found at {image_dir}")
        return []
    samples = [
        CaptionSample(image_path=str(p), source="Flickr30k")
        for p in image_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    ]
    log(f"  Flickr30k     → {len(samples):>6} images")
    return samples

def load_bnature(root: Path) -> List[CaptionSample]:
    image_dir = root / "Pictures"
    if not image_dir.exists():
        log(f"  [WARN] BNature not found at {image_dir}")
        return []
    samples = [
        CaptionSample(image_path=str(p), source="BNATURE")
        for p in image_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    ]
    log(f"  BNature       → {len(samples):>6} images")
    return samples

def load_coco(root: Path) -> List[CaptionSample]:
    samples = []
    for split in ("train2017", "val2017", "test2017"):
        image_dir = root / split
        if not image_dir.exists():
            log(f"  [WARN] COCO/{split} not found")
            continue
        chunk = [
            CaptionSample(image_path=str(p), source=f"COCO-{split}")
            for p in image_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        ]
        log(f"  COCO-{split:<12} → {len(chunk):>6} images")
        samples.extend(chunk)
    return samples


# ═══════════════════════════════════════════════════════════
#  Manifest builder
# ═══════════════════════════════════════════════════════════
def build_unified_manifest(target_images: int = TARGET_IMAGES) -> List[CaptionSample]:
    log_sep("STEP 1 — Building image manifest")
    random.seed(SEED)

    banglalekha_root = Path("/kaggle/input/datasets/ezharuddinjubaer/banglalekhaimagecaptions")
    flickr_root      = Path("/kaggle/input/datasets/adityajn105/flickr30k")
    bnature_root     = Path("/kaggle/input/datasets/almominfaruk/bnaturebengali-image-captioning-dataset")
    coco_root        = Path("/kaggle/input/datasets/awsaf49/coco-2017-dataset/coco2017")

    pools = [
        load_banglalekha(banglalekha_root),
        load_flickr30k(flickr_root),
        load_bnature(bnature_root),
        load_coco(coco_root),
    ]

    all_samples: List[CaptionSample] = []
    for pool in pools:
        random.shuffle(pool)
        all_samples.extend(pool)

    random.shuffle(all_samples)
    if len(all_samples) > target_images:
        all_samples = all_samples[:target_images]

    for idx, sample in enumerate(all_samples):
        original_name = Path(sample.image_path).name
        prefix = SOURCE_PREFIX.get(sample.source, "mix")
        sample.output_image_name = f"{prefix}_{idx:06d}_{original_name}"

    log(f"Manifest ready → {len(all_samples)} images selected (target={target_images})")
    return all_samples


def batch_iter(items: List[CaptionSample], batch_size: int) -> Iterable[List[CaptionSample]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


# ═══════════════════════════════════════════════════════════
#  HF Auth
# ═══════════════════════════════════════════════════════════
def configure_hf_auth() -> None:
    token = None
    try:
        from kaggle_secrets import UserSecretsClient
        token = UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        token = os.environ.get("HF_TOKEN")

    if token:
        try:
            from huggingface_hub import login
            login(token=token)
            log("HuggingFace login successful.")
        except Exception as e:
            log(f"HF login failed (continuing): {e}")
    else:
        log("No HF token found — using public models only.")


# ═══════════════════════════════════════════════════════════
#  Model loaders
# ═══════════════════════════════════════════════════════════
def create_caption_model(device: torch.device):
    log_sep("Loading caption model (Qwen2-VL)")
    last_error = None
    dtype = torch.bfloat16

    for model_id in CAPTION_MODEL_CANDIDATES:
        log(f"  Trying {model_id} ...")
        t0 = time.time()
        try:
            processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_id,
                dtype=dtype,
                device_map="cpu",       # load to CPU first, then move
            )
            model = model.to(device)
            model.eval()
            log(f"  ✓ Loaded {model_id} in {time.time()-t0:.1f}s")
            return processor, model
        except Exception as e:
            last_error = e
            log(f"  ✗ Failed ({type(e).__name__}): {e}")
            gc.collect()
            if XLA_AVAILABLE:
                xm.mark_step()

    raise RuntimeError("Could not load any caption model.") from last_error


def create_translate_model(device: torch.device):
    log_sep("Loading translation model (MADLAD / NLLB)")
    last_error = None
    dtype = torch.bfloat16 if str(device) != "cpu" else torch.float32

    for model_id in TRANSLATE_MODEL_CANDIDATES:
        log(f"  Trying {model_id} ...")
        t0 = time.time()
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_id, dtype=dtype)
            model = model.to(device)
            model.eval()
            mode = "nllb" if "nllb" in model_id.lower() else "madlad"
            log(f"  ✓ Loaded {model_id} in {time.time()-t0:.1f}s  (mode={mode})")
            return tokenizer, model, mode
        except Exception as e:
            last_error = e
            log(f"  ✗ Failed ({type(e).__name__}): {e}")
            gc.collect()
            if XLA_AVAILABLE:
                xm.mark_step()

    raise RuntimeError("Could not load any translation model.") from last_error


# ═══════════════════════════════════════════════════════════
#  Caption generation
# ═══════════════════════════════════════════════════════════
def build_qwen_messages(image: Image.Image, prompt: str) -> list:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text",  "text": prompt},
            ],
        }
    ]


def generate_english_captions(
    batch       : List[CaptionSample],
    processor,
    model,
    device      : torch.device,
) -> None:
    images       = [load_image(x.image_path) for x in batch]
    all_messages = [build_qwen_messages(img, PROMPT_EN) for img in images]

    texts = [
        processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        for msgs in all_messages
    ]

    inputs = processor(text=texts, images=images, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # torch.no_grad() — NOT inference_mode() — required for XLA compatibility
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens      = 80,
            num_beams           = 4,
            repetition_penalty  = 1.2,
            no_repeat_ngram_size= 3,
            length_penalty      = 0.95,
            do_sample           = False,
        )

    input_len  = inputs["input_ids"].shape[1]
    new_tokens = out[:, input_len:]
    decoded    = processor.batch_decode(new_tokens, skip_special_tokens=True)

    for sample, text in zip(batch, decoded):
        sample.en_caption = clean_caption(text)

    if XLA_AVAILABLE:
        xm.mark_step()


# ═══════════════════════════════════════════════════════════
#  Translation
# ═══════════════════════════════════════════════════════════
def translate_en_to_bn(
    batch    : List[CaptionSample],
    tokenizer,
    model,
    device   : torch.device,
    mode     : str,
) -> None:
    if mode == "nllb":
        tokenizer.src_lang = "eng_Latn"
        texts  = [x.en_caption for x in batch]
        inputs = tokenizer(texts, return_tensors="pt",
                           padding=True, truncation=True, max_length=256)
        forced_bos_token_id = tokenizer.convert_tokens_to_ids("ben_Beng")
    else:
        texts  = [f"<2bn> {x.en_caption}" for x in batch]
        inputs = tokenizer(texts, return_tensors="pt",
                           padding=True, truncation=True, max_length=256)
        forced_bos_token_id = None

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens       = 96,
            num_beams            = 5,
            repetition_penalty   = 1.2,
            no_repeat_ngram_size = 3,
            length_penalty       = 1.0,
            forced_bos_token_id  = forced_bos_token_id,
        )
        decoded = tokenizer.batch_decode(out, skip_special_tokens=True)

    for sample, text in zip(batch, decoded):
        sample.bn_caption = clean_caption(text)

    if XLA_AVAILABLE:
        xm.mark_step()


# ═══════════════════════════════════════════════════════════
#  I/O helpers
# ═══════════════════════════════════════════════════════════
def copy_images_to_output(image_dir: Path, rows: List[CaptionSample]) -> None:
    image_dir.mkdir(parents=True, exist_ok=True)
    for r in rows:
        dst = image_dir / r.output_image_name
        if not dst.exists():
            shutil.copy2(r.image_path, dst)

def write_caption_rows(path: Path, rows: List[CaptionSample], append: bool) -> None:
    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if not append:
            writer.writerow(["image_name", "en_caption", "bn_caption"])
        for r in rows:
            writer.writerow([r.output_image_name, r.en_caption, r.bn_caption])

def write_source_rows(path: Path, rows: List[CaptionSample], append: bool) -> None:
    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if not append:
            writer.writerow(["source", "image_name"])
        for r in rows:
            writer.writerow([r.source, r.output_image_name])

def write_en_rows(path: Path, rows: List[CaptionSample], append: bool) -> None:
    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if not append:
            writer.writerow(["image_name", "en_caption"])
        for r in rows:
            writer.writerow([r.output_image_name, r.en_caption])

def read_en_rows(path: Path) -> List[CaptionSample]:
    rows = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(CaptionSample(
                image_path="", source="",
                output_image_name=row["image_name"],
                en_caption=row["en_caption"],
            ))
    return rows


# ═══════════════════════════════════════════════════════════
#  Speed estimator
# ═══════════════════════════════════════════════════════════
class SpeedTracker:
    """Tracks images/sec and prints ETA every N images."""

    def __init__(self, total: int, print_every: int = 100) -> None:
        self.total       = total
        self.print_every = print_every
        self.start       = time.time()
        self.last_log    = time.time()
        self.last_count  = 0

    def update(self, done: int) -> None:
        if done % self.print_every != 0:
            return
        now      = time.time()
        elapsed  = now - self.start
        interval = now - self.last_log
        delta    = done - self.last_count

        speed_overall = done / elapsed if elapsed > 0 else 0
        speed_recent  = delta / interval if interval > 0 else 0
        remaining     = self.total - done
        eta_sec       = remaining / speed_recent if speed_recent > 0 else 0
        eta_h         = int(eta_sec // 3600)
        eta_m         = int((eta_sec % 3600) // 60)
        pct           = 100.0 * done / self.total

        log(
            f"  Progress: {done:>6}/{self.total}  ({pct:5.1f}%)  "
            f"speed={speed_recent:.2f} img/s  "
            f"ETA={eta_h:02d}h{eta_m:02d}m"
        )
        self.last_log   = now
        self.last_count = done


# ═══════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════
def main() -> None:
    log_sep("BanglaVision90K — Qwen2-VL-7B + MADLAD-400")

    output_dir       = Path("/kaggle/working/banglavision90k_outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_images_dir = output_dir / "BanglaVision90K" / "images"
    en_csv_path       = output_dir / "banglavision90k_en_only.csv"
    csv_path          = output_dir / "banglavision90k_captions.csv"
    source_csv_path   = output_dir / "banglavision90k_source_name.csv"
    state_path        = output_dir / "state.json"

    samples = build_unified_manifest(TARGET_IMAGES)

    if not samples:
        log("ERROR: No images found. Check dataset mount paths.")
        return

    # ── Resume state ──────────────────────────────────────
    phase          = "caption"
    caption_done   = 0
    translate_done = 0
    if state_path.exists():
        state          = json.loads(state_path.read_text(encoding="utf-8"))
        phase          = state.get("phase", "caption")
        caption_done   = int(state.get("caption_done", 0))
        translate_done = int(state.get("translate_done", 0))
        log_sep("Resuming previous run")
        log(f"  phase          = {phase}")
        log(f"  caption_done   = {caption_done}/{len(samples)}")
        log(f"  translate_done = {translate_done}")
    else:
        log("Fresh run — no previous state found.")

    if not source_csv_path.exists():
        write_source_rows(source_csv_path, samples, append=False)
        log(f"Source manifest written → {source_csv_path}")

    device = get_device()
    log(f"Device: {device}  |  XLA available: {XLA_AVAILABLE}")
    configure_hf_auth()

    caption_batch_size   = 1   # Qwen2-VL-7B @bfloat16 ≈14 GB — keep batch=1
    translate_batch_size = 4

    # ══════════════════════════════════════════════════════
    #  PHASE 1 — CAPTION
    # ══════════════════════════════════════════════════════
    if phase == "caption":
        log_sep(f"PHASE 1 — Captioning  (start={caption_done}, total={len(samples)})")
        caption_processor, caption_model = create_caption_model(device)

        en_append_mode = caption_done > 0 and en_csv_path.exists()
        processed      = caption_done
        tracker        = SpeedTracker(total=len(samples), print_every=100)

        for chunk in batch_iter(samples[caption_done:], caption_batch_size):
            generate_english_captions(chunk, caption_processor, caption_model, device)
            copy_images_to_output(output_images_dir, chunk)
            write_en_rows(en_csv_path, chunk, append=en_append_mode)
            en_append_mode = True
            processed     += len(chunk)

            # ── per-image live log (every 100) ──
            tracker.update(processed)

            # ── checkpoint every 1 000 ──
            if processed % CHECKPOINT_EVERY == 0:
                state_path.write_text(
                    json.dumps({
                        "phase": "caption",
                        "caption_done": processed,
                        "translate_done": translate_done,
                    }, ensure_ascii=False),
                    encoding="utf-8",
                )
                log(f"  ✔ Checkpoint saved  [{processed}/{len(samples)}]")

        # phase transition
        state_path.write_text(
            json.dumps({
                "phase": "translate",
                "caption_done": processed,
                "translate_done": translate_done,
            }, ensure_ascii=False),
            encoding="utf-8",
        )
        log_sep("PHASE 1 complete — all captions generated")
        log(f"  Total captioned: {processed}")

        del caption_model, caption_processor
        gc.collect()
        if XLA_AVAILABLE:
            xm.mark_step()

    # ══════════════════════════════════════════════════════
    #  PHASE 2 — TRANSLATE
    # ══════════════════════════════════════════════════════
    log_sep("PHASE 2 — Translation EN → BN")
    en_rows = read_en_rows(en_csv_path)
    log(f"  Rows to translate: {len(en_rows)}  (already done: {translate_done})")

    trans_tokenizer, trans_model, translate_mode = create_translate_model(device)
    append_mode         = translate_done > 0 and csv_path.exists()
    processed_translate = translate_done
    tracker2            = SpeedTracker(total=len(en_rows), print_every=200)

    for chunk in batch_iter(en_rows[translate_done:], translate_batch_size):
        translate_en_to_bn(chunk, trans_tokenizer, trans_model, device, mode=translate_mode)
        write_caption_rows(csv_path, chunk, append=append_mode)
        append_mode          = True
        processed_translate += len(chunk)

        tracker2.update(processed_translate)

        if processed_translate % CHECKPOINT_EVERY == 0:
            state_path.write_text(
                json.dumps({
                    "phase": "translate",
                    "caption_done": len(en_rows),
                    "translate_done": processed_translate,
                }, ensure_ascii=False),
                encoding="utf-8",
            )
            log(f"  ✔ Checkpoint saved  [{processed_translate}/{len(en_rows)}]")

    state_path.write_text(
        json.dumps({
            "phase": "done",
            "caption_done": len(en_rows),
            "translate_done": processed_translate,
        }, ensure_ascii=False),
        encoding="utf-8",
    )

    log_sep("ALL DONE")
    log(f"  Final CSV  → {csv_path}")
    log(f"  EN-only CSV→ {en_csv_path}")
    log(f"  Images     → {output_images_dir}")
    log(f"  Total time : {(time.time()-_run_start)/3600:.2f} h")


if __name__ == "__main__":
    main()