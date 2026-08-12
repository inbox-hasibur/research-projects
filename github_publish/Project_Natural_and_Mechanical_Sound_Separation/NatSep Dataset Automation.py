import sys
import os
import glob
import re
import random
import subprocess
import warnings
from pathlib import Path


# Suppress Python 3.12 SyntaxWarnings from pydub regex strings
warnings.filterwarnings("ignore", category=SyntaxWarning)

from pydub import AudioSegment
from pydub.effects import normalize

# ==========================================
# DATASET PATH CONFIGURATION
# ==========================================
DATASET_PATH = Path("/content/drive/MyDrive/Education/Research/Conf. Kahf NatSep - Natural Sound Preserving Music Remover/NetSep Audio Dataset")

INSTRUMENTAL_DIR = DATASET_PATH / "Mechanical Sounds" / "instrumentals"
NATURAL_DIR = DATASET_PATH / "Natural Sounds" / "naturals"
VOCAL_DIR = DATASET_PATH / "Natural Sounds" / "vocals"

MIX_NATURAL_INSTR = DATASET_PATH / "Mix Sounds" / "natural+instrumental"
MIX_VOCAL_INSTR = DATASET_PATH / "Mix Sounds" / "vocal+instrumental"
MIX_VOCAL_NATURAL_INSTR = DATASET_PATH / "Mix Sounds" / "vocal+natural+instrumental"

RAW_DIR = DATASET_PATH / "RAW File" / "Complete"

# Ensure output directories exist
for p in [INSTRUMENTAL_DIR, NATURAL_DIR, VOCAL_DIR, MIX_NATURAL_INSTR, MIX_VOCAL_INSTR, MIX_VOCAL_NATURAL_INSTR, RAW_DIR]:
    p.mkdir(parents=True, exist_ok=True)


# ==========================================
# TITLE CLEANING & NAMING UTILITIES
# ==========================================
def clean_title(title: str) -> str:
    """
    Cleans title: removes special characters like ', :, ", ?, !, /, \, |, *, <, >, (, ), [, ], etc.
    Converts to lowercase and cleans extra spaces.
    """
    # Remove special characters
    title = re.sub(r"['\":?!/\\|*<>()[\]{}]", "", title)
    # Clean multiple spaces and strip
    title = re.sub(r"\s+", " ", title).strip().lower()
    return title

def extract_song_name(filename: str) -> str:
    """
    Extracts song name from file formatted like '1 vocal - neffex cold.wav'
    Returns 'neffex cold'
    """
    stem = Path(filename).stem
    if " - " in stem:
        parts = stem.split(" - ", 1)
        return clean_title(parts[1])
    return clean_title(stem)


# ==========================================
# 1. YOUTUBE DOWNLOAD & CONVERT AUTOMATION
# ==========================================
def ensure_yt_dlp():
    try:
        import yt_dlp
        return yt_dlp
    except ImportError:
        print("📦 Installing yt-dlp package...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "yt-dlp"], check=True)
        import yt_dlp
        return yt_dlp

def download_youtube_audio(url_list, target_folder, category_type="vocal"):
    """
    Downloads YouTube audio using native yt_dlp Python API, cleans title, renames to pattern:
    '{index} {category_type} - {clean_title}.wav'
    """
    yt_dlp = ensure_yt_dlp()
    target_folder = Path(target_folder)
    target_folder.mkdir(parents=True, exist_ok=True)
    
    if isinstance(url_list, str):
        url_list = [url_list]
        
    existing_files = [p for p in target_folder.iterdir() if p.suffix.lower() in ('.wav', '.mp3', '.m4a', '.flac')]
    start_index = len(existing_files) + 1
    
    print(f"\n📥 Starting Download to: {target_folder.name} (Category: {category_type})")
    for idx, url in enumerate(url_list):
        current_index = start_index + idx
        output_template = str(target_folder / f"temp_%(title)s.%(ext)s")
        
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '0',
            }],
            'outtmpl': output_template,
            'quiet': False,
        }
        
        try:
            print(f"Downloading [{idx+1}/{len(url_list)}]: {url}")
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            
            # Find the temp file downloaded and rename according to pattern
            temp_files = list(target_folder.glob("temp_*"))
            for temp_f in temp_files:
                raw_title = temp_f.stem.replace("temp_", "")
                cleaned = clean_title(raw_title)
                new_filename = f"{current_index} {category_type} - {cleaned}.wav"
                new_path = target_folder / new_filename
                temp_f.rename(new_path)
                print(f"  ✅ Saved as: {new_filename}")
        except Exception as e:
            print(f"  ❌ Error downloading {url}: {e}")


def process_form_download(category, url_input, auto_slice=True, chunk_sec=120):
    urls = [u.strip() for u in url_input.replace("\n", ",").split(",") if u.strip()]
    if not urls:
        print("⚠️ No URLs provided!")
        return

    if category == "Vocals":
        download_youtube_audio(urls, VOCAL_DIR, category_type="vocal")
    elif category == "Instrumentals":
        download_youtube_audio(urls, INSTRUMENTAL_DIR, category_type="instrumental")
    elif category == "Long Natural Sound":
        download_youtube_audio(urls, RAW_DIR, category_type="natural")
        if auto_slice:
            for raw_file in load_audio_files(RAW_DIR):
                slice_long_audio(raw_file, NATURAL_DIR, chunk_sec=chunk_sec)
    else:
        print(f"❌ Unknown Category: {category}")


# ==========================================
# 2. AUDIO SLICER (For 7-8 Hour Natural Audio)
# ==========================================
def slice_long_audio(input_file, target_folder, chunk_sec=120):
    input_file = Path(input_file)
    target_folder = Path(target_folder)
    target_folder.mkdir(parents=True, exist_ok=True)
    
    song_name = extract_song_name(input_file.name)
    existing_files = load_audio_files(target_folder)
    start_index = len(existing_files) + 1
    
    print(f"\n✂️ Slicing long audio: {input_file.name} ...")
    audio = AudioSegment.from_file(input_file)
    chunk_ms = chunk_sec * 1000
    total_ms = len(audio)
    num_chunks = total_ms // chunk_ms
    
    for i in range(num_chunks):
        curr_idx = start_index + i
        chunk = audio[i * chunk_ms : (i + 1) * chunk_ms]
        chunk_name = f"{curr_idx} natural - {song_name}.wav"
        chunk_path = target_folder / chunk_name
        chunk.export(chunk_path, format="wav")
        if (i + 1) % 20 == 0 or i == num_chunks - 1:
            print(f"  Exported [{i+1}/{num_chunks}] -> {chunk_name}")


# ==========================================
# 3. DATASET MIXING AUTOMATION
# ==========================================
def load_audio_files(directory):
    valid_exts = ('.wav', '.mp3', '.m4a', '.flac', '.aac', '.ogg')
    return sorted([p for p in Path(directory).iterdir() if p.suffix.lower() in valid_exts], key=lambda x: x.name)

def match_duration(audio, target_ms):
    if len(audio) > target_ms:
        return audio[:target_ms]
    elif len(audio) < target_ms:
        repeats = (target_ms // len(audio)) + 1
        looped = audio * repeats
        return looped[:target_ms]
    return audio

def mix_and_generate_dataset(target_duration_sec=120, max_samples=None, random_combination=False):
    vocals = load_audio_files(VOCAL_DIR)
    naturals = load_audio_files(NATURAL_DIR)
    instrumentals = load_audio_files(INSTRUMENTAL_DIR)
    
    print(f"\n📊 Found Audio Stems:")
    print(f"  - Vocals: {len(vocals)} files")
    print(f"  - Naturals: {len(naturals)} files")
    print(f"  - Instrumentals: {len(instrumentals)} files")
    
    if not vocals or not naturals or not instrumentals:
        print("❌ Error: One or more source directories are empty. Please add audio files first.")
        return
        
    target_ms = target_duration_sec * 1000
    
    if random_combination:
        random.shuffle(vocals)
        random.shuffle(naturals)
        random.shuffle(instrumentals)
        
    num_mixes = min(len(vocals), len(naturals), len(instrumentals))
    if max_samples:
        num_mixes = min(num_mixes, max_samples)
        
    print(f"\n🎛️ Generating {num_mixes} Dataset Mix Combinations ({target_duration_sec}s each)...")
    
    for i in range(num_mixes):
        v_path = vocals[i]
        n_path = naturals[i % len(naturals)]
        i_path = instrumentals[i % len(instrumentals)]
        
        idx_num = i + 1
        song_name = extract_song_name(v_path.name)
        print(f"[{idx_num}/{num_mixes}] Mixing: '{song_name}'")
        
        # Load audio segments
        v_seg = match_duration(AudioSegment.from_file(v_path), target_ms) - 2.0
        n_seg = match_duration(AudioSegment.from_file(n_path), target_ms) - 3.0
        i_seg = match_duration(AudioSegment.from_file(i_path), target_ms) - 2.0
        
        # 1. Natural + Instrumental
        name_ni = f"{idx_num} natural+instrumental - {song_name}.wav"
        normalize(n_seg.overlay(i_seg)).export(MIX_NATURAL_INSTR / name_ni, format="wav")
        
        # 2. Vocal + Instrumental
        name_vi = f"{idx_num} vocal+instrumental - {song_name}.wav"
        normalize(v_seg.overlay(i_seg)).export(MIX_VOCAL_INSTR / name_vi, format="wav")
        
        # 3. Vocal + Natural + Instrumental
        name_vni = f"{idx_num} vocal+natural+instrumental - {song_name}.wav"
        normalize(v_seg.overlay(n_seg).overlay(i_seg)).export(MIX_VOCAL_NATURAL_INSTR / name_vni, format="wav")
        
    print("\n🎉 Dataset Generation Complete!")


def launch_interactive_ui():
    try:
        import ipywidgets as widgets
        from IPython.display import display
    except ImportError:
        print("ipywidgets not installed. Run: pip install ipywidgets")
        return

    print("==================================================")
    print("  🎵 NatSep Audio Dataset Automation Dashboard 🎵")
    print("==================================================")

    category_w = widgets.Dropdown(
        options=['Vocals', 'Instrumentals', 'Long Natural Sound'],
        value='Vocals',
        description='Category:',
        style={'description_width': 'initial'}
    )
    
    url_w = widgets.Textarea(
        value='',
        placeholder='Paste YouTube URLs here...',
        description='YouTube URLs:',
        layout=widgets.Layout(width='90%', height='100px'),
        style={'description_width': 'initial'}
    )
    
    slice_w = widgets.Checkbox(
        value=True,
        description='Auto-slice Long Natural Sound into 2-min (120s) chunks'
    )
    
    download_btn = widgets.Button(
        description="⬇️ Start Download & Process",
        button_style='success',
        layout=widgets.Layout(width='60%', height='40px')
    )
    
    dl_output = widgets.Output()

    def on_download_click(b):
        with dl_output:
            dl_output.clear_output()
            process_form_download(
                category=category_w.value,
                url_input=url_w.value,
                auto_slice=slice_w.value,
                chunk_sec=120
            )

    download_btn.on_click(on_download_click)

    mix_btn = widgets.Button(
        description="🎛️ Generate Dataset Mixes",
        button_style='primary',
        layout=widgets.Layout(width='60%', height='40px')
    )
    
    mix_output = widgets.Output()

    def on_mix_click(b):
        with mix_output:
            mix_output.clear_output()
            mix_and_generate_dataset(target_duration_sec=120)

    mix_btn.on_click(on_mix_click)

    display(widgets.VBox([
        widgets.HTML("<h3>1. YouTube Audio Downloader</h3>"),
        category_w, url_w, slice_w, download_btn, dl_output,
        widgets.HTML("<hr><h3>2. Dataset Audio Mixer</h3>"),
        mix_btn, mix_output
    ]))


if __name__ == "__main__":
    print("NatSep Dataset Automation Pipeline Loaded.")
    launch_interactive_ui()

