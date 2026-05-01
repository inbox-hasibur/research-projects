# ---------------------------------------------------------
# BanglaVision60K Dataset Advanced EDA for Kaggle
# ---------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import urllib.request
import matplotlib.font_manager as fm
from collections import Counter
from PIL import Image
import warnings
from wordcloud import WordCloud

warnings.filterwarnings('ignore')
sns.set(style="whitegrid")

# Create output directory for figures
OUTPUT_DIR = '/kaggle/working/figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------
# Bengali Font Setup (Crucial for visualization)
# ---------------------------------------------------------
import requests

font_urls = [
    # 3. Nikosh: Official documents, academic, clean design
    "https://raw.githubusercontent.com/maateen/bengali-fonts/master/fonts/Nikosh/Nikosh.ttf",
    # 4. Bangla Academy: Educational books and materials (Standard Fallback)
    "https://raw.githubusercontent.com/MinhasKamal/BengaliDictionary/master/BengaliFont/Kalpurush.ttf"
    # 1. Noto Sans Bengali: Highly legible, Google global collection
    "https://raw.githubusercontent.com/google/fonts/main/ofl/notosansbengali/static/NotoSansBengali-Regular.ttf",
    # 2. Tiro Bangla: Classic serif for academic/literary publishing
    "https://raw.githubusercontent.com/google/fonts/main/ofl/tirobangla/TiroBangla-Regular.ttf",
]

font_path = "/kaggle/working/BengaliFont.ttf"
font_downloaded = False

if not os.path.exists(font_path) or os.path.getsize(font_path) < 10000:
    print("Downloading Bengali font for visualizations...")
    for url in font_urls:
        try:
            print(f"Trying to fetch: {url.split('/')[-1]}...")
            response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
            response.raise_for_status()
            with open(font_path, 'wb') as f:
                f.write(response.content)
            
            # Verify file size to ensure it's not a corrupted HTML page
            if os.path.getsize(font_path) > 10000:
                print(f"✅ Font {url.split('/')[-1]} downloaded successfully!")
                font_downloaded = True
                break
            else:
                print("Downloaded file is too small. Trying next...")
        except requests.exceptions.ConnectionError:
            print("\n[!] CRITICAL ERROR: NO INTERNET CONNECTION DETECTED.")
            print("    Kaggle notebooks have Internet OFF by default.")
            print("    Please go to the right sidebar -> 'Settings' -> Turn 'Internet' ON.")
            print("    Without Internet, the Bengali font cannot be downloaded.\n")
            break # No point trying others if no internet
        except Exception as e:
            print(f"Failed: {e}")

if os.path.exists(font_path) and os.path.getsize(font_path) > 10000:
    fm.fontManager.addfont(font_path)
    prop = fm.FontProperties(fname=font_path)
    plt.rcParams['font.family'] = prop.get_name()
else:
    prop = None

# 1. Load Dataset
BASE_PATH = '/kaggle/input/banglavision60k/' 
if not os.path.exists(BASE_PATH):
    BASE_PATH = '/kaggle/input/datasets/inboxhasibur/banglavision60k/'

csv_path = os.path.join(BASE_PATH, 'Annotation.csv')
if not os.path.exists(csv_path):
    for root, dirs, files in os.walk('/kaggle/input'):
        if 'Annotation.csv' in files:
            csv_path = os.path.join(root, 'Annotation.csv')
            BASE_PATH = root
            break

print(f"\nLoading dataset from: {csv_path}")
df = pd.read_csv(csv_path)

# Helper function to find images robustly
def get_actual_img_path(base_path, filename):
    for folder in ['Images1 BN', 'Images2 FL', 'Images3 CC']:
        path = os.path.join(base_path, folder, str(filename))
        if os.path.exists(path):
            return path
    return None

# ---------------------------------------------------------
# 2. Basic Inspection
# ---------------------------------------------------------
print("\n" + "="*50)
print("► Dataset Vital Signs")
print("="*50)
print(f"Total Captions (Rows): {df.shape[0]}")
print(f"Unique Images: {df['original_filename'].nunique()}")
print(f"Features: {df.columns.tolist()}")

if 'split' in df.columns:
    print(f"\nSplit Distribution:\n{df.drop_duplicates(subset=['original_filename'])['split'].value_counts().to_string()}")


# ---------------------------------------------------------
# 3. Source Distribution
# ---------------------------------------------------------
print("\n" + "="*50)
print("► Source Distribution")
print("="*50)
unique_images_df = df.drop_duplicates(subset=['original_filename', 'source_folder'])

fig, ax = plt.subplots(figsize=(10, 6))
sns.countplot(data=unique_images_df, x='source_folder', palette='magma', ax=ax)
ax.set_title('Unique Image Distribution across Sources', fontsize=14)
ax.set_ylabel('Number of Images')
plt.savefig(os.path.join(OUTPUT_DIR, 'source_distribution.png'), dpi=300, bbox_inches='tight')
plt.show()

# ---------------------------------------------------------
# 3.5. Data Split Ratio (Train/Val/Test)
# ---------------------------------------------------------
if 'split' in unique_images_df.columns:
    print("\n" + "="*50)
    print("► Data Split Distribution")
    print("="*50)
    
    plt.figure(figsize=(8, 8))
    split_counts = unique_images_df['split'].value_counts()
    colors = sns.color_palette("pastel")[0:len(split_counts)]
    
    # Donut chart
    plt.pie(split_counts, labels=split_counts.index, autopct='%1.1f%%', 
            startangle=140, colors=colors, wedgeprops={'edgecolor': 'white', 'linewidth': 2})
    
    centre_circle = plt.Circle((0,0),0.70,fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)
    
    plt.title('Dataset Split Distribution (Train / Val / Test)', fontsize=16, fontweight='bold')
    plt.savefig(os.path.join(OUTPUT_DIR, 'split_distribution.png'), dpi=300, bbox_inches='tight')
    plt.show()

# ---------------------------------------------------------
# 4. Advanced Linguistic Autopsy
# ---------------------------------------------------------
print("\n" + "="*50)
print("► Advanced Linguistic Autopsy")
print("="*50)
df_clean = df.dropna(subset=['caption']).copy()

df_clean['word_count'] = df_clean['caption'].apply(lambda x: len(str(x).split()))
df_clean['char_count'] = df_clean['caption'].apply(lambda x: len(str(x)))

print(f"Total Valid Captions: {len(df_clean)}")
print(f"Average Words per Caption: {df_clean['word_count'].mean():.2f}")
print(f"Max Words: {df_clean['word_count'].max()} | Min Words: {df_clean['word_count'].min()}")

text = " ".join(caption for caption in df_clean.caption.astype(str))
words = text.split()
unique_words = set(words)
print(f"Total Vocabulary Size (Unique Tokens): {len(unique_words)}")

fig, ax = plt.subplots(1, 2, figsize=(16, 5))
sns.histplot(df_clean['word_count'], bins=30, kde=True, color='teal', ax=ax[0])
ax[0].set_title('Word Count Distribution')
ax[0].set_xlabel('Number of Words')

sns.histplot(df_clean['char_count'], bins=30, kde=True, color='coral', ax=ax[1])
ax[1].set_title('Character Count Distribution')
ax[1].set_xlabel('Number of Characters')

plt.savefig(os.path.join(OUTPUT_DIR, 'linguistic_autopsy.png'), dpi=300, bbox_inches='tight')
plt.show()

# New: Caption Length by Source
plt.figure(figsize=(10, 6))
sns.boxplot(data=df_clean, x='source_folder', y='word_count', palette='Set2')
plt.title('Caption Length (Word Count) by Source', fontproperties=prop, fontsize=16)
plt.savefig(os.path.join(OUTPUT_DIR, 'caption_length_source.png'), dpi=300, bbox_inches='tight')
plt.show()

# New: WordCloud
try:
    # Check if the font was downloaded successfully and has a reasonable size (> 10KB)
    if os.path.exists(font_path) and os.path.getsize(font_path) > 10000:
        wc = WordCloud(font_path=font_path, width=800, height=400, background_color='white').generate(text)
    else:
        print("Warning: Valid Bengali font not found. Generating WordCloud with default font (Bengali text may show as boxes). Please turn ON 'Internet' in Kaggle settings.")
        wc = WordCloud(width=800, height=400, background_color='white').generate(text)
        
    plt.figure(figsize=(12, 6))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    if prop:
        plt.title('Most Frequent Words WordCloud', fontproperties=prop, fontsize=16)
    else:
        plt.title('Most Frequent Words WordCloud', fontsize=16)
    plt.savefig(os.path.join(OUTPUT_DIR, 'wordcloud.png'), dpi=300, bbox_inches='tight')
    plt.show()
except Exception as e:
    print(f"Failed to generate WordCloud: {e}")

# NEW VISUALIZATION 3: Cumulative Vocabulary Coverage
print("\n► Generating Cumulative Vocabulary Coverage Plot...")
word_counts = Counter(words)
total_words = sum(word_counts.values())
sorted_counts = sorted(word_counts.values(), reverse=True)
cumulative_sum = np.cumsum(sorted_counts)
cumulative_percent = (cumulative_sum / total_words) * 100

milestones = [50, 80, 90, 95, 99]
milestone_indices = []
for m in milestones:
    idx = np.argmax(cumulative_percent >= m)
    milestone_indices.append((idx + 1, cumulative_percent[idx]))

plt.figure(figsize=(12, 6))
plt.plot(range(1, len(cumulative_percent) + 1), cumulative_percent, linewidth=2.5, color='#1f77b4')
plt.fill_between(range(1, len(cumulative_percent) + 1), cumulative_percent, color='#1f77b4', alpha=0.1)

for m_val, m_pct in milestone_indices:
    plt.plot(m_val, m_pct, marker='o', markersize=6, color='#d62728')
    plt.annotate(f"{m_val:,} words\n→ {int(m_pct)}%", 
                 (m_val, m_pct),
                 xytext=(15, -20), textcoords='offset points',
                 fontsize=9, color='darkred',
                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2", color='gray'))

plt.xscale('log')
plt.ylim(0, 105)
plt.xlabel('Vocabulary Size (log scale)', fontsize=12)
plt.ylabel('Corpus Coverage (%)', fontsize=12)

if prop:
    plt.title('Vocabulary Coverage Curve (Cumulative)', fontproperties=prop, fontsize=16)
else:
    plt.title('Vocabulary Coverage Curve (Cumulative)', fontsize=16)

plt.grid(True, which="both", ls="-", alpha=0.2)
plt.savefig(os.path.join(OUTPUT_DIR, 'vocab_coverage.png'), dpi=300, bbox_inches='tight')
plt.show()

# ---------------------------------------------------------
# 5. Token & N-Gram Analysis
# ---------------------------------------------------------
print("\n" + "="*50)
print("► Token & Bigram Analysis")
print("="*50)
top_words = Counter(words).most_common(15)
print("Top 10 Words:")
for word, freq in top_words[:10]:
    print(f"  {word}: {freq}")

# Bigrams
bigrams = zip(words[:-1], words[1:])
top_bigrams = Counter(bigrams).most_common(5)
print("\nTop 5 Bigrams:")
for bg, freq in top_bigrams:
    print(f"  {bg[0]} {bg[1]}: {freq}")

# Visualization of Top Tokens
words_list, freqs_list = zip(*top_words)
plt.figure(figsize=(10, 6))
# Using proper font properties if available
ax = sns.barplot(x=list(freqs_list), y=list(words_list), palette='viridis')
if prop:
    plt.yticks(fontproperties=prop, fontsize=12)
    plt.title('Top 15 Most Frequent Bengali Tokens', fontproperties=prop, fontsize=16)
else:
    plt.title('Top 15 Most Frequent Bengali Tokens')
plt.xlabel('Frequency')
plt.savefig(os.path.join(OUTPUT_DIR, 'top_tokens.png'), dpi=300, bbox_inches='tight')
plt.show()

# ---------------------------------------------------------
# 6. Image Forensics (Resolution & Aspect Ratio)
# ---------------------------------------------------------
print("\n" + "="*50)
print("► Image Forensics (Resolution)")
print("="*50)

def get_image_resolution(row):
    path = get_actual_img_path(BASE_PATH, row['original_filename'])
    if path:
        try:
            with Image.open(path) as img:
                return img.size 
        except:
            return (None, None)
    return (None, None)

print("Analyzing Image Resolutions (Sampling 1000 unique images for speed)...")
sample_df = unique_images_df.sample(min(1000, len(unique_images_df)))
sample_img_info = sample_df.apply(get_image_resolution, axis=1)
widths = [i[0] for i in sample_img_info if i[0] is not None]
heights = [i[1] for i in sample_img_info if i[1] is not None]

if widths and heights:
    plt.figure(figsize=(8, 6))
    plt.scatter(widths, heights, alpha=0.5, c='purple')
    plt.title('Image Resolution Distribution (Width vs Height)')
    plt.xlabel('Width (pixels)')
    plt.ylabel('Height (pixels)')
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, 'image_forensics.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    avg_w = np.mean(widths)
    avg_h = np.mean(heights)
    print(f"Average Resolution: {avg_w:.0f}x{avg_h:.0f}")
else:
    print("Could not load image resolutions. Check image paths.")

# ---------------------------------------------------------
# 7. Final Gallery (Visual Proof with Valid Images)
# ---------------------------------------------------------
print("\n" + "="*50)
print("► Final Gallery: 3 Examples per Source (with 2 Captions each)")
print("="*50)

sources = df['source_folder'].unique()

for source in sources:
    if pd.isna(source): continue
    
    source_df = df[df['source_folder'] == source]
    unique_imgs = source_df['original_filename'].unique()
    
    # Filter to ensure we only use images that actually exist on disk
    valid_imgs = []
    for img_name in unique_imgs:
        if get_actual_img_path(BASE_PATH, img_name) is not None:
            valid_imgs.append(img_name)
        if len(valid_imgs) >= 3: # We only need 3 valid ones per source
            break
            
    if len(valid_imgs) == 0:
        continue
    
    # We will plot exactly the valid ones (up to 3)
    sample_imgs = valid_imgs[:3]
    
    plt.figure(figsize=(20, 6))
    if prop:
        plt.suptitle(f"Source: {source}", fontsize=18, fontweight='bold', fontproperties=prop)
    else:
        plt.suptitle(f"Source: {source}", fontsize=18, fontweight='bold')
    
    for i, img_name in enumerate(sample_imgs):
        img_captions = source_df[source_df['original_filename'] == img_name]['caption'].tolist()[:2]
        img_path = get_actual_img_path(BASE_PATH, img_name)
        
        plt.subplot(1, len(sample_imgs), i+1)
        img = Image.open(img_path)
        plt.imshow(img)
        
        title = ""
        for j, cap in enumerate(img_captions):
            wrapped_cap = cap[:45] + '...' if len(cap) > 45 else cap
            title += f"C{j+1}: {wrapped_cap}\n"
            
        if prop:
            plt.title(title.strip(), fontproperties=prop, fontsize=14, pad=10)
        else:
            plt.title(title.strip(), fontsize=11, pad=10)
            
        plt.axis('off')
        
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'gallery_{source}.png'), dpi=300, bbox_inches='tight')
    plt.show()
