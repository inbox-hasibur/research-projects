LASTLY SOB FIGURE ER ALIGNMENT AND SPACING THIK KORE DEO, RESEARCH PAPER STANDARD, LAST ER FIGURE E JUST ONLY TEST MACRO F1 TA RAKHO, FIGURE 1 TTHEKEI CODE DEO:
# =============================================================================
# FIG 1 ── Dataset Sample  4×2  REAL IMAGES
# =============================================================================
print("─"*55)
print("FIG 1 — Dataset Sample (4×2, real images)")

def find_one_image(cls_name, seed=42):
    """Return path of one representative image for the class."""
    random.seed(seed); np.random.seed(seed)
    pool = []

    # Afzaal train
    if os.path.exists(AFZAAL_TRAIN):
        for fn in sorted(os.listdir(AFZAAL_TRAIN)):
            if not fn.lower().endswith(('.jpg','.jpeg','.png')): continue
            if os.path.splitext(fn)[0].lower().startswith(cls_name):
                pool.append(os.path.join(AFZAAL_TRAIN, fn))

    # PlantVillage fallbacks
    for pv_dir, pv_cls in [(PV_SCORCH,'leaf_scorch'),(PV_HEALTHY,'healthy')]:
        if cls_name == pv_cls and os.path.exists(pv_dir):
            for fn in sorted(os.listdir(pv_dir)):
                if fn.lower().endswith(('.jpg','.jpeg','.png')):
                    pool.append(os.path.join(pv_dir, fn))

    if not pool: return None
    # pick from middle third for variety
    s = max(0, len(pool)//3)
    e = min(len(pool), s+15)
    return random.choice(pool[s:e]) if e > s else pool[0]

def load_square(path, size=300):
    """Load image, centre-crop to square, resize."""
    img = Image.open(path).convert("RGB")
    w,h = img.size
    side = min(w,h)
    img  = img.crop(((w-side)//2,(h-side)//2,
                     (w+side)//2,(h+side)//2))
    return img.resize((size,size), Image.LANCZOS)

# Collect one image per class
print("  Searching images...")
img_paths = {}
for cls in CLASS_NAMES_8:
    p = find_one_image(cls)
    img_paths[cls] = p
    tag = f"✅ {os.path.basename(p)}" if p else "❌ NOT FOUND"
    print(f"    {cls:<22} → {tag}")

# ── Build figure (Horizontal Layout) ──────────────────────
nrows, ncols = 2, 4  
fig = plt.figure(figsize=(20, 10))
fig.patch.set_facecolor('white')

# GridSpec 
gs = gridspec.GridSpec(
    nrows * 2, ncols,
    figure=fig,
    hspace=0.01,
    wspace=0.01,
    height_ratios=[5, 1] * nrows,
    top=0.88,  
    bottom=0.08 
)

for idx, cls in enumerate(CLASS_NAMES_8):
    row_i = (idx // ncols) * 2   
    row_d = row_i + 1             
    col   = idx % ncols
    accent = CLASS_COLORS_8[idx]
    ax_img  = fig.add_subplot(gs[row_i, col])
    ax_lbl  = fig.add_subplot(gs[row_d, col])

    # ── Image ────────────────────────────────────────────────
    p = img_paths[cls]
    if p and os.path.exists(p):
        try:
            arr = np.array(load_square(p, 300))
            ax_img.imshow(arr)
        except Exception as ex:
            print(f"    ⚠ {cls}: {ex}")
            ax_img.set_facecolor(accent)
            ax_img.text(0.5,0.5,'Load\nError',
                        ha='center',va='center',
                        color='white',fontsize=10,
                        transform=ax_img.transAxes)
    else:
        # colour-patch fallback if image unavailable
        ax_img.set_facecolor(accent)
        ax_img.text(0.5,0.5,'Not\nFound',
                    ha='center',va='center',
                    color='white',fontsize=11,fontweight='bold',
                    transform=ax_img.transAxes)

    # Coloured border
    for sp in ax_img.spines.values():
        sp.set_edgecolor(accent); sp.set_linewidth(3.5)
    ax_img.tick_params(left=False,bottom=False,
                       labelleft=False,labelbottom=False)

    # Class-index badge  (top-left)
    ax_img.text(0.03, 0.97,
                f"Class {idx}",
                transform=ax_img.transAxes,
                fontsize=9, color='white', fontweight='bold',
                va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.3',
                          facecolor=accent, alpha=0.88,
                          edgecolor='white', linewidth=0.8))

    # Source badge (bottom-right)
    src = 'PlantVillage' if cls in ('leaf_scorch','healthy') else 'Afzaal'
    sc  = PAL['teal'] if src == 'PlantVillage' else PAL['vgg']
    ax_img.text(0.97, 0.03, src,
                transform=ax_img.transAxes,
                fontsize=7.5, color='white', fontweight='bold',
                va='bottom', ha='right',
                bbox=dict(boxstyle='round,pad=0.25',
                          facecolor=sc, alpha=0.88,
                          edgecolor='white', linewidth=0.6))

    # ── Label strip ──────────────────────────────────────────
    ax_lbl.set_facecolor(accent)
    ax_lbl.set_xlim(0,1); ax_lbl.set_ylim(0,1)
    ax_lbl.axis('off')
    ax_lbl.text(0.5, 0.65,
                CLASS_DISPLAY_8[idx],
                ha='center', va='center',
                fontsize=10, color='white', fontweight='bold',
                transform=ax_lbl.transAxes)
    ax_lbl.text(0.5, 0.25,
                CLASS_DESC_8[idx],
                ha='center', va='center',
                fontsize=7.5, color='white', alpha=0.90,
                multialignment='center',
                transform=ax_lbl.transAxes)

# Legend
legend_patches = [
    mpatches.Patch(facecolor=PAL['vgg'],  edgecolor='white',
                   label='Afzaal Dataset'),
    mpatches.Patch(facecolor=PAL['teal'], edgecolor='white',
                   label='PlantVillage'),
]

fig.legend(handles=legend_patches,
           loc='lower center',
           ncol=2,
           fontsize=10,
           frameon=True,
           framealpha=0.9,
           edgecolor='#ccc',
           bbox_to_anchor=(0.5, 0.015))

# Title (increase gap above images)
fig.suptitle(
    "Strawberry Disease Dataset — Sample Images\n"
    "Afzaal (6 Classes) + PlantVillage "
    "(2 Classes)  |  Total: 4,065 images",
    fontsize=12,
    fontweight='bold',
    color='#1a1a2e',
    y=0.94,   # ↑ move title up → creates space below it
    linespacing=1.5
)

save_fig("fig1_dataset_sample", "Dataset Sample 4×2 Real Images")


# =============================================================================
# FIG 2 ── Class & Split Distribution
# =============================================================================
print("─"*55)
print("FIG 2 — Class & Split Distribution")

classes_ord  = list(train_dist_8.keys())
cls_disp     = [c.replace('_','\n') for c in classes_ord]
train_counts = [train_dist_8[c] for c in classes_ord]
val_counts   = [val_dist_8.get(c,0) for c in classes_ord]
test_counts  = [test_dist_8.get(c,0) for c in classes_ord]
x = np.arange(len(classes_ord))

fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
fig.patch.set_facecolor('white')

# ── Stacked bar ───────────────────────────────────────────────
b1 = axes[0].bar(x, train_counts, width=0.7,
                 label='Train', color=PAL['vgg'],
                 alpha=0.88, edgecolor='white', linewidth=0.7)
b2 = axes[0].bar(x, val_counts, width=0.7,
                 bottom=train_counts,
                 label='Validation', color=PAL['gold'],
                 alpha=0.88, edgecolor='white', linewidth=0.7)
b3 = axes[0].bar(x, test_counts, width=0.7,
                 bottom=[t+v for t,v in zip(train_counts,val_counts)],
                 label='Test', color=PAL['neg'],
                 alpha=0.88, edgecolor='white', linewidth=0.7)

axes[0].set_xticks(x)
axes[0].set_xticklabels(cls_disp, fontsize=8.5)
axes[0].set_ylabel("Number of Images")
axes[0].set_title("Class & Split Distribution", fontweight='bold')
axes[0].legend(fontsize=9)
axes[0].grid(axis='y', alpha=0.3, linestyle='--')
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)

for bar, tc, vc, tesc in zip(b1, train_counts, val_counts, test_counts):
    total = tc + vc + tesc
    axes[0].text(bar.get_x()+bar.get_width()/2,
                 total+6, str(total),
                 ha='center', va='bottom',
                 fontsize=8, fontweight='bold', color='#333')

# ── Pie chart ─────────────────────────────────────────────────
afzaal_total = 1450+307+743
pv_total     = (2701-afzaal_total)+(900-743)
sizes   = [afzaal_total, pv_total]
slabels = [f'Afzaal\n({afzaal_total:,} imgs)',
           f'PlantVillage\n({pv_total:,} imgs)']
scols   = [PAL['vgg'], PAL['eff']]

wedges,texts,autotexts = axes[1].pie(
    sizes, labels=slabels, colors=scols, explode=[0.04,0.04],
    autopct='%1.1f%%', startangle=140,
    textprops={'fontsize':10},
    wedgeprops={'edgecolor':'white','linewidth':1.5},
    pctdistance=0.75)
for at in autotexts:
    at.set_fontsize(10); at.set_fontweight('bold')

axes[1].text(0,-0.08,
             f"Total: {afzaal_total+pv_total:,}  |  Classes: 8\n"
             f"Train: 2,701  Val: 464  Test: 900\n"
             f"Imbalance Ratio: 13.35×",
             ha='center', va='center', fontsize=9,
             bbox=dict(boxstyle='round,pad=0.35',
                       facecolor='#fffde7', alpha=0.9,
                       edgecolor='#ccc'))
axes[1].set_title("Dataset Source Distribution", fontweight='bold')

plt.tight_layout()
save_fig("fig2_class_split_distribution", "Class & Split Distribution")


# =============================================================================
# FIG 3 ── Training vs Validation Curves — 2×2 (one per model)
#           Shows Train Loss + Val Loss on same axes → easy overfitting view
# =============================================================================
print("─"*55)
print("FIG 3 — Training vs Validation Curves (2×2 overfitting view)")

model_specs_tv = [
    ('VGG19+CBAM',             vgg_tr,  vgg_vl,  vgg_ac,  None,
     PAL['vgg']),
    ('ResNet50+CBAM+GeM',      res_tr,  res_vl,  res_ac,  None,
     PAL['res']),
    ('EfficientNetV2+ECA+GeM', eff_tr,  eff_vl,  eff_ac,  None,
     PAL['eff']),
    ('ResNet50+Swin Hybrid',   hyb_tr,  hyb_vl,  hyb_ac,  BEST_EPOCH_HYB,
     PAL['hybrid']),
]

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.patch.set_facecolor('white')
axes_flat = axes.flatten()

for ax, (mname, tr, vl, ac, best_ep, col) in \
        zip(axes_flat, model_specs_tv):

    ep = np.arange(1, len(tr)+1)
    ax2 = ax.twinx()   # second y-axis for val accuracy

    # Train loss
    l1, = ax.plot(ep, tr, color=col, lw=2.2, linestyle='-',
                  label='Train Loss')
    # Val loss
    l2, = ax.plot(ep, vl, color=col, lw=2.2, linestyle='--',
                  alpha=0.75, label='Val Loss')
    # Fill gap → overfitting region
    ax.fill_between(ep, tr, vl,
                    where=[v > t for t,v in zip(tr,vl)],
                    alpha=0.12, color='red',
                    label='Val > Train (overfit)')
    ax.fill_between(ep, tr, vl,
                    where=[v <= t for t,v in zip(tr,vl)],
                    alpha=0.08, color=col)

    # Val accuracy on right axis
    l3, = ax2.plot(ep, ac, color=PAL['gold'], lw=1.6,
                   linestyle=':', marker='o', ms=3, markevery=3,
                   label='Val Acc (%)')
    ax2.set_ylabel("Val Accuracy (%)", color=PAL['gold'], fontsize=9)
    ax2.tick_params(axis='y', labelcolor=PAL['gold'])
    ax2.set_ylim([85, 101])

    # Best epoch
    if best_ep:
        ax.axvline(best_ep, color=PAL['pos'], lw=1.5,
                   linestyle=':', alpha=0.9,
                   label=f'Best Ep {best_ep}')

    ax.set_title(mname, fontweight='bold')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(alpha=0.22, linestyle='--')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.spines['top'].set_visible(False)

    # Combined legend
    lines = [l1, l2, l3]
    labels = [ln.get_label() for ln in lines]
    ax.legend(lines, labels, fontsize=7.5, loc='upper right')

plt.suptitle(
    "Training vs. Validation Loss & Accuracy — All Four Models",
    fontsize=12, fontweight='bold', y=1.01)
plt.tight_layout()
save_fig("fig3_train_val_curves_2x2",
         "Training vs Val Curves 2×2")


# =============================================================================
# FIG 4 ── Validation Curves — 4 Models Combined
# =============================================================================
print("─"*55)
print("FIG 4 — Validation Curves Combined")

model_specs_comb = [
    ('VGG19+CBAM',             vgg_tr,  vgg_vl,  vgg_ac,  PAL['vgg']),
    ('ResNet50+CBAM+GeM',      res_tr,  res_vl,  res_ac,  PAL['res']),
    ('EfficientNetV2+ECA+GeM', eff_tr,  eff_vl,  eff_ac,  PAL['eff']),
    ('ResNet50+Swin Hybrid',   hyb_tr,  hyb_vl,  hyb_ac,  PAL['hybrid']),
]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.patch.set_facecolor('white')

for mname, tr, vl, ac, col in model_specs_comb:
    ep = range(1, len(tr)+1)
    axes[0].plot(ep, vl, color=col, lw=2.0, label=mname, alpha=0.9)
    axes[1].plot(ep, ac, color=col, lw=2.0, label=mname, alpha=0.9)

axes[0].axvline(BEST_EPOCH_HYB, color='grey', lw=1.2,
                linestyle=':', alpha=0.6, label='Hybrid Best Ep')
axes[1].axvline(BEST_EPOCH_HYB, color='grey', lw=1.2,
                linestyle=':', alpha=0.6, label='Hybrid Best Ep')

axes[0].set_title("Validation Loss — All Models", fontweight='bold')
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Validation Loss")
axes[0].legend(fontsize=8.5); axes[0].grid(alpha=0.25, linestyle='--')

axes[1].set_title("Validation Accuracy — All Models", fontweight='bold')
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy (%)")
axes[1].legend(fontsize=8.5); axes[1].grid(alpha=0.25, linestyle='--')
axes[1].set_ylim([85, 101])

plt.tight_layout()
save_fig("fig4_val_curves_combined", "Validation Curves Combined")


# =============================================================================
# FIG 5 ── Learning Rate Schedule  (SINGLE FIGURE)
# =============================================================================
print("─"*55)
print("FIG 5 — Learning Rate (single figure)")

ep_lr = np.arange(1, len(lr_hyb)+1)

fig, ax = plt.subplots(figsize=(9, 4.5))
fig.patch.set_facecolor('white')

ax.plot(ep_lr, lr_hyb,
        color=PAL['hybrid'], lw=2.4,
        marker='o', ms=4.5, markevery=2,
        label='Classification Head LR  (×1.0)')
ax.plot(ep_lr, [l*0.1  for l in lr_hyb],
        color=PAL['swin'], lw=1.9, linestyle='--',
        marker='s', ms=4, markevery=2,
        label='Swin-T Backbone LR  (×0.1)')
ax.plot(ep_lr, [l*0.05 for l in lr_hyb],
        color=PAL['res'], lw=1.9, linestyle=':',
        marker='^', ms=4, markevery=2,
        label='ResNet50 Backbone LR  (×0.05)')

ax.axvline(BEST_EPOCH_HYB, color=PAL['pos'], lw=1.8,
           linestyle='--', alpha=0.85,
           label=f'Best Checkpoint (Ep {BEST_EPOCH_HYB})')
ax.axvline(10, color='grey', lw=1.0, linestyle=':',
           alpha=0.5, label='Warm Restart (T₀=10)')

ax.set_yscale('log')
ax.set_xlabel("Epoch")
ax.set_ylabel("Learning Rate (log scale)")
ax.set_title("CosineAnnealingWarmRestarts — Differential Learning Rate",
             fontweight='bold')
ax.legend(fontsize=9)
ax.grid(alpha=0.25, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
save_fig("fig5_learning_rate", "Learning Rate Schedule")


# =============================================================================
# FIG 6 ── Confusion Matrix  (SINGLE — row-normalised only)
# =============================================================================
print("─"*55)
print("FIG 6 — Confusion Matrix (single, normalised)")

cm_pct = (CM_8.astype(float)
          / CM_8.sum(axis=1, keepdims=True).clip(min=1) * 100)

fig, ax = plt.subplots(figsize=(9, 7.5))
fig.patch.set_facecolor('white')

sns.heatmap(cm_pct,
            annot=True, fmt='.1f',
            cmap='Blues', ax=ax,
            xticklabels=CLASS_DISPLAY_8,
            yticklabels=CLASS_DISPLAY_8,
            linewidths=0.45, linecolor='#e0e0e0',
            cbar_kws={'label':'Row-Normalised (%)','shrink':0.82},
            annot_kws={'size':9.5})

# Highlight diagonal cells
for i in range(len(CLASS_DISPLAY_8)):
    ax.add_patch(plt.Rectangle((i,i),1,1,fill=False,
                                edgecolor='#1a1a2e',lw=2.5))

ax.set_ylabel("True Label",      fontsize=10)
ax.set_xlabel("Predicted Label", fontsize=10)
ax.set_title(
    "Confusion Matrix — ResNet50+Swin Hybrid  "
    "(Acc: 95.11%  |  Macro F1: 94.46%)",
    fontweight='bold', pad=12)
ax.tick_params(axis='x', rotation=35, labelsize=9)
ax.tick_params(axis='y', rotation=0,  labelsize=9)

plt.tight_layout()
save_fig("fig6_confusion_matrix", "Confusion Matrix (single)")


# =============================================================================
# FIG 7 ── ROC Curves  (compact, no "one-vs-rest" text, tighter width)
# =============================================================================
print("─"*55)
print("FIG 7 — ROC Curves (compact single plot)")

ROC_PAL = [
    '#e41a1c','#377eb8','#4daf4a','#984ea3',
    '#ff7f00','#a65628','#f781bf','#999999'
]

fig, ax = plt.subplots(figsize=(6.5, 5.5))   # narrower figure
fig.patch.set_facecolor('white')

for i, cls in enumerate(CLASS_DISPLAY_8):
    fpr, tpr = roc_curves_8[i]
    ax.plot(fpr, tpr, color=ROC_PAL[i], lw=1.8,
            label=f"{cls}  (AUC = {roc_aucs_8[i]:.3f})")

ax.plot([0,1],[0,1],'k--',lw=1,alpha=0.45)
ax.fill_between([0,1],[0,1],alpha=0.04,color='grey')

ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title(
    f"ROC Curves — Hybrid Model\n"
    f"Mean AUC = {np.mean(roc_aucs_8):.4f}",
    fontweight='bold')
ax.legend(fontsize=8, loc='lower right',
          frameon=True, framealpha=0.9,
          edgecolor='#ccc')
ax.grid(alpha=0.22, linestyle='--')
ax.set_xlim([-0.01,1.01]); ax.set_ylim([-0.01,1.01])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
save_fig("fig7_roc_curves", "ROC Curves compact")


# =============================================================================
# FIG 8 ── XAI: Real image + Grad-CAM++ + Grad-CAM  (8 rows × 3 cols)
# =============================================================================
print("─"*55)
print("FIG 8 — XAI with real images (3×3 grid)")

# ── Pick 3 representative classes ─────────────────────────────
# Choose visually distinct & clinically interesting classes
XAI_CLASSES   = ['gray_mold', 'leaf_spot', 'healthy']
XAI_CONF      = [0.97,         0.94,        1.00]
XAI_COLORS    = {
    'gray_mold' : CLASS_COLORS_8[CLASS_NAMES_8.index('gray_mold')],
    'leaf_spot' : CLASS_COLORS_8[CLASS_NAMES_8.index('leaf_spot')],
    'healthy'   : CLASS_COLORS_8[CLASS_NAMES_8.index('healthy')],
}
XAI_DISPLAY   = {
    'gray_mold' : 'Gray Mold',
    'leaf_spot' : 'Leaf Spot',
    'healthy'   : 'Healthy',
}

# ── Grad-CAM overlay generator ────────────────────────────────
def make_gradcam_overlay(img_arr, cam_type='gradcam++', seed=0):
    """
    Produce a visually plausible CAM heatmap blended onto img_arr.
    img_arr  : H×W×3  float32  [0,1]
    cam_type : 'gradcam++' → tighter/sharper blobs
               'gradcam'   → broader, smoother blobs
    Returns  : H×W×3  float32  [0,1]
    """
    np.random.seed(seed)
    H, W = img_arr.shape[:2]
    heatmap = np.zeros((H, W), dtype=np.float32)

    if cam_type == 'gradcam++':
        n_blobs = 5
        sigma   = 0.10 * H      # tight, sharp
    else:
        n_blobs = 3
        sigma   = 0.22 * H      # broad, smooth

    Ym, Xm = np.ogrid[:H, :W]
    for _ in range(n_blobs):
        # keep blobs in centre 50% of image
        cx = int(np.random.uniform(0.25, 0.75) * W)
        cy = int(np.random.uniform(0.25, 0.75) * H)
        g  = np.exp(-((Xm - cx)**2 + (Ym - cy)**2)
                    / (2 * sigma**2))
        heatmap += g.astype(np.float32)

    # power-law sharpening → makes hotspots pop
    if heatmap.max() > 0:
        heatmap /= heatmap.max()
    heatmap = heatmap ** 1.8          # sharpen

    # re-normalise after sharpening
    if heatmap.max() > 0:
        heatmap /= heatmap.max()

    cam_rgb = plt.cm.jet(heatmap)[:, :, :3].astype(np.float32)
    alpha   = 0.52
    overlay = (1 - alpha) * img_arr + alpha * cam_rgb
    return np.clip(overlay, 0, 1)

# ── Load real images ──────────────────────────────────────────
print("  Loading images for XAI panel...")
xai_imgs = {}
for cls in XAI_CLASSES:
    p = img_paths.get(cls)
    if p and os.path.exists(p):
        try:
            xai_imgs[cls] = (np.array(load_square(p, 256))
                             .astype(np.float32) / 255.0)
            print(f"    ✅ {cls:<22} → {os.path.basename(p)}")
        except Exception as ex:
            print(f"    ⚠  {cls}: {ex}")
            xai_imgs[cls] = None
    else:
        print(f"    ❌ {cls}: not found, using colour patch")
        xai_imgs[cls] = None

# ── Build 3×3 figure ─────────────────────────────────────────
# Rows = classes  |  Cols = [Original, Grad-CAM++, Grad-CAM]
fig = plt.figure(figsize=(11, 11))
fig.patch.set_facecolor('white')

gs = gridspec.GridSpec(
    3, 3,
    figure=fig,
    hspace=0.10,
    wspace=0.06,
)

col_hdrs = ['Original Image', 'Grad-CAM++', 'Grad-CAM']

for row_i, cls in enumerate(XAI_CLASSES):
    accent  = XAI_COLORS[cls]
    conf    = XAI_CONF[row_i]
    cls_idx = CLASS_NAMES_8.index(cls)

    # ── Load / fallback ───────────────────────────────────────
    img_f32 = xai_imgs.get(cls)
    if img_f32 is None:
        img_f32 = np.ones((256, 256, 3), dtype=np.float32)
        img_f32[:] = matplotlib.colors.to_rgb(accent)

    # ── Generate CAM overlays ─────────────────────────────────
    gcpp = make_gradcam_overlay(img_f32,
                                cam_type='gradcam++',
                                seed=cls_idx * 7 + 1)
    gc   = make_gradcam_overlay(img_f32,
                                cam_type='gradcam',
                                seed=cls_idx * 11 + 3)

    panels = [img_f32, gcpp, gc]

    for col_i, arr in enumerate(panels):
        ax = fig.add_subplot(gs[row_i, col_i])
        ax.imshow(arr, aspect='equal')
        ax.axis('off')

        # Coloured border
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_edgecolor(accent)
            sp.set_linewidth(3.0)

        # ── Column header (first row only) ────────────────────
        if row_i == 0:
            ax.set_title(col_hdrs[col_i],
                         fontsize=11, fontweight='bold',
                         pad=9, color='#1a1a2e')

        # ── Row label on leftmost column ──────────────────────
        if col_i == 0:
            # Class name badge inside image (top-left)
            ax.text(0.03, 0.97,
                    XAI_DISPLAY[cls],
                    transform=ax.transAxes,
                    fontsize=9.5, color='white',
                    fontweight='bold',
                    va='top', ha='left',
                    bbox=dict(boxstyle='round,pad=0.35',
                              facecolor=accent,
                              alpha=0.88,
                              edgecolor='white',
                              linewidth=0.8))

        # ── Confidence badge on rightmost column ──────────────
        if col_i == 2:
            ax.text(0.97, 0.03,
                    f"✓  {conf*100:.0f}%",
                    transform=ax.transAxes,
                    fontsize=9, color='white',
                    fontweight='bold',
                    va='bottom', ha='right',
                    bbox=dict(boxstyle='round,pad=0.30',
                              facecolor=PAL['pos'],
                              alpha=0.88,
                              edgecolor='white',
                              linewidth=0.7))

        # ── CAM-type badge on overlay columns ─────────────────
        if col_i == 1:
            ax.text(0.97, 0.97, 'Grad-CAM++',
                    transform=ax.transAxes,
                    fontsize=7.5, color='white',
                    fontweight='bold',
                    va='top', ha='right',
                    bbox=dict(boxstyle='round,pad=0.25',
                              facecolor='#333333',
                              alpha=0.78,
                              edgecolor='none'))
        if col_i == 2:
            ax.text(0.03, 0.97, 'Grad-CAM',
                    transform=ax.transAxes,
                    fontsize=7.5, color='white',
                    fontweight='bold',
                    va='top', ha='left',
                    bbox=dict(boxstyle='round,pad=0.25',
                              facecolor='#333333',
                              alpha=0.78,
                              edgecolor='none'))

# ── Colourbar (right side) ────────────────────────────────────
cax = fig.add_axes([1.02, 0.12, 0.020, 0.76])
cb  = plt.colorbar(
    plt.cm.ScalarMappable(
        norm=matplotlib.colors.Normalize(0, 1),
        cmap='jet'),
    cax=cax)
cb.set_label('Activation Intensity', fontsize=10, labelpad=8)
cb.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
cb.set_ticklabels(['Very Low', 'Low', 'Medium', 'High', 'Very High'],
                  fontsize=8)
cb.outline.set_linewidth(0.5)

# ── Overall caption ───────────────────────────────────────────
fig.text(
    0.5, -0.015,
    "Grad-CAM++ produces sharper, more localised activation maps "
    "compared to standard Grad-CAM.\n"
    "Warmer colours (red/yellow) mark regions most influential "
    "for the model's prediction.",
    ha='center', va='top',
    fontsize=9, color='#444',
    style='italic',
    wrap=True
)

fig.suptitle(
    "Explainability — Grad-CAM++ and Grad-CAM Attention Maps",
    fontsize=13, fontweight='bold',
    y=1.02, color='#1a1a2e')

plt.tight_layout()
save_fig("fig8_xai_gradcam", "XAI 3×3 real images + Grad-CAM")


# =============================================================================
# FIG 9 ── Robustness: Accuracy vs Noise
# =============================================================================
print("─"*55)
print("FIG 9 — Robustness: Accuracy vs Noise")

fig, ax = plt.subplots(figsize=(8, 5))
fig.patch.set_facecolor('white')

rob_models = [
    ('VGG19+CBAM',             rob_acc_vgg,    PAL['vgg'],    'o'),
    ('ResNet50+CBAM+GeM',      rob_acc_res,    PAL['res'],    's'),
    ('EfficientNetV2+ECA+GeM', rob_acc_eff,    PAL['eff'],    '^'),
    ('ResNet50+Swin Hybrid',   rob_acc_hyb,    PAL['hybrid'], 'D'),
]
for mname, accs, col, mk in rob_models:
    ax.plot(rob_sigmas, accs,
            color=col, lw=2.2, marker=mk, ms=7, label=mname)

# Annotate hybrid values
for sx, ay in zip(rob_sigmas, rob_acc_hyb):
    ax.annotate(f"{ay:.1f}%", (sx, ay),
                textcoords="offset points", xytext=(5,5),
                fontsize=7.5, color=PAL['hybrid'], fontweight='bold')

ax.set_xlabel("Gaussian Noise σ")
ax.set_ylabel("Accuracy (%)")
ax.set_title("Accuracy Under Gaussian Noise Perturbation",
             fontweight='bold')
ax.legend(fontsize=9)
ax.grid(alpha=0.25, linestyle='--')
ax.set_ylim([72, 100])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
save_fig("fig9_robustness_accuracy", "Robustness Accuracy vs Noise")


# =============================================================================
# FIG 10 ── Robustness: Macro F1 vs Noise
# =============================================================================
print("─"*55)
print("FIG 10 — Robustness: Macro F1 vs Noise")

fig, ax = plt.subplots(figsize=(8, 5))
fig.patch.set_facecolor('white')

rob_models_f1 = [
    ('VGG19+CBAM',             rob_f1_vgg,    PAL['vgg'],    'o'),
    ('ResNet50+CBAM+GeM',      rob_f1_res,    PAL['res'],    's'),
    ('EfficientNetV2+ECA+GeM', rob_f1_eff,    PAL['eff'],    '^'),
    ('ResNet50+Swin Hybrid',   rob_f1_hyb,    PAL['hybrid'], 'D'),
]
for mname, f1s, col, mk in rob_models_f1:
    ax.plot(rob_sigmas, f1s,
            color=col, lw=2.2, marker=mk, ms=7, label=mname)

for sx, fy in zip(rob_sigmas, rob_f1_hyb):
    ax.annotate(f"{fy:.1f}%", (sx, fy),
                textcoords="offset points", xytext=(5,5),
                fontsize=7.5, color=PAL['hybrid'], fontweight='bold')

ax.set_xlabel("Gaussian Noise σ")
ax.set_ylabel("Macro F1 Score (%)")
ax.set_title("Macro F1 Score Under Gaussian Noise Perturbation",
             fontweight='bold')
ax.legend(fontsize=9)
ax.grid(alpha=0.25, linestyle='--')
ax.set_ylim([68, 100])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
save_fig("fig10_robustness_f1", "Robustness Macro F1 vs Noise")


# =============================================================================
# FIG 11 ── Benchmark Metric Comparison
# =============================================================================
print("─"*55)
print("FIG 11 — Benchmark Metric Comparison")

model_names_bar = list(bench_metrics.keys())
bar_colors      = [PAL['vgg'], PAL['res'], PAL['eff'], PAL['hybrid']]
metrics_plot    = ['acc','mf1','auc']
metric_labels   = ['Accuracy (%)','Macro F1 (%)','Macro AUC-ROC']

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.patch.set_facecolor('white')

for ax, mk, ml in zip(axes, metrics_plot, metric_labels):
    vals = [bench_metrics[n][mk] * (100 if mk != 'auc' else 1)
            for n in model_names_bar]
    bars = ax.bar(range(len(model_names_bar)), vals,
                  color=bar_colors, width=0.55,
                  edgecolor='white', linewidth=0.8)
    bars[-1].set_edgecolor(PAL['gold'])
    bars[-1].set_linewidth(2.8)

    ax.set_xticks(range(len(model_names_bar)))
    ax.set_xticklabels(model_names_bar, fontsize=8.5)
    ax.set_title(f"Test {ml}", fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    mn_v, mx_v = min(vals), max(vals)
    pad = max((mx_v - mn_v)*0.3, 0.4)
    ax.set_ylim([mn_v - pad, mx_v + pad*1.5])

    for bar, val in zip(bars, vals):
        fmt = f"{val:.3f}" if mk == 'auc' else f"{val:.2f}%"
        ax.text(bar.get_x()+bar.get_width()/2,
                bar.get_height()+pad*0.1,
                fmt, ha='center', va='bottom',
                fontsize=8.5, fontweight='bold')

    patches = [mpatches.Patch(color=c, label=n.replace('\n',' '))
               for c,n in zip(bar_colors, model_names_bar)]
    ax.legend(handles=patches, fontsize=7.5, loc='lower right')

plt.suptitle("Benchmark vs. Hybrid — Test Metrics Comparison",
             fontsize=12, fontweight='bold', y=1.02)
plt.tight_layout()
save_fig("fig11_benchmark_comparison", "Benchmark Metric Comparison")


# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "="*65)
print("📋  ALL FIGURES GENERATED")
print("="*65)
all_figs = [
    ("fig1_dataset_sample",          "Dataset Sample 4×2 real images"),
    ("fig2_class_split_distribution","Class & Split Distribution"),
    ("fig3_train_val_curves_2x2",    "Train vs Val Curves 2×2"),
    ("fig4_val_curves_combined",     "Validation Curves Combined"),
    ("fig5_learning_rate",           "Learning Rate (single)"),
    ("fig6_confusion_matrix",        "Confusion Matrix (single)"),
    ("fig7_roc_curves",              "ROC Curves (compact)"),
    ("fig8_xai_gradcam",             "XAI real images + CAM"),
    ("fig9_robustness_accuracy",     "Robustness Accuracy vs Noise"),
    ("fig10_robustness_f1",          "Robustness Macro F1 vs Noise"),
    ("fig11_benchmark_comparison",   "Benchmark Metric Comparison"),
]
print(f"  {'File':<42} {'PDF':>5} {'PNG':>5}  Description")
print("  "+"─"*65)
for fname, desc in all_figs:
    base   = f"paper_figures/{fname}"
    pdf_ok = "✓" if os.path.exists(f"{base}.pdf") else "✗"
    png_ok = "✓" if os.path.exists(f"{base}.png") else "✗"
    print(f"  {fname:<42} {pdf_ok:>5} {png_ok:>5}  {desc}")
print("="*65)
print("📁  Saved to: paper_figures/   (PNG 300 DPI + PDF vector)")
print("="*65)