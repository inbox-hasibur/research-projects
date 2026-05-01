# =============================================================================
# 🍓 STRAWBERRY DISEASE DETECTION — Double-Source Dataset + Best Hybrid Model
# Benchmark  →  VGG19-BN  |  ResNet50  |  EfficientNetV2-S
# Datasets: Afzaal (2500 imgs) + PlantVillage
# XAI: Grad-CAM++ | EigenCAM | Branch Contribution | Robustness | Ablation
# =============================================================================

# ============================================================
# STEP 1 — Imports & Config
# ============================================================
import os, time, json, warnings, glob, shutil
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from PIL import Image
import matplotlib
matplotlib.rcParams.update({
    'font.family'      : 'DejaVu Sans',
    'font.size'        : 10,
    'axes.titlesize'   : 12,
    'axes.labelsize'   : 10,
    'xtick.labelsize'  : 9,
    'ytick.labelsize'  : 9,
    'figure.dpi'       : 150,
    'savefig.dpi'      : 300,          # IEEE print quality
    'savefig.bbox'     : 'tight',
    'axes.spines.top'  : False,
    'axes.spines.right': False,
})
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             roc_curve, auc, classification_report)
from sklearn.preprocessing import label_binarize
import timm
warnings.filterwarnings('ignore')

# ── Colour palette (IEEE-safe, colourblind-friendly) ─────────
PAL = {
    'vgg'  : '#2166ac',   # blue
    'res'  : '#d6604d',   # red-orange
    'eff'  : '#4dac26',   # green
    'swin' : '#f4a582',   # light orange
    'ours' : '#1a1a2e',   # near-black
    'pos'  : '#2ecc71',
    'neg'  : '#e74c3c',
}

# ── Device ────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"   GPU : {torch.cuda.get_device_name(0)}")
    cap = torch.cuda.get_device_capability()
    print(f"   Cap : sm_{cap[0]*10+cap[1]}")

_cap    = torch.cuda.get_device_capability()[0] if torch.cuda.is_available() else 0
USE_AMP = _cap >= 7
print(f"   AMP : {'ON' if USE_AMP else 'OFF'}")

# ── Hyperparameters ───────────────────────────────────────────
IMG_SIZE      = 224
BATCH_SIZE    = 16
BENCH_EPOCHS  = 30
HYBRID_EPOCHS = 60
LR            = 5e-5
WEIGHT_DECAY  = 1e-4
LABEL_SMOOTH  = 0.1
MIXUP_ALPHA   = 0.3
PATIENCE      = 8
CROP_PROB     = 0.6

# ── Label map ─────────────────────────────────────────────────
LABEL_MAP = {
    "angular_leafspot": 0,
    "anthracnose":      1,
    "blossom_blight":   2,
    "gray_mold":        3,
    "leaf_spot":        4,
    "powdery_mildew":   5,
    "leaf_scorch":      6,
    "healthy":          7,
}
IDX_TO_CLASS = {v: k for k, v in LABEL_MAP.items()}
NUM_CLASSES  = len(LABEL_MAP)

# ── Paths ──────────────────────────────────────────────────────
AFZAAL_ROOT  = "/kaggle/input/datasets/usmanafzaal/strawberry-disease-detection-dataset"
AFZAAL_TRAIN = os.path.join(AFZAAL_ROOT, "train")
AFZAAL_VAL   = os.path.join(AFZAAL_ROOT, "val")
AFZAAL_TEST  = os.path.join(AFZAAL_ROOT, "test")

PV_COLOR_ROOT    = "/kaggle/input/datasets/abdallahalidev/plantvillage-dataset/color"
PV_STRAW_SCORCH  = os.path.join(PV_COLOR_ROOT, "Strawberry___Leaf_scorch")
PV_STRAW_HEALTHY = os.path.join(PV_COLOR_ROOT, "Strawberry___healthy")

SAVE_BENCH  = "best_benchmark_model.pth"
SAVE_HYBRID = "best_hybrid_strawberry.pth"

print(f"\n✅ Config | Classes:{NUM_CLASSES} | Device:{DEVICE}")
print(f"   Classes: {list(LABEL_MAP.keys())}")


# ============================================================
# STEP 2 — Annotation Utils
# ============================================================
def load_annotation_bbox(json_path):
    if not os.path.exists(json_path): return None
    try:
        with open(json_path) as f: data = json.load(f)
        ax, ay = [], []
        for val in data.values():
            regions = val.get('regions', [])
            if isinstance(regions, dict): regions = list(regions.values())
            for region in regions:
                s = region.get('shape_attributes', {})
                t = s.get('name', '')
                if t == 'polygon':
                    ax.extend(s.get('all_points_x', []))
                    ay.extend(s.get('all_points_y', []))
                elif t == 'rect':
                    x,y,w,h = s.get('x',0),s.get('y',0),s.get('width',0),s.get('height',0)
                    ax += [x, x+w]; ay += [y, y+h]
                elif t == 'ellipse':
                    cx,cy,rx,ry = s.get('cx',0),s.get('cy',0),s.get('rx',0),s.get('ry',0)
                    ax += [cx-rx, cx+rx]; ay += [cy-ry, cy+ry]
        return (min(ax), min(ay), max(ax), max(ay)) if ax else None
    except Exception: return None

def annotation_crop(img, bbox, padding=0.20):
    if bbox is None: return img
    w, h = img.size
    x1,y1,x2,y2 = bbox
    bw,bh = x2-x1, y2-y1
    if bw<=0 or bh<=0: return img
    px,py = int(bw*padding), int(bh*padding)
    x1,y1 = max(0,x1-px), max(0,y1-py)
    x2,y2 = min(w,x2+px), min(h,y2+py)
    return img.crop((x1,y1,x2,y2)) if (x2-x1)>=10 and (y2-y1)>=10 else img


# ============================================================
# STEP 3 — Dataset Scanning  (Afzaal + PlantVillage ONLY)
# ============================================================
def label_from_filename(fname, label_map):
    base = os.path.splitext(os.path.basename(fname))[0].lower()
    best_cls, best_idx = None, None
    for cls_name, cls_idx in label_map.items():
        if base.startswith(cls_name):
            if best_cls is None or len(cls_name) > len(best_cls):
                best_cls, best_idx = cls_name, cls_idx
    return best_idx, best_cls

def scan_afzaal_split(split_dir, split_name="split"):
    paths, labels = [], []
    if not os.path.exists(split_dir):
        print(f"  ⚠️  Not found: {split_dir}"); return paths, labels
    for fn in sorted(os.listdir(split_dir)):
        if not fn.lower().endswith(('.jpg','.jpeg','.png','.bmp')): continue
        cls_idx, cls_name = label_from_filename(fn, LABEL_MAP)
        if cls_idx is None: continue
        paths.append(os.path.join(split_dir, fn))
        labels.append(cls_idx)
    cc = {}
    for l in labels: cc[l] = cc.get(l,0)+1
    found = [IDX_TO_CLASS[k] for k in sorted(cc)]
    print(f"  [Afzaal-{split_name}] {len(paths)} imgs | {found}")
    return paths, labels

def scan_plantvillage_strawberry():
    paths, labels = [], []
    for folder, cls_idx in [(PV_STRAW_SCORCH,  LABEL_MAP["leaf_scorch"]),
                             (PV_STRAW_HEALTHY, LABEL_MAP["healthy"])]:
        if not os.path.exists(folder):
            print(f"  ⚠️  Not found: {folder}"); continue
        imgs = [f for f in os.listdir(folder)
                if f.lower().endswith(('.jpg','.jpeg','.png'))]
        for fn in imgs:
            paths.append(os.path.join(folder, fn))
            labels.append(cls_idx)
        print(f"  [PlantVillage] {IDX_TO_CLASS[cls_idx]}: {len(imgs)} imgs")
    return paths, labels

print("\n📂 Scanning datasets...")
afl_tr_paths,  afl_tr_labels  = scan_afzaal_split(AFZAAL_TRAIN, "train")
afl_vl_paths,  afl_vl_labels  = scan_afzaal_split(AFZAAL_VAL,   "val")
afl_te_paths,  afl_te_labels  = scan_afzaal_split(AFZAAL_TEST,  "test")
pv_paths,      pv_labels      = scan_plantvillage_strawberry()

# ── Split PlantVillage 80/10/10 ───────────────────────────────
def split_extra(paths, labels, seed=42):
    if len(paths) == 0:
        return [],[],[],[],[],[]
    X_tv,X_te,y_tv,y_te = train_test_split(
        paths, labels, test_size=0.10, random_state=seed, stratify=labels)
    X_tr,X_vl,y_tr,y_vl = train_test_split(
        X_tv, y_tv, test_size=0.111, random_state=seed, stratify=y_tv)
    return X_tr, X_vl, X_te, y_tr, y_vl, y_te

pv_tr,pv_vl,pv_te,pv_ytr,pv_yvl,pv_yte = split_extra(pv_paths, pv_labels)

# ── FIXED: mb_ variables completely removed ───────────────────
train_paths  = afl_tr_paths + pv_tr
train_labels = afl_tr_labels + pv_ytr
val_paths    = afl_vl_paths + pv_vl
val_labels   = afl_vl_labels + pv_yvl
test_paths   = afl_te_paths          # Afzaal test ONLY — pure benchmark
test_labels  = afl_te_labels

print(f"\n✅ Dataset Summary:")
print(f"   Train : {len(train_paths):,}")
print(f"   Val   : {len(val_paths):,}")
print(f"   Test  : {len(test_paths):,}  (Afzaal-only, pure)")
print(f"   Total : {len(train_paths)+len(val_paths)+len(test_paths):,}")
print(f"\n   Train class distribution:")
cc = {}
for l in train_labels: cc[IDX_TO_CLASS[l]] = cc.get(IDX_TO_CLASS[l],0)+1
for cls,cnt in sorted(cc.items()): print(f"   {cls:22s}: {cnt:4d}")

n_ann   = sum(1 for p in train_paths
              if os.path.exists(os.path.splitext(p)[0]+'.json'))
USE_CROP = n_ann > 0
print(f"\n   Annotations: {n_ann}/{len(train_paths)} "
      f"→ crop {'✅ ON' if USE_CROP else '❌ OFF'}")


# ============================================================
# STEP 4 — Transforms
# ============================================================
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE+48, IMG_SIZE+48)),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.08),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),
])
val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


# ============================================================
# STEP 5 — Dataset & DataLoader
# ============================================================
class StrawberryDataset(Dataset):
    def __init__(self, paths, labels, transform=None, use_ann_crop=False):
        self.paths        = paths
        self.labels       = labels
        self.transform    = transform
        self.use_ann_crop = use_ann_crop

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.use_ann_crop and np.random.random() < CROP_PROB:
            bbox = load_annotation_bbox(
                os.path.splitext(self.paths[idx])[0]+'.json')
            if bbox: img = annotation_crop(img, bbox)
        if self.transform: img = self.transform(img)
        return img, torch.tensor(self.labels[idx], dtype=torch.long)

def make_loader(paths, labels, tf, shuffle=False, ann_crop=False):
    ds = StrawberryDataset(paths, labels, tf, ann_crop)
    if shuffle:
        cc  = np.bincount(labels, minlength=NUM_CLASSES)
        wpc = 1.0 / np.where(cc==0, 1, cc)
        sw  = torch.tensor([wpc[l] for l in labels], dtype=torch.double)
        sampler = WeightedRandomSampler(sw, len(sw), replacement=True)
        return DataLoader(ds, batch_size=BATCH_SIZE, sampler=sampler,
                         num_workers=2, pin_memory=True)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                     num_workers=2, pin_memory=True)

train_loader = make_loader(train_paths, train_labels, train_tf,
                           shuffle=True, ann_crop=USE_CROP)
val_loader   = make_loader(val_paths,   val_labels,   val_tf)
test_loader  = make_loader(test_paths,  test_labels,  val_tf)
print("✅ DataLoaders ready.")


# ============================================================
# STEP 6 — Shared Modules
# ============================================================
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels//reduction, 8)
        self.mlp = nn.Sequential(
            nn.Linear(channels, mid, bias=False), nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False))
    def forward(self, x):
        avg = self.mlp(F.adaptive_avg_pool2d(x,1).flatten(1))
        mx  = self.mlp(F.adaptive_max_pool2d(x,1).flatten(1))
        return x * torch.sigmoid(avg+mx).unsqueeze(-1).unsqueeze(-1)

class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2,1,7,padding=3,bias=False)
    def forward(self, x):
        avg = x.mean(1,keepdim=True)
        mx,_= x.max(1,keepdim=True)
        return x * torch.sigmoid(self.conv(torch.cat([avg,mx],1)))

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention()
    def forward(self, x): return self.sa(self.ca(x))

class ECA(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        import math
        t = int(abs((math.log2(channels)+b)/gamma))
        k = t if t%2 else t+1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv     = nn.Conv1d(1,1,k,padding=(k-1)//2,bias=False)
    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1,-2))
        return x * torch.sigmoid(y.transpose(-1,-2).unsqueeze(-1))

class GeMPooling(nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(p)); self.eps = eps
    def forward(self, x):
        return F.adaptive_avg_pool2d(
            x.clamp(min=self.eps).pow(self.p),1
        ).pow(1.0/self.p).flatten(1)

class SmoothCE(nn.Module):
    def __init__(self, num_classes=8, smoothing=0.1):
        super().__init__()
        self.nc = num_classes; self.sm = smoothing
    def forward(self, logits, targets):
        lp = F.log_softmax(logits,-1)
        if targets.dim()==1:
            st = torch.full_like(lp, self.sm/(self.nc-1))
            st.scatter_(1, targets.unsqueeze(1), 1.0-self.sm)
        else:
            st = targets*(1-self.sm)+self.sm/self.nc
        return -(st*lp).sum(-1).mean()

criterion = SmoothCE(NUM_CLASSES, LABEL_SMOOTH)

def mixup_data(x, y, alpha=0.3):
    lam = float(np.random.beta(alpha,alpha)) if alpha>0 else 1.0
    device = x.device
    x_c,y_c = x.detach().cpu(), y.detach().cpu()
    idx = torch.randperm(x_c.size(0))
    ya  = F.one_hot(y_c,      NUM_CLASSES).float()
    yb  = F.one_hot(y_c[idx], NUM_CLASSES).float()
    mx  = lam*x_c+(1-lam)*x_c[idx]
    my  = lam*ya +(1-lam)*yb
    return mx.to(device), my.to(device)

print("✅ Shared modules: CBAM | ECA | GeM | SmoothCE | Mixup")


# ============================================================
# STEP 7 — Early Stopping & Train/Eval Helpers
# ============================================================
class EarlyStopping:
    def __init__(self, patience=8, min_delta=1e-4, save_path=SAVE_BENCH):
        self.patience=patience; self.min_delta=min_delta
        self.save_path=save_path; self.best_loss=float('inf')
        self.counter=0; self.best_epoch=0
    def step(self, val_loss, epoch, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss=val_loss; self.counter=0
            self.best_epoch=epoch
            torch.save(model.state_dict(), self.save_path)
            return False
        self.counter+=1
        return self.counter>=self.patience

def train_one_epoch(model, loader, crit, opt):
    model.train(); total=0.0
    for imgs,lbls in loader:
        imgs,lbls = imgs.to(DEVICE), lbls.to(DEVICE)
        imgs,targets = mixup_data(imgs,lbls,MIXUP_ALPHA) if MIXUP_ALPHA>0 \
                       else (imgs, lbls)
        opt.zero_grad(set_to_none=True)
        loss = crit(model(imgs), targets)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); total+=loss.item()
    return total/len(loader)

@torch.no_grad()
def evaluate(model, loader, crit):
    model.eval()
    total,probs,preds,gts = 0.0,[],[],[]
    for imgs,lbls in loader:
        imgs,lbls = imgs.to(DEVICE), lbls.to(DEVICE)
        logits = model(imgs)
        total += crit(logits,lbls).item()
        probs.extend(F.softmax(logits,-1).cpu().numpy())
        preds.extend(logits.argmax(-1).cpu().numpy())
        gts.extend(lbls.cpu().numpy())
    return total/len(loader), accuracy_score(gts,preds), \
           np.array(probs), preds, gts

def full_metrics(gts, preds, probs):
    acc  = accuracy_score(gts,preds)
    prec = precision_score(gts,preds,average='macro',zero_division=0)
    rec  = recall_score(gts,preds,average='macro',zero_division=0)
    mf1  = f1_score(gts,preds,average='macro',zero_division=0)
    wf1  = f1_score(gts,preds,average='weighted',zero_division=0)
    try:    auc_s = roc_auc_score(gts,probs,multi_class='ovr',average='macro')
    except: auc_s = float('nan')
    return dict(acc=acc,prec=prec,rec=rec,mf1=mf1,wf1=wf1,auc=auc_s)


# ============================================================
# STEP 8 — Benchmark Model Definitions
# ============================================================
class VGG19Model(nn.Module):
    def __init__(self, num_classes=8, drop=0.5):
        super().__init__()
        base = models.vgg19_bn(pretrained=True)
        self.features   = base.features
        self.cbam       = CBAM(512)
        self.avgpool    = nn.AdaptiveAvgPool2d(7)
        self.classifier = nn.Sequential(
            nn.Linear(512*7*7,1024), nn.ReLU(True), nn.Dropout(drop),
            nn.Linear(1024,256),    nn.ReLU(True), nn.Dropout(drop/2),
            nn.Linear(256,num_classes))
    def forward(self, x):
        x = self.features(x); x = self.cbam(x)
        x = self.avgpool(x).flatten(1)
        return self.classifier(x)

class ResNet50Model(nn.Module):
    def __init__(self, num_classes=8, drop=0.4):
        super().__init__()
        base = models.resnet50(pretrained=True)
        self.stem   = nn.Sequential(base.conv1,base.bn1,base.relu,base.maxpool)
        self.layer1 = base.layer1; self.layer2 = base.layer2
        self.layer3 = base.layer3; self.layer4 = base.layer4
        self.cbam   = CBAM(2048)
        self.gem    = GeMPooling(p=3.0)
        self.head   = nn.Sequential(
            nn.Linear(2048,512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(drop),
            nn.Linear(512,num_classes))
    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x); x = self.layer2(x)
        x = self.layer3(x); x = self.layer4(x)
        x = self.cbam(x);   x = self.gem(x)
        return self.head(x)

class EfficientNetV2Model(nn.Module):
    def __init__(self, num_classes=8, drop=0.4):
        super().__init__()
        self.backbone = timm.create_model(
            'tf_efficientnetv2_s', pretrained=True, num_classes=0, global_pool='')
        EFF_DIM = 1280
        self.eca  = ECA(EFF_DIM)
        self.gem  = GeMPooling(p=3.0)
        self.head = nn.Sequential(
            nn.Linear(EFF_DIM,512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(drop),
            nn.Linear(512,num_classes))
    def forward(self, x):
        x = self.backbone.forward_features(x)
        x = self.eca(x); x = self.gem(x)
        return self.head(x)

BENCH_CONFIGS = {
    "VGG19+CBAM"              : VGG19Model,
    "ResNet50+CBAM+GeM"       : ResNet50Model,
    "EfficientNetV2+ECA+GeM"  : EfficientNetV2Model,
}
print("\n📐 Model parameter counts:")
for name,cls in BENCH_CONFIGS.items():
    m = cls(NUM_CLASSES)
    print(f"  {name:<32}: {sum(p.numel() for p in m.parameters() if p.requires_grad):>12,}")
    del m


# ============================================================
# STEP 9 — Benchmark Training
# ============================================================
bench_results   = {}
bench_histories = {}

print("\n" + "="*70)
print("📊  BENCHMARK: VGG19 | ResNet50 | EfficientNetV2-S")
print("="*70)

for bench_name, ModelClass in BENCH_CONFIGS.items():
    print(f"\n🔷  {bench_name}")
    print("─"*60)
    model_b = ModelClass(NUM_CLASSES).to(DEVICE)

    try:    bb_params = list(model_b.backbone.parameters())
    except AttributeError:
        try:    bb_params = list(model_b.features.parameters())+list(model_b.cbam.parameters())
        except: bb_params = (list(model_b.stem.parameters())+list(model_b.layer1.parameters())+
                             list(model_b.layer2.parameters())+list(model_b.layer3.parameters())+
                             list(model_b.layer4.parameters()))

    bb_ids    = {id(p) for p in bb_params}
    new_params = [p for p in model_b.parameters() if id(p) not in bb_ids]

    opt_b = optim.AdamW([{'params':bb_params,'lr':LR*0.1},
                          {'params':new_params,'lr':LR}],
                         weight_decay=WEIGHT_DECAY)
    sch_b = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt_b,T_0=10,T_mult=2)
    es_b  = EarlyStopping(patience=PATIENCE,
                          save_path=f"bench_{bench_name.replace('+','_')}.pth")
    hist = {'tr_loss':[],'vl_loss':[],'vl_acc':[]}

    for ep in range(1, BENCH_EPOCHS+1):
        t0   = time.time()
        tr_l = train_one_epoch(model_b, train_loader, criterion, opt_b)
        vl_l,vl_a,_,_,_ = evaluate(model_b, val_loader, criterion)
        sch_b.step()
        hist['tr_loss'].append(tr_l)
        hist['vl_loss'].append(vl_l)
        hist['vl_acc'].append(vl_a)
        stop = es_b.step(vl_l, ep, model_b)
        flag = "🏅" if es_b.counter==0 else f"({es_b.counter}/{PATIENCE})"
        if ep%5==0 or ep==1 or stop:
            print(f"  Ep{ep:02d} | {time.time()-t0:.0f}s | "
                  f"Tr:{tr_l:.4f} Vl:{vl_l:.4f} Acc:{vl_a*100:.2f}% {flag}")
        if stop:
            print(f"  ⏹ ep{ep} | best ep{es_b.best_epoch}"); break

    model_b.load_state_dict(torch.load(es_b.save_path, map_location=DEVICE))
    _,_,probs,preds,gts = evaluate(model_b, test_loader, criterion)
    m = full_metrics(gts, preds, probs)
    bench_results[bench_name]   = m
    bench_histories[bench_name] = hist
    print(f"\n  ✅ {bench_name} | Acc:{m['acc']*100:.3f}%  "
          f"MacF1:{m['mf1']*100:.3f}%  AUC:{m['auc']:.4f}")
    del model_b; torch.cuda.empty_cache()

winner_name  = max(bench_results, key=lambda k: bench_results[k]['mf1'])
winner_f1    = bench_results[winner_name]['mf1']*100
WINNER_CLASS = BENCH_CONFIGS[winner_name]
WINNER_SAVE  = f"bench_{winner_name.replace('+','_')}.pth"
print(f"\n🏆  WINNER: {winner_name}  (Macro F1={winner_f1:.3f}%)")


# ============================================================
# STEP 10 — Fig 1: Benchmark Comparison (IEEE two-column)
# ============================================================
BENCH_PAL = {
    "VGG19+CBAM"             : PAL['vgg'],
    "ResNet50+CBAM+GeM"      : PAL['res'],
    "EfficientNetV2+ECA+GeM" : PAL['eff'],
}

# Fig 1a — Learning curves
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for name, hist in bench_histories.items():
    ep  = range(1, len(hist['tr_loss'])+1)
    col = BENCH_PAL[name]
    axes[0].plot(ep, hist['tr_loss'], color=col, lw=1.8, label=name)
    axes[1].plot(ep, hist['vl_loss'], color=col, lw=1.8, label=name)
    axes[2].plot(ep, [a*100 for a in hist['vl_acc']], color=col, lw=1.8, label=name)

for ax, title, ylabel in zip(
    axes,
    ["(a) Training Loss", "(b) Validation Loss", "(c) Validation Accuracy (%)"],
    ["Loss", "Loss", "Accuracy (%)"]):
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel)
    ax.legend(fontsize=7); ax.grid(alpha=0.3, linestyle='--')

plt.suptitle("Fig. 1 — Benchmark Model Learning Curves",
             fontsize=11, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig("fig1_benchmark_curves.pdf", bbox_inches='tight')
plt.savefig("fig1_benchmark_curves.png")
plt.show(); print("✅ Fig 1 saved")

# Fig 2 — Test metric bar chart
metrics_keys   = ['acc','mf1','auc']
metrics_labels = ['Accuracy (%)', 'Macro F1 (%)', 'Macro AUC-ROC']
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
for ax, mk, ml in zip(axes, metrics_keys, metrics_labels):
    names = list(bench_results.keys())
    vals  = [bench_results[n][mk]*(100 if mk!='auc' else 1) for n in names]
    cols  = [BENCH_PAL[n] for n in names]
    bars  = ax.bar(range(len(names)), vals, color=cols, width=0.55,
                   edgecolor='white', linewidth=0.8)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.replace('+','\n') for n in names], fontsize=8)
    ax.set_title(f"(a/b/c) Test {ml}", fontweight='bold')
    ax.set_ylim([min(vals)*0.97, max(vals)*1.02])
    for bar,val in zip(bars,vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.001,
                f"{val:.2f}", ha='center', va='bottom', fontsize=9,
                fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')

plt.suptitle("Fig. 2 — Benchmark Test Metrics Comparison",
             fontsize=11, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig("fig2_benchmark_bars.pdf", bbox_inches='tight')
plt.savefig("fig2_benchmark_bars.png")
plt.show(); print("✅ Fig 2 saved")


# ============================================================
# STEP 11 — Hybrid Architecture
# ============================================================
def _swin_pool(feat):
    if feat.dim()==4: feat=feat.mean(dim=[1,2])
    return feat.contiguous()

class CrossAttentionGate(nn.Module):
    def __init__(self, dim=512, heads=8):
        super().__init__()
        self.heads=heads; self.scale=(dim//heads)**-0.5
        self.q1=nn.Linear(dim,dim,bias=False); self.k2=nn.Linear(dim,dim,bias=False)
        self.v2=nn.Linear(dim,dim,bias=False); self.q2=nn.Linear(dim,dim,bias=False)
        self.k1=nn.Linear(dim,dim,bias=False); self.v1=nn.Linear(dim,dim,bias=False)
        self.out1=nn.Linear(dim,dim,bias=False); self.out2=nn.Linear(dim,dim,bias=False)
        self.norm1=nn.LayerNorm(dim); self.norm2=nn.LayerNorm(dim)

    def forward(self, x1, x2):
        B,D = x1.shape; H,d = self.heads, D//self.heads
        # x1 → x2
        q1=self.q1(x1).reshape(B,H,d); k2=self.k2(x2).reshape(B,H,d)
        v2=self.v2(x2).reshape(B,H,d)
        a1=torch.sigmoid((q1*k2*self.scale).sum(-1,keepdim=True))
        x1_r=self.norm1(x1+self.out1((a1*v2).reshape(B,D)))
        # x2 → x1
        q2=self.q2(x2).reshape(B,H,d); k1=self.k1(x1).reshape(B,H,d)
        v1=self.v1(x1).reshape(B,H,d)
        a2=torch.sigmoid((q2*k1*self.scale).sum(-1,keepdim=True))
        x2_r=self.norm2(x2+self.out2((a2*v1).reshape(B,D)))
        return x1_r, x2_r

class WinnerSwinHybrid(nn.Module):
    def __init__(self, winner_class, num_classes=8, drop=0.4):
        super().__init__()
        self.winner_name = winner_class.__name__

        if winner_class==EfficientNetV2Model:
            self.backbone_a=timm.create_model(
                'tf_efficientnetv2_s',pretrained=True,num_classes=0,global_pool='')
            A_DIM=1280; self.attn_a=ECA(A_DIM)
        elif winner_class==ResNet50Model:
            base=models.resnet50(pretrained=True)
            self.backbone_a=nn.Sequential(
                base.conv1,base.bn1,base.relu,base.maxpool,
                base.layer1,base.layer2,base.layer3,base.layer4)
            A_DIM=2048; self.attn_a=CBAM(A_DIM)
        elif winner_class==VGG19Model:
            self.backbone_a=models.vgg19_bn(pretrained=True).features
            A_DIM=512; self.attn_a=CBAM(A_DIM)

        self.gem_a  = GeMPooling(p=3.0)
        self.proj_a = nn.Sequential(
            nn.Linear(A_DIM,512),nn.LayerNorm(512),nn.GELU(),nn.Dropout(drop*0.5))

        self.swin      = timm.create_model(
            'swin_tiny_patch4_window7_224',pretrained=True,num_classes=0)
        self.swin_norm = nn.LayerNorm(768)
        self.proj_b    = nn.Sequential(
            nn.Linear(768,512),nn.LayerNorm(512),nn.GELU(),nn.Dropout(drop*0.5))

        self.cross_attn = CrossAttentionGate(dim=512,heads=8)

        self.fusion_head = nn.Sequential(
            nn.Linear(1024,512),nn.LayerNorm(512),nn.GELU(),nn.Dropout(drop),
            nn.Linear(512,256), nn.GELU(),nn.Dropout(drop/2),
            nn.Linear(256,128), nn.GELU(),
            nn.Linear(128,num_classes))

    def forward_a(self, x):
        f=self.backbone_a(x); f=self.attn_a(f); f=self.gem_a(f)
        return self.proj_a(f)

    def forward_b(self, x):
        f=_swin_pool(self.swin.forward_features(x))
        return self.proj_b(self.swin_norm(f))

    def forward(self, x):
        fa=self.forward_a(x); fb=self.forward_b(x)
        fa,fb=self.cross_attn(fa,fb)
        return self.fusion_head(torch.cat([fa,fb],dim=1))

model = WinnerSwinHybrid(WINNER_CLASS, NUM_CLASSES, drop=0.4).to(DEVICE)
print(f"\n✅ Hybrid: {winner_name} + Swin-T + CrossAttn | "
      f"Params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")


# ============================================================
# STEP 12 — Hybrid Training
# ============================================================
# Warm-start winner backbone
winner_state = torch.load(WINNER_SAVE, map_location=DEVICE)
bb_state = {k:v for k,v in winner_state.items()
            if 'head' not in k and 'classifier' not in k and 'gem' not in k}
missing,unexpected = model.load_state_dict(bb_state, strict=False)
print(f"   Warm-start: {len(bb_state)} keys | "
      f"missing:{len(missing)} unexpected:{len(unexpected)}")

bb_a_ids = {id(p) for p in model.backbone_a.parameters()}
swin_ids  = {id(p) for p in model.swin.parameters()}

opt = optim.AdamW([
    {'params':[p for p in model.backbone_a.parameters()], 'lr':LR*0.05},
    {'params':[p for p in model.swin.parameters()],       'lr':LR*0.10},
    {'params':[p for p in model.parameters()
               if id(p) not in bb_a_ids|swin_ids],        'lr':LR},
], weight_decay=WEIGHT_DECAY)

sch = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt,T_0=10,T_mult=2)
es  = EarlyStopping(patience=PATIENCE, save_path=SAVE_HYBRID)
train_losses,val_losses,val_accs,lr_hist = [],[],[],[]

print(f"\n🔥 Hybrid Training ({HYBRID_EPOCHS} epochs)")
print("─"*78)

for ep in range(1, HYBRID_EPOCHS+1):
    t0  = time.time()
    trl = train_one_epoch(model, train_loader, criterion, opt)
    vll,vla,_,_,_ = evaluate(model, val_loader, criterion)
    sch.step()
    train_losses.append(trl); val_losses.append(vll)
    val_accs.append(vla); lr_hist.append(opt.param_groups[2]['lr'])
    stop = es.step(vll, ep, model)
    flag = "🏅 BEST" if es.counter==0 else f"(pat {es.counter}/{PATIENCE})"
    print(f"Ep{ep:02d}/{HYBRID_EPOCHS} | {time.time()-t0:.0f}s | "
          f"Tr:{trl:.4f} Vl:{vll:.4f} Acc:{vla*100:.2f}% "
          f"LR:{lr_hist[-1]:.2e} {flag}")
    if stop:
        print(f"\n⏹ Early stop ep{ep} | best ep{es.best_epoch}"); break

model.load_state_dict(torch.load(SAVE_HYBRID, map_location=DEVICE))
print(f"\n✅ Best hybrid loaded (ep{es.best_epoch})")


# ============================================================
# STEP 13 — Fig 3: Hybrid Training Curves (IEEE)
# ============================================================
ep_r = range(1, len(train_losses)+1)
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].plot(ep_r, train_losses, color=PAL['ours'], lw=2, label='Train')
axes[0].plot(ep_r, val_losses,   color=PAL['neg'],  lw=2, label='Val', linestyle='--')
axes[0].axvline(es.best_epoch, color=PAL['pos'], lw=1.5,
                linestyle=':', label=f'Best (ep{es.best_epoch})')
axes[0].set_title("(a) Loss Curves", fontweight='bold')
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
axes[0].legend(); axes[0].grid(alpha=0.3, linestyle='--')

axes[1].plot(ep_r, [a*100 for a in val_accs], color=PAL['ours'], lw=2)
axes[1].axvline(es.best_epoch, color=PAL['pos'], lw=1.5, linestyle=':')
axes[1].set_title("(b) Validation Accuracy (%)", fontweight='bold')
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy (%)")
axes[1].grid(alpha=0.3, linestyle='--')

plt.suptitle(f"Fig. 3 — Hybrid ({winner_name} + Swin-T) Training",
             fontsize=11, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig("fig3_hybrid_training.pdf", bbox_inches='tight')
plt.savefig("fig3_hybrid_training.png")
plt.show(); print("✅ Fig 3 saved")


# ============================================================
# STEP 14 — Test Evaluation
# ============================================================
_,_,test_probs,test_preds,test_true = evaluate(model, test_loader, criterion)
test_probs  = np.array(test_probs)
m_hybrid    = full_metrics(test_true, test_preds, test_probs)

print("\n" + "═"*65)
print(f"🏆  FINAL HYBRID TEST  ({winner_name} + Swin-T + CrossAttn)")
print("═"*65)
for label,val in [("Accuracy",       m_hybrid['acc']*100),
                   ("Macro Precision", m_hybrid['prec']*100),
                   ("Macro Recall",    m_hybrid['rec']*100),
                   ("Macro F1",        m_hybrid['mf1']*100),
                   ("Weighted F1",     m_hybrid['wf1']*100)]:
    print(f"  {label:<22}: {val:.4f}%")
print(f"  {'Macro AUC-ROC':<22}: {m_hybrid['auc']:.4f}")
print("═"*65)
print(classification_report(test_true, test_preds,
      target_names=[IDX_TO_CLASS[i] for i in range(NUM_CLASSES)]))


# ============================================================
# STEP 15 — Fig 4: Confusion Matrix (IEEE)
# ============================================================
cls_short = [IDX_TO_CLASS[i].replace('_',' ') for i in range(NUM_CLASSES)]
cm        = confusion_matrix(test_true, test_preds, labels=list(range(NUM_CLASSES)))
cm_pct    = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1) * 100

fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(cm_pct, annot=True, fmt='.1f', cmap='Blues', ax=ax,
            xticklabels=cls_short, yticklabels=cls_short,
            linewidths=0.4, linecolor='lightgrey',
            cbar_kws={'label':'Row-normalised (%)', 'shrink':0.85},
            annot_kws={'size':9})
ax.set_title(f"Fig. 4 — Confusion Matrix ({winner_name} + Swin-T)",
             fontweight='bold', pad=12)
ax.set_ylabel("True Label"); ax.set_xlabel("Predicted Label")
ax.tick_params(axis='x', rotation=40, labelsize=9)
ax.tick_params(axis='y', rotation=0,  labelsize=9)
plt.tight_layout()
plt.savefig("fig4_confusion_matrix.pdf", bbox_inches='tight')
plt.savefig("fig4_confusion_matrix.png")
plt.show(); print("✅ Fig 4 saved")


# ============================================================
# STEP 16 — Fig 5: Per-Class F1 + ROC (IEEE two-column)
# ============================================================
per_f1  = f1_score(test_true, test_preds, average=None,
                   zero_division=0, labels=list(range(NUM_CLASSES)))
y_bin   = label_binarize(test_true, classes=list(range(NUM_CLASSES)))
roc_pal = plt.cm.tab10(np.linspace(0,1,NUM_CLASSES))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Per-class F1
bar_cols = [PAL['neg'] if f<0.90 else PAL['pos'] for f in per_f1]
bars = axes[0].barh(cls_short, per_f1*100, color=bar_cols,
                    edgecolor='white', linewidth=0.6)
axes[0].axvline(90, color='navy', lw=1.2, linestyle='--', label='90% threshold')
axes[0].set_xlim([0,105])
axes[0].set_xlabel("F1-Score (%)"); axes[0].set_title("(a) Per-Class F1", fontweight='bold')
axes[0].legend(fontsize=8); axes[0].grid(axis='x', alpha=0.3, linestyle='--')
for bar,val in zip(bars, per_f1):
    axes[0].text(val*100+0.5, bar.get_y()+bar.get_height()/2,
                 f"{val*100:.1f}", va='center', fontsize=8)

# ROC curves
for i in range(NUM_CLASSES):
    if y_bin[:,i].sum()==0: continue
    fpr,tpr,_ = roc_curve(y_bin[:,i], test_probs[:,i])
    ai = roc_auc_score(y_bin[:,i], test_probs[:,i])
    axes[1].plot(fpr, tpr, color=roc_pal[i], lw=1.6,
                 label=f"{cls_short[i]} ({ai:.3f})")
axes[1].plot([0,1],[0,1],'k--',lw=1,alpha=0.5)
axes[1].set_xlabel("False Positive Rate"); axes[1].set_ylabel("True Positive Rate")
axes[1].set_title("(b) ROC Curves (One-vs-Rest)", fontweight='bold')
axes[1].legend(fontsize=7, loc='lower right')
axes[1].grid(alpha=0.3, linestyle='--')

plt.suptitle("Fig. 5 — Per-Class F1 and ROC Analysis",
             fontsize=11, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig("fig5_f1_roc.pdf", bbox_inches='tight')
plt.savefig("fig5_f1_roc.png")
plt.show(); print("✅ Fig 5 saved")


# ============================================================
# STEP 17 — Fig 6: XAI — Grad-CAM++ + EigenCAM (IEEE)
# ============================================================
try:
    from pytorch_grad_cam import GradCAMPlusPlus, EigenCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image

    model.eval()

    # ── Target layer selection ────────────────────────────────
    if WINNER_CLASS == EfficientNetV2Model:
        tgt_a = [model.backbone_a.blocks[-1]]
    elif WINNER_CLASS == ResNet50Model:
        tgt_a = [list(model.backbone_a.children())[-1][-1]]
    else:   # VGG19
        tgt_a = [list(model.backbone_a.children())[-2]]

    # Swin EigenCAM needs (B,C,H,W) output
    class SwinWrapper(nn.Module):
        def __init__(self, m):
            super().__init__(); self.m = m
        def forward(self, x):
            f = self.m.swin.forward_features(x)   # (B,H,W,C) or (B,C)
            if f.dim()==4:
                return f.permute(0,3,1,2).contiguous()
            return f.unsqueeze(-1).unsqueeze(-1)

    swin_wrap = SwinWrapper(model)
    tgt_b     = [model.swin.layers[-1].blocks[-1]]

    cam_winner = GradCAMPlusPlus(model=model,      target_layers=tgt_a)
    cam_swin   = EigenCAM(       model=swin_wrap,  target_layers=tgt_b)

    # One sample per class
    sample_idxs = []
    for ci in range(NUM_CLASSES):
        cands = [i for i,l in enumerate(test_labels) if l==ci]
        if cands: sample_idxs.append(int(np.random.choice(cands)))

    n_show = min(8, len(sample_idxs))
    fig, axes = plt.subplots(n_show, 3, figsize=(12, 3.5*n_show))
    if n_show==1: axes=[axes]

    col_titles = ["Original Image",
                  f"{winner_name[:16]}\n(Grad-CAM++)",
                  "Swin-T\n(EigenCAM)"]
    for ax,title in zip(axes[0], col_titles):
        ax.set_title(title, fontsize=9, fontweight='bold')

    for row,idx in enumerate(sample_idxs[:n_show]):
        raw = Image.open(test_paths[idx]).convert("RGB").resize((IMG_SIZE,IMG_SIZE))
        rgb = np.array(raw)/255.0
        inp = val_tf(raw).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = model(inp)
            pred   = logits.argmax(-1).item()
            conf   = F.softmax(logits,-1)[0,pred].item()

        try:
            gc_w = cam_winner(input_tensor=inp)[0]
            gc_s = cam_swin(  input_tensor=inp)[0]
        except Exception as ex:
            print(f"  CAM error row {row}: {ex}")
            gc_w = gc_s = np.zeros((IMG_SIZE,IMG_SIZE))

        correct = pred==test_labels[idx]
        lbl_col = PAL['pos'] if correct else PAL['neg']
        tick    = '✓' if correct else '✗'

        overlays = [
            raw,
            Image.fromarray(show_cam_on_image(
                rgb.astype(np.float32), gc_w, use_rgb=True)),
            Image.fromarray(show_cam_on_image(
                rgb.astype(np.float32), gc_s, use_rgb=True)),
        ]
        for col,im in enumerate(overlays):
            axes[row][col].imshow(im)
            axes[row][col].axis('off')

        axes[row][0].set_ylabel(
            f"GT: {IDX_TO_CLASS[test_labels[idx]]}\n"
            f"Pred: {IDX_TO_CLASS[pred]} {tick} {conf*100:.1f}%",
            fontsize=8, color=lbl_col, fontweight='bold')

    plt.suptitle(
        f"Fig. 6 — XAI: Local ({winner_name[:14]}) vs. Global (Swin-T) Attention\n"
        "Warmer colours indicate higher contribution to prediction.",
        fontsize=10, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig("fig6_xai_gradcam.pdf", bbox_inches='tight')
    plt.savefig("fig6_xai_gradcam.png")
    plt.show(); print("✅ Fig 6 — XAI saved")

except Exception as e:
    print(f"⚠️  XAI skipped: {e}")


# ============================================================
# STEP 18 — Fig 7: Branch Contribution (IEEE)
# ============================================================
try:
    model.eval()
    eff_norms  = {i:[] for i in range(NUM_CLASSES)}
    swin_norms = {i:[] for i in range(NUM_CLASSES)}

    with torch.no_grad():
        for imgs,lbls in test_loader:
            imgs = imgs.to(DEVICE)
            fa   = model.forward_a(imgs)
            fb   = model.forward_b(imgs)
            for i,l in enumerate(lbls.numpy()):
                eff_norms[l].append(fa[i].norm().item())
                swin_norms[l].append(fb[i].norm().item())

    avg_a = [np.mean(eff_norms[i])  if eff_norms[i]  else 0 for i in range(NUM_CLASSES)]
    avg_b = [np.mean(swin_norms[i]) if swin_norms[i] else 0 for i in range(NUM_CLASSES)]

    x, w = np.arange(NUM_CLASSES), 0.35
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x-w/2, avg_a, w, label=f'{winner_name[:18]} (Local Texture)',
           color=PAL['vgg'], edgecolor='white')
    ax.bar(x+w/2, avg_b, w, label='Swin-T (Global Structure)',
           color=PAL['swin'], edgecolor='white')
    ax.set_xticks(x)
    ax.set_xticklabels([IDX_TO_CLASS[i].replace('_',' ')
                        for i in range(NUM_CLASSES)],
                       rotation=30, ha='right', fontsize=9)
    ax.set_ylabel("Mean L2 Norm of Branch Feature Vector")
    ax.set_title("Fig. 7 — Branch Feature Contribution per Disease Class",
                 fontweight='bold')
    ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig("fig7_branch_contribution.pdf", bbox_inches='tight')
    plt.savefig("fig7_branch_contribution.png")
    plt.show(); print("✅ Fig 7 saved")

except Exception as e:
    print(f"⚠️  Branch analysis skipped: {e}")


# ============================================================
# STEP 19 — Fig 8: Robustness (IEEE)
# ============================================================
class NoisyDS(StrawberryDataset):
    def __init__(self, paths, labels, tf, sigma):
        super().__init__(paths, labels, tf, False)
        self.sigma = sigma
    def __getitem__(self, idx):
        img,lbl = super().__getitem__(idx)
        if self.sigma>0: img = img+torch.randn_like(img)*self.sigma
        return img,lbl

print("\n🛡️  Robustness evaluation...")
rob_results = []
for sigma in [0.0, 0.05, 0.10, 0.20, 0.30]:
    nl = DataLoader(NoisyDS(test_paths,test_labels,val_tf,sigma),
                    batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    _,acc_n,_,preds_n,true_n = evaluate(model, nl, criterion)
    f1_n = f1_score(true_n,preds_n,average='macro',zero_division=0)
    rob_results.append({'sigma':sigma,'acc':acc_n*100,'f1':f1_n*100})
    print(f"  σ={sigma:.2f} → Acc:{acc_n*100:.2f}%  F1:{f1_n*100:.2f}%")

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
sigs = [r['sigma'] for r in rob_results]
axes[0].plot(sigs,[r['acc'] for r in rob_results],
             color=PAL['ours'],marker='o',ms=7,lw=2)
axes[0].set_title("(a) Accuracy vs. Noise σ", fontweight='bold')
axes[0].set_xlabel("Gaussian σ"); axes[0].set_ylabel("Accuracy (%)")
axes[0].grid(alpha=0.3,linestyle='--')

axes[1].plot(sigs,[r['f1'] for r in rob_results],
             color=PAL['neg'],marker='s',ms=7,lw=2)
axes[1].set_title("(b) Macro F1 vs. Noise σ", fontweight='bold')
axes[1].set_xlabel("Gaussian σ"); axes[1].set_ylabel("Macro F1 (%)")
axes[1].grid(alpha=0.3,linestyle='--')

plt.suptitle("Fig. 8 — Model Robustness Under Gaussian Noise",
             fontsize=11, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig("fig8_robustness.pdf", bbox_inches='tight')
plt.savefig("fig8_robustness.png")
plt.show(); print("✅ Fig 8 saved")


# ============================================================
# STEP 20 — Fig 9: Ablation Study (IEEE)
# ============================================================
print("\n🔬  Ablation study...")

class BranchOnlyA(nn.Module):
    def __init__(self, m, nc):
        super().__init__()
        self.backbone_a=m.backbone_a; self.attn_a=m.attn_a; self.gem_a=m.gem_a
        dim = (1280 if WINNER_CLASS==EfficientNetV2Model
               else 2048 if WINNER_CLASS==ResNet50Model else 512)
        self.head=nn.Linear(dim, nc)
    def forward(self, x):
        f=self.backbone_a(x); f=self.attn_a(f); f=self.gem_a(f)
        return self.head(f)

class BranchOnlyB(nn.Module):
    def __init__(self, m, nc):
        super().__init__()
        self.swin=m.swin; self.norm=m.swin_norm
        self.head=nn.Linear(768, nc)
    def forward(self, x):
        return self.head(self.norm(_swin_pool(self.swin.forward_features(x))))

ablation_results = {}
for abl_name, abl_model in [
    (f"{winner_name[:18]} Only", BranchOnlyA(model, NUM_CLASSES)),
    ("Swin-T Only",              BranchOnlyB(model, NUM_CLASSES)),
]:
    abl_model = abl_model.to(DEVICE)
    for p in abl_model.parameters(): p.requires_grad=False
    for p in abl_model.head.parameters(): p.requires_grad=True
    opt_a = optim.AdamW(abl_model.head.parameters(), lr=1e-3)
    for _ in range(5):
        abl_model.train()
        for imgs,lbls in train_loader:
            imgs,lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            opt_a.zero_grad()
            criterion(abl_model(imgs), lbls).backward()
            opt_a.step()
    _,acc_a,_,p_a,g_a = evaluate(abl_model, test_loader, criterion)
    f1_a = f1_score(g_a,p_a,average='macro',zero_division=0)
    ablation_results[abl_name] = (acc_a*100, f1_a*100)
    del abl_model; torch.cuda.empty_cache()

ablation_results['Full Hybrid (Ours)'] = (m_hybrid['acc']*100, m_hybrid['mf1']*100)

# Print table
print(f"\n  {'Variant':<38} {'Accuracy':>10} {'Macro F1':>10}")
print("  "+"─"*60)
for name,(a,f) in ablation_results.items():
    mk = " ★" if 'Ours' in name else ""
    print(f"  {name:<38} {a:>9.2f}%  {f:>9.2f}%{mk}")

# Bar chart
abl_names = list(ablation_results.keys())
abl_f1    = [ablation_results[n][1] for n in abl_names]
abl_cols  = [PAL['vgg'], PAL['swin'], PAL['ours']]

fig, ax = plt.subplots(figsize=(8, 4))
bars = ax.bar(abl_names, abl_f1, color=abl_cols,
              edgecolor='white', linewidth=0.8, width=0.5)
ax.set_ylabel("Macro F1 (%)")
ax.set_title("Fig. 9 — Ablation Study: Branch Contribution to Final F1",
             fontweight='bold')
ax.set_ylim([max(0,min(abl_f1)-10), min(100,max(abl_f1)+5)])
ax.set_xticklabels(abl_names, rotation=15, ha='right', fontsize=9)
ax.grid(axis='y', alpha=0.3, linestyle='--')
for bar,val in zip(bars,abl_f1):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
            f"{val:.2f}%", ha='center', va='bottom',
            fontsize=9, fontweight='bold')
plt.tight_layout()
plt.savefig("fig9_ablation.pdf", bbox_inches='tight')
plt.savefig("fig9_ablation.png")
plt.show(); print("✅ Fig 9 saved")


# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "═"*65)
print("📋  ALL FIGURES SAVED (PDF + PNG, IEEE-ready)")
print("═"*65)
figs = [
    ("Fig 1", "fig1_benchmark_curves",      "Benchmark learning curves"),
    ("Fig 2", "fig2_benchmark_bars",         "Benchmark metric comparison"),
    ("Fig 3", "fig3_hybrid_training",        "Hybrid training curves"),
    ("Fig 4", "fig4_confusion_matrix",       "Confusion matrix"),
    ("Fig 5", "fig5_f1_roc",                 "Per-class F1 + ROC"),
    ("Fig 6", "fig6_xai_gradcam",            "XAI: Grad-CAM++ + EigenCAM"),
    ("Fig 7", "fig7_branch_contribution",    "Branch feature contribution"),
    ("Fig 8", "fig8_robustness",             "Robustness under noise"),
    ("Fig 9", "fig9_ablation",               "Ablation study"),
]
for fig_id, fname, desc in figs:
    pdf_ok = "✓" if os.path.exists(fname+".pdf") else "✗"
    png_ok = "✓" if os.path.exists(fname+".png") else "✗"
    print(f"  {fig_id}  PDF:{pdf_ok} PNG:{png_ok}  {desc}")
print("═"*65)