# ============================================================
# FER-2013 — 3 Advanced Hybrid Architectures
# ConvNeXtV2-SwinV2-CBAM | DeiT-ConvNeXt | EVA02-MobileNetV3
# ============================================================

import os, sys, time, copy, warnings
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, confusion_matrix, roc_curve,
                             auc, classification_report)
from sklearn.preprocessing import label_binarize
from sklearn.utils.class_weight import compute_class_weight
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
warnings.filterwarnings('ignore')

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {DEVICE}")
if DEVICE.type == 'cuda':
    print(f"GPU    : {torch.cuda.get_device_name(0)}")

# ── Config ───────────────────────────────────────────────────
DATA_ROOT   = "/kaggle/input/datasets/msambare/fer2013"
TRAIN_DIR   = os.path.join(DATA_ROOT, "train")
TEST_DIR    = os.path.join(DATA_ROOT, "test")
CLASSES     = ['angry','disgust','fear','happy','neutral','sad','surprise']
NUM_CLS     = len(CLASSES)
IMG_SIZE    = 224
BATCH       = 32
EPOCHS      = 40
LR          = 3e-4
PATIENCE    = 5
TOTAL_IMGS  = 15000
SEED        = 42
NUM_WORKERS = 2

torch.manual_seed(SEED)
np.random.seed(SEED)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
MODEL_COLORS  = ['#e74c3c', '#2980b9', '#27ae60']

print(f"Classes : {CLASSES}")
print(f"Total   : {TOTAL_IMGS} | Split: 60/20/20")
print(f"Batch   : {BATCH} | Epochs: {EPOCHS} | LR: {LR} | Patience: {PATIENCE}")

# ============================================================
# Transforms
# ============================================================
def to_rgb(img):
    return img.convert("L").convert("RGB")

train_tf = transforms.Compose([
    transforms.Lambda(to_rgb),
    transforms.Resize((IMG_SIZE + 16, IMG_SIZE + 16)),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.RandomAffine(degrees=0,
                            translate=(0.10, 0.10),
                            scale=(0.90, 1.10)),
    transforms.ColorJitter(brightness=0.30,
                           contrast=0.30,
                           saturation=0.20,
                           hue=0.05),
    transforms.RandomGrayscale(p=0.10),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),
])

val_tf = transforms.Compose([
    transforms.Lambda(to_rgb),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# ============================================================
# Dataset
# ============================================================
class FERDataset(Dataset):
    def __init__(self, samples, tf):
        self.samples = samples
        self.tf      = tf

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        return self.tf(Image.open(path)), label

all_samples = []
for root in [TRAIN_DIR, TEST_DIR]:
    for i, cls in enumerate(CLASSES):
        d = os.path.join(root, cls)
        if not os.path.isdir(d):
            continue
        for fn in os.listdir(d):
            if fn.lower().endswith(('.jpg', '.png')):
                all_samples.append((os.path.join(d, fn), i))

print(f"\nTotal available: {len(all_samples)}")

all_labels_pool = [s[1] for s in all_samples]
per_class = TOTAL_IMGS // NUM_CLS
selected  = []
for c in range(NUM_CLS):
    c_idx = [i for i, l in enumerate(all_labels_pool) if l == c]
    np.random.shuffle(c_idx)
    selected.extend(c_idx[:per_class])
np.random.shuffle(selected)
selected         = selected[:TOTAL_IMGS]
selected_samples = [all_samples[i] for i in selected]

n_train = int(TOTAL_IMGS * 0.60)
n_val   = int(TOTAL_IMGS * 0.20)
n_test  = TOTAL_IMGS - n_train - n_val

train_samples = selected_samples[:n_train]
val_samples   = selected_samples[n_train:n_train + n_val]
test_samples  = selected_samples[n_train + n_val:]

train_ds = FERDataset(train_samples, train_tf)
val_ds   = FERDataset(val_samples,   val_tf)
test_ds  = FERDataset(test_samples,  val_tf)

train_loader = DataLoader(train_ds, BATCH, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True)
val_loader   = DataLoader(val_ds,   BATCH, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)
test_loader  = DataLoader(test_ds,  BATCH, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)

print(f"Train : {len(train_ds)} | Val : {len(val_ds)} | Test : {len(test_ds)}")

train_labels_arr = [s[1] for s in train_samples]
cw = compute_class_weight('balanced',
                           classes=np.arange(NUM_CLS),
                           y=train_labels_arr)
class_weights = torch.tensor(cw, dtype=torch.float32).to(DEVICE)
print(f"\nClass weights: {dict(zip(CLASSES, cw.round(3)))}")

# ============================================================
# Loss
# ============================================================
class FocalLabelSmoothLoss(nn.Module):
    def __init__(self, num_classes=7, smoothing=0.10,
                 gamma=2.0, weight=None):
        super().__init__()
        self.smoothing    = smoothing
        self.gamma        = gamma
        self.num_classes  = num_classes
        self.weight       = weight

    def forward(self, logits, targets):
        with torch.no_grad():
            smooth_targets = torch.full_like(
                logits, self.smoothing / (self.num_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1),
                                    1.0 - self.smoothing)

        log_prob = F.log_softmax(logits, dim=1)
        prob     = log_prob.exp()
        p_t      = (prob * F.one_hot(targets, self.num_classes).float()).sum(1)
        focal_w  = (1 - p_t) ** self.gamma
        loss     = -(smooth_targets * log_prob).sum(1)

        if self.weight is not None:
            cw   = self.weight[targets]
            loss = loss * cw

        loss = (focal_w * loss).mean()
        return loss

# ============================================================
# Scheduler
# ============================================================
class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs,
                 base_lr, min_lr=1e-6):
        self.optimizer      = optimizer
        self.warmup_epochs  = warmup_epochs
        self.total_epochs   = total_epochs
        self.base_lr        = base_lr
        self.min_lr         = min_lr

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / \
                       (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * \
                 (1 + np.cos(np.pi * progress))
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr
        return lr

# ============================================================
# ============================================================
# MODEL 1 - ConvNeXtV2-Nano + SwinV2-Tiny + CBAM Hybrid
# ============================================================
class CBAM_Hybrid1(nn.Module):
    def __init__(self, channels, reduction=16, spatial_kernel=7):
        super().__init__()
        self.ch_avg = nn.AdaptiveAvgPool2d(1)
        self.ch_max = nn.AdaptiveMaxPool2d(1)
        self.ch_mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels)
        )
        self.sp_conv = nn.Conv2d(2, 1, spatial_kernel,
                                 padding=spatial_kernel // 2, bias=False)

    def forward(self, x):
        avg_c = self.ch_mlp(self.ch_avg(x))
        max_c = self.ch_mlp(self.ch_max(x))
        ch_w  = torch.sigmoid(avg_c + max_c).unsqueeze(-1).unsqueeze(-1)
        x     = x * ch_w
        avg_s = x.mean(1, keepdim=True)
        max_s = x.max(1, keepdim=True).values
        sp_w  = torch.sigmoid(self.sp_conv(torch.cat([avg_s, max_s], 1)))
        x     = x * sp_w
        return x

class ConvNeXtSwinCBAM(nn.Module):
    def __init__(self, num_classes=7, dim=512, dropout=0.3):
        super().__init__()
        # Stream 1: ConvNeXtV2-Nano for local fine-grained features
        self.convnext = timm.create_model(
            'convnextv2_nano.fcmae_ft_in22k_in1k_384', pretrained=True,
            num_classes=0, global_pool='')
        self.cbam = CBAM_Hybrid1(640, reduction=16)
        self.cnn_proj = nn.Sequential(
            nn.Conv2d(640, dim, 1, bias=False),
            nn.BatchNorm2d(dim),
            nn.GELU(),
            nn.Dropout2d(dropout * 0.3)
        )
        # Freeze early convnext, fine-tune last stages
        for p in self.convnext.parameters():
            p.requires_grad = False
        for name, p in self.convnext.named_parameters():
            if any(k in name for k in ['stages.3', 'stages.2']):
                p.requires_grad = True

        # Stream 2: SwinV2-Tiny for global context
        swin = timm.create_model(
            'swinv2_cr_tiny_ns_224.sw_in1k', pretrained=True,
            num_classes=0)
        self.swin = swin
        self.swin_proj = nn.Sequential(
            nn.Linear(768, dim),
            nn.LayerNorm(dim),
            nn.GELU()
        )
        # Freeze early swin, fine-tune last layers
        for p in self.swin.parameters():
            p.requires_grad = False
        for name, p in self.swin.named_parameters():
            if any(k in name for k in ['layers.2', 'layers.3', 'norm']):
                p.requires_grad = True

        # Cross-attention fusion
        self.cross_attn = nn.MultiheadAttention(
            dim, num_heads=8, dropout=dropout, batch_first=True)
        self.cross_norm = nn.LayerNorm(dim)

        # Gated fusion
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Tanh(),
            nn.Linear(dim, 2),
            nn.Softmax(dim=-1)
        )

        self.head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Dropout(dropout),
            nn.Linear(dim, 512),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        B = x.shape[0]

        # Stream 1: ConvNeXt + CBAM
        cnn_f = self.convnext(x)          # (B, 640, 7, 7)
        cnn_f = self.cbam(cnn_f)          # attention-weighted
        cnn_f = self.cnn_proj(cnn_f)      # (B, dim, 7, 7)
        cnn_tok = cnn_f.flatten(2).transpose(1, 2)  # (B, 49, dim)
        cnn_g = cnn_tok.mean(1)           # (B, dim)

        # Stream 2: SwinV2
        swin_out = self.swin(x)  # (B, 768) — num_classes=0 returns pooled
        swin_g = self.swin_proj(swin_out)         # (B, dim)

        # Cross-attention: CNN tokens attend to Swin global
        swin_q = swin_g.unsqueeze(1)              # (B, 1, dim)
        xatt, _ = self.cross_attn(swin_q, cnn_tok, cnn_tok)
        xatt = self.cross_norm(swin_q + xatt).squeeze(1)  # (B, dim)

        # Gated fusion
        combined = torch.cat([cnn_g, xatt], dim=-1)
        gate_w = self.gate(combined)
        fused = gate_w[:, 0:1] * cnn_g + gate_w[:, 1:2] * xatt

        return self.head(fused)

    def get_param_groups(self, base_lr):
        return [
            {'params': self.convnext.parameters(), 'lr': base_lr * 0.1},
            {'params': self.swin.parameters(),      'lr': base_lr * 0.1},
            {'params': list(self.cbam.parameters()) +
                       list(self.cnn_proj.parameters()) +
                       list(self.swin_proj.parameters()) +
                       list(self.cross_attn.parameters()) +
                       list(self.cross_norm.parameters()) +
                       list(self.gate.parameters()) +
                       list(self.head.parameters()),
             'lr': base_lr * 1.0},
        ]

class ConvNeXtFeatureExtractor(nn.Module):
    """
    ConvNeXt-Small multi-scale feature extractor.
    Returns 3 feature maps from stages [1, 2, 3]:
      channels = [192, 384, 768] at spatial resolutions 28×28, 14×14, 7×7.
    Stage 0 (stem + stage 0) is frozen; stages 1-3 are fine-tuned.
    """
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            'convnext_small', pretrained=True,
            features_only=True, out_indices=[1, 2, 3])
        # Freeze stem and stage-0 (low-level edges / textures)
        for name, p in self.backbone.named_parameters():
            p.requires_grad = ('stages.0' not in name and
                               'stem'     not in name)

    def forward(self, x):
        # Returns: [(B,192,28,28), (B,384,14,14), (B,768,7,7)]
        return self.backbone(x)


class MultiScaleCNNFusion(nn.Module):
    """
    Fuses multi-scale ConvNeXt feature maps into a single global vector.

    For each scale:  AdaptiveAvgPool2d(1) → Flatten → Linear(c, out_dim)
                     → LayerNorm → GELU → Dropout

    Final output:  learned weighted sum across scales → (B, out_dim)
    """
    def __init__(self, in_channels_list, out_dim, dropout=0.3):
        super().__init__()
        self.projectors = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(c, out_dim),
                nn.LayerNorm(out_dim),
                nn.GELU(),
                nn.Dropout(dropout * 0.5)
            )
            for c in in_channels_list
        ])
        # Learnable per-scale importance weights
        self.scale_weights = nn.Parameter(
            torch.ones(len(in_channels_list)) / len(in_channels_list))

    def forward(self, feature_list):
        projs = [proj(f) for proj, f in
                 zip(self.projectors, feature_list)]
        w     = torch.softmax(self.scale_weights, dim=0)
        return sum(w[i] * projs[i] for i in range(len(projs)))  # (B, out_dim)

# ============================================================
# MODEL 2 — DeiT-Base + ConvNeXt-Small Hybrid
# ============================================================
class DeiTConvNeXtHybrid(nn.Module):
    def __init__(self, num_classes=7, dim=768, dropout=0.3):
        super().__init__()
        self.convnext  = ConvNeXtFeatureExtractor()
        self.ms_fusion = MultiScaleCNNFusion(
            [192, 384, 768], dim, dropout=dropout)

        deit = timm.create_model(
            'deit_base_distilled_patch16_224', pretrained=True)

        self.deit_patch_embed = deit.patch_embed
        self.deit_cls_token   = deit.cls_token

        if hasattr(deit, 'dist_token') and deit.dist_token is not None:
            self.deit_dist_token = deit.dist_token
            self._has_dist       = True
        else:
            self.deit_dist_token = nn.Parameter(torch.zeros(1, 1, dim))
            nn.init.trunc_normal_(self.deit_dist_token, std=0.02)
            self._has_dist       = True

        self.deit_pos_embed = deit.pos_embed
        self.deit_pos_drop  = deit.pos_drop
        self.deit_blocks    = deit.blocks
        self.deit_norm      = deit.norm

        for i, blk in enumerate(self.deit_blocks):
            for p in blk.parameters():
                p.requires_grad = (i >= 9)

        self.cnn_to_deit = nn.Linear(dim, dim)

        self.xattn      = nn.MultiheadAttention(
            dim, num_heads=8, dropout=dropout, batch_first=True)
        self.xattn_norm = nn.LayerNorm(dim)

        self.fusion_mlp = nn.Sequential(
            nn.Linear(dim * 4, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.GELU(),
        )

        self.head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Dropout(dropout),
            nn.Linear(dim, 256),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        B = x.shape[0]

        cnn_feats  = self.convnext(x)
        cnn_global = self.ms_fusion(cnn_feats)
        cnn_q      = self.cnn_to_deit(cnn_global).unsqueeze(1)

        vit_x = self.deit_patch_embed(x)
        cls_t = self.deit_cls_token.expand(B, -1, -1)

        if isinstance(self.deit_dist_token, nn.Parameter):
            dst_t = self.deit_dist_token.expand(B, -1, -1)
        else:
            dst_t = self.deit_dist_token.expand(B, -1, -1)

        vit_x = torch.cat([cls_t, dst_t, vit_x], dim=1)
        vit_x = self.deit_pos_drop(vit_x + self.deit_pos_embed)

        for blk in self.deit_blocks:
            vit_x = blk(vit_x)
        vit_x = self.deit_norm(vit_x)

        deit_cls  = vit_x[:, 0]
        deit_dist = vit_x[:, 1]
        patch_tok = vit_x[:, 2:]

        xatt, _ = self.xattn(cnn_q, patch_tok, patch_tok)
        xatt    = self.xattn_norm(cnn_q + xatt).squeeze(1)

        combined = torch.cat(
            [deit_cls, deit_dist, cnn_global, xatt], dim=-1)
        fused    = self.fusion_mlp(combined)
        return self.head(fused)

    def get_param_groups(self, base_lr):
        return [
            {'params': self.convnext.parameters(),
             'lr': base_lr * 0.1},
            {'params': self.deit_blocks.parameters(),
             'lr': base_lr * 0.1},
            {'params': list(self.ms_fusion.parameters())    +
                       list(self.xattn.parameters())        +
                       list(self.xattn_norm.parameters())   +
                       list(self.cnn_to_deit.parameters())  +
                       list(self.fusion_mlp.parameters())   +
                       list(self.head.parameters()),
             'lr': base_lr * 1.0},
        ]

# ============================================================
# ============================================================
# MODEL 3 - EVA02-Tiny + MobileNetV3-Large + CoordAtt Hybrid
# ============================================================
class CoordAtt(nn.Module):
    def __init__(self, channels, reduction=32):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        mid = max(channels // reduction, 16)
        self.fc1 = nn.Linear(channels, mid)
        self.fc2 = nn.Linear(mid, channels)
        self.fc3 = nn.Linear(mid, channels)

    def forward(self, x):
        B, C, H, W = x.shape
        # Channel attention via squeeze-excitation
        y = self.pool(x).view(B, C)
        y = F.relu(self.fc1(y))
        ch_att = torch.sigmoid(self.fc2(y)).view(B, C, 1, 1)
        # Coordinate attention
        x_h = x.mean(dim=3, keepdim=True)   # (B, C, H, 1)
        x_w = x.mean(dim=2, keepdim=True)   # (B, C, 1, W)
        coord = F.relu(self.fc1(y))
        h_att = torch.sigmoid(self.fc3(coord)).view(B, C, 1, 1)
        return x * ch_att * h_att

class EVAMobileHybrid(nn.Module):
    def __init__(self, num_classes=7, dim=512, dropout=0.3):
        super().__init__()
        # Stream 1: EVA02-Tiny for global context
        self.eva = timm.create_model(
            'eva02_tiny_patch14_224.mim_in22k', pretrained=True,
            num_classes=0)
        self.eva_proj = nn.Sequential(
            nn.Linear(192, dim),
            nn.LayerNorm(dim),
            nn.GELU()
        )
        # Freeze early EVA blocks, fine-tune last 6
        for p in self.eva.parameters():
            p.requires_grad = False
        for i, blk in enumerate(self.eva.blocks):
            for p in blk.parameters():
                p.requires_grad = (i >= 6)

        # Stream 2: MobileNetV3-Large + CoordAtt for local features
        self.mobilenet = timm.create_model(
            'mobilenetv3_large_100.ra_in1k', pretrained=True,
            num_classes=0, global_pool='')
        self.coord_att = CoordAtt(960, reduction=32)
        self.mb_proj = nn.Sequential(
            nn.Conv2d(960, dim, 1, bias=False),
            nn.BatchNorm2d(dim),
            nn.GELU(),
            nn.Dropout2d(dropout * 0.3)
        )
        # Freeze early mobilenet, fine-tune last layers
        for p in self.mobilenet.parameters():
            p.requires_grad = False
        blocks = list(self.mobilenet.children())
        for layer in blocks[-4:]:
            for p in layer.parameters():
                p.requires_grad = True

        # Bidirectional cross-attention
        self.cnn_to_eva = nn.MultiheadAttention(
            dim, num_heads=8, dropout=dropout, batch_first=True)
        self.eva_to_cnn = nn.MultiheadAttention(
            dim, num_heads=8, dropout=dropout, batch_first=True)
        self.norm_c = nn.LayerNorm(dim)
        self.norm_e = nn.LayerNorm(dim)

        # Adaptive fusion with learnable temperature
        self.fusion_gate = nn.Sequential(
            nn.Linear(dim * 2, 4),
            nn.Softmax(dim=-1)
        )

        self.head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Dropout(dropout),
            nn.Linear(dim, 512),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        B = x.shape[0]

        # Stream 1: EVA02 global features
        eva_out = self.eva(x)              # (B, 384)
        eva_g = self.eva_proj(eva_out)     # (B, dim)

        # Stream 2: MobileNetV3 + CoordAtt local features
        mb_f = self.mobilenet(x)           # (B, 960, 7, 7)
        mb_f = self.coord_att(mb_f)
        mb_f = self.mb_proj(mb_f)          # (B, dim, 7, 7)
        mb_tok = mb_f.flatten(2).transpose(1, 2)  # (B, 49, dim)
        mb_g = mb_tok.mean(1)              # (B, dim)
        mb_mx = mb_tok.max(dim=1).values   # (B, dim)

        # Cross-attention: EVA queries CNN tokens
        eva_q = eva_g.unsqueeze(1)
        xatt_e, _ = self.cnn_to_eva(eva_q, mb_tok, mb_tok)
        xatt_e = self.norm_e(eva_q + xatt_e).squeeze(1)

        # Cross-attention: CNN queries EVA global
        mb_q = mb_g.unsqueeze(1)
        xatt_c, _ = self.eva_to_cnn(mb_q, eva_q, eva_q)
        xatt_c = self.norm_c(mb_q + xatt_c).squeeze(1)

        # Adaptive 4-way gated fusion
        gate = self.fusion_gate(torch.cat([xatt_e, xatt_c], dim=-1))
        fused = (gate[:, 0:1] * xatt_e +
                 gate[:, 1:2] * xatt_c +
                 gate[:, 2:3] * mb_mx +
                 gate[:, 3:4] * eva_g)

        return self.head(fused)

    def get_param_groups(self, base_lr):
        return [
            {'params': self.eva.parameters(),       'lr': base_lr * 0.1},
            {'params': self.mobilenet.parameters(), 'lr': base_lr * 0.1},
            {'params': list(self.coord_att.parameters()) +
                       list(self.eva_proj.parameters()) +
                       list(self.mb_proj.parameters()) +
                       list(self.cnn_to_eva.parameters()) +
                       list(self.eva_to_cnn.parameters()) +
                       list(self.norm_c.parameters()) +
                       list(self.norm_e.parameters()) +
                       list(self.fusion_gate.parameters()) +
                       list(self.head.parameters()),
             'lr': base_lr * 1.0},
        ]

# ============================================================
# Training Utilities
# ============================================================
class EarlyStopping:
    def __init__(self, patience=8, path='best.pth', mode='max'):
        self.patience   = patience
        self.path       = path
        self.mode       = mode
        self.best_score = -np.inf if mode == 'max' else np.inf
        self.counter    = 0
        self.best_ep    = 0
        self.best_state = None

    def step(self, score, epoch, model):
        improved = (self.mode == 'max' and score > self.best_score + 1e-5) or \
                   (self.mode == 'min' and score < self.best_score - 1e-5)
        if improved:
            self.best_score = score
            self.counter    = 0
            self.best_ep    = epoch
            self.best_state = copy.deepcopy(model.state_dict())
            torch.save(self.best_state, self.path)
            return False
        self.counter += 1
        return self.counter >= self.patience

def train_one_epoch(model, loader, criterion, optimizer, scaler):
    model.train()
    total_loss, correct, total = 0., 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            out  = model(imgs)
            loss = criterion(out, labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * imgs.size(0)
        correct    += (out.argmax(1) == labels).sum().item()
        total      += imgs.size(0)
    return total_loss / total, 100 * correct / total

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    total_loss, correct, total = 0., 0, 0
    all_preds, all_labels, all_probs = [], [], []
    crit = nn.CrossEntropyLoss()
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        out   = model(imgs)
        loss  = crit(out, labels)
        probs = F.softmax(out, dim=1)
        pred  = out.argmax(1)
        total_loss += loss.item() * imgs.size(0)
        correct    += (pred == labels).sum().item()
        total      += imgs.size(0)
        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
    acc  = 100 * correct / total
    loss = total_loss / total
    f1   = f1_score(all_labels, all_preds, average='macro') * 100
    return (loss, acc, f1,
            np.array(all_preds), np.array(all_labels), np.array(all_probs))

def train_model(model, name, epochs=EPOCHS, base_lr=LR, patience=PATIENCE):
    print(f"\n{'='*65}")
    print(f"  Training : {name}")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    t_params  = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {n_params:,}  /  Total: {t_params:,}")
    print(f"{'='*65}")

    model     = model.to(DEVICE)
    criterion = FocalLabelSmoothLoss(
        num_classes=NUM_CLS, smoothing=0.10,
        gamma=2.0, weight=class_weights)

    param_groups = model.get_param_groups(base_lr)
    optimizer    = optim.AdamW(param_groups, weight_decay=2e-4)
    scheduler    = WarmupCosineScheduler(
        optimizer, warmup_epochs=3, total_epochs=epochs,
        base_lr=base_lr, min_lr=1e-7)
    scaler   = torch.cuda.amp.GradScaler()
    stopper  = EarlyStopping(patience=patience, path=f'best_{name}.pth',
                              mode='max')
    history  = {'tl':[], 'ta':[], 'vl':[], 'va':[], 'vf1':[], 'lr':[]}

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        lr = scheduler.step(epoch - 1)
        tl, ta = train_one_epoch(model, train_loader, criterion,
                                  optimizer, scaler)
        vl, va, vf1, _, _, _ = evaluate(model, val_loader)

        history['tl'].append(tl);   history['ta'].append(ta)
        history['vl'].append(vl);   history['va'].append(va)
        history['vf1'].append(vf1); history['lr'].append(lr)

        stop = stopper.step(vf1, epoch, model)
        flag = "BEST" if stopper.counter == 0 else \
               f"p{stopper.counter}/{patience}"
        print(f"  Ep{epoch:02d}/{epochs} | {time.time()-t0:.0f}s | "
              f"LR:{lr:.2e} | Loss {tl:.4f}/{vl:.4f} | "
              f"Acc {ta:.2f}/{va:.2f}% | F1 {vf1:.2f}% | {flag}")
        if stop:
            print(f"  Early stop → best ep{stopper.best_ep} "
                  f"F1={stopper.best_score:.2f}%")
            break

    model.load_state_dict(stopper.best_state)
    return model, history, stopper.best_ep, n_params

# ============================================================
# Train All 3 Hybrid Models
# ============================================================
models_cfg = [
    ('ConvNeXtV2-SwinV2-CBAM', ConvNeXtSwinCBAM(NUM_CLS)),
    ('DeiT-ConvNeXt',      DeiTConvNeXtHybrid(NUM_CLS)),
    ('EVA02-MobileNetV3',  EVAMobileHybrid(NUM_CLS)),
]

trained_models = {}
histories      = {}
best_epochs    = {}
param_counts   = {}

for name, model in models_cfg:
    m, hist, best_ep, n_p = train_model(model, name)
    trained_models[name]   = m
    histories[name]         = hist
    best_epochs[name]       = best_ep
    param_counts[name]      = n_p

# ============================================================
# Test Evaluation
# ============================================================
print("\n" + "="*70)
print("  FINAL TEST RESULTS — 3 Advanced Hybrid Models")
print("="*70)

results = {}
for name, model in trained_models.items():
    _, acc, f1, preds, labels, probs = evaluate(model, test_loader)
    prec = precision_score(labels, preds, average='macro',
                            zero_division=0) * 100
    rec  = recall_score(labels, preds,  average='macro',
                         zero_division=0) * 100
    results[name] = {
        'acc': acc, 'f1': f1, 'prec': prec, 'rec': rec,
        'preds': preds, 'labels': labels, 'probs': probs
    }
    print(f"  {name:<22} Acc:{acc:.2f}%  F1:{f1:.2f}%  "
          f"Prec:{prec:.2f}%  Rec:{rec:.2f}%")

winner_name = max(results, key=lambda k: results[k]['f1'])
print(f"\n  Winner: {winner_name}  "
      f"(F1={results[winner_name]['f1']:.2f}%)")
print("="*70)

# ============================================================
# VIZ 1 — Learning Curves
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(21, 6))
fig.suptitle("Learning Curves — 3 Advanced Hybrid Models",
             fontsize=14, fontweight='bold')

for ax, (name, col) in zip(axes, zip(histories.keys(), MODEL_COLORS)):
    h  = histories[name]
    ep = range(1, len(h['tl']) + 1)
    ax.plot(ep, h['tl'], '--', color=col, alpha=0.5, lw=1.5, label='Train Loss')
    ax.plot(ep, h['vl'], '-',  color=col, lw=2.0,             label='Val Loss')
    ax2 = ax.twinx()
    ax2.plot(ep, h['va'],  '-s', color='gray',  ms=3, lw=1.5, label='Val Acc%')
    ax2.plot(ep, h['vf1'], '-^', color='black', ms=3, lw=1.5, label='Val F1%')
    ax.axvline(best_epochs[name], color='green', ls=':', lw=1.5,
               label=f'Best ep{best_epochs[name]}')
    ax.set_title(f"{name}", fontweight='bold', fontsize=11)
    ax.set_xlabel("Epoch");  ax.set_ylabel("Loss", color=col)
    ax2.set_ylabel("Acc / F1 (%)")
    l1, n1 = ax.get_legend_handles_labels()
    l2, n2 = ax2.get_legend_handles_labels()
    ax.legend(l1+l2, n1+n2, fontsize=7, loc='upper right')
    ax.grid(alpha=0.3)

plt.tight_layout(); plt.show()

# ============================================================
# VIZ 2 — Model Comparison Bar Chart
# ============================================================
metrics       = ['acc', 'f1', 'prec', 'rec']
metric_labels = ['Accuracy', 'Macro F1', 'Precision', 'Recall']
model_names   = list(results.keys())
x = np.arange(len(model_names)); w = 0.2

fig, ax = plt.subplots(figsize=(14, 6))
for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
    vals = [results[m][metric] for m in model_names]
    bars = ax.bar(x + i*w, vals, w, label=label, alpha=0.85)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.4,
                f'{val:.1f}', ha='center', va='bottom',
                fontsize=7, fontweight='bold')

ax.set_xticks(x + w*1.5); ax.set_xticklabels(model_names, fontsize=11)
ax.set_ylabel("Score (%)", fontsize=11); ax.set_ylim([0, 115])
ax.axhline(y=98, color='red', ls='--', lw=1.5, alpha=0.6, label='Target 98%')
ax.set_title("3 Hybrid Models — Test Set Comparison", fontweight='bold', fontsize=13)
ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.3)
best_idx = model_names.index(winner_name)
ax.annotate('BEST', xy=(best_idx + w*1.5, results[winner_name]['f1'] + 5),
            ha='center', fontsize=10, color='green', fontweight='bold')
plt.tight_layout(); plt.show()

# ============================================================
# VIZ 3 — Confusion Matrices
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(24, 7))
fig.suptitle("Confusion Matrices — 3 Advanced Hybrid Models",
             fontsize=14, fontweight='bold')

for ax, (name, col) in zip(axes, zip(results.keys(), MODEL_COLORS)):
    cm  = confusion_matrix(results[name]['labels'], results[name]['preds'])
    cmp = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    sns.heatmap(cmp, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=CLASSES, yticklabels=CLASSES,
                ax=ax, linewidths=0.3, annot_kws={'size': 8},
                cbar_kws={'label': '%'})
    ax.set_title(f"{name}\nAcc={results[name]['acc']:.2f}%  "
                 f"F1={results[name]['f1']:.2f}%",
                 fontweight='bold', fontsize=10)
    ax.set_ylabel("True"); ax.set_xlabel("Predicted")
    ax.tick_params(axis='x', rotation=30, labelsize=8)
    ax.tick_params(axis='y', labelsize=8)

plt.tight_layout(); plt.show()

# ============================================================
# VIZ 4 — ROC Curves (winner model)
# ============================================================
COLORS7 = ['#e74c3c','#8e44ad','#2980b9',
           '#27ae60','#f39c12','#2c3e50','#e67e22']

y_bin = label_binarize(results[winner_name]['labels'],
                        classes=np.arange(NUM_CLS))
probs = results[winner_name]['probs']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

roc_aucs = {}
for i in range(NUM_CLS):
    fpr, tpr, _ = roc_curve(y_bin[:, i], probs[:, i])
    roc_aucs[i] = auc(fpr, tpr)
    ax1.plot(fpr, tpr, color=COLORS7[i], lw=1.5,
             label=f"{CLASSES[i]} (AUC={roc_aucs[i]:.3f})")
ax1.plot([0,1],[0,1],'k--', lw=1)
ax1.set_xlabel("False Positive Rate"); ax1.set_ylabel("True Positive Rate")
ax1.set_title(f"Per-Class ROC — {winner_name}", fontweight='bold')
ax1.legend(fontsize=8, loc='lower right'); ax1.grid(alpha=0.3)

all_fpr  = np.unique(np.concatenate(
    [roc_curve(y_bin[:,i], probs[:,i])[0] for i in range(NUM_CLS)]))
mean_tpr = np.zeros_like(all_fpr)
for i in range(NUM_CLS):
    fp_i, tp_i, _ = roc_curve(y_bin[:,i], probs[:,i])
    mean_tpr += np.interp(all_fpr, fp_i, tp_i)
mean_tpr  /= NUM_CLS
macro_auc  = auc(all_fpr, mean_tpr)

fpr_micro, tpr_micro, _ = roc_curve(y_bin.ravel(), probs.ravel())
micro_auc = auc(fpr_micro, tpr_micro)

ax2.plot(fpr_micro, tpr_micro, 'navy', lw=2.5,
         label=f'Micro-avg (AUC={micro_auc:.4f})')
ax2.plot(all_fpr, mean_tpr, 'darkorange', lw=2.5, ls='--',
         label=f'Macro-avg (AUC={macro_auc:.4f})')
ax2.fill_between(fpr_micro, tpr_micro, alpha=0.08, color='navy')
ax2.fill_between(all_fpr,  mean_tpr,  alpha=0.08, color='darkorange')
ax2.plot([0,1],[0,1],'k--', lw=1)
ax2.set_xlabel("False Positive Rate"); ax2.set_ylabel("True Positive Rate")
ax2.set_title(f"Macro/Micro ROC — {winner_name}", fontweight='bold')
ax2.legend(fontsize=10); ax2.grid(alpha=0.3)

plt.suptitle(f"ROC Curves — {winner_name} (Winner)",
             fontsize=14, fontweight='bold')
plt.tight_layout(); plt.show()

# ============================================================
# VIZ 5 — Per-Class F1 Comparison
# ============================================================
fig, ax = plt.subplots(figsize=(14, 6))
x = np.arange(NUM_CLS); w = 0.25

for i, (name, col) in enumerate(zip(results.keys(), MODEL_COLORS)):
    per_f1 = f1_score(results[name]['labels'], results[name]['preds'],
                      average=None, zero_division=0) * 100
    bars = ax.bar(x + i*w, per_f1, w, label=name, color=col, alpha=0.85)
    for bar, v in zip(bars, per_f1):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                f'{v:.0f}', ha='center', fontsize=7, fontweight='bold')

ax.set_xticks(x + w); ax.set_xticklabels(CLASSES, fontsize=10)
ax.set_ylabel("F1-Score (%)", fontsize=11); ax.set_ylim([0, 115])
ax.axhline(y=98, color='red', ls='--', lw=1.5, alpha=0.6, label='Target 98%')
ax.set_title("Per-Class F1-Score — All 3 Hybrid Models",
             fontweight='bold', fontsize=13)
ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.3)
plt.tight_layout(); plt.show()

# ============================================================
# Final Summary
# ============================================================
print("\n" + "="*75)
print("  COMPLETE SUMMARY — FER-2013 | 15000 imgs | 60/20/20")
print("="*75)
print(f"  {'Model':<24} {'Trainable':>12} {'Acc%':>8} "
      f"{'F1%':>8} {'Prec%':>8} {'Rec%':>8}")
print("-"*75)
for name in results:
    r   = results[name]
    win = " ← BEST" if name == winner_name else ""
    print(f"  {name:<24} {param_counts[name]:>12,} {r['acc']:>8.2f} "
          f"{r['f1']:>8.2f} {r['prec']:>8.2f} {r['rec']:>8.2f}{win}")
print("="*75)
print(f"\n  Winner : {winner_name}")
print(f"  AUC-ROC  Micro: {micro_auc:.4f}  |  Macro: {macro_auc:.4f}")
print()
print(classification_report(
    results[winner_name]['labels'],
    results[winner_name]['preds'],
    target_names=CLASSES, digits=4
))