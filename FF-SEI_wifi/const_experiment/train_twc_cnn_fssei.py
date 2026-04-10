#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_twc_cnn_fssei.py

TWC-CNN adapted baseline for your current FS-SEI pipeline.

Core idea:
- Branch 1: IQ encoder (2-channel 1D CNN)
- Branch 2: Time-Wavelet Spectrum encoder (2D CNN)
- Pretrain stage:
    NT-Xent(z_orig, z_aug) + CE(logits_orig, y) + CE(logits_aug, y)
    + cosine consistency(feat_orig, feat_aug)
- Fine-tune stage:
    freeze pretrained encoders by default, train classifier head on target dataset

Input:
- FeatureMatrix_3.mat, which should contain iqTensor + label_id/device_id (+ file_id optional)
- split_indices_fssei.npz

Output:
- twc_cnn_pretrain_best.pth
- twc_cnn_cls_best.pth
- pretrain_log.csv
- finetune_log.csv
- summary.json

Example:
python train_twc_cnn_fssei.py \
  --mat_all ../FeatureMatrix_OSU_Stable_WiFi_Wireless_unified.mat  \
  --split_npz ../split_indices_fssei_osu_stable_wireless.npz \
  --save_dir ./experiments/e2_dl_twc_cnn \
  --pre_epochs 40 \
  --ft_epochs 80 \
  --batch_size 64 \
  --eval_split val
"""

import os
import csv
import json
import copy
import math
import argparse
from typing import Optional, Tuple

import numpy as np
import scipy.io as sio
from scipy.signal import cwt, morlet2, resample

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    from sklearn.metrics import f1_score
except Exception:
    f1_score = None

CWD_DIR = os.getcwd()


# =========================================================
# basic utils
# =========================================================
def resolve_path(p: str, base_dir: Optional[str] = None) -> str:
    p = os.path.expanduser(str(p).strip())
    if os.path.isabs(p):
        return os.path.normpath(p)
    return os.path.normpath(os.path.join(base_dir or CWD_DIR, p))


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_mat_v73_h5py(path):
    import h5py

    def _fix_matlab_order(arr):
        if isinstance(arr, np.ndarray) and arr.ndim >= 2:
            arr = np.transpose(arr, tuple(range(arr.ndim))[::-1])
        return arr

    def _h5_to_obj(obj):
        if isinstance(obj, h5py.Dataset):
            data = np.array(obj[()])
            return _fix_matlab_order(data)
        if isinstance(obj, h5py.Group):
            return {k: _h5_to_obj(obj[k]) for k in obj.keys()}
        return obj

    out = {"__backend__": "h5py"}
    with h5py.File(path, "r") as f:
        for k in f.keys():
            out[k] = _h5_to_obj(f[k])
    return out


def load_mat_auto(path):
    try:
        mat = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
        return {"__backend__": "scipy", **mat}
    except Exception as e:
        msg = str(e).lower()
        if ("version 73" in msg) or ("unknown mat file type" in msg) or ("matlab 7.3" in msg):
            return load_mat_v73_h5py(path)
        raise


def pick_first_existing(d, keys):
    for k in keys:
        if k in d:
            return k
    return None


def to_1d_int(y):
    y = np.array(y).reshape(-1).astype(np.int64)
    _, y_new = np.unique(y, return_inverse=True)
    return y_new.astype(np.int64)


def load_split_npz(path):
    Z = np.load(path, allow_pickle=True)

    def _get(*names):
        for n in names:
            if n in Z:
                return Z[n]
        return None

    tr = _get("train_idx", "idx_train", "train_indices")
    va = _get("val_idx", "idx_val", "val_indices")
    te = _get("test_idx", "idx_test", "test_indices")
    if tr is None or va is None or te is None:
        raise KeyError(f"split_npz missing train/val/test keys, keys={list(Z.keys())}")
    return tr.astype(np.int64), va.astype(np.int64), te.astype(np.int64)


def load_label_map_csv(path: str, num_classes: int):
    class_names = [str(i) for i in range(num_classes)]
    if not path or (not os.path.isfile(path)):
        return class_names
    try:
        import pandas as pd
        df = pd.read_csv(path)
        df.columns = [str(c).lower().strip() for c in df.columns]
        if "class_id" in df.columns and "name" in df.columns:
            mp = {}
            for _, r in df.iterrows():
                try:
                    mp[int(r["class_id"])] = str(r["name"])
                except Exception:
                    pass
            class_names = [mp.get(i, str(i)) for i in range(num_classes)]
    except Exception:
        pass
    return class_names


# =========================================================
# iq loading
# =========================================================
def _ensure_iq_ncl(arr: np.ndarray, n_expected: int = None) -> np.ndarray:
    """
    Convert iqTensor to [N, 2, L].

    Supported common layouts:
      - complex [N, L]
      - [N, 2, L]
      - [N, L, 2]
      - [2, L, N]
      - [L, 2, N]

    If n_expected is provided (recommended), use it to infer which axis is N.
    """
    arr = np.asarray(arr)

    # case 1: complex [N, L]
    if np.iscomplexobj(arr):
        if arr.ndim != 2:
            raise ValueError(f"Unsupported complex iqTensor shape: {arr.shape}")

        # case A: already [N, L]
        if n_expected is None or arr.shape[0] == n_expected:
            out = np.stack([arr.real, arr.imag], axis=1).astype(np.float32)  # [N,2,L]
            if n_expected is None or out.shape[0] == n_expected:
                return out

        # case B: actually [L, N] -> transpose to [N, L]
        if n_expected is None or arr.shape[1] == n_expected:
            arr_t = arr.T
            out = np.stack([arr_t.real, arr_t.imag], axis=1).astype(np.float32)  # [N,2,L]
            if n_expected is None or out.shape[0] == n_expected:
                return out

        raise ValueError(f"Complex iqTensor converted shape mismatch: raw={arr.shape}, expected N={n_expected}")


    # case 2: real 3D tensor
    if arr.ndim == 3:
        candidates = []

        # [N, 2, L]
        if arr.shape[1] == 2 and (n_expected is None or arr.shape[0] == n_expected):
            candidates.append(arr.astype(np.float32))

        # [N, L, 2] -> [N, 2, L]
        if arr.shape[2] == 2 and (n_expected is None or arr.shape[0] == n_expected):
            candidates.append(np.transpose(arr, (0, 2, 1)).astype(np.float32))

        # [2, L, N] -> [N, 2, L]
        if arr.shape[0] == 2 and (n_expected is None or arr.shape[2] == n_expected):
            candidates.append(np.transpose(arr, (2, 0, 1)).astype(np.float32))

        # [L, 2, N] -> [N, 2, L]
        if arr.shape[1] == 2 and (n_expected is None or arr.shape[2] == n_expected):
            candidates.append(np.transpose(arr, (2, 1, 0)).astype(np.float32))

        # 去重后选合法项
        valid = []
        for c in candidates:
            if c.ndim == 3 and c.shape[1] == 2:
                if n_expected is None or c.shape[0] == n_expected:
                    valid.append(c)

        if len(valid) == 1:
            return valid[0]

        if len(valid) > 1:
            # 优先选样本数匹配且长度维更像时间长度的
            valid = sorted(valid, key=lambda x: x.shape[2], reverse=True)
            return valid[0]

    raise ValueError(f"Unsupported iqTensor shape: {arr.shape}, expected N={n_expected}")
    

def load_featurematrix3_iq(mat_path: str):
    S = load_mat_auto(mat_path)

    kiq = pick_first_existing(S, ["iqTensor", "iq_tensor", "X_iq", "iq"])
    if kiq is None:
        raise KeyError("MAT missing iqTensor/iq_tensor/X_iq/iq")

    ky = pick_first_existing(S, ["label_id", "device_id", "y", "labels"])
    if ky is None:
        raise KeyError("MAT missing label_id/device_id/y/labels")

    y = to_1d_int(S[ky])
    iq = _ensure_iq_ncl(S[kiq], n_expected=len(y))

    if iq.shape[0] != len(y):
        raise ValueError(f"iqTensor N mismatch labels: {iq.shape[0]} vs {len(y)}")

    file_id = None
    kfid = pick_first_existing(S, ["file_id", "fileId", "fileID", "file_idx", "file_index_id"])
    if kfid is not None:
        tmp = np.asarray(S[kfid]).reshape(-1)
        if len(tmp) == len(y):
            file_id = tmp.astype(np.int64)
    
    print("[Debug] raw iqTensor shape =", np.asarray(S[kiq]).shape)
    print("[Debug] label length =", len(y))

    return iq.astype(np.float32), y.astype(np.int64), file_id, S


# =========================================================
# cwt transform
# =========================================================
def iq_to_wavelet_map(
    iq_2l: np.ndarray,
    scales: int = 32,
    time_bins: int = 256,
    w: float = 6.0,
) -> np.ndarray:
    """
    Build one time-wavelet spectrum map from IQ signal.
    Input:
      iq_2l: [2, L]
    Output:
      wt_map: [1, scales, time_bins]
    """
    assert iq_2l.ndim == 2 and iq_2l.shape[0] == 2

    i_sig = iq_2l[0]
    q_sig = iq_2l[1]
    amp = np.sqrt(i_sig ** 2 + q_sig ** 2)

    # resample time axis first for speed
    if len(amp) != time_bins:
        sig = resample(amp, time_bins).astype(np.float32)
    else:
        sig = amp.astype(np.float32)

    widths = np.arange(1, scales + 1)
    wt = cwt(sig, lambda M, s: morlet2(M, s, w=w), widths)  # [S, T]
    wt = np.abs(wt).astype(np.float32)
    wt = np.log1p(wt)

    # per-sample normalize
    wt = (wt - wt.mean()) / (wt.std() + 1e-6)
    return wt[None, :, :].astype(np.float32)  # [1,S,T]


# =========================================================
# augmentations
# =========================================================
def augment_iq(
    iq_2l: np.ndarray,
    noise_std: float = 0.01,
    amp_scale_low: float = 0.9,
    amp_scale_high: float = 1.1,
    max_shift_ratio: float = 0.05,
    phase_rot_deg: float = 15.0,
    p_flip: float = 0.5,
) -> np.ndarray:
    """
    Simple RF-style augmentations on IQ:
      - phase rotation
      - amplitude scaling
      - additive noise
      - cyclic shift
      - optional time reverse
    """
    x = iq_2l.copy().astype(np.float32)
    I, Q = x[0], x[1]

    # amplitude scaling
    scale = np.random.uniform(amp_scale_low, amp_scale_high)
    I = I * scale
    Q = Q * scale

    # phase rotation
    theta = np.deg2rad(np.random.uniform(-phase_rot_deg, phase_rot_deg))
    c, s = np.cos(theta), np.sin(theta)
    I_rot = c * I - s * Q
    Q_rot = s * I + c * Q
    I, Q = I_rot, Q_rot

    # cyclic shift
    max_shift = max(1, int(len(I) * max_shift_ratio))
    shift = np.random.randint(-max_shift, max_shift + 1)
    I = np.roll(I, shift)
    Q = np.roll(Q, shift)

    # time reverse
    if np.random.rand() < p_flip:
        I = I[::-1].copy()
        Q = Q[::-1].copy()

    # additive noise
    sig_std = float(np.sqrt(np.mean(I ** 2 + Q ** 2)) + 1e-6)
    eps = np.random.randn(*I.shape).astype(np.float32) * (noise_std * sig_std)
    eta = np.random.randn(*Q.shape).astype(np.float32) * (noise_std * sig_std)
    I = I + eps
    Q = Q + eta

    return np.stack([I, Q], axis=0).astype(np.float32)


# =========================================================
# dataset
# =========================================================
class TWCPretrainDataset(Dataset):
    def __init__(self, iq, y, indices, mean, std, wt_scales, wt_time_bins, wt_w):
        self.iq = iq[indices].astype(np.float32)
        self.y = y[indices].astype(np.int64)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.mean = float(mean)
        self.std = float(std)
        self.wt_scales = int(wt_scales)
        self.wt_time_bins = int(wt_time_bins)
        self.wt_w = float(wt_w)

    def __len__(self):
        return len(self.indices)

    def _norm_iq(self, x):
        return (x - self.mean) / max(self.std, 1e-6)

    def __getitem__(self, idx):
        iq0 = self.iq[idx].copy()
        iq1 = augment_iq(iq0)

        iq0 = self._norm_iq(iq0).astype(np.float32)
        iq1 = self._norm_iq(iq1).astype(np.float32)

        wt0 = iq_to_wavelet_map(iq0, self.wt_scales, self.wt_time_bins, self.wt_w)
        wt1 = iq_to_wavelet_map(iq1, self.wt_scales, self.wt_time_bins, self.wt_w)

        return (
            torch.from_numpy(iq0),
            torch.from_numpy(wt0),
            torch.from_numpy(iq1),
            torch.from_numpy(wt1),
            int(self.y[idx]),
        )


class TWCFineTuneDataset(Dataset):
    def __init__(self, iq, y, indices, mean, std, wt_scales, wt_time_bins, wt_w):
        self.iq = iq[indices].astype(np.float32)
        self.y = y[indices].astype(np.int64)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.mean = float(mean)
        self.std = float(std)
        self.wt_scales = int(wt_scales)
        self.wt_time_bins = int(wt_time_bins)
        self.wt_w = float(wt_w)

    def __len__(self):
        return len(self.indices)

    def _norm_iq(self, x):
        return (x - self.mean) / max(self.std, 1e-6)

    def __getitem__(self, idx):
        iq0 = self._norm_iq(self.iq[idx]).astype(np.float32)
        wt0 = iq_to_wavelet_map(iq0, self.wt_scales, self.wt_time_bins, self.wt_w)
        return torch.from_numpy(iq0), torch.from_numpy(wt0), int(self.y[idx])


# =========================================================
# model
# =========================================================
class Conv1DBlock(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(cin, cout, 5, stride=stride, padding=2, bias=False),
            nn.BatchNorm1d(cout),
            nn.ReLU(inplace=True),
            nn.Conv1d(cout, cout, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(cout),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class IQEncoder(nn.Module):
    def __init__(self, in_ch=2, width=32, feat_dim=1024, dropout=0.0):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, width, 5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(width),
            nn.ReLU(inplace=True),
        )
        self.b1 = Conv1DBlock(width, width * 2, stride=2)
        self.b2 = Conv1DBlock(width * 2, width * 4, stride=2)
        self.b3 = Conv1DBlock(width * 4, width * 8, stride=2)
        self.b4 = Conv1DBlock(width * 8, width * 8, stride=2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(width * 8, feat_dim)

    def forward(self, x):
        x = self.stem(x)
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.pool(x).squeeze(-1)
        x = self.drop(x)
        return self.fc(x)


class Conv2DBlock(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class WTEncoder(nn.Module):
    def __init__(self, in_ch=1, width=32, feat_dim=1024, dropout=0.0):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, width, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
        )
        self.b1 = Conv2DBlock(width, width * 2, stride=2)
        self.b2 = Conv2DBlock(width * 2, width * 4, stride=2)
        self.b3 = Conv2DBlock(width * 4, width * 8, stride=2)
        self.b4 = Conv2DBlock(width * 8, width * 8, stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(width * 8, feat_dim)

    def forward(self, x):
        x = self.stem(x)
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.pool(x).flatten(1)
        x = self.drop(x)
        return self.fc(x)


class TWCBackbone(nn.Module):
    def __init__(self, iq_width=32, wt_width=32, feat_dim=1024, dropout=0.0):
        super().__init__()
        self.iq_encoder = IQEncoder(in_ch=2, width=iq_width, feat_dim=feat_dim, dropout=dropout)
        self.wt_encoder = WTEncoder(in_ch=1, width=wt_width, feat_dim=feat_dim, dropout=dropout)

    def forward_features(self, iq, wt):
        fiq = self.iq_encoder(iq)
        fwt = self.wt_encoder(wt)
        fused = torch.cat([fiq, fwt], dim=1)  # [B, 2*feat_dim]
        return fused, fiq, fwt


class TWCPretrainModel(nn.Module):
    def __init__(self, backbone, feat_dim=1024, proj_dim=256, num_classes=10, dropout=0.0):
        super().__init__()
        self.backbone = backbone
        fused_dim = feat_dim * 2
        self.projector = nn.Sequential(
            nn.Linear(fused_dim, fused_dim),
            nn.ReLU(inplace=True),
            nn.Linear(fused_dim, proj_dim),
        )
        self.cls_head = nn.Sequential(
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(fused_dim, num_classes),
        )

    def forward(self, iq, wt):
        feat, fiq, fwt = self.backbone.forward_features(iq, wt)
        z = self.projector(feat)
        logits = self.cls_head(feat)
        return feat, z, logits, fiq, fwt


class TWCClassifier(nn.Module):
    def __init__(self, backbone, feat_dim=1024, num_classes=10, dropout=0.0):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(feat_dim * 2, num_classes),
        )

    def forward_features(self, iq, wt):
        feat, _, _ = self.backbone.forward_features(iq, wt)
        return feat

    def forward(self, iq, wt):
        feat = self.forward_features(iq, wt)
        return self.head(feat)


# ===== complexity / export helper =====
def create_backbone(
    iq_width=32,
    wt_width=32,
    feat_dim=1024,
    dropout=0.0,
    **kwargs
):
    return TWCBackbone(
        iq_width=iq_width,
        wt_width=wt_width,
        feat_dim=feat_dim,
        dropout=dropout,
    )


def create_model(
    num_classes,
    feat_dim=1024,
    proj_dim=256,
    iq_width=32,
    wt_width=32,
    dropout=0.0,
    mode="classifier",
    **kwargs
):
    """
    统一给外部脚本调用的构造入口。

    mode="classifier" -> 返回 TWCClassifier（用于闭集分类/复杂度统计）
    mode="pretrain"   -> 返回 TWCPretrainModel（若你想统计预训练头）
    """
    backbone = create_backbone(
        iq_width=iq_width,
        wt_width=wt_width,
        feat_dim=feat_dim,
        dropout=dropout,
    )

    if str(mode).lower() in ["pretrain", "pre", "ssl"]:
        return TWCPretrainModel(
            backbone=backbone,
            feat_dim=feat_dim,
            proj_dim=proj_dim,
            num_classes=num_classes,
            dropout=dropout,
        )

    return TWCClassifier(
        backbone=backbone,
        feat_dim=feat_dim,
        num_classes=num_classes,
        dropout=dropout,
    )


def load_pretrained(model, ckpt_path, map_location="cpu", strict=False):
    """
    统一给复杂度/导出脚本调用的权重加载入口。

    兼容两类 checkpoint:
    1) twc_cnn_pretrain_best.pth
       {"backbone_state": ...}
    2) twc_cnn_cls_best.pth
       {"model_state": ...}
    """
    ckpt = torch.load(ckpt_path, map_location=map_location)

    if "model_state" in ckpt:
        state = ckpt["model_state"]
        missing, unexpected = model.load_state_dict(state, strict=strict)
        return model, {
            "type": "model_state",
            "missing": missing,
            "unexpected": unexpected,
        }

    if "backbone_state" in ckpt:
        if hasattr(model, "backbone"):
            missing, unexpected = model.backbone.load_state_dict(
                ckpt["backbone_state"], strict=strict
            )
            return model, {
                "type": "backbone_state",
                "missing": missing,
                "unexpected": unexpected,
            }
        else:
            raise RuntimeError(
                "Checkpoint contains backbone_state, but model has no .backbone"
            )

    raise KeyError(
        f"Unsupported checkpoint keys: {list(ckpt.keys())}"
    )

# =========================================================
# losses
# =========================================================
def nt_xent_loss(z1, z2, temperature=0.2):
    """
    NT-Xent on paired batch.
    """
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    B = z1.size(0)
    z = torch.cat([z1, z2], dim=0)  # [2B, D]
    sim = torch.matmul(z, z.t()) / temperature  # [2B, 2B]

    mask = torch.eye(2 * B, device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(mask, -1e9)

    targets = torch.arange(B, device=z.device)
    targets = torch.cat([targets + B, targets], dim=0)

    loss = F.cross_entropy(sim, targets)
    return loss


# =========================================================
# train / eval
# =========================================================
@torch.no_grad()
def evaluate_pretrain(model, loader, device):
    model.eval()
    total_loss = 0.0
    ys, ps = [], []
    n = 0

    for iq0, wt0, iq1, wt1, yb in loader:
        iq0 = iq0.to(device, non_blocking=True)
        wt0 = wt0.to(device, non_blocking=True)
        iq1 = iq1.to(device, non_blocking=True)
        wt1 = wt1.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        feat0, z0, logits0, _, _ = model(iq0, wt0)
        feat1, z1, logits1, _, _ = model(iq1, wt1)

        loss_nt = nt_xent_loss(z0, z1)
        loss_ce0 = F.cross_entropy(logits0, yb)
        loss_ce1 = F.cross_entropy(logits1, yb)
        loss_cos = (1.0 - F.cosine_similarity(feat0, feat1, dim=1)).mean()
        loss = loss_nt + loss_ce0 + loss_ce1 + loss_cos

        total_loss += float(loss.item()) * yb.size(0)
        pred = torch.argmax(logits0, dim=1)
        ys.append(yb.cpu().numpy())
        ps.append(pred.cpu().numpy())
        n += yb.size(0)

    ys = np.concatenate(ys)
    ps = np.concatenate(ps)
    acc = float((ys == ps).mean())
    macro_f1 = float(f1_score(ys, ps, average="macro")) if f1_score is not None else float("nan")
    return {"loss": total_loss / max(1, n), "acc": acc, "macro_f1": macro_f1}


@torch.no_grad()
def evaluate_cls(model, loader, device):
    model.eval()
    total_loss = 0.0
    ys, ps = [], []
    n = 0
    for iq0, wt0, yb in loader:
        iq0 = iq0.to(device, non_blocking=True)
        wt0 = wt0.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        logits = model(iq0, wt0)
        loss = F.cross_entropy(logits, yb)
        total_loss += float(loss.item()) * yb.size(0)
        pred = torch.argmax(logits, dim=1)
        ys.append(yb.cpu().numpy())
        ps.append(pred.cpu().numpy())
        n += yb.size(0)

    ys = np.concatenate(ys)
    ps = np.concatenate(ps)
    acc = float((ys == ps).mean())
    macro_f1 = float(f1_score(ys, ps, average="macro")) if f1_score is not None else float("nan")
    return {"loss": total_loss / max(1, n), "acc": acc, "macro_f1": macro_f1}


# =========================================================
# main
# =========================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mat_all", type=str, required=True)
    ap.add_argument("--split_npz", type=str, required=True)
    ap.add_argument("--save_dir", type=str, required=True)
    ap.add_argument("--label_map_csv", type=str, default="")

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=4)

    ap.add_argument("--feat_dim", type=int, default=1024)
    ap.add_argument("--proj_dim", type=int, default=256)
    ap.add_argument("--iq_width", type=int, default=32)
    ap.add_argument("--wt_width", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.0)

    ap.add_argument("--wt_scales", type=int, default=32)
    ap.add_argument("--wt_time_bins", type=int, default=256)
    ap.add_argument("--wt_morlet_w", type=float, default=6.0)

    ap.add_argument("--do_pretrain", type=int, default=1, choices=[0, 1])
    ap.add_argument("--pre_epochs", type=int, default=80)
    ap.add_argument("--pre_lr", type=float, default=1e-3)
    ap.add_argument("--pre_wd", type=float, default=1e-4)
    ap.add_argument("--temperature", type=float, default=0.2)

    ap.add_argument("--ft_epochs", type=int, default=120)
    ap.add_argument("--ft_lr", type=float, default=1e-3)
    ap.add_argument("--ft_wd", type=float, default=1e-4)
    ap.add_argument("--freeze_backbone", type=int, default=1, choices=[0, 1])
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--eval_split", type=str, default="val", choices=["val", "test"])

    args = ap.parse_args()

    set_seed(args.seed)
    args.mat_all = resolve_path(args.mat_all, CWD_DIR)
    args.split_npz = resolve_path(args.split_npz, CWD_DIR)
    args.save_dir = resolve_path(args.save_dir, CWD_DIR)
    os.makedirs(args.save_dir, exist_ok=True)

    iq, y, file_id, S = load_featurematrix3_iq(args.mat_all)
    tr_idx, va_idx, te_idx = load_split_npz(args.split_npz)
    num_classes = int(np.unique(y).size)
    class_names = load_label_map_csv(args.label_map_csv, num_classes)

    # train-only global normalization
    mean = float(iq[tr_idx].mean())
    std = float(iq[tr_idx].std() + 1e-6)
    stats = {"mean": mean, "std": std}

    ds_pre_tr = TWCPretrainDataset(iq, y, tr_idx, mean, std, args.wt_scales, args.wt_time_bins, args.wt_morlet_w)
    ds_pre_va = TWCPretrainDataset(iq, y, va_idx, mean, std, args.wt_scales, args.wt_time_bins, args.wt_morlet_w)

    ds_ft_tr = TWCFineTuneDataset(iq, y, tr_idx, mean, std, args.wt_scales, args.wt_time_bins, args.wt_morlet_w)
    ds_ft_va = TWCFineTuneDataset(iq, y, va_idx, mean, std, args.wt_scales, args.wt_time_bins, args.wt_morlet_w)
    ds_ft_te = TWCFineTuneDataset(iq, y, te_idx, mean, std, args.wt_scales, args.wt_time_bins, args.wt_morlet_w)

    dl_pre_tr = DataLoader(ds_pre_tr, batch_size=args.batch_size, shuffle=True,
                           num_workers=args.num_workers, pin_memory=True, drop_last=True)
    dl_pre_va = DataLoader(ds_pre_va, batch_size=args.batch_size, shuffle=False,
                           num_workers=args.num_workers, pin_memory=True)

    dl_ft_tr = DataLoader(ds_ft_tr, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, pin_memory=True)
    dl_ft_va = DataLoader(ds_ft_va, batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers, pin_memory=True)
    dl_ft_te = DataLoader(ds_ft_te, batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers, pin_memory=True)

    eval_loader = dl_ft_va if args.eval_split == "val" else dl_ft_te

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -----------------------------------------------------
    # stage 1: pretrain
    # -----------------------------------------------------
    pre_ckpt = os.path.join(args.save_dir, "twc_cnn_pretrain_best.pth")
    pre_log_csv = os.path.join(args.save_dir, "pretrain_log.csv")

    backbone = TWCBackbone(
        iq_width=args.iq_width,
        wt_width=args.wt_width,
        feat_dim=args.feat_dim,
        dropout=args.dropout,
    ).to(device)

    if args.do_pretrain == 1:
        pre_model = TWCPretrainModel(
            backbone=backbone,
            feat_dim=args.feat_dim,
            proj_dim=args.proj_dim,
            num_classes=num_classes,
            dropout=args.dropout,
        ).to(device)

        opt = torch.optim.AdamW(pre_model.parameters(), lr=args.pre_lr, weight_decay=args.pre_wd)
        best_eval = -1.0
        best_state = None
        logs = []

        for ep in range(1, args.pre_epochs + 1):
            pre_model.train()
            total_loss = 0.0
            total_seen = 0
            total_correct = 0

            for iq0, wt0, iq1, wt1, yb in dl_pre_tr:
                iq0 = iq0.to(device, non_blocking=True)
                wt0 = wt0.to(device, non_blocking=True)
                iq1 = iq1.to(device, non_blocking=True)
                wt1 = wt1.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)

                feat0, z0, logits0, _, _ = pre_model(iq0, wt0)
                feat1, z1, logits1, _, _ = pre_model(iq1, wt1)

                loss_nt = nt_xent_loss(z0, z1, args.temperature)
                loss_ce0 = F.cross_entropy(logits0, yb)
                loss_ce1 = F.cross_entropy(logits1, yb)
                loss_cos = (1.0 - F.cosine_similarity(feat0, feat1, dim=1)).mean()
                loss = loss_nt + loss_ce0 + loss_ce1 + loss_cos

                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

                total_loss += float(loss.item()) * yb.size(0)
                total_seen += yb.size(0)
                total_correct += int((logits0.argmax(dim=1) == yb).sum().item())

            tr_loss = total_loss / max(1, total_seen)
            tr_acc = total_correct / max(1, total_seen)
            ev = evaluate_pretrain(pre_model, dl_pre_va, device)

            logs.append({
                "epoch": ep,
                "train_loss": tr_loss,
                "train_acc": tr_acc,
                "eval_loss": ev["loss"],
                "eval_acc": ev["acc"],
                "eval_macro_f1": ev["macro_f1"],
            })

            print(f"[Pretrain {ep:03d}] train_loss={tr_loss:.4f}, train_acc={tr_acc:.4f} | "
                  f"eval_loss={ev['loss']:.4f}, eval_acc={ev['acc']:.4f}, eval_f1={ev['macro_f1']:.4f}")

            if ev["acc"] > best_eval:
                best_eval = ev["acc"]
                best_state = copy.deepcopy(pre_model.backbone.state_dict())
                torch.save({
                    "backbone_state": best_state,
                    "args": vars(args),
                    "stats": stats,
                    "num_classes": num_classes,
                    "class_names": class_names,
                    "input_shape_iq": list(iq.shape[1:]),
                    "wavelet_shape": [1, args.wt_scales, args.wt_time_bins],
                }, pre_ckpt)

        with open(pre_log_csv, "w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "train_acc", "eval_loss", "eval_acc", "eval_macro_f1"])
            wr.writeheader()
            wr.writerows(logs)

        if best_state is not None:
            backbone.load_state_dict(best_state, strict=True)
    else:
        print("[Pretrain] skipped, backbone remains randomly initialized.")

    # -----------------------------------------------------
    # stage 2: fine-tune classifier
    # -----------------------------------------------------
    cls_ckpt = os.path.join(args.save_dir, "twc_cnn_cls_best.pth")
    ft_log_csv = os.path.join(args.save_dir, "finetune_log.csv")

    clf_model = TWCClassifier(
        backbone=backbone,
        feat_dim=args.feat_dim,
        num_classes=num_classes,
        dropout=args.dropout,
    ).to(device)

    if args.freeze_backbone == 1:
        for p in clf_model.backbone.parameters():
            p.requires_grad = False

    trainable = [p for p in clf_model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.ft_lr, weight_decay=args.ft_wd)

    best_eval = -1.0
    best_state = None
    bad = 0
    logs = []

    for ep in range(1, args.ft_epochs + 1):
        clf_model.train()
        total_loss = 0.0
        total_seen = 0
        total_correct = 0

        for iq0, wt0, yb in dl_ft_tr:
            iq0 = iq0.to(device, non_blocking=True)
            wt0 = wt0.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            logits = clf_model(iq0, wt0)
            loss = F.cross_entropy(logits, yb)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            total_loss += float(loss.item()) * yb.size(0)
            total_seen += yb.size(0)
            total_correct += int((logits.argmax(dim=1) == yb).sum().item())

        tr_loss = total_loss / max(1, total_seen)
        tr_acc = total_correct / max(1, total_seen)
        ev = evaluate_cls(clf_model, eval_loader, device)

        logs.append({
            "epoch": ep,
            "train_loss": tr_loss,
            "train_acc": tr_acc,
            "eval_loss": ev["loss"],
            "eval_acc": ev["acc"],
            "eval_macro_f1": ev["macro_f1"],
        })

        print(f"[FineTune {ep:03d}] train_loss={tr_loss:.4f}, train_acc={tr_acc:.4f} | "
              f"eval_loss={ev['loss']:.4f}, eval_acc={ev['acc']:.4f}, eval_f1={ev['macro_f1']:.4f}")

        if ev["acc"] > best_eval:
            best_eval = ev["acc"]
            best_state = copy.deepcopy(clf_model.state_dict())
            bad = 0
            torch.save({
                "model_state": best_state,
                "args": vars(args),
                "stats": stats,
                "num_classes": num_classes,
                "class_names": class_names,
                "input_shape_iq": list(iq.shape[1:]),
                "wavelet_shape": [1, args.wt_scales, args.wt_time_bins],
            }, cls_ckpt)
        else:
            bad += 1
            if bad >= args.patience:
                print("[EarlyStop] patience reached.")
                break

    with open(ft_log_csv, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "train_acc", "eval_loss", "eval_acc", "eval_macro_f1"])
        wr.writeheader()
        wr.writerows(logs)

    if best_state is not None:
        clf_model.load_state_dict(best_state, strict=True)

    test_stats = evaluate_cls(clf_model, dl_ft_te, device)
    print(f"[Test] loss={test_stats['loss']:.4f}, acc={test_stats['acc']:.4f}, macro_f1={test_stats['macro_f1']:.4f}")

    summary = {
        "model_name": "TWC-CNN adapted baseline",
        "paper_alignment": {
            "iq_branch": True,
            "wavelet_branch": True,
            "nt_xent_pretrain": True,
            "ce_pretrain_original_aug": True,
            "cosine_consistency": True,
            "freeze_encoder_during_finetune": bool(args.freeze_backbone),
            "input_used": "iqTensor",
            "wavelet_online_cwt": True,
        },
        "best_eval_acc": float(best_eval),
        "test_metrics": test_stats,
        "num_samples": int(len(y)),
        "num_classes": int(num_classes),
        "class_names": class_names,
        "stats": stats,
    }
    with open(os.path.join(args.save_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("[Save]", os.path.join(args.save_dir, "summary.json"))


if __name__ == "__main__":
    main()