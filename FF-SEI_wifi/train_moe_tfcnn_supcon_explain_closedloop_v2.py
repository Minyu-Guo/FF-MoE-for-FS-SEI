#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_moe_tfcnn_supcon_explain_closedloop_v2.py

Closed-loop pretraining script:
- Load FeatureMatrix_3.mat (featureMatrix(39) + specTensor + label_id + fsVector + file_id optional)
- Load topk_indices.mat (union indices) -> gate-only extra features
- Train FeatureFamilyMoE (4 experts: time/freq/tf(inst+spec)/inst) with:
    CE main + AUX(expert CE) + SupCon(h_fusion) + entropy regularization
- Save:
    - moe_tfcnn_best.pth
    - train_log.csv
    - expert_load_curve.png
    - val_confusion_counts.png / val_confusion_norm.png (+ .npy)
    - gate_profile_val.csv  (with class_name)
    - top_confusions_val.csv

Label names:
- Prefer --label_map label_map.csv with columns: class_id,name
- Else try read from MAT: labelVector (N strings) or file_index.labelStr (per file)

说明：
BT数据集：FeatureMatrix_3.mat    topk_indices.mat
WiFi数据集：FeatureMatrix_Indoor_OSU.mat   topk_indices_wifi.mat

冻结专家只训gate:
python train_moe_tfcnn_supcon_explain_closedloop_v2.py  --mat_all FeatureMatrix_Indoor_OSU.mat   --topk_mat topk_indices_wifi.mat   --save_dir ./stage2_gate_only_occ_entropy1e-2    --init_from ./stage2_gate_only_occ/moe_injected_experts_occ_1.pth   --train_gate_only  --gate_align_level batch_mean
端到端：
python train_moe_tfcnn_supcon_explain_closedloop_v2.py   --mat_all FeatureMatrix_3.mat   --topk_mat topk_indices.mat   --save_dir ./exp_end2end_occ_e1e2e3e4_entropy3e-2 
消融实验后加：
--disable_expert time   --save_dir ablation/exp_no_time   去掉 time expert
--disable_expert freq   --save_dir ablation/exp_no_freq   去掉 freq（I/Q 占用图专家）
--disable_expert tf   --save_dir ablation/exp_no_tf       去掉 tf（TF + spectrogram）
--disable_expert inst   --save_dir ablation/exp_no_inst   去掉 inst（瞬时特征）
--save_dir ablation/exp_full     全专家

1、noinst + 零正则 gate-only warmup
python train_moe_tfcnn_supcon_explain_closedloop_v2.py \
  --mat_all FeatureMatrix_OSU_Stable_WiFi_Wireless_unified.mat \
  --topk_mat topk_indices_wifi.mat \
  --save_dir ./result/pretrain/stage3a_gatewarmup_osu_stable \
  --init_from ./moe_injected_experts_osu_stable.pth \
  --train_gate_only \
  --disable_expert inst \
  --epochs 8 \
  --batch_size 32 \
  --lr 1e-3 \
  --weight_decay 1e-4 \
  --gate_align_coef 0 \
  --entropy_coef 0 \
  --sample_entropy_coef 0 \
  --supcon_weight 0 \
  --aux_weight 0 \
  --seed 42
2、联合微调(解冻专家一起训)
python train_moe_tfcnn_supcon_explain_closedloop_v2.py \
  --mat_all FeatureMatrix_OSU_Stable_WiFi_Wireless_unified.mat \
  --topk_mat topk_indices_wifi.mat \
  --save_dir ./result/pretrain/stage3b_joint_osu_stable \
  --init_from ./result/pretrain/stage3a_gatewarmup_osu_stable/moe_tfcnn_best.pth \
  --disable_expert inst \
  --epochs 12 \
  --batch_size 32 \
  --lr 3e-4 \
  --weight_decay 1e-4 \
  --gate_align_coef 0 \
  --entropy_coef 0 \
  --sample_entropy_coef 0 \
  --supcon_weight 0 \
  --aux_weight 0.1 \
  --seed 42

消融：
python train_moe_tfcnn_supcon_explain_closedloop_v2.py \
  --mat_all FeatureMatrix_OSU_Stable_WiFi_Wireless_unified.mat \
  --topk_mat topk_indices_wifi.mat \
  --save_dir ./ablation/no_inst/stageA \
  --init_from ./moe_injected_experts_osu_stable.pth \
  --train_gate_only \
  --disable_expert inst \
  --epochs 8 \
  --batch_size 32 \
  --lr 1e-3 \
  --weight_decay 1e-4 \
  --gate_align_coef 0 \
  --entropy_coef 0 \
  --sample_entropy_coef 0 \
  --supcon_weight 0 \
  --aux_weight 0 \
  --seed 42

python train_moe_tfcnn_supcon_explain_closedloop_v2.py \
  --mat_all FeatureMatrix_OSU_Stable_WiFi_Wireless_unified.mat \
  --topk_mat topk_indices_wifi.mat \
  --save_dir ./ablation/no_inst/stageB \
  --init_from ./ablation/no_inst/stageA/moe_tfcnn_best.pth \
  --disable_expert inst \
  --epochs 12 \
  --batch_size 32 \
  --lr 3e-4 \
  --weight_decay 1e-4 \
  --gate_align_coef 0 \
  --entropy_coef 0 \
  --sample_entropy_coef 0 \
  --supcon_weight 0 \
  --aux_weight 0.1 \
  --seed 42
"""

import os
import time
import random
import argparse
from collections import Counter

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import scipy.io as sio
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

# optional plotting
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

try:
    import seaborn as sns
except Exception:
    sns = None

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Liberation Serif", "Nimbus Roman", "DejaVu Serif"]
plt.rcParams["axes.unicode_minus"] = False
# ==================== MAT load (v7 / v7.3 auto) ====================

def load_mat_auto(path):
    try:
        mat = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
        return {"__backend__": "scipy", **mat}
    except Exception as e:
        msg = str(e).lower()
        if ("version 73" in msg) or ("unknown mat file type" in msg) or ("matlab 7.3" in msg):
            try:
                import h5py  # noqa
            except Exception:
                raise RuntimeError(
                    f"Failed to read '{path}' via scipy (likely v7.3). "
                    f"Please `pip install h5py`.\nOriginal error: {e}"
                )
            return load_mat_v73_h5py(path)
        raise


def load_mat_v73_h5py(path):
    """
    NOTE: This is a *numeric-friendly* loader.
    For MATLAB cell/struct string fields, v7.3 stores refs -> hard to fully decode automatically.
    If you rely on MAT strings for label names and this loader is used, please provide label_map.csv.
    """
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


def pick_first_existing(d, keys):
    for k in keys:
        if k in d:
            return k
    return None


def ensure_2d_feature_matrix(X):
    X = np.array(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.ndim != 2:
        raise ValueError(f"featureMatrix must be 2D, got shape={X.shape}")
    return X


def to_1d_int(y):
    y = np.array(y).reshape(-1).astype(np.int64)
    _, y_new = np.unique(y, return_inverse=True)
    return y_new.astype(np.int64)


def normalize_fs(fs_vec):
    fs_vec = np.array(fs_vec).reshape(-1).astype(np.float32)
    fs_vec = np.log10(fs_vec + 1.0)
    fs_vec = (fs_vec - fs_vec.mean()) / (fs_vec.std() + 1e-12)
    return fs_vec.astype(np.float32)


def ensure_nchw(spec, N):
    spec = np.array(spec)
    if spec.ndim == 3:
        if spec.shape[2] == N:      # (H,W,N)
            return np.transpose(spec, (2, 0, 1))[:, None, :, :]
        if spec.shape[0] == N:      # (N,H,W)
            return spec[:, None, :, :]
        raise ValueError(f"3D specTensor cannot match N={N}, got {spec.shape}")

    if spec.ndim == 4:
        if spec.shape[3] == N:      # (H,W,C,N)
            return np.transpose(spec, (3, 2, 0, 1))
        if spec.shape[0] == N:      # (N,H,W,C)
            return np.transpose(spec, (0, 3, 1, 2))
        if spec.shape[2] == N:      # (H,W,N,C)
            return np.transpose(spec, (2, 3, 0, 1))
        if spec.shape[1] == N:      # (H,N,W,C)
            return np.transpose(spec, (1, 3, 0, 2))
        raise ValueError(f"4D specTensor cannot match N={N}, got {spec.shape}")

    raise ValueError(f"specTensor ndim must be 3/4, got {spec.ndim}, shape={spec.shape}")


def load_topk_union_indices(topk_path, P):
    T = load_mat_auto(topk_path)
    keys = [k for k in T.keys() if not k.startswith("__")]
    if len(keys) == 0:
        return np.array([], dtype=np.int64)

    def _as_indices(arr):
        arr = np.array(arr)
        if arr.dtype == np.bool_ and arr.size == P:
            return np.where(arr.reshape(-1))[0].astype(np.int64)
        if arr.dtype == object:
            return None
        flat = arr.reshape(-1)
        if np.issubdtype(flat.dtype, np.number):
            flat = flat[np.isfinite(flat)]
            if flat.size == 0:
                return None
            v = np.round(flat).astype(np.int64)
            v = v[v >= 0]
            if v.size == 0:
                return None
            # 1-based?
            if v.min() >= 1 and v.max() <= P:
                v = v - 1
            v = v[(v >= 0) & (v < P)]
            if v.size == 0:
                return None
            return np.unique(v)
        return None

    preferred = ["topk_idx", "topk_indices", "union_idx", "union_indices", "selected_idx", "selected_indices", "idx"]
    for k in preferred:
        if k in T:
            idx = _as_indices(T[k])
            if idx is not None and idx.size > 0:
                return idx

    best, best_sz = None, 0
    for k in keys:
        idx = _as_indices(T[k])
        if idx is not None and idx.size > best_sz:
            best, best_sz = idx, idx.size
    return np.array([], dtype=np.int64) if best is None else best.astype(np.int64)


# ==================== label name mapping ====================

def _mat_char_to_str(x) -> str:
    """Best-effort convert MATLAB char/cell element into Python str (scipy loadmat case)."""
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, (bytes, np.bytes_)):
        try:
            return x.decode("utf-8", errors="ignore")
        except Exception:
            return str(x)
    if isinstance(x, np.ndarray):
        if x.size == 0:
            return ""
        # already unicode array?
        if x.dtype.kind == "U":
            return "".join([str(v) for v in x.reshape(-1).tolist()]).strip()
        if x.dtype.kind == "S":
            parts = []
            for v in x.reshape(-1):
                try:
                    parts.append(v.decode("utf-8", errors="ignore"))
                except Exception:
                    parts.append(str(v))
            return "".join(parts).strip()

        # numeric char codes (uint16 typical)
        if np.issubdtype(x.dtype, np.integer):
            flat = x.reshape(-1).tolist()
            flat = [int(v) for v in flat if int(v) != 0]
            try:
                return "".join([chr(v) for v in flat]).strip()
            except Exception:
                return str(x)

        # object array: take first element
        if x.dtype == object:
            return _mat_char_to_str(x.reshape(-1)[0])
    # fallback
    return str(x).strip()


def load_label_map_csv(path: str):
    if path is None or path == "":
        return None
    if not os.path.isfile(path):
        return None
    df = pd.read_csv(path)
    # tolerant column naming
    cols = [c.lower().strip() for c in df.columns]
    df.columns = cols
    if "class_id" not in df.columns:
        # common alternatives
        for alt in ["id", "class", "label_id", "y"]:
            if alt in df.columns:
                df = df.rename(columns={alt: "class_id"})
                break
    if "name" not in df.columns:
        for alt in ["label", "device", "class_name", "label_name"]:
            if alt in df.columns:
                df = df.rename(columns={alt: "name"})
                break
    if "class_id" not in df.columns or "name" not in df.columns:
        raise ValueError("label_map.csv must have columns: class_id,name (or compatible aliases).")

    mp = {}
    for _, r in df.iterrows():
        try:
            cid = int(r["class_id"])
        except Exception:
            continue
        mp[cid] = str(r["name"])
    return mp


def infer_label_map_from_mat(S, y, file_id):
    """
    Try infer class_id -> name from MAT fields.
    Priority:
      1) label_map / class_names arrays if exist
      2) labelVector (N strings) + y
      3) file_index(labelStr) + file_id + y  (most common labelStr per class)
    """
    # 1) direct class names array (rare)
    for key in ["class_names", "label_names", "names", "label_names_all"]:
        if key in S:
            arr = np.array(S[key]).reshape(-1)
            mp = {}
            for i, v in enumerate(arr.tolist()):
                s = _mat_char_to_str(v)
                if s != "":
                    mp[int(i)] = s
            if len(mp) > 0:
                return mp, f"[LabelMap] from MAT `{key}` (len={len(mp)})"

    # 2) labelVector per sample (your MATLAB screenshot has it)
    for key in ["labelVector", "label_vector", "labelStrVector", "labelStr", "labels_str"]:
        if key in S:
            try:
                arr = np.array(S[key]).reshape(-1)
                if arr.size == y.size:
                    names = [_mat_char_to_str(v) for v in arr.tolist()]
                    mp = {}
                    for c in np.unique(y):
                        c = int(c)
                        idx = np.where(y == c)[0]
                        if idx.size == 0:
                            continue
                        cnt = Counter([names[i] for i in idx if names[i] != ""])
                        if len(cnt) > 0:
                            mp[c] = cnt.most_common(1)[0][0]
                    if len(mp) > 0:
                        return mp, f"[LabelMap] from MAT `{key}` via majority vote per class"
            except Exception:
                pass

    # 3) file_index struct: labelStr per file
    for key in ["file_index", "fileIndex", "fileindex"]:
        if key in S and file_id is not None:
            try:
                fi = S[key]
                # scipy loadmat: fi is ndarray of mat_struct
                fi_arr = np.array(fi).reshape(-1)
                n_files = fi_arr.size

                file_id = np.asarray(file_id).reshape(-1).astype(np.int64)
                # detect fid base
                fid_min, fid_max = int(file_id.min()), int(file_id.max())
                if fid_min == 1 and fid_max == n_files:
                    offset = -1
                elif fid_min == 0 and fid_max == n_files - 1:
                    offset = 0
                else:
                    # guess: if most fids are within 1..n_files, assume 1-based
                    offset = -1 if (fid_min >= 1 and fid_max <= n_files) else 0

                fid_to_label = {}
                for i in range(n_files):
                    obj = fi_arr[i]
                    # mat_struct has attribute names
                    if hasattr(obj, "labelStr"):
                        s = _mat_char_to_str(getattr(obj, "labelStr"))
                    elif hasattr(obj, "labelstr"):
                        s = _mat_char_to_str(getattr(obj, "labelstr"))
                    elif hasattr(obj, "label"):
                        s = _mat_char_to_str(getattr(obj, "label"))
                    else:
                        s = ""
                    fid_to_label[i] = s

                mp = {}
                for c in np.unique(y):
                    c = int(c)
                    idx = np.where(y == c)[0]
                    if idx.size == 0:
                        continue
                    fids = file_id[idx] + offset
                    names = []
                    for f in fids.tolist():
                        if 0 <= int(f) < n_files:
                            s = fid_to_label[int(f)]
                            if s != "":
                                names.append(s)
                    if len(names) > 0:
                        mp[c] = Counter(names).most_common(1)[0][0]
                if len(mp) > 0:
                    return mp, f"[LabelMap] from MAT `{key}.labelStr` via file_id (offset={offset})"
            except Exception:
                pass

    return None, "[LabelMap][Warn] cannot infer from MAT (maybe v7.3 strings). Please provide --label_map label_map.csv."


def build_class_names(num_classes, label_map):
    names = []
    for cid in range(num_classes):
        if label_map is not None and cid in label_map:
            names.append(str(label_map[cid]))
        else:
            names.append(str(cid))
    return names


def shorten_labels(labels, max_len=18):
    if max_len <= 0:
        return labels
    out = []
    for s in labels:
        s = str(s)
        out.append(s if len(s) <= max_len else (s[:max_len - 1] + "…"))
    return out


# ==================== SupCon ====================

class SupConLoss(nn.Module):
    """Supervised Contrastive Loss (single-view). features [B,D], labels [B]."""
    def __init__(self, temperature=0.07, eps=1e-12):
        super().__init__()
        self.temperature = temperature
        self.eps = eps

    def forward(self, features, labels):
        B = features.size(0)
        if B <= 1:
            return features.new_tensor(0.0)

        x = F.normalize(features, dim=1)
        y = labels.view(-1, 1)

        sim = (x @ x.T) / self.temperature
        sim = sim - sim.max(dim=1, keepdim=True).values

        mask = torch.eq(y, y.T).to(sim.dtype)
        eye = torch.eye(B, device=sim.device, dtype=sim.dtype)
        mask_pos = mask * (1.0 - eye)
        logits_mask = (1.0 - eye)

        exp_sim = torch.exp(sim) * logits_mask
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + self.eps)

        pos_cnt = mask_pos.sum(dim=1)
        valid = pos_cnt > 0
        if valid.sum().item() == 0:
            return features.new_tensor(0.0)

        mean_log_prob_pos = (mask_pos * log_prob).sum(dim=1) / (pos_cnt + self.eps)
        return -mean_log_prob_pos[valid].mean()


# ==================== Dataset ====================

class MoEDataset(Dataset):
    def __init__(self, X_all, X_topk, fs_vec, X_spec, X_occ, y, indices):
        self.X_all  = torch.tensor(X_all[indices]).float()
        self.X_topk = torch.tensor(X_topk[indices]).float()
        self.fs     = torch.tensor(fs_vec[indices]).float().unsqueeze(1)
        self.X_spec = torch.tensor(X_spec[indices]).float()
        self.X_occ  = torch.tensor(X_occ[indices]).float()
        self.y      = torch.tensor(y[indices]).long()

    def __len__(self):
        return int(self.y.numel())

    def __getitem__(self, i):
        return self.X_all[i], self.X_topk[i], self.fs[i], self.X_spec[i], self.X_occ[i], self.y[i]


# ==================== Model ====================

class MLPExpert(nn.Module):
    def __init__(self, in_dim, num_classes, hidden=64, emb_dim=64, p_drop=0.2):
        super().__init__()
        self.feat = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(p_drop),
            nn.Linear(hidden, emb_dim), nn.ReLU(),
        )
        self.cls = nn.Linear(emb_dim, num_classes)

    def forward(self, x):
        h = self.feat(x)
        return self.cls(h), h


class Gate(nn.Module):
    def __init__(self, in_dim, hidden, num_exp):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, num_exp)
        )

    def forward(self, x):
        return torch.softmax(self.net(x), dim=1)


class SmallSpecCNN(nn.Module):
    def __init__(self, in_channels, cnn_emb_dim=64):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, cnn_emb_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.fc(self.features(x))


class TFExpert(nn.Module):
    def __init__(self, tf_feat_dim, spec_in_ch, cnn_emb_dim, num_classes, hidden=64, emb_dim=64, p_drop=0.2):
        super().__init__()
        self.cnn = SmallSpecCNN(spec_in_ch, cnn_emb_dim=cnn_emb_dim)
        self.feat = nn.Sequential(
            nn.Linear(tf_feat_dim + cnn_emb_dim, hidden), nn.ReLU(), nn.Dropout(p_drop),
            nn.Linear(hidden, emb_dim), nn.ReLU(),
        )
        self.cls = nn.Linear(emb_dim, num_classes)

    def forward(self, x_tf, x_spec):
        h_spec = self.cnn(x_spec)
        h = self.feat(torch.cat([x_tf, h_spec], dim=1))
        return self.cls(h), h


class ImgExpert(nn.Module):
    def __init__(self, img_in_ch, cnn_emb_dim, num_classes, hidden=64, emb_dim=64, p_drop=0.2):
        super().__init__()
        self.cnn = SmallSpecCNN(img_in_ch, cnn_emb_dim=cnn_emb_dim)
        self.feat = nn.Sequential(
            nn.Linear(cnn_emb_dim, hidden), nn.ReLU(), nn.Dropout(p_drop),
            nn.Linear(hidden, emb_dim), nn.ReLU(),
        )
        self.cls = nn.Linear(emb_dim, num_classes)

    def forward(self, x_img):
        h0 = self.cnn(x_img)
        h  = self.feat(h0)
        return self.cls(h), h


class FeatureFamilyMoE(nn.Module):
    """
    all39 slices:
      time: 0:17 (17)
      freq: 17:22 (5)
      tf  : 22:27 (5)
      inst: 27:39 (12)

    Gate input = [all39, topk_union, fs]
    """
    def __init__(self, num_classes, spec_in_ch, topk_dim, occ_in_ch=1, cnn_emb_dim=64, emb_dim=64, p_drop=0.2, gate_hidden=64):
        super().__init__()
        self.num_experts = 4
        self.register_buffer("expert_mask", torch.ones(self.num_experts))   # [E], 默认全 1  选专家
        self.time_slice = slice(0, 17)
        self.freq_slice = slice(17, 22)
        self.tf_slice   = slice(22, 27)
        self.inst_slice = slice(27, 39)

        self.expert_time = MLPExpert(17, num_classes, hidden=64, emb_dim=emb_dim, p_drop=p_drop)
        # self.expert_freq = MLPExpert(5,  num_classes, hidden=64, emb_dim=emb_dim, p_drop=p_drop)
        self.expert_freq = ImgExpert(occ_in_ch, cnn_emb_dim, num_classes, hidden=64, emb_dim=emb_dim, p_drop=p_drop)  # <-- 替换
        self.expert_tf   = TFExpert(5, spec_in_ch, cnn_emb_dim, num_classes, hidden=64, emb_dim=emb_dim, p_drop=p_drop)
        self.expert_inst = MLPExpert(12, num_classes, hidden=64, emb_dim=emb_dim, p_drop=p_drop)

        gate_in_dim = 39 + topk_dim + 1
        self.gate = Gate(gate_in_dim, hidden=gate_hidden, num_exp=self.num_experts)

    def forward(self, x_all, x_topk, fs_feat, x_spec, x_occ):
        x_time = x_all[:, self.time_slice]
        # x_freq = x_all[:, self.freq_slice]
        x_tf   = x_all[:, self.tf_slice]
        x_inst = x_all[:, self.inst_slice]

        gate_in = torch.cat([x_all, x_topk, fs_feat], dim=1)   
        # gate_w  = self.gate(gate_in)
        gate_w = self.gate(gate_in)                     # [B, E]
        gate_w = gate_w * self.expert_mask.unsqueeze(0)  # mask
        gate_w = gate_w / (gate_w.sum(dim=1, keepdim=True) + 1e-12)

        logits_time, h_time = self.expert_time(x_time)
        if x_occ is None:
            raise ValueError("x_occ is required because expert_freq is ImgExpert now.")
        logits_freq, h_freq = self.expert_freq(x_occ)
        logits_tf,   h_tf   = self.expert_tf(x_tf, x_spec)
        logits_inst, h_inst = self.expert_inst(x_inst)

        expert_logits = [logits_time, logits_freq, logits_tf, logits_inst]
        expert_embs   = [h_time, h_freq, h_tf, h_inst]

        logits_stack = torch.stack(expert_logits, dim=2)     # [B,C,E]
        gate_exp     = gate_w.unsqueeze(1)                   # [B,1,E]
        logits_fused = torch.sum(logits_stack * gate_exp, dim=2)

        emb_stack = torch.stack(expert_embs, dim=2)          # [B,emb,E]
        h_fusion  = torch.sum(emb_stack * gate_exp, dim=2)   # [B,emb]    输出
        return logits_fused, gate_w, {
            "time": h_time,
            "freq": h_freq,
            "tf":   h_tf,
            "inst": h_inst,
            },expert_logits, h_fusion


# ==================== plotting / report helpers ====================

def save_confusion(cm, labels, out_counts_png, out_norm_png, out_npy_prefix):
    os.makedirs(os.path.dirname(out_counts_png) or ".", exist_ok=True)
    np.save(out_npy_prefix + "_counts.npy", cm.astype(np.int64))
    cmn = cm.astype(np.float64) / (cm.sum(axis=1, keepdims=True) + 1e-12)
    np.save(out_npy_prefix + "_norm.npy", cmn.astype(np.float32))

    if plt is None:
        print("[Plot][Warn] matplotlib missing -> skip cm png.")
        return

    tick = labels
    # counts
    plt.figure(figsize=(10, 8))
    if sns is not None:
        sns.heatmap(cm, cmap="Blues", annot=False, xticklabels=tick, yticklabels=tick)
    else:
        plt.imshow(cm, aspect="auto")
        plt.colorbar()
        plt.xticks(range(len(tick)), tick, rotation=45, ha="right")
        plt.yticks(range(len(tick)), tick)
    plt.xlabel("Pred"); plt.ylabel("True"); plt.title("VAL Confusion (Counts)")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(out_counts_png, dpi=300)
    plt.close()

    # norm
    plt.figure(figsize=(10, 8))
    if sns is not None:
        sns.heatmap(cmn, cmap="Blues", vmin=0, vmax=1, annot=False, xticklabels=tick, yticklabels=tick)
    else:
        plt.imshow(cmn, aspect="auto", vmin=0, vmax=1)
        plt.colorbar()
        plt.xticks(range(len(tick)), tick, rotation=45, ha="right")
        plt.yticks(range(len(tick)), tick)
    plt.xlabel("Pred"); plt.ylabel("True"); plt.title("VAL Confusion (Row-norm)")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(out_norm_png, dpi=300)
    plt.close()


def top_confusions(cm, class_names_full, topk=30, min_count=5):
    cm = cm.astype(np.int64)
    cmn = cm.astype(np.float64) / (cm.sum(axis=1, keepdims=True) + 1e-12)
    rows = []
    K = cm.shape[0]
    for i in range(K):
        for j in range(K):
            if i == j:
                continue
            cnt = int(cm[i, j])
            if cnt < min_count:
                continue
            rows.append({
                "true_id": i,
                "pred_id": j,
                "true_name": class_names_full[i],
                "pred_name": class_names_full[j],
                "count": cnt,
                "row_norm": float(cmn[i, j]),
            })
    rows.sort(key=lambda r: (r["row_norm"], r["count"]), reverse=True)
    return rows[:topk]


def plot_expert_load_curve(df_log, out_png):
    if plt is None:
        print("[Plot][Warn] matplotlib missing -> skip expert load curve.")
        return
    cols = [c for c in df_log.columns if c.startswith("gate_val_e")]
    if len(cols) == 0:
        cols = [c for c in df_log.columns if c.startswith("gate_e")]
    if len(cols) == 0:
        return
    plt.figure(figsize=(10, 6))
    for c in cols:
        plt.plot(df_log[c].values, label=c.replace("gate_val_", "").replace("gate_", ""))
    plt.xlabel("Epoch")
    plt.ylabel("Mean Gate Weight (VAL)")
    plt.title("Expert Load Curve (VAL mean gate)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


# ==================== main ====================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mat_all", type=str, default="FeatureMatrix_Indoor_OSU_unified.mat")
    ap.add_argument("--topk_mat", type=str, default="topk_indices_wifi.mat")
    ap.add_argument("--save_dir", type=str, default="./stage2_gate_only_occ")    #./exp_end2end_occ_e1e2e3e4_entropy3e-2 
    ap.add_argument("--label_map", type=str, default="label_map.csv", help="CSV: class_id,name (optional)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--disable_expert",type=str,default="",help="Disable expert by name: time | freq | tf | inst")  

    # ===== stage2: init / gate-only / gate-align =====
    ap.add_argument("--init_from", type=str, default="",
                    help="optional: init model from ckpt (e.g., moe_injected_experts.pth)")
    ap.add_argument("--train_gate_only", action="store_true",
                    help="freeze experts, train only gate")
    ap.add_argument("--gate_align_coef", type=float, default=0.005,    # 0.005
                    help="coef for gate alignment loss (0 disables)")
    ap.add_argument("--gate_align_temp", type=float, default=0.50,
                    help="temperature for expert-loss -> target gate distribution")
    ap.add_argument("--gate_align_level", type=str, default="per_sample", choices=["per_sample", "batch_mean"], 
                    help="align gate per-sample or align batch-mean distribution")

    # training
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=64)  # 128
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.2)

    # losses
    ap.add_argument("--entropy_coef", type=float, default=3e-3)
    ap.add_argument("--aux_weight", type=float, default=0.3)
    ap.add_argument("--supcon_weight", type=float, default=0)
    ap.add_argument("--supcon_temp", type=float, default=0.07)
    ap.add_argument("--emb_dim", type=int, default=64)
    ap.add_argument("--sample_entropy_coef", type=float, default=0.001,
                help="sample-wise gate entropy penalty; >0 encourages sharper per-sample routing")

    # split
    ap.add_argument("--val_ratio", type=float, default=0.2)

    # plotting labels
    ap.add_argument("--max_label_len", type=int, default=18, help="truncate labels in plots; 0 means no truncate")

    args = ap.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    # seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------- Load MAT ----------
    print(f"[Load] {args.mat_all}")
    S = load_mat_auto(args.mat_all)
    print(f"  backend = {S.get('__backend__')}")

    kX = pick_first_existing(S, ["featureMatrix", "feature_matrix", "X"])
    if kX is None:
        raise KeyError("MAT missing featureMatrix")
    X_all = ensure_2d_feature_matrix(S[kX]).astype(np.float32)
    N, P = X_all.shape
    if P != 39:
        raise ValueError(f"Expect all39, got P={P}")

    ky = pick_first_existing(S, ["label_id", "device_id", "y", "labels"])
    if ky is None:
        raise KeyError("MAT missing label_id")
    y = to_1d_int(S[ky])
    num_classes = int(np.unique(y).size)

    kfs = pick_first_existing(S, ["fsVector", "fs", "fs_vec"])
    if kfs is None:
        fs_vec = np.zeros((N,), dtype=np.float32)
    else:
        fs_vec = normalize_fs(S[kfs])
        if fs_vec.shape[0] != N:
            raise ValueError("fsVector length mismatch")

    kfid = pick_first_existing(S, ["file_id", "fileId", "fileID", "file_index_id", "file_idx"])
    file_id = None
    if kfid is not None:
        fid = np.array(S[kfid]).reshape(-1)
        if fid.shape[0] == N:
            file_id = fid.astype(np.int64)
            print(f"  file_id unique = {np.unique(file_id).size}")
        else:
            print("  file_id length mismatch -> ignore")

    kspec = pick_first_existing(S, ["specTensor", "spec", "spec_tensor", "SpecTensor"])
    if kspec is None:
        raise KeyError("MAT missing specTensor")
    X_spec = ensure_nchw(S[kspec], N).astype(np.float32)
    X_spec = (X_spec - X_spec.mean()) / (X_spec.std() + 1e-8)
    print(f"  X_all={X_all.shape}, X_spec={X_spec.shape}, classes={num_classes}")

    kocc = pick_first_existing(S, ["occTensor", "occ_tensor", "densityTensor", "occ"])
    if kocc is None:
        raise KeyError("MAT missing occTensor (I/Q density map)")
    X_occ = ensure_nchw(S[kocc], N).astype(np.float32)
    X_occ = (X_occ - X_occ.mean()) / (X_occ.std() + 1e-8)
    print(" X_occ=", X_occ.shape)

    # ---------- label names ----------
    label_map = load_label_map_csv(args.label_map)
    if label_map is not None:
        print(f"[LabelMap] loaded from CSV: {args.label_map}")
    else:
        mp, info = infer_label_map_from_mat(S, y, file_id)
        print(info)
        label_map = mp

    class_names_full = build_class_names(num_classes, label_map)
    class_names_plot = shorten_labels(class_names_full, max_len=args.max_label_len)

    # ---------- preprocessing ----------
    scaler = StandardScaler()
    X_all_z = scaler.fit_transform(X_all).astype(np.float32)
    # save scaler for exporter (exact reproducibility)
    np.savez(os.path.join(args.save_dir, "preproc_scaler.npz"),
             mean=scaler.mean_.astype(np.float32),
             scale=scaler.scale_.astype(np.float32))
    print("[Preproc] saved scaler to preproc_scaler.npz")

    # ---------- topk union ----------
    print(f"[Load] TOPK={args.topk_mat}")
    topk_union = load_topk_union_indices(args.topk_mat, P).astype(np.int64)
    if topk_union.size > 0:
        print(f"  [TopK] union size={topk_union.size}, min={topk_union.min()}, max={topk_union.max()}")
    else:
        print("  [TopK][Warn] parse failed -> gate uses only all39+fs")
    X_topk = X_all_z[:, topk_union] if topk_union.size > 0 else np.zeros((N, 0), dtype=np.float32)

    # ---------- split ----------
    idx_all = np.arange(N)
    if file_id is not None:
        gss = GroupShuffleSplit(n_splits=1, test_size=args.val_ratio, random_state=args.seed)
        train_idx, val_idx = next(gss.split(idx_all, y, groups=file_id))
        print("[Split] GroupShuffleSplit by file_id.")
    else:
        train_idx, val_idx = train_test_split(idx_all, test_size=args.val_ratio, stratify=y, random_state=args.seed)
        print("[Split] Stratified random split.")

    # ---------- dataset ----------
    train_ds = MoEDataset(X_all_z, X_topk, fs_vec, X_spec, X_occ, y, train_idx)
    val_ds   = MoEDataset(X_all_z, X_topk, fs_vec, X_spec, X_occ, y, val_idx)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0)

    # ---------- model ----------
    spec_in_ch = X_spec.shape[1]
    topk_dim = X_topk.shape[1]

    occ_in_ch = int(X_occ.shape[1])

    model = FeatureFamilyMoE(
        num_classes=num_classes,
        spec_in_ch=spec_in_ch,
        occ_in_ch=occ_in_ch,
        topk_dim=topk_dim,
        cnn_emb_dim=64,
        emb_dim=args.emb_dim,
        p_drop=args.dropout,
        gate_hidden=64
    ).to(device)

    expert_name_to_idx = {
    "time": 0,
    "freq": 1,
    "tf":   2,
    "inst": 3
    }

    # ===== init from ckpt (optional) =====
    if args.init_from and os.path.isfile(args.init_from):
        print(f"[Init] load from: {args.init_from}")
        ckpt = torch.load(args.init_from, map_location="cpu")
        if isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
            state = ckpt["model"]
        else:
            state = ckpt

        # 可选保险：不让 ckpt 里的 expert_mask 覆盖当前运行时设置
        if "expert_mask" in state:
            state.pop("expert_mask")

        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"[Init] missing={len(missing)}, unexpected={len(unexpected)}")

    # ===== ablation: disable experts AFTER loading ckpt =====
    if args.disable_expert:
        disable_list = [x.strip().lower() for x in args.disable_expert.split(",") if x.strip()]
        for ex_name in disable_list:
            if ex_name not in expert_name_to_idx:
                raise ValueError(f"Unknown expert in --disable_expert: {ex_name}")
            ei = expert_name_to_idx[ex_name]
            print(f"[Ablation] Disable expert: {ex_name} (idx={ei})")
            model.expert_mask[ei] = 0.0

    print("[Ablation] expert_names =", list(expert_name_to_idx.keys()))
    print("[Ablation] expert_mask  =", model.expert_mask.detach().cpu().numpy())

    # ===== freeze experts, train only gate =====
    if args.train_gate_only:
        print("[GateOnly] Freeze experts; train only gate.")
        for name, p in model.named_parameters():
            p.requires_grad = name.startswith("gate.")
        # gate-only 时 aux_loss 对 gate 没梯度，纯浪费
        if args.aux_weight != 0:
            print("[GateOnly] aux_weight forced to 0.")
            args.aux_weight = 0.0

    criterion = nn.CrossEntropyLoss()
    supcon = SupConLoss(temperature=args.supcon_temp).to(device)
    # optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    print(f"[Optim] trainable params = {sum(p.numel() for p in params)}")


    # ---------- train ----------
    best_acc = -1.0
    best_path = os.path.join(args.save_dir, "moe_tfcnn_best.pth")
    log_rows = []

    print(f"[Train] save_dir={args.save_dir}")
    print(f"[Model] gate_in={39}+{topk_dim}+1={39+topk_dim+1}, experts={model.num_experts}")

    # expert name order follows forward(): [time, freq, tf, inst]
    expert_names = ["time", "freq", "tf", "inst"]
    if model.num_experts != len(expert_names):
        expert_names = [f"e{i+1}" for i in range(model.num_experts)]

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()

        sum_loss = 0.0
        sum_correct = 0
        sum_n = 0
        sum_correct_exp = np.zeros((model.num_experts,), dtype=np.int64)
        sum_sup = 0.0
        sup_cnt = 0

        # track mean gate (train)
        gate_sum_train = torch.zeros(model.num_experts, dtype=torch.float32, device=device)

        for xb_all, xb_topk, xfs, xb_spec, xb_occ, yb in train_loader:
            xb_all  = xb_all.to(device)
            xb_topk = xb_topk.to(device)
            xfs     = xfs.to(device)
            xb_spec = xb_spec.to(device)
            xb_occ  = xb_occ.to(device)
            yb      = yb.to(device)

            optimizer.zero_grad()
            logits, gate_w, _, expert_logits, h_fusion = model(xb_all, xb_topk, xfs, xb_spec, xb_occ)

            # per-expert train acc (each expert's own head)
            with torch.no_grad():
                for ei, le in enumerate(expert_logits):
                    sum_correct_exp[ei] += int((le.argmax(dim=1) == yb).sum().item())

            loss_main = criterion(logits, yb)

            if args.aux_weight > 0:
                loss_aux = 0.0
                for le in expert_logits:
                    loss_aux = loss_aux + criterion(le, yb)
                loss_aux = loss_aux / len(expert_logits)
            else:
                loss_aux = 0.0

            # entropy on mean gate (batch)
            mean_w = gate_w.mean(dim=0)
            entropy = -(mean_w * torch.log(mean_w + 1e-12)).sum()
            
            # sample-level entropy
            entropy_sample = -(gate_w * torch.log(gate_w + 1e-12)).sum(dim=1).mean()

            loss_sup = supcon(h_fusion, yb)
            if torch.isfinite(loss_sup):
                sum_sup += float(loss_sup.item())
                sup_cnt += 1
            else:
                loss_sup = torch.zeros([], device=yb.device, dtype=logits.dtype)

            # ===== gate alignment (optional) =====
            if args.gate_align_coef > 0:
                # 用“每个专家对当前样本的 CE loss”生成一个 target 分布：loss 越小权重越大
                with torch.no_grad():
                    per_exp_loss = []
                    for le in expert_logits:  # list of [B,C]
                        per_exp_loss.append(F.cross_entropy(le, yb, reduction="none"))  # [B]
                    L = torch.stack(per_exp_loss, dim=1)  # [B,E]
                    target = torch.softmax(-L / max(args.gate_align_temp, 1e-6), dim=1)  # [B,E]
                    if args.gate_align_level == "batch_mean":    
                        target = target.mean(dim=0, keepdim=True).expand_as(gate_w)

                gate_align = -(target * torch.log(gate_w + 1e-12)).sum(dim=1).mean()
            else:
                # gate_align = 0.0
                gate_align = torch.zeros([], device=yb.device, dtype=logits.dtype)

            loss = (
                    loss_main 
                    + args.aux_weight * loss_aux 
                    + args.supcon_weight * loss_sup 
                    + args.gate_align_coef * gate_align 
                    + args.sample_entropy_coef * entropy_sample   # 惩罚单样本高熵
                    - args.entropy_coef * entropy                 # 保持全局负载均衡
                    )
            loss.backward()
            optimizer.step()

            bs = int(yb.numel())
            sum_loss += float(loss.item()) * bs
            sum_correct += int((logits.argmax(dim=1) == yb).sum().item())
            sum_n += bs
            gate_sum_train += gate_w.sum(dim=0)

        train_loss = sum_loss / max(1, sum_n)
        train_acc = sum_correct / max(1, sum_n)
        train_acc_exp = (sum_correct_exp / max(1, sum_n)).astype(np.float64)
        supcon_mean = sum_sup / max(1, sup_cnt)
        gate_train = (gate_sum_train / max(1, sum_n)).detach().cpu().numpy()

        # ---------- val ----------
        model.eval()
        val_correct = 0
        val_n = 0
        val_correct_exp = np.zeros((model.num_experts,), dtype=np.int64)
        gate_sum_val = torch.zeros(model.num_experts, dtype=torch.float32)

        val_credit = torch.zeros(model.num_experts, device=device)     # 每个专家拿到的“正确功劳”总和
        val_gate_mass = torch.zeros(model.num_experts, device=device)  # 每个专家被 gate 分配的总权重

        # ===== 路由后的“专家条件准确率”统计 =====
        val_route_cnt = torch.zeros(model.num_experts, device=device)   # 每个专家被分配到的样本数(硬路由)
        val_route_cor = torch.zeros(model.num_experts, device=device)   # 硬路由下该专家预测正确的样本数

        # ===== soft 使用准确率（可选，更平滑）=====
        val_soft_mass = torch.zeros(model.num_experts, device=device)   # Σ gate_w
        val_soft_cor  = torch.zeros(model.num_experts, device=device)   # Σ gate_w * 1(expert_pred==y)


        # for confusion + gate profile
        all_true = []
        all_pred = []

        # per-class gate mean on val:
        gate_sum_cls = np.zeros((num_classes, model.num_experts), dtype=np.float64)
        cnt_cls = np.zeros((num_classes,), dtype=np.int64)
        cor_cls = np.zeros((num_classes,), dtype=np.int64)

        with torch.no_grad():
            for xb_all, xb_topk, xfs, xb_spec, xb_occ, yb in val_loader:
                xb_all  = xb_all.to(device)
                xb_topk = xb_topk.to(device)
                xfs     = xfs.to(device)
                xb_spec = xb_spec.to(device)
                xb_occ  = xb_occ.to(device)
                yb      = yb.to(device)
           
                logits, gate_w, _, expert_logits, _ = model(xb_all, xb_topk, xfs, xb_spec, xb_occ)        
                pred = logits.argmax(dim=1)

                # --------- per-expert standalone acc (你已有的 exp_val_acc) ----------
                exp_pred = torch.stack([le.argmax(dim=1) for le in expert_logits], dim=1)  # [B,E]
                for ei in range(model.num_experts):
                    val_correct_exp[ei] += int((exp_pred[:, ei] == yb).sum().item())

                # --------- Hard 路由：gate 把样本交给哪个专家负责 ----------
                route = gate_w.argmax(dim=1)  # [B]
                for ei in range(model.num_experts):
                    m = (route == ei)
                    if m.any():
                        val_route_cnt[ei] += m.sum()
                        val_route_cor[ei] += (exp_pred[m, ei] == yb[m]).float().sum()

                # --------- Soft 使用准确率（可选）----------
                exp_correct_mat = (exp_pred == yb.unsqueeze(1)).float()         # [B,E]
                val_soft_mass += gate_w.sum(dim=0)                               # [E]
                val_soft_cor  += (gate_w * exp_correct_mat).sum(dim=0)           # [E]

                bs = int(yb.numel())
                val_correct += int((pred == yb).sum().item())
                val_n += bs
                gate_sum_val += gate_w.sum(dim=0).cpu()

                correct = (pred == yb).float()                 # [B]
                val_credit += (gate_w * correct.unsqueeze(1)).sum(dim=0)   # [E]
                val_gate_mass += gate_w.sum(dim=0)                           # [E]

                all_true.append(yb.cpu().numpy())
                all_pred.append(pred.cpu().numpy())

                gw = gate_w.cpu().numpy()
                yt = yb.cpu().numpy()
                pr = pred.cpu().numpy()
                for i in range(bs):
                    c = int(yt[i])
                    gate_sum_cls[c] += gw[i]
                    cnt_cls[c] += 1
                    if int(pr[i]) == c:
                        cor_cls[c] += 1

        val_acc_exp = (val_correct_exp / max(1, val_n)).astype(np.float64)
        val_acc = val_correct / max(1, val_n)
        gate_val = (gate_sum_val / max(1, val_n)).numpy()

        eps = 1e-12
        val_route_acc = (val_route_cor / (val_route_cnt + eps)).detach().cpu().numpy()  # [E]
        val_route_cov = (val_route_cnt / max(1, val_n)).detach().cpu().numpy()          # [E]
        val_soft_acc  = (val_soft_cor  / (val_soft_mass + eps)).detach().cpu().numpy() # [E]

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_path)

        # log row
        row = {
            "epoch": epoch,
            "loss": train_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "supcon": supcon_mean,
        }
        for i, nm in enumerate(expert_names):
            row[f"val_route_acc_{nm}"] = float(val_route_acc[i])
            row[f"val_route_cov_{nm}"] = float(val_route_cov[i])
            row[f"val_soft_acc_{nm}"]  = float(val_soft_acc[i])

        for i in range(model.num_experts):
            row[f"gate_train_e{i+1}"] = float(gate_train[i])
            row[f"gate_val_e{i+1}"] = float(gate_val[i])
        log_rows.append(row)

        print(
            f"Epoch {epoch:03d} | train_acc={train_acc:.3f} val_acc={val_acc:.3f} best={best_acc:.3f} | "
            f"exp_val_acc={np.round(val_acc_exp,3)} | "
            f"route_acc={np.round(val_route_acc,3)} cov={np.round(val_route_cov,3)} soft_acc={np.round(val_soft_acc,3)} | "
            f"supcon={supcon_mean:.4f} | "
            f"gate_train={np.round(gate_train,3)} gate_val={np.round(gate_val,3)} | "
            f"time={time.time()-t0:.1f}s"
        )

    print(f"\nBest val acc = {best_acc:.6f}")
    print(f"Best ckpt saved to: {best_path}")

    # ---------- save train log ----------
    df_log = pd.DataFrame(log_rows)
    df_log.to_csv(os.path.join(args.save_dir, "train_log.csv"), index=False)
    print("Saved:", os.path.join(args.save_dir, "train_log.csv"))

    # expert load curve
    plot_expert_load_curve(df_log, os.path.join(args.save_dir, "expert_load_curve.png"))
    print("Saved:", os.path.join(args.save_dir, "expert_load_curve.png"))

    # ---------- final val reports using best ckpt ----------
    model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()

    all_true = []
    all_pred = []
    gate_sum_cls = np.zeros((num_classes, model.num_experts), dtype=np.float64)
    cnt_cls = np.zeros((num_classes,), dtype=np.int64)
    cor_cls = np.zeros((num_classes,), dtype=np.int64)

    # ===== route/soft/exp 的统计量
    best_val_n = 0
    best_exp_cor   = torch.zeros(model.num_experts, device=device)
    best_route_cnt = torch.zeros(model.num_experts, device=device)
    best_route_cor = torch.zeros(model.num_experts, device=device)
    best_soft_mass = torch.zeros(model.num_experts, device=device)
    best_soft_cor  = torch.zeros(model.num_experts, device=device)

    with torch.no_grad():
        for xb_all, xb_topk, xfs, xb_spec, xb_occ, yb in val_loader:
            xb_all  = xb_all.to(device)
            xb_topk = xb_topk.to(device)
            xfs     = xfs.to(device)
            xb_spec = xb_spec.to(device)
            xb_occ  = xb_occ.to(device)
            yb      = yb.to(device)

            logits, gate_w, _, expert_logits, _ = model(xb_all, xb_topk, xfs, xb_spec, xb_occ)
            pred = logits.argmax(dim=1)

            # ===== 1) 混淆矩阵需要的 y_true/y_pred（融合输出）=====
            all_true.append(yb.cpu().numpy())
            all_pred.append(pred.cpu().numpy())

            # ===== 2) 专家预测（每个专家自己的 argmax）=====
            exp_pred = torch.stack([le.argmax(dim=1) for le in expert_logits], dim=1)  # [B,E]
            best_exp_cor += (exp_pred == yb.unsqueeze(1)).float().sum(dim=0)

            # ===== 3) Hard route：gate 把样本交给哪个专家 =====
            route = gate_w.argmax(dim=1)
            for ei in range(model.num_experts):
                m = (route == ei)
                if m.any():
                    best_route_cnt[ei] += m.sum()
                    best_route_cor[ei] += (exp_pred[m, ei] == yb[m]).float().sum()

            # ===== 4) Soft 使用准确率 =====
            exp_correct_mat = (exp_pred == yb.unsqueeze(1)).float()
            best_soft_mass += gate_w.sum(dim=0)
            best_soft_cor  += (gate_w * exp_correct_mat).sum(dim=0)

            best_val_n += int(yb.numel())

            
            # ===== 5) gate_profile_val.csv 统计（按类聚合 gate 均值 + 类准确率）=====
            gw = gate_w.cpu().numpy()
            yt = yb.cpu().numpy()
            pr = pred.cpu().numpy()
            bs = yt.shape[0]
            for i in range(bs):
                c = int(yt[i])
                gate_sum_cls[c] += gw[i]
                cnt_cls[c] += 1
                if int(pr[i]) == c:
                    cor_cls[c] += 1

    eps = 1e-12
    best_route_acc = (best_route_cor/(best_route_cnt+eps)).detach().cpu().numpy()
    best_route_cov = (best_route_cnt/max(1,best_val_n)).detach().cpu().numpy()
    best_soft_acc  = (best_soft_cor /(best_soft_mass+eps)).detach().cpu().numpy()
    best_exp_acc   = (best_exp_cor  /max(1,best_val_n)).detach().cpu().numpy()

    print("[BestCkpt] exp_acc =", np.round(best_exp_acc,3))
    print("[BestCkpt] route_acc=", np.round(best_route_acc,3), "cov=", np.round(best_route_cov,3), "soft_acc=", np.round(best_soft_acc,3))

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)

    cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))

    save_confusion(
        cm=cm,
        labels=class_names_plot,
        out_counts_png=os.path.join(args.save_dir, "val_confusion_counts.png"),
        out_norm_png=os.path.join(args.save_dir, "val_confusion_norm.png"),
        out_npy_prefix=os.path.join(args.save_dir, "val_confusion")
    )
    print("Saved:", os.path.join(args.save_dir, "val_confusion_counts.png"))
    print("Saved:", os.path.join(args.save_dir, "val_confusion_norm.png"))

    # gate_profile_val.csv (with class_name)
    rows = []
    for c in range(num_classes):
        if cnt_cls[c] <= 0:
            mean_gate = np.zeros((model.num_experts,), dtype=np.float64)
            acc_c = 0.0
        else:
            mean_gate = gate_sum_cls[c] / float(cnt_cls[c])
            acc_c = float(cor_cls[c] / max(1, cnt_cls[c]))

        r = {
            "class_id": c,
            "class_name": class_names_full[c],
            "val_count": int(cnt_cls[c]),
            "val_acc": acc_c,
        }
        for e in range(model.num_experts):
            r[f"gate_mean_e{e+1}"] = float(mean_gate[e])
        r["top_expert"] = int(np.argmax(mean_gate) + 1) if cnt_cls[c] > 0 else -1
        rows.append(r)

    df_gate = pd.DataFrame(rows).sort_values("class_id")
    df_gate.to_csv(os.path.join(args.save_dir, "gate_profile_val.csv"), index=False)
    print("Saved:", os.path.join(args.save_dir, "gate_profile_val.csv"))

    # top confusions table (with names)
    tc = top_confusions(cm, class_names_full, topk=30, min_count=5)
    pd.DataFrame(tc).to_csv(os.path.join(args.save_dir, "top_confusions_val.csv"), index=False)
    print("Saved:", os.path.join(args.save_dir, "top_confusions_val.csv"))


if __name__ == "__main__":
    main()
