# -*- coding: utf-8 -*-
"""
SEI baseline common utils for FeatureMatrix_3.mat
- Robust MAT loader (scipy + h5py fallback)
- specTensor / label_id parsing
- split_npz loading
- training/eval loops
"""

import os, json, time, random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import scipy.io as sio

try:
    import h5py
except Exception:
    h5py = None

try:
    import pandas as pd
except Exception:
    pd = None

try:
    from sklearn.metrics import f1_score
except Exception:
    f1_score = None


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_mat_auto(mat_path):
    """
    支持 v7(via scipy) / v7.3(via h5py) 的基础读取
    """
    try:
        m = sio.loadmat(mat_path)
        m["__backend__"] = "scipy.io.loadmat"
        return m
    except Exception:
        if h5py is None:
            raise
    # h5py fallback
    out = {}
    with h5py.File(mat_path, "r") as f:
        for k in f.keys():
            out[k] = np.array(f[k])
    out["__backend__"] = "h5py"
    return out


def pick_first_existing(d, keys):
    for k in keys:
        if k in d:
            return k
    return None


def to_1d_int(x):
    x = np.asarray(x).squeeze()
    if x.ndim != 1:
        x = x.reshape(-1)
    x = x.astype(np.int64)
    uniq = np.unique(x)
    # 若标签是 1..C，转成 0..C-1
    if uniq.min() == 1 and uniq.max() == len(uniq):
        x = x - 1
    return x


def ensure_nhw_from_64x64xN(arr, name="tensor"):
    """
    将 (64,64,N) 或 (N,64,64) 统一成 (N,64,64)
    """
    x = np.asarray(arr)
    if x.ndim != 3:
        raise ValueError(f"{name} must be 3D, got {x.shape}")
    if x.shape[0] == 64 and x.shape[1] == 64:
        x = np.transpose(x, (2, 0, 1))
    elif x.shape[1] == 64 and x.shape[2] == 64:
        pass
    else:
        raise ValueError(f"Unrecognized {name} shape={x.shape}")
    return x.astype(np.float32)


def load_featurematrix3_spec_label(mat_path):
    """
    读取 FeatureMatrix_3.mat 中的 specTensor + label_id(labelVector)
    返回:
      X_spec: [N,64,64] float32
      y     : [N] int64 (0-based)
      meta  : dict
    """
    S = load_mat_auto(mat_path)

    kspec = pick_first_existing(S, ["specTensor", "spec_tensor"])
    ky = pick_first_existing(S, ["label_id", "labelVector", "labels", "y"])
    if kspec is None:
        raise KeyError("MAT file missing specTensor")
    if ky is None:
        raise KeyError("MAT file missing label_id/labelVector")

    X_spec = ensure_nhw_from_64x64xN(S[kspec], "specTensor")
    y = to_1d_int(S[ky])

    if len(X_spec) != len(y):
        raise ValueError(f"len(X_spec)={len(X_spec)} != len(y)={len(y)}")

    meta = {
        "mat_backend": str(S.get("__backend__", "unknown")),
        "spec_key": kspec,
        "label_key": ky,
        "num_samples": int(len(y)),
        "num_classes": int(np.unique(y).size),
        "input_hw": list(X_spec.shape[1:]),
    }
    return X_spec, y, meta


def load_indices_from_npz(split_npz):
    Z = np.load(split_npz, allow_pickle=True)
    keys = list(Z.keys())
    candidates = [
        ("train_idx", "val_idx", "test_idx"),
        ("train_indices", "val_indices", "test_indices"),
        ("idx_train", "idx_val", "idx_test"),
    ]
    for a, b, c in candidates:
        if a in Z and b in Z and c in Z:
            return Z[a].astype(np.int64), Z[b].astype(np.int64), Z[c].astype(np.int64)
    raise KeyError(f"Cannot find train/val/test indices in {split_npz}; keys={keys}")


def stratified_split_indices(y, train_ratio=0.7, val_ratio=0.15, seed=42):
    rng = np.random.RandomState(seed)
    y = np.asarray(y)
    tr, va, te = [], [], []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        n = len(idx)
        n_tr = max(1, int(round(n * train_ratio)))
        n_va = max(1, int(round(n * val_ratio)))
        if n_tr + n_va >= n:
            n_tr = max(1, n - 2)
            n_va = 1
        n_te = n - n_tr - n_va
        if n_te <= 0:
            n_te = 1
            if n_tr > 1:
                n_tr -= 1
            else:
                n_va = max(1, n_va - 1)
        tr.extend(idx[:n_tr])
        va.extend(idx[n_tr:n_tr+n_va])
        te.extend(idx[n_tr+n_va:])
    return np.array(tr), np.array(va), np.array(te)


class SpecClsDataset(Dataset):
    def __init__(self, X, y, mean=None, std=None):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)
        if mean is None:
            mean = float(self.X.mean())
        if std is None:
            std = float(self.X.std() + 1e-6)
        self.mean = float(mean)
        self.std = float(std)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        x = (self.X[i] - self.mean) / self.std
        x = torch.from_numpy(x[None, ...])  # [1,H,W]
        y = torch.tensor(self.y[i], dtype=torch.long)
        return x, y


class SpecOnlyDataset(Dataset):
    def __init__(self, X, mean=None, std=None):
        self.X = X.astype(np.float32)
        if mean is None:
            mean = float(self.X.mean())
        if std is None:
            std = float(self.X.std() + 1e-6)
        self.mean = float(mean)
        self.std = float(std)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        x = (self.X[i] - self.mean) / self.std
        x = torch.from_numpy(x[None, ...])  # [1,H,W]
        return x


def make_loaders_for_cls(X, y, tr_idx, va_idx, te_idx, batch_size=128, num_workers=4):
    Xtr, ytr = X[tr_idx], y[tr_idx]
    Xva, yva = X[va_idx], y[va_idx]
    Xte, yte = X[te_idx], y[te_idx]

    mean = float(Xtr.mean())
    std = float(Xtr.std() + 1e-6)

    ds_tr = SpecClsDataset(Xtr, ytr, mean, std)
    ds_va = SpecClsDataset(Xva, yva, mean, std)
    ds_te = SpecClsDataset(Xte, yte, mean, std)

    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    dl_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    dl_te = DataLoader(ds_te, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    stats = {"mean": mean, "std": std}
    return dl_tr, dl_va, dl_te, stats


@torch.no_grad()
def evaluate_cls(model, loader, device):
    model.eval()
    ce = nn.CrossEntropyLoss()
    total, correct = 0, 0
    loss_sum = 0.0
    ys, ps = [], []

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        logits = model(xb)
        loss = ce(logits, yb)

        pred = logits.argmax(dim=1)
        total += yb.numel()
        correct += (pred == yb).sum().item()
        loss_sum += loss.item() * yb.size(0)
        ys.append(yb.cpu().numpy())
        ps.append(pred.cpu().numpy())

    acc = correct / max(1, total)
    if len(ys) > 0 and f1_score is not None:
        y_true = np.concatenate(ys)
        y_pred = np.concatenate(ps)
        macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    else:
        macro_f1 = float("nan")

    return {
        "loss": loss_sum / max(1, total),
        "acc": acc,
        "macro_f1": macro_f1,
    }


def train_one_epoch_cls(model, loader, optimizer, device, scaler=None):
    model.train()
    ce = nn.CrossEntropyLoss()
    total, correct = 0, 0
    loss_sum = 0.0

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        if scaler is None:
            logits = model(xb)
            loss = ce(logits, yb)
            loss.backward()
            optimizer.step()
        else:
            with torch.cuda.amp.autocast():
                logits = model(xb)
                loss = ce(logits, yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        pred = logits.argmax(dim=1)
        total += yb.numel()
        correct += (pred == yb).sum().item()
        loss_sum += loss.item() * yb.size(0)

    return {"loss": loss_sum / max(1, total), "acc": correct / max(1, total)}


def save_logs_csv(logs, path):
    if pd is None:
        np.save(path.replace(".csv", ".npy"), logs, allow_pickle=True)
        return
    pd.DataFrame(logs).to_csv(path, index=False)


def dump_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)