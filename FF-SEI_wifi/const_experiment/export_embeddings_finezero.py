#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export_embeddings_finezero.py

Export embeddings from a trained FineZero model into an NPZ that can be directly used by
`downstream_fssei_fewshot_SNR.py`.

Output NPZ keys:
  - H: [N,D]
  - y: [N]
  - logits / prob / pred
  - file_id (if available)
  - class_names
  - train_idx / val_idx / test_idx (relative to the exported subset)
  - orig_indices

说明：
BT数据集：FeatureMatrix_3.mat    topk_indices.mat
WiFi数据集：FeatureMatrix_Indoor_OSU.mat   topk_indices_wifi.mat

Example:
python export_embeddings_finezero.py \
  --mat_all ../FeatureMatrix_OSU_Stable_WiFi_Wireless_unified.mat  \
  --split_npz ../split_indices_fssei_osu_stable_wireless.npz \
  --ckpt ./experiments/e2_dl_finezero/finezero_cls_best.pth \
  --save_dir ./experiments/e2_dl_finezero/finezero_export \
  --subset all
"""

import os
import json
import argparse
from typing import Optional

import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

CWD_DIR = os.getcwd()


def resolve_path(p: str, base_dir: Optional[str] = None) -> str:
    p = os.path.expanduser(str(p).strip())
    if os.path.isabs(p):
        return os.path.normpath(p)
    return os.path.normpath(os.path.join(base_dir or CWD_DIR, p))


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


def _ensure_spec_nchw(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 3:
        if arr.shape[2] >= arr.shape[0] and arr.shape[2] >= arr.shape[1]:
            arr = np.transpose(arr, (2, 0, 1))
        arr = arr[:, None, :, :]
        return arr.astype(np.float32)
    if arr.ndim == 4:
        if arr.shape[-1] == 1:
            arr = np.transpose(arr, (0, 3, 1, 2))
            return arr.astype(np.float32)
        if arr.shape[2] == 1:
            arr = np.transpose(arr, (3, 2, 0, 1))
            return arr.astype(np.float32)
        if arr.shape[-1] > 8 and arr.shape[0] > 8:
            arr = np.transpose(arr, (3, 2, 0, 1))
            return arr.astype(np.float32)
        return arr.astype(np.float32)
    raise ValueError(f"Unsupported specTensor shape: {arr.shape}")


def load_featurematrix3_spec(mat_path: str):
    S = load_mat_auto(mat_path)
    kspec = pick_first_existing(S, ["specTensor", "spec_tensor", "X_spec", "spec"])
    if kspec is None:
        raise KeyError("MAT missing specTensor/spec_tensor/X_spec/spec")
    ky = pick_first_existing(S, ["label_id", "device_id", "y", "labels"])
    if ky is None:
        raise KeyError("MAT missing label_id/device_id/y/labels")
    spec = _ensure_spec_nchw(S[kspec])
    y = to_1d_int(S[ky])
    if spec.shape[0] != len(y):
        raise ValueError(f"specTensor N mismatch labels: {spec.shape[0]} vs {len(y)}")
    file_id = None
    kfid = pick_first_existing(S, ["file_id", "fileId", "fileID", "file_idx", "file_index_id"])
    if kfid is not None:
        tmp = np.asarray(S[kfid]).reshape(-1)
        if len(tmp) == len(y):
            file_id = tmp.astype(np.int64)
    return spec.astype(np.float32), y.astype(np.int64), file_id, S


class ConvBlock(nn.Module):
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


class FineZeroEncoder(nn.Module):
    def __init__(self, in_ch=1, feat_dim=1024, width=32, dropout=0.0):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, width, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
        )
        self.b1 = ConvBlock(width, width * 2, stride=2)
        self.b2 = ConvBlock(width * 2, width * 4, stride=2)
        self.b3 = ConvBlock(width * 4, width * 8, stride=2)
        self.b4 = ConvBlock(width * 8, width * 8, stride=2)
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


class FineZeroClassifier(nn.Module):
    def __init__(self, encoder, feat_dim=1024, num_classes=10, dropout=0.0):
        super().__init__()
        self.encoder = encoder
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.head = nn.Linear(feat_dim, num_classes)
    def forward_features(self, x):
        return self.encoder(x)
    def forward(self, x):
        h = self.forward_features(x)
        return self.head(self.drop(h))


class SpecSubsetDataset(Dataset):
    def __init__(self, X, y, indices, mean, std):
        self.X = X[indices].astype(np.float32)
        self.y = y[indices].astype(np.int64)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.mean = float(mean)
        self.std = float(std)
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, idx):
        x = self.X[idx]
        x = (x - self.mean) / max(self.std, 1e-6)
        return torch.from_numpy(x), int(self.y[idx]), int(self.indices[idx])


@torch.no_grad()
def softmax_np(x, axis=1):
    x = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=axis, keepdims=True)


@torch.no_grad()
def forward_export(model, loader, device):
    model.eval()
    Hs, logits_all, ys, orig_idx = [], [], [], []
    for xb, yb, ib in loader:
        xb = xb.to(device, non_blocking=True)
        h = model.forward_features(xb)
        logits = model.head(model.drop(h))
        Hs.append(h.cpu().numpy().astype(np.float32))
        logits_all.append(logits.cpu().numpy().astype(np.float32))
        ys.append(np.asarray(yb, dtype=np.int64))
        orig_idx.append(np.asarray(ib, dtype=np.int64))
    H = np.concatenate(Hs, axis=0)
    logits = np.concatenate(logits_all, axis=0)
    y = np.concatenate(ys, axis=0)
    idx = np.concatenate(orig_idx, axis=0)
    pred = logits.argmax(axis=1).astype(np.int64)
    prob = softmax_np(logits, axis=1).astype(np.float32)
    return H, logits, prob, pred, y, idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mat_all", type=str, required=True)
    ap.add_argument("--split_npz", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--save_dir", type=str, required=True)
    ap.add_argument("--subset", type=str, default="test", choices=["train", "val", "test", "all"])
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=4)
    args = ap.parse_args()

    args.mat_all = resolve_path(args.mat_all, CWD_DIR)
    args.split_npz = resolve_path(args.split_npz, CWD_DIR)
    args.ckpt = resolve_path(args.ckpt, CWD_DIR)
    args.save_dir = resolve_path(args.save_dir, CWD_DIR)
    os.makedirs(args.save_dir, exist_ok=True)

    X, y, file_id, S = load_featurematrix3_spec(args.mat_all)
    tr_idx, va_idx, te_idx = load_split_npz(args.split_npz)

    if args.subset == "train":
        subset_idx = tr_idx
    elif args.subset == "val":
        subset_idx = va_idx
    elif args.subset == "test":
        subset_idx = te_idx
    else:
        subset_idx = np.arange(len(y), dtype=np.int64)

    ck = torch.load(args.ckpt, map_location="cpu")
    ck_args = ck.get("args", {})
    stats = ck.get("stats", {})
    mean = float(stats.get("mean", float(X[tr_idx].mean())))
    std = float(stats.get("std", float(X[tr_idx].std() + 1e-6)))
    num_classes = int(ck.get("num_classes", int(np.unique(y).size)))
    class_names = ck.get("class_names", [str(i) for i in range(num_classes)])

    feat_dim = int(ck_args.get("feat_dim", 1024))
    width = int(ck_args.get("width", 32))
    dropout = float(ck_args.get("dropout", 0.0))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = FineZeroEncoder(in_ch=X.shape[1], feat_dim=feat_dim, width=width, dropout=dropout)
    model = FineZeroClassifier(encoder, feat_dim=feat_dim, num_classes=num_classes, dropout=dropout)
    state = ck["model_state"] if "model_state" in ck else ck
    model.load_state_dict(state, strict=True)
    model.to(device)

    ds = SpecSubsetDataset(X, y, subset_idx, mean, std)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, pin_memory=True)

    H, logits, prob, pred, y_sub, orig_idx = forward_export(model, dl, device)

    out_npz = os.path.join(args.save_dir, f"finezero_embeddings_{args.subset}.npz")
    save_dict = {
        "H": H.astype(np.float32),
        "y": y_sub.astype(np.int64),
        "logits": logits.astype(np.float32),
        "prob": prob.astype(np.float32),
        "pred": pred.astype(np.int64),
        "orig_indices": orig_idx.astype(np.int64),
        "class_names": np.array(class_names, dtype=object),
    }
    if file_id is not None:
        save_dict["file_id"] = file_id[orig_idx].astype(np.int64)

    n_sub = len(orig_idx)
    if args.subset == "train":
        save_dict["train_idx"] = np.arange(n_sub, dtype=np.int64)
        save_dict["val_idx"] = np.array([], dtype=np.int64)
        save_dict["test_idx"] = np.array([], dtype=np.int64)
    elif args.subset == "val":
        save_dict["train_idx"] = np.array([], dtype=np.int64)
        save_dict["val_idx"] = np.arange(n_sub, dtype=np.int64)
        save_dict["test_idx"] = np.array([], dtype=np.int64)
    elif args.subset == "test":
        save_dict["train_idx"] = np.array([], dtype=np.int64)
        save_dict["val_idx"] = np.array([], dtype=np.int64)
        save_dict["test_idx"] = np.arange(n_sub, dtype=np.int64)
    else:
        save_dict["train_idx"] = np.nonzero(np.isin(orig_idx, tr_idx))[0].astype(np.int64)
        save_dict["val_idx"] = np.nonzero(np.isin(orig_idx, va_idx))[0].astype(np.int64)
        save_dict["test_idx"] = np.nonzero(np.isin(orig_idx, te_idx))[0].astype(np.int64)

    np.savez_compressed(out_npz, **save_dict)
    print("[Saved]", out_npz)

    with open(os.path.join(args.save_dir, f"summary_export_{args.subset}.json"), "w", encoding="utf-8") as f:
        json.dump({
            "subset": args.subset,
            "num_exported": int(n_sub),
            "embedding_dim": int(H.shape[1]),
            "npz": os.path.basename(out_npz),
            "note": "Use this NPZ directly with downstream_fssei_fewshot_SNR.py. Prefer subset=test or subset=val to avoid leakage.",
        }, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()