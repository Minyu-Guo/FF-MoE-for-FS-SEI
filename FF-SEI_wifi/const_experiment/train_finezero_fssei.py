#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_finezero_fssei.py

FineZero baseline for your FS-SEI pipeline.

Definition aligned with SA2SEI paper:
- no pre-training phase
- no knowledge transfer
- train the feature extractor (encoder) and classifier from scratch
- use the original labeled target dataset directly

This script matches your current engineering flow:
  FeatureMatrix_3.mat + split_indices_fssei.npz
    -> supervised training from scratch on train split
    -> best checkpoint by val/test accuracy
    -> export embeddings by a separate script
    -> reuse downstream_fssei_fewshot_SNR.py for strict few-shot evaluation

说明：
BT数据集：FeatureMatrix_3.mat    topk_indices.mat
WiFi数据集：FeatureMatrix_Indoor_OSU.mat   topk_indices_wifi.mat

Outputs:
  save_dir/
    - finezero_cls_best.pth
    - train_log.csv
    - summary.json

Example:
python train_finezero_fssei.py \
  --mat_all ../FeatureMatrix_OSU_Stable_WiFi_Wireless_unified.mat  \
  --split_npz ../split_indices_fssei_osu_stable_wireless.npz \
  --save_dir ./experiments/e2_dl_finezero \
  --ft_epochs 30 \
  --batch_size 128 \
  --eval_split val
"""

import os
import json
import csv
import copy
import argparse
from typing import Optional

import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    from sklearn.metrics import f1_score
except Exception:
    f1_score = None

CWD_DIR = os.getcwd()


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
            return [mp.get(i, str(i)) for i in range(num_classes)]
    except Exception:
        pass
    return class_names


def _ensure_spec_nchw(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 2:
        raise ValueError(f"specTensor shape seems wrong: {arr.shape}")

    if arr.ndim == 3:
        if arr.shape[0] > 8 and arr.shape[1] > 8 and arr.shape[2] > 8:
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


class SpecClsDataset(Dataset):
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
        return torch.from_numpy(x), int(self.y[idx])


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
    """
    Keep the same encoder family as SA2SEI-like so the comparison isolates
    the effect of pre-training / knowledge transfer.
    """
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


@torch.no_grad()
def evaluate_cls(model, loader, device):
    model.eval()
    total_loss = 0.0
    ys, ps = [], []
    n = 0
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        logits = model(xb)
        loss = F.cross_entropy(logits, yb)
        total_loss += float(loss.item()) * yb.size(0)
        pred = torch.argmax(logits, dim=1)
        ys.append(yb.detach().cpu().numpy())
        ps.append(pred.detach().cpu().numpy())
        n += yb.size(0)
    ys = np.concatenate(ys)
    ps = np.concatenate(ps)
    acc = float((ys == ps).mean())
    macro_f1 = float(f1_score(ys, ps, average="macro")) if f1_score is not None else float("nan")
    return {"loss": total_loss / max(1, n), "acc": acc, "macro_f1": macro_f1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mat_all", type=str, required=True)
    ap.add_argument("--split_npz", type=str, required=True)
    ap.add_argument("--save_dir", type=str, required=True)
    ap.add_argument("--label_map_csv", type=str, default="")

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=4)

    ap.add_argument("--feat_dim", type=int, default=1024)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.0)

    ap.add_argument("--ft_epochs", type=int, default=100)
    ap.add_argument("--ft_lr", type=float, default=1e-3)
    ap.add_argument("--ft_wd", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--eval_split", type=str, default="val", choices=["val", "test"])

    args = ap.parse_args()

    set_seed(args.seed)
    args.mat_all = resolve_path(args.mat_all, CWD_DIR)
    args.split_npz = resolve_path(args.split_npz, CWD_DIR)
    args.save_dir = resolve_path(args.save_dir, CWD_DIR)
    os.makedirs(args.save_dir, exist_ok=True)

    X, y, file_id, S = load_featurematrix3_spec(args.mat_all)
    tr_idx, va_idx, te_idx = load_split_npz(args.split_npz)
    num_classes = int(np.unique(y).size)
    class_names = load_label_map_csv(args.label_map_csv, num_classes)

    # normalize using train split only
    mean = float(X[tr_idx].mean())
    std = float(X[tr_idx].std() + 1e-6)
    stats = {"mean": mean, "std": std}

    ds_tr = SpecClsDataset(X, y, tr_idx, mean, std)
    ds_va = SpecClsDataset(X, y, va_idx, mean, std)
    ds_te = SpecClsDataset(X, y, te_idx, mean, std)

    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True,
                       num_workers=args.num_workers, pin_memory=True)
    dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False,
                       num_workers=args.num_workers, pin_memory=True)
    dl_te = DataLoader(ds_te, batch_size=args.batch_size, shuffle=False,
                       num_workers=args.num_workers, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = FineZeroEncoder(in_ch=X.shape[1], feat_dim=args.feat_dim, width=args.width, dropout=args.dropout)
    model = FineZeroClassifier(encoder, feat_dim=args.feat_dim, num_classes=num_classes, dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.ft_lr, weight_decay=args.ft_wd)

    ckpt_path = os.path.join(args.save_dir, "finezero_cls_best.pth")
    eval_loader = dl_va if args.eval_split == "val" else dl_te

    best_eval = -1.0
    best_state = None
    bad = 0
    logs = []

    for ep in range(1, args.ft_epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_seen = 0

        for xb, yb in dl_tr:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            total_loss += float(loss.item()) * yb.size(0)
            total_correct += int((logits.argmax(dim=1) == yb).sum().item())
            total_seen += yb.size(0)

        tr_loss = total_loss / max(1, total_seen)
        tr_acc = total_correct / max(1, total_seen)
        ev = evaluate_cls(model, eval_loader, device)
        row = {
            "epoch": ep,
            "train_loss": tr_loss,
            "train_acc": tr_acc,
            "eval_loss": ev["loss"],
            "eval_acc": ev["acc"],
            "eval_macro_f1": ev["macro_f1"],
        }
        logs.append(row)
        print(f"[FineZero {ep:03d}] train_loss={tr_loss:.4f}, train_acc={tr_acc:.4f} | eval_loss={ev['loss']:.4f}, eval_acc={ev['acc']:.4f}, eval_f1={ev['macro_f1']:.4f}")

        if ev["acc"] > best_eval:
            best_eval = ev["acc"]
            best_state = copy.deepcopy(model.state_dict())
            bad = 0
            torch.save({
                "model_state": best_state,
                "args": vars(args),
                "stats": stats,
                "num_classes": num_classes,
                "class_names": class_names,
                "input_shape": list(X.shape[1:]),
            }, ckpt_path)
        else:
            bad += 1
            if bad >= args.patience:
                print("[EarlyStop] patience reached.")
                break

    with open(os.path.join(args.save_dir, "train_log.csv"), "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "train_acc", "eval_loss", "eval_acc", "eval_macro_f1"])
        wr.writeheader()
        wr.writerows(logs)

    if best_state is not None:
        model.load_state_dict(best_state, strict=True)
    test_stats = evaluate_cls(model, dl_te, device)
    print(f"[Test] loss={test_stats['loss']:.4f}, acc={test_stats['acc']:.4f}, macro_f1={test_stats['macro_f1']:.4f}")

    summary = {
        "model_name": "FineZero (same encoder as SA2SEI-like, no pretrain, train from scratch)",
        "paper_alignment": {
            "pretrain_phase": False,
            "knowledge_transfer": False,
            "random_initialization": True,
            "same_encoder_family_as_sa2sei_like": True,
            "input_used": "specTensor",
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