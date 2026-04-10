# -*- coding: utf-8 -*-
"""
Export embeddings + predictions for GLFormer-like TIFS2023 baseline
依赖:
  - sei_mat_utils.py
  - train_glformer_like_tifs2023.py  (提供 LightSEITransformer)

python export_embeddings_glformer_like.py \
  --mat_all ../FeatureMatrix_OSU_Stable_WiFi_Wireless_unified.mat   \
  --split_npz ../split_indices_fssei_osu_stable_wireless.npz \
  --ckpt ./experiments/e2_glformer_like_tifs2023/glformer_like_best.pth \
  --save_dir ./experiments/e2_glformer_like_tifs2023/glformer_test \
  --subset test
"""

import os
import json
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from sei_mat_utils import (
    set_seed, load_featurematrix3_spec_label, load_mat_auto,
    load_indices_from_npz, stratified_split_indices, SpecClsDataset
)
from train_glformer_like_tifs2023 import LightSEITransformer

try:
    from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score, average_precision_score
    from sklearn.preprocessing import label_binarize
except Exception:
    f1_score = None
    confusion_matrix = None
    roc_auc_score = None
    average_precision_score = None
    label_binarize = None


def softmax_np(x, axis=1):
    x = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=axis, keepdims=True)


@torch.no_grad()
def forward_all(model, loader, device):
    model.eval()
    feats_all, logits_all, y_all = [], [], []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        feat = model.forward_features(xb)   # [B,D]
        logits = model.head(feat)           # [B,C]
        feats_all.append(feat.cpu().numpy().astype(np.float32))
        logits_all.append(logits.cpu().numpy().astype(np.float32))
        y_all.append(yb.numpy().astype(np.int64))
    H = np.concatenate(feats_all, axis=0)
    logits = np.concatenate(logits_all, axis=0)
    y = np.concatenate(y_all, axis=0)
    pred = logits.argmax(axis=1).astype(np.int64)
    prob = softmax_np(logits, axis=1).astype(np.float32)
    return H, logits, prob, pred, y


def compute_metrics(y_true, y_pred, prob=None):
    out = {}
    y_true = np.asarray(y_true).astype(np.int64)
    y_pred = np.asarray(y_pred).astype(np.int64)

    out["acc"] = float((y_true == y_pred).mean())

    if f1_score is not None:
        out["macro_f1"] = float(f1_score(y_true, y_pred, average="macro"))
    else:
        out["macro_f1"] = float("nan")

    C = int(np.unique(y_true).size)
    if confusion_matrix is not None:
        cm = confusion_matrix(y_true, y_pred, labels=np.arange(C))
        row_sum = cm.sum(axis=1, keepdims=True).clip(min=1)
        row_acc = (cm / row_sum).diagonal()
        out["mean_per_class_acc"] = float(np.mean(row_acc))
        k = min(5, len(row_acc))
        out["worst_5_class_acc"] = float(np.mean(np.sort(row_acc)[:k]))
    else:
        out["mean_per_class_acc"] = float("nan")
        out["worst_5_class_acc"] = float("nan")

    # 多分类 AUROC / AUPRC（若有概率）
    out["auroc_macro_ovr"] = float("nan")
    out["auprc_macro_ovr"] = float("nan")
    if (prob is not None) and (roc_auc_score is not None) and (average_precision_score is not None) and (label_binarize is not None):
        try:
            classes = np.unique(y_true)
            Y = label_binarize(y_true, classes=classes)
            # prob shape [N,C], 列顺序需与类别索引对齐（这里默认0..C-1）
            P = prob[:, classes]
            if Y.shape[1] > 1:
                out["auroc_macro_ovr"] = float(roc_auc_score(Y, P, average="macro", multi_class="ovr"))
                out["auprc_macro_ovr"] = float(average_precision_score(Y, P, average="macro"))
        except Exception:
            pass

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mat_all", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True, help=".../glformer_like_best.pth")
    ap.add_argument("--split_npz", type=str, default="")
    ap.add_argument("--save_dir", type=str, required=True)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--subset", type=str, default="all", choices=["all", "train", "val", "test"],
                help="which subset to export embeddings for")

    # 若 ckpt 中没有 args（通常会有），可手动指定
    ap.add_argument("--patch", type=int, default=8)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[Device]", device)

    # 读取数据（spec + y）
    X_spec, y, meta = load_featurematrix3_spec_label(args.mat_all)
    N = len(y)

    # 读取 MAT 原始结构（为了导出 file_id/metaInfo 等）
    S = load_mat_auto(args.mat_all)
    file_id = np.asarray(S["file_id"]).reshape(-1).astype(np.int64) if "file_id" in S else None
    fsVector = np.asarray(S["fsVector"]).reshape(-1).astype(np.float32) if "fsVector" in S else None
    metaInfo = np.asarray(S["metaInfo"]) if "metaInfo" in S else None
    file_index = np.asarray(S["file_index"]) if "file_index" in S else None

    # split
    if args.split_npz and os.path.isfile(args.split_npz):
        tr_idx, va_idx, te_idx = load_indices_from_npz(args.split_npz)
        print("[Split] use npz:", len(tr_idx), len(va_idx), len(te_idx))
    else:
        tr_idx, va_idx, te_idx = stratified_split_indices(y, seed=args.seed)
        print("[Split] stratified:", len(tr_idx), len(va_idx), len(te_idx))

    # 选择导出子集
    if args.subset == "train":
        export_idx = tr_idx
    elif args.subset == "val":
        export_idx = va_idx
    elif args.subset == "test":
        export_idx = te_idx
    else:
        export_idx = np.arange(N, dtype=np.int64)

    print(f"[Export subset] {args.subset}: n={len(export_idx)}")

    # 载入 ckpt
    ckpt = torch.load(args.ckpt, map_location="cpu")
    ckpt_args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
    stats = ckpt.get("stats", None) if isinstance(ckpt, dict) else None
    mean = float(stats["mean"]) if isinstance(stats, dict) and ("mean" in stats) else float(X_spec[tr_idx].mean())
    std = float(stats["std"]) if isinstance(stats, dict) and ("std" in stats) else float(X_spec[tr_idx].std() + 1e-6)

    patch = int(ckpt_args.get("patch", args.patch))
    dim = int(ckpt_args.get("dim", args.dim))
    depth = int(ckpt_args.get("depth", args.depth))
    heads = int(ckpt_args.get("heads", args.heads))
    drop = float(ckpt_args.get("dropout", args.dropout))

    num_classes = int(np.unique(y).size)
    model = LightSEITransformer(
        num_classes=num_classes,
        img_size=X_spec.shape[1],
        patch=patch, dim=dim, depth=depth, heads=heads, drop=drop
    )
    state = ckpt["model_state"] if isinstance(ckpt, dict) and ("model_state" in ckpt) else ckpt
    model.load_state_dict(state, strict=True)
    model.to(device)

    # 全量 dataloader（顺序不能乱）
    ds_all = SpecClsDataset(X_spec, y, mean=mean, std=std)
    dl_all = DataLoader(ds_all, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    H, logits, prob, pred, y_all = forward_all(model, dl_all, device)
    assert len(H) == N and len(y_all) == N

    # 选择导出子集
    if args.subset == "train":
        export_idx = tr_idx
    elif args.subset == "val":
        export_idx = va_idx
    elif args.subset == "test":
        export_idx = te_idx
    else:
        export_idx = np.arange(N, dtype=np.int64)

    print(f"[Export subset] {args.subset}: n={len(export_idx)}")

    # 测试集指标
    te = compute_metrics(y_all[te_idx], pred[te_idx], prob[te_idx])
    print("[Test metrics]", te)

    # 保存 embeddings（给 few-shot 下游直接用）
    emb_path = os.path.join(args.save_dir, "glformer_embeddings.npz")
    save_dict = {
        "H": H.astype(np.float32),
        "y": y_all.astype(np.int64),
        "logits": logits.astype(np.float32),
        "prob": prob.astype(np.float32),
        "pred": pred.astype(np.int64),
        # "train_idx": tr_idx.astype(np.int64),
        # "val_idx": va_idx.astype(np.int64),
        # "test_idx": te_idx.astype(np.int64),
        "orig_index": export_idx.astype(np.int64),   # 原始全量数据中的索引
        "subset": np.array(args.subset, dtype=object),
        "norm_mean": np.array([mean], dtype=np.float32),
        "norm_std": np.array([std], dtype=np.float32),
    }
    # if file_id is not None and len(file_id) == N:
    #     save_dict["file_id"] = file_id.astype(np.int64)
    # if fsVector is not None and len(fsVector) == N:
    #     save_dict["fsVector"] = fsVector.astype(np.float32)
    # if metaInfo is not None and metaInfo.shape[0] == N:
    #     save_dict["metaInfo"] = metaInfo
    # if file_index is not None:
    #     save_dict["file_index"] = file_index
    if file_id is not None and len(file_id) == N:
        save_dict["file_id"] = file_id[export_idx].astype(np.int64)
    if fsVector is not None and len(fsVector) == N:
        save_dict["fsVector"] = fsVector[export_idx].astype(np.float32)
    if metaInfo is not None and metaInfo.shape[0] == N:
        save_dict["metaInfo"] = metaInfo[export_idx]
    if file_index is not None:
        save_dict["file_index"] = file_index
    np.savez_compressed(emb_path, **save_dict)
    print("[Save]", emb_path)

    # 保存 test 预测（统一评估脚本优先读取）
    pred_path = os.path.join(args.save_dir, "test_predictions.npz")
    np.savez_compressed(
        pred_path,
        y_true=y_all[te_idx].astype(np.int64),
        y_pred=pred[te_idx].astype(np.int64),
        prob=prob[te_idx].astype(np.float32),
        logits=logits[te_idx].astype(np.float32),
        test_idx=te_idx.astype(np.int64),
    )
    print("[Save]", pred_path)

    with open(os.path.join(args.save_dir, "test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(te, f, indent=2, ensure_ascii=False)

    # 也补一份 summary.json（便于统一聚合）
    summary = {
        "model_name": "GLFormer-like (TIFS2023 style)",
        "export_embeddings": os.path.basename(emb_path),
        "test_metrics": te,
        "meta": meta,
        "num_samples": int(N),
        "num_classes": num_classes,
        "ckpt": args.ckpt,
    }
    with open(os.path.join(args.save_dir, "summary_export.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()