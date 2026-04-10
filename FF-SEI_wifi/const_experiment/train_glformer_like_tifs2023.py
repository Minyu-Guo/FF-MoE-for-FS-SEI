# -*- coding: utf-8 -*-
"""
GLFormer-like lightweight Transformer baseline for SEI (TIFS 2023 style)
Input: FeatureMatrix_3.mat (use specTensor + label_id)

说明：
BT数据集：FeatureMatrix_3.mat    topk_indices.mat
WiFi数据集：FeatureMatrix_Indoor_OSU.mat   topk_indices_wifi.mat

python train_glformer_like_tifs2023.py \
  --mat_all ../FeatureMatrix_OSU_Stable_WiFi_Wireless_unified.mat  \
  --split_npz ../split_indices_fssei_osu_stable_wireless.npz \
  --save_dir ./experiments/e2_glformer_like_tifs2023 \
  --epochs 40 \
  --batch_size 128 \
  --lr 1e-3 \
  --patch 8 --dim 144 --depth 4 --heads 4
"""

import os, time, argparse
import numpy as np
import torch
import torch.nn as nn

from sei_mat_utils import (
    set_seed, load_featurematrix3_spec_label,
    load_indices_from_npz, stratified_split_indices,
    make_loaders_for_cls, train_one_epoch_cls, evaluate_cls,
    save_logs_csv, dump_json
)


class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, dim),
            nn.Dropout(drop),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, dim=128, num_heads=4, mlp_ratio=3.0, drop=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=drop, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio=mlp_ratio, drop=drop)

    def forward(self, x):
        z = self.norm1(x)
        a, _ = self.attn(z, z, z, need_weights=False)
        x = x + a
        x = x + self.mlp(self.norm2(x))
        return x


class LightSEITransformer(nn.Module):
    """
    工程复现风格 lightweight Transformer:
      specTensor(1x64x64) -> conv stem -> patch tokens -> transformer -> cls head
    """
    def __init__(self, num_classes, img_size=64, patch=8, dim=128, depth=4, heads=4, drop=0.1):
        super().__init__()
        assert img_size % patch == 0
        self.img_size = img_size
        self.patch = patch
        self.dim = dim

        self.stem = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.Conv2d(16, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
        )

        self.patch_embed = nn.Conv2d(32, dim, kernel_size=patch, stride=patch)
        num_patches = (img_size // patch) * (img_size // patch)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + num_patches, dim))
        self.drop = nn.Dropout(drop)

        self.blocks = nn.ModuleList([
            TransformerBlock(dim=dim, num_heads=heads, mlp_ratio=3.0, drop=drop)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward_features(self, x):
        x = self.stem(x)                       # [B,32,64,64]
        x = self.patch_embed(x)                # [B,D,64/p,64/p]
        x = x.flatten(2).transpose(1, 2)       # [B,L,D]
        B = x.size(0)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed[:, :x.size(1), :]
        x = self.drop(x)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0]                         # CLS token

    def forward(self, x):
        feat = self.forward_features(x)
        return self.head(feat)

# ===== complexity / export helper =====
def create_model(
    num_classes,
    img_size=64,
    patch=8,
    dim=128,
    depth=4,
    heads=4,
    drop=0.1,
    dropout=None,
    **kwargs
):
    """
    统一给外部脚本调用的构造入口。
    dropout 若传入，则覆盖 drop，便于和外部命名对齐。
    """
    if dropout is not None:
        drop = dropout

    return LightSEITransformer(
        num_classes=num_classes,
        img_size=img_size,
        patch=patch,
        dim=dim,
        depth=depth,
        heads=heads,
        drop=drop,
    )


def load_pretrained(model, ckpt_path, map_location="cpu", strict=True):
    """
    统一给复杂度/导出脚本调用的权重加载入口。
    兼容当前训练脚本保存的 glformer_like_best.pth:
      {"model_state": ..., "epoch": ..., "val_acc": ..., ...}
    """
    ckpt = torch.load(ckpt_path, map_location=map_location)
    state = ckpt.get("model_state", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=strict)
    return model, {"missing": missing, "unexpected": unexpected}
    
    
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mat_all", type=str, required=True)
    ap.add_argument("--split_npz", type=str, default="")
    ap.add_argument("--save_dir", type=str, required=True)

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=15)

    ap.add_argument("--patch", type=int, default=8)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None
    print("[Device]", device)

    X_spec, y, meta = load_featurematrix3_spec_label(args.mat_all)
    print("[Data]", X_spec.shape, y.shape, meta)

    if args.split_npz and os.path.isfile(args.split_npz):
        tr_idx, va_idx, te_idx = load_indices_from_npz(args.split_npz)
        print("[Split] use npz:", args.split_npz, len(tr_idx), len(va_idx), len(te_idx))
    else:
        tr_idx, va_idx, te_idx = stratified_split_indices(y, seed=args.seed)
        print("[Split] stratified random:", len(tr_idx), len(va_idx), len(te_idx))

    dl_tr, dl_va, dl_te, stats = make_loaders_for_cls(
        X_spec, y, tr_idx, va_idx, te_idx,
        batch_size=args.batch_size, num_workers=args.num_workers
    )

    num_classes = int(np.unique(y).size)
    model = LightSEITransformer(
        num_classes=num_classes,
        img_size=X_spec.shape[1],
        patch=args.patch,
        dim=args.dim,
        depth=args.depth,
        heads=args.heads,
        drop=args.dropout
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[Model] params={n_params:.3f} M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    logs = []
    best_val_acc = -1.0
    best_epoch = -1
    wait = 0

    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        tr = train_one_epoch_cls(model, dl_tr, optimizer, device, scaler=scaler)
        va = evaluate_cls(model, dl_va, device)
        scheduler.step()

        row = {
            "epoch": ep,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": tr["loss"],
            "train_acc": tr["acc"],
            "val_loss": va["loss"],
            "val_acc": va["acc"],
            "val_macro_f1": va["macro_f1"],
            "sec": time.time() - t0
        }
        logs.append(row)

        print(f"[{ep:03d}] "
              f"train_loss={tr['loss']:.4f}, train_acc={tr['acc']:.4f} | "
              f"val_loss={va['loss']:.4f}, val_acc={va['acc']:.4f}, val_f1={va['macro_f1']:.4f}")

        if va["acc"] > best_val_acc:
            best_val_acc = va["acc"]
            best_epoch = ep
            wait = 0
            torch.save({
                "model_state": model.state_dict(),
                "epoch": ep,
                "val_acc": va["acc"],
                "stats": stats,
                "args": vars(args),
            }, os.path.join(args.save_dir, "glformer_like_best.pth"))
        else:
            wait += 1
            if wait >= args.patience:
                print("[EarlyStop] patience reached.")
                break

    ckpt = torch.load(os.path.join(args.save_dir, "glformer_like_best.pth"), map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    model.to(device)

    te = evaluate_cls(model, dl_te, device)
    print(f"[Test] loss={te['loss']:.4f}, acc={te['acc']:.4f}, macro_f1={te['macro_f1']:.4f}")

    save_logs_csv(logs, os.path.join(args.save_dir, "train_log.csv"))
    dump_json({
        "args": vars(args),
        "meta": meta,
        "norm_stats": stats,
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "test_metrics": te,
        "model_params_M": n_params,
    }, os.path.join(args.save_dir, "summary.json"))


if __name__ == "__main__":
    main()