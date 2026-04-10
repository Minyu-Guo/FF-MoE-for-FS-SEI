"""
single_expert_train_v2.py

目标：
    注入到 MoE 时天然 matched_keys=6/6（MLP 专家）/ 全部匹配（TF）
说明：
    BT数据集：FeatureMatrix_3.mat    topk_indices.mat
    WiFi数据集：FeatureMatrix_OSU_Stable_WiFi_Wireless.mat    topk_indices_wifi.mat

运行示例：
  BT:
  python single_expert_train_v2.py --mat_all FeatureMatrix_3.mat --topk_mat topk_indices.mat --expert inst    # 只跑time专家
  wifi:
  python single_expert_train_v2.py \
  --mat_all FeatureMatrix_OSU_Stable_WiFi_Wireless_unified.mat \
  --save_dir ./single_expert_runs_osu_stable \
  --expert all \
  --epochs 60 \
  --batch_size 64 \
  --lr 1e-3 \
  --weight_decay 1e-4 \
  --dropout 0.2 \
  --inst_epochs 100 \
  --inst_batch_size 64 \
  --inst_lr 2e-4 \
  --inst_weight_decay 1e-4 \
  --inst_dropout 0.08 \
  --inst_label_smoothing 0.05 \
  --val_ratio 0.2 \
  --seed 42
"""

import os, time, argparse
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

# 直接复用 MoE 脚本里的 MLPExpert / TFExpert（结构完全一致）
def import_moe_experts():
    try:
        from train_moe_tfcnn_supcon_explain_closedloop_v2 import MLPExpert, TFExpert, ImgExpert
        return MLPExpert, TFExpert, ImgExpert
    except Exception:
        from train_moe_tfcnn_supcon_explain_closedloop import MLPExpert, TFExpert
        return MLPExpert, TFExpert

MLPExpert, TFExpert, ImgExpert = import_moe_experts()


# ========= MAT load =========
def load_mat_auto(path):
    try:
        mat = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
        return {"__backend__": "scipy", **mat}
    except Exception as e:
        msg = str(e).lower()
        if ("version 73" in msg) or ("unknown mat file type" in msg) or ("matlab 7.3" in msg):
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
        raise


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
        raise ValueError(f"4D specTensor cannot match N={N}, got {spec.shape}")
    raise ValueError(f"specTensor ndim must be 3/4, got {spec.ndim}, shape={spec.shape}")


def load_topk_union_indices(topk_path, P):
    if (not topk_path) or (not os.path.isfile(topk_path)):
        return np.array([], dtype=np.int64)
    T = load_mat_auto(topk_path)
    keys = [k for k in T.keys() if not k.startswith("__")]
    if len(keys) == 0:
        return np.array([], dtype=np.int64)

    def _as_indices(arr):
        arr = np.array(arr)
        if arr.dtype == np.bool_ and arr.size == P:
            return np.where(arr.reshape(-1))[0].astype(np.int64)
        flat = arr.reshape(-1)
        if np.issubdtype(flat.dtype, np.number):
            flat = flat[np.isfinite(flat)]
            if flat.size == 0:
                return None
            v = np.round(flat).astype(np.int64)
            v = v[v >= 0]
            if v.size == 0:
                return None
            if v.min() >= 1 and v.max() <= P:  # matlab 1-based
                v = v - 1
            v = v[(v >= 0) & (v < P)]
            if v.size == 0:
                return None
            return np.unique(v)
        return None

    preferred = ["topk_idx", "topk_indices", "union_idx", "union_indices", "idx", "indices"]
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


# ========= Dataset =========
class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, X_all_z, X_spec, X_occ, y, indices):
        self.X_all = torch.tensor(X_all_z[indices]).float()
        self.X_spec = torch.tensor(X_spec[indices]).float()
        self.X_occ  = torch.tensor(X_occ[indices]).float()   # <-- 新增
        self.y = torch.tensor(y[indices]).long()

    def __len__(self):
        return int(self.y.numel())

    def __getitem__(self, i):
        return self.X_all[i], self.X_spec[i], self.X_occ[i], self.y[i]
        

class SingleExpertReuseMoE(nn.Module):
    """
    只包含 1 个专家子模块，但模块名严格与 MoE 一致：
      - time:  expert_time
      - freq:  expert_freq
      - inst:  expert_inst
      - tf:    expert_tf
    保存 state_dict 后，inject 时天然对齐。
    """
    def __init__(self, expert_name, num_classes, spec_in_ch, occ_in_ch = 1,cnn_emb_dim=64, emb_dim=64, p_drop=0.2, inst_p_drop=None):
        super().__init__()
        self.expert_name = expert_name
        self.time_slice = slice(0, 17)
        self.freq_slice = slice(17, 22)
        self.tf_slice   = slice(22, 27)
        self.inst_slice = slice(27, 39)

        if expert_name == "time":
            self.expert_time = MLPExpert(17, num_classes, hidden=64, emb_dim=emb_dim, p_drop=p_drop)
        elif expert_name == "freq":
            # self.expert_freq = MLPExpert(5,  num_classes, hidden=64, emb_dim=emb_dim, p_drop=p_drop)
            self.expert_freq = ImgExpert(occ_in_ch, cnn_emb_dim, num_classes, hidden=64, emb_dim=emb_dim, p_drop=p_drop)  # <-- 替换
        elif expert_name == "inst":
            drop = p_drop if inst_p_drop is None else inst_p_drop
            self.expert_inst = MLPExpert(12, num_classes, hidden=64, emb_dim=emb_dim, p_drop=p_drop)
        elif expert_name == "tf":
            self.expert_tf = TFExpert(5, spec_in_ch, cnn_emb_dim, num_classes, hidden=64, emb_dim=emb_dim, p_drop=p_drop)
        else:
            raise ValueError("expert_name must be one of: time,freq,tf,inst")

    def forward(self, x_all, x_spec=None, x_occ=None):
        if self.expert_name == "time":
            return self.expert_time(x_all[:, self.time_slice])
        
        if self.expert_name == "freq":
            if x_occ is None:
                raise ValueError("freq expert requires x_occ (occTensor NCHW)")
            return self.expert_freq(x_occ)

        if self.expert_name == "inst":
            return self.expert_inst(x_all[:, self.inst_slice])

        if self.expert_name == "tf":
            return self.expert_tf(x_all[:, self.tf_slice], x_spec)
        raise RuntimeError("unreachable")


@torch.no_grad()
def eval_one(model, loader, device):
    model.eval()
    ys, ps = [], []
    for x, s, o, y in loader:
        x, s, o, y = x.to(device), s.to(device), o.to(device), y.to(device)
        logits, _ = model(x, x_spec=s, x_occ=o)
        pred = logits.argmax(1)
        ys.append(y.cpu().numpy())
        ps.append(pred.cpu().numpy())
    y_true = np.concatenate(ys)
    y_pred = np.concatenate(ps)
    acc = float((y_true == y_pred).mean())
    return acc, y_true, y_pred


def train_one(expert_name, args, X_all_z, X_spec, X_occ, y, train_idx, val_idx, num_classes, device):
    model = SingleExpertReuseMoE(
        expert_name=expert_name,
        num_classes=num_classes,
        spec_in_ch=int(X_spec.shape[1]),
        occ_in_ch=int(X_occ.shape[1]),  
        cnn_emb_dim=args.cnn_emb_dim,
        emb_dim=args.emb_dim,
        p_drop=args.dropout,
        inst_p_drop=(args.inst_dropout if expert_name == "inst" else None),
    ).to(device)
    # -----------------------------
    # expert-specific training hyperparams
    # -----------------------------
    if expert_name == "inst":
        epochs_cur = args.inst_epochs
        batch_cur = args.inst_batch_size
        lr_cur = args.inst_lr
        wd_cur = args.inst_weight_decay
    else:
        epochs_cur = args.epochs
        batch_cur = args.batch_size
        lr_cur = args.lr
        wd_cur = args.weight_decay

    # -----------------------------
    # expert-specific training hyperparams
    # -----------------------------
    train_loader = DataLoader(
        SimpleDataset(X_all_z, X_spec, X_occ, y, train_idx),
        batch_size=batch_cur, shuffle=True, num_workers=0)
    val_loader = DataLoader(
        SimpleDataset(X_all_z, X_spec, X_occ, y, val_idx),
        batch_size=batch_cur, shuffle=False, num_workers=0)

    # -----------------------------
    # optimizer
    # -----------------------------
    opt = optim.Adam(model.parameters(), lr=lr_cur, weight_decay=wd_cur)
    
    # -----------------------------
    # loss
    # inst: optional weighted CE + label smoothing
    # -----------------------------
    if expert_name == "inst":
        if args.inst_use_weighted_ce:
            cnt = np.bincount(y[train_idx], minlength=num_classes).astype(np.float32)
            w = cnt.sum() / np.maximum(cnt, 1.0)
            w = w / w.mean()
            ce = nn.CrossEntropyLoss(
                weight=torch.tensor(w, dtype=torch.float32, device=device),
                label_smoothing=args.inst_label_smoothing
            )
        else:
            ce = nn.CrossEntropyLoss(label_smoothing=args.inst_label_smoothing)
    else:
        ce = nn.CrossEntropyLoss()

    # -----------------------------
    # scheduler
    # -----------------------------
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", factor=0.5,
        patience=(args.inst_sched_patience if expert_name == "inst" else 8),
        verbose=True
    )

    best = -1.0
    os.makedirs(args.save_dir, exist_ok=True)
    out_best = os.path.join(args.save_dir, f"best_{expert_name}.pth")

    # -----------------------------
    # train loop
    # -----------------------------
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        loss_sum, n_sum = 0.0, 0

        for x, s, o, yb in train_loader:
            # x, s, yb = x.to(device), s.to(device), yb.to(device)
            x, s, o, yb = x.to(device), s.to(device), o.to(device), yb.to(device)
            opt.zero_grad()
            logits, _ = model(x, x_spec=s, x_occ=o)
            loss = ce(logits, yb)
            loss.backward()
            opt.step()
            bs = int(yb.numel())
            loss_sum += float(loss.item()) * bs
            n_sum += bs

        tr_loss = loss_sum / max(1, n_sum)
        val_acc, _, _ = eval_one(model, val_loader, device)
        print(f"[{expert_name}][{ep:03d}] loss={tr_loss:.4f} val_acc={val_acc:.4f} time={time.time()-t0:.1f}s")

        if val_acc > best:
            best = val_acc
            os.makedirs(args.save_dir, exist_ok=True)
            torch.save(model.state_dict(), out_best)
            print(f"  ✓ saved -> {out_best}")

        scheduler.step(val_acc)

    # -----------------------------
    # train loop
    # -----------------------------
    val_acc, y_true, y_pred = eval_one(model, val_loader, device)
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))
    np.save(os.path.join(args.save_dir, f"cm_{expert_name}_counts.npy"), cm)

    rows = []
    for c in range(num_classes):
        idx = (y_true == c)
        n = int(idx.sum())
        corr = int((y_pred[idx] == c).sum()) if n > 0 else 0
        acc = float(corr / n) if n > 0 else 0.0
        rows.append((c, n, corr, acc))
    pd.DataFrame(rows, columns=["class_id", "n", "correct", "acc"]).to_csv(
        os.path.join(args.save_dir, f"per_class_{expert_name}.csv"), index=False)

    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mat_all", type=str, default="FeatureMatrix_Indoor_OSU.mat")    
    ap.add_argument("--topk_mat", type=str, default="topk_indices_wifi.mat")
    ap.add_argument("--save_dir", type=str, default="./single_expert_runs_reuse_occ_1")
    ap.add_argument("--expert", type=str, default="all", choices=["all", "time", "freq", "tf", "inst"])

    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--emb_dim", type=int, default=64)
    ap.add_argument("--cnn_emb_dim", type=int, default=64)

    ap.add_argument("--inst_epochs", type=int, default=100)
    ap.add_argument("--inst_batch_size", type=int, default=128)
    ap.add_argument("--inst_lr", type=float, default=2e-4)
    ap.add_argument("--inst_weight_decay", type=float, default=1e-4)
    ap.add_argument("--inst_dropout", type=float, default=0.08)
    ap.add_argument("--inst_noise_std", type=float, default=0.01)
    ap.add_argument("--inst_label_smoothing", type=float, default=0.05)
    ap.add_argument("--inst_use_weighted_ce", action="store_true")
    ap.add_argument("--inst_sched_patience", type=int, default=5)

    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    S = load_mat_auto(args.mat_all)

    kX = pick_first_existing(S, ["featureMatrix", "feature_matrix", "X"])
    ky = pick_first_existing(S, ["label_id", "device_id", "y", "labels"])
    kspec = pick_first_existing(S, ["specTensor", "spec", "spec_tensor", "SpecTensor"])
    kocc = pick_first_existing(S, ["occTensor", "occ_tensor", "densityTensor", "occ", "occMap"])
    kfs = pick_first_existing(S, ["fsVector", "fs", "fs_vec"])
    kfid = pick_first_existing(S, ["file_id", "fileId", "fileID", "file_idx"])

    if kX is None or ky is None or kspec is None:
        raise KeyError("MAT 必须包含 featureMatrix / label_id / specTensor")

    if kocc is None:
        raise KeyError("MAT missing occTensor. 你需要先在 MATLAB 导出 occTensor (density map).")

    X_all = ensure_2d_feature_matrix(S[kX]).astype(np.float32)
    y = to_1d_int(S[ky])
    N, P = X_all.shape
    if P != 39:
        raise ValueError(f"Expect 39-dim, got {P}")

    # ---------- basic data ----------
    X_all = ensure_2d_feature_matrix(S[kX]).astype(np.float32)
    y = to_1d_int(S[ky])
    N, P = X_all.shape
    if P != 39:
        raise ValueError(f"Expect 39-dim, got {P}")

    # ---------- split first ----------
    idx_all = np.arange(N)
    if kfid is not None:
        file_id = to_1d_int(S[kfid])
        gss = GroupShuffleSplit(n_splits=1, test_size=args.val_ratio, random_state=args.seed)
        train_idx, val_idx = next(gss.split(idx_all, y, groups=file_id))
        print("[Split] GroupShuffleSplit by file_id.")
    else:
        train_idx, val_idx = train_test_split(
            idx_all, test_size=args.val_ratio, stratify=y, random_state=args.seed
        )
        print("[Split] Stratified split by y.")

    # ---------- then load raw spec/occ ----------
    X_occ = ensure_nchw(S[kocc], N).astype(np.float32)   # [N,C,H,W]
    X_spec = ensure_nchw(S[kspec], N).astype(np.float32) # [N,C,H,W]

    # ---------- normalize using TRAIN statistics only ----------
    # all39
    scaler = StandardScaler()
    scaler.fit(X_all[train_idx])
    X_all_z = scaler.transform(X_all).astype(np.float32)

    # occ
    occ_mu = X_occ[train_idx].mean()
    occ_std = X_occ[train_idx].std() + 1e-8
    X_occ = ((X_occ - occ_mu) / occ_std).astype(np.float32)

    # spec
    spec_mu = X_spec[train_idx].mean()
    spec_std = X_spec[train_idx].std() + 1e-8
    X_spec = ((X_spec - spec_mu) / spec_std).astype(np.float32)

    print(f"[Data] spec={X_spec.shape}, occ={X_occ.shape}")

    num_classes = int(np.unique(y).size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Data] N={N}, classes={num_classes}, spec={X_spec.shape}, occ={X_occ.shape}, device={device}")

    experts = ["time", "freq", "tf", "inst"] if args.expert == "all" else [args.expert]
    os.makedirs(args.save_dir, exist_ok=True)

    for ex in experts:
        best = train_one(ex, args, X_all_z, X_spec, X_occ, y, train_idx, val_idx, num_classes, device)
        print(f"[Done] {ex} best_val_acc={best:.4f}")

    print("All done.")


if __name__ == "__main__":
    main()
