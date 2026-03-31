#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
python export_hfusion_supcon_closedloop_v2_consistent.py \
  --mat_all ./FeatureMatrix_OSU_Stable_WiFi_Wireless_unified.mat \
  --topk_mat ./topk_indices_wifi.mat \
  --ckpt ./result/pretrain/stage3b_joint_osu_stable/moe_tfcnn_best.pth \
  --out_npz ./result/exports/hfusion_osu_stable_joint_noinst_consistent_test.npz \
  --preproc_npz ./result/pretrain/stage3b_joint_osu_stable/preproc_scaler.npz \
  --label_map ./label_map_osu_stable_wifi_wireless.csv \
  --split_npz ./split_indices_fssei_osu_stable_wireless.npz \
  --subset test
time:
python export_hfusion_supcon_closedloop_v2_consistent.py \
  --mat_all ./FeatureMatrix_OSU_Stable_WiFi_Wireless_unified.mat \
  --topk_mat ./topk_indices_wifi.mat \
  --ckpt ./ablation/no_time/stageB/moe_tfcnn_best.pth \
  --out_npz ./ablation/exports/no_time_embeddings.npz  \
  --preproc_npz ./ablation/no_time/stageB/preproc_scaler.npz \
  --label_map ./label_map_osu_stable_wifi_wireless.csv \
  --split_npz ./split_indices_fssei_osu_stable_wireless.npz \
  --subset test
freq:
python export_hfusion_supcon_closedloop_v2_consistent.py \
  --mat_all ./FeatureMatrix_OSU_Stable_WiFi_Wireless_unified.mat \
  --topk_mat ./topk_indices_wifi.mat \
  --ckpt ./ablation/no_freq/stageB/moe_tfcnn_best.pth \
  --out_npz ./ablation/exports/no_freq_embeddings.npz \
  --preproc_npz ./ablation/no_freq/stageB/preproc_scaler.npz \
  --label_map ./label_map_osu_stable_wifi_wireless.csv \
  --split_npz ./split_indices_fssei_osu_stable_wireless.npz \
  --subset test
tf:
python export_hfusion_supcon_closedloop_v2_consistent.py \
  --mat_all ./FeatureMatrix_OSU_Stable_WiFi_Wireless_unified.mat \
  --topk_mat ./topk_indices_wifi.mat \
  --ckpt ./ablation/no_tf/stageB/moe_tfcnn_best.pth \
  --out_npz ./ablation/exports/no_tf_embeddings.npz \
  --preproc_npz ./ablation/no_tf/stageB/preproc_scaler.npz \
  --label_map ./label_map_osu_stable_wifi_wireless.csv \
  --split_npz ./split_indices_fssei_osu_stable_wireless.npz \
  --subset test
inst:
python export_hfusion_supcon_closedloop_v2_consistent.py \
  --mat_all ./FeatureMatrix_OSU_Stable_WiFi_Wireless_unified.mat \
  --topk_mat ./topk_indices_wifi.mat \
  --ckpt ./ablation/no_inst/stageB/moe_tfcnn_best.pth \
  --out_npz ./ablation/exports/no_inst_embeddings.npz \
  --preproc_npz ./ablation/no_inst/stageB/preproc_scaler.npz \
  --label_map ./label_map_osu_stable_wifi_wireless.csv \
  --split_npz ./split_indices_fssei_osu_stable_wireless.npz \
  --subset test
"""
import os, argparse
from collections import Counter
import numpy as np, scipy.io as sio, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader

def load_mat_v73_h5py(path):
    import h5py
    def _fix(arr):
        if isinstance(arr, np.ndarray) and arr.ndim >= 2:
            arr = np.transpose(arr, tuple(range(arr.ndim))[::-1])
        return arr
    def _h5(obj):
        if isinstance(obj, h5py.Dataset):
            return _fix(np.array(obj[()]))
        if isinstance(obj, h5py.Group):
            return {k: _h5(obj[k]) for k in obj.keys()}
        return obj
    out = {"__backend__":"h5py"}
    with h5py.File(path, "r") as f:
        for k in f.keys():
            out[k] = _h5(f[k])
    return out

def load_mat_auto(path):
    try:
        return {"__backend__":"scipy", **sio.loadmat(path, squeeze_me=True, struct_as_record=False)}
    except Exception as e:
        msg = str(e).lower()
        if "73" in msg or "unknown mat file type" in msg or "matlab 7.3" in msg:
            return load_mat_v73_h5py(path)
        raise

def pick_first_existing(d, keys):
    for k in keys:
        if k in d: return k
    return None

def ensure_2d_feature_matrix(X):
    X = np.asarray(X)
    if X.ndim == 1: X = X.reshape(-1, 1)
    if X.ndim != 2: raise ValueError(f"featureMatrix must be 2D, got {X.shape}")
    return X

def to_1d_int(y):
    y = np.asarray(y).reshape(-1).astype(np.int64)
    _, y_new = np.unique(y, return_inverse=True)
    return y_new.astype(np.int64)

def load_split_indices(npz_path):
    if not os.path.isfile(npz_path):
        raise FileNotFoundError(f"split file not found: {npz_path}")

    S = np.load(npz_path, allow_pickle=True)

    def pick(keys):
        for k in keys:
            if k in S:
                return np.asarray(S[k]).astype(np.int64)
        return None

    train_idx = pick(["train_idx", "train_indices", "idx_train"])
    val_idx   = pick(["val_idx", "val_indices", "idx_val"])
    test_idx  = pick(["test_idx", "test_indices", "idx_test"])

    if train_idx is None or val_idx is None or test_idx is None:
        raise KeyError(f"{npz_path} missing train/val/test indices")

    return train_idx, val_idx, test_idx

def ensure_nchw(spec, N):
    spec = np.asarray(spec)
    if spec.ndim == 3:
        if spec.shape[2] == N: return np.transpose(spec, (2,0,1))[:,None,:,:]
        if spec.shape[0] == N: return spec[:,None,:,:]
        raise ValueError(f"3D specTensor cannot match N={N}, got {spec.shape}")
    if spec.ndim == 4:
        if spec.shape[3] == N: return np.transpose(spec, (3,2,0,1))
        if spec.shape[0] == N: return np.transpose(spec, (0,3,1,2))
        if spec.shape[2] == N: return np.transpose(spec, (2,3,0,1))
        if spec.shape[1] == N: return np.transpose(spec, (1,3,0,2))
        raise ValueError(f"4D specTensor cannot match N={N}, got {spec.shape}")
    raise ValueError(f"specTensor ndim must be 3/4, got {spec.ndim}")

def load_topk_union_indices(topk_path, P):
    if not topk_path or (not os.path.isfile(topk_path)):
        return np.array([], dtype=np.int64)
    T = load_mat_auto(topk_path)
    keys = [k for k in T.keys() if not k.startswith("__")]
    def _as_indices(arr):
        arr = np.asarray(arr)
        if arr.dtype == np.bool_ and arr.size == P:
            return np.where(arr.reshape(-1))[0].astype(np.int64)
        if arr.dtype == object: return None
        flat = arr.reshape(-1)
        if np.issubdtype(flat.dtype, np.number):
            flat = flat[np.isfinite(flat)]
            if flat.size == 0: return None
            v = np.round(flat).astype(np.int64)
            v = v[v >= 0]
            if v.size == 0: return None
            if v.min() >= 1 and v.max() <= P: v = v - 1
            v = v[(v >= 0) & (v < P)]
            if v.size == 0: return None
            return np.unique(v)
        return None
    for k in ["topk_idx","topk_indices","union_idx","union_indices","selected_idx","selected_indices","idx"]:
        if k in T:
            idx = _as_indices(T[k])
            if idx is not None and idx.size > 0: return idx
    best, best_sz = None, 0
    for k in keys:
        idx = _as_indices(T[k])
        if idx is not None and idx.size > best_sz:
            best, best_sz = idx, idx.size
    return np.array([], dtype=np.int64) if best is None else best.astype(np.int64)

def _mat_char_to_str(x):
    if x is None: return ""
    if isinstance(x, str): return x
    if isinstance(x, (bytes, np.bytes_)):
        try: return x.decode("utf-8", errors="ignore")
        except Exception: return str(x)
    arr = np.asarray(x)
    if arr.dtype.kind in ["U","S"]:
        try: return "".join(arr.reshape(-1).tolist()).strip()
        except Exception: return str(arr)
    if arr.dtype.kind in ["i","u"]:
        try: return "".join(chr(int(v)) for v in arr.reshape(-1).tolist() if int(v) != 0).strip()
        except Exception: return str(arr)
    return str(x)

def load_label_map_csv(path):
    if not path or (not os.path.isfile(path)): return None
    import csv
    mp = {}
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for row in csv.DictReader(f):
            mp[int(row["class_id"])] = str(row["name"])
    return mp if mp else None

def infer_label_map_from_mat(S, y):
    for key in ["labelVector","label_vector","labelStrVector","labelStr","labels_str"]:
        if key in S:
            try:
                arr = np.asarray(S[key]).reshape(-1)
                if arr.size == y.size:
                    names = [_mat_char_to_str(v) for v in arr.tolist()]
                    mp = {}
                    for c in np.unique(y):
                        idx = np.where(y == int(c))[0]
                        cnt = Counter([names[i] for i in idx if names[i] != ""])
                        if cnt: mp[int(c)] = cnt.most_common(1)[0][0]
                    if mp: return mp, f"from MAT `{key}`"
            except Exception:
                pass
    return None, "cannot infer (use --label_map)"

def build_class_names(C, label_map):
    return [str(label_map[c]) if label_map is not None and c in label_map else str(c) for c in range(C)]

class MLPExpert(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=64, emb_dim=64, p_drop=0.2):
        super().__init__()
        self.feat = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(p_drop), nn.Linear(hidden, emb_dim), nn.ReLU())
        self.cls = nn.Linear(emb_dim, out_dim)
    def forward(self, x):
        h = self.feat(x); return self.cls(h), h

class Gate(nn.Module):
    def __init__(self, in_dim, hidden, num_exp):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, num_exp))
    def forward(self, x): return torch.softmax(self.net(x), dim=1)

class SmallSpecCNN(nn.Module):
    def __init__(self, in_channels, cnn_emb_dim=64):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels,32,3,padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.AdaptiveAvgPool2d((4,4)))
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(128*4*4, cnn_emb_dim), nn.ReLU())
    def forward(self, x): return self.fc(self.features(x))

class ImgExpert(nn.Module):
    def __init__(self, img_in_ch, cnn_emb_dim, out_dim, hidden=64, emb_dim=64, p_drop=0.2):
        super().__init__()
        self.cnn = SmallSpecCNN(img_in_ch, cnn_emb_dim)
        self.mlp = nn.Sequential(nn.Linear(cnn_emb_dim, hidden), nn.ReLU(), nn.Dropout(p_drop), nn.Linear(hidden, emb_dim), nn.ReLU())
        self.cls = nn.Linear(emb_dim, out_dim)
    def forward(self, x_img):
        h = self.mlp(self.cnn(x_img)); return self.cls(h), h

class TFExpert(nn.Module):
    def __init__(self, feat_in_dim, spec_in_ch, cnn_emb_dim, out_dim, hidden=64, emb_dim=64, p_drop=0.2):
        super().__init__()
        self.cnn = SmallSpecCNN(spec_in_ch, cnn_emb_dim)
        self.mlp = nn.Sequential(nn.Linear(feat_in_dim+cnn_emb_dim, hidden), nn.ReLU(), nn.Dropout(p_drop), nn.Linear(hidden, emb_dim), nn.ReLU())
        self.cls = nn.Linear(emb_dim, out_dim)
    def forward(self, x_feat, x_spec):
        h = self.mlp(torch.cat([x_feat, self.cnn(x_spec)], dim=1)); return self.cls(h), h

class FeatureFamilyMoE(nn.Module):
    def __init__(self, num_classes, spec_in_ch, topk_dim, occ_in_ch=1, cnn_emb_dim=64, emb_dim=64, p_drop=0.2, gate_hidden=64):
        super().__init__()
        self.num_experts = 4
        self.register_buffer("expert_mask", torch.ones(self.num_experts))
        self.time_slice, self.tf_slice, self.inst_slice = slice(0,17), slice(22,27), slice(27,39)
        self.expert_time = MLPExpert(17, num_classes, 64, emb_dim, p_drop)
        self.expert_freq = ImgExpert(occ_in_ch, cnn_emb_dim, num_classes, 64, emb_dim, p_drop)
        self.expert_tf = TFExpert(5, spec_in_ch, cnn_emb_dim, num_classes, 64, emb_dim, p_drop)
        self.expert_inst = MLPExpert(12, num_classes, 64, emb_dim, p_drop)
        self.gate = Gate(39 + topk_dim + 1, gate_hidden, self.num_experts)
    def forward(self, x_all, x_topk, fs_feat, x_spec, x_occ):
        x_time, x_tf, x_inst = x_all[:, self.time_slice], x_all[:, self.tf_slice], x_all[:, self.inst_slice]
        gate_w = self.gate(torch.cat([x_all, x_topk, fs_feat], dim=1))
        gate_w = gate_w * self.expert_mask.unsqueeze(0)
        gate_w = gate_w / (gate_w.sum(dim=1, keepdim=True) + 1e-12)
        logits_time, h_time = self.expert_time(x_time)
        logits_freq, h_freq = self.expert_freq(x_occ)
        logits_tf, h_tf = self.expert_tf(x_tf, x_spec)
        logits_inst, h_inst = self.expert_inst(x_inst)
        expert_logits = [logits_time, logits_freq, logits_tf, logits_inst]
        expert_embs = [h_time, h_freq, h_tf, h_inst]
        logits_stack = torch.stack(expert_logits, dim=2)
        gate_exp = gate_w.unsqueeze(1)
        logits_fused = torch.sum(logits_stack * gate_exp, dim=2)
        emb_stack = torch.stack(expert_embs, dim=2)
        h_fusion = torch.sum(emb_stack * gate_exp, dim=2)
        return logits_fused, gate_w, {"time":h_time,"freq":h_freq,"tf":h_tf,"inst":h_inst}, expert_logits, h_fusion

class FullDataset(Dataset):
    def __init__(self, X_all_z, X_topk, fs_vec, X_spec, X_occ, y):
        self.X_all = np.asarray(X_all_z, np.float32)
        self.X_topk = np.asarray(X_topk, np.float32)
        self.fs = np.asarray(fs_vec, np.float32).reshape(-1,1)
        self.X_spec = np.asarray(X_spec, np.float32)
        self.X_occ = np.asarray(X_occ, np.float32)
        self.y = np.asarray(y, np.int64)
    def __len__(self): return self.y.shape[0]
    def __getitem__(self, i):
        return (torch.from_numpy(self.X_all[i]).float(),
                torch.from_numpy(self.X_topk[i]).float(),
                torch.from_numpy(self.fs[i]).float(),
                torch.from_numpy(self.X_spec[i]).float(),
                torch.from_numpy(self.X_occ[i]).float(),
                torch.tensor(self.y[i], dtype=torch.long))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mat_all", required=True)
    ap.add_argument("--topk_mat", default="")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out_npz", required=True)
    ap.add_argument("--preproc_npz", required=True)
    ap.add_argument("--label_map", default="")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--emb_dim", type=int, default=64)
    ap.add_argument("--cnn_emb_dim", type=int, default=64)
    ap.add_argument("--gate_hidden", type=int, default=64)
    ap.add_argument("--split_npz", type=str, default="", help="optional split file; used when subset != all")
    ap.add_argument("--subset", type=str, default="all", choices=["all", "train", "val", "test"], help="which subset to export")
    args = ap.parse_args()

    S = load_mat_auto(args.mat_all)
    kX, ky = pick_first_existing(S, ["featureMatrix","feature_matrix","X"]), pick_first_existing(S, ["label_id","device_id","y","labels"])
    if kX is None or ky is None: raise KeyError("MAT missing featureMatrix / label_id")
    X_all = ensure_2d_feature_matrix(S[kX]).astype(np.float32)
    y = to_1d_int(S[ky])
    N, P = X_all.shape
    if P != 39: raise ValueError(f"Expect all39, got P={P}")
    kfs = pick_first_existing(S, ["fsVector","fs","fs_vec"])
    fs_raw = np.zeros((N,), np.float32) if kfs is None else np.asarray(S[kfs]).reshape(-1).astype(np.float32)
    kfid = pick_first_existing(S, ["file_id","fileId","fileID","file_index_id","file_idx"])
    file_id = None
    if kfid is not None:
        fid = np.asarray(S[kfid]).reshape(-1)
        if fid.shape[0] == N: file_id = fid.astype(np.int64)
    kspec = pick_first_existing(S, ["specTensor","spec","spec_tensor","SpecTensor"])
    kocc = pick_first_existing(S, ["occTensor","occ_tensor","densityTensor","occ","occMap"])
    if kspec is None or kocc is None: raise KeyError("MAT missing specTensor / occTensor")
    X_spec = ensure_nchw(S[kspec], N).astype(np.float32, copy=False)
    X_occ = ensure_nchw(S[kocc], N).astype(np.float32, copy=False)

    # PZ = np.load(args.preproc_npz, allow_pickle=True)
    # mean = np.asarray(PZ["mean"]).reshape(1,-1).astype(np.float32)
    # scale = np.asarray(PZ["scale"]).reshape(1,-1).astype(np.float32)
    # X_all_z = ((X_all - mean) / (scale + 1e-12)).astype(np.float32)
    # fs_mean, fs_std = float(PZ["fs_mean"]), float(PZ["fs_std"])
    # fs_vec = ((np.log10(fs_raw + 1.0) - fs_mean) / (fs_std + 1e-12)).astype(np.float32)
    # spec_mu, spec_std = float(PZ["spec_mu"]), float(PZ["spec_std"])
    # X_spec = X_spec.astype(np.float32, copy=True); X_spec -= spec_mu; X_spec /= (spec_std + 1e-12)
    # occ_mu, occ_std = float(PZ["occ_mu"]), float(PZ["occ_std"])
    # X_occ = X_occ.astype(np.float32, copy=True); X_occ -= occ_mu; X_occ /= (occ_std + 1e-12)
    PZ = np.load(args.preproc_npz, allow_pickle=True)

    # all39
    mean = np.asarray(PZ["mean"]).reshape(1, -1).astype(np.float32)
    scale = np.asarray(PZ["scale"]).reshape(1, -1).astype(np.float32)
    X_all_z = ((X_all - mean) / (scale + 1e-12)).astype(np.float32)

    # fs / spec / occ:
    # 优先使用 preproc_scaler.npz 中保存的训练统计量；
    # 如果旧版 npz 没有这些键，则回退为“当前数据全量统计量”
    # 这样至少脚本能跑通，但与训练分布不完全一致。
    if ("fs_mean" in PZ) and ("fs_std" in PZ):
        fs_mean = float(PZ["fs_mean"])
        fs_std = float(PZ["fs_std"])
    else:
        fs_log = np.log10(fs_raw + 1.0)
        fs_mean = float(fs_log.mean())
        fs_std = float(fs_log.std() + 1e-12)
        print("[Warn] preproc npz missing fs_mean/fs_std -> fallback to full-data stats")

    fs_vec = ((np.log10(fs_raw + 1.0) - fs_mean) / (fs_std + 1e-12)).astype(np.float32)

    if ("spec_mu" in PZ) and ("spec_std" in PZ):
        spec_mu = float(PZ["spec_mu"])
        spec_std = float(PZ["spec_std"])
    else:
        spec_mu = float(X_spec.mean())
        spec_std = float(X_spec.std() + 1e-12)
        print("[Warn] preproc npz missing spec_mu/spec_std -> fallback to full-data stats")

    X_spec = X_spec.astype(np.float32, copy=True)
    X_spec -= spec_mu
    X_spec /= (spec_std + 1e-12)

    if ("occ_mu" in PZ) and ("occ_std" in PZ):
        occ_mu = float(PZ["occ_mu"])
        occ_std = float(PZ["occ_std"])
    else:
        occ_mu = float(X_occ.mean())
        occ_std = float(X_occ.std() + 1e-12)
        print("[Warn] preproc npz missing occ_mu/occ_std -> fallback to full-data stats")

    X_occ = X_occ.astype(np.float32, copy=True)
    X_occ -= occ_mu
    X_occ /= (occ_std + 1e-12)

    topk_union = load_topk_union_indices(args.topk_mat, P).astype(np.int64)
    X_topk = X_all_z[:, topk_union] if topk_union.size > 0 else np.zeros((N,0), np.float32)

    label_map = load_label_map_csv(args.label_map)
    if label_map is None:
        label_map, _ = infer_label_map_from_mat(S, y)
    class_names_full = build_class_names(int(np.unique(y).size), label_map)

    # ---------- optional subset filtering ----------
    sel_idx = None
    if args.split_npz and args.subset != "all":
        train_idx, val_idx, test_idx = load_split_indices(args.split_npz)

        if args.subset == "train":
            sel_idx = train_idx
        elif args.subset == "val":
            sel_idx = val_idx
        elif args.subset == "test":
            sel_idx = test_idx
        else:
            raise ValueError(f"unknown subset: {args.subset}")

        print(f"[Subset] export subset={args.subset}, n={len(sel_idx)}")

        X_all_z = X_all_z[sel_idx]
        X_topk  = X_topk[sel_idx]
        fs_vec  = fs_vec[sel_idx]
        X_spec  = X_spec[sel_idx]
        X_occ   = X_occ[sel_idx]
        y       = y[sel_idx]

        if file_id is not None:
            file_id = file_id[sel_idx]
    else:
        print(f"[Subset] export subset=all, n={len(y)}")
    # ------------------------------------------------------------

    ds = FullDataset(X_all_z, X_topk, fs_vec, X_spec, X_occ, y)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device(args.device)
    model = FeatureFamilyMoE(int(np.unique(y).size), X_spec.shape[1], X_topk.shape[1], X_occ.shape[1], args.cnn_emb_dim, args.emb_dim, args.dropout, args.gate_hidden).to(device)

    if not os.path.isfile(args.ckpt): raise FileNotFoundError(args.ckpt)
    state = torch.load(args.ckpt, map_location=device)
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict): state = state["model"]
    model.load_state_dict(state, strict=False)
    model.eval()

    H_list=[]; G_list=[]; y_list=[]; Ht=[]; Hf=[]; Htf=[]; Hi=[]
    with torch.no_grad():
        for xb_all, xb_topk, xfs, xb_spec, xb_occ, yb in loader:
            xb_all, xb_topk, xfs, xb_spec, xb_occ = xb_all.to(device), xb_topk.to(device), xfs.to(device), xb_spec.to(device), xb_occ.to(device)
            _, gate_w, h_dict, _, h_fusion = model(xb_all, xb_topk, xfs, xb_spec, xb_occ)
            H_list.append(h_fusion.cpu().numpy()); G_list.append(gate_w.cpu().numpy()); y_list.append(yb.numpy())
            Ht.append(h_dict["time"].cpu().numpy()); Hf.append(h_dict["freq"].cpu().numpy()); Htf.append(h_dict["tf"].cpu().numpy()); Hi.append(h_dict["inst"].cpu().numpy())
    H=np.concatenate(H_list,0); G=np.concatenate(G_list,0); y_np=np.concatenate(y_list,0)
    np.savez(args.out_npz,
             H=H.astype(np.float32),
             H_exp0=np.concatenate(Ht,0).astype(np.float32),
             H_exp1=np.concatenate(Hf,0).astype(np.float32),
             H_exp2=np.concatenate(Htf,0).astype(np.float32),
             H_exp3=np.concatenate(Hi,0).astype(np.float32),
             y=y_np.astype(np.int64),
             file_id=(file_id if file_id is not None else np.full((N,), -1, dtype=np.int64)),
             fs_vec=fs_vec.astype(np.float32),
             gate_w=G.astype(np.float32),
             class_names=np.asarray(class_names_full, dtype=object))
    print("Saved:", args.out_npz)

if __name__ == "__main__":
    main()
