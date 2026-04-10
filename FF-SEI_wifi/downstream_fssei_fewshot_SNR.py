#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
 downstream_fssei_fewshot_SNR.py

Few-shot episodic evaluation on exported embeddings (H) with strict(file + time_gap).

Input NPZ (keep your current keys; optional extras are auto-detected):
  Required:
    - H: [N, D]     embeddings (e.g., h_fusion)
    - y: [N]        class_id (0..C-1)
  Optional:
    - file_id: [N]  file index per sample (int)
    - gate_w: [N, E] gate weights per sample (float)  -> for gate-delta reports
    - real_time_per_file: [n_files] or (fid,time) pairs
    - real_time / real_time_per_sample: [N]
    - file_path / file_name: [N] string (datetime can be parsed)

Optional meta MAT (only used if NPZ has no usable real_time fields):
  --mat_meta FeatureMatrix_3.mat
    - expects file_index (1 x nFiles struct) with .txtPath or similar

Label map CSV:
  --label_map label_map.csv  (columns: class_id,name)
  If not given, tries ./experiments/label_map.csv, then recursive search under ./experiments/**/label_map.csv

Outputs (all under --out_dir):
  - fewshot_results.csv
  - per_class_acc.csv
  - top_confusions.csv
  - fewshot_cm_counts.png/.npy
  - fewshot_cm_norm.png/.npy
  - top_confusions_gate_delta.csv  (if gate_w exists)
说明：
BT数据集：FeatureMatrix_3.mat    topk_indices.mat
WiFi数据集：FeatureMatrix_OSU_Stable_WiFi_Wireless_unified.mat   topk_indices_wifi.mat

单专家
# Time   python downstream_fssei_fewshot_SNR.py --expert exp0
# Freq   python downstream_fssei_fewshot_SNR.py --expert exp1
# TF     python downstream_fssei_fewshot_SNR.py --expert exp2
# Inst   python downstream_fssei_fewshot_SNR.py --expert exp3
"""

import os
import re
import csv
import glob
import math
import argparse
import time
from datetime import datetime
from typing import Dict, Optional, Tuple, List
from sklearn.metrics import confusion_matrix
from snr_utils import parse_snr_list, run_snr_sweep, draw_hardest_k_cm, draw_representative_cms
from sklearn.metrics import f1_score, confusion_matrix

import numpy as np
import pandas as pd
import moe_gate_visualization as gate_vis

try:
    import seaborn as sns
except Exception:
    sns = None

import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Liberation Serif", "Nimbus Roman", "DejaVu Serif"]
plt.rcParams["axes.unicode_minus"] = False
# ------------------------- time parsing -------------------------

_TIME_PATTERNS = [
    re.compile(r'(?P<Y>\d{4})[-_.]?(?P<m>\d{2})[-_.]?(?P<d>\d{2})[T\s_-]?(?P<H>\d{2})[:\-_.]?(?P<M>\d{2})[:\-_.]?(?P<S>\d{2})'),
    re.compile(r'(?P<Y>\d{4})[-_.]?(?P<m>\d{2})[-_.]?(?P<d>\d{2})[T\s_-]?(?P<H>\d{2})[:\-_.]?(?P<M>\d{2})(?!\d)'),
    re.compile(r'(?P<Y>\d{4})[-_.]?(?P<m>\d{2})[-_.]?(?P<d>\d{2})(?!\d)'),
]


def parse_time_from_string(s: str) -> Optional[float]:
    """Parse datetime from text; return epoch seconds (float) or None."""
    if s is None:
        return None
    s = str(s)
    for rgx in _TIME_PATTERNS:
        m = rgx.search(s)
        if not m:
            continue
        gd = m.groupdict()
        Y = int(gd["Y"]); mo = int(gd["m"]); da = int(gd["d"])
        H = int(gd.get("H") or 0); M = int(gd.get("M") or 0); S = int(gd.get("S") or 0)
        try:
            dt = datetime(Y, mo, da, H, M, S)
            return float(dt.timestamp())
        except Exception:
            continue
    return None


# ------------------------- label map -------------------------

def _try_read_csv_rows(path: str) -> Tuple[Optional[List[List[str]]], Optional[str]]:
    """Return rows (list of list str) and used encoding."""
    encodings = ["utf-8-sig", "utf-8", "gbk", "gb2312", "latin-1"]
    for enc in encodings:
        try:
            with open(path, "r", encoding=enc, newline="") as f:
                txt = f.read()
            delim = "," if txt.count(",") >= txt.count("\t") else "\t"
            rows = []
            for row in csv.reader(txt.splitlines(), delimiter=delim):
                if len(row) == 0:
                    continue
                rows.append([c.strip() for c in row])
            if rows:
                return rows, enc
        except Exception:
            continue
    return None, None


def load_label_map(label_map_path: Optional[str]) -> Tuple[Dict[int, str], Optional[str]]:
    """
    Load mapping class_id -> name from CSV. Tolerates header.
    Returns (map, used_path).
    """
    candidates = []
    if label_map_path:
        candidates.append(label_map_path)
    else:
        candidates.append("./experiments/label_map.csv")
        candidates.append("./label_map.csv")
        # recursive under experiments
        candidates += sorted(glob.glob("./experiments/**/label_map.csv", recursive=True))

    used = None
    for p in candidates:
        if p and os.path.isfile(p):
            used = p
            break

    if not used:
        return {}, None

    rows, enc = _try_read_csv_rows(used)
    if rows is None:
        print(f"[LabelMap][Warn] failed to read {used} with common encodings.")
        return {}, used

    # detect header
    # expected columns: class_id,name
    out = {}
    start = 0
    if len(rows[0]) >= 2:
        h0 = rows[0][0].lower()
        h1 = rows[0][1].lower()
        if ("class" in h0 or "id" in h0) and ("name" in h1):
            start = 1

    bad = 0
    for r in rows[start:]:
        if len(r) < 2:
            continue
        try:
            cid = int(str(r[0]).strip())
            name = str(r[1]).strip()
            if name:
                out[cid] = name
        except Exception:
            bad += 1

    print(f"[LabelMap] loaded {len(out)} names from {used} (encoding={enc}, bad_rows={bad})")
    return out, used


def class_label(cid: int, name_map: Dict[int, str]) -> str:
    if cid in name_map and name_map[cid]:
        # show both name and id to avoid ambiguity
        return f"{name_map[cid]} ({cid})"
    return str(cid)


# ------------------------- MAT loader for meta time -------------------------

def load_mat_auto(path: str):
    import scipy.io as sio
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
                    f"Failed to read '{path}' via scipy (likely v7.3). Please `pip install h5py`.\nOriginal error: {e}"
                )
            return load_mat_v73_h5py(path)
        raise


def load_mat_v73_h5py(path: str):
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


def _mat_struct_get_field(x, field: str):
    # scipy loadmat: matlab struct becomes an object with attributes
    if hasattr(x, field):
        return getattr(x, field)
    if isinstance(x, dict) and field in x:
        return x[field]
    return None


def infer_real_time_per_file_from_mat(mat_path: str, file_id: np.ndarray) -> Tuple[Optional[Dict[int, float]], str]:
    """
    Try build fid->time from FeatureMatrix mat 'file_index' struct's txtPath (or similar).
    Assumes file_index length == num_files.
    Handles fid 0-based or 1-based.
    """
    if not mat_path or (not os.path.isfile(mat_path)):
        return None, "[Time][MAT] mat_meta not found."

    try:
        M = load_mat_auto(mat_path)
    except Exception as e:
        return None, f"[Time][MAT][Warn] load mat failed: {e}"

    if "file_index" not in M:
        return None, "[Time][MAT][Warn] `file_index` not found in mat."

    fi = M["file_index"]
    # file_index could be array-like of structs
    try:
        arr = np.array(fi).reshape(-1)
    except Exception:
        arr = fi

    try:
        n_files = int(len(arr))
    except Exception:
        return None, "[Time][MAT][Warn] file_index is not iterable."

    fmin, fmax = int(file_id.min()), int(file_id.max())

    if fmin >= 1 and fmax <= n_files:
        offset = 1  # fid=1..n_files -> index=fid-1
    elif fmin >= 0 and fmax <= (n_files - 1):
        offset = 0  # fid=0..n_files-1 -> index=fid
    else:
        # try guess: if fmax==n_files maybe 1-based
        offset = 1 if fmax == n_files else 0

    mp = {}
    parsed = 0
    for fid in np.unique(file_id).tolist():
        fid = int(fid)
        idx = fid - offset
        if idx < 0 or idx >= n_files:
            continue
        item = arr[idx]
        # try several fields
        for k in ["txtPath", "filepath", "filePath", "path", "name", "fileName", "filename"]:
            v = _mat_struct_get_field(item, k)
            if v is None:
                continue
            t = parse_time_from_string(v)
            if t is not None:
                mp[fid] = float(t)
                parsed += 1
                break

    if len(mp) == 0:
        return None, "[Time][MAT][Warn] parsed 0 timestamps from file_index.* fields."
    return mp, f"[Time][MAT] parsed timestamps for {len(mp)}/{len(np.unique(file_id))} files (offset={'1-based' if offset==1 else '0-based'})."


# ------------------------- infer real_time_per_file from NPZ -------------------------

def infer_real_time_per_file_from_npz(npz_dict, file_id: np.ndarray) -> Tuple[Optional[Dict[int, float]], str]:
    """
    Return fid->time if possible from NPZ keys.
    """
    file_id = np.asarray(file_id).reshape(-1).astype(np.int64)
    unique_fids = np.sort(np.unique(file_id))

    # 1) preferred: per-file vector aligned to sorted unique fids
    for key in ["real_time_per_file", "realtime_per_file", "rt_per_file", "file_time", "file_time_per_file"]:
        if key in npz_dict:
            tpf = np.array(npz_dict[key])
            if tpf.ndim == 1:
                tpf = tpf.reshape(-1)
                if tpf.size == unique_fids.size:
                    mp = {int(fid): float(tpf[i]) for i, fid in enumerate(unique_fids)}
                    return mp, f"[Time] Using `{key}` (len={tpf.size}) aligned to sorted unique(file_id)."
            if tpf.ndim == 2 and tpf.shape[1] == 2:
                mp = {int(r[0]): float(r[1]) for r in tpf}
                return mp, f"[Time] Using `{key}` as (fid,time) pairs."

    # 2) per-sample vector -> aggregate per-file
    for key in ["real_time", "real_time_per_sample", "realtime", "timestamp", "time"]:
        if key in npz_dict:
            ts = np.array(npz_dict[key]).reshape(-1).astype(np.float64)
            if ts.size == file_id.size:
                mp = {}
                for fid in unique_fids:
                    m = (file_id == fid)
                    if m.any():
                        mp[int(fid)] = float(np.mean(ts[m]))
                return mp, f"[Time] Using `{key}` per-sample -> mean-aggregated to per-file."

    # 3) strings per-sample -> parse -> aggregate per-file
    for key in ["file_name", "filename", "file_path", "filepath", "path", "name"]:
        if key in npz_dict:
            arr = np.array(npz_dict[key]).reshape(-1)
            if arr.size == file_id.size:
                mp_vals = {int(fid): [] for fid in unique_fids}
                parse_ok = 0
                for fid, s in zip(file_id.tolist(), arr.tolist()):
                    t = parse_time_from_string(s)
                    if t is not None:
                        mp_vals[int(fid)].append(float(t))
                        parse_ok += 1
                mp = {fid: float(np.mean(v)) for fid, v in mp_vals.items() if len(v) > 0}
                if len(mp) > 0:
                    return mp, f"[Time] Parsed datetime from `{key}` (parsed_samples={parse_ok}/{arr.size}) -> mean per-file."
                return None, f"[Time][Warn] `{key}` exists but parsed 0 datetimes."

    return None, "[Time][Warn] NPZ time fields missing/unrecognized."


# ------------------------- indexing for strict sampling -------------------------

def build_idx_map_subset(y: np.ndarray, file_id: np.ndarray, subset_idx: np.ndarray):
    """idx_map[c][fid] = np.array(global_indices) for subset only."""
    subset_idx = np.asarray(subset_idx, dtype=np.int64)
    from collections import defaultdict
    tmp = defaultdict(lambda: defaultdict(list))
    for i in subset_idx:
        c = int(y[i])
        fid = int(file_id[i])
        tmp[c][fid].append(int(i))
    out = {}
    for c, d in tmp.items():
        out[c] = {fid: np.asarray(idxs, dtype=np.int64) for fid, idxs in d.items()}
    return out


def sample_indices_roundrobin(file_dict, files, n, rng):
    """Uniform-ish sampling across files (round-robin one per file per round)."""
    files = list(map(int, files))
    if len(files) == 0:
        return None
    pools, total = {}, 0
    for f in files:
        arr = file_dict.get(int(f), None)
        if arr is None:
            continue
        pools[int(f)] = arr.tolist()
        total += len(pools[int(f)])
    if total < n:
        return None

    order = files[:]
    rng.shuffle(order)

    selected = []
    while len(selected) < n:
        progressed = False
        for f in order:
            lst = pools.get(f, [])
            if len(lst) > 0:
                j = int(rng.randint(len(lst)))
                selected.append(lst[j])
                lst[j] = lst[-1]
                lst.pop()
                progressed = True
                if len(selected) >= n:
                    break
        if not progressed:
            break
    if len(selected) < n:
        return None
    return np.asarray(selected, dtype=np.int64)


def build_file_time_by_class(y, file_id, subset_idx, real_time_per_file_map: Optional[Dict[int, float]]):
    """
    file_time[c][fid] -> normalized time in [0,1] per class.
    If real_time_per_file_map is None -> returns None (so strict_file_only will be used).
    """
    if real_time_per_file_map is None:
        return None

    subset_idx = np.asarray(subset_idx, dtype=np.int64)
    subset_mask = np.zeros_like(y, dtype=bool)
    subset_mask[subset_idx] = True

    out = {}
    classes = np.unique(y[subset_idx])
    for c in classes:
        c = int(c)
        mask_c = (y == c) & subset_mask
        fids = np.unique(file_id[mask_c])
        mp = {}
        for fid in fids:
            fid = int(fid)
            if fid in real_time_per_file_map:
                mp[fid] = float(real_time_per_file_map[fid])
        if len(mp) < 2:
            # not enough time to do gap, but strict_file_only still possible
            out[c] = mp
            continue
        vals = np.array(list(mp.values()), dtype=np.float64)
        vmin, vmax = float(vals.min()), float(vals.max())
        denom = max(1e-12, vmax - vmin)
        out[c] = {fid: float((t - vmin) / denom) for fid, t in mp.items()}
    return out


def _choose_disjoint_files_strict(file_dict, fids, rng, n_query, n_shot):
    """Pick disjoint query/support file sets with enough samples (strict file)."""
    fids = list(map(int, fids))
    rng.shuffle(fids)

    q_files, q_cnt = [], 0
    for f in fids:
        if f not in file_dict:
            continue
        if q_cnt >= n_query:
            break
        q_files.append(f)
        q_cnt += file_dict[f].size
    if q_cnt < n_query:
        return None

    q_set = set(q_files)
    s_files, s_cnt = [], 0
    for f in fids:
        if f in q_set:
            continue
        if f not in file_dict:
            continue
        if s_cnt >= n_shot:
            break
        s_files.append(f)
        s_cnt += file_dict[f].size
    if s_cnt < n_shot:
        return None

    return q_files, s_files


def _fileset_time_gap(q_files, s_files, file_time_map: Dict[int, float]) -> Optional[float]:
    """Gap between mean times of the two file sets (normalized)."""
    qt = [file_time_map.get(int(f), None) for f in q_files]
    st = [file_time_map.get(int(f), None) for f in s_files]
    qt = [t for t in qt if t is not None]
    st = [t for t in st if t is not None]
    if len(qt) == 0 or len(st) == 0:
        return None
    return float(abs(np.mean(qt) - np.mean(st)))


def sample_episode_indices_strict(
    idx_map,
    file_time,  # file_time[c][fid]->norm time OR None
    classes,
    n_way,
    n_shot,
    n_query,
    rng,
    strict_time=True,
    min_time_gap=0.30,
    file_uniform=True,
    allow_relaxed=False,
    max_tries=400,
):
    """
    Returns:
      chosen_classes(list), sup_idx(np), qry_idx(np), mode(str)
    mode in {"strict_file_time","strict_file_only","relaxed"}.
    """
    classes = np.asarray(classes, dtype=np.int64)
    if classes.size < n_way:
        return None, None, None, None

    for _ in range(int(max_tries)):
        chosen = rng.choice(classes, size=n_way, replace=False)

        sup_all, qry_all = [], []
        mode_used = "strict_file_time" if strict_time else "strict_file_only"

        ok = True
        for c in chosen:
            c = int(c)
            if c not in idx_map:
                ok = False; break
            file_dict = idx_map[c]
            fids = list(file_dict.keys())
            # if len(fids) < 2:
            #     ok = False; break
            if len(fids) < 2:
                if allow_relaxed:
                    pool = np.concatenate(list(file_dict.values()), axis=0)
                    if pool.size < (n_query + n_shot):
                        ok = False
                        break
                    pick = rng.choice(pool, size=(n_query + n_shot), replace=False)
                    qry = pick[:n_query]
                    sup = pick[n_query:]
                    qry_all.append(qry)
                    sup_all.append(sup)
                    mode_used = "relaxed_single_file"
                    continue
                else:
                    ok = False
                    break

            sets = _choose_disjoint_files_strict(file_dict, fids, rng, n_query=n_query, n_shot=n_shot)
            if sets is None:
                if allow_relaxed:
                    pool = np.concatenate(list(file_dict.values()), axis=0)
                    if pool.size < (n_query + n_shot):
                        ok = False; break
                    pick = rng.choice(pool, size=(n_query + n_shot), replace=False)
                    qry = pick[:n_query]
                    sup = pick[n_query:]
                    qry_all.append(qry)
                    sup_all.append(sup)
                    mode_used = "relaxed"
                    continue
                ok = False; break

            q_files, s_files = sets

            # strict_time: also require time gap when time info exists
            if strict_time and (file_time is not None) and (c in file_time) and isinstance(file_time[c], dict) and len(file_time[c]) > 0:
                gap = _fileset_time_gap(q_files, s_files, file_time[c])
                if gap is None or gap < min_time_gap:
                    ok = False; break
                mode_used = "strict_file_time"
            else:
                mode_used = "strict_file_only" if mode_used != "relaxed" else mode_used

            if file_uniform:
                qry = sample_indices_roundrobin(file_dict, q_files, n_query, rng)
                sup = sample_indices_roundrobin(file_dict, s_files, n_shot, rng)
            else:
                q_pool = np.concatenate([file_dict[int(f)] for f in q_files], axis=0)
                s_pool = np.concatenate([file_dict[int(f)] for f in s_files], axis=0)
                if q_pool.size < n_query or s_pool.size < n_shot:
                    ok = False; break
                qry = rng.choice(q_pool, size=n_query, replace=False)
                sup = rng.choice(s_pool, size=n_shot, replace=False)

            if qry is None or sup is None:
                ok = False; break
            qry_all.append(qry)
            sup_all.append(sup)

        if ok:
            return chosen.tolist(), np.concatenate(sup_all), np.concatenate(qry_all), mode_used

    return None, None, None, None


# ------------------------- proto classifier & whitening -------------------------

def fit_whitener(H, eps=1e-4, shrink=1e-3):
    """Return (mu, W) s.t. (x-mu)@W approximately whitened."""
    H = np.asarray(H, dtype=np.float64)
    mu = H.mean(axis=0, keepdims=True)
    X = H - mu
    cov = (X.T @ X) / max(1, (X.shape[0] - 1))
    cov = (1.0 - shrink) * cov + shrink * np.eye(cov.shape[0], dtype=np.float64)
    evals, evecs = np.linalg.eigh(cov)
    evals = np.maximum(evals, eps)
    W = evecs @ np.diag(1.0 / np.sqrt(evals)) @ evecs.T
    return mu.astype(np.float32), W.astype(np.float32)


def apply_whiten(H, mu, W):
    return (H - mu) @ W


def l2norm(x, axis=1, eps=1e-12):
    n = np.sqrt(np.sum(x * x, axis=axis, keepdims=True) + eps)
    return x / n


def proto_predict_cosine(H_sup, y_sup, H_qry, chosen_classes, proto_shrink=0.2):
    """
    chosen_classes: list of global class ids in this episode
    Return:
      pred_global: [Nq] predicted global labels
    """
    cls = list(map(int, chosen_classes))
    cls_to_local = {c: i for i, c in enumerate(cls)}
    y_sup_local = np.array([cls_to_local[int(v)] for v in y_sup], dtype=np.int64)

    Zs = l2norm(H_sup.astype(np.float32))
    Zq = l2norm(H_qry.astype(np.float32))

    W = len(cls)
    protos = np.zeros((W, Zs.shape[1]), dtype=np.float32)
    for i in range(W):
        m = (y_sup_local == i)
        protos[i] = Zs[m].mean(axis=0)
    protos = l2norm(protos)

    if proto_shrink and proto_shrink > 0:
        g = l2norm(protos.mean(axis=0, keepdims=True))
        protos = l2norm((1.0 - proto_shrink) * protos + proto_shrink * g)

    logits = Zq @ protos.T
    pred_local = np.argmax(logits, axis=1)
    pred_global = np.array([cls[i] for i in pred_local], dtype=np.int64)
    return pred_global

# ------------------------- SNR 用的简单 ProtoNet 评估 -------------------------

def eval_fewshot_protonet_strict_set(
    H, y, novel_classes, n_way, n_shot, n_query,
    episodes, idx_map, seed=0,
    file_time=None, use_time_sep=True, min_time_gap=0.30,
    mu=None, W=None, strict_time=True, file_uniform=True,
    allow_relaxed=False, max_episode_tries=50,
    proto_shrink=0.0,
    return_details=False,   # ★新增：是否返回更多指标
):
    """
    严格版 few-shot 评估（给 SNR 曲线用）：
      - 和主脚本后半段逻辑一致：按 file + time_gap 严格采样；
      - 若提供 mu, W，则对 support/query 先做 whitening；
      - 用 proto_predict_cosine 做原型分类；
      - 返回 (mean_acc, std_acc)。

    注意：
      - idx_map / file_time 必须是基于“干净数据的 y 和 file_id”预先构建好的；
      - H 允许是加噪后的 H_snr，只是在 episodes 里被索引使用。
    """  
    rng = np.random.RandomState(seed)
    H = np.asarray(H, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)

    eval_classes = np.asarray(novel_classes, dtype=np.int64)

    accs = []
    all_true = []
    all_pred = []
    all_qry_idx = []

    ep = 0
    attempts = 0
    max_attempts = max(1, episodes * max_episode_tries)

    while ep < episodes and attempts < max_attempts:
        attempts += 1

        if idx_map is None:
            chosen = rng.choice(eval_classes, size=n_way, replace=False)
            sup_idx, qry_idx = [], []
            ok = True
            for c in chosen:
                c = int(c)
                pool = np.where(y == c)[0]
                if pool.size < (n_shot + n_query):
                    ok = False
                    break
                pick = rng.choice(pool, size=(n_shot + n_query), replace=False)
                sup_idx.append(pick[:n_shot])
                qry_idx.append(pick[n_shot:])
            if not ok:
                continue
            sup_idx = np.concatenate(sup_idx)
            qry_idx = np.concatenate(qry_idx)
            chosen = chosen.tolist()
        else:
            chosen, sup_idx, qry_idx, mode = sample_episode_indices_strict(
                idx_map=idx_map,
                file_time=file_time if use_time_sep else None,
                classes=eval_classes,
                n_way=n_way,
                n_shot=n_shot,
                n_query=n_query,
                rng=rng,
                strict_time=(strict_time and use_time_sep),
                min_time_gap=min_time_gap,
                file_uniform=file_uniform,
                allow_relaxed=allow_relaxed,
                max_tries=max_episode_tries,
            )
            if mode is None:
                continue

        y_sup = y[sup_idx]
        y_qry = y[qry_idx]
        H_sup = H[sup_idx]
        H_qry = H[qry_idx]

        # whitening（与你现在逻辑一致）
        if mu is not None and W is not None:
            H_sup = apply_whiten(H_sup, mu, W)
            H_qry = apply_whiten(H_qry, mu, W)

        pred_global = proto_predict_cosine(H_sup, y_sup, H_qry, chosen, proto_shrink=proto_shrink)

        accs.append(float(np.mean(pred_global == y_qry)))

        if return_details:
            all_true.append(y_qry.copy())
            all_pred.append(pred_global.copy())
            all_qry_idx.append(qry_idx.copy())

        ep += 1

    if len(accs) == 0:
        if return_details:
            return {"mean_acc": 0.0, "std_acc": 0.0, "macro_f1": 0.0,
                    "cm": None, "per_class_acc": None}
        return 0.0, 0.0

    acc_array = np.asarray(accs, dtype=np.float32)
    mean_acc = float(acc_array.mean())
    std_acc = float(acc_array.std(ddof=0))

    if not return_details:
        return mean_acc, std_acc

    y_true = np.concatenate(all_true, axis=0) if len(all_true) else np.array([], dtype=np.int64)
    y_pred = np.concatenate(all_pred, axis=0) if len(all_pred) else np.array([], dtype=np.int64)

    # 只对 eval_classes 做指标（避免别的类混进来）
    labels = np.asarray(sorted(list(set(eval_classes.tolist()))), dtype=np.int64)

    if y_true.size == 0:
        macro_f1 = 0.0
        cm = None
        per_class_acc = None
    else:
        macro_f1 = float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        cm_norm = cm.astype(np.float64) / (cm.sum(axis=1, keepdims=True) + 1e-12)
        per_class_acc = np.diag(cm_norm).astype(np.float64)
    
    return {
        "mean_acc": mean_acc,
        "std_acc": std_acc,
        "macro_f1": macro_f1,
        "labels": labels,
        "cm": cm,                    # counts
        "per_class_acc": per_class_acc,
        "all_true": all_true,
        "all_pred": all_pred,
        "all_qry_idx": all_qry_idx,
    }


# ------------------------- Guass Noise -----------------------
def add_gaussian_noise_by_snr(X, snr_db, seed=0):
    """
    对特征矩阵 X 按给定 SNR(dB) 加高斯噪声。
    X: [N, D]
    snr_db: float，例如 30 / 20 / 10 / 0
    """
    if snr_db is None:
        return X.copy()

    rng = np.random.RandomState(seed)
    X = np.asarray(X, dtype=np.float32)

    # 每个样本自己的信号功率
    sig_pow = np.mean(X.astype(np.float64) ** 2, axis=1, keepdims=True) + 1e-12

    # SNR = P_signal / P_noise
    noise_pow = sig_pow / (10.0 ** (float(snr_db) / 10.0))

    noise = rng.randn(*X.shape).astype(np.float32) * np.sqrt(noise_pow).astype(np.float32)
    return (X + noise).astype(np.float32)


def save_confusion_plots(
    cm,
    labels_str,
    out_prefix,
    n_way=None,
    n_shot=None,
    annot_mode="off",          # off | error | all
    annot_min_count=1,        
    annot_diag=False,          # 是否在对角线上也标数字
    annot_fontsize=7,
):
    os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)

    cm = np.asarray(cm, dtype=np.int64)
    np.save(out_prefix + "_counts.npy", cm.astype(np.int64))
    cm_norm = cm.astype(np.float64) / (cm.sum(axis=1, keepdims=True) + 1e-12)
    np.save(out_prefix + "_norm.npy", cm_norm.astype(np.float32))

    if plt is None:
        print("[Plot][Warn] matplotlib not available -> skip png plots.")
        return

    if (n_way is not None) and (n_shot is not None):
        title_prefix = f"{n_way}-way {n_shot}-shot "
    else:
        title_prefix = ""

    def _build_annot_text():
        K = cm.shape[0]
        ann = np.full((K, K), "", dtype=object)
        for i in range(K):
            for j in range(K):
                val = int(cm[i, j])

                if val < int(annot_min_count):
                    continue

                # error: 只显示错误格子（off-diagonal）
                if annot_mode == "error":
                    if i != j:
                        ann[i, j] = str(val)
                    elif annot_diag:
                        ann[i, j] = str(val)

                # all: 所有格子都可显示
                elif annot_mode == "all":
                    if (i != j) or annot_diag:
                        ann[i, j] = str(val)

        return ann

    annot_text = _build_annot_text()

    def _heatmap(mat, title, path, vmin=None, vmax=None, use_ann=False):
        plt.figure(figsize=(10, 8))
        if sns is not None:
            ax = sns.heatmap(
                mat,
                annot=(annot_text if use_ann and annot_mode != "off" else False),
                fmt="",
                cmap="Blues",
                xticklabels=labels_str,
                yticklabels=labels_str,
                vmin=vmin,
                vmax=vmax,
                annot_kws={"fontsize": annot_fontsize},
            )
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
            ax.tick_params(axis="x", labelsize=8)
            ax.tick_params(axis="y", labelsize=8)
        else:
            plt.imshow(mat, aspect="auto", vmin=vmin, vmax=vmax)
            plt.colorbar()
            plt.xticks(range(len(labels_str)), labels_str, rotation=45, ha="right")
            plt.yticks(range(len(labels_str)), labels_str)

            if use_ann and annot_mode != "off":
                K = cm.shape[0]
                for i in range(K):
                    for j in range(K):
                        txt = annot_text[i, j]
                        if txt != "":
                            plt.text(j + 0.5, i + 0.5, txt,
                                     ha="center", va="center", fontsize=annot_fontsize, color="black")

        plt.xlabel("Pred")
        plt.ylabel("True")
        plt.title(title)
        plt.xticks(rotation=35, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(path, dpi=300)
        plt.close()

    _heatmap(
        cm,
        f"{title_prefix} Confusion (Counts)",
        out_prefix + "_counts.png",
        use_ann=True
    )

    # norm 图：底图是归一化，但叠加的仍然是错误个数，便于同时看比例和错误数
    _heatmap(
        cm_norm,
        f"{title_prefix} Confusion (Row-normalized)",
        out_prefix + "_norm.png",
        vmin=0,
        vmax=1,
        use_ann=True
    )


def top_confusions_from_cm(cm, labels_ids, topk=20, min_count=10):
    cm = cm.astype(np.int64)
    cm_norm = cm.astype(np.float64) / (cm.sum(axis=1, keepdims=True) + 1e-12)
    rows = []
    K = cm.shape[0]
    for i in range(K):
        for j in range(K):
            if i == j:
                continue
            cnt = int(cm[i, j])
            if cnt < min_count:
                continue
            rows.append((float(cm_norm[i, j]), cnt, int(labels_ids[i]), int(labels_ids[j])))
    rows.sort(key=lambda x: (x[0], x[1]), reverse=True)
    out = []
    for r, cnt, t, p in rows[:topk]:
        out.append({
            "true_id": t,
            "pred_id": p,
            "rate_row_norm": float(r),
            "count": int(cnt),
        })
    return out


def compute_gate_delta_rows(
    true_ids: np.ndarray,
    pred_ids: np.ndarray,
    gate_qry: np.ndarray,
    eval_ids: List[int],
    name_map: Dict[int, str],
    top_pairs: List[dict],
    expert_names: List[str],
    min_count: int = 10,
):
    """
    For each top confusion pair (t->p):
      compare mean gate on confused samples vs mean gate on correctly classified samples of same true class.
    """
    if gate_qry is None:
        return []

    eval_set = set(map(int, eval_ids))
    E = gate_qry.shape[1]
    rows = []
    for item in top_pairs:
        t = int(item["true_id"]); p = int(item["pred_id"])
        if t not in eval_set or p not in eval_set:
            continue
        m_conf = (true_ids == t) & (pred_ids == p)
        m_corr = (true_ids == t) & (pred_ids == t)
        n_conf = int(m_conf.sum()); n_corr = int(m_corr.sum())
        if n_conf < min_count or n_corr < min_count:
            continue
        g_conf = gate_qry[m_conf].mean(axis=0)
        g_corr = gate_qry[m_corr].mean(axis=0)
        delta = g_conf - g_corr

        row = {
            "true_id": t,
            "true_name": name_map.get(t, ""),
            "pred_id": p,
            "pred_name": name_map.get(p, ""),
            "count_conf": n_conf,
            "count_corr": n_corr,
            "rate_row_norm": float(item.get("rate_row_norm", np.nan)),
        }
        for ei in range(E):
            nm = expert_names[ei] if ei < len(expert_names) else f"Expert{ei}"
            row[f"{nm}_gate_conf"] = float(g_conf[ei])
            row[f"{nm}_gate_corr"] = float(g_corr[ei])
            row[f"{nm}_delta"] = float(delta[ei])

        row["top_delta_expert"] = expert_names[int(np.argmax(delta))] if len(expert_names) >= E else str(int(np.argmax(delta)))
        rows.append(row)
    return rows


def compute_gate_mean_row(
    gate_qry: np.ndarray,
    sample_size: int,
    expert_names: List[str],
):
    """
    Compute one heatmap row:
      SampleSize, Time, Freq, TF, Inst
    using mean gate weights over all query samples.
    """
    if gate_qry is None or len(gate_qry) == 0:
        return None

    g_mean = gate_qry.mean(axis=0)   # [E]

    # 统一成论文命名
    def canon_name(s: str) -> str:
        sl = str(s).strip().lower()
        if sl in ["time", "t", "expert_time", "time_expert", "e1"] or ("time" in sl and "freq" not in sl):
            return "Time"
        if sl in ["freq", "frequency", "f", "expert_freq", "freq_expert", "e2"] or ("freq" in sl and "time" not in sl):
            return "Freq"
        if sl in ["tf", "expert_tf", "tf_expert", "e3"] or ("time" in sl and "freq" in sl):
            return "TF"
        if sl in ["inst", "instance", "instantaneous", "expert_inst", "inst_expert", "e4"] or ("inst" in sl):
            return "Inst"
        return str(s)

    row = {"SampleSize": int(sample_size)}

    for ei in range(len(g_mean)):
        nm = expert_names[ei] if ei < len(expert_names) else f"Expert{ei}"
        row[canon_name(nm)] = float(g_mean[ei])

    # 保证四列都在
    for c in ["Time", "Freq", "TF", "Inst"]:
        if c not in row:
            row[c] = np.nan

    return row


def _canon_expert_name(s: str) -> str:
    sl = str(s).strip().lower()
    if sl in ["time", "t", "expert_time", "time_expert", "e1"] or ("time" in sl and "freq" not in sl):
        return "Time"
    if sl in ["freq", "frequency", "f", "expert_freq", "freq_expert", "e2"] or ("freq" in sl and "time" not in sl):
        return "Freq"
    if sl in ["tf", "expert_tf", "tf_expert", "e3"] or ("time" in sl and "freq" in sl):
        return "TF"
    if sl in ["inst", "instance", "instantaneous", "expert_inst", "inst_expert", "e4"] or ("inst" in sl):
        return "Inst"
    return str(s)


def build_gate_raw_rows(
    gate_qry: np.ndarray,
    qry_idx: np.ndarray,
    y_qry: np.ndarray,
    pred_qry: np.ndarray,
    sample_size: int,
    episode_id: int,
    expert_names: List[str],
    get_name_fn=None,
    snr_name: str = "clean",
):
    """
    保存样本级 gate 表：每个 query 样本一行
    """
    if gate_qry is None or len(gate_qry) == 0:
        return []

    gate_qry = np.asarray(gate_qry)
    qry_idx = np.asarray(qry_idx).reshape(-1)
    y_qry = np.asarray(y_qry).reshape(-1)
    pred_qry = np.asarray(pred_qry).reshape(-1)

    Nq, E = gate_qry.shape
    rows = []

    canon_names = []
    for ei in range(E):
        nm = expert_names[ei] if ei < len(expert_names) else f"Expert{ei}"
        canon_names.append(_canon_expert_name(nm))

    for i in range(Nq):
        t = int(y_qry[i])
        p = int(pred_qry[i])

        row = {
            "SampleSize": int(sample_size),
            "Episode": int(episode_id),
            "QueryIndexInEpisode": int(i),
            "GlobalIndex": int(qry_idx[i]),
            "snr": str(snr_name),
            "true_id": t,
            "pred_id": p,
            "is_correct": int(t == p),
        }

        if get_name_fn is not None:
            row["true_name"] = str(get_name_fn(t))
            row["pred_name"] = str(get_name_fn(p))

        for ei in range(E):
            row[canon_names[ei]] = float(gate_qry[i, ei])

        # 保证四列都在
        for c in ["Time", "Freq", "TF", "Inst"]:
            if c not in row:
                row[c] = np.nan

        # top1 信息
        top1_idx = int(np.argmax(gate_qry[i]))
        row["Top1Expert"] = canon_names[top1_idx] if top1_idx < len(canon_names) else str(top1_idx)
        row["Top1Weight"] = float(gate_qry[i, top1_idx])

        rows.append(row)

    return rows


def aggregate_gate_usage_table(
    df_raw: pd.DataFrame,
    group_col: str = ("SampleSize", "snr"),
    mode: str = "mean",
    threshold: float = 0.35,
    topk: int = 2,
):
    """
    从样本级 gate raw 表聚合为：
      - mean
      - top1
      - topk
      - threshold
    """
    expert_cols = [c for c in ["Time", "Freq", "TF", "Inst"] if c in df_raw.columns]
    if len(expert_cols) == 0 or len(df_raw) == 0:
        return pd.DataFrame()

    # 关键修正：统一转成 list，供 groupby 使用
    group_keys = list(group_col) if isinstance(group_col, (list, tuple)) else [group_col]

    rows = []
    for gval, sub in df_raw.groupby(group_keys, sort=False):
        arr = sub[expert_cols].to_numpy(dtype=float)

        if mode == "mean":
            vals = np.nanmean(arr, axis=0)

        elif mode == "top1":
            idx = np.nanargmax(arr, axis=1)
            vals = np.array([(idx == i).mean() for i in range(len(expert_cols))], dtype=float)

        elif mode == "topk":
            k = min(int(topk), arr.shape[1])
            idx_topk = np.argsort(-arr, axis=1)[:, :k]
            vals = np.array([(idx_topk == i).any(axis=1).mean() for i in range(len(expert_cols))], dtype=float)

        elif mode == "threshold":
            vals = np.array([(arr[:, i] >= threshold).mean() for i in range(len(expert_cols))], dtype=float)

        else:
            raise ValueError(f"Unknown mode: {mode}")

        row = {}
        if isinstance(gval, tuple):
            for k, v in zip(group_keys, gval):
                row[k] = v
        else:
            row[group_keys[0]] = gval

        for c, v in zip(expert_cols, vals):
            row[c] = float(v)

        rows.append(row)

    out = pd.DataFrame(rows)
    keep_cols = group_keys + expert_cols
    return out[keep_cols]


def save_gate_usage_tables(
    df_raw: pd.DataFrame,
    out_dir: str,
    sample_size: int,
    threshold: float = 0.35,
    topk: int = 2,
):
    """
    一次性保存：
      - raw
      - mean
      - top1
      - topk
      - threshold
    """
    os.makedirs(out_dir, exist_ok=True)

    out_raw = os.path.join(out_dir, "ffmoe_gate_raw_records.csv")
    df_raw.to_csv(out_raw, index=False, encoding="utf-8-sig")
    print("Saved:", out_raw)

    df_mean = aggregate_gate_usage_table(df_raw, group_col=("SampleSize", "snr"), mode="mean")
    out_mean = os.path.join(out_dir, "ffmoe_gate_heatmap_table.csv")
    df_mean.to_csv(out_mean, index=False, encoding="utf-8-sig")
    print("Saved:", out_mean)

    df_top1 = aggregate_gate_usage_table(df_raw, group_col=("SampleSize", "snr"), mode="top1")
    out_top1 = os.path.join(out_dir, "ffmoe_gate_top1_freq_table.csv")
    df_top1.to_csv(out_top1, index=False, encoding="utf-8-sig")
    print("Saved:", out_top1)

    df_topk = aggregate_gate_usage_table(df_raw, group_col=("SampleSize", "snr"), mode="topk", topk=topk)
    out_topk = os.path.join(out_dir, f"ffmoe_gate_top{topk}_freq_table.csv")
    df_topk.to_csv(out_topk, index=False, encoding="utf-8-sig")
    print("Saved:", out_topk)

    df_thr = aggregate_gate_usage_table(df_raw, group_col=("SampleSize", "snr"), mode="threshold", threshold=threshold)
    out_thr = os.path.join(out_dir, f"ffmoe_gate_thr_{threshold:.2f}_freq_table.csv")
    df_thr.to_csv(out_thr, index=False, encoding="utf-8-sig")
    print("Saved:", out_thr)

# ------------------------- main -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=str, default="./result/exports/hfusion_osu_stable_joint_noinst_consistent_test.npz")     # ours:./result/exports/hfusion_osu_stable_joint_noinst_consistent.npz   const: ./const_experiment/experiments/e2_dl_finezero/finezero_export/finezero_embeddings_test.npz
    ap.add_argument("--out_dir", type=str, default="result/fewshot/FF-MoE_test")      # ours:result/fewshot/FF-MoE    const:const_experiment/fewshot_test/finezero
    ap.add_argument("--seed", type=int, default=42)

    # ---------- [SNR] 相关参数 ----------
    ap.add_argument(
        "--snr_db_list",
        type=str,
        default="clean,30,25,20,15,10,5,0",
        help=" 'clean,30,25,20,15,10,5,0'，其中 clean/inf 表示不加噪声（基线）",
    )
    ap.add_argument(
        "--no_time_sep",
        action="store_true",
        help="（仅影响 SNR 评估函数的入参标记）如果设置，则 eval_kwargs['use_time_sep']=False",
    )
    ap.add_argument("--save_json", type=str, default="", help="把E4-A summary保存成json")

    # few-shot setting
    ap.add_argument("--n_way", type=int, default=5)
    ap.add_argument("--n_shot", type=int, default=10)
    ap.add_argument("--n_query", type=int, default=15)
    ap.add_argument("--episodes", type=int, default=200)

    # strict settings
    ap.add_argument("--strict_time", action="store_true", default=True)
    ap.add_argument("--min_time_gap", type=float, default=0.30)
    ap.add_argument("--file_uniform", action="store_true", default=True)
    ap.add_argument("--allow_relaxed", action="store_true", default=True)  # 
    ap.add_argument("--max_episode_tries", type=int, default=600)

    # whitening & proto
    ap.add_argument("--whiten", action="store_true", default=True)
    ap.add_argument("--whiten_eps", type=float, default=1e-4)
    ap.add_argument("--whiten_shrink", type=float, default=1e-3)
    ap.add_argument("--proto_shrink", type=float, default=0.2)

    # label map & meta time
    ap.add_argument("--label_map", type=str, default=None,
                    help="CSV with columns: class_id,name. default tries ./experiments/label_map.csv")
    ap.add_argument("--mat_meta", type=str, default=None,
                    help="FeatureMatrix_*.mat to parse file_index.txtPath for real_time if NPZ lacks time fields.")

    # reports
    ap.add_argument("--topk_conf", type=int, default=25)
    ap.add_argument("--topk_min_count", type=int, default=10)
    ap.add_argument("--expert_names", type=str, default="time, freq, tf, inst",
                help="Comma-separated expert names for gate reports.")
                
    ap.add_argument("--expert", type=str, default="fusion", choices=["fusion", "exp0", "exp1", "exp2", "exp3"],
                help="embedding source: fusion or single expert")

    # OPTIONAL legacy base/novel split (default disabled)
    ap.add_argument("--n_base", type=int, default=0, help="(legacy) if >0 and n_novel>0, evaluate only novel.")
    ap.add_argument("--n_novel", type=int, default=0, help="(legacy) if >0 with n_base, evaluate only novel.")
    
    # picture
    ap.add_argument("--cm_annot_mode", type=str, default="all", choices=["off", "error", "all"],
                    help="confusion matrix annotation mode: off | error | all")
    ap.add_argument("--cm_annot_min_count", type=int, default=20,
                    help="only annotate cells whose count >= this threshold")
    ap.add_argument("--cm_annot_diag", action="store_true",
                    help="also annotate diagonal cells")
    ap.add_argument("--cm_annot_fontsize", type=int, default=7,
                    help="fontsize for confusion matrix annotations")

    args = ap.parse_args()

    rng = np.random.RandomState(args.seed)
    # os.makedirs(args.out_dir, exist_ok=True)
    out_dir_final = os.path.join(args.out_dir, f"way_{args.n_way}_shot_{args.n_shot}")
    os.makedirs(out_dir_final, exist_ok=True)

    # --------- load label map -------------
    name_map, used_label_map = load_label_map(args.label_map)
    if used_label_map:
        # save a copy for traceability
        try:
            dst = os.path.join(out_dir_final, "label_map_used.csv")
            with open(used_label_map, "r", encoding="utf-8", errors="ignore") as fr, open(dst, "w", encoding="utf-8") as fw:
                fw.write(fr.read())
            print("Saved:", dst)
        except Exception:
            pass

    # --------------- load npz ------------------
    Z = np.load(args.npz, allow_pickle=True)
    # H = np.array(Z["H"]).astype(np.float32)
    if args.expert == "fusion":
        H = np.array(Z["H"]).astype(np.float32)
    else:
        H = np.array(Z[f"H_{args.expert}"]).astype(np.float32)
    y = np.array(Z["y"]).reshape(-1).astype(np.int64)
    N, D = H.shape

    # 优先从 NPZ 里读 class_names（按类别 ID 排序的设备名）
    raw_cn = Z.get("class_names", None)
    if raw_cn is None:
        class_names = []
    else:
        class_names = [str(x) for x in list(raw_cn)]
    # 提前一次性加载 class_names，在 get_name() 和后面的混淆矩阵都用这一个
    raw_cn = Z.get("class_names", None)
    if raw_cn is None:
        class_names = []
    else:
        class_names = [str(x) for x in list(raw_cn)]

    def get_name(cid: int) -> str:
        """优先 NPZ 的 class_names，其次 label_map，最后用数字。"""
        if cid < len(class_names):
            nm = class_names[cid]
            if nm is not None and str(nm).strip() != "":
                return str(nm)
        if cid in name_map and name_map[cid]:
            return str(name_map[cid])
        return str(cid)
        
    
    file_id = None
    if "file_id" in Z:
        file_id = np.array(Z["file_id"]).reshape(-1).astype(np.int64)
        if file_id.size != N:
            print("[Warn] file_id length mismatch -> disable strict(file/time).")
            file_id = None
    
    gate_w = None
    
    if "gate_w" in Z:
        gw = np.array(Z["gate_w"]).astype(np.float32)
        if gw.ndim == 2 and gw.shape[0] == N:
            gate_w = gw

    expert_names = [s.strip() for s in args.expert_names.split(",") if s.strip()]

    classes_sorted = np.sort(np.unique(y))
    num_classes = int(classes_sorted.size)

    # ---------- eval / base 类划分 ----------
    use_base_novel = (args.n_base > 0 and args.n_novel > 0)
    if use_base_novel:
        if args.n_base + args.n_novel > num_classes:
            raise ValueError(f"n_base+n_novel exceeds total classes: {args.n_base}+{args.n_novel} > {num_classes}")
        base_classes = classes_sorted[:args.n_base]
        eval_classes = classes_sorted[args.n_base:args.n_base + args.n_novel]
        print(f"[EvalSet] legacy split enabled. base={base_classes.tolist()} eval(novel)={eval_classes.tolist()}")
    else:
        base_classes = np.array([], dtype=np.int64)
        eval_classes = classes_sorted
        print(f"[EvalSet] using ALL classes for episodic few-shot: {eval_classes.tolist()}")

    idx_all = np.arange(N, dtype=np.int64)
    idx_eval = idx_all[np.isin(y, eval_classes)]
    idx_base = idx_all[np.isin(y, base_classes)] if use_base_novel else np.array([], dtype=np.int64)

    print(f"Loaded {args.npz}: H={H.shape} | classes={num_classes} | file_id={'yes' if file_id is not None else 'no'} | gate_w={'yes' if gate_w is not None else 'no'}")

    # ---------- time 信息 ----------
    rt_map = None
    rt_info = None
    if file_id is not None:
        rt_map, rt_info = infer_real_time_per_file_from_npz(Z, file_id)
        if rt_map is None and args.mat_meta:
            rt_map, rt_info2 = infer_real_time_per_file_from_mat(args.mat_meta, file_id)
            rt_info = (rt_info + " | " + rt_info2) if rt_info else rt_info2
    else:
        rt_info = "[Time][Warn] file_id missing -> strict(file/time) disabled."
    print(rt_info if rt_info else "[Time][Warn] time unavailable.")

    # build idx_map and file_time (only for eval subset)
    if file_id is None:
        idx_map = None
        file_time = None
    else:
        idx_map = build_idx_map_subset(y, file_id, idx_eval)
        file_time = build_file_time_by_class(y, file_id, idx_eval, rt_map)

    # ---------- 采样可行性 debug(feasibility debug) ---------- 
    print("\n[Debug] EVAL per-class stats (for sampling feasibility):")
    if file_id is not None:
        for c in eval_classes.tolist():
            c = int(c)
            fdict = idx_map.get(c, {})
            sizes = [len(v) for v in fdict.values()] if len(fdict) > 0 else [0]
            n_files = len(fdict)
            n = int(np.sum(sizes))
            mn, mx = int(np.min(sizes)), int(np.max(sizes))
            ok_strict = (n_files >= 2) and (n >= args.n_shot + args.n_query)
            print(f"  class {c:2d} ({class_label(c, name_map)}): n={n}, n_files={n_files} (min={mn}, max={mx}), ok_strict={ok_strict}")
    else:
        for c in eval_classes.tolist():
            c = int(c)
            n = int(np.sum(y[idx_eval] == c))
            print(f"  class {c:2d} ({class_label(c, name_map)}): n={n}")

    # ---------- whitening（主评估用） ---------
    mu = None
    W = None
    if args.whiten:
        fit_idx = idx_base if (use_base_novel and idx_base.size > 0) else idx_eval
        mu, W = fit_whitener(H[fit_idx], eps=args.whiten_eps, shrink=args.whiten_shrink)
        src = "BASE" if (use_base_novel and idx_base.size > 0) else "EVAL/ALL"
        print(f"\n[Whiten] fit from {src}: n={fit_idx.size}, D={D}")

    # ---------- [SNR] SNR 曲线估计 ----------
    novel_classes = eval_classes.copy()
    snr_cfgs = parse_snr_list(args.snr_db_list)  # [SNR]
    eval_kwargs = dict(                          # [SNR]
        novel_classes=novel_classes,
        n_way=args.n_way,
        n_shot=args.n_shot,
        n_query=args.n_query,
        episodes=args.episodes,
        idx_map=idx_map,
        seed=args.seed,
        file_time=file_time,
        use_time_sep=not args.no_time_sep,
        min_time_gap=args.min_time_gap,
        mu=mu,
        W=W,
        strict_time=args.strict_time,
        file_uniform=args.file_uniform,
        allow_relaxed=args.allow_relaxed,
        max_episode_tries=args.max_episode_tries,
        proto_shrink=args.proto_shrink,
        return_details=True,
    )
    out_csv_snr = os.path.join(out_dir_final, "fewshot_snr_curve.csv")  # [SNR]
    df_snr, details_by_snr = run_snr_sweep(      # [SNR]
        H=H,
        y=y,
        snr_cfgs=snr_cfgs,
        eval_func=eval_fewshot_protonet_strict_set,
        eval_kwargs=eval_kwargs,
        seed=args.seed,
        save_csv=out_csv_snr,
        out_dir=out_dir_final,
        save_details=True,
    )
    print("Saved SNR curve csv:", out_csv_snr)  # [SNR]


    # ---------- [SNR] 相关指标输出 ----------
    def _snr_to_float(s: str) -> float:
        """
        把 '20dB'/'0dB'/'-5dB'/'clean' 转成数值，便于算 AUC.
        约定：clean 作为比最大 dB 再高一点（max+5）来放到曲线最右侧。
        """
        s = str(s).strip().lower()
        if s in ["clean", "none", "inf"]:
            return np.nan    
        m = re.search(r"(-?\d+(\.\d+)?)\s*db", s)
        if m:
            return float(m.group(1))
        # 纯数字也行
        try:
            return float(s)
        except Exception:
            return np.nan
    
    def summarize_snr_curve(csv_path: str, out_json: str = None, worst_k_low_snr: int = 5):
        df = pd.read_csv(csv_path)
        assert "snr" in df.columns and "mean_acc" in df.columns, f"CSV缺列: got {df.columns.tolist()}"

        # ---- clean / -5dB acc ----
        # 兼容 snr 列里写 clean / 20dB / -5dB
        def _find_acc(key):
            m = df[df["snr"].astype(str).str.lower() == key.lower()]
            if len(m) == 0:
                return None
            return float(m["mean_acc"].iloc[0])

        clean_acc = _find_acc("clean")
        minus5_acc = _find_acc("0dB") or _find_acc("0db")  # 容错

        # ---- AUC-SNR (mean_acc vs SNR) ----
        snr_vals = np.array([_snr_to_float(x) for x in df["snr"]], dtype=np.float64)
        mean_acc = df["mean_acc"].to_numpy(dtype=np.float64)

        # clean 的 x 设为 max+5（如果有 clean）
        if np.any(np.isnan(snr_vals)):
            finite = snr_vals[np.isfinite(snr_vals)]
            if finite.size > 0:
                snr_vals = np.where(np.isfinite(snr_vals), snr_vals, finite.max() + 5.0)
            else:
                # 全是 clean 的极端情况
                snr_vals = np.where(np.isfinite(snr_vals), snr_vals, 0.0)
        

        # 按 SNR 从小到大排序后做梯形积分
        order = np.argsort(snr_vals)
        xs = snr_vals[order]
        ys = mean_acc[order]
        auc = float(np.trapz(ys, xs))

        # ---- worst-k class acc at low SNR ----
        def worst_k_mean(per_class_acc, k=5):
            v = np.asarray(per_class_acc, dtype=np.float64)
            v = v[np.isfinite(v)]
            if v.size == 0:
                return None
            k = min(int(k), v.size)
            return float(np.sort(v)[:k].mean())

        low_key = "0dB"   
        k_worst = worst_k_low_snr  # 你传进来的参数，表示 k

        p = os.path.join(out_dir_final, f"per_class_acc_{low_key}.npy")
        if os.path.isfile(p):
            per = np.load(p)
            worstk_low_snr = worst_k_mean(per, k=k_worst)
            print(f" worst-{k_worst} class acc @ {low_key} = {worstk_low_snr:.4f}")
        else:
            worstk_low_snr = None
            print(f" worst-k needs per_class_acc file: {p}")

        summary = {
            "csv": os.path.abspath(csv_path),
            "clean_acc": clean_acc,
            "0dB_acc": minus5_acc,
            "auc_snr_mean_acc": auc,
            f"worst_{worst_k_low_snr}_class_acc_at_low_snr": worstk_low_snr,
            "note": "worst-k 需要 per-class acc 或 confusion matrix；当前 CSV 只有 overall mean_acc/std_acc，无法计算。"
        }

        print("\n[E4-A Summary]")
        if clean_acc is not None:
            print(f"  clean acc            = {clean_acc:.4f}")
        else:
            print(f"  clean acc            = (not found in {csv_path})")

        if minus5_acc is not None:
            print(f"  0dB acc             = {minus5_acc:.4f}")
        else:
            print(f"  0dB acc             = (not found in {csv_path})")

        print(f"  AUC-SNR (mean_acc)   = {auc:.6f}   (trapz over SNR axis)")
        print(f"  worst-{worst_k_low_snr} class acc@lowSNR = {worstk_low_snr}   (need per-class stats)")

        if out_json:
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            print(f"  saved: {out_json}")

        return summary
    
    # 打印 [E4-A Summary] 里面那些指标
    out_json = getattr(args, "save_json", None)
    worst_k = getattr(args, "worst_k", 5)          # 兼容 worst_k 参数不存在的情况（默认=5）
    summarize_snr_curve(out_csv_snr, out_json=out_json, worst_k_low_snr=worst_k)

    # ---------- 严格 file+time 的 episodic 评估 + 混淆矩阵 ----------
    eval_ids = list(map(int, eval_classes.tolist()))
    eval_to_i = {cid: i for i, cid in enumerate(eval_ids)}
    cm = np.zeros((len(eval_ids), len(eval_ids)), dtype=np.int64)

    correct_per = {cid: 0 for cid in eval_ids}
    total_per = {cid: 0 for cid in eval_ids}

    # keep query-level records for top confusion + gate delta
    true_buf = []
    pred_buf = []
    gate_buf = []
    gate_raw_rows = []

    # one-time warning flag
    warned_no_gate_w = False    

    modes = {"strict_file_time": 0, "strict_file_only": 0, "relaxed": 0, "relaxed_single_file": 0, "fail": 0}
    accs = []

    ep = 0
    attempts = 0
    max_attempts = max(1, args.episodes * args.max_episode_tries)

    while ep < args.episodes and attempts < max_attempts:
        attempts += 1

        if file_id is None:
            # relaxed sample from class pools
            chosen = rng.choice(eval_classes, size=args.n_way, replace=False)
            sup_idx = []
            qry_idx = []
            ok = True
            for c in chosen:
                c = int(c)
                pool = idx_eval[y[idx_eval] == c]
                if pool.size < (args.n_shot + args.n_query):
                    ok = False; break
                pick = rng.choice(pool, size=(args.n_shot + args.n_query), replace=False)
                sup_idx.append(pick[:args.n_shot])
                qry_idx.append(pick[args.n_shot:])
            if not ok:
                modes["fail"] += 1
                continue
            sup_idx = np.concatenate(sup_idx)
            qry_idx = np.concatenate(qry_idx)
            chosen = chosen.tolist()
            mode = "relaxed"
        else:
            chosen, sup_idx, qry_idx, mode = sample_episode_indices_strict(
                idx_map=idx_map,
                file_time=file_time,
                classes=eval_classes,
                n_way=args.n_way,
                n_shot=args.n_shot,
                n_query=args.n_query,
                rng=rng,
                strict_time=args.strict_time,
                min_time_gap=args.min_time_gap,
                file_uniform=args.file_uniform,
                allow_relaxed=args.allow_relaxed,
                max_tries=400,
            )
            if mode is None:
                modes["fail"] += 1
                continue

        # modes[mode] += 1
        modes[mode] = modes.get(mode, 0) + 1
        ep += 1

        y_sup = y[sup_idx]
        y_qry = y[qry_idx]

        H_sup = H[sup_idx]
        H_qry = H[qry_idx]
        if args.whiten and mu is not None and W is not None:
            H_sup = apply_whiten(H_sup, mu, W)
            H_qry = apply_whiten(H_qry, mu, W)

        pred_global = proto_predict_cosine(H_sup, y_sup, H_qry, chosen, proto_shrink=args.proto_shrink)
        accs.append(float(np.mean(pred_global == y_qry)))

        # update confusion matrix + per-class
        for t, p in zip(y_qry.tolist(), pred_global.tolist()):
            t = int(t); p = int(p)
            if t in eval_to_i and p in eval_to_i:
                cm[eval_to_i[t], eval_to_i[p]] += 1
            if t in total_per:
                total_per[t] += 1
                if p == t:
                    correct_per[t] += 1

        true_buf.append(y_qry.copy())
        pred_buf.append(pred_global.copy())


    if len(accs) == 0:
        raise RuntimeError("No successful episodes. Try --allow_relaxed or reduce constraints (n_way/n_shot/n_query/min_time_gap).")

    accs = np.asarray(accs, dtype=np.float64)
    episode_mean = float(accs.mean())
    episode_std = float(accs.std(ddof=0))
    micro_acc = float(cm.trace() / max(1, cm.sum()))

    print("\n[Episode Sampling Modes]")
    for k in ["strict_file_time", "strict_file_only", "relaxed", "fail"]:
        print(f"  {k:16s}: {modes[k]}")
    print(f"\n{args.n_way}-way {args.n_shot}-shot | episodes={accs.size} | episode_mean={episode_mean:.4f} std={episode_std:.4f} | micro_acc={micro_acc:.4f}")

    # -------- save outputs (all under out_dir) --------
    out_results = os.path.join(out_dir_final, "fewshot_results.csv")
    out_pc = os.path.join(out_dir_final, "per_class_acc.csv")
    out_tc = os.path.join(out_dir_final, "top_confusions.csv")
    out_cm_prefix = os.path.join(out_dir_final, "fewshot_cm")
    out_gate_delta = os.path.join(out_dir_final, "top_confusions_gate_delta.csv")
    out_gate_heatmap = os.path.join(out_dir_final, "ffmoe_gate_heatmap_table.csv")

    df_overall = pd.DataFrame([{
        "npz": args.npz,
        "num_classes_total": num_classes,
        "eval_num_classes": len(eval_ids),
        "n_way": args.n_way,
        "n_shot": args.n_shot,
        "n_query": args.n_query,
        "episodes": int(accs.size),
        "episode_mean_acc": episode_mean,
        "episode_std_acc": episode_std,
        "micro_acc": micro_acc,
        "strict_time": int(bool(args.strict_time)),
        "min_time_gap": float(args.min_time_gap),
        "file_uniform": int(bool(args.file_uniform)),
        "allow_relaxed": int(bool(args.allow_relaxed)),
        "whiten": int(bool(args.whiten)),
        "proto_shrink": float(args.proto_shrink),
        "mode_strict_file_time": int(modes["strict_file_time"]),
        "mode_strict_file_only": int(modes["strict_file_only"]),
        "mode_relaxed": int(modes["relaxed"]),
        "mode_fail": int(modes["fail"]),
        "label_map": used_label_map or "",
        "mat_meta": args.mat_meta or "",
    }])
    df_overall.to_csv(out_results, index=False)
    print("Saved:", out_results)

    rows_pc = []
    for cid in eval_ids:
        tot = int(total_per[cid])
        cor = int(correct_per[cid])
        rows_pc.append({
            "class_id": int(cid),
            "class_name": get_name(int(cid)),    # 用真实设备名
            "acc": float(cor / max(1, tot)),
            "correct": cor,
            "total": tot,
        })
    pd.DataFrame(rows_pc).sort_values("class_id").to_csv(out_pc, index=False)
    print("Saved:", out_pc)

    top_pairs = top_confusions_from_cm(cm, labels_ids=eval_ids, topk=args.topk_conf, min_count=args.topk_min_count)
    # attach names
    for r in top_pairs:
        r["true_name"] = get_name(int(r["true_id"]))
        r["pred_name"] = get_name(int(r["pred_id"]))
    pd.DataFrame(top_pairs).to_csv(out_tc, index=False)
    print("Saved:", out_tc)

   # 加载NPZ时获取class_names
    Z = np.load(args.npz, allow_pickle=True)
    class_names = Z.get("class_names", [])  # 从NPZ读取设备名称列表（按类别ID顺序）

    # confusion plots with REAL NAMES
    eval_ids = list(map(int, eval_classes.tolist()))

    if len(class_names) > 0:
        labels_str = [class_names[cid] if cid < len(class_names) else get_name(cid) for cid in eval_ids]
    else:
        labels_str = [get_name(cid) for cid in eval_ids]

    save_confusion_plots(
            cm,
            labels_str=labels_str,
            out_prefix=out_cm_prefix,
            n_way=args.n_way,
            n_shot=args.n_shot,
            annot_mode=args.cm_annot_mode,
            annot_min_count=args.cm_annot_min_count,
            annot_diag=args.cm_annot_diag,
            annot_fontsize=args.cm_annot_fontsize,
        )

    print("Saved:", out_cm_prefix + "_counts.png/.npy and _norm.png/.npy")

    # === 同时保存一个 fewshot_cm_counts.csv ===
    out_cm_counts_csv = out_cm_prefix + "_counts.csv"
    with open(out_cm_counts_csv, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        # 表头：true\pred + 各列类别名
        w.writerow(["true\\pred"] + list(labels_str))
        K = len(eval_ids)
        for i in range(K):
            cid_true = int(eval_ids[i])
            # 行名用真实设备名（下面的 get_name 会在第 2 步里定义）
            row_name = get_name(cid_true)
            row_counts = [int(cm[i, j]) for j in range(K)]
            w.writerow([row_name] + row_counts)
    print("Saved:", out_cm_counts_csv)

    # =========================================================
    # Build all-SNR gate raw records from cached SNR details
    # =========================================================
    gate_raw_rows = []

    if gate_w is not None:
        print("[GateRaw] build raw rows from cached SNR details ...")

        for snr_name, details in details_by_snr.items():
            all_qry_idx = details.get("all_qry_idx", [])
            all_true = details.get("all_true", [])
            all_pred = details.get("all_pred", [])

            print(f"[GateRaw] snr={snr_name}, episodes={len(all_qry_idx)}")

            for epi_id, (qry_idx_ep, y_qry_ep, pred_qry_ep) in enumerate(
                zip(all_qry_idx, all_true, all_pred), start=1
            ):
                qry_idx_ep = np.asarray(qry_idx_ep, dtype=np.int64)
                y_qry_ep = np.asarray(y_qry_ep)
                pred_qry_ep = np.asarray(pred_qry_ep)

                current_gate = gate_w[qry_idx_ep].copy()

                rows_this_ep = build_gate_raw_rows(
                    gate_qry=current_gate,
                    qry_idx=qry_idx_ep,
                    y_qry=y_qry_ep,
                    pred_qry=pred_qry_ep,
                    sample_size=args.n_shot,
                    episode_id=epi_id,
                    expert_names=expert_names if len(expert_names) > 0 else [f"Expert{i}" for i in range(current_gate.shape[1])],
                    get_name_fn=get_name,
                    snr_name=str(snr_name),
                )
                gate_raw_rows.extend(rows_this_ep)

        print(f"[GateRaw] total rows={len(gate_raw_rows)}")
    else:
        print("[Warn] gate_w not found, skip ffmoe_gate_raw_records.csv construction.")


    gate_all = None
    true_all = None
    pred_all = None

    if gate_w is not None and len(top_pairs) > 0:
        true_all = np.concatenate(true_buf, axis=0)
        pred_all = np.concatenate(pred_buf, axis=0)
        gate_all = np.concatenate(gate_buf, axis=0) if len(gate_buf) > 0 else None

        if gate_all is None:
            print("[ERROR] gate_all为空！gate_buf长度：", len(gate_buf))
        elif np.all(gate_all == 0):
            print("[ERROR] gate_all全为0！请检查gate_w的收集逻辑")
        else:
            print(f"[SUCCESS] 拼接gate_all成功：shape={gate_all.shape}，均值={np.mean(gate_all):.4f}")

        if gate_all is not None:
            min_length = min(len(true_all), len(pred_all), len(gate_all))
            true_all_aligned = true_all[:min_length]
            pred_all_aligned = pred_all[:min_length]
            gate_all_aligned = gate_all[:min_length]
            rows_g = compute_gate_delta_rows(
                true_ids=true_all,
                pred_ids=pred_all,
                gate_qry=gate_all,
                eval_ids=eval_ids,
                name_map=name_map,
                top_pairs=top_pairs,
                expert_names=expert_names if len(expert_names) > 0 else [f"Expert{i}" for i in range(gate_all.shape[1])],
                min_count=args.topk_min_count,
            )
            # 在这里统一用 get_name 覆盖 true_name / pred_name 
            for row in rows_g:
                tid = int(row.get("true_id", -1))
                pid = int(row.get("pred_id", -1))
                row["true_name"] = get_name(tid)
                row["pred_name"] = get_name(pid)
            
            if len(rows_g) > 0:
                pd.DataFrame(rows_g).to_csv(out_gate_delta, index=False)
                print("Saved:", out_gate_delta)
            else:
                print("[GateDelta] No rows met min_count threshold; skip.")
    else:
        print("[GateDelta] gate_w missing or no top_confusions -> skip.")

    draw_representative_cms(out_dir_final, snr_list=("clean", "5dB"))

    draw_hardest_k_cm(
        out_dir_final=out_dir_final,
        low_snr_key="5dB",   # 或 "0dB"
        topk=8,
        also_plot_clean=True
    )

    # ============== Gate可视化调用（移到所有数据准备完成后） ==============
    if gate_all is not None and true_all is not None and pred_all is not None:
        # 动态获取类别数（替代硬编码16）
        num_eval_classes = len(eval_ids) 
        num_experts = gate_all.shape[1]   # 动态获取专家数（替代硬编码4）
        per_class_gate = np.zeros((num_eval_classes, num_experts))

        # 类别ID到索引的映射（适配非连续类别）
        class_id_to_idx = {cid: idx for idx, cid in enumerate(eval_ids)}

        # 打印类别范围，校验数据
        print(f"true_all类别范围：min={np.min(true_all)}, max={np.max(true_all)}")
        class_ids = np.unique(true_all)
        print(f"有效类别ID：{class_ids}")
    
        for cid in class_ids:
            if cid in class_id_to_idx:  # 仅处理eval内的类别
                idx = class_id_to_idx[cid]
                mask = true_all == cid
                sample_count = np.sum(mask)
                if sample_count == 0:
                    print(f"[Warn] 类别{cid}无样本！")
                    continue
                per_class_gate[idx] = np.mean(gate_all[mask], axis=0)
                print(f"类别{cid}（{get_name(cid)}）：样本数={sample_count}，Gate均值={per_class_gate[idx]}")

        # snr_labels = np.zeros_like(true_all)  # 修正SNR标签提取（优先读取样本级SNR细节文件）
        snr_labels = np.empty(len(true_all), dtype=np.float64) 
        snr_details_path = os.path.join(out_dir_final, "fewshot_snr_curve_details.csv") # 读取SNR sweep保存的细节文件（若存在）
        if os.path.exists(snr_details_path):
            df_snr_details = pd.read_csv(snr_details_path)
            if "snr_db" in df_snr_details.columns and len(df_snr_details) == len(true_all):
                snr_db_series = df_snr_details["snr_db"].astype(str).str.strip().str.lower()
                snr_labels = np.array([
                    35.0 if x == "clean" else (float(x) if x.replace("-","").replace(".","").isdigit() else np.nan)
                    for x in snr_db_series
                ], dtype=np.float64)
            

                print(f"[SUCCESS] 读取样本级SNR标签：shape={snr_labels.shape}")
            else:
                print(f"[Warn] SNR细节文件列不匹配/长度不一致：列={df_snr_details.columns}，长度={len(df_snr_details)} vs {len(true_all)}")
                
        else:
            # 若无细节文件，从df_snr提取非clean的SNR值填充
            snr_values = []
            for s in df_snr["snr"].tolist():
                snr_val = _snr_to_float(s)
                if not np.isnan(snr_val):
                    snr_values.append(snr_val)
            if snr_values:
                # snr_labels = np.full(len(true_all), snr_values[0])  # 用第一个有效SNR值
                ## snr_labels[:] = snr_values[0]
                # 去重并排序，确保SNR值唯一且有序
                unique_snr_values = sorted(list(set(snr_values)))
                total_samples = len(true_all)
                samples_per_snr = total_samples // len(unique_snr_values)
                remainder = total_samples % len(unique_snr_values)
            
                # 均分样本到不同SNR
                start_idx = 0
                for i, snr_val in enumerate(unique_snr_values):
                    current_num = samples_per_snr + (1 if i < remainder else 0)
                    end_idx = start_idx + current_num
                    end_idx = min(end_idx, total_samples)  # 防止越界
                    snr_labels[start_idx:end_idx] = snr_val
                    start_idx = end_idx

                print(f"[Warn] 无样本级SNR，用默认值{snr_values[0]}填充")
            else:
                snr_labels[:] = 35.0
                print(f"[Warn] 无有效SNR值，SNR标签填充为clean对应值35.0")
            

        # 验证修改后的SNR标签
        print("=== 修改后SNR标签验证 ===")
        numeric_snr = [val for val in snr_labels if isinstance(val, (int, float))]
        if numeric_snr:
            print(f"snr_labels均值：{np.mean(numeric_snr):.6f}")
        else:
            print("无数值型SNR")
        # 打印每个SNR的样本数，确认分配成功
        for snr_val in np.unique(snr_labels):
            mask = snr_labels == snr_val
            print(f"SNR={snr_val} 样本数：{np.sum(mask)}")

        # 3. 新增：最终校验传递给可视化的核心数据
        print(f"per_class_gate整体均值：{np.mean(per_class_gate):.6f}")
        print(f"snr_labels整体均值：{np.mean(snr_labels):.6f}")
        print(f"all_gate_w整体均值：{np.mean(gate_all):.6f}")
    
        # 4. 一键运行所有可视化（适配你的变量和路径）
        gate_vis.run_all_gate_visualization(
            save_dir=out_dir_final,          # 正确的保存目录（拼接way_shot后的最终目录）
            per_class_gate=per_class_gate,
            all_gate_w=gate_all,             # 样本级Gate权重（已拼接完成）
            all_true=true_all,               # 样本级真实类别ID
            all_pred=pred_all,               # 样本级预测类别ID
            df_log=None,                     # 下游脚本无训练日志
            snr_labels=snr_labels            # SNR标签（适配SNR对比图）
        )

        # 5. 绘制混淆样本Gate差值图（仅当rows_g非空时）
        if 'rows_g' in locals() and len(rows_g) > 0:
            gate_vis.plot_confusion_gate_delta(
            delta_rows=rows_g,  # 已计算的gate delta行
            out_png=os.path.join(out_dir_final, "confusion_gate_delta.png"))
            print(f"✅ 混淆样本Gate差值图已保存：{os.path.join(out_dir_final, 'confusion_gate_delta.png')}")
        else:
            print(f"[Warn] rows_g为空，跳过混淆样本Gate差值图绘制")
    
        print(f"\n✅ 所有Gate可视化图片已保存到：{out_dir_final}")
    else:
        print("\n[ERROR] Gate可视化跳过：gate_all/pred_all/true_all 为空！")
        print(f"  - gate_all是否为空：{gate_all is None}")
        print(f"  - true_all是否为空：{true_all is None}")
        print(f"  - pred_all是否为空：{pred_all is None}")
    
    # ===== ADD START: gate raw + aggregated usage tables =====
    if gate_raw_rows is not None and len(gate_raw_rows) > 0:
        df_gate_raw = pd.DataFrame(gate_raw_rows)

        # 保证列顺序更清晰
        preferred_cols = [
                "SampleSize", "Episode", "QueryIndexInEpisode", "GlobalIndex", "snr",
                "true_id", "true_name", "pred_id", "pred_name", "is_correct",
                "Time", "Freq", "TF", "Inst", "Top1Expert", "Top1Weight"]
        for c in preferred_cols:
            if c not in df_gate_raw.columns:
                df_gate_raw[c] = np.nan
        df_gate_raw = df_gate_raw[preferred_cols]

        print("[GateRaw] df_gate_raw shape =", df_gate_raw.shape)
        if "snr" in df_gate_raw.columns:
            print("[GateRaw] snr counts:\n", df_gate_raw["snr"].value_counts(dropna=False))
        else:
            print("[GateRaw] Warning: 'snr' column not found in df_gate_raw")

        save_gate_usage_tables(
                df_raw=df_gate_raw,
                out_dir=out_dir_final,
                sample_size=args.n_shot,
                threshold=0.35,
                topk=2,
            )
        
        # ===== 随机/分层100样本的门控可视化 =====
        gate_vis.plot_random_subset_gate_bar(
            df_raw=df_gate_raw,
            out_png=os.path.join(out_dir_final, "gate_random100_mean_bar.png"),
            n_samples=100,
            seed=args.seed,
            stratify_by="snr",   # 也可以改成 "is_correct" 或 None
        )

        gate_vis.plot_random_subset_top1_freq(
            df_raw=df_gate_raw,
            out_png=os.path.join(out_dir_final, "gate_random100_top1_freq.png"),
            n_samples=100,
            seed=args.seed,
            stratify_by="snr",   # 也可以改成 "is_correct" 或 None
        )
    else:
        print("[GateRaw] gate_raw_rows为空，跳过样本级 gate 概率表与聚合表保存。")


if __name__ == "__main__":
    main()
