"""
make_split_indices_osu_stable_wifi_daywise.py

Create a fixed Day-1 / Day-2 / Day-3 split file for the OSU Stable WiFi 2024
Scenario 2 MAT exported by preprocess_osu_stable_wifi_ffmoe.py.

Output NPZ keys:
  - train_idx / val_idx / test_idx
  - train_indices / val_indices / test_indices
  - idx_train / idx_val / idx_test

运行命令：
python make_split_indices_osu_stable_wifi_daywise.py \
  --mat ./FeatureMatrix_OSU_Stable_WiFi_Wireless_unified.mat \
  --out split_indices_fssei_osu_stable_wireless.npz \
  --train_day Day-1 \
  --val_day Day-2 \
  --test_day Day-3
"""

import os
import argparse
import numpy as np
import scipy.io as sio


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

    out = {"__backend__": "h5py"}
    with h5py.File(path, "r") as f:
        for k in f.keys():
            out[k] = _h5(f[k])
    return out


def load_mat_auto(path):
    try:
        return {"__backend__": "scipy", **sio.loadmat(path, squeeze_me=True, struct_as_record=False)}
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


def to_str_list(x):
    arr = np.asarray(x).reshape(-1)
    out = []
    for v in arr.tolist():
        if isinstance(v, str):
            out.append(v)
        elif isinstance(v, (bytes, np.bytes_)):
            out.append(v.decode("utf-8", errors="ignore"))
        else:
            out.append(str(v))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mat", required=True)
    ap.add_argument("--out", default="split_indices_fssei_osu_stable_wireless.npz")
    ap.add_argument("--train_day", default="Day-1")
    ap.add_argument("--val_day", default="Day-2")
    ap.add_argument("--test_day", default="Day-3")
    args = ap.parse_args()

    S = load_mat_auto(args.mat)
    ky = pick_first_existing(S, ["label_id", "device_id", "y", "labels"])
    if ky is None:
        raise KeyError("MAT missing label_id")
    y = np.asarray(S[ky]).reshape(-1).astype(np.int64)
    kfid = pick_first_existing(S, ["file_id", "fileId", "fileID", "file_idx", "file_index_id"])
    if kfid is None:
        raise KeyError("MAT missing file_id")
    file_id = np.asarray(S[kfid]).reshape(-1).astype(np.int64)

    kseg = pick_first_existing(S, ["segment_name_per_file", "segment_name"])
    if kseg is None:
        raise KeyError("MAT missing segment_name_per_file")
    seg_names = to_str_list(S[kseg])

    # file_id in exported MAT is 1-based
    uniq_files = np.unique(file_id)
    if uniq_files.min() < 1 or uniq_files.max() > len(seg_names):
        raise ValueError("file_id range does not match segment_name_per_file")

    train_files = {i + 1 for i, s in enumerate(seg_names) if s == args.train_day}
    val_files = {i + 1 for i, s in enumerate(seg_names) if s == args.val_day}
    test_files = {i + 1 for i, s in enumerate(seg_names) if s == args.test_day}

    if not train_files or not val_files or not test_files:
        raise RuntimeError(
            f"Empty split groups. train={len(train_files)} val={len(val_files)} test={len(test_files)}. "
            f"Available segments={sorted(set(seg_names))}"
        )

    train_idx = np.where(np.isin(file_id, list(train_files)))[0].astype(np.int64)
    val_idx = np.where(np.isin(file_id, list(val_files)))[0].astype(np.int64)
    test_idx = np.where(np.isin(file_id, list(test_files)))[0].astype(np.int64)

    np.savez(
        args.out,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        train_indices=train_idx,
        val_indices=val_idx,
        test_indices=test_idx,
        idx_train=train_idx,
        idx_val=val_idx,
        idx_test=test_idx,
        y=y.astype(np.int64),
        file_id=file_id.astype(np.int64),
        split_mode=np.array("daywise_fixed", dtype=object),
        train_day=np.array(args.train_day, dtype=object),
        val_day=np.array(args.val_day, dtype=object),
        test_day=np.array(args.test_day, dtype=object),
    )

    print("=" * 80)
    print("Saved:", args.out)
    print("train_day:", args.train_day, "| train_idx:", len(train_idx), "| files:", len(train_files))
    print("val_day  :", args.val_day,   "| val_idx  :", len(val_idx),   "| files:", len(val_files))
    print("test_day :", args.test_day,  "| test_idx :", len(test_idx),  "| files:", len(test_files))
    print("=" * 80)


if __name__ == "__main__":
    main()
