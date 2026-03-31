#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
preprocess_osu_stable_wifi_ffmoe.py

Fixed preprocessing script for OSU Stable WiFi RF Datasets (2024).

Goal
----
Read the original HDF5 packet files from the OSU Stable WiFi dataset and
convert them directly into an FF-MoE-compatible MAT (HDF5/v7.3-style) file.

Dataset assumptions (from the official release note)
----------------------------------------------------
1) Each HDF5 file contains a dataset named 'data'.
2) The shape of 'data' is (N, 50340), where N is the number of packets in that file.
3) For each packet row:
   - first 25170 values: I samples
   - next 25170 values: Q samples
4) Sample rate defaults to 45e6.

Design choice used here
-----------------------
Each packet is treated as one candidate sample. Since each packet is much
longer than the frame length used by the current FF-MoE pipeline, one fixed
sub-window is selected from each packet (max-energy / center / random), and
that sub-window is used to build:
  - X_time (17)
  - X_freq (5)
  - X_tf   (5)
  - X_inst (12)
  - featureMatrix (39)
  - specTensor
  - occTensor

This keeps the downstream train/export/few-shot scripts compatible and avoids
exploding the dataset size.
python preprocess_osu_stable_wifi_ffmoe.py \
  --root /hy-tmp/osu_stable_wifi/Wireless-Dataset \
  --scenario wireless \
  --out_mat /root/GMY/MoE_FS_SEI/Only_FFSEI_stable_wifi/FeatureMatrix_OSU_Stable_WiFi_Wireless_unified.mat \
  --label_csv /root/GMY/MoE_FS_SEI/Only_FFSEI_stable_wifi/result/preprocess/label_map_osu_stable_wifi_wireless.csv \
  --file_index_csv /root/GMY/MoE_FS_SEI/Only_FFSEI_stable_wifi/result/preprocess/file_index_osu_stable_wifi_wireless.csv \
  --sample_rate 45e6 \
  --packet_len 25170 \
  --frame_len 1024 \
  --window_mode max_energy \
  --save_iq 1 \
  --H 64 \
  --W 64 \
  --max_keep_per_file 1000 \
  --packet_energy_prctile 0 \
  --flush_size 512 \
  --compression_level 4 \
  --seed 42
"""

import os
import re
import csv
import math
import gc
import argparse
from typing import List, Tuple, Dict, Optional

import numpy as np
import h5py
from tqdm import tqdm
from scipy.signal import stft, get_window
from scipy.ndimage import zoom


# ==============================
# Utility
# ==============================

def ensure_dir_for_file(path: str):
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.isdir(d):
        os.makedirs(d, exist_ok=True)


def rms_safe(x: np.ndarray) -> float:
    x = np.asarray(x)
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.abs(x) ** 2) + 1e-12))


def kurtosis_np(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.size < 2:
        return 0.0
    mu = x.mean()
    sd = x.std() + 1e-12
    z = (x - mu) / sd
    return float(np.mean(z ** 4))


def skewness_np(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.size < 2:
        return 0.0
    mu = x.mean()
    sd = x.std() + 1e-12
    z = (x - mu) / sd
    return float(np.mean(z ** 3))


def corrcoef_safe(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1)
    if x.size < 2 or y.size < 2:
        return 0.0
    xs = x.std()
    ys = y.std()
    if xs < 1e-12 or ys < 1e-12:
        return 0.0
    c = np.corrcoef(x, y)
    if c.shape[0] < 2 or not np.isfinite(c[0, 1]):
        return 0.0
    return float(c[0, 1])


def percentile_np(x: np.ndarray, q: float) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.size == 0:
        return 0.0
    return float(np.percentile(x, q))


def utf8_vlen_dtype():
    return h5py.string_dtype(encoding="utf-8")


# ==============================
# Feature extraction
# ==============================

def make_spec(s: np.ndarray, fs: float, nperseg: int = 256, noverlap: int = 128, nfft: int = 512) -> np.ndarray:
    s = np.asarray(s).reshape(-1)
    if s.size < 8:
        return np.zeros((64, 64), dtype=np.float32)
    _, _, Z = stft(
        s,
        fs=fs if np.isfinite(fs) and fs > 0 else 1.0,
        window=get_window("hamming", nperseg),
        nperseg=nperseg,
        noverlap=min(noverlap, nperseg - 1),
        nfft=nfft,
        boundary=None,
        padded=False,
        return_onesided=False,
    )
    spec = np.abs(Z) ** 2
    return spec.astype(np.float32)


def resize_to_hw(img: np.ndarray, H: int, W: int) -> np.ndarray:
    img = np.asarray(img, dtype=np.float32)
    if img.ndim != 2:
        raise ValueError(f"resize_to_hw expects 2D, got {img.shape}")
    if img.shape == (H, W):
        return img
    z0 = H / max(1, img.shape[0])
    z1 = W / max(1, img.shape[1])
    out = zoom(img, (z0, z1), order=1)
    if out.shape != (H, W):
        out = out[:H, :W]
        if out.shape[0] < H or out.shape[1] < W:
            pad_h = H - out.shape[0]
            pad_w = W - out.shape[1]
            out = np.pad(out, ((0, max(0, pad_h)), (0, max(0, pad_w))), mode="edge")
            out = out[:H, :W]
    return out.astype(np.float32)


def make_iq_density_map(I: np.ndarray, Q: np.ndarray, H: int, W: int, lim: float = 3.0) -> np.ndarray:
    x = np.clip(np.asarray(I).reshape(-1), -lim, lim)
    y = np.clip(np.asarray(Q).reshape(-1), -lim, lim)
    ex = np.linspace(-lim, lim, W + 1)
    ey = np.linspace(-lim, lim, H + 1)
    H2, _, _ = np.histogram2d(y, x, bins=[ey, ex])
    H2 = np.log1p(H2)
    H2 = H2 / (np.max(H2) + 1e-12)
    return H2.astype(np.float32)


def extract_time_features(I: np.ndarray, Q: np.ndarray) -> np.ndarray:
    sig = I + 1j * Q
    amp = np.abs(sig)
    phase = np.angle(sig)
    amp_phase_prod = amp * phase
    feat = np.array([
        np.mean(I), np.std(I), kurtosis_np(I), skewness_np(I),
        np.mean(Q), np.std(Q), kurtosis_np(Q), skewness_np(Q),
        np.mean(amp), np.std(amp), np.max(amp), np.min(amp),
        np.mean(phase), np.std(phase),
        corrcoef_safe(amp, phase), np.mean(amp_phase_prod), np.std(amp_phase_prod)
    ], dtype=np.float32)
    return feat


def extract_freq_features(s: np.ndarray, fs: float) -> np.ndarray:
    s = np.asarray(s).reshape(-1)
    N = s.size
    if N < 4:
        return np.zeros((5,), dtype=np.float32)
    Y = np.fft.fftshift(np.fft.fft(s))
    P = (np.abs(Y) ** 2) / max(1, N)
    f = np.linspace(-fs / 2.0, fs / 2.0, N) if np.isfinite(fs) and fs > 0 else np.linspace(-0.5, 0.5, N)
    idx = int(np.argmax(P))
    f_center = f[idx]
    bw = float(np.sum(P > np.max(P) * 0.5) * ((fs / N) if np.isfinite(fs) and fs > 0 else (1.0 / N)))
    f_center_norm = f_center / ((fs / 2.0) + 1e-12) if np.isfinite(fs) and fs > 0 else float(f_center)
    bw_norm = bw / (fs + 1e-12) if np.isfinite(fs) and fs > 0 else float(bw)
    Pn = P / (np.sum(P) + 1e-12)
    spec_entropy = -np.sum(Pn * np.log(Pn + 1e-12))
    feat = np.array([f_center_norm, bw_norm, np.mean(P), np.var(P), spec_entropy], dtype=np.float32)
    return feat


def extract_tf_features(s: np.ndarray, fs: float) -> np.ndarray:
    spec = make_spec(s, fs)
    x = spec.reshape(-1)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.zeros((5,), dtype=np.float32)
    mean_energy = float(np.mean(x))
    var_energy = float(np.var(x))
    p = x / (np.sum(x) + 1e-12)
    tf_entropy = float(-np.sum(p * np.log(p + 1e-12)))
    spec_flatness = float(np.exp(np.mean(np.log(x + 1e-12))) / (np.mean(x) + 1e-12))
    xs = np.sort(x)[::-1]
    k = max(1, int(round(0.10 * xs.size)))
    energy_conc = float(np.sum(xs[:k]) / (np.sum(xs) + 1e-12))
    return np.array([mean_energy, var_energy, tf_entropy, spec_flatness, energy_conc], dtype=np.float32)


def extract_inst_features(I: np.ndarray, Q: np.ndarray, fs: float) -> np.ndarray:
    s = (np.asarray(I).reshape(-1) + 1j * np.asarray(Q).reshape(-1)).astype(np.complex64)
    if s.size < 4:
        return np.zeros((12,), dtype=np.float32)
    s = s - np.mean(s)
    s = s / (rms_safe(s) + 1e-12)
    dphi = np.angle(s[1:] * np.conj(s[:-1]))
    dphi = dphi[np.isfinite(dphi)]
    if dphi.size == 0:
        return np.zeros((12,), dtype=np.float32)
    w0 = float(np.median(dphi))
    cfo_hz = float(w0 * fs / (2.0 * np.pi)) if np.isfinite(fs) and fs > 0 else w0
    r = dphi - w0
    mu_r = float(np.mean(r))
    std_r = float(np.std(r) + 1e-12)
    mad_r = float(np.mean(np.abs(r - mu_r)))
    q25 = percentile_np(r, 25)
    q75 = percentile_np(r, 75)
    iqr_r = q75 - q25
    maxabs_r = float(np.max(np.abs(r))) if r.size > 0 else 0.0
    if r.size >= 3 and np.std(r[:-1]) > 1e-12 and np.std(r[1:]) > 1e-12:
        ac1 = corrcoef_safe(r[:-1], r[1:])
    else:
        ac1 = 0.0
    rr = r.astype(np.float64) * np.hanning(r.size)
    P = np.abs(np.fft.fft(rr)) ** 2
    P = P[:max(1, P.size // 2)]
    Pn = P / (np.sum(P) + 1e-12)
    L = Pn.size
    low_frac = float(np.sum(Pn[:max(1, round(0.05 * L))]))
    high_start = min(L - 1, max(0, round(0.30 * L)))
    high_frac = float(np.sum(Pn[high_start:]))
    flatness = float(np.exp(np.mean(np.log(P + 1e-12))) / (np.mean(P) + 1e-12))
    dr = np.diff(r)
    mean_abs_dr = float(np.mean(np.abs(dr))) if dr.size > 0 else 0.0
    ku = kurtosis_np(r)
    return np.array([
        cfo_hz, std_r, mad_r, iqr_r, maxabs_r,
        ac1, low_frac, high_frac, flatness,
        mean_abs_dr, mu_r, ku
    ], dtype=np.float32)


def build_frame_outputs(s_frame: np.ndarray, fs: float, H: int, W: int):
    s_frame = np.asarray(s_frame).astype(np.complex64).reshape(-1)
    s_frame = s_frame - np.mean(s_frame)
    s_frame = s_frame / (rms_safe(s_frame) + 1e-12)
    I = np.real(s_frame).astype(np.float32)
    Q = np.imag(s_frame).astype(np.float32)

    x_time = extract_time_features(I, Q)
    x_freq = extract_freq_features(s_frame, fs)
    x_tf = extract_tf_features(s_frame, fs)
    x_inst = extract_inst_features(I, Q, fs)
    feat39 = np.concatenate([x_time, x_freq, x_tf, x_inst], axis=0).astype(np.float32)

    spec = make_spec(s_frame, fs)
    spec = 10.0 * np.log10(spec + 1e-12)
    spec = spec - np.min(spec)
    spec = spec / (np.max(spec) + 1e-12)
    spec = resize_to_hw(spec, H, W).astype(np.float32)

    occ = make_iq_density_map(I, Q, H, W).astype(np.float32)
    return x_time, x_freq, x_tf, x_inst, feat39, spec, occ, s_frame


# ==============================
# Packet window selection
# ==============================

def pick_packet_window(s: np.ndarray, frame_len: int, mode: str = "max_energy", seed: int = 42) -> np.ndarray:
    s = np.asarray(s).reshape(-1)
    n = s.size
    if n < frame_len:
        pad = frame_len - n
        s = np.pad(s, (0, pad), mode="constant")
        return s[:frame_len]

    if mode == "center":
        start = max(0, (n - frame_len) // 2)
        return s[start:start + frame_len]

    if mode == "random":
        rng = np.random.default_rng(seed)
        start = int(rng.integers(0, n - frame_len + 1))
        return s[start:start + frame_len]

    # default: max_energy
    p = np.abs(s) ** 2
    cs = np.concatenate([[0.0], np.cumsum(p, dtype=np.float64)])
    e = cs[frame_len:] - cs[:-frame_len]
    start = int(np.argmax(e))
    return s[start:start + frame_len]


def packet_row_to_complex(row: np.ndarray, packet_len: int = 25170) -> np.ndarray:
    row = np.asarray(row, dtype=np.float32).reshape(-1)
    if row.size != packet_len * 2:
        raise ValueError(f"Expect row size {packet_len*2}, got {row.size}")
    I = row[:packet_len]
    Q = row[packet_len:packet_len * 2]
    return (I + 1j * Q).astype(np.complex64)


# ==============================
# Stable WiFi path parsing
# ==============================

def list_h5_files(root: str) -> List[str]:
    out = []
    for r, _, files in os.walk(root):
        for fn in files:
            if fn.lower().endswith(".h5") or fn.lower().endswith(".hdf5"):
                out.append(os.path.join(r, fn))
    out.sort()
    return out


def parse_device_from_filename(path: str) -> str:
    base = os.path.basename(path)
    m = re.search(r'(dev(?:ice)?[_-]?(\d+))', base.lower())
    if m:
        num = int(m.group(2))
        return f"device_{num}"
    m = re.search(r'(\d+)', base)
    if m:
        return f"device_{int(m.group(1))}"
    return os.path.splitext(base)[0]


def parse_segment_from_path(path: str) -> Tuple[str, str, int]:
    """
    Return (segment_type, segment_name, segment_order)
    segment_type in {day, location, phase, unknown}
    """
    norm_parts = [p for p in os.path.normpath(path).split(os.sep) if p]
    lower_parts = [p.lower() for p in norm_parts]

    for p in lower_parts:
        m = re.match(r'day[-_ ]?(\d+)', p)
        if m:
            d = int(m.group(1))
            return "day", f"Day-{d}", d

    for p in lower_parts:
        m = re.match(r'location[-_ ]?([abc])', p)
        if m:
            tag = m.group(1).upper()
            return "location", f"Location-{tag}", {"A": 1, "B": 2, "C": 3}[tag]
        if p in {"a", "b", "c"}:
            tag = p.upper()
            return "location", f"Location-{tag}", {"A": 1, "B": 2, "C": 3}[tag]

    for p in lower_parts:
        if "enroll" in p:
            return "phase", "enrollment", 1
        if "deploy" in p:
            return "phase", "deployment", 2

    return "unknown", "unknown", 0


def infer_scenario_name(path: str) -> str:
    low = os.path.normpath(path).lower()
    if "wireless-dataset" in low:
        return "wireless"
    if "wired-dataset" in low:
        return "wired"
    if "location-dataset" in low:
        return "location"
    if "random" in low:
        return "random"
    return "unknown"


# ==============================
# MAT writer (compatible with current Python pipeline)
# ==============================
class MatH5Writer:
    def __init__(self, out_path: str, H: int, W: int, frame_len: int, save_iq: bool = False, compression: str = "gzip", compression_level: int = 4):
        ensure_dir_for_file(out_path)
        self.out_path = out_path
        self.H = H
        self.W = W
        self.frame_len = frame_len
        self.save_iq = save_iq
        self.f = h5py.File(out_path, "w")
        self.n = 0
        self._str_dt = utf8_vlen_dtype()
        comp = compression
        clevel = compression_level

        self.ds_featureMatrix = self.f.create_dataset("featureMatrix", shape=(39, 0), maxshape=(39, None), dtype="float32", compression=comp, compression_opts=clevel, chunks=(39, 1024))
        self.ds_X_time = self.f.create_dataset("X_time", shape=(17, 0), maxshape=(17, None), dtype="float32", compression=comp, compression_opts=clevel, chunks=(17, 1024))
        self.ds_X_freq = self.f.create_dataset("X_freq", shape=(5, 0), maxshape=(5, None), dtype="float32", compression=comp, compression_opts=clevel, chunks=(5, 1024))
        self.ds_X_tf = self.f.create_dataset("X_tf", shape=(5, 0), maxshape=(5, None), dtype="float32", compression=comp, compression_opts=clevel, chunks=(5, 1024))
        self.ds_X_inst = self.f.create_dataset("X_inst", shape=(12, 0), maxshape=(12, None), dtype="float32", compression=comp, compression_opts=clevel, chunks=(12, 1024))
        self.ds_specTensor = self.f.create_dataset("specTensor", shape=(0, W, H), maxshape=(None, W, H), dtype="float32", compression=comp, compression_opts=clevel, chunks=(64, W, H))
        self.ds_occTensor = self.f.create_dataset("occTensor", shape=(0, W, H), maxshape=(None, W, H), dtype="float32", compression=comp, compression_opts=clevel, chunks=(64, W, H))

        if save_iq:
            self.ds_iqTensor = self.f.create_dataset("iqTensor", shape=(0, frame_len), maxshape=(None, frame_len), dtype=np.complex64, compression=comp, compression_opts=clevel, chunks=(64, frame_len))
        else:
            self.ds_iqTensor = None

        self.ds_label_id = self.f.create_dataset("label_id", shape=(0,), maxshape=(None,), dtype="int64", compression=comp, compression_opts=clevel, chunks=(4096,))
        self.ds_fsVector = self.f.create_dataset("fsVector", shape=(0,), maxshape=(None,), dtype="float32", compression=comp, compression_opts=clevel, chunks=(4096,))
        self.ds_file_id = self.f.create_dataset("file_id", shape=(0,), maxshape=(None,), dtype="int64", compression=comp, compression_opts=clevel, chunks=(4096,))
        self.ds_real_time = self.f.create_dataset("real_time", shape=(0,), maxshape=(None,), dtype="float64", compression=comp, compression_opts=clevel, chunks=(4096,))
        self.ds_labelVector = self.f.create_dataset("labelVector", shape=(0,), maxshape=(None,), dtype=self._str_dt, compression=comp, compression_opts=clevel, chunks=(1024,))
        self.ds_file_path = self.f.create_dataset("file_path", shape=(0,), maxshape=(None,), dtype=self._str_dt, compression=comp, compression_opts=clevel, chunks=(1024,))

        self.f.create_dataset("feature_names_time", data=np.array([
            'mean_I','std_I','kurt_I','skew_I','mean_Q','std_Q','kurt_Q','skew_Q',
            'mean_amp','std_amp','max_amp','min_amp','mean_phase','std_phase',
            'corr_amp_phase','mean_amp_phase_prod','std_amp_phase_prod'
        ], dtype=object), dtype=self._str_dt)
        self.f.create_dataset("feature_names_freq", data=np.array([
            'f_center_norm','bw_norm','mean_P','var_P','spec_entropy_P'
        ], dtype=object), dtype=self._str_dt)
        self.f.create_dataset("feature_names_tf", data=np.array([
            'tf_mean_energy','tf_var_energy','tf_entropy','tf_flatness','tf_energy_conc_top10'
        ], dtype=object), dtype=self._str_dt)
        self.f.create_dataset("feature_names_inst", data=np.array([
            'cfo_hz','std_inst','mad_inst','iqr_inst','maxabs_inst',
            'ac1_inst','low_frac_inst','high_frac_inst','flatness_inst',
            'mean_abs_df','mu_inst_res','kurt_inst'
        ], dtype=object), dtype=self._str_dt)
        self.f.create_dataset("feature_names_all", data=np.array([
            'mean_I','std_I','kurt_I','skew_I','mean_Q','std_Q','kurt_Q','skew_Q',
            'mean_amp','std_amp','max_amp','min_amp','mean_phase','std_phase',
            'corr_amp_phase','mean_amp_phase_prod','std_amp_phase_prod',
            'f_center_norm','bw_norm','mean_P','var_P','spec_entropy_P',
            'tf_mean_energy','tf_var_energy','tf_entropy','tf_flatness','tf_energy_conc_top10',
            'cfo_hz','std_inst','mad_inst','iqr_inst','maxabs_inst',
            'ac1_inst','low_frac_inst','high_frac_inst','flatness_inst','mean_abs_df','mu_inst_res','kurt_inst'
        ], dtype=object), dtype=self._str_dt)

    def append_batch(self, X_time, X_freq, X_tf, X_inst, featureMatrix, specTensor, occTensor,
                     label_id, fsVector, file_id, labelVector, file_path, real_time, iqTensor=None):
        k = int(featureMatrix.shape[0])
        if k <= 0:
            return
        s0 = self.n
        s1 = self.n + k

        self.ds_featureMatrix.resize((39, s1))
        self.ds_featureMatrix[:, s0:s1] = featureMatrix.T.astype(np.float32)
        self.ds_X_time.resize((17, s1)); self.ds_X_time[:, s0:s1] = X_time.T.astype(np.float32)
        self.ds_X_freq.resize((5, s1)); self.ds_X_freq[:, s0:s1] = X_freq.T.astype(np.float32)
        self.ds_X_tf.resize((5, s1)); self.ds_X_tf[:, s0:s1] = X_tf.T.astype(np.float32)
        self.ds_X_inst.resize((12, s1)); self.ds_X_inst[:, s0:s1] = X_inst.T.astype(np.float32)

        self.ds_specTensor.resize((s1, self.W, self.H))
        self.ds_specTensor[s0:s1] = np.transpose(specTensor, (0, 2, 1)).astype(np.float32)
        self.ds_occTensor.resize((s1, self.W, self.H))
        self.ds_occTensor[s0:s1] = np.transpose(occTensor, (0, 2, 1)).astype(np.float32)

        if self.ds_iqTensor is not None and iqTensor is not None:
            self.ds_iqTensor.resize((s1, self.frame_len))
            self.ds_iqTensor[s0:s1] = iqTensor.astype(np.complex64)

        self.ds_label_id.resize((s1,)); self.ds_label_id[s0:s1] = np.asarray(label_id, dtype=np.int64).reshape(-1)
        self.ds_fsVector.resize((s1,)); self.ds_fsVector[s0:s1] = np.asarray(fsVector, dtype=np.float32).reshape(-1)
        self.ds_file_id.resize((s1,)); self.ds_file_id[s0:s1] = np.asarray(file_id, dtype=np.int64).reshape(-1)
        self.ds_real_time.resize((s1,)); self.ds_real_time[s0:s1] = np.asarray(real_time, dtype=np.float64).reshape(-1)
        self.ds_labelVector.resize((s1,)); self.ds_labelVector[s0:s1] = np.asarray(labelVector, dtype=object)
        self.ds_file_path.resize((s1,)); self.ds_file_path[s0:s1] = np.asarray(file_path, dtype=object)

        self.n = s1

    def finalize(self, meta: Dict):
        for k, v in meta.items():
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], str):
                self.f.create_dataset(k, data=np.asarray(v, dtype=object), dtype=self._str_dt)
            elif isinstance(v, list):
                self.f.create_dataset(k, data=np.asarray(v))
            else:
                self.f.create_dataset(k, data=np.asarray(v))
        self.f.flush()
        self.f.close()


class BatchBuffer:
    def __init__(self, save_iq: bool):
        self.save_iq = save_iq
        self.clear()

    def clear(self):
        self.x_time = []
        self.x_freq = []
        self.x_tf = []
        self.x_inst = []
        self.feat39 = []
        self.spec = []
        self.occ = []
        self.label_id = []
        self.fs = []
        self.file_id = []
        self.label_str = []
        self.file_path = []
        self.real_time = []
        self.iq = []

    def __len__(self):
        return len(self.feat39)

    def add(self, x_time, x_freq, x_tf, x_inst, feat39, spec, occ, label_id, fs, file_id, label_str, file_path, real_time, iq=None):
        self.x_time.append(x_time)
        self.x_freq.append(x_freq)
        self.x_tf.append(x_tf)
        self.x_inst.append(x_inst)
        self.feat39.append(feat39)
        self.spec.append(spec)
        self.occ.append(occ)
        self.label_id.append(int(label_id))
        self.fs.append(float(fs))
        self.file_id.append(int(file_id))
        self.label_str.append(str(label_str))
        self.file_path.append(str(file_path))
        self.real_time.append(float(real_time))
        if self.save_iq and iq is not None:
            self.iq.append(iq.astype(np.complex64))

    def flush_to(self, writer: MatH5Writer):
        if len(self) == 0:
            return
        iq_np = np.stack(self.iq, axis=0).astype(np.complex64) if self.save_iq and len(self.iq) > 0 else None
        writer.append_batch(
            X_time=np.stack(self.x_time, axis=0).astype(np.float32),
            X_freq=np.stack(self.x_freq, axis=0).astype(np.float32),
            X_tf=np.stack(self.x_tf, axis=0).astype(np.float32),
            X_inst=np.stack(self.x_inst, axis=0).astype(np.float32),
            featureMatrix=np.stack(self.feat39, axis=0).astype(np.float32),
            specTensor=np.stack(self.spec, axis=0).astype(np.float32),
            occTensor=np.stack(self.occ, axis=0).astype(np.float32),
            label_id=np.asarray(self.label_id, dtype=np.int64),
            fsVector=np.asarray(self.fs, dtype=np.float32),
            file_id=np.asarray(self.file_id, dtype=np.int64),
            labelVector=list(self.label_str),
            file_path=list(self.file_path),
            real_time=np.asarray(self.real_time, dtype=np.float64),
            iqTensor=iq_np,
        )
        self.clear()


# ==============================
# Main
# ==============================

def build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True, help="root directory of OSU Stable WiFi HDF5 files")
    ap.add_argument("--scenario", type=str, default="auto", choices=["auto", "wireless", "wired", "location", "random"], help="optional scenario tag for bookkeeping")
    ap.add_argument("--out_mat", type=str, default="FeatureMatrix_OSU_Stable_WiFi.mat")
    ap.add_argument("--label_csv", type=str, default="label_map_osu_stable_wifi.csv")
    ap.add_argument("--file_index_csv", type=str, default="file_index_osu_stable_wifi.csv")

    ap.add_argument("--sample_rate", type=float, default=45e6)
    ap.add_argument("--packet_len", type=int, default=25170)
    ap.add_argument("--frame_len", type=int, default=1024)
    ap.add_argument("--window_mode", type=str, default="max_energy", choices=["max_energy", "center", "random"])
    ap.add_argument("--save_iq", type=int, default=0)

    ap.add_argument("--H", type=int, default=64)
    ap.add_argument("--W", type=int, default=64)
    ap.add_argument("--max_files", type=int, default=0)
    ap.add_argument("--max_keep_per_file", type=int, default=1000, help="maximum packets kept from each .h5 file; <=0 means keep all packets")
    ap.add_argument("--packet_energy_prctile", type=float, default=0.0, help="optional packet-level energy filtering within each file; 0 disables")
    ap.add_argument("--flush_size", type=int, default=512)
    ap.add_argument("--compression_level", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    return ap


def main():
    args = build_argparser().parse_args()
    np.random.seed(args.seed)

    files = list_h5_files(args.root)
    if args.max_files > 0:
        files = files[:args.max_files]
    if len(files) == 0:
        raise RuntimeError(f"No .h5 files found under: {args.root}")

    ensure_dir_for_file(args.out_mat)
    ensure_dir_for_file(args.label_csv)
    ensure_dir_for_file(args.file_index_csv)

    writer = MatH5Writer(
        out_path=args.out_mat,
        H=args.H,
        W=args.W,
        frame_len=args.frame_len,
        save_iq=bool(args.save_iq),
        compression="gzip",
        compression_level=args.compression_level,
    )
    buffer = BatchBuffer(save_iq=bool(args.save_iq))

    label_to_id: Dict[str, int] = {}
    id_to_label: Dict[int, str] = {}

    file_path_per_file = []
    label_str_per_file = []
    real_time_per_file = []
    fs_per_file = []
    scenario_per_file = []
    segment_type_per_file = []
    segment_name_per_file = []
    segment_order_per_file = []
    device_name_per_file = []
    num_packets_total_per_file = []
    num_packets_kept_per_file = []
    num_dropped_energy_per_file = []

    total_kept = 0
    valid_file_counter = 0
    rows_csv = []

    pbar = tqdm(files, ncols=120, desc="OSU Stable WiFi preprocess")
    for path in pbar:
        try:
            scenario_name = infer_scenario_name(path) if args.scenario == "auto" else args.scenario
            seg_type, seg_name, seg_order = parse_segment_from_path(path)
            device_name = parse_device_from_filename(path)
            label_str = device_name
            if label_str not in label_to_id:
                cid = len(label_to_id)
                label_to_id[label_str] = cid
                id_to_label[cid] = label_str
            label_id = label_to_id[label_str]

            with h5py.File(path, "r") as f:
                if "data" not in f:
                    raise KeyError(f"Missing dataset 'data' in {path}")
                ds = f["data"]
                if ds.ndim != 2 or ds.shape[1] != args.packet_len * 2:
                    raise ValueError(f"Expect 'data' shape (N,{args.packet_len*2}), got {ds.shape} in {path}")
                num_packets = int(ds.shape[0])

                if args.max_keep_per_file > 0 and num_packets > args.max_keep_per_file:
                    pick_idx = np.unique(np.round(np.linspace(0, num_packets - 1, args.max_keep_per_file)).astype(np.int64))
                else:
                    pick_idx = np.arange(num_packets, dtype=np.int64)

                energies = []
                if args.packet_energy_prctile > 0:
                    for pi in pick_idx.tolist():
                        row = ds[pi]
                        s = packet_row_to_complex(row, args.packet_len)
                        frame = pick_packet_window(s, args.frame_len, args.window_mode, args.seed + int(pi))
                        energies.append(rms_safe(frame))
                    thr = np.percentile(np.asarray(energies), args.packet_energy_prctile) if len(energies) > 0 else -np.inf
                else:
                    thr = -np.inf

                kept_this_file = 0
                dropped_this_file = 0
                current_file_id = valid_file_counter + 1
                pseudo_time = float(seg_order)

                for kk, pi in enumerate(pick_idx.tolist()):
                    row = ds[pi]
                    s = packet_row_to_complex(row, args.packet_len)
                    frame = pick_packet_window(s, args.frame_len, args.window_mode, args.seed + int(pi))
                    e = rms_safe(frame)
                    if args.packet_energy_prctile > 0 and e < thr:
                        dropped_this_file += 1
                        continue

                    x_time, x_freq, x_tf, x_inst, feat39, spec, occ, frame_norm = build_frame_outputs(frame, args.sample_rate, args.H, args.W)
                    buffer.add(
                        x_time=x_time,
                        x_freq=x_freq,
                        x_tf=x_tf,
                        x_inst=x_inst,
                        feat39=feat39,
                        spec=spec,
                        occ=occ,
                        label_id=label_id,
                        fs=args.sample_rate,
                        file_id=current_file_id,
                        label_str=label_str,
                        file_path=path,
                        real_time=pseudo_time,
                        iq=frame_norm if args.save_iq else None,
                    )
                    kept_this_file += 1
                    total_kept += 1

                    if len(buffer) >= args.flush_size:
                        buffer.flush_to(writer)

                if kept_this_file > 0:
                    valid_file_counter += 1
                    file_path_per_file.append(path)
                    label_str_per_file.append(label_str)
                    real_time_per_file.append(pseudo_time)
                    fs_per_file.append(float(args.sample_rate))
                    scenario_per_file.append(scenario_name)
                    segment_type_per_file.append(seg_type)
                    segment_name_per_file.append(seg_name)
                    segment_order_per_file.append(seg_order)
                    device_name_per_file.append(device_name)
                    num_packets_total_per_file.append(num_packets)
                    num_packets_kept_per_file.append(kept_this_file)
                    num_dropped_energy_per_file.append(dropped_this_file)

                    rows_csv.append({
                        "file_id": valid_file_counter,
                        "path": path,
                        "scenario": scenario_name,
                        "segment_type": seg_type,
                        "segment_name": seg_name,
                        "segment_order": seg_order,
                        "device": device_name,
                        "label_str": label_str,
                        "label_id": label_id,
                        "fs": args.sample_rate,
                        "num_packets_total": num_packets,
                        "num_packets_kept": kept_this_file,
                        "num_dropped_energy": dropped_this_file,
                    })

                pbar.set_postfix({
                    "samples": total_kept,
                    "valid_files": valid_file_counter,
                    "classes": len(label_to_id),
                })

        except Exception as e:
            print(f"[Warn] skip file: {path} | err={e}")
            continue

    if len(buffer) > 0:
        buffer.flush_to(writer)

    writer.finalize({
        "file_path_per_file": file_path_per_file,
        "label_str_per_file": label_str_per_file,
        "real_time_per_file": real_time_per_file,
        "fs_per_file": fs_per_file,
        "scenario_per_file": scenario_per_file,
        "segment_type_per_file": segment_type_per_file,
        "segment_name_per_file": segment_name_per_file,
        "segment_order_per_file": segment_order_per_file,
        "device_name_per_file": device_name_per_file,
        "num_packets_total_per_file": num_packets_total_per_file,
        "num_packets_kept_per_file": num_packets_kept_per_file,
        "num_dropped_energy_per_file": num_dropped_energy_per_file,
        "frameLen": np.array(args.frame_len, dtype=np.int64),
        "sample_rate": np.array(args.sample_rate, dtype=np.float32),
        "packet_len": np.array(args.packet_len, dtype=np.int64),
        "window_mode": np.array(args.window_mode, dtype=object),
        "scenario": np.array(args.scenario, dtype=object),
    })

    with open(args.label_csv, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class_id", "name"])
        for cid in sorted(id_to_label.keys()):
            w.writerow([cid, id_to_label[cid]])

    with open(args.file_index_csv, "w", encoding="utf-8-sig", newline="") as f:
        fieldnames = [
            "file_id", "path", "scenario", "segment_type", "segment_name", "segment_order",
            "device", "label_str", "label_id", "fs", "num_packets_total", "num_packets_kept",
            "num_dropped_energy"
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows_csv:
            w.writerow(r)

    print("=" * 90)
    print(f"Saved MAT : {args.out_mat}")
    print(f"Saved CSV : {args.label_csv}")
    print(f"Saved IDX : {args.file_index_csv}")
    print(f"Samples   : {writer.n}")
    print(f"Classes   : {len(label_to_id)}")
    print(f"ValidFile : {valid_file_counter}")
    print("Done.")


if __name__ == "__main__":
    main()
