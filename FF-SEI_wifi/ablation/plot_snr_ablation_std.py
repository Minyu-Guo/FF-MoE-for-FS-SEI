#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Support two modes:
1) table mode: read one xlsx/csv table (backward-compatible)
2) folder mode: read multiple experiment folders, each containing
   fewshot/<shot_dir>/fewshot_snr_curve.csv, then plot mean_acc with std_acc band.

Example (folder mode):
python plot_snr_ablation_std.py \
  --root_dir ./fewshot_test\
  --full_dir exp_full \
  --shot_dir way_5_shot_10 \
  --out_png figs/snr_ablation_way5shot10.png \
  --out_pdf figs/snr_ablation_way5shot10.pdf \
  --show_ci \
  --no_std
"""

import argparse
import os
import re
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# =========================
# IEEE / TIFS style
# =========================
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
mpl.rcParams["mathtext.fontset"] = "stix"
mpl.rcParams["axes.unicode_minus"] = False
mpl.rcParams["axes.linewidth"] = 1.1
mpl.rcParams["xtick.direction"] = "in"
mpl.rcParams["ytick.direction"] = "in"
mpl.rcParams["xtick.major.width"] = 1.0
mpl.rcParams["ytick.major.width"] = 1.0
mpl.rcParams["xtick.minor.width"] = 0.8
mpl.rcParams["ytick.minor.width"] = 0.8

# plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.linewidth"] = 1.2
plt.rcParams["xtick.major.width"] = 1.0
plt.rcParams["ytick.major.width"] = 1.0
plt.rcParams["xtick.major.size"] = 5
plt.rcParams["ytick.major.size"] = 5
plt.rcParams["legend.frameon"] = True
plt.rcParams["legend.framealpha"] = 0.9


METHOD_NAME_MAP = {
    # "ffmoe": "FF-MoE",
    # "ff-moe": "FF-MoE",
    # "full": "FF-MoE",
    # "baseline": "FF-MoE",
    # "exp_full": "FF-MoE",   # 加这一行
    # "exp_no_time": "w/o Time",
    # "exp_no_freq": "w/o Freq",
    # "exp_no_tf": "w/o TF",
    # "exp_no_inst": "w/o Inst",
    # "no_time": "w/o Time",
    # "no_freq": "w/o Freq",
    # "no_tf": "w/o TF",
    # "no_inst": "w/o Inst",
    "exp_full": "FF-MoE",   # 加这一行
    "time": "Time",
    "freq": "Freq",
    "tf": "TF",
    "inst": "Inst",
}

def parse_args():
    ap = argparse.ArgumentParser(description="Plot TIFS-style SNR ablation curve")

    # old table mode (backward-compatible)
    ap.add_argument("--input", type=str, default=None, help="Input .xlsx/.xls/.csv table")
    ap.add_argument("--sheet", type=str, default="0", help="Excel sheet name or index; ignored for csv")
    ap.add_argument("--snr_col", type=str, default="SNR (dB)",
                    help="SNR column name in table mode. If not found, auto-detects a column containing 'snr'.")

    # new folder mode
    ap.add_argument("--root_dir", type=str, default=None,
                    help="Root dir containing experiment folders such as exp_no_freq, exp_no_inst, ...")
    ap.add_argument("--shot_dir", type=str, default="way_5_shot_10",
                    help="Few-shot subfolder name under each experiment folder")
    ap.add_argument("--curve_name", type=str, default="fewshot_snr_curve.csv",
                    help="CSV file name under fewshot/<shot_dir>/")
    ap.add_argument("--metric_col", type=str, default="mean_acc",
                    help="Mean metric column in fewshot_snr_curve.csv")
    ap.add_argument("--std_col", type=str, default="std_acc",
                    help="Std column in fewshot_snr_curve.csv")
    ap.add_argument("--folder_order", type=str,
                    # default="FF-MoE,exp_no_time,exp_no_freq,exp_no_tf,exp_no_inst",
                    default="FF-MoE,time,freq,tf,inst",
                    help="Comma-separated experiment folder order. Unknown/missing items are skipped.")
    ap.add_argument("--folder_labels", type=str, default="",
                    help=("Optional custom mapping like 'exp_full=FF-MoE,exp_no_time=w/o Time,"
                          "exp_no_freq=w/o Freq,exp_no_tf=w/o TF,exp_no_inst=w/o Inst'"))
    ap.add_argument("--full_dir", type=str, default="",
                    help="Optional folder name for the full model inside root_dir, e.g. exp_full or baseline")

    ap.add_argument("--out_png", type=str, required=True)
    ap.add_argument("--out_pdf", type=str, default=None)
    ap.add_argument("--out_svg", type=str, default=None)

    ap.add_argument("--figsize", type=str, default="7.2,5.2",
                    help="Figure size, e.g. 7.2,5.2")
    ap.add_argument("--dpi", type=int, default=500)

    ap.add_argument("--xlabel", type=str, default="SNR (dB)")
    ap.add_argument("--ylabel", type=str, default="Recognition Accuracy (%)")
    ap.add_argument("--legend_loc", type=str, default="lower right")
    ap.add_argument("--title", type=str, default="")

    ap.add_argument("--ylim", type=str, default="0,100",
                    help="y-axis range, e.g. 15,80")
    ap.add_argument("--highlight", type=str, default="FF-MoE",
                    help="Method to highlight by thicker/darker line")
    ap.add_argument("--clean_label", type=str, default="Noise-free",
                    help="Display label for clean condition")

    ap.add_argument("--show_ci", action="store_true",
                    help="Show shaded band. In folder mode, use std_acc; if --n_repeats>1, convert std to 95%% CI.")
    ap.add_argument("--n_repeats", type=int, default=0,
                    help="Number of repeats used to convert std to 95%% CI. If <=1, std is shown directly.")
    ap.add_argument("--band_alpha", type=float, default=0.12)

    ap.add_argument("--legend_ncol", type=int, default=1)
    ap.add_argument("--grid_alpha", type=float, default=0.28)
    ap.add_argument("--no_std", action="store_true", help="Do not plot std shaded region")

    return ap.parse_args()


# ---------- common helpers ----------
def parse_pair(s):
    vals = [float(x.strip()) for x in s.split(",")]
    if len(vals) != 2:
        raise ValueError(f"Invalid pair string: {s}")
    return vals[0], vals[1]


def auto_detect_snr_col(df, preferred):
    if preferred in df.columns:
        return preferred
    for c in df.columns:
        if "snr" in str(c).lower():
            return c
    raise KeyError(f"SNR column '{preferred}' not found, and auto-detect failed. Columns: {list(df.columns)}")


def normalize_method_name(name: str) -> str:
    s = str(name).strip()
    mapping = {
        # "FF-MoE": "FF-MoE",
        # "w/o time": "w/o Time",
        # "w/o f": "w/o Freq",
        # "w/o freq": "w/o Freq",
        # "w/o frequency": "w/o Freq",
        # "w/o tf": "w/o TF",
        # "w/o inst": "w/o Inst",
        "exp_full": "FF-MoE",   # 加这一行
        "time": "Time",
        "freq": "Freq",
        "tf": "TF",
        "inst": "Inst",
    }
    return mapping.get(s, s)


def base_method_name(col_name: str):
    s = str(col_name).strip()
    if re.search(r"(_std|_stderr|_se)$", s, flags=re.IGNORECASE):
        s = re.sub(r"(_std|_stderr|_se)$", "", s, flags=re.IGNORECASE)
    return s


def is_std_col(col_name: str):
    s = str(col_name).strip().lower()
    return s.endswith("_std") or s.endswith("_stderr") or s.endswith("_se")


def snr_to_float(v):
    s = str(v).strip().lower()
    if s in ["clean", "inf", "infty", "infinite", "none"]:
        return np.nan

    m = re.search(r"(-?\d+(\.\d+)?)\s*db", s)
    if m:
        return float(m.group(1))

    try:
        return float(s)
    except Exception:
        return np.nan


def sort_snr_values(vals):
    numeric_items = []
    clean_items = []

    for v in vals:
        fv = snr_to_float(v)
        if np.isfinite(fv):
            numeric_items.append((fv, v))
        else:
            clean_items.append(v)

    numeric_items = sorted(numeric_items, key=lambda x: x[0])
    out = [v for _, v in numeric_items]
    out.extend(clean_items)   # clean 放最后
    return out


def snr_to_label(v, clean_label="Noise-free"):
    fv = snr_to_float(v)
    if np.isfinite(fv):
        return str(int(fv)) if float(fv).is_integer() else str(fv)
    return clean_label

# def style_map(name: str, highlight: str):
#     n = name.lower()
#     h = highlight.lower()

#     if name == highlight or n == h:
#         return {"color": "#d62728", "marker": "o", "lw": 1.8, "ms": 7.0, "zorder": 5}
#     if "time" in n:
#         return {"color": "#1f77b4", "marker": "^", "lw": 1.8, "ms": 6.0, "zorder": 3}
#     if "freq" in n or "w/o f" in n:
#         return {"color": "#2ca02c", "marker": "s", "lw": 1.8, "ms": 6.0, "zorder": 3}
#     if "w/o tf" in n:
#         return {"color": "#ff7f0e", "marker": "D", "lw": 1.8, "ms": 6.0, "zorder": 3}
#     if "inst" in n:
#         return {"color": "#9467bd", "marker": "v", "lw": 1.8, "ms": 6.0, "zorder": 3}
#     return {"color": "#444444", "marker": "o", "lw": 1.8, "ms": 6.0, "zorder": 2}


def style_map(name, highlight="FF-MoE"):
    n = str(name).strip().lower()
    h = str(highlight).strip().lower()

    if name == highlight or n == h:
        return {
            "color": "#d62728",   # 红色
            "marker": "o",
            "lw": 1.8,
            "ms": 6.5,
            "zorder": 6,
            "alpha": 1.0,
            "band_alpha": 0.14,
        }

    color_map = {
        # "w/o time": "#1f77b4",
        # "w/o freq": "#2ca02c",
        # "w/o tf":   "#ff7f0e",
        # "w/o inst": "#9467bd",
        "time": "#1f77b4",
        "freq": "#2ca02c",
        "tf":   "#ff7f0e",
        "inst": "#9467bd",
    }

    return {
        "color": color_map.get(n, "#666666"),
        "marker": {
            # "w/o time": "^",
            # "w/o freq": "s",
            # "w/o tf":   "D",
            # "w/o inst": "v",
            "time": "^",
            "freq": "s",
            "tf":   "D",
            "inst": "v",
        }.get(n, "o"),
        "lw": 2.0,
        "ms": 5.5,
        "zorder": 4,
        "alpha": 0.95,
        "band_alpha": 0.08,
    }

    
def maybe_to_percent(df, cols: List[str]):
    max_val = np.nanmax(df[cols].to_numpy(dtype=float))
    use_percent = max_val <= 1.5
    if use_percent:
        for c in cols:
            df[c] = pd.to_numeric(df[c], errors="coerce") * 100.0
        print("[Info] Detected values in [0,1], converted to percentage.")
    return df, use_percent


# ---------- table mode ----------
def load_table(path, sheet="0"):
    ext = os.path.splitext(path)[1].lower()
    if ext in [".xlsx", ".xls"]:
        if str(sheet).isdigit():
            return pd.read_excel(path, sheet_name=int(sheet))
        return pd.read_excel(path, sheet_name=sheet)
    if ext in [".csv", ".txt"]:
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file type: {ext}")


def prepare_from_table(args):
    df = load_table(args.input, args.sheet)
    snr_col = auto_detect_snr_col(df, args.snr_col)

    raw_cols = [c for c in df.columns if c != snr_col]
    mean_cols = [c for c in raw_cols if not is_std_col(c)]
    std_cols = [c for c in raw_cols if is_std_col(c)]

    method_map = {}
    for c in mean_cols:
        method = normalize_method_name(base_method_name(c))
        method_map.setdefault(method, {})["mean_col"] = c
    for c in std_cols:
        method = normalize_method_name(base_method_name(c))
        method_map.setdefault(method, {})["std_col"] = c

    methods = list(method_map.keys())
    if not methods:
        raise RuntimeError("No method columns found.")

    snr_sorted = sort_snr_values(df[snr_col].tolist())
    df = df.set_index(snr_col).loc[snr_sorted].reset_index()

    for m in methods:
        df[method_map[m]["mean_col"]] = pd.to_numeric(df[method_map[m]["mean_col"]], errors="coerce")
        if "std_col" in method_map[m]:
            df[method_map[m]["std_col"]] = pd.to_numeric(df[method_map[m]["std_col"]], errors="coerce")

    value_cols = [method_map[m]["mean_col"] for m in methods]
    std_value_cols = [method_map[m]["std_col"] for m in methods if "std_col" in method_map[m]]
    df, use_percent = maybe_to_percent(df, value_cols + std_value_cols)

    x = np.arange(len(df))
    labels = [snr_to_label(v, args.clean_label) for v in df[snr_col].tolist()]

    curves = {}
    for m in methods:
        y = df[method_map[m]["mean_col"]].to_numpy(dtype=float)
        band = None
        if args.show_ci and ("std_col" in method_map[m]):
            std = df[method_map[m]["std_col"]].to_numpy(dtype=float)
            band = 1.96 * std / np.sqrt(args.n_repeats) if args.n_repeats and args.n_repeats > 1 else std
        curves[m] = {"y": y, "band": band}
    return x, labels, curves


# ---------- folder mode ----------
def parse_folder_labels(s: str) -> Dict[str, str]:
    out = {}
    if not s.strip():
        return out
    for item in s.split(","):
        item = item.strip()
        if not item or "=" not in item:
            continue
        k, v = item.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def infer_label(folder_name: str, custom_map: Dict[str, str]) -> str:
    if folder_name in custom_map:
        return custom_map[folder_name]
    return METHOD_NAME_MAP.get(folder_name, folder_name)


def detect_snr_col_from_curve(df: pd.DataFrame) -> str:
    candidates = ["snr", "snr_db", "SNR", "SNR(dB)", "SNR (dB)"]
    for c in candidates:
        if c in df.columns:
            return c
    return auto_detect_snr_col(df, "SNR (dB)")


# def find_curve_csv(root_dir: str, exp_folder: str, shot_dir: str, curve_name: str) -> str:
#     p = os.path.join(root_dir, exp_folder, shot_dir, curve_name)
#     return p
def find_curve_csv(root_dir: str, exp_folder: str, shot_dir: str, curve_name: str) -> str:
    # 优先尝试 root/exp_folder/shot_dir/curve_name
    p1 = os.path.join(root_dir, exp_folder, shot_dir, curve_name)
    if os.path.isfile(p1):
        return p1

    # 再尝试 root/exp_folder/curve_name
    p2 = os.path.join(root_dir, exp_folder, curve_name)
    if os.path.isfile(p2):
        return p2

    # 兼容 fewshot 子目录
    p3 = os.path.join(root_dir, exp_folder, "fewshot", shot_dir, curve_name)
    if os.path.isfile(p3):
        return p3

    return p1


def prepare_from_folders(args):
    if not os.path.isdir(args.root_dir):
        raise FileNotFoundError(f"root_dir not found: {args.root_dir}")

    custom_labels = parse_folder_labels(args.folder_labels)

    order = [x.strip() for x in args.folder_order.split(",") if x.strip()]
    exp_folders = []
    for name in order:
        if name == "FF-MoE" and args.full_dir:
            exp_folders.append(args.full_dir)
        else:
            exp_folders.append(name)

    curves = {}
    snr_master = None

    for exp_folder in exp_folders:
        csv_path = find_curve_csv(args.root_dir, exp_folder, args.shot_dir, args.curve_name)
        if not os.path.isfile(csv_path):
            print(f"[Warn] Missing file, skipped: {csv_path}")
            continue

        df = pd.read_csv(csv_path)
        snr_col = detect_snr_col_from_curve(df)
        if args.metric_col not in df.columns:
            raise KeyError(f"'{args.metric_col}' not found in {csv_path}. Columns: {list(df.columns)}")
        if args.show_ci and args.std_col not in df.columns:
            print(f"[Warn] '{args.std_col}' not found in {csv_path}; no band for this curve.")

        df[args.metric_col] = pd.to_numeric(df[args.metric_col], errors="coerce")
        if args.std_col in df.columns:
            df[args.std_col] = pd.to_numeric(df[args.std_col], errors="coerce")

        snr_sorted = sort_snr_values(df[snr_col].tolist())
        df = df.set_index(snr_col).loc[snr_sorted].reset_index()

        value_cols = [args.metric_col] + ([args.std_col] if args.std_col in df.columns else [])
        df, _ = maybe_to_percent(df, value_cols)

        if snr_master is None:
            snr_master = df[snr_col].tolist()
        else:
            # align to first curve's SNR order
            df = df.set_index(snr_col).reindex(snr_master).reset_index()

        label = infer_label(exp_folder, custom_labels)
        y = df[args.metric_col].to_numpy(dtype=float)
        band = None
        if args.show_ci and args.std_col in df.columns:
            std = df[args.std_col].to_numpy(dtype=float)
            band = 1.96 * std / np.sqrt(args.n_repeats) if args.n_repeats and args.n_repeats > 1 else std

        curves[label] = {"y": y, "band": band}
        print(f"[Loaded] {label}: {csv_path}")

    if not curves:
        raise RuntimeError("No valid curves found from root_dir.")

    x = np.arange(len(snr_master))
    labels = [snr_to_label(v, args.clean_label) for v in snr_master]
    return x, labels, curves


def plot_curves(args, x, labels, curves):
    fig_w, fig_h = parse_pair(args.figsize)
    y0, y1 = parse_pair(args.ylim)

    plt.figure(figsize=(fig_w, fig_h))

    methods = list(curves.keys())
    methods_sorted = [m for m in methods if m != args.highlight] + ([args.highlight] if args.highlight in methods else [])

    for m in methods_sorted:
        y = curves[m]["y"]
        band = curves[m].get("band", None)
        style = style_map(m, args.highlight)

        # if args.show_ci and band is not None:
        #     y_low = y - band
        #     y_high = y + band

        #     # 半透明填充
        #     plt.fill_between(
        #         x, y_low, y_high,
        #         color=style["color"],
        #         alpha=args.band_alpha,
        #         linewidth=0,
        #         zorder=style["zorder"] - 1
        #     )

        #     # 上下边界虚线
        #     plt.plot(
        #         x, y_low,
        #         color=style["color"],
        #         linestyle="--",
        #         linewidth=1.0,
        #         alpha=0.9,
        #         zorder=style["zorder"] - 0.5
        #     )
        #     plt.plot(
        #         x, y_high,
        #         color=style["color"],
        #         linestyle="--",
        #         linewidth=1.0,
        #         alpha=0.9,
        #         zorder=style["zorder"] - 0.5
        #     )

        if args.show_ci and band is not None:
            y_low = y - band
            y_high = y + band
            
            if not args.no_std:
                plt.fill_between(
                    x, y_low, y_high,
                    color=style["color"],
                    alpha=style.get("band_alpha", 0.08),
                    linewidth=0,
                    zorder=style["zorder"] - 1
                )

                plt.plot(
                    x, y_low,
                    color=style["color"],
                    linestyle=(0, (4, 3)),
                    linewidth=1.0 if m == "FF-MoE" else 0.9,
                    alpha=0.55,
                    zorder=style["zorder"] - 0.5
                )
                plt.plot(
                    x, y_high,
                    color=style["color"],
                    linestyle=(0, (4, 3)),
                    linewidth=1.0 if m == "FF-MoE" else 0.9,
                    alpha=0.55,
                    zorder=style["zorder"] - 0.5
                )
            plt.plot(
                x, y,
                color=style["color"], marker=style["marker"],
                linewidth=style["lw"], markersize=style["ms"],
                markerfacecolor=style["color"], markeredgecolor=style["color"],
                label=m, zorder=style["zorder"]
            )

    plt.xticks(x, labels, fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel(args.xlabel, fontsize=13)
    plt.ylabel(args.ylabel, fontsize=13)
    if args.title.strip():
        plt.title(args.title, fontsize=10)

    plt.ylim(y0, y1)
    plt.grid(True, which="major", axis="both", linestyle="--", alpha=0.25)

    for tick in plt.gca().get_xticklabels():
        if tick.get_text().lower() == args.clean_label.lower():
            tick.set_fontweight("bold")

    leg = plt.legend(
        loc=args.legend_loc, fontsize=10, frameon=True,
        ncol=args.legend_ncol, borderpad=0.5,
        labelspacing=0.4, handlelength=2.0
    )
    for txt in leg.get_texts():
        if txt.get_text() == args.highlight:
            txt.set_fontweight("bold")

    plt.tight_layout()

    out_png = os.path.abspath(args.out_png)
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=args.dpi, bbox_inches="tight")
    print(f"[Saved] {out_png}")

    if args.out_pdf:
        out_pdf = os.path.abspath(args.out_pdf)
        os.makedirs(os.path.dirname(out_pdf), exist_ok=True)
        plt.savefig(out_pdf, bbox_inches="tight")
        print(f"[Saved] {out_pdf}")

    if args.out_svg:
        out_svg = os.path.abspath(args.out_svg)
        os.makedirs(os.path.dirname(out_svg), exist_ok=True)
        plt.savefig(out_svg, bbox_inches="tight")
        print(f"[Saved] {out_svg}")

    plt.close()


def main():
    args = parse_args()

    if args.root_dir:
        x, labels, curves = prepare_from_folders(args)
    elif args.input:
        x, labels, curves = prepare_from_table(args)
    else:
        raise ValueError("Please provide either --input (table mode) or --root_dir (folder mode).")

    plot_curves(args, x, labels, curves)


if __name__ == "__main__":
    main()
