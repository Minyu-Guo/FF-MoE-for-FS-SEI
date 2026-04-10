'''
python plot_ablation_clean_boxplot.py \
  --root_dir ./fewshot_test \
  --out_png ./figs/ablation_clean_boxplot.png \
  --out_pdf ./figs/ablation_clean_boxplot.pdf \
  --show_points
'''

import os
import re
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# METHOD_NAME_MAP = {
#     "exp_full": "FF-MoE (Ours)",
#     "exp_no_time": "w/o Time",
#     "exp_no_freq": "w/o Freq",
#     "exp_no_tf": "w/o TF",
#     "exp_no_inst": "w/o Inst",
# }
METHOD_NAME_MAP = {
    "exp_full": "FF-MoE (Ours)",
    "time": "Time",
    "freq": "Freq",
    "tf": "TF",
    "inst": "Inst",
}

# DEFAULT_ORDER = ["exp_full", "exp_no_time", "exp_no_freq", "exp_no_tf", "exp_no_inst"]
DEFAULT_ORDER = ["exp_full", "time", "freq", "tf", "inst"]


def parse_args():
    ap = argparse.ArgumentParser(description="Plot clean-accuracy boxplot for ablation experiments.")
    ap.add_argument("--root_dir", type=str, required=True,
                    help="Root ablation directory, e.g. ./ablation")
    ap.add_argument("--out_png", type=str, default="figs/ablation_clean_boxplot.png")
    ap.add_argument("--out_pdf", type=str, default="figs/ablation_clean_boxplot.pdf")
    ap.add_argument("--curve_name", type=str, default="fewshot_snr_curve.csv")
    ap.add_argument("--results_name", type=str, default="fewshot_results.csv")
    ap.add_argument("--shot_pattern", type=str, default=r"way_\d+_shot_\d+",
                    help="Regex for selecting fewshot subfolders")
    ap.add_argument("--show_points", action="store_true",
                    help="Overlay individual clean-accuracy points")
    # ap.add_argument("--title", type=str, default="Clean Accuracy Distribution of Ablation Models")
    ap.add_argument("--ylabel", type=str, default="Recognition Accuracy (%)")
    return ap.parse_args()


def ensure_dir(path):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def normalize_acc(v):
    """If in [0,1], convert to percentage."""
    v = float(v)
    return v * 100.0 if 0.0 <= v <= 1.0 else v


def get_clean_acc_from_curve(csv_path):
    df = pd.read_csv(csv_path)

    # possible snr column names
    snr_col = None
    for c in ["snr", "SNR", "snr_db", "snr_name"]:
        if c in df.columns:
            snr_col = c
            break
    if snr_col is None:
        raise KeyError(f"No SNR column found in {csv_path}. Columns: {list(df.columns)}")

    # possible accuracy column names
    acc_col = None
    for c in ["mean_acc", "acc", "accuracy", "episode_mean_acc"]:
        if c in df.columns:
            acc_col = c
            break
    if acc_col is None:
        raise KeyError(f"No accuracy column found in {csv_path}. Columns: {list(df.columns)}")

    snr_vals = df[snr_col].astype(str).str.strip().str.lower()
    mask = snr_vals.isin(["clean", "inf", "infty", "infinite"])
    if not mask.any():
        raise ValueError(f"No clean row found in {csv_path}")

    val = df.loc[mask, acc_col].iloc[0]
    return normalize_acc(val)


def get_clean_acc_from_results(csv_path):
    df = pd.read_csv(csv_path)
    for c in ["episode_mean_acc", "mean_acc", "acc", "accuracy"]:
        if c in df.columns:
            return normalize_acc(df[c].iloc[0])
    raise KeyError(f"No usable accuracy column in {csv_path}. Columns: {list(df.columns)}")


def collect_method_values(root_dir, exp_dir, curve_name, results_name, shot_regex):
    """
    Collect clean accuracies over all shot folders for one method.
    """
    method_root = os.path.join(root_dir, exp_dir)
    if not os.path.isdir(method_root):
        print(f"[Warn] Missing folder: {method_root}")
        return []

    vals = []
    shot_dirs = sorted(os.listdir(method_root))
    for sd in shot_dirs:
        if not re.fullmatch(shot_regex, sd):
            continue

        curve_csv = os.path.join(method_root, sd, curve_name)
        results_csv = os.path.join(method_root, sd, results_name)

        try:
            if os.path.isfile(curve_csv):
                acc = get_clean_acc_from_curve(curve_csv)
                vals.append(acc)
                print(f"[Loaded] {exp_dir}/{sd} <- {curve_csv} | clean={acc:.2f}")
            elif os.path.isfile(results_csv):
                acc = get_clean_acc_from_results(results_csv)
                vals.append(acc)
                print(f"[Loaded] {exp_dir}/{sd} <- {results_csv} | clean={acc:.2f}")
            else:
                print(f"[Warn] No curve/results file found in: {os.path.join(method_root, sd)}")
        except Exception as e:
            print(f"[Warn] Failed to parse {exp_dir}/{sd}: {e}")

    return vals


def main():
    args = parse_args()

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
    plt.rcParams["axes.unicode_minus"] = False

    shot_regex = re.compile(args.shot_pattern)

    data = []
    labels = []

    for exp_dir in DEFAULT_ORDER:
        vals = collect_method_values(
            root_dir=args.root_dir,
            exp_dir=exp_dir,
            curve_name=args.curve_name,
            results_name=args.results_name,
            shot_regex=shot_regex,
        )
        if len(vals) == 0:
            print(f"[Warn] No valid clean accuracies found for {exp_dir}")
            continue

        data.append(vals)
        labels.append(METHOD_NAME_MAP.get(exp_dir, exp_dir))

    if len(data) == 0:
        raise RuntimeError("No valid data found. Check root_dir and file structure.")

    ensure_dir(args.out_png)
    ensure_dir(args.out_pdf)

    fig, ax = plt.subplots(figsize=(9, 6), dpi=180)

    bp = ax.boxplot(
        data,
        labels=labels,
        patch_artist=True,
        showmeans=False,
        showfliers=True,
        medianprops=dict(linewidth=1.8),
        boxprops=dict(linewidth=1.5),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
    )

    # 不主动指定颜色时，matplotlib 默认样式也行；
    # 这里为了区分方法，给一个比较稳的轻配色
    facecolors = ["#f4cccc", "#cfe2f3", "#d9ead3", "#fce5cd", "#d9d2e9"]
    for patch, fc in zip(bp["boxes"], facecolors[:len(bp["boxes"])]):
        patch.set_facecolor(fc)
        patch.set_alpha(0.85)
    ax.set_ylim(0, 100)   # 纵坐标从0开始

    # 可选：叠加每个shot的具体点
    if args.show_points:
        rng = np.random.RandomState(0)
        for i, vals in enumerate(data, start=1):
            x = rng.normal(loc=i, scale=0.04, size=len(vals))
            ax.scatter(x, vals, s=24, alpha=0.75, zorder=3)

    # ax.set_title(args.title, fontsize=16)
    ax.set_ylabel(args.ylabel, fontsize=15)
    ax.tick_params(axis="x", labelsize=15)
    ax.tick_params(axis="y", labelsize=15)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)

    plt.tight_layout()
    fig.savefig(args.out_png, bbox_inches="tight")
    fig.savefig(args.out_pdf, bbox_inches="tight")
    print(f"[Saved] {os.path.abspath(args.out_png)}")
    print(f"[Saved] {os.path.abspath(args.out_pdf)}")


if __name__ == "__main__":
    main()