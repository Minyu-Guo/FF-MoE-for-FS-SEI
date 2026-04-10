"""
python plot_fewshot_snr_violin.py \
  --result_dir ./batch_moe_downstream/moe_ours/fewshot_out_v2/way_5_shot_10 \
  --snr_order 0,5,10,15,20,25,30 \
  --out_png ./figs/violin_moe_10shot.png \
  --out_pdf ./figs/violin_moe_10shot.pdf

python plot_fewshot_snr_violin.py \
  --result_dir ./ablation/fewshot_test/exp_full/way_5_shot_10 \
  --snr_order 0,5,10,15,20,25,30 \
  --out_png ./figs/violin_moe_10shot.png \
  --out_pdf ./figs/violin_moe_10shot.pdf
"""

import os
import re
import glob
import json
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# =========================
# style
# =========================
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
plt.rcParams["axes.unicode_minus"] = False


# =========================
# helpers
# =========================
def parse_args():
    ap = argparse.ArgumentParser(
        description="Plot violin plot of episode accuracies across SNRs for few-shot experiments."
    )

    # 方式1：给一个结果目录（推荐）
    ap.add_argument("--result_dir", type=str, default=None,
                    help="Path to one shot result dir, e.g. .../fewshot_out_v2/way_5_shot_10")

    # 方式2：直接给宽表（csv/xlsx）
    ap.add_argument("--wide_file", type=str, default=None,
                    help="Optional wide-format CSV/XLSX file. Columns are SNRs, rows are episodes.")

    # 输出
    ap.add_argument("--out_png", type=str, required=True)
    ap.add_argument("--out_pdf", type=str, default=None)

    # 轴与样式
    ap.add_argument("--xlabel", type=str, default="SNR (dB)")
    ap.add_argument("--ylabel", type=str, default="Accuracy (%)")
    ap.add_argument("--ylim", type=str, default="10,100",
                    help="e.g. 10,100")
    ap.add_argument("--figsize", type=str, default="9,7",
                    help="e.g. 9,7")
    ap.add_argument("--dpi", type=int, default=600)

    # x 轴顺序（按你实验常见顺序）
    ap.add_argument("--snr_order", type=str, default="0,5,10,15,20,25,30",
                    help="Comma-separated SNR order for x-axis. e.g. clean,30dB,25dB,... or 0,5,10,15,20,25,30")

    # 标题
    ap.add_argument("--title", type=str, default=None,
                    help="If omitted, no title is drawn (recommended for paper figures).")

    # 读宽表时的选项
    ap.add_argument("--skiprows", type=int, default=0,
                    help="For wide excel/csv input: rows to skip before data.")
    ap.add_argument("--header", type=str, default="infer",
                    help="'infer' or 'none' for wide file.")

    return ap.parse_args()


def parse_figsize(s):
    parts = [float(x.strip()) for x in s.split(",")]
    if len(parts) != 2:
        raise ValueError(f"Invalid figsize: {s}")
    return tuple(parts)


def parse_ylim(s):
    parts = [float(x.strip()) for x in s.split(",")]
    if len(parts) != 2:
        raise ValueError(f"Invalid ylim: {s}")
    return tuple(parts)


def normalize_snr_label(lbl: str) -> str:
    """
    Convert variants to canonical labels:
      "30", "30dB", "30DB" -> "30"
      "clean" -> "clean"
      "-5dB" -> "-5"
    """
    s = str(lbl).strip()
    sl = s.lower()

    if sl in ["clean", "none", "inf"]:
        return "Noise-free"

    sl = sl.replace("db", "").replace(" ", "")
    # keep sign/number only
    m = re.match(r"^-?\d+(\.\d+)?$", sl)
    if m:
        # if integer-like, drop .0
        try:
            v = float(sl)
            if abs(v - round(v)) < 1e-9:
                return str(int(round(v)))
            else:
                return str(v)
        except Exception:
            return sl
    return s


def parse_snr_order(order_str):
    return [normalize_snr_label(x) for x in order_str.split(",") if x.strip()]


def infer_shot_from_dir(path):
    # match way_5_shot_10
    m = re.search(r"way_(\d+)_shot_(\d+)", path.replace("\\", "/"))
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))


def to_percent_if_needed(vals):
    vals = np.asarray(vals, dtype=float)
    if vals.size == 0:
        return vals
    # If values look like 0~1, convert to %
    if np.nanmax(vals) <= 1.5:
        vals = vals * 100.0
    return vals


# =========================
# loaders
# =========================
def load_from_wide_file(wide_file, skiprows=0, header="infer"):
    """
    Accept wide CSV/XLSX:
      columns = SNR labels
      rows = episode accuracies
    """
    ext = os.path.splitext(wide_file)[1].lower()
    header_arg = None if header.lower() == "none" else "infer"

    if ext in [".xlsx", ".xls"]:
        if header_arg == "infer":
            df = pd.read_excel(wide_file, skiprows=skiprows)
        else:
            df = pd.read_excel(wide_file, header=None, skiprows=skiprows)
    elif ext in [".csv", ".txt"]:
        if header_arg == "infer":
            df = pd.read_csv(wide_file, skiprows=skiprows)
        else:
            df = pd.read_csv(wide_file, header=None, skiprows=skiprows)
    else:
        raise ValueError(f"Unsupported wide_file extension: {ext}")

    # if no header, use default incremental columns and let user rename externally (not ideal)
    # For your use-case, wide file usually has headers or can be manually set before calling.
    if df.shape[1] == 0:
        raise RuntimeError("Empty wide file")

    # Melt to long
    df_long = df.melt(var_name="SNR", value_name="Accuracy")
    df_long = df_long.dropna()

    # Normalize labels
    df_long["SNR"] = df_long["SNR"].astype(str).map(normalize_snr_label)
    df_long["Accuracy"] = pd.to_numeric(df_long["Accuracy"], errors="coerce")
    df_long = df_long.dropna(subset=["Accuracy"])

    df_long["Accuracy"] = to_percent_if_needed(df_long["Accuracy"].values)
    return df_long[["SNR", "Accuracy"]]


def try_read_episode_acc_csv(csv_path):
    """
    Try multiple common CSV formats and return 1D array of episode acc.
    Supported columns:
      acc / episode_acc / accuracy / mean_acc (per-row episode)
    Or single-column numeric CSV.
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None

    if df.shape[0] == 0:
        return None

    # common column names
    for c in ["acc", "episode_acc", "accuracy", "Accuracy"]:
        if c in df.columns:
            vals = pd.to_numeric(df[c], errors="coerce").dropna().values
            if len(vals) > 0:
                return vals

    # if only one column and numeric
    if df.shape[1] == 1:
        vals = pd.to_numeric(df.iloc[:, 0], errors="coerce").dropna().values
        if len(vals) > 0:
            return vals

    return None


def detect_snr_from_filename(path):
    """
    Infer SNR label from file name.
    Accept patterns like:
      episode_accs_5dB.csv, episode_accs_5.csv, snr_5dB_episode_accs.csv, acc_clean.npy
    """
    base = os.path.basename(path)
    low = base.lower()

    if "clean" in low:
        return "clean"

    # prefer patterns with dB
    m = re.search(r'(-?\d+)\s*db', low)
    if m:
        return normalize_snr_label(m.group(1))

    # generic numeric token near separators
    m2 = re.search(r'[_\-](-?\d+)(?:[_\.-]|$)', low)
    if m2:
        return normalize_snr_label(m2.group(1))

    return None


def load_from_result_dir(result_dir):
    """
    Load episode-level accs by SNR from a downstream few-shot result directory.

    Priority:
      1) long-format file if exists (snr_episode_accs.csv etc.)
      2) wide-format file if exists
      3) per-SNR CSV/NPY files
    """
    if not os.path.isdir(result_dir):
        raise FileNotFoundError(f"result_dir not found: {result_dir}")

    # ---- 1) Try long-format combined files ----
    long_candidates = [
        "snr_episode_accs.csv",
        "episode_accs_by_snr.csv",
        "snr_episode_results.csv",
    ]
    for fn in long_candidates:
        p = os.path.join(result_dir, fn)
        if os.path.isfile(p):
            df = pd.read_csv(p)
            cols = {c.lower(): c for c in df.columns}
            snr_col = None
            acc_col = None
            for c in ["snr", "snr_db"]:
                if c in cols:
                    snr_col = cols[c]
                    break
            for c in ["acc", "episode_acc", "accuracy"]:
                if c in cols:
                    acc_col = cols[c]
                    break
            if snr_col and acc_col:
                out = df[[snr_col, acc_col]].copy()
                out.columns = ["SNR", "Accuracy"]
                out["SNR"] = out["SNR"].astype(str).map(normalize_snr_label)
                out["Accuracy"] = pd.to_numeric(out["Accuracy"], errors="coerce")
                out = out.dropna(subset=["Accuracy"])
                out["Accuracy"] = to_percent_if_needed(out["Accuracy"].values)
                return out[["SNR", "Accuracy"]]

    # ---- 2) Try wide-format combined files ----
    wide_candidates = [
        "snr_episode_accs_wide.csv",
        "snr_episode_accs.xlsx",
        "snr_acc_episodes.xlsx",
    ]
    for fn in wide_candidates:
        p = os.path.join(result_dir, fn)
        if os.path.isfile(p):
            return load_from_wide_file(p, skiprows=0, header="infer")

    # ---- 3) Try per-SNR CSV/NPY files ----
    rows = []

    # common filename patterns (broad search)
    cand_files = []
    cand_files += glob.glob(os.path.join(result_dir, "*episode*acc*.csv"))
    cand_files += glob.glob(os.path.join(result_dir, "*episode*.csv"))
    cand_files += glob.glob(os.path.join(result_dir, "*acc*.npy"))
    cand_files += glob.glob(os.path.join(result_dir, "*episode*.npy"))
    cand_files = sorted(set(cand_files))

    # 排除 summary 文件（不是 episode-level）
    skip_names = {
        "fewshot_results.csv", "snr_results.csv", "snr_metrics.csv",
        "per_class_acc.csv", "per_class_acc_test.csv", "top_confusions.csv"
    }
    cand_files = [p for p in cand_files if os.path.basename(p) not in skip_names]

    for p in cand_files:
        snr_lbl = detect_snr_from_filename(p)
        if snr_lbl is None:
            continue

        vals = None
        ext = os.path.splitext(p)[1].lower()
        try:
            if ext == ".csv":
                vals = try_read_episode_acc_csv(p)
            elif ext == ".npy":
                arr = np.load(p, allow_pickle=True)
                arr = np.asarray(arr).reshape(-1)
                # filter numeric
                vals = pd.to_numeric(pd.Series(arr), errors="coerce").dropna().values
            else:
                continue
        except Exception:
            vals = None

        if vals is None or len(vals) == 0:
            continue

        vals = to_percent_if_needed(vals)
        for v in vals:
            rows.append({"SNR": snr_lbl, "Accuracy": float(v)})

    if len(rows) == 0:
        # as a friendly check, maybe only summary exists
        summary_candidates = [
            os.path.join(result_dir, "snr_results.csv"),
            os.path.join(result_dir, "snr_metrics.csv")
        ]
        have_summary = [p for p in summary_candidates if os.path.isfile(p)]
        if have_summary:
            raise RuntimeError(
                "在结果目录中只找到了 SNR 汇总结果（mean/std），没找到每个 SNR 的 episode-level 精度文件。\n"
                "小提琴图需要每个 SNR 下 200 episodes 的精度列表。\n"
                "请在 downstream 脚本里保存每个 SNR 的 episode_accs（csv 或 npy），再运行本脚本。"
            )
        else:
            raise RuntimeError(
                "未在 result_dir 中识别到可用的 episode-level SNR 精度文件（CSV/NPY/宽表）。\n"
                "请检查文件命名，或使用 --wide_file 直接指定宽表文件。"
            )

    return pd.DataFrame(rows, columns=["SNR", "Accuracy"])


def palette_for_snr_order(snr_order):
    """
    Return a dict: {snr_label: color}
    你可以按自己的审美改颜色。
    """
    # 固定映射（推荐）
    base = {
        "clean": "#d62728",  # red
        "30":    "#e377c2",  # pink
        "25":    "#8c564b",  # brown
        "20":    "#9467bd",  # purple
        "15":    "#ff7f0e",  # orange
        "10":    "#2ca02c",  # green
        "5":     "#1f77b4",  # blue
        "0":     "#17becf",  # cyan
        "-5":    "#7f7f7f",  # gray
    }

    # 如果出现未预设的SNR，补默认颜色循环
    fallback_cycle = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
        "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
        "#bcbd22", "#17becf"
    ]
    palette = {}
    j = 0
    for s in snr_order:
        if s in base:
            palette[s] = base[s]
        else:
            palette[s] = fallback_cycle[j % len(fallback_cycle)]
            j += 1
    return palette

# =========================
# plotting
# =========================
def plot_violin(df_long, snr_order, out_png, out_pdf=None,
                xlabel="SNR (dB)", ylabel="Accuracy (%)",
                ylim=(10, 100), figsize=(9, 7), dpi=600, title=None):
    # Keep only requested order (if provided), but also append unseen labels at end
    df = df_long.copy()
    df["SNR"] = df["SNR"].astype(str).map(normalize_snr_label)

    # Filter to existing labels in order, preserve order
    existing = list(df["SNR"].dropna().unique())
    ordered = [x for x in snr_order if x in existing]
    extra = [x for x in existing if x not in ordered]
    final_order = ordered + extra

    # Convert to categorical order
    df["SNR"] = pd.Categorical(df["SNR"], categories=final_order, ordered=True)
    df = df.sort_values("SNR")

    plt.figure(figsize=figsize)

    palette = palette_for_snr_order(final_order)

    sns.violinplot(
        x="SNR",
        y="Accuracy",
        data=df,
        order=final_order,
        palette=palette,
        cut=2,
        inner="box",   # 显示箱体更直观（也可改 'quartile'）
        linewidth=1.2
    )

    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel(ylabel, fontsize=15)

    if title is not None and str(title).strip() != "":
        plt.title(title, fontsize=18)

    plt.grid(axis="y", linestyle="--", alpha=0.35)
    plt.ylim(*ylim)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
    plt.savefig(out_png, dpi=dpi, bbox_inches="tight")
    print(f"[Saved] {out_png}")

    if out_pdf:
        os.makedirs(os.path.dirname(os.path.abspath(out_pdf)), exist_ok=True)
        plt.savefig(out_pdf, bbox_inches="tight")
        print(f"[Saved] {out_pdf}")

    plt.close()


def main():
    args = parse_args()

    if (args.result_dir is None) and (args.wide_file is None):
        raise ValueError("请至少提供 --result_dir 或 --wide_file 之一。")

    if args.wide_file is not None:
        df_long = load_from_wide_file(args.wide_file, skiprows=args.skiprows, header=args.header)
    else:
        df_long = load_from_result_dir(args.result_dir)

    snr_order = parse_snr_order(args.snr_order)
    ylim = parse_ylim(args.ylim)
    figsize = parse_figsize(args.figsize)

    # 若未指定标题，默认不画（论文图更清爽）
    title = args.title

    # 打印统计（方便核对）
    print("\n[Summary by SNR]")
    for snr in sorted(df_long["SNR"].astype(str).unique(), key=lambda x: (x == "clean", float(x) if x not in ["clean"] and re.match(r"^-?\d+(\.\d+)?$", x) else 1e9)):
        vals = df_long[df_long["SNR"].astype(str) == snr]["Accuracy"].values
        if len(vals) == 0:
            continue
        print(f"  SNR={snr:>5s} | n={len(vals):3d} | mean={np.mean(vals):.3f} | std={np.std(vals):.3f}")

    plot_violin(
        df_long=df_long,
        snr_order=snr_order,
        out_png=args.out_png,
        out_pdf=args.out_pdf,
        xlabel=args.xlabel,
        ylabel=args.ylabel,
        ylim=ylim,
        figsize=figsize,
        dpi=args.dpi,
        title=title
    )


if __name__ == "__main__":
    main()
    