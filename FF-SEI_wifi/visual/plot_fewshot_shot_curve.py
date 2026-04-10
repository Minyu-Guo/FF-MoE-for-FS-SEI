"""
Plot few-shot shot-accuracy curves for multiple methods from fewshot_results.csv files.

Now supports BOTH directory styles:

(1) Old:
    <root>/<method_tag>/fewshot_out_v2/way_<NWAY>_shot_<SHOT>/fewshot_results.csv

(2) New (your current layout):
    <root>/fewshot_<method>_like/way_<NWAY>_shot_<SHOT>/fewshot_results.csv

Script scans recursively and relies on path containing:
    .../way_<NWAY>_shot_<SHOT>/fewshot_results.csv

--root ./batch_ml_downstream \

python plot_fewshot_shot_curve.py \
  --root ./const_experiment/fewshot_test \
  --root ./result/fewshot/FF-MoE_test \
  --n_way 5 \
  --shots 1,5,10,15,20,25,30 \
  --out_png ./figs/fewshot_shot_curve_way5.png \
  --out_pdf ./figs/fewshot_shot_curve_way5.pdf \
  --use_percent \
  --legend_loc "lower right"
"""

import os
import re
import csv
import argparse
from typing import Dict, List, Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt

# ===== Global font: Times New Roman =====
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
mpl.rcParams["mathtext.fontset"] = "stix"
mpl.rcParams["axes.unicode_minus"] = False


# =========================
# CLI
# =========================
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        action="append",
        required=True,
        help="Root dir to scan. Can be given multiple times."
    )
    ap.add_argument("--n_way", type=int, default=5, help="Way value to match in way_<n_way>_shot_<shot>")
    ap.add_argument("--shots", type=str, default="5,10,15,20,25,30", help="Comma-separated shots to plot")
    ap.add_argument("--metric_col", type=str, default=None)
    ap.add_argument("--std_col", type=str, default=None)
    ap.add_argument("--out_png", type=str, required=True)
    ap.add_argument("--out_pdf", type=str, default=None)
    ap.add_argument("--title", type=str, default=None)
    ap.add_argument("--ylabel", type=str, default="Recognition Accuracy")
    ap.add_argument("--legend_loc", type=str, default="best")
    ap.add_argument("--dpi", type=int, default=300)

    ap.add_argument("--show_errorbar", action="store_true")
    ap.add_argument("--use_percent", action="store_true")
    ap.add_argument("--ylim", type=str, default=None)
    ap.add_argument("--xlim", type=str, default=None)

    ap.add_argument("--include_methods", type=str, default=None)
    ap.add_argument("--exclude_methods", type=str, default=None)
    ap.add_argument("--save_used_csv", type=str, default=None)
    return ap.parse_args()


# =========================
# CSV / path helpers
# =========================
def safe_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(str(x).strip())
    except Exception:
        return None


def read_csv_first_row(path: str) -> Optional[Dict[str, str]]:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        if len(rows) == 0:
            return None
        return rows[0]
    except Exception as e:
        print(f"[WARN] Failed reading CSV: {path} | {e}")
        return None


def extract_way_shot_from_path(path: str) -> Optional[Tuple[int, int]]:
    m = re.search(r"way_(\d+)_shot_(\d+)", path.replace("\\", "/"))
    if m is None:
        return None
    return int(m.group(1)), int(m.group(2))


def infer_method_tag_from_csv_path(root: str, csv_path: str) -> str:
    """
    Robust method tag inference for BOTH directory styles.

    Strategy:
      1) Find folder name that contains "way_<n>_shot_<s>".
      2) Take the parent folder right before it as method tag.
         - If parent is 'fewshot_out_v2', take one more level up.
    """
    rel = os.path.relpath(csv_path, root)
    parts = rel.split(os.sep)

    # locate the "way_x_shot_y" folder
    way_idx = None
    for i, p in enumerate(parts):
        if re.match(r"^way_\d+_shot_\d+$", p):
            way_idx = i
            break

    if way_idx is not None:
        # parent candidate
        if way_idx - 1 >= 0:
            cand = parts[way_idx - 1]
            if cand in ["fewshot_out_v2", "fewshot_out", "fewshot_results", "results"]:
                if way_idx - 2 >= 0:
                    return parts[way_idx - 2]
            return cand

    # fallback (old logic)
    if len(parts) >= 4 and parts[1] == "fewshot_out_v2":
        return parts[0]
    if "fewshot_out_v2" in parts:
        idx = parts.index("fewshot_out_v2")
        if idx - 1 >= 0:
            return parts[idx - 1]

    return os.path.basename(root.rstrip(os.sep))


def discover_fewshot_csvs(root: str, n_way: int) -> List[Tuple[str, int, str]]:
    out = []
    for dirpath, _, filenames in os.walk(root):
        if "fewshot_results.csv" not in filenames:
            continue
        csv_path = os.path.join(dirpath, "fewshot_results.csv")
        ws = extract_way_shot_from_path(csv_path)
        if ws is None:
            continue
        way, shot = ws
        if way != n_way:
            continue

        method_tag = infer_method_tag_from_csv_path(root, csv_path)
        out.append((method_tag, shot, csv_path))
    return out


def pick_metric_from_row(row: Dict[str, str], preferred: Optional[str]) -> Optional[Tuple[str, float]]:
    if preferred:
        v = safe_float(row.get(preferred))
        if v is not None:
            return preferred, v

    candidates = ["episode_mean", "mean_acc", "acc", "micro_acc", "overall_acc", "mean_accuracy"]
    for k in candidates:
        v = safe_float(row.get(k))
        if v is not None:
            return k, v
    return None


def pick_std_from_row(row: Dict[str, str], preferred: Optional[str]) -> Tuple[Optional[str], Optional[float]]:
    if preferred:
        v = safe_float(row.get(preferred))
        if v is not None:
            return preferred, v

    candidates = ["std_acc", "episode_std", "std", "stddev", "acc_std"]
    for k in candidates:
        v = safe_float(row.get(k))
        if v is not None:
            return k, v
    return None, None


# =========================
# Pretty names / styles
# =========================
def pretty_method_name(tag: str) -> str:
    """
    Update mapping to your new DL baselines:
      fewshot_glformer_like  -> GLFormer
      fewshot_adpmae_like    -> TF-CSS
    """
    mapping = {
        # NEW DL baselines (your folders)
        "ff-moe": "FF-MoE (Ours)",
        "finezero": "FineZero",
        "glformer": "GLFormer",
        "tfc_cnn": "TWC-CNN",
        
        "moe_ours": "FF-MoE (Ours)",
        "FF-MoE_test":"FF-MoE (Ours)",
        "moe": "FF-MoE (Ours)",
        "fewshot_finezero": "FineZero",
        "fewshot_glformer_like": "GLFormer",
        "glformer_like": "GLFormer",
        "fewshot_twc_cnn": "TWC-CNN",
        "tfc_cnn": "TWC-CNN",
        
        # Existing
        "fewshot_sa2sei_like":"SA2SEI",
        "dl_smallcnn_concat_fusionnet": "SmallCNN + Concat",
        "dl_smallcnn_attn_mb": "SmallCNN + Attn",
        "dl_resnet18_concat_mb": "ResNet18 + Concat",
        "moe_ours": "FF-MoE (Ours)",
        "moe": "FF-MoE (Ours)",
        "lightgbm_feat39_proto": "Handcrafted-39D + Proto",
        "feat39_svm_rbf": "SVM",
        "feat39_lgbm":"lightGBM",
        "feat39_knn": "Handcrafted-39D + kNN",
    }
    return mapping.get(tag, tag)


def marker_for_name(name: str) -> str:
    n = name.lower()
    if "moe" in n:
        return "*"
    if "twc-cnn" in n or "twc_cnn" in n:
        return "o"
    if "glformer" in n:
        return "D"
    if "lightgbm" in n:
        return "^"
    if "proto" in n:
        return "x"
    if "finezero" in n:
        return "v"
    if "resnet" in n:
        return "s"
    if "smallcnn" in n:
        return "p"
    return "o"


def linestyle_for_name(name: str) -> str:
    n = name.lower()
    if "moe" in n:
        return "-"
    
    # GLFormer
    if "glformer" in n:
        return "--"

    # TFC-CNN / TWC-CNN
    if "tfc" in n or "twc" in n or "tfc-cnn" in n or "twc-cnn" in n:
        return "-."

    # FineZero
    if "finezero" in n:
        return ":"

    # 其他方法默认
    return "-"


def sort_method_tags(tags: List[str]) -> List[str]:
    """
    Prefer showing Ours + DL baselines prominently:
      0: MoE
      1: GLFormer / TF-CSS
      2: ML (feat39...)
      3: other DL (smallcnn/resnet)
    """
    def rank(tag):
        t = pretty_method_name(tag).lower()
        if "moe" in t:
            return (0, t)
        if "glformer" in t or "tf-css" in t or "tf_css" in t:
            return (1, t)
        if "handcrafted" in t or "knn" in t or "proto" in t or "lightgbm" in t:
            return (2, t)
        if "smallcnn" in t:
            return (3, t)
        if "resnet" in t:
            return (4, t)
        if "finezero" in t:
            return (10, t)
        return (9, t)
    return sorted(tags, key=rank)


def color_for_name(name: str):
    n = name.lower()
    if "moe" in n:
        return "#d62728"  # red
    if "glformer" in n:
        return "#1f77b4"  # blue
    if "twc" in n or "twc_cnn" in n or "twc-cnn" in n:
        return "#ff7f0e"  # orange
    if "finezero" in n:
        return "#2ca02c"  # green for sa2sei

    # ✅ add these two
    # if "svm" in n:
    #     return "#2ca02c"  # green for SVM
    if "handcrafted" in n:
        return "#8c564b"  # brown-ish for handcrafted (optional)

    # if "resnet" in n:
    #     return "#2ca02c"  # green
    if "lgbm" in n:
        return "#9467bd"  # purple
    return None


# =========================
# Main
# =========================
def main():
    args = parse_args()

    shots_target = [int(s.strip()) for s in args.shots.split(",") if s.strip()]
    shots_target_set = set(shots_target)

    include_set = None
    if args.include_methods:
        include_set = set(x.strip() for x in args.include_methods.split(",") if x.strip())

    exclude_set = set()
    if args.exclude_methods:
        exclude_set = set(x.strip() for x in args.exclude_methods.split(",") if x.strip())

    entries = []
    for root in args.root:
        root_abs = os.path.abspath(root)
        if not os.path.isdir(root_abs):
            print(f"[WARN] root not found: {root_abs}")
            continue
        found = discover_fewshot_csvs(root_abs, args.n_way)
        entries.extend([(root_abs, m, s, p) for (m, s, p) in found])
        print(f"[INFO] root={root_abs} | found {len(found)} fewshot_results.csv for way={args.n_way}")

    if len(entries) == 0:
        raise RuntimeError("No fewshot_results.csv found. Check --root and --n_way.")

    data: Dict[str, Dict[int, Dict[str, object]]] = {}
    rows_used_for_export = []

    for root_abs, method_tag, shot, csv_path in entries:
        if shot not in shots_target_set:
            continue
        if include_set is not None and method_tag not in include_set:
            continue
        if method_tag in exclude_set:
            continue

        row = read_csv_first_row(csv_path)
        if row is None:
            continue

        metric_pick = pick_metric_from_row(row, args.metric_col)
        if metric_pick is None:
            print(f"[WARN] No usable metric column in {csv_path}")
            continue
        metric_col_used, metric_val = metric_pick

        std_col_used, std_val = pick_std_from_row(row, args.std_col)

        if method_tag not in data:
            data[method_tag] = {}

        prev = data[method_tag].get(shot, None)
        cur_mtime = os.path.getmtime(csv_path)
        use_this = True
        if prev is not None:
            prev_mtime = prev.get("mtime", -1)
            if cur_mtime <= prev_mtime:
                use_this = False

        if use_this:
            data[method_tag][shot] = {
                "metric": float(metric_val),
                "std": float(std_val) if std_val is not None else None,
                "csv_path": csv_path,
                "root": root_abs,
                "metric_col_used": metric_col_used,
                "std_col_used": std_col_used,
                "mtime": cur_mtime,
            }

    method_tags = [m for m in data.keys() if any(s in data[m] for s in shots_target)]
    method_tags = sort_method_tags(method_tags)

    if len(method_tags) == 0:
        raise RuntimeError("No methods contain requested shots after filtering.")

    plt.figure(figsize=(8.8, 5.8))

    for method_tag in method_tags:
        xvals, yvals, evals = [], [], []
        for s in shots_target:
            if s not in data[method_tag]:
                continue
            rec = data[method_tag][s]
            y = rec["metric"]
            e = rec["std"]

            if args.use_percent:
                y = y * 100.0
                if e is not None:
                    e = e * 100.0

            xvals.append(s)
            yvals.append(y)
            evals.append(e if e is not None else float("nan"))

            rows_used_for_export.append({
                "method_tag": method_tag,
                "method_name": pretty_method_name(method_tag),
                "shot": s,
                "metric": rec["metric"],
                "std": rec["std"],
                "metric_col_used": rec["metric_col_used"],
                "std_col_used": rec["std_col_used"],
                "csv_path": rec["csv_path"],
            })

        if len(xvals) == 0:
            continue

        name = pretty_method_name(method_tag)
        marker = marker_for_name(name)
        ls = linestyle_for_name(name)
        color = color_for_name(name)
        marker_size = 9 if "moe" in name.lower() else 7

        if args.show_errorbar and any((v == v) for v in evals):
            yerr = [0.0 if not (v == v) else v for v in evals]
            plt.errorbar(
                xvals, yvals, yerr=yerr,
                marker=marker, linestyle=ls, linewidth=2, markersize=marker_size,
                capsize=3, label=name,
                color=color
            )
        else:
            plt.plot(
                xvals, yvals,
                marker=marker, linestyle=ls, linewidth=2, markersize=marker_size,
                color=color,
                label=name
            )

    plt.xlabel("Shot", fontsize=15)

    if args.use_percent:
        ylabel = args.ylabel
        if "%" not in ylabel:
            ylabel = ylabel + " (%)"
        plt.ylabel(ylabel, fontsize=15)
    else:
        plt.ylabel(args.ylabel, fontsize=15)

    plt.ylim(10, 100)
    plt.xticks(shots_target, fontsize=13)
    plt.yticks(fontsize=13)
    plt.grid(True, linestyle="--", alpha=0.35)
    leg = plt.legend(loc=args.legend_loc, fontsize=13, frameon=True)


    for txt in leg.get_texts():
        if "FF-MoE" in txt.get_text():
            txt.set_fontweight("bold")

    if args.ylim:
        lo, hi = [float(x.strip()) for x in args.ylim.split(",")]
        plt.ylim(lo, hi)
    if args.xlim:
        lo, hi = [float(x.strip()) for x in args.xlim.split(",")]
        plt.xlim(lo, hi)

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

    print("\n[Summary used for plot]")
    for method_tag in method_tags:
        name = pretty_method_name(method_tag)
        chunks = []
        for s in shots_target:
            if s in data[method_tag]:
                rec = data[method_tag][s]
                chunks.append(f"{s}:{rec['metric']:.4f}")
        print(f"  {name} -> " + ", ".join(chunks))

    if args.save_used_csv:
        out_csv = os.path.abspath(args.save_used_csv)
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        fieldnames = [
            "method_tag", "method_name", "shot", "metric", "std",
            "metric_col_used", "std_col_used", "csv_path"
        ]
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows_used_for_export)
        print(f"[Saved] {out_csv}")


if __name__ == "__main__":
    main()