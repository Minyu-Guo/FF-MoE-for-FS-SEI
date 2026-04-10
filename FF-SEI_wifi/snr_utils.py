import math
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# 全局字体：Times New Roman
mpl.rcParams["font.family"] = "Times New Roman"
mpl.rcParams["font.serif"] = ["Times New Roman"]  # 保险
mpl.rcParams["mathtext.fontset"] = "stix"         # 公式/希腊字母更像 Times 系
mpl.rcParams["axes.unicode_minus"] = False        # 负号正常显示


def parse_snr_list(s):
    """
    把命令行里的 --snr_db_list 解析成 [(name, snr_db or None), ...]
    e.g. 'clean,20,10' -> [('clean', None), ('20dB', 20.0), ('10dB', 10.0)]
    """
    out = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        tl = tok.lower()
        if tl in ["clean", "none", "inf"]:
            out.append(("clean", None))
        else:
            v = float(tok)
            out.append((f"{v:g}dB", v))
    return out


def add_awgn(H, snr_db, rng):
    """
    在特征 H 上加零均值高斯白噪声，达到给定 SNR（dB）。
    snr_db=None 时返回原始 H 的拷贝（不加噪声）。
    """
    H = np.asarray(H, dtype=np.float32)

    if snr_db is None:
        return H.copy()

    sig_pow = float(np.mean(H ** 2))
    if sig_pow <= 0:
        return H.copy()

    snr_lin = 10.0 ** (snr_db / 10.0)
    noise_pow = sig_pow / snr_lin
    noise_std = math.sqrt(noise_pow)

    noise = rng.normal(loc=0.0, scale=noise_std, size=H.shape).astype(np.float32)
    return H + noise


def run_snr_sweep(
    H, y,
    snr_cfgs,
    eval_func,
    eval_kwargs=None,
    seed=0,
    save_csv=None,
    out_dir=None,          
    save_details=False,    
):
    """
    在多种 SNR 下重复调用 eval_func(H_snr, y, **eval_kwargs)，
    返回一个 DataFrame，并可选保存为 csv。

    参数：
        H, y          : 原始特征和标签
        snr_cfgs      : parse_snr_list 的输出列表 [(name, snr_db or None), ...]
        eval_func     : 评估函数，比如 eval_fewshot_protonet_strict_set
        eval_kwargs   : 传给 eval_func 的其他关键字参数（dict）
        seed          : 噪声随机数种子
        save_csv      : 若不为 None，则保存 SNR-ACC 表到此路径

    兼容两种返回形式：
        eval_func -> (mean_acc, std_acc)
        eval_func -> (mean_acc, std_acc, acc_all)
    """
    eval_kwargs = eval_kwargs or {}
    rows = []
    details_by_snr = {}
    
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)

    for i, (snr_name, snr_db) in enumerate(snr_cfgs):
        rng_noise = np.random.RandomState(seed + 100 * i)
        H_snr = add_awgn(H, snr_db, rng_noise)

        out = eval_func(H_snr, y, **eval_kwargs)
        details_by_snr[str(snr_name)] = out

        # ---------- 兼容三种返回 ----------
        macro_f1 = None
        cm = None
        per_class_acc = None
        labels = None

        if isinstance(out, dict):
            mean_acc = float(out.get("mean_acc", np.nan))
            std_acc  = float(out.get("std_acc", np.nan))
            macro_f1 = out.get("macro_f1", None)
            cm = out.get("cm", None)
            per_class_acc = out.get("per_class_acc", None)
            labels = out.get("labels", None)

        elif isinstance(out, (tuple, list)):
            if len(out) == 3:
                mean_acc, std_acc, _acc_all = out
            elif len(out) == 2:
                mean_acc, std_acc = out
            else:
                raise ValueError(f"eval_func should return 2/3 values or dict, got {len(out)}")
            mean_acc = float(mean_acc)
            std_acc  = float(std_acc)

        else:
            mean_acc = float(out)
            std_acc  = float("nan")

        print(f"[SNR={snr_name}] mean_acc={mean_acc:.4f} std={std_acc:.4f}"
              + (f" macro_f1={float(macro_f1):.4f}" if macro_f1 is not None else ""))

        # CSV 行：多加一列 macro_f1（没有就 NaN）
        rows.append({
            "snr": snr_name,
            "mean_acc": mean_acc,
            "std_acc": std_acc,
            "macro_f1": (float(macro_f1) if macro_f1 is not None else np.nan),
        })

        # 细节保存：cm / per_class_acc / labels
        if save_details and out_dir is not None:
            if cm is not None:
                np.save(os.path.join(out_dir, f"cm_counts_{snr_name}.npy"), np.asarray(cm))
            if per_class_acc is not None:
                np.save(os.path.join(out_dir, f"per_class_acc_{snr_name}.npy"), np.asarray(per_class_acc))
            if labels is not None:
                np.save(os.path.join(out_dir, f"labels_{snr_name}.npy"), np.asarray(labels))

    df = pd.DataFrame(rows, columns=["snr", "mean_acc", "std_acc", "macro_f1"])

    if save_csv is not None:
        df.to_csv(save_csv, index=False)
        print("Saved SNR curve csv ->", save_csv)

    return df, details_by_snr

def plot_cm_png(
    cm_counts, labels, save_png,
    title="Confusion (Counts)",
    normalize=False,         
    cmap="Blues",
    figsize=(12, 10),        
    dpi=200,
    xtick_rotation=45,
    tick_fontsize=9,
    title_fontsize=14,
    vmin=None,
    vmax=None,
    square=True,            
    grid=False,               
    cbar_label=None,          
    max_classes=None,         
):
    cm = np.asarray(cm_counts, dtype=np.float64)
    labels = [str(x) for x in np.asarray(labels).reshape(-1).tolist()]

    # --- 截断前 max_classes 类 ---
    C = cm.shape[0]
    if (max_classes is not None) and (C > int(max_classes)):
        m = int(max_classes)
        cm = cm[:m, :m]
        labels = labels[:m]
        C = m

    # --- normalize ---
    if normalize:
        den = cm.sum(axis=1, keepdims=True) + 1e-12
        cm_plot = cm / den
        vmin_use = 0.0 if vmin is None else vmin
        vmax_use = 1.0 if vmax is None else vmax
    else:
        cm_plot = cm
        vmin_use = 0.0 if vmin is None else vmin 
        vmax_use = vmax 

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    im = ax.imshow(
        cm_plot,
        interpolation="nearest",
        cmap=cmap,
        vmin=vmin_use,
        vmax=vmax_use,
        aspect="equal" if square else "auto"
    )

    ax.set_title(title, fontsize=title_fontsize, pad=12)
    ax.set_xlabel("Pred")
    ax.set_ylabel("True")

    ax.set_xticks(np.arange(C))
    ax.set_yticks(np.arange(C))
    ax.set_xticklabels(labels, fontsize=tick_fontsize)
    ax.set_yticklabels(labels, fontsize=tick_fontsize)
    plt.setp(ax.get_xticklabels(), rotation=xtick_rotation, ha="right", rotation_mode="anchor")

    
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=tick_fontsize)
  
    try:     
        cbar.outline.set_visible(False)
    except Exception:
        pass

    for spine in cbar.ax.spines.values():  
        spine.set_visible(False)

    if cbar_label:
        cbar.set_label(cbar_label, fontsize=tick_fontsize)

    if grid:
        ax.set_xticks(np.arange(-.5, C, 1), minor=True)
        ax.set_yticks(np.arange(-.5, C, 1), minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=0.6)
        ax.tick_params(which="minor", bottom=False, left=False)

    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.tight_layout()
    fig.savefig(save_png, bbox_inches="tight")
    plt.close(fig)


def draw_representative_cms(out_dir_final, snr_list=("clean", "5dB")):
    for snr_name in snr_list:
        cm_path = os.path.join(out_dir_final, f"cm_counts_{snr_name}.npy")
        lb_path = os.path.join(out_dir_final, f"labels_{snr_name}.npy")

        if (not os.path.isfile(cm_path)) or (not os.path.isfile(lb_path)):
            print(f"[Warn] missing CM files for {snr_name}:")
            print("  ", cm_path)
            print("  ", lb_path)
            continue

        cm = np.load(cm_path)
        labels = np.load(lb_path)

        # 1) 行归一化（每类召回）
        plot_cm_png(
            cm, labels,
            save_png=os.path.join(out_dir_final, f"cm_norm_{snr_name}.png"),
            title=f"Confusion Matrix ({snr_name}) - Row Normalized",
            normalize=True,
            figsize=(8, 6),
            xtick_rotation=0,  # 设置为0，标签水平显示
            title_fontsize=10
        )

        # 2) 原始计数
        plot_cm_png(
            cm, labels,
            save_png=os.path.join(out_dir_final, f"cm_counts_{snr_name}.png"),
            title=f"Confusion Matrix ({snr_name}) - Counts",
            normalize=False,
            figsize=(8, 6),
            xtick_rotation=0, 
            title_fontsize=10
        )

        print(f"[CM] saved: cm_norm_{snr_name}.png / cm_counts_{snr_name}.png")


def _subset_cm_by_labels(cm_counts, labels, keep_labels):
    """
    从全 cm_counts (C,C) 中挑选 keep_labels 对应的子矩阵
    labels: 长度 C 的 label 列表（与 cm 的行列顺序一致）——通常是 class_id
    keep_labels: 要保留的 label 列表（class_id）
    """
    labels = np.asarray(labels).reshape(-1)
    cm = np.asarray(cm_counts)

    # 允许 labels 是 numpy int/str
    def _to_key(x):
        try:
            return int(x)
        except Exception:
            return str(x)

    idx_map = {_to_key(l): i for i, l in enumerate(labels.tolist())}

    keep = []
    for l in keep_labels:
        key = _to_key(l)
        if key in idx_map:
            keep.append(idx_map[key])

    keep = np.asarray(keep, dtype=np.int64)
    sub_cm = cm[np.ix_(keep, keep)]
    sub_labels = labels[keep]
    return sub_cm, sub_labels


def draw_hardest_k_cm(
    out_dir_final: str,
    low_snr_key: str = "-5dB",
    topk: int = 10,
    also_plot_clean: bool = True,
):
    """
    自动挑选 lowSNR 下 per_class_acc 最低的 top-k 类，
    画 lowSNR 的子混淆矩阵（以及可选 clean 的相同子矩阵对比）。
    """
    p_acc = os.path.join(out_dir_final, f"per_class_acc_{low_snr_key}.npy")
    p_lbl = os.path.join(out_dir_final, f"labels_{low_snr_key}.npy")
    p_cm  = os.path.join(out_dir_final, f"cm_counts_{low_snr_key}.npy")

    if not (os.path.isfile(p_acc) and os.path.isfile(p_lbl) and os.path.isfile(p_cm)):
        print("[Hard-CM] missing files:")
        print(" ", p_acc)
        print(" ", p_lbl)
        print(" ", p_cm)
        return None

    per = np.load(p_acc).astype(np.float64).reshape(-1)
    labels = np.load(p_lbl).reshape(-1)
    cm_counts = np.load(p_cm)

    ok = np.isfinite(per)
    per2 = per[ok]
    lab2 = labels[ok]
    if per2.size == 0:
        print("[Hard-CM] per_class_acc is empty after filtering.")
        return None

    k = min(int(topk), int(per2.size))
    order = np.argsort(per2)    # 越小越难
    hard_labels = lab2[order[:k]].tolist()

    print(f"[Hard-CM] lowSNR={low_snr_key} top-{k} hardest labels =", hard_labels)
    print(f"[Hard-CM] their per-class acc =", per2[order[:k]])

    sub_cm, sub_labels = _subset_cm_by_labels(cm_counts, labels, hard_labels)

    # lowSNR 子矩阵
    plot_cm_png(
        sub_cm, sub_labels,
        save_png=os.path.join(out_dir_final, f"cm_norm_{low_snr_key}_hard{k}.png"),
        title=f"Hardest-{k} CM ({low_snr_key}) - Row Normalized",
        normalize=True,
        figsize=(8, 6),
        xtick_rotation=0,  # 设置为0，标签水平显示
        title_fontsize=10
    )
    plot_cm_png(
        sub_cm, sub_labels,
        save_png=os.path.join(out_dir_final, f"cm_counts_{low_snr_key}_hard{k}.png"),
        title=f"Hardest-{k} CM ({low_snr_key}) - Counts",
        normalize=False,
        figsize=(8, 6),
        xtick_rotation=0,  # 设置为0，标签水平显示
        title_fontsize=10
    )

    # clean 对比=
    if also_plot_clean:
        p_lbl_c = os.path.join(out_dir_final, "labels_clean.npy")
        p_cm_c  = os.path.join(out_dir_final, "cm_counts_clean.npy")
        if os.path.isfile(p_lbl_c) and os.path.isfile(p_cm_c):
            labels_c = np.load(p_lbl_c).reshape(-1)
            cm_c = np.load(p_cm_c)
            sub_cm_c, sub_labels_c = _subset_cm_by_labels(cm_c, labels_c, hard_labels)

            plot_cm_png(
                sub_cm_c, sub_labels_c,
                save_png=os.path.join(out_dir_final, f"cm_norm_clean_hard{k}.png"),
                title=f"Hardest-{k} CM (clean) - Row Normalized",
                normalize=True,
                figsize=(8, 6),
                xtick_rotation=0, 
                title_fontsize=10
            )
            plot_cm_png(
                sub_cm_c, sub_labels_c,
                save_png=os.path.join(out_dir_final, f"cm_counts_clean_hard{k}.png"),
                title=f"Hardest-{k} CM (clean) - Counts",
                normalize=False,
                figsize=(8, 6),
                xtick_rotation=0, 
                title_fontsize=10
            )
            print(f"[Hard-CM] saved clean + {low_snr_key} hardest-{k} sub-CMs.")
        else:
            print("[Hard-CM] clean cm/labels not found, skip clean comparison.")

    return hard_labels