"""
MoE Gate权重可视化脚本（适配SEI小样本任务）
适配文件：
- 训练脚本：train_moe_tfcnn_supcon_explain_closedloop_v2.py
- 下游脚本：downstream_fssei_fewshot_SNR.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

matplotlib.use('Agg')  

plt.rcParams['figure.dpi'] = 300  # 提高分辨率
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['axes.facecolor'] = 'white'  # 画布背景设为白色（默认透明可能看起来空白）
plt.rcParams['figure.facecolor'] = 'white'
# sns.set_style("whitegrid")  # 若用seaborn，强制显示网格/背景

# ====================== 固定配置======================
EXPERT_NAMES = ["Time", "Freq", "Tf", "Inst"]
# BT数据集类别名称映射，实际上自动搜索了
NAME_MAP = {
    0: "Iphone_4s",
    1: "Iphone_5",
    2: "Iphone_5s",
    3: "Iphone_6",
    4: "Iphone_6S",
    5: "Iphone_7",
    6: "Iphone_7plus",
    7: "LG_G4",
    8: "LG_V20",
    9: "Samsung_J7",
    10: "Samsung_Note2",
    11: "Samsung_S5",
    12: "Samsung_S7 edge",
    13: "Samsung_note3",
    14: "Sony_XperiaM5",
    15: "Xiaomi_Mi6"
}

plt.rcParams["font.serif"] = ["Times New Roman", "Liberation Serif"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 12

# ====================== 基础工具函数 ======================
def compute_gate_entropy(gate_w: np.ndarray, epsilon: float = 1e-8):
    """计算每个样本的Gate熵（熵高=多专家协同，熵低=单专家主导）"""
    gate_w_norm = gate_w / (gate_w.sum(axis=1, keepdims=True) + epsilon)
    entropy = -np.sum(gate_w_norm * np.log(gate_w_norm + epsilon), axis=1)
    return entropy

# ====================== 核心可视化函数 ======================
def plot_expert_load_curve(df_log, out_png, expert_names: list = EXPERT_NAMES):
    """绘制训练/验证阶段各Epoch的专家Gate均值曲线（复用你训练脚本的逻辑并优化）"""
    try:
        cols = [c for c in df_log.columns if c.startswith("gate_val_e")]
        if len(cols) == 0:
            cols = [c for c in df_log.columns if c.startswith("gate_e")]
        if len(cols) == 0:
            print("[Warn] 未找到Gate列，跳过Epoch维度Gate曲线绘制")
            return
        
        cols = sorted(cols, key=lambda x: int(x.split("e")[-1]))
        plt.figure(figsize=(10, 6))
        for idx, c in enumerate(cols):
            label = expert_names[idx] if idx < len(expert_names) else c.replace("gate_val_", "").replace("gate_", "")
            plt.plot(df_log[c].values, label=label, linewidth=1.5)
        
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Mean Gate Weight (Validation Set)", fontsize=12)
        plt.title("Expert Load Curve (Validation Set Mean Gate Weight)", fontsize=14)
        plt.legend(fontsize=10, loc="best")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"✅ 已保存Epoch维度Gate曲线：{out_png}")
    except Exception as e:
        print(f"[Error] 绘制Epoch维度Gate曲线失败：{e}")

def plot_per_class_gate_distribution(
    per_class_gate: np.ndarray,
    out_png: str,
    class_names: list = [NAME_MAP[c] for c in range(16)],
    expert_names: list = EXPERT_NAMES,
    figsize: tuple = (14, 8)
):
    """绘制按类别的Gate权重分布（核心：不同设备依赖不同特征专家）"""
    try:
        plt.figure(figsize=figsize)
        sns.heatmap(
            per_class_gate,
            annot=True, fmt=".3f", cmap="Blues",
            xticklabels=expert_names,
            yticklabels=class_names,
            cbar_kws={"label": "Mean Gate Weight"}
        )
        plt.xlabel("Feature Experts", fontsize=15)
        plt.ylabel("Device Class", fontsize=15)
        # plt.title("Per-Device-Class Mean Gate Weight Distribution", fontsize=14)
        plt.tight_layout()
        plt.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"✅ 已保存按类别Gate分布：{out_png}")
    except Exception as e:
        print(f"[Error] 绘制按类别Gate分布失败：{e}")

def plot_conditional_gate_comparison(
    all_gate_w: np.ndarray,
    condition_labels: np.ndarray,
    out_png: str,
    condition_name: str = "SNR",
    expert_names: list = EXPERT_NAMES,
    figsize: tuple = (10, 6)
):
    """绘制按SNR的Gate权重对比（核心：MoE自适应选择特征来源）"""
    try:
        plt.figure(figsize=figsize)
        unique_conditions = sorted(np.unique(condition_labels))
        cond_gate_mean = []
        for cond in unique_conditions:
            mask = (condition_labels == cond)
            cond_gate = all_gate_w[mask].mean(axis=0)
            cond_gate_mean.append(cond_gate)
        cond_gate_mean = np.array(cond_gate_mean)
        
        # 分组柱状图核心参数设置
        n_conditions = len(unique_conditions)  # SNR数量
        n_experts = len(expert_names)          # 专家数量（固定4个）
        width = 0.8 / n_conditions             # 单个柱子宽度（适配多SNR并列，不重叠）
        x = np.arange(len(expert_names))
        width = 0.2
        # 自定义配色（替换默认Set2，更美观且区分度高）
        colors = ["#8c564b", "#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#9467bd"]  # 蓝/紫/橙/红 …，可按需改
        # 若条件数超过4个，自动适配配色（备用方案）
        if len(unique_conditions) > len(colors):
            colors = plt.cm.Set3(np.linspace(0, 1, len(unique_conditions)))
        # colors = plt.cm.Set2(np.linspace(0, 1, len(unique_conditions)))
        
        for i, (cond, color) in enumerate(zip(unique_conditions, colors)):
            offset = width * (i - len(unique_conditions)/2 + 0.5)
            plt.bar(
                x + offset, cond_gate_mean[i],
                width=width, label=f"{condition_name}={cond}", color=color, alpha=0.8)
        for j, val in enumerate(cond_gate_mean[i]):
                plt.text(
                    x[j] + offset, val + 0.01, f"{val:.3f}", 
                    ha='center', va='bottom', fontsize=8
                )

        plt.xlabel("Feature Experts ", fontsize=12)
        plt.ylabel("Mean Gate Weight", fontsize=12)
        # plt.title(f"Gate Weight Distribution by {condition_name}", fontsize=14)
        plt.xticks(x, expert_names, fontsize=10)
        plt.ylim(0, np.max(cond_gate_mean) * 1.15)
        plt.legend(fontsize=9, loc='upper right', frameon=False)  # 图例更紧凑（无边框）
        plt.grid(True, alpha=0.2, axis="y")  # 弱化网格线，更简洁
        plt.tight_layout(pad=0.8)            # pad减小边距，布局更紧凑

        plt.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"✅ 已保存按{condition_name}的Gate对比：{out_png}")
    except Exception as e:
        print(f"[Error] 绘制按条件Gate对比失败：{e}")

def plot_gate_entropy_analysis(
    all_gate_w: np.ndarray,
    all_true: np.ndarray,
    all_pred: np.ndarray,
    out_png: str,
    class_names: list = [NAME_MAP[c] for c in range(16)],
    figsize: tuple = (10, 6)
):
    """绘制Gate熵分析（核心：困难样本多专家协同）"""
    try:
        gate_entropy = compute_gate_entropy(all_gate_w)
        correct_mask = (all_true == all_pred)
        entropy_correct = gate_entropy[correct_mask]
        entropy_incorrect = gate_entropy[~correct_mask]
        
        plt.figure(figsize=figsize)
        box_data = [entropy_correct, entropy_incorrect]
        box_labels = ["Correct Classification", "Incorrect Classification"]
        plt.boxplot(
            box_data, labels=box_labels, patch_artist=True,
            boxprops={"facecolor": "lightblue"}, medianprops={"color": "red"}
        )
        
        plt.xlabel("Classification Result", fontsize=12)
        plt.ylabel("Gate Entropy (Higher = More Experts Collaboration)", fontsize=12)
        plt.title("Gate Entropy Distribution: Correct vs Incorrect Samples", fontsize=14)
        plt.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        plt.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"✅ 已保存Gate熵分析图：{out_png}")
    except Exception as e:
        print(f"[Error] 绘制Gate熵分析失败：{e}")

# ====================== 整合调用函数（一键运行） ======================
def run_all_gate_visualization(
    save_dir: str,
    per_class_gate: np.ndarray,
    all_gate_w: np.ndarray,
    all_true: np.ndarray,
    all_pred: np.ndarray,
    df_log: pd.DataFrame = None,
    snr_labels: np.ndarray = None
):
    """
    一键运行所有Gate可视化（适配你的SEI小样本任务）
    参数说明：
        save_dir: 图片保存目录（对应你脚本中的args.save_dir）
        per_class_gate: [16, 4] 16个设备类别的平均Gate权重
        all_gate_w: [total_samples, 4] 样本级Gate权重矩阵
        all_true: [total_samples] 真实设备类别ID（0-15）
        all_pred: [total_samples] 预测设备类别ID（0-15）
        df_log: （可选）训练日志DataFrame（含gate_val_e*列）
        snr_labels: （可选）[total_samples] 每个样本的SNR标签
    """
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)
    # 步骤1：统一转为字符串（处理混合类型），去除空格+小写化，避免"Clean"/" clean"等格式问题
    snr_str = np.char.lower(np.char.strip(np.array(snr_labels, dtype=str)))
    # 步骤2：创建清晰的布尔掩码（判断是否为clean），消除逐元素比较警告
    is_clean = (snr_str == "clean")
    # 步骤3：初始化浮点数组，逐元素处理（容错非数值）
    snr_labels_float = np.zeros_like(snr_labels, dtype=np.float64)
    for i in range(len(snr_labels)):
        if is_clean[i]:
            snr_labels_float[i] = 35.0  
        else:
            try:
                snr_labels_float[i] = float(snr_labels[i])
            except (ValueError, TypeError):
                snr_labels_float[i] = 35.0

    snr_labels_float = np.nan_to_num(snr_labels_float, nan=35.0)

    snr_labels = snr_labels_float

    # 1. Epoch维度Gate曲线（补充）
    if df_log is not None:
        plot_expert_load_curve(
            df_log=df_log,
            out_png=os.path.join(save_dir, "expert_load_curve.png")
        )
    
    # 2. 核心1：按设备类别的Gate分布
    plot_per_class_gate_distribution(
        per_class_gate=per_class_gate,
        out_png=os.path.join(save_dir, "per_device_gate_dist.png")
    )
    
    # 3. 核心2：按SNR的Gate对比（下游脚本重点）
    if snr_labels is not None:
        plot_conditional_gate_comparison(
            all_gate_w=all_gate_w,
            condition_labels=snr_labels,
            out_png=os.path.join(save_dir, "snr_gate_comparison.png"),
            condition_name="SNR"
        )
    
    # 4. 核心3：Gate熵分析
    plot_gate_entropy_analysis(
        all_gate_w=all_gate_w,
        all_true=all_true,
        all_pred=all_pred,
        out_png=os.path.join(save_dir, "gate_entropy_analysis.png")
    )

# ====================== 下游脚本专用：混淆样本Gate差值分析 ======================
def plot_confusion_gate_delta(
    delta_rows: list,
    out_png: str,
    figsize: tuple = (12, 8)
):
    """
    适配你下游脚本的compute_gate_delta_rows函数，绘制混淆样本vs正确样本的Gate差值
    delta_rows: compute_gate_delta_rows的返回值
    """
    try:
        # 转换为DataFrame
        df_delta = pd.DataFrame(delta_rows)
        if df_delta.empty:
            print("[Warn] 无有效混淆样本Gate差值数据，跳过绘制")
            return
        
        # 提取差值列
        delta_cols = [col for col in df_delta.columns if col.endswith("_delta")]
        # 构建差值矩阵（行：混淆对，列：专家）
        delta_matrix = df_delta[delta_cols].values
        # 混淆对名称
        confusion_pairs = [f"{row['true_name']}→{row['pred_name']}" for _, row in df_delta.iterrows()]
        # 专家名称（去掉_delta后缀）
        expert_cols = [col.replace("_delta", "") for col in delta_cols]
        
        plt.figure(figsize=figsize)
        sns.heatmap(
            delta_matrix,
            annot=True, fmt=".3f", cmap="RdBu_r", center=0,
            xticklabels=expert_cols,
            yticklabels=confusion_pairs,
            cbar_kws={"label": "Gate Delta (Confused - Correct)"}
        )
        plt.xlabel("Feature Experts (I/Q Occupancy Map)", fontsize=12)
        plt.ylabel("Confusion Pair (True→Pred)", fontsize=12)
        plt.title("Gate Weight Delta: Confused Samples vs Correct Samples", fontsize=14)
        plt.tight_layout()
        plt.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"已保存混淆样本Gate差值图：{out_png}")
    except Exception as e:
        print(f"[Error] 绘制混淆样本Gate差值失败：{e}")


def plot_random_subset_gate_bar(
    df_raw: pd.DataFrame,
    out_png: str,
    n_samples: int = 100,
    seed: int = 42,
    stratify_by: str = None,   # None / "is_correct" / "snr"
    expert_cols: list = None,
    figsize: tuple = (7, 5),
):
    """
    从样本级 gate raw 表中抽取 n_samples 个样本，
    画四个专家的均值柱状图 + 标准差误差线。
    """
    try:
        if expert_cols is None:
            expert_cols = [c for c in ["Time", "Freq", "TF", "Inst"] if c in df_raw.columns]

        if len(expert_cols) == 0:
            print("[Warn] No expert columns found in df_raw.")
            return

        df = df_raw.copy()
        rng = np.random.RandomState(seed)

        # ---------- 抽样 ----------
        if stratify_by is None or stratify_by not in df.columns:
            n = min(n_samples, len(df))
            df_sub = df.sample(n=n, random_state=seed, replace=False)

        else:
            groups = []
            uniq = list(df[stratify_by].dropna().unique())
            if len(uniq) == 0:
                n = min(n_samples, len(df))
                df_sub = df.sample(n=n, random_state=seed, replace=False)
            else:
                per_group = max(1, n_samples // len(uniq))
                for g in uniq:
                    dfg = df[df[stratify_by] == g]
                    if len(dfg) == 0:
                        continue
                    take = min(per_group, len(dfg))
                    groups.append(dfg.sample(n=take, random_state=seed, replace=False))
                df_sub = pd.concat(groups, axis=0).reset_index(drop=True)

                # 若还没到 n_samples，剩余部分随机补齐
                if len(df_sub) < min(n_samples, len(df)):
                    remain = df.drop(df_sub.index, errors="ignore")
                    need = min(n_samples, len(df)) - len(df_sub)
                    if len(remain) > 0 and need > 0:
                        extra = remain.sample(n=min(need, len(remain)), random_state=seed, replace=False)
                        df_sub = pd.concat([df_sub, extra], axis=0).reset_index(drop=True)

        vals = df_sub[expert_cols].to_numpy(dtype=float)
        means = np.nanmean(vals, axis=0)
        stds = np.nanstd(vals, axis=0)

        plt.figure(figsize=figsize)
        x = np.arange(len(expert_cols))
        colors = {
            "Time": "#4C72B0",
            "Freq": "#DD8452",
            "TF":   "#55A868",
            "Inst": "#C44E52",
        }
        bar_colors = [colors.get(c, "#4C72B0") for c in expert_cols]

        # plt.bar(x, means, yerr=stds, capsize=4, alpha=0.85)
        plt.bar(
            x, means,
            # yerr=stds,
            # capsize=4,
            alpha=0.9,
            color=bar_colors,
            edgecolor="black",
            linewidth=0.8
        )
        plt.xticks(x, expert_cols)
        plt.ylabel("Mean Gate Weight")
        plt.xlabel("Experts")
        # plt.title(f"Random-{len(df_sub)} Sample Mean Gate Weights")
        plt.ylim(0, max(1e-6, np.max(means + stds)) * 1.2)
        plt.grid(True, axis="y", alpha=0.25)
        plt.tight_layout()
        plt.savefig(out_png, dpi=300, bbox_inches="tight")
        out_pdf = os.path.splitext(out_png)[0] + ".pdf"
        plt.savefig(out_pdf, bbox_inches="tight")
        plt.close()
        print(f"已保存随机子集Gate均值柱状图：{out_png}")
        print(f"已保存随机子集Gate均值柱状图：{out_pdf}")

    except Exception as e:
        print(f"[Error] plot_random_subset_gate_bar failed: {e}")


def plot_random_subset_top1_freq(
    df_raw: pd.DataFrame,
    out_png: str,
    n_samples: int = 100,
    seed: int = 42,
    stratify_by: str = None,   # None / "is_correct" / "snr"
    expert_col: str = "Top1Expert",
    figsize: tuple = (7, 5),
):
    """
    从样本级 gate raw 表中抽取 n_samples 个样本，
    统计 Top1Expert 频数并画柱状图。
    """
    try:
        if expert_col not in df_raw.columns:
            print(f"[Warn] `{expert_col}` not found in df_raw.")
            return

        df = df_raw.copy()

        # ---------- 抽样 ----------
        if stratify_by is None or stratify_by not in df.columns:
            n = min(n_samples, len(df))
            df_sub = df.sample(n=n, random_state=seed, replace=False)
        else:
            groups = []
            uniq = list(df[stratify_by].dropna().unique())
            if len(uniq) == 0:
                n = min(n_samples, len(df))
                df_sub = df.sample(n=n, random_state=seed, replace=False)
            else:
                per_group = max(1, n_samples // len(uniq))
                for g in uniq:
                    dfg = df[df[stratify_by] == g]
                    if len(dfg) == 0:
                        continue
                    take = min(per_group, len(dfg))
                    groups.append(dfg.sample(n=take, random_state=seed, replace=False))
                df_sub = pd.concat(groups, axis=0).reset_index(drop=True)

        cnt = df_sub[expert_col].value_counts()
        order = [e for e in ["Time", "Freq", "TF", "Inst"] if e in cnt.index] + \
                [e for e in cnt.index if e not in ["Time", "Freq", "TF", "Inst"]]
        cnt = cnt.reindex(order).fillna(0)

        plt.figure(figsize=figsize)
        x = np.arange(len(cnt))
        plt.bar(x, cnt.values, alpha=0.85)
        plt.xticks(x, cnt.index)
        plt.ylabel("Count")
        plt.xlabel("Top-1 Expert")
        plt.title(f"Random-{len(df_sub)} Sample Top-1 Expert Frequency")
        plt.grid(True, axis="y", alpha=0.25)
        plt.tight_layout()
        plt.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"已保存随机子集Top1专家频数图：{out_png}")

    except Exception as e:
        print(f"[Error] plot_random_subset_top1_freq failed: {e}")