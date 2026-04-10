"""
inject_experts_into_moe.py

兼容两种单专家 ckpt 保存格式：
1) ckpt 本身就是 state_dict（keys 可能是 feat.0.weight / cls.weight ...）
2) ckpt 是 dict，里面有 model/state_dict/model_state_dict 等字段 （keys 可能已经带 expert_time.* 前缀，也可能不带）

说明：
BT数据集：FeatureMatrix_3.mat    topk_indices.mat
WiFi数据集：FeatureMatrix_Indoor_OSU_unified.mat     topk_indices_wifi.mat

bash:
python inject_experts_into_moe_v2.py   --mat_all FeatureMatrix_Indoor_OSU_wififix_unified.mat   --topk_mat topk_indices_wifi.mat   --time_ckpt ./single_expert_runs_reuse/best_time.pth   --freq_ckpt ./single_expert_runs_reuse//best_freq.pth   --tf_ckpt ./single_expert_runs_reuse//best_tf.pth   --inst_ckpt ./single_expert_runs_reuse//best_inst.pth  --out_ckpt  moe_injected_experts.pth
python inject_experts_into_moe_v2.py \
  --mat_all FeatureMatrix_OSU_Stable_WiFi_Wireless_unified.mat \
  --time_ckpt ./single_expert_runs_osu_stable/best_time.pth \
  --freq_ckpt ./single_expert_runs_osu_stable/best_freq.pth \
  --tf_ckpt ./single_expert_runs_osu_stable/best_tf.pth \
  --inst_ckpt ./single_expert_runs_osu_stable/best_inst.pth \
  --out_ckpt ./moe_injected_experts_osu_stable.pth
"""

import os
import argparse
import numpy as np
import torch
import re

from train_moe_tfcnn_supcon_explain_closedloop_v2 import (  # noqa
    FeatureFamilyMoE,
    load_mat_auto,
    pick_first_existing,
    ensure_2d_feature_matrix,
    to_1d_int,
    ensure_nchw,
    normalize_fs,
    load_topk_union_indices,
)

from sklearn.preprocessing import StandardScaler


def _load_state_dict_any(path: str):
    obj = torch.load(path, map_location="cpu")

    if isinstance(obj, dict) and all(isinstance(k, str) for k in obj.keys()):
        for k in ["state_dict", "model_state_dict", "model", "net", "weights"]:
            if k in obj and isinstance(obj[k], dict):
                return obj[k]
        return obj
    raise ValueError(f"Unrecognized checkpoint format: {path}")


def _strip_common_prefix(k: str) -> str:
    if k.startswith("module."):
        k = k[len("module."):]
    if k.startswith("model."):
        k = k[len("model."):]
    if k.startswith("net."):
        k = k[len("net."):]
    return k


def _build_injected_keys(sd_raw: dict, expert_prefix: str, moe_sd: dict):
    moe_keys = set(moe_sd.keys())

    # 1) strip module/model/net 前缀
    sd = { _strip_common_prefix(k): v for k, v in sd_raw.items() if isinstance(k, str) }

    # 2) 如果 ckpt 里已经带 expert_prefix，先剥掉，避免 expert_time.expert_time.xxx
    sd_rel = {}
    for k, v in sd.items():
        if k.startswith(expert_prefix + "."):
            sd_rel[k[len(expert_prefix) + 1:]] = v
        else:
            sd_rel[k] = v

    injected = {}

    # A) 先尝试：ckpt 已经是 feat/cls/cnn 的同名结构（最理想）
    for k, v in sd_rel.items():
        kk = expert_prefix + "." + k
        if kk in moe_keys and tuple(moe_sd[kk].shape) == tuple(v.shape):
            injected[kk] = v

    # B) 再处理：mlp.<idx>.* -> feat.0/feat.3/cls（确定性映射，不猜 shape）
    # 找到所有 mlp 的 Linear 层 index（只看 weight）
    mlp_weight_idxs = []
    for k in sd_rel.keys():
        m = re.match(r"^mlp\.(\d+)\.weight$", k)
        if m:
            mlp_weight_idxs.append(int(m.group(1)))
    mlp_weight_idxs = sorted(set(mlp_weight_idxs))

    # 常见结构：mlp = [Linear, ReLU, Drop, Linear, ReLU, Drop, Linear]
    # 我们用：第1个Linear -> feat.0；第2个Linear -> feat.3；最后一个Linear -> cls
    idx_map = {}
    if len(mlp_weight_idxs) >= 3:
        idx_map[mlp_weight_idxs[0]] = "feat.0"
        idx_map[mlp_weight_idxs[1]] = "feat.3"
        idx_map[mlp_weight_idxs[-1]] = "cls"
    elif len(mlp_weight_idxs) == 2:
        idx_map[mlp_weight_idxs[0]] = "feat.0"
        idx_map[mlp_weight_idxs[-1]] = "cls"

    for k, v in sd_rel.items():
        m = re.match(r"^mlp\.(\d+)\.(weight|bias)$", k)
        if not m:
            continue
        idx = int(m.group(1))
        wb = m.group(2)
        if idx not in idx_map:
            continue
        kk = f"{expert_prefix}.{idx_map[idx]}.{wb}"
        if kk in moe_keys and tuple(moe_sd[kk].shape) == tuple(v.shape):
            injected[kk] = v

    return injected, len(injected)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mat_all", type=str, default="FeatureMatrix_Indoor_OSU_wififix_unified.mat")
    ap.add_argument("--topk_mat", type=str, default="topk_indices_wifi.mat")

    # 4 experts ckpt
    ap.add_argument("--time_ckpt", type=str, default="best_time.pth")
    ap.add_argument("--freq_ckpt", type=str, default="best_freq.pth")
    ap.add_argument("--tf_ckpt",   type=str, default="best_tf.pth")
    ap.add_argument("--inst_ckpt", type=str, default="best_inst.pth")

    # 如果你想保留某个已有 MoE 的 gate 初始化（或其它），先加载它，再覆盖 experts
    ap.add_argument("--base_moe_ckpt", type=str, default="")

    # MoE 超参
    ap.add_argument("--cnn_emb_dim", type=int, default=64)
    ap.add_argument("--emb_dim", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--gate_hidden", type=int, default=64)

    ap.add_argument("--out_ckpt", type=str, default="moe_injected_experts.pth")

    args = ap.parse_args()

    for k in ["time_ckpt", "freq_ckpt", "tf_ckpt", "inst_ckpt"]:
        p = getattr(args, k)
        print(f"[Arg] {k} = {p} | exists={os.path.isfile(p)}")
        if not os.path.isfile(p):
            raise FileNotFoundError(p)
    # ---------- infer dims from MAT ----------
    if not os.path.isfile(args.mat_all):
        raise FileNotFoundError(args.mat_all)

    print(f"[Load MAT] {args.mat_all}")
    S = load_mat_auto(args.mat_all)

    kX = pick_first_existing(S, ["featureMatrix", "feature_matrix", "X"])
    if kX is None:
        raise KeyError("MAT missing featureMatrix")
    X_all = ensure_2d_feature_matrix(S[kX]).astype(np.float32)
    N, P = X_all.shape
    if P != 39:
        raise ValueError(f"Expect P=39, got {P}")

    ky = pick_first_existing(S, ["label_id", "device_id", "y", "labels"])
    if ky is None:
        raise KeyError("MAT missing label_id/device_id")
    y = to_1d_int(S[ky])
    num_classes = int(np.unique(y).size)

    kfs = pick_first_existing(S, ["fsVector", "fs", "fs_vec"])
    fs_vec = normalize_fs(S[kfs]) if kfs is not None else np.zeros((N,), dtype=np.float32)
    if fs_vec.shape[0] != N:
        raise ValueError("fsVector length mismatch")

    kspec = pick_first_existing(S, ["specTensor", "spec", "spec_tensor", "SpecTensor"])
    if kspec is None:
        raise KeyError("MAT missing specTensor")
    X_spec = ensure_nchw(S[kspec], N).astype(np.float32)
    spec_in_ch = int(X_spec.shape[1])

    # topk_dim（StandardScaler -> X_all_z -> slice union_idx）
    scaler = StandardScaler()
    X_all_z = scaler.fit_transform(X_all).astype(np.float32)

    topk_union = load_topk_union_indices(args.topk_mat, P).astype(np.int64) if os.path.isfile(args.topk_mat) else np.array([], dtype=np.int64)
    X_topk = X_all_z[:, topk_union] if topk_union.size > 0 else np.zeros((N, 0), dtype=np.float32)
    topk_dim = int(X_topk.shape[1])

    print(f"[Infer] num_classes={num_classes}, spec_in_ch={spec_in_ch}, topk_dim={topk_dim}")

    kocc = pick_first_existing(S, ["occTensor","occ_tensor","densityTensor","occ","occMap"])
    if kocc is None:
        raise KeyError("MAT missing occTensor (density map).")
    X_occ = ensure_nchw(S[kocc], N).astype(np.float32)
    occ_in_ch = int(X_occ.shape[1])

    # ---------- build MoE ----------
    moe = FeatureFamilyMoE(
        num_classes=num_classes,
        spec_in_ch=spec_in_ch,
        topk_dim=topk_dim,
        occ_in_ch=occ_in_ch,
        cnn_emb_dim=args.cnn_emb_dim,
        emb_dim=args.emb_dim,
        p_drop=args.dropout,
        gate_hidden=args.gate_hidden
    )

    # 先加载一个 base MoE（比如你之前训过 gate 的 ckpt）
    if args.base_moe_ckpt and os.path.isfile(args.base_moe_ckpt):
        print(f"[Base MoE] loading: {args.base_moe_ckpt}")
        base_sd = _load_state_dict_any(args.base_moe_ckpt)
        moe.load_state_dict(base_sd, strict=False)

    moe_sd = moe.state_dict()
    moe_keys = set(moe_sd.keys())

    # ---------- inject experts ----------
    inject_total = {}

    def do_inject(path, prefix):
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        sd_raw = _load_state_dict_any(path)
        inj, hit = _build_injected_keys(sd_raw, prefix, moe_sd)
        print(f"[Inject] {prefix} <- {os.path.basename(path)} | matched_keys={hit}/{len(sd_raw)}")
        if hit == 0:
            print(f"  [Warn] 0 keys matched for {prefix}. 你的 ckpt 结构可能不一致（比如维度/层名不同）。")
        return inj
    
    inject_total.update(do_inject(args.time_ckpt, "expert_time"))
    inject_total.update(do_inject(args.freq_ckpt, "expert_freq"))
    inject_total.update(do_inject(args.tf_ckpt,   "expert_tf"))
    inject_total.update(do_inject(args.inst_ckpt, "expert_inst"))

    merged = dict(moe_sd)
    merged.update(inject_total)
    moe.load_state_dict(merged, strict=True)

    # ---------- save ----------
    out_dir = os.path.dirname(args.out_ckpt)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    torch.save(moe.state_dict(), args.out_ckpt)
    print(f"[Saved] {args.out_ckpt}")
    print("Done.")


if __name__ == "__main__":
    main()
