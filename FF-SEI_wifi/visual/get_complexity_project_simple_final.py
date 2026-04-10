"""
Complexity-only profiler for this project.
- No checkpoint loading
- No test metrics
- Just instantiate model and compute Params / MACs / FLOPs

Recommended:
python get_complexity_project_simple_final.py \
  --run_all \
  --mat_all FeatureMatrix_OSU_Stable_WiFi_Wireless_unified.mat \
  --project_root . \
  --save_csv ./complexity_summary.csv
"""

import os
import csv
import json
import inspect
import argparse
import importlib.util
import sys
from contextlib import contextmanager
from types import ModuleType
from thop import profile

import numpy as np
import torch
import torch.nn as nn

try:
    import scipy.io as sio
except Exception:
    sio = None
try:
    import h5py
except Exception:
    h5py = None

MODEL_CFG = {
    "ffmoe": {
        "script": "train_moe_tfcnn_supcon_explain_closedloop_v2.py",
        "input_kind": "moe",
    },
    "finezero": {
        "script": "const_experiment/train_finezero_fssei.py",
        "builder_candidates": ["create_model"],
        "encoder_candidates": ["FineZeroEncoder", "Encoder", "Backbone"],
        "input_kind": "spec",
    },
    "twc_cnn": {
        "script": "const_experiment/train_twc_cnn_fssei.py",
        "alt_scripts": ["const_experiment/train_twc_cnn.py"],
        "builder_candidates": ["create_model"],
        "input_kind": "iq_wt",
    },
    "glformer": {
        "script": "const_experiment/train_glformer_like_tifs2023.py",
        "builder_candidates": [
            "create_model", "build_model", "GLFormerLike", "GLFormerLikeClassifier",
            "GLFormer", "GLFormerClassifier"
        ],
        "input_kind": "spec",
    },
}


def resolve_path(p: str, root: str) -> str:
    if os.path.isabs(p):
        return os.path.normpath(p)
    return os.path.normpath(os.path.join(root, p))


def find_existing_script(root: str, cfg: dict) -> str:
    cands = [cfg.get("script", "")] + cfg.get("alt_scripts", [])
    search_roots = [
        os.path.abspath(root),
        os.getcwd(),
        os.path.dirname(os.path.abspath(__file__)),
        os.path.dirname(os.path.abspath(root)),
    ]
    tried = []
    for sr in search_roots:
        for c in cands:
            p = resolve_path(c, sr)
            tried.append(p)
            if os.path.isfile(p):
                return p
    raise FileNotFoundError(f"No script found from candidates={cands}. Tried={tried}")


def load_mat_auto(path: str):
    if sio is not None:
        try:
            return sio.loadmat(path)
        except (NotImplementedError, ValueError):
            pass
    if h5py is not None:
        out = {}
        with h5py.File(path, "r") as f:
            for k in f.keys():
                out[k] = np.array(f[k])
        return out
    raise RuntimeError("Cannot read MAT: scipy.io and h5py both unavailable")


def pick_first_existing(d, keys):
    for k in keys:
        if k in d:
            return k
    return None


def ensure_2d_feature_matrix(x):
    x = np.asarray(x)
    if x.ndim != 2:
        x = x.reshape(x.shape[0], -1) if x.shape[0] < x.shape[-1] else x.reshape(-1, x.shape[-1])
    if x.shape[0] < x.shape[1] and x.shape[0] < 256:
        return x.T
    return x


def to_1d_int(x):
    y = np.asarray(x).reshape(-1)
    if y.dtype.kind in "f":
        y = y.astype(np.int64)
    elif y.dtype.kind not in "iu":
        y = y.astype(np.int64)
    return y


def ensure_nchw(x, N_expected=None):
    x = np.asarray(x)
    if x.ndim == 3:
        if N_expected is None or x.shape[-1] == N_expected:
            return np.transpose(x, (2, 0, 1))[:, None, :, :]
        return x[:, None, :, :]
    if x.ndim == 4:
        if N_expected is None or x.shape[0] == N_expected:
            return x
        if x.shape[-1] == N_expected:
            return np.transpose(x, (3, 2, 0, 1))
        if x.shape[2] == N_expected:
            return np.transpose(x, (2, 3, 0, 1))
    raise ValueError(f"Unsupported tensor shape: {x.shape}")


def infer_shared_shapes(mat_path: str):
    S = load_mat_auto(mat_path)
    kx = pick_first_existing(S, ["featureMatrix", "feature_matrix", "X"])
    ky = pick_first_existing(S, ["label_id", "device_id", "y", "labels"])
    kspec = pick_first_existing(S, ["specTensor", "spec", "spec_tensor", "SpecTensor"])
    kocc = pick_first_existing(S, ["occTensor", "occ_tensor", "densityTensor", "occ"])
    if kx is None or ky is None or kspec is None:
        raise KeyError("MAT must contain featureMatrix/label_id/specTensor")

    X = ensure_2d_feature_matrix(S[kx]).astype(np.float32)
    y = to_1d_int(S[ky])
    N = X.shape[0]
    Xs = ensure_nchw(S[kspec], N).astype(np.float32)
    Xo = ensure_nchw(S[kocc], N).astype(np.float32) if kocc is not None else None
    return {
        "num_classes": int(np.unique(y).size),
        "feat_dim": int(X.shape[1]),
        "spec_shape": tuple(int(v) for v in Xs.shape[1:]),
        "occ_shape": tuple(int(v) for v in Xo.shape[1:]) if Xo is not None else (1, 64, 64),
    }


@contextmanager
def temp_sys_path(paths):
    old = list(sys.path)
    try:
        for p in reversed([os.path.abspath(p) for p in paths if p]):
            if p not in sys.path:
                sys.path.insert(0, p)
        yield
    finally:
        sys.path[:] = old


class _StopImportAfterDefs(Exception):
    pass


@contextmanager
def patched_argparse_for_import():
    old_parse_args = argparse.ArgumentParser.parse_args
    old_parse_known_args = argparse.ArgumentParser.parse_known_args
    old_parse_intermixed_args = getattr(argparse.ArgumentParser, "parse_intermixed_args", None)
    old_parse_known_intermixed_args = getattr(argparse.ArgumentParser, "parse_known_intermixed_args", None)
    old_error = argparse.ArgumentParser.error
    old_exit = argparse.ArgumentParser.exit
    old_argv = sys.argv[:]

    def stop_parse(*args, **kwargs):
        raise _StopImportAfterDefs()

    def stop_known_parse(*args, **kwargs):
        raise _StopImportAfterDefs()

    def no_error(self, message):
        raise _StopImportAfterDefs()

    def no_exit(self, status=0, message=None):
        raise _StopImportAfterDefs()

    try:
        argparse.ArgumentParser.parse_args = stop_parse
        argparse.ArgumentParser.parse_known_args = stop_known_parse
        if old_parse_intermixed_args is not None:
            argparse.ArgumentParser.parse_intermixed_args = stop_parse
        if old_parse_known_intermixed_args is not None:
            argparse.ArgumentParser.parse_known_intermixed_args = stop_known_parse
        argparse.ArgumentParser.error = no_error
        argparse.ArgumentParser.exit = no_exit
        sys.argv = [old_argv[0]]
        yield
    finally:
        argparse.ArgumentParser.parse_args = old_parse_args
        argparse.ArgumentParser.parse_known_args = old_parse_known_args
        if old_parse_intermixed_args is not None:
            argparse.ArgumentParser.parse_intermixed_args = old_parse_intermixed_args
        if old_parse_known_intermixed_args is not None:
            argparse.ArgumentParser.parse_known_intermixed_args = old_parse_known_intermixed_args
        argparse.ArgumentParser.error = old_error
        argparse.ArgumentParser.exit = old_exit
        sys.argv = old_argv


def load_module_from_path(path: str, project_root: str) -> ModuleType:
    name = f"mod_{abs(hash((path, project_root)))}"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    script_dir = os.path.dirname(os.path.abspath(path))
    extra = [script_dir, project_root, os.path.dirname(script_dir)]
    with temp_sys_path(extra), patched_argparse_for_import():
        try:
            spec.loader.exec_module(module)
        except _StopImportAfterDefs:
            pass
    return module


def filter_kwargs(fn_or_cls, kwargs: dict):
    sig = inspect.signature(fn_or_cls)
    out = {}
    required = []
    for n, p in sig.parameters.items():
        if n == "self":
            continue
        if n in kwargs:
            out[n] = kwargs[n]
        elif p.default is inspect._empty and p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            required.append(n)
    return out, required


def discover_builder_candidates(module):
    scored = []
    for name, obj in module.__dict__.items():
        if name.startswith("_"):
            continue
        lname = name.lower()
        if any(bad in lname for bad in ["dataset", "loader", "loss", "metric", "utils", "collate"]):
            continue
        score = 0
        if inspect.isfunction(obj):
            if any(tok in lname for tok in ["model", "build", "create"]):
                score += 5
            if any(tok in lname for tok in ["cnn", "former", "classifier", "net"]):
                score += 3
            scored.append((score, name))
        elif inspect.isclass(obj):
            try:
                is_mod = issubclass(obj, nn.Module)
            except Exception:
                is_mod = False
            if is_mod:
                if "classifier" in lname:
                    score += 6
                if any(tok in lname for tok in ["model", "net"]):
                    score += 5
                if any(tok in lname for tok in ["cnn", "former"]):
                    score += 4
                if any(tok in lname for tok in ["encoder", "backbone", "block"]):
                    score -= 2
                scored.append((score, name))
        scored.sort(key=lambda x: (-x[0], x[1]))
    return [n for _, n in scored]


def try_build_from_candidates(module, candidates, kwargs, encoder_candidates=None, model_name=""):
    last_err = None
    tried = []
    for name in candidates:
        if not hasattr(module, name):
            continue
        obj = getattr(module, name)
        try:
            sub_kwargs, required = filter_kwargs(obj, kwargs)
            tried.append((name, list(sub_kwargs.keys()), list(required)))
            if "encoder" in required and encoder_candidates:
                enc = None
                for en in encoder_candidates:
                    if hasattr(module, en):
                        enc_cls = getattr(module, en)
                        enc_kwargs, enc_required = filter_kwargs(enc_cls, kwargs)
                        if enc_required:
                            continue
                        enc = enc_cls(**enc_kwargs)
                        break
                if enc is not None:
                    sub_kwargs["encoder"] = enc
                    required = [r for r in required if r != "encoder"]
            if required:
                continue
            model = obj(**sub_kwargs) if inspect.isclass(obj) else obj(**sub_kwargs)
            print(f"[DEBUG] selected builder for {model_name}: {name}")
            return model
        except Exception as e:
            last_err = e
    if last_err is not None:
        msg = f"Cannot build {model_name} from candidates={candidates}. Tried={tried}. Last error={last_err}"
        raise RuntimeError(msg)
    raise RuntimeError(f"Cannot build {model_name} from candidates={candidates}. Tried={tried}")


def build_model(model_name: str, project_root: str, shared: dict):
    if model_name == "ffmoe":
        script = find_existing_script(project_root, MODEL_CFG["ffmoe"])
        mod = load_module_from_path(script, project_root)
        model = mod.FeatureFamilyMoE(
            num_classes=shared["num_classes"],
            spec_in_ch=shared["spec_shape"][0],
            occ_in_ch=shared["occ_shape"][0],
            topk_dim=shared["feat_dim"],
            cnn_emb_dim=64,
            emb_dim=64,
            p_drop=0.2,
            gate_hidden=64,
        )
        return model, "moe"

    cfg = MODEL_CFG[model_name]
    script = find_existing_script(project_root, cfg)
    mod = load_module_from_path(script, project_root)
    common_kwargs = {
        "num_classes": shared["num_classes"],
        "n_classes": shared["num_classes"],
        "num_class": shared["num_classes"],
        "classes": shared["num_classes"],
        "feat_dim": shared["feat_dim"],
        "input_dim": shared["feat_dim"],
        "feature_dim": shared["feat_dim"],
        "in_ch": shared["spec_shape"][0],
        "in_channels": shared["spec_shape"][0],
        "input_channels": shared["spec_shape"][0],
        "img_size": shared["spec_shape"][1],
        "image_size": shared["spec_shape"][1],
        "input_shape": shared["spec_shape"],
        "dropout": 0.2,
        "p_drop": 0.2,
    }

    user_cands = cfg.get("builder_candidates", [])
    auto_cands = discover_builder_candidates(mod)
    merged_cands = []
    for n in user_cands + auto_cands:
        if n not in merged_cands:
            merged_cands.append(n)

    model = try_build_from_candidates(
        mod,
        merged_cands,
        common_kwargs,
        encoder_candidates=cfg.get("encoder_candidates", []),
        model_name=model_name,
    )
    return model, cfg["input_kind"]


# def get_dummy_inputs(input_kind: str, shared: dict, device):
#     if input_kind == "feat":
#         return (torch.randn(1, shared["feat_dim"], device=device),)

#     elif input_kind == "spec":
#         c, h, w = shared["spec_shape"]
#         return (torch.randn(1, c, h, w, device=device),)

#     elif input_kind == "occ":
#         c, h, w = shared["occ_shape"]
#         return (torch.randn(1, c, h, w, device=device),)

#     elif input_kind == "moe":
#         c1, h1, w1 = shared["spec_shape"]
#         c2, h2, w2 = shared["occ_shape"]
#         return (
#             torch.randn(1, shared["feat_dim"], device=device),
#             torch.randn(1, shared["feat_dim"], device=device),
#             torch.randn(1, 1, device=device),
#             torch.randn(1, c1, h1, w1, device=device),
#             torch.randn(1, c2, h2, w2, device=device),
#         )

#     elif input_kind == "iq_wt":
#         # iq: [B, 2, L]
#         # wt: [B, 1, S, T]
#         iq_len = int(shared.get("iq_len", 4096))
#         wt_scales = int(shared.get("wt_scales", 32))
#         wt_time_bins = int(shared.get("wt_time_bins", 256))

#         dummy_iq = torch.randn(1, 2, iq_len).to(device)
#         dummy_wt = torch.randn(1, 1, wt_scales, wt_time_bins).to(device)
#         dummy_inputs = (dummy_iq, dummy_wt)

#     else:
#         raise ValueError(f"Unsupported input_kind: {input_kind}")

#     flops, params = profile(model, inputs=dummy_inputs, verbose=False)

def get_dummy_inputs(input_kind: str, shared: dict, device):
    if input_kind == "feat":
        return (torch.randn(1, shared["feat_dim"], device=device),)

    elif input_kind == "spec":
        c, h, w = shared["spec_shape"]
        return (torch.randn(1, c, h, w, device=device),)

    elif input_kind == "occ":
        c, h, w = shared["occ_shape"]
        return (torch.randn(1, c, h, w, device=device),)

    elif input_kind == "moe":
        c1, h1, w1 = shared["spec_shape"]
        c2, h2, w2 = shared["occ_shape"]
        return (
            torch.randn(1, shared["feat_dim"], device=device),
            torch.randn(1, shared["feat_dim"], device=device),
            torch.randn(1, 1, device=device),
            torch.randn(1, c1, h1, w1, device=device),
            torch.randn(1, c2, h2, w2, device=device),
        )

    elif input_kind == "iq_wt":
        # iq: [B, 2, L]
        # wt: [B, 1, S, T]
        iq_len = int(shared.get("iq_len", 4096))
        wt_scales = int(shared.get("wt_scales", 32))
        wt_time_bins = int(shared.get("wt_time_bins", 256))
        return (
            torch.randn(1, 2, iq_len, device=device),
            torch.randn(1, 1, wt_scales, wt_time_bins, device=device),
        )

    else:
        raise ValueError(f"Unsupported input_kind: {input_kind}")


def profile_model(model, dummy_inputs):
    params = sum(p.numel() for p in model.parameters())
    try:
        from thop import profile
        model.eval()
        with torch.no_grad():
            macs, _ = profile(model, inputs=dummy_inputs, verbose=False)
        flops = 2 * macs
        return int(params), int(macs), int(flops), "thop"
    except Exception as e:
        return int(params), None, None, f"param_only ({e})"


def run_one(model_name: str, mat_all: str, project_root: str, device: str):
    shared = infer_shared_shapes(mat_all)
    model, input_kind = build_model(model_name, project_root, shared)
    dev = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
    model = model.to(dev)
    dummy_inputs = get_dummy_inputs(input_kind, shared, dev)
    params, macs, flops, backend = profile_model(model, dummy_inputs)
    return {
        "model": model_name,
        "input_kind": input_kind,
        "num_classes": shared["num_classes"],
        "feat_dim": shared["feat_dim"],
        "spec_shape": str(shared["spec_shape"]),
        "occ_shape": str(shared["occ_shape"]),
        "params": params,
        "params_M": round(params / 1e6, 6),
        "macs": macs,
        "flops": flops,
        "mflops": round(flops / 1e6, 6) if flops is not None else None,
        "backend": backend,
    }


def save_csv(path: str, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    keys = [
        "model", "input_kind", "num_classes", "feat_dim", "spec_shape", "occ_shape",
        "params", "params_M", "macs", "flops", "mflops", "backend"
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="ffmoe", choices=["ffmoe", "finezero", "twc_cnn", "glformer"])
    ap.add_argument("--run_all", action="store_true")
    ap.add_argument("--mat_all", type=str, required=True)
    ap.add_argument("--project_root", type=str, default=".")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--save_json", type=str, default="")
    ap.add_argument("--save_csv", type=str, default="")
    args = ap.parse_args()

    rows = []
    model_list = ["ffmoe", "finezero", "twc_cnn", "glformer"] if args.run_all else [args.model]
    for m in model_list:
        try:
            row = run_one(m, args.mat_all, args.project_root, args.device)
            rows.append(row)
            print(f"[{m}] params={row['params_M']:.6f} M, mflops={row['mflops']}, backend={row['backend']}")
        except Exception as e:
            print(f"[ERROR] {m} failed: {e}")

    if args.save_json and rows:
        path = resolve_path(args.save_json, args.project_root)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(rows if args.run_all else rows[0], f, indent=2, ensure_ascii=False)
    if args.save_csv and rows:
        save_csv(resolve_path(args.save_csv, args.project_root), rows)


if __name__ == "__main__":
    main()
