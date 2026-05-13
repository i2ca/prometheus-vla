#!/usr/bin/env python3
# Cheap diagnostic probes for pi05 checkpoints. Runs entirely on the loaded
# policy + synthetic inputs (no sim, no real robot) so it can finish in a
# couple of minutes per checkpoint.
#
# Probe 1 -- parameter norms grouped by top-level module. Useful to confirm
#   that depth-related submodules (pointnet, depth_proj, fuse, ...) carry
#   non-trivial weight in the "depth" checkpoint but are absent / zero in
#   the vanilla one.
#
# Probe 2 -- action-chunk distance under input perturbations. For each
#   checkpoint, draw a few synthetic batches whose shapes are taken from
#   policy.config.input_features, then measure
#       || predict_action_chunk(perturbed) - predict_action_chunk(clean) ||
#   for {drop-rgb, drop-depth, drop-proprio, drop-pressure, swap-depth-noise}.
#   The depth checkpoint should react much more strongly to drop-depth /
#   swap-depth-noise than the vanilla one. If it doesn't, depth is inert
#   at inference time and the trained "pi05+depth" run is effectively
#   ignoring the depth modality.

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "lerobot-ext"))

from lerobot.policies.factory import make_pre_post_processors  # noqa: E402
from lerobot.policies.pi05.modeling_pi05 import PI05Policy  # noqa: E402
import safetensors.torch as st  # noqa: E402


def load_policy(checkpoint_dir: str, fusion_mode: str, device: torch.device) -> PI05Policy:
    policy = PI05Policy.from_pretrained(checkpoint_dir, strict=False).to(device).eval()
    if fusion_mode == "none":
        return policy
    if fusion_mode == "depth_only":
        from train.pi05_depth_injector import inject_pi05_depth
        inject_pi05_depth(policy, device=device)
    elif fusion_mode == "full":
        from train.pi05_d_injector import inject_pi05_d
        inject_pi05_d(policy, device=device)
    else:
        raise ValueError(f"unknown fusion_mode={fusion_mode}")
    sd = st.load_file(str(Path(checkpoint_dir) / "model.safetensors"), device=str(device))
    policy.load_state_dict(sd, strict=False)
    return policy


# ---- Probe 1: param norms ---------------------------------------------------

DEPTH_HINTS = ("depth", "point", "pressure", "inject", "fuse", "proj")


def probe_params(policy: PI05Policy, name: str) -> dict:
    grouped: dict[str, dict] = {}
    for n, p in policy.named_parameters():
        parts = n.split(".")
        key = ".".join(parts[:2]) if len(parts) >= 2 else parts[0]
        g = grouped.setdefault(key, {"n": 0, "sumsq": 0.0, "max_abs": 0.0})
        g["n"] += p.numel()
        g["sumsq"] += float((p.detach().float() ** 2).sum())
        g["max_abs"] = max(g["max_abs"], float(p.detach().abs().max()))

    total_n = sum(g["n"] for g in grouped.values())
    print(f"\n[{name}] params total={total_n:,}  groups={len(grouped)}")
    rows = []
    for k, g in grouped.items():
        rms = (g["sumsq"] / max(1, g["n"])) ** 0.5
        rows.append((k, g["n"], rms, g["max_abs"]))
    for k, n, rms, mx in sorted(rows):
        tag = "  <-- depth/inject" if any(s in k.lower() for s in DEPTH_HINTS) else ""
        print(f"  {k:55s} n={n:>11,}  rms={rms:.3e}  max|w|={mx:.3e}{tag}")
    return {k: {"n": v["n"], "rms": (v["sumsq"] / max(1, v["n"])) ** 0.5,
                "max_abs": v["max_abs"]} for k, v in grouped.items()}


# ---- Probe 2: action-chunk distance under perturbation ----------------------

def synth_batch(policy: PI05Policy, device: torch.device, seed: int) -> dict:
    """Build a synthetic batch matching policy.config.input_features."""
    g = torch.Generator(device=device).manual_seed(seed)
    batch: dict = {}
    for name, feat in policy.config.input_features.items():
        shape = tuple(feat.shape)
        if "image" in name:
            batch[name] = torch.rand((1, *shape), generator=g, device=device)
        else:
            batch[name] = torch.randn((1, *shape), generator=g, device=device) * 0.1
    batch["task"] = "Pick up the cup"
    return batch


def clone_batch(b: dict) -> dict:
    return {k: (v.clone() if torch.is_tensor(v) else v) for k, v in b.items()}


def perturb(batch: dict, kind: str, device: torch.device, seed: int) -> dict:
    b = clone_batch(batch)
    if kind == "drop-rgb":
        b["observation.images.head_camera"] = torch.zeros_like(b["observation.images.head_camera"])
    elif kind == "drop-depth":
        b["observation.images.head_camera_depth"] = torch.zeros_like(b["observation.images.head_camera_depth"])
    elif kind == "drop-proprio":
        b["observation.state"] = torch.zeros_like(b["observation.state"])
    elif kind == "drop-pressure":
        for k in ("observation.left_hand_pressure", "observation.right_hand_pressure"):
            if k in b:
                b[k] = torch.zeros_like(b[k])
    elif kind == "swap-depth-noise":
        t = b["observation.images.head_camera_depth"]
        g = torch.Generator(device=device).manual_seed(seed * 31 + 7)
        b["observation.images.head_camera_depth"] = torch.rand(t.shape, generator=g, device=device, dtype=t.dtype)
    else:
        raise ValueError(kind)
    return b


def chunk(policy, preproc, batch) -> torch.Tensor:
    b = preproc(clone_batch(batch))
    with torch.no_grad():
        c = policy.predict_action_chunk(b)
    return c.detach()


PERTURBS = ("drop-rgb", "drop-depth", "drop-proprio", "drop-pressure", "swap-depth-noise")


def probe_action_distance(policy, preproc, name: str, device: torch.device, n_seeds: int) -> dict:
    print(f"\n[{name}] action-distance probe ({n_seeds} seeds)")
    print(f"  {'perturb':16s} | mean abs ||Δ|| | mean relΔ")
    per_seed: dict[str, list[tuple[float, float]]] = {k: [] for k in PERTURBS}
    for seed in range(n_seeds):
        b_clean = synth_batch(policy, device, seed)
        c_clean = chunk(policy, preproc, b_clean)
        cl_norm = float(c_clean.norm()) + 1e-9
        for kind in PERTURBS:
            b_p = perturb(b_clean, kind, device, seed)
            c_p = chunk(policy, preproc, b_p)
            d = float((c_p - c_clean).norm())
            per_seed[kind].append((d, d / cl_norm))

    out: dict = {}
    for kind in PERTURBS:
        ds = [v[0] for v in per_seed[kind]]
        rs = [v[1] for v in per_seed[kind]]
        d_mean, r_mean = float(np.mean(ds)), float(np.mean(rs))
        d_std, r_std = float(np.std(ds)), float(np.std(rs))
        print(f"  {kind:16s} | {d_mean:>8.4f} ±{d_std:.3f} | {r_mean:>5.1%} ±{r_std:.1%}")
        out[kind] = {"abs_mean": d_mean, "abs_std": d_std,
                     "rel_mean": r_mean, "rel_std": r_std,
                     "abs_per_seed": ds, "rel_per_seed": rs}
    return out


# ---- main -------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--fusion-modes", nargs="+", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-seeds", type=int, default=3)
    args = ap.parse_args()
    if len(args.checkpoints) != len(args.fusion_modes):
        sys.exit(f"got {len(args.checkpoints)} ckpts vs {len(args.fusion_modes)} fusion modes")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    report: dict = {}
    for ckpt, fusion in zip(args.checkpoints, args.fusion_modes):
        ckpt_path = Path(ckpt)
        # Tag = the run name (parent of "checkpoints/")
        name = next((p.parent.name for p in ckpt_path.parents if p.name == "checkpoints"),
                    ckpt_path.name)
        print(f"\n=== {name} ({fusion}) ===")
        policy = load_policy(ckpt, fusion, device)
        preproc, _post = make_pre_post_processors(policy_cfg=policy.config, pretrained_path=ckpt)
        params = probe_params(policy, name)
        actions = probe_action_distance(policy, preproc, name, device, args.n_seeds)
        report[name] = {
            "checkpoint": ckpt,
            "fusion_mode": fusion,
            "param_groups": params,
            "action_distance": actions,
        }
        del policy, preproc
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    out_path = out_dir / "probes.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\n[probes] wrote {out_path}")

    # Quick verdict block: contrast depth ckpt vs vanilla on the depth-related
    # perturbations -- the central test of the project's hypothesis.
    if len(report) >= 2:
        tags = list(report.keys())
        print(f"\n=== quick verdict (depth vs depth) ===")
        for kind in ("drop-depth", "swap-depth-noise"):
            print(f"  {kind}:")
            for tag in tags:
                rel = report[tag]["action_distance"][kind]["rel_mean"]
                print(f"    {tag:30s} relΔ={rel:.2%}")


if __name__ == "__main__":
    main()
