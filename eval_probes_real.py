#!/usr/bin/env python3
# Probes on REAL training frames from the dataset used to train pi05_*_cup3.
#
# Closed-loop sim eval is currently broken (sim shuts down before main loop;
# this issue predates the current investigation -- see he_pilot/he_eval/* logs,
# all of which spam "Waiting for robot state..." indefinitely). To still answer
# the central question -- "is the depth checkpoint actually using depth, or
# was real-robot success just proprio leakage?" -- this runs cheap diagnostics
# on the same frames the policies were trained on.
#
# Two diagnostics:
#  1. Action-distance under perturbation, per ablation, on real frames
#     (more meaningful than synth because the proprio is the actual trained
#     manifold, so any "leakage to proprio" pattern shows up here).
#  2. Saliency: gradient of ||predict_action_chunk(b)||^2 w.r.t. each input
#     modality. Reports per-modality gradient RMS and the depth/RGB ratio.
#     If depth ckpt has depth_grad/rgb_grad >> 1 vs vanilla, depth genuinely
#     drives the policy in distribution.

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
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
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
        raise ValueError(fusion_mode)
    sd = st.load_file(str(Path(checkpoint_dir) / "model.safetensors"), device=str(device))
    policy.load_state_dict(sd, strict=False)
    return policy


def make_batch_from_sample(sample: dict, policy: PI05Policy, device: torch.device,
                           task: str = "Pick up the cup") -> dict:
    """LeRobotDataset returns CHW float images in [0,1] for video features and
    1-D tensors for STATE features. Add batch dim and select keys the policy
    actually expects."""
    out: dict = {}
    for name, _feat in policy.config.input_features.items():
        if name not in sample:
            raise KeyError(f"sample missing {name}; have {list(sample.keys())[:8]}")
        v = sample[name]
        if not torch.is_tensor(v):
            v = torch.as_tensor(v)
        out[name] = v.to(device).unsqueeze(0).float()
    out["task"] = task
    return out


def clone_batch(b: dict) -> dict:
    return {k: (v.clone() if torch.is_tensor(v) else v) for k, v in b.items()}


PERTURBS = ("drop-rgb", "drop-depth", "drop-proprio", "drop-pressure", "swap-depth-noise")


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
    return b


def run_chunk(policy, preproc, batch) -> torch.Tensor:
    b = preproc(clone_batch(batch))
    with torch.no_grad():
        return policy.predict_action_chunk(b).detach()


def saliency(policy, preproc, batch: dict, device: torch.device,
             eps: float = 0.05, n_probes: int = 4) -> dict:
    """Finite-difference sensitivity per modality. predict_action_chunk is
    decorated with @torch.no_grad() so autograd is unavailable; instead we
    add small Gaussian noise to each modality at the raw input, recompute the
    chunk, and measure ||Δaction|| / ||noise||. Per-modality 'gradient RMS'
    here is the average finite-difference sensitivity across n_probes seeds.
    """
    clean = run_chunk(policy, preproc, batch)
    keys = [k for k in batch if k.startswith("observation.") and torch.is_tensor(batch[k])]
    out: dict = {}
    for k in keys:
        sens = []
        for s in range(n_probes):
            g = torch.Generator(device=device).manual_seed(s * 17 + 3)
            base = batch[k]
            noise = torch.randn(base.shape, generator=g, device=device, dtype=base.dtype) * eps
            bp = clone_batch(batch)
            bp[k] = base + noise
            chunk_p = run_chunk(policy, preproc, bp)
            d_action = float((chunk_p - clean).norm())
            d_input = float(noise.norm()) + 1e-12
            sens.append(d_action / d_input)
        out[k] = {"grad_rms": float(np.mean(sens)),
                  "grad_max_abs": float(np.max(sens)),
                  "is_zero": False}
    rgb = out.get("observation.images.head_camera", {}).get("grad_rms", 0.0)
    dep = out.get("observation.images.head_camera_depth", {}).get("grad_rms", 0.0)
    out["_summary"] = {"rgb_rms": rgb, "depth_rms": dep,
                       "depth_over_rgb": dep / (rgb + 1e-12)}
    return out


def evaluate_ckpt(ckpt: str, fusion_mode: str, samples: list, device: torch.device,
                  n_seeds: int = 1) -> dict:
    name = next((p.parent.name for p in Path(ckpt).parents if p.name == "checkpoints"),
                Path(ckpt).name)
    print(f"\n=== {name} ({fusion_mode}) ===")
    policy = load_policy(ckpt, fusion_mode, device)
    preproc, _ = make_pre_post_processors(policy_cfg=policy.config, pretrained_path=ckpt)

    abs_per_kind: dict[str, list[float]] = {k: [] for k in PERTURBS}
    rel_per_kind: dict[str, list[float]] = {k: [] for k in PERTURBS}
    sal_summaries: list[dict] = []
    sal_full: list[dict] = []

    for i, sample in enumerate(samples):
        batch = make_batch_from_sample(sample, policy, device)
        clean = run_chunk(policy, preproc, batch)
        cl_norm = float(clean.norm()) + 1e-9

        for kind in PERTURBS:
            for seed in range(n_seeds):
                bp = perturb(batch, kind, device, seed=seed + i * 11)
                cp = run_chunk(policy, preproc, bp)
                d = float((cp - clean).norm())
                abs_per_kind[kind].append(d)
                rel_per_kind[kind].append(d / cl_norm)

        s = saliency(policy, preproc, batch, device)
        sal_summaries.append(s["_summary"])
        sal_full.append({k: v for k, v in s.items() if k != "_summary"})
        print(f"  [{i+1}/{len(samples)}] clean ||a||={cl_norm:.3f}  "
              f"sal rgb={s['_summary']['rgb_rms']:.2e} depth={s['_summary']['depth_rms']:.2e}  "
              f"d/rgb={s['_summary']['depth_over_rgb']:.2f}")

    print(f"  -- action-distance summary --")
    print(f"  {'perturb':16s} | mean abs Δ |  mean relΔ")
    summary_action: dict = {}
    for kind in PERTURBS:
        a = np.array(abs_per_kind[kind])
        r = np.array(rel_per_kind[kind])
        print(f"  {kind:16s} | {a.mean():>8.4f}   | {r.mean():>5.1%} (±{r.std():.1%})")
        summary_action[kind] = {
            "abs_mean": float(a.mean()), "abs_std": float(a.std()),
            "rel_mean": float(r.mean()), "rel_std": float(r.std()),
            "n": int(len(a)),
        }

    rgb_rms = np.array([s["rgb_rms"] for s in sal_summaries])
    dep_rms = np.array([s["depth_rms"] for s in sal_summaries])
    print(f"  -- saliency summary --")
    print(f"  rgb_rms   : mean={rgb_rms.mean():.3e}  std={rgb_rms.std():.3e}")
    print(f"  depth_rms : mean={dep_rms.mean():.3e}  std={dep_rms.std():.3e}")
    print(f"  depth/rgb : mean={(dep_rms/(rgb_rms+1e-12)).mean():.3f}")

    return {
        "checkpoint": ckpt,
        "fusion_mode": fusion_mode,
        "n_frames": len(samples),
        "action_distance": summary_action,
        "saliency": {
            "rgb_rms_mean": float(rgb_rms.mean()),
            "rgb_rms_std": float(rgb_rms.std()),
            "depth_rms_mean": float(dep_rms.mean()),
            "depth_rms_std": float(dep_rms.std()),
            "depth_over_rgb_mean": float((dep_rms / (rgb_rms + 1e-12)).mean()),
            "per_frame": sal_full,
        },
    }


def sample_dataset_frames(dataset: LeRobotDataset, n_frames: int) -> list:
    """Pick frames evenly spread across the dataset (start/mid/end of episodes)."""
    total = len(dataset)
    idxs = np.linspace(0, total - 1, n_frames, dtype=int).tolist()
    print(f"sampling {n_frames} frames from {total} total at idx={idxs[:5]}..{idxs[-3:]}")
    return [dataset[i] for i in idxs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--fusion-modes", nargs="+", required=True)
    ap.add_argument("--repo-id", default="Mrwlker/pick_up_the_cup3")
    ap.add_argument("--n-frames", type=int, default=8)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    if len(args.checkpoints) != len(args.fusion_modes):
        sys.exit(f"got {len(args.checkpoints)} ckpts vs {len(args.fusion_modes)} fusion modes")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"loading dataset {args.repo_id}")
    ds = LeRobotDataset(args.repo_id)
    samples = sample_dataset_frames(ds, args.n_frames)

    report: dict = {}
    for ckpt, fusion in zip(args.checkpoints, args.fusion_modes):
        r = evaluate_ckpt(ckpt, fusion, samples, device)
        report[Path(ckpt).parents[2].name] = r
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    out_path = out_dir / "probes_real.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\n[probes_real] wrote {out_path}")

    if len(report) >= 2:
        print(f"\n=== verdict (real frames) ===")
        for kind in ("drop-rgb", "drop-depth", "drop-proprio", "swap-depth-noise"):
            print(f"  {kind}:")
            for tag, r in report.items():
                rel = r["action_distance"][kind]["rel_mean"]
                print(f"    {tag:30s} relΔ={rel:.2%}")
        print(f"\n  saliency depth/rgb ratio:")
        for tag, r in report.items():
            print(f"    {tag:30s} depth/rgb={r['saliency']['depth_over_rgb_mean']:.3f}")


if __name__ == "__main__":
    main()
