"""Evaluate trained pi05 best/ checkpoints offline against Humanoid Everyday G1 dataset.

Bypasses LeRobotDataset (HE is v2.1, our lerobot needs v3.0). Reads parquets and mp4s directly.
"""
from __future__ import annotations
import argparse
import json
import sys
import io
from pathlib import Path

import numpy as np
import torch
import safetensors.torch as st
from huggingface_hub import hf_hub_download

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "lerobot-ext"))
import policies  # noqa F401
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi05.modeling_pi05 import PI05Policy

HE_REPO = "USC-PSI-Lab/Humanoid-Everyday-G1"


def load_policy(checkpoint_dir: Path, fusion_mode: str, device: torch.device):
    policy = PI05Policy.from_pretrained(str(checkpoint_dir), strict=False, torch_dtype=torch.bfloat16).to(device).eval()
    if fusion_mode != "none":
        if fusion_mode == "depth_only":
            from train.pi05_depth_injector import inject_pi05_depth
            inject_pi05_depth(policy, device=device)
        elif fusion_mode == "full":
            from train.pi05_d_injector import inject_pi05_d
            inject_pi05_d(policy, device=device)
        sd = st.load_file(str(checkpoint_dir / "model.safetensors"), device=str(device))
        policy.load_state_dict(sd, strict=False)
    return policy


def get_video_frame(video_path: Path, frame_idx: int) -> np.ndarray:
    """Return RGB uint8 frame [H,W,3]."""
    from torchcodec.decoders import VideoDecoder
    dec = VideoDecoder(str(video_path), seek_mode="approximate")
    f = dec.get_frame_at(index=frame_idx)
    arr = f.data  # [C,H,W] uint8 tensor
    if torch.is_tensor(arr):
        arr = arr.permute(1, 2, 0).numpy()
    return arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--fusion-mode", choices=["none", "depth_only", "full"], default="none")
    ap.add_argument("--task-filter", default="pick")
    ap.add_argument("--max-episodes", type=int, default=10)
    ap.add_argument("--frames-per-episode", type=int, default=10)
    ap.add_argument("--task-prompt", default="Pick up the cup")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] device={device}")

    # Fetch metadata
    print("[*] fetching meta ...")
    tasks_file = hf_hub_download(repo_id=HE_REPO, filename="meta/tasks.jsonl", repo_type="dataset")
    eps_file = hf_hub_download(repo_id=HE_REPO, filename="meta/episodes.jsonl", repo_type="dataset")
    info_file = hf_hub_download(repo_id=HE_REPO, filename="meta/info.json", repo_type="dataset")
    info = json.loads(Path(info_file).read_text())
    chunks_size = info.get("chunks_size", 1000)
    data_path_tpl = info["data_path"]
    video_path_tpl = info["video_path"]

    tasks = {}
    with open(tasks_file) as f:
        for line in f:
            d = json.loads(line)
            tasks[d["task_index"]] = d
    matching_idx = {ti for ti, d in tasks.items() if args.task_filter.lower() in d["task"].lower()}
    print(f"    {len(matching_idx)} tasks match {args.task_filter}")

    eps = []
    with open(eps_file) as f:
        for line in f:
            d = json.loads(line)
            ti_field = d.get("tasks", [d.get("task_index")])
            if isinstance(ti_field, list):
                if not ti_field or isinstance(ti_field[0], str):
                    continue
                ti = ti_field[0]
            else:
                ti = ti_field
            if ti in matching_idx:
                eps.append((d["episode_index"], ti, d.get("length", 0)))
    print(f"    {len(eps)} episodes match. Capping at {args.max_episodes}.")
    eps = eps[: args.max_episodes]

    print("[*] loading policy ...")
    ckpt = Path(args.checkpoint)
    policy = load_policy(ckpt, args.fusion_mode, device)
    preproc, postproc = make_pre_post_processors(policy_cfg=policy.config, pretrained_path=str(ckpt))

    import pandas as pd
    per_task_mse = {}
    total_done = 0

    for ep_idx, ti, length in eps:
        chunk = ep_idx // chunks_size
        parquet_rel = data_path_tpl.format(episode_chunk=chunk, episode_index=ep_idx)
        video_rel = video_path_tpl.format(episode_chunk=chunk, episode_index=ep_idx)
        try:
            parquet_path = hf_hub_download(repo_id=HE_REPO, filename=parquet_rel, repo_type="dataset")
            video_path = hf_hub_download(repo_id=HE_REPO, filename=video_rel, repo_type="dataset")
        except Exception as e:
            print(f"  download failed for ep {ep_idx}: {e}")
            continue
        df = pd.read_parquet(parquet_path)
        n = len(df)
        if n == 0:
            continue
        cat = tasks[ti]["category"]
        task_name = tasks[ti]["task"]
        print(f"\n[*] ep={ep_idx} task={task_name}  frames={n}")

        picked = np.linspace(0, n - 1, args.frames_per_episode).astype(int)
        mses = []
        with torch.no_grad():
            for k in picked:
                row = df.iloc[int(k)]
                arm = np.asarray(row["observation.arm_joints"], dtype=np.float32)
                hand = np.asarray(row["observation.hand_joints"], dtype=np.float32)
                state = torch.from_numpy(np.concatenate([arm, hand])).to(device).unsqueeze(0)
                rgb = get_video_frame(Path(video_path), int(row["frame_index"]))
                rgb_t = torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255.0).to(device).unsqueeze(0)
                depth_z = torch.zeros((1, 3, 480, 640), device=device, dtype=torch.float32)
                p_z = torch.zeros((1, 33), device=device, dtype=torch.float32)
                batch = {
                    "observation.state": state,
                    "observation.images.head_camera": rgb_t,
                    "observation.images.head_camera_depth": depth_z,
                    "observation.left_hand_pressure": p_z,
                    "observation.right_hand_pressure": p_z,
                    "task": args.task_prompt,
                }
                batch = preproc(batch)
                try:
                    pred = policy.predict_action_chunk(batch)
                except Exception as e:
                    print(f"  pred fail @ k={k}: {type(e).__name__}: {e}")
                    continue
                gt = np.asarray(row["action"], dtype=np.float32)
                gt_t = torch.from_numpy(gt).to(device)
                pred_first = pred[0, 0] if pred.dim() == 3 else pred[0]
                dim = min(pred_first.shape[-1], gt_t.shape[-1])
                mse = float(((pred_first[..., :dim].float() - gt_t[..., :dim]) ** 2).mean().cpu())
                mses.append(mse)
        if mses:
            per_task_mse.setdefault(cat, []).extend(mses)
            print(f"  ep MSE mean={np.mean(mses):.4f} min={np.min(mses):.4f} max={np.max(mses):.4f}")
            total_done += len(mses)

    summary = {
        "checkpoint": str(ckpt),
        "fusion_mode": args.fusion_mode,
        "task_filter": args.task_filter,
        "n_frames_evaluated": total_done,
        "per_category": {k: {"mean": float(np.mean(v)), "std": float(np.std(v)), "n": len(v)} for k, v in per_task_mse.items()},
    }
    if per_task_mse:
        all_mse = [m for v in per_task_mse.values() for m in v]
        summary["overall_mse_mean"] = float(np.mean(all_mse))
        summary["overall_mse_std"] = float(np.std(all_mse))
    Path(args.out).write_text(json.dumps(summary, indent=2))
    print(f"\n[*] wrote {args.out}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
