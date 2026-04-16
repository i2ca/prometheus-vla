"""Utilitário para carregar uma policy pi05-D treinada e predizer uma ação.

Uso:
    python -m train.inference_pi05_d \
        --checkpoint=/home/hercules/prometheus-vla/train/output/pi05/checkpoints/last/pretrained_model \
        --dataset=Mrwlker/pick_up_the_cup3 --episode=10 --frame=0

Cuidado: como o pi05-D adiciona PointNet + pressure_proj, é preciso re-injetar esses
modulos ANTES de carregar o state_dict, senão as chaves caem em "unexpected keys".
"""
import argparse
import sys
from pathlib import Path

import torch
import safetensors.torch as st

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.policies.factory import make_pre_post_processors


def load_pi05_d(checkpoint_dir: str, device: torch.device):
    from train.pi05_d_injector import inject_pi05_d

    policy = PI05Policy.from_pretrained(checkpoint_dir, strict=False)
    policy.to(device).eval()

    inject_pi05_d(policy, device=device)

    sd_path = Path(checkpoint_dir) / "model.safetensors"
    sd = st.load_file(str(sd_path), device=str(device))
    missing, unexpected = policy.load_state_dict(sd, strict=False)
    injected_loaded = [k for k in sd.keys() if "pointnet" in k or "pressure_proj" in k]
    print(f"Loaded {len(injected_loaded)} injected tensors; missing={len(missing)}, unexpected={len(unexpected)}")
    return policy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dataset", default="Mrwlker/pick_up_the_cup3")
    ap.add_argument("--episode", type=int, default=10)
    ap.add_argument("--frame", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy = load_pi05_d(args.checkpoint, device)

    ds = LeRobotDataset(args.dataset, episodes=[args.episode])
    sample = ds[args.frame]
    batch = {k: (v.unsqueeze(0).to(device) if torch.is_tensor(v) else v) for k, v in sample.items()}

    preprocessor, _ = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=args.checkpoint,
    )
    batch = preprocessor(batch)

    with torch.no_grad():
        action_chunk = policy.predict_action_chunk(batch)
    print("action_chunk shape:", tuple(action_chunk.shape))
    print("first action:", action_chunk[0, 0].cpu().tolist())


if __name__ == "__main__":
    main()
