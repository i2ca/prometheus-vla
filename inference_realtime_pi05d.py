#!/usr/bin/env python3
"""Real-time inference loop for the pi05-D policy on the Unitree G1 Dex3.

Replaces the policy_server + robot_client async-gRPC pair, which cannot run a
pi05-D model because:
  1. policy_server instantiates a vanilla PI05Policy and never calls
     inject_pi05_d(), so the pointnet/pressure_proj weights load but stay
     disconnected from the forward pass.
  2. the vanilla pipeline has no slot for the custom depth/pressure tokens the
     model was trained with.

What this script does:
  - Connects to the robot through the ZMQ bridge (run_g1_server.py +
    realsense_server.py running on the robot).
  - Loads the checkpoint via load_pi05_d() (from lerobot-ext), which runs
    inject_pi05_d() so the extra prefix tokens are active.
  - Runs a synchronous loop: observation → preprocess → predict_action_chunk →
    postprocess → execute chunk at the specified FPS → repeat.

Usage (after the robot-side services are up):
    python inference_realtime_pi05d.py \
        --checkpoint /home/hercules/prometheus-vla/train/output/pi05/checkpoints/best/pretrained_model \
        --robot-ip 10.9.8.73 \
        --task "Pick up the cup" \
        --fps 30 \
        --actions-per-chunk 50
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch
import cv2

# Import lerobot-ext modules for the pi05-D loader. lerobot-ext lives beside the
# main prometheus-vla repo but isn't a package, so extend sys.path by convention.
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "lerobot-ext"))

from train.inference_pi05_d import load_pi05_d  # noqa: E402  (needs sys.path first)
from lerobot.cameras.zmq.camera_zmq import ZMQCamera  # noqa: E402
from lerobot.cameras.zmq.configuration_zmq import ZMQCameraConfig  # noqa: E402
from lerobot.policies.factory import make_pre_post_processors  # noqa: E402
from lerobot.robots.unitree_g1.unitree_g1_dex3 import (  # noqa: E402
    UnitreeG1Dex3,
    UnitreeG1Dex3Config,
)

logger = logging.getLogger("pi05d_runtime")


def build_observation_batch(
    robot: UnitreeG1Dex3,
    depth_camera: ZMQCamera,
    task: str,
    device: torch.device,
    image_shape_hw: tuple[int, int] = (480, 640),
) -> dict:
    """Convert a raw robot observation into the batched dict the policy expects.

    Matches the exact feature names in the trained checkpoint's config.json:
      - observation.state                 (28,)
      - observation.images.head_camera    (3, 480, 640) float32 in [0, 1]
      - observation.images.head_camera_depth (3, 480, 640)
      - observation.left_hand_pressure    (33,)
      - observation.right_hand_pressure   (33,)
      - task                              (str, later tokenized by preprocessor)

    The driver emits the RGB camera under key ``cam_rgb_high`` (historic name
    used by the ACT policy). The depth stream is fetched out-of-band via
    ``depth_camera`` because the shared driver config can't be extended without
    breaking ACT.
    """
    obs = robot.get_observation()

    # State: body joints followed by hand joints, in the order used during training.
    # observation_features dict ordering is stable on Python 3.7+. pressure features
    # use tuple specs and are filtered out here (they go to their own top-level keys).
    state_vec: list[float] = []
    for name, kind in robot.observation_features.items():
        if kind is float:
            state_vec.append(float(obs[name]))
    state_tensor = torch.tensor(state_vec, dtype=torch.float32, device=device).unsqueeze(0)

    H, W = image_shape_hw

    def to_tensor(img: np.ndarray) -> torch.Tensor:
        if img is None:
            img = np.zeros((H, W, 3), dtype=np.uint8)
        if img.shape[:2] != (H, W):
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        # HWC uint8 -> CHW float [0,1]
        return torch.from_numpy(img).to(device).permute(2, 0, 1).float().div_(255.0).unsqueeze(0)

    rgb_img = obs["cam_rgb_high"]
    depth_img = depth_camera.async_read()

    batch = {
        "observation.state": state_tensor,
        "observation.images.head_camera": to_tensor(rgb_img),
        "observation.images.head_camera_depth": to_tensor(depth_img),
        "observation.left_hand_pressure": torch.from_numpy(
            np.asarray(obs["left_hand_pressure"], dtype=np.float32)
        ).to(device).unsqueeze(0),
        "observation.right_hand_pressure": torch.from_numpy(
            np.asarray(obs["right_hand_pressure"], dtype=np.float32)
        ).to(device).unsqueeze(0),
        "task": task,
    }
    return batch


def action_tensor_to_robot_action(action_vec: torch.Tensor, robot: UnitreeG1Dex3) -> dict:
    """Convert a 28-dim action tensor into the dict send_action() expects."""
    action = action_vec.detach().cpu().numpy().astype(float).tolist()
    out: dict = {}
    for name, _ in robot.action_features.items():
        if not action:
            break
        out[name] = action.pop(0)
    return out


class GracefulKiller:
    """Ctrl+C handler that sets a flag so the main loop can exit cleanly."""

    def __init__(self):
        self.kill = False
        signal.signal(signal.SIGINT, self._handler)
        signal.signal(signal.SIGTERM, self._handler)

    def _handler(self, *_):
        logger.warning("shutdown requested; finishing current chunk and stopping...")
        self.kill = True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="pi05-D pretrained_model directory")
    parser.add_argument("--robot-ip", default="10.9.8.73")
    parser.add_argument("--task", default="Pick up the cup")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--actions-per-chunk", type=int, default=50,
                        help="Number of actions to execute from each predicted chunk. "
                             "Training chunk_size is 50; <=50 is safe, >50 is invalid.")
    parser.add_argument("--control-mode", default="upper_body")
    parser.add_argument("--arm", default="G1_29")
    parser.add_argument("--dry-run", action="store_true",
                        help="Log predicted actions but don't send to the robot.")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"device={device}")

    logger.info("loading pi05-D policy (this loads ~7GB of weights)...")
    policy = load_pi05_d(args.checkpoint, device)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config, pretrained_path=args.checkpoint
    )
    logger.info("policy ready")

    robot_cfg = UnitreeG1Dex3Config(robot_ip=args.robot_ip, control_mode=args.control_mode)
    robot = UnitreeG1Dex3(robot_cfg)
    robot.connect()
    logger.info(f"robot connected at {args.robot_ip}")

    depth_camera = ZMQCamera(
        ZMQCameraConfig(
            server_address=args.robot_ip,
            port=5555,
            camera_name="head_camera_depth",
            width=640,
            height=480,
        )
    )
    depth_camera.connect()
    logger.info("depth camera connected at {}:5555/head_camera_depth".format(args.robot_ip))

    killer = GracefulKiller()
    step_period = 1.0 / args.fps
    chunk_counter = 0

    try:
        while not killer.kill:
            # 1) Fetch a fresh observation and predict a chunk.
            t_obs_start = time.perf_counter()
            batch = build_observation_batch(robot, depth_camera, args.task, device)
            batch = preprocessor(batch)
            with torch.no_grad():
                action_chunk = policy.predict_action_chunk(batch)  # (1, chunk_size, action_dim)
            t_inf = time.perf_counter() - t_obs_start
            logger.info(
                f"chunk {chunk_counter}: predicted in {t_inf*1000:.0f}ms "
                f"(shape {tuple(action_chunk.shape)})"
            )

            # 2) Execute the chunk at the policy FPS.
            steps_to_run = min(args.actions_per_chunk, action_chunk.shape[1])
            for i in range(steps_to_run):
                if killer.kill:
                    break
                loop_start = time.perf_counter()
                action_norm = action_chunk[:, i, :]
                action_out = postprocessor(action_norm).squeeze(0)
                if args.dry_run:
                    if i == 0:
                        logger.info(f"dry-run first action: {action_out.cpu().numpy().round(3).tolist()}")
                else:
                    robot.send_action(action_tensor_to_robot_action(action_out, robot))
                elapsed = time.perf_counter() - loop_start
                sleep_for = step_period - elapsed
                if sleep_for > 0:
                    time.sleep(sleep_for)
            chunk_counter += 1
    finally:
        logger.info("disconnecting robot")
        try:
            depth_camera.disconnect()
        except Exception as e:
            logger.warning(f"depth disconnect raised: {e}")
        try:
            robot.disconnect()
        except Exception as e:
            logger.warning(f"disconnect raised: {e}")


if __name__ == "__main__":
    main()
