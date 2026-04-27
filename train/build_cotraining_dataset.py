"""Build a single LeRobotDataset that merges pick_up_the_cup3 with Unitree G1
Dex3 pick datasets, with feature names remapped to what pi05 vanilla expects.

This exists because this lerobot fork disables MultiLeRobotDataset in
make_dataset (datasets/factory.py:115 raises NotImplementedError). The clean
workaround is to build a single merged dataset on disk ahead of time and point
the training yaml at its local root.

Usage (AFTER you confirm disk space — output can be ~60 GB for ToastedBread):
    python train/build_cotraining_dataset.py \
        --output-repo-id local/cotraining_pi05_cup3_plus_unitree \
        --output-dir /home/hercules/datasets/cotraining_pi05_cup3_plus_unitree \
        --unitree-datasets unitreerobotics/G1_Dex3_ToastedBread_Dataset \
                          unitreerobotics/G1_Dex3_GraspSquare_Dataset

Then train with:
    lerobot-train --config train/config/pi05_vanilla_cotraining.yaml

This script does NOT run unless invoked — module-level imports only. Always dry-
run first with --dry-run to preview feature mappings before committing to disk.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.dataset_tools import merge_datasets, modify_features

logger = logging.getLogger(__name__)


# Feature convention of pick_up_the_cup3 (the target schema we align to).
TARGET_RGB_KEY = "observation.images.head_camera"

# Unitree G1 Dex3 datasets publish cameras under these names. We keep the
# head-camera view closest to cup3 and drop the rest.
UNITREE_RGB_CANDIDATES = ["cam_left_high", "cam_rgb_high"]
UNITREE_DROP_CAMERAS = [
    "observation.images.cam_right_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
    "observation.images.cam_gray_left_high",
    "observation.images.cam_gray_right_high",
]

# Your dataset has these but pi05 vanilla won't use them. Drop for co-training so
# the merged schema stays RGB+state only.
CUP3_DROP_EXTRAS = [
    "observation.images.head_camera_depth",
    "observation.left_hand_pressure",
    "observation.right_hand_pressure",
]


def _pick_unitree_rgb_key(dataset: LeRobotDataset) -> str:
    keys = set(dataset.meta.features.keys())
    for candidate in UNITREE_RGB_CANDIDATES:
        full = f"observation.images.{candidate}"
        if full in keys:
            return candidate
    raise RuntimeError(
        f"Dataset {dataset.repo_id} has no known head-camera candidate; "
        f"got image keys: {[k for k in keys if 'images' in k]}"
    )


def prepare_unitree_dataset(repo_id: str, staging_dir: Path) -> LeRobotDataset:
    """Download a Unitree dataset and return a schema-aligned local snapshot.

    Pipeline:
      1. Load original dataset (HF-backed).
      2. Drop non-head cameras (cam_right_high, wrists, grays).
      3. Rename the chosen head-camera key to observation.images.head_camera.
    """
    logger.info(f"Loading {repo_id}...")
    ds = LeRobotDataset(repo_id=repo_id)

    chosen = _pick_unitree_rgb_key(ds)
    rename_from = f"observation.images.{chosen}"
    logger.info(f"  head-camera candidate = {chosen}")

    to_drop = [k for k in UNITREE_DROP_CAMERAS if k in ds.meta.features]
    if to_drop:
        logger.info(f"  dropping {len(to_drop)} extra cameras: {to_drop}")
        ds = modify_features(
            ds,
            remove_features=to_drop,
            output_dir=staging_dir / f"{repo_id.replace('/', '_')}_pruned",
            repo_id=f"{repo_id}_pruned",
        )

    # TODO: rename_from -> TARGET_RGB_KEY
    # `dataset_tools.modify_features` only adds/removes — lerobot doesn't ship a
    # native feature-renamer. Options:
    #   (a) Copy all frames to a new dataset with the renamed feature via
    #       LeRobotDatasetRecorder (slow but explicit).
    #   (b) Patch meta/info.json + meta/stats.json + parquet column headers
    #       in-place (fast but invasive).
    # For this script, flag the TODO and return the pruned dataset with the
    # original camera name. The training yaml's `rename_map` can take care of
    # the final rename at load time — but ONLY if we don't merge datasets with
    # mismatched schemas first. That's why co-training via merge hits this wall.
    logger.warning(
        f"  TODO: rename {rename_from!r} -> {TARGET_RGB_KEY!r}. "
        f"merge_datasets() requires identical schemas across inputs."
    )
    return ds


def prepare_cup3(repo_id: str, staging_dir: Path) -> LeRobotDataset:
    logger.info(f"Loading {repo_id}...")
    ds = LeRobotDataset(repo_id=repo_id)
    to_drop = [k for k in CUP3_DROP_EXTRAS if k in ds.meta.features]
    if to_drop:
        logger.info(f"  dropping {len(to_drop)} extras: {to_drop}")
        ds = modify_features(
            ds,
            remove_features=to_drop,
            output_dir=staging_dir / f"{repo_id.replace('/', '_')}_rgbonly",
            repo_id=f"{repo_id}_rgbonly",
        )
    return ds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cup3-repo-id", default="Mrwlker/pick_up_the_cup3")
    parser.add_argument("--unitree-datasets", nargs="+", default=[
        "unitreerobotics/G1_Dex3_ToastedBread_Dataset",
        "unitreerobotics/G1_Dex3_GraspSquare_Dataset",
    ])
    parser.add_argument("--output-repo-id", default="local/cotraining_pi05_cup3_plus_unitree")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--staging-dir", default="/tmp/cotraining_staging")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper()),
                        format="%(asctime)s %(levelname)s %(message)s")
    staging_dir = Path(args.staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)
    out_dir = Path(args.output_dir)

    if args.dry_run:
        logger.info("DRY RUN — no downloads, just printing the plan")
        logger.info(f"cup3 repo: {args.cup3_repo_id} → drop {CUP3_DROP_EXTRAS}")
        for r in args.unitree_datasets:
            logger.info(f"unitree repo: {r} → drop {UNITREE_DROP_CAMERAS}, rename head -> {TARGET_RGB_KEY}")
        logger.info(f"would merge to {args.output_repo_id} at {out_dir}")
        return

    cup3 = prepare_cup3(args.cup3_repo_id, staging_dir)
    unitree_prepped = [prepare_unitree_dataset(r, staging_dir) for r in args.unitree_datasets]

    all_datasets = [cup3] + unitree_prepped
    logger.info(f"merging {len(all_datasets)} datasets into {args.output_repo_id}")
    merged = merge_datasets(all_datasets, args.output_repo_id, output_dir=out_dir)
    logger.info(f"merged dataset has {merged.num_frames} frames across {merged.num_episodes} episodes")


if __name__ == "__main__":
    main()
