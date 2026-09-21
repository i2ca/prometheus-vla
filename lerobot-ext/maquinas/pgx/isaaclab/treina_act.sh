#!/usr/bin/env bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate prometheus-vla
cd ~/DEV/prometheus-vla/lerobot-ext
exec lerobot-train --config_path=config/train/act_redblock_isaaclab.yaml "$@"
