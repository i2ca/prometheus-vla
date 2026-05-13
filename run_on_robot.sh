#!/usr/bin/env bash
# Pre-flight + deploy wrapper for inference_realtime_pi05d.py
# Usage:
#   ./run_on_robot.sh <checkpoint_name> [--live]
#
# checkpoint_name: pi05_vanilla_cup3 | pi05_depth_cup3 | pi05_droid | pi05_libero | pi05_vanilla_unitree_toast_then_cup3
# --live         : remove --dry-run (default is dry-run)
#
# Pre-conditions on the ROBOT (NOT this host):
#   - run_g1_server.py running   (ZMQ port 5555 / DDS rt/lowstate)
#   - realsense_server.py running (ZMQ camera publish)
#   - Robot ready, e-stop accessible

set -euo pipefail
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export UNITREE_ROBOT_IP=${ROBOT_IP:-10.9.8.73}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

ROBOT_IP=${ROBOT_IP:-10.9.8.73}
TASK=${TASK:-"Pick up the cup"}
FPS=${FPS:-30}
ACTIONS_PER_CHUNK=${ACTIONS_PER_CHUNK:-50}

REPO=$(cd "$(dirname "$0")" && pwd)
CKPT_NAME=${1:-}
MODE=${2:-"--dry-run"}

if [ -z "$CKPT_NAME" ]; then
    echo "Usage: $0 <checkpoint_name> [--live]"
    echo
    echo "Available checkpoints:"
    ls -1 "$REPO/train/output" | grep -v "^$" | sed "s/^/  /"
    exit 1
fi

CKPT="$REPO/train/output/$CKPT_NAME/checkpoints/best/pretrained_model"
if [ ! -d "$CKPT" ]; then
    echo "[FATAL] checkpoint not found: $CKPT"
    exit 1
fi

if [ "$MODE" = "--live" ]; then
    DRY_FLAG=""
    export PROMETHEUS_DRY_RUN=0
    echo
    echo "============================================================"
    echo " LIVE MODE — actions WILL be sent to robot at $ROBOT_IP"
    echo " Press Ctrl+C within 5s to abort."
    echo "============================================================"
    sleep 5
else
    DRY_FLAG="--dry-run"
    export PROMETHEUS_DRY_RUN=1
fi

# 1. ENV CHECK
echo "[1/5] env check..."
if true; then
    source ~/miniforge3/etc/profile.d/conda.sh
fi
conda activate g1
python -c "import torch; assert torch.cuda.is_available(), \"CUDA unavailable\"; print(\"  cuda OK,\", torch.cuda.get_device_name(0))"

# 2. HF AUTH (PaliGemma tokenizer is gated)
echo "[2/5] HF auth..."
HF_USER=$(huggingface-cli whoami 2>/dev/null | tail -1 || echo "ANONYMOUS")
if [ "$HF_USER" = "ANONYMOUS" ] || echo "$HF_USER" | grep -qi "not logged"; then
    echo "  [WARN] HuggingFace not logged in. PaliGemma tokenizer may fail."
    echo "  Run: huggingface-cli login"
fi
echo "  HF user: $HF_USER"

# 3. ROBOT REACHABLE
echo "[3/5] robot reachable @ $ROBOT_IP..."
if ! ping -c 2 -W 2 "$ROBOT_IP" >/dev/null 2>&1; then
    echo "  [FATAL] robot not pingable at $ROBOT_IP"
    exit 1
fi
echo "  ping OK"

# 4. ZMQ camera port (5555)
echo "[4/5] ZMQ port 5555..."
if ! nc -z -w 3 "$ROBOT_IP" 5555 2>/dev/null; then
    echo "  [WARN] port 5555 not open. Make sure realsense_server.py is running on the robot."
fi

# 5. CHECKPOINT INFO
echo "[5/5] checkpoint: $CKPT_NAME"
echo "  size: $(du -sh "$CKPT" | cut -f1)"
echo "  task prompt: \"$TASK\""
echo "  fps: $FPS  actions_per_chunk: $ACTIONS_PER_CHUNK"

echo
echo "[*] launching inference_realtime_pi05d.py ($MODE)..."
exec python -u "$REPO/inference_realtime_pi05d.py" \
    --checkpoint "$CKPT" \
    --robot-ip "$ROBOT_IP" \
    --task "$TASK" \
    --fps "$FPS" \
    --actions-per-chunk "$ACTIONS_PER_CHUNK" \
    $DRY_FLAG
