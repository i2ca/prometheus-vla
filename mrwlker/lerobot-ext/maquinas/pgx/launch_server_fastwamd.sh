#!/usr/bin/env bash
# Servidor de inferência do FastWAM-D na ThinkStation PGX (GB10, aarch64).
#
# É o `maquinas/athena/launch_server_fastwamd.sh` sem as três gambiarras da athena, e cada
# ausência foi conferida na PGX em 15/09/2026:
#  - sem LD_PRELOAD do libstdc++: o do DGX OS já tem GLIBCXX_3.4.33, e
#      python -c "import zmq, torch, lerobot.policies.fastwam"
#    passa sem ele. O segfault calado da athena não existe aqui;
#  - um cache só do HuggingFace, o padrão em ~/.cache: a PGX tem um usuário só, e o
#    fastwam_base + o VAE/text encoder do Wan já foram baixados para lá;
#  - sem CUDA_VISIBLE_DEVICES: há uma GPU só, e ela divide os 121 GiB com a CPU.
#    O `nvidia-smi` mostra a memória como "Not Supported"; acompanhe com `free -h`.
#
# Os checkpoints de LoRA apontam para `lerobot/fastwam_base` pelo id do Hub, não por
# caminho da athena, e por isso carregam aqui sem editar o adapter_config.json.
#
#   uso: bash maquinas/pgx/launch_server_fastwamd.sh [checkpoint]
set -euo pipefail
CKPT="${1:-$HOME/prometheus-dados/train_output/fastwamd_cotreino_sim_real/checkpoints/best/pretrained_model}"
ENV=$HOME/miniforge3/envs/prometheus-vla
export OMP_NUM_THREADS=1

LOG="${LOG:-$HOME/inferencia_server.log}"

cd ~/DEV/prometheus-vla/mrwlker/lerobot-ext
echo "== checkpoint $CKPT | log $LOG =="
exec "$ENV/bin/python" -u init_lerobot_inference_fastwamd_server.py \
    --checkpoint="$CKPT" --port=5600 --debug \
    2>&1 | tee -a "$LOG"
