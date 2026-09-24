#!/usr/bin/env bash
# Servidor do UnifoLM-WMA + ponte ZMQ, para o cliente do FastWAM-D falar com ele.
#
# Sobe DOIS processos: o `real_eval_server.py` (FastAPI, no env `unifolm-wma`)
# e a `pontes/wma/ponte_wma.py` (ZMQ, no mesmo env). O cliente no seu PC continua
# apontando para a porta 5600, como se nada tivesse mudado.
#
#   uso: bash launch_wma_server.sh [GPU] [checkpoint]
set -euo pipefail
GPU="${1:-0}"
CKPT="${2:-}"
ENV=$HOME/miniconda3/envs/unifolm-wma
RAIZ=$HOME/DEV/prometheus-vla/mrwlker

if [ ! -x "$ENV/bin/python" ]; then
    echo "❌ env unifolm-wma não existe. Rode: bash $RAIZ/lerobot-ext/install-unifolm-wma.sh"
    exit 1
fi

# O checkpoint padrão é um POS-TREINADO NOSSO, e não o unifolm_wma_dual.ckpt
# cru, de propósito: com agent_action_dim=29 a cabeça de ação do Dual não
# carrega (shape diferente) e o servidor responderia ruído com cara de ação.
# Ver docs/UNIFOLM_WMA.md §4 e `pontes/wma/confere_checkpoint.py`.
CKPT="${CKPT:-/data/train_output/wma_g1_dex3/checkpoints/last.ckpt}"
if [ ! -f "$CKPT" ]; then
    echo "❌ checkpoint não encontrado: $CKPT"
    echo "   Você ainda não pós-treinou. Ver docs/UNIFOLM_WMA.md §6."
    exit 1
fi

# Mesmo motivo do launch_server_fastwamd.sh: o libzmq puxa o libstdc++ do
# SISTEMA antes de qualquer coisa e as extensões nativas do torch estouram em
# Segmentation fault sem mensagem. Forçar o do conda a vir primeiro resolve.
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="$ENV/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/data/.cache/huggingface/hub}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export CUDA_VISIBLE_DEVICES="$GPU"
# Sem isto o ipopt/BLAS do conda-forge abre uma thread por núcleo e o FPS
# despenca (medido: 83 ms contra 0,8 ms por solve de IK).
export OMP_NUM_THREADS=1

LOG="${LOG:-/data/train_output/wma_server.log}"
touch "$LOG" 2>/dev/null || LOG="$HOME/wma_server.log"

RES="${RES:-$HOME/wma_resultados}"
mkdir -p "$RES"

cd "$RAIZ/unifolm-wma"
echo "== GPU $GPU | checkpoint $CKPT | log $LOG =="
echo "== vídeos preditos em $RES =="

"$ENV/bin/python" -u scripts/evaluation/real_eval_server.py \
    --seed 123 \
    --ckpt_path "$CKPT" \
    --config "$RAIZ/lerobot-ext/config/wma/inferencia_g1_dex3.yaml" \
    --savedir "$RES" \
    --bs 1 --height 320 --width 512 \
    --unconditional_guidance_scale 1.0 \
    --ddim_steps 16 --ddim_eta 1.0 \
    --video_length 16 --frame_stride 2 \
    --timestep_spacing 'uniform_trailing' \
    --guidance_rescale 0.7 --perframe_ae \
    2>&1 | tee -a "$LOG" &
PID_WMA=$!
trap 'kill $PID_WMA 2>/dev/null || true' EXIT

# A ponte só serve depois que o modelo carregou; subir antes faria o cliente
# receber ECONNREFUSED e desistir no primeiro ciclo.
echo "== esperando o WMA responder em :8000 (5 B de pesos, leva minutos) =="
for _ in $(seq 1 120); do
    if "$ENV/bin/python" -c "
import socket,sys
s=socket.socket(); s.settimeout(1)
sys.exit(0 if s.connect_ex(('127.0.0.1',8000))==0 else 1)" 2>/dev/null; then
        break
    fi
    sleep 5
done

cd "$RAIZ/lerobot-ext"
exec "$ENV/bin/python" -u pontes/wma/ponte_wma.py --wma=http://127.0.0.1:8000 --port=5600
