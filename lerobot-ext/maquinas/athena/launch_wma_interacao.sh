#!/usr/bin/env bash
# Modo simulação interativa do WMA-0 — o ÚNICO teste que roda zero-shot.
#
# Aqui o modelo não controla nada: recebe um quadro e uma instrução e prevê o
# vídeo do futuro. Não passa pela cabeça de ação, então `agent_action_dim`
# não importa e o `unifolm_wma_dual.ckpt` cru serve como está.
#
# É o teste que responde: o modelo de mundo da Unitree entende uma cena do
# G1? Se entender, pós-treinar a política tem chance; se não, não tem.
#
# O `--dataset unitree_g1_pack_camera` usa o prompt de exemplo que já vem no
# submódulo (imagem + h5 + stats) — não precisa do nosso dataset convertido.
#
#   uso: bash launch_wma_interacao.sh [GPU] [checkpoint] [dataset]
set -euo pipefail
GPU="${1:-0}"
ENV=$HOME/miniconda3/envs/unifolm-wma
RAIZ=$HOME/DEV/prometheus-vla
CKPT="${2:-$(ls -d "$HOME"/.cache/huggingface/hub/models--unitreerobotics--UnifoLM-WMA-0-Dual/snapshots/*/unifolm_wma_dual.ckpt 2>/dev/null | head -1)}"
DATASET="${3:-unitree_g1_pack_camera}"

if [ ! -x "$ENV/bin/python" ]; then
    echo "❌ env unifolm-wma não existe. Rode: bash $RAIZ/lerobot-ext/install-unifolm-wma.sh"
    exit 1
fi
if [ -z "$CKPT" ] || [ ! -f "$CKPT" ]; then
    echo "❌ unifolm_wma_dual.ckpt não está no cache. O repositório é GATED:"
    echo "     hf auth login"
    echo "     # aceite em https://huggingface.co/unitreerobotics/UnifoLM-WMA-0-Dual"
    echo "     hf download unitreerobotics/UnifoLM-WMA-0-Dual --exclude 'assets/*'"
    exit 1
fi

export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export CUDA_VISIBLE_DEVICES="$GPU"
export OMP_NUM_THREADS=1

RES="${RES:-$HOME/wma_interacao}"
mkdir -p "$RES"

cd "$RAIZ/unifolm-wma"
echo "== GPU $GPU | $DATASET | checkpoint $(basename "$CKPT") =="
echo "== vídeos em $RES/$DATASET =="
exec "$ENV/bin/python" -u scripts/evaluation/world_model_interaction.py \
    --seed 123 \
    --ckpt_path "$CKPT" \
    --config "$RAIZ/lerobot-ext/config/wma/interacao_exemplo_g1.yaml" \
    --savedir "$RES/$DATASET" \
    --bs 1 --height 320 --width 512 \
    --unconditional_guidance_scale 1.0 \
    --ddim_steps 50 --ddim_eta 1.0 \
    --prompt_dir "$RAIZ/unifolm-wma/examples/world_model_interaction_prompts" \
    --dataset "$DATASET" \
    --video_length 16 --frame_stride 6 \
    --n_action_steps 16 --exe_steps 16 --n_iter 11 \
    --timestep_spacing 'uniform_trailing' \
    --guidance_rescale 0.7 --perframe_ae
