#!/usr/bin/env bash
# Modo de simulação interativa do WMA-0 na PGX, dentro do container `unifolm-wma:pgx`.
#
# Faz o mesmo teste do `athena/launch_wma_interacao.sh`: o modelo recebe um quadro e uma
# instrução e prevê o vídeo do futuro, sem passar pela cabeça de ação. A diferença é
# onde ele roda. O env do upstream não instala em ARM, então tudo vai num container
# (ver `pgx/wma/Dockerfile`).
#
# Com --user, os vídeos saem com o dono certo, mas o $HOME do container não é gravável.
# Por isso o HOME aponta para /tmp e o cache do HF é montado à parte. É lá que o
# open_clip grava o ViT-H-14 na primeira execução.
#
#   uso: bash pgx/launch_wma_interacao.sh [dataset]
set -euo pipefail
RAIZ=$HOME/DEV/prometheus-vla
DATASET="${1:-unitree_g1_pack_camera}"
HF=$HOME/.cache/huggingface
CKPT=$(ls -d "$HF"/hub/models--unitreerobotics--UnifoLM-WMA-0-Dual/snapshots/*/unifolm_wma_dual.ckpt 2>/dev/null | head -1)
RES="${RES:-$HOME/wma_interacao}"

if ! docker image inspect unifolm-wma:pgx >/dev/null 2>&1; then
    echo "❌ imagem unifolm-wma:pgx não existe. Rode:"
    echo "     docker build -t unifolm-wma:pgx $RAIZ/lerobot-ext/pgx/wma/"
    exit 1
fi
if [ -z "$CKPT" ]; then
    echo "❌ unifolm_wma_dual.ckpt não está em $HF/hub"
    exit 1
fi
mkdir -p "$RES/$DATASET"

# O `interacao_exemplo_g1.yaml` fixa o `data_dir` em caminho absoluto DO NOTEBOOK
# (/home/miguel/...). Dentro do container o repo é /work, e a 1ª rodada na PGX, em 15/09,
# morreu em 28 s com FileNotFoundError no `unitree_z1_stackbox.csv`. Em vez de editar o
# config compartilhado (que o notebook usa como está), cada rodada grava uma cópia só com o
# `data_dir` trocado em $RES, que o container enxerga como /res.
CONFIG="$RES/interacao_exemplo_g1.pgx.yaml"
sed -E "s#^([[:space:]]*data_dir:[[:space:]]*).*#\1'/work/unifolm-wma/examples/world_model_interaction_prompts'#" \
    "$RAIZ/lerobot-ext/config/wma/interacao_exemplo_g1.yaml" > "$CONFIG"
grep -q "/work/unifolm-wma/examples/world_model_interaction_prompts" "$CONFIG" \
    || { echo "❌ não achei o data_dir para trocar em $CONFIG"; exit 1; }

echo "== $DATASET | checkpoint $(basename "$CKPT") | vídeos em $RES/$DATASET =="
exec docker run --rm --gpus all --ipc=host \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    --user "$(id -u):$(id -g)" -e HOME=/tmp \
    -v "$RAIZ":/work -v "$HF":/hf -v "$RES":/res \
    -e HF_HOME=/hf -e OMP_NUM_THREADS=1 \
    -e PYTHONPATH=/work/unifolm-wma/src \
    -w /work/unifolm-wma \
    unifolm-wma:pgx \
    python -u scripts/evaluation/world_model_interaction.py \
        --seed 123 \
        --ckpt_path "/hf/${CKPT#"$HF"/}" \
        --config "/res/$(basename "$CONFIG")" \
        --savedir "/res/$DATASET" \
        --bs 1 --height 320 --width 512 \
        --unconditional_guidance_scale 1.0 \
        --ddim_steps 50 --ddim_eta 1.0 \
        --prompt_dir /work/unifolm-wma/examples/world_model_interaction_prompts \
        --dataset "$DATASET" \
        --video_length 16 --frame_stride 6 \
        --n_action_steps 16 --exe_steps 16 --n_iter 11 \
        --timestep_spacing 'uniform_trailing' \
        --guidance_rescale 0.7 --perframe_ae
