#!/usr/bin/env bash
# Grounding AO VIVO da câmera do robô, sem controlar o robô.
#
#   uso: bash launch_grounding_vivo.sh [GPU] [args extras]
#
#   bash launch_grounding_vivo.sh 2
#   bash launch_grounding_vivo.sh 2 --task="pick up the black kettle"
#
# PRÉ-REQUISITO, no robô (10.9.8.73) e à mão:
#
#   python Scripts_Prometheus_int/full_realsenser_server.py       # cabeça, :5555
#   python Scripts_Prometheus_int/right_arm_realsense_server.py   # pulso,  :5556
#
# O bridge (`dex3_g1_server_v2.py`) NÃO é necessário: este modo não manda ação
# nenhuma. Sem ele não há propriocepção, e o script usa uma pose fixa — ver a
# nota no cabeçalho do `grounding_ao_vivo.py`.
set -euo pipefail

PY="$HOME/miniconda3/envs/prometheus-vla/bin/python"
[ -x "$PY" ] || { echo "ERRO: $PY não existe. Rode o ./install.sh." >&2; exit 1; }

# ── LD_PRELOAD do libstdc++ do conda ─────────────────────────────────────
# Este script carrega zmq E torch no mesmo processo, que é exatamente a
# combinação que mata o servidor de inferência sem imprimir uma linha: o
# libstdc++ do sistema não tem `GLIBCXX_3.4.29`, que o numpy exige, e o libzmq
# carrega o do sistema primeiro. O import do `zmq` antes do `torch` no topo do
# .py resolve na maioria dos casos; isto é o cinto de segurança.
LIBSTDCPP="$(dirname "$PY")/../lib/libstdc++.so.6"
[ -f "$LIBSTDCPP" ] && export LD_PRELOAD="$LIBSTDCPP${LD_PRELOAD:+:$LD_PRELOAD}"

# Os DOIS caches do HuggingFace separados: leitura compartilhada dos 25 GB de
# pesos, escrita no home de quem rodou. Ver maquinas/athena/README.md.
export HF_HUB_CACHE="${HF_HUB_CACHE:-/data/.cache/huggingface/hub}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1

if [[ "${1:-}" =~ ^[0-9]+$ ]]; then GPU="$1"; shift; else
    GPU=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
          | sort -t, -k2 -n -r | head -1 | cut -d, -f1 | tr -d " ")
fi
export CUDA_VISIBLE_DEVICES="$GPU"
export OMP_NUM_THREADS=1

ROBO="${ROBO:-10.9.8.73}"
PORTA="${PORTA:-8089}"
CHECKPOINT="${CHECKPOINT:-/data/train_output/fastwamd_corrida2_step1000/pretrained_model}"

echo "== GPU $GPU ($(nvidia-smi -i "$GPU" --query-gpu=memory.free --format=csv,noheader) livres) | grounding ao vivo =="
echo "   robô $ROBO  →  painel em http://$(hostname -I | awk '{print $1}'):$PORTA/"

# As câmeras têm que estar no ar ANTES: sem isto o script sobe, carrega 6 B de
# pesos por dois minutos e só então descobre que não há quadro.
for p in 5555 5556; do
    timeout 2 bash -c "echo > /dev/tcp/$ROBO/$p" 2>/dev/null \
        && echo "   :$p ok" \
        || echo "   :$p FECHADA — suba o servidor de câmera no robô"
done

cd "$(dirname "${BASH_SOURCE[0]}")/.."
exec "$PY" grounding_ao_vivo.py \
    --checkpoint="$CHECKPOINT" --robo="$ROBO" --porta="$PORTA" "$@"
