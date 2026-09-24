#!/usr/bin/env bash
# Lançamento do ACT-D na athena.
#
#   uso: bash launch_actd.sh [GPU] [args extras para o run_train]
#
# Irmão do `launch_fastwamd.sh`, com as mesmas armadilhas de ambiente. A
# diferença de fundo é o tamanho: o ACT-D tem ~50 M de parâmetros contra os
# 6 B do FastWAM-D, então cabe em qualquer GPU e roda ao lado de outro treino.
set -euo pipefail

# ── wandb ────────────────────────────────────────────────────────────────
# A conta `hercules` tem um `~/.netrc` com a chave de OUTRA pessoa. Sem esta
# guarda o wandb o usa em silêncio e a corrida sobe na conta errada.
CHAVE=~/.wandb_key
if [ -f "$CHAVE" ]; then
    WANDB_API_KEY=$(tr -d "[:space:]" < "$CHAVE")
    export WANDB_API_KEY
else
    echo "ERRO: $CHAVE não existe." >&2
    echo "Sem ele o wandb usaria a chave do ~/.netrc, que é de outra conta." >&2
    [[ " $* " == *" --wandb.enable=false "* ]] || exit 1
fi

if [[ "${1:-}" =~ ^[0-9]+$ ]]; then
    GPU="$1"; shift
else
    GPU=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
          | sort -t, -k2 -n -r | head -1 | cut -d, -f1 | tr -d " ")
fi
LIVRE=$(nvidia-smi -i "$GPU" --query-gpu=memory.free --format=csv,noheader)
echo "== GPU $GPU ($LIVRE livres) | ACT-D | $(date '+%F %T') =="

cd ~/DEV/prometheus-vla/lerobot-ext
# O disco de sistema tem 21 GB livres de 6,9 TB: nada de cache nem de saída de
# treino pode cair nele. O `output_dir` do YAML é relativo (`train_output/...`,
# escrito para o PC do Miguel) e é sobrescrito abaixo pelo mesmo motivo.
# Os caminhos abaixo eram FIXOS no hercules e o script inteiro só funcionava
# com aquele login: `HF_HOME` e `TORCH_HOME` em `/data/.cache/*` (que é do
# hercules e outros logins não escrevem), o python em `/home/hercules/...`, e o
# `output_dir`/log em `/data/train_output` (idem). Agora tudo tem default igual
# ao de antes e aceita ser sobrescrito — o hercules não vê diferença.
export HF_HOME="${HF_HOME:-/data/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-/data/.cache/torch}"   # pesos ImageNet da ResNet18

# TMPDIR fora do disco de sistema: o `/` da athena vive perto de 100% e o
# `tempfile` do Python cai em `/tmp` quando `$TMPDIR` não existe. Foi assim que
# a corrida do pi05 morreu em 02/09 (`Errno 28` num `/tmp/pymp-*`).
export TMPDIR="${TMPDIR:-/data/$USER/tmp}"
mkdir -p "$TMPDIR"
export CUDA_VISIBLE_DEVICES="$GPU"
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# `--num_workers=8`: knob de máquina, não de experimento — o 4 do YAML é do
# notebook. Cada amostra decodifica RGB e profundidade em 848×480, e a athena
# tem núcleo de sobra. Nada mais do YAML é tocado aqui de propósito: o
# `batch_size: 8` fica como está para a corrida ser a que o arquivo descreve.
PY="${PY:-/home/hercules/miniconda3/envs/prometheus-vla/bin/python}"
CONFIG="${CONFIG:-config/train/actdepth_white_cup_on_dripper.yaml}"
SAIDA="${SAIDA:-/data/train_output/actdepth_white_cup_on_dripper}"
NOME="${NOME:-actdepth_white_cup_on_dripper}"
LOG="${LOG:-/data/train_output/actdepth.log}"
mkdir -p "$(dirname "$SAIDA")"

exec "$PY" -m policies.act_depth.run_train \
    --config_path="$CONFIG" \
    --output_dir="$SAIDA" \
    --job_name="$NOME" \
    --num_workers=8 \
    --wandb.enable=true "$@" \
    2>&1 | tee -a "$LOG"
