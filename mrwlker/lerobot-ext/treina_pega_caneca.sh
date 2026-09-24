#!/usr/bin/env bash
# Afina o nosso π0.5 nos 97 episódios de ROBÔ REAL de "Pick up the white cup".
#
#   bash treina_pega_caneca.sh              # a corrida de verdade (15.000 passos)
#   TESTE=1 bash treina_pega_caneca.sh      # corrida curta de ~40 min, para ver a curva andar
#   PASSOS=10 bash treina_pega_caneca.sh    # teste de fumaça: só conferir que carrega e roda
#
# Irmão do `maquinas/athena/launch_pi05.sh`, com três diferenças que valem para ESTA máquina:
#  · a athena tem várias GPUs e escolhe a mais livre; a GB10 tem uma só, então nada de
#    CUDA_VISIBLE_DEVICES;
#  · os caminhos de `/data/$USER` daquele script não existem aqui — tudo mora em $HOME;
#  · a poda de disco nao precisa de codigo: `save_freq` igual a `steps` faz a gravacao
#    periodica acontecer so no fim, e o `best_val_checkpoint` e sempre a mesma pasta.
set -euo pipefail

CONFIG="${CONFIG:-config/train/pi05_pega_caneca_real.yaml}"
ENV="${ENV:-$HOME/miniforge3/envs/prometheus-vla}"
PY="$ENV/bin/python"

cd "$(dirname "$(readlink -f "$0")")"

# O tokenizer do π0.5 é o do `google/paligemma-3b-pt-224`, repo fechado. Aqui ele JÁ está em
# `~/.cache/huggingface/hub/models--google--paligemma-3b-pt-224`, então a corrida sobe sem
# token; o `~/.hf_token` só é necessário numa máquina limpa.
[ -f ~/.hf_token ] && { HF_TOKEN=$(tr -d "[:space:]" < ~/.hf_token); export HF_TOKEN HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"; }

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SAIDA=$("$PY" -c "import yaml;print(yaml.safe_load(open('$CONFIG'))['output_dir'])")
# NAO criar "$SAIDA": o `cfg.validate()` do LeRobot ABORTA se a pasta de saida ja existe e
# `resume` e falso, para nao sobrescrever uma corrida anterior sem querer. Só o pai.
mkdir -p "$(dirname "$SAIDA")"

# Os modos curtos escrevem em OUTRA pasta, senao queimariam o nome da corrida de verdade
# (o `cfg.validate()` recusa uma saida que ja existe).
EXTRA=()
if [ -n "${PASSOS:-}" ]; then
    SAIDA="${SAIDA}_fumaca"
    rm -rf "$SAIDA"
    EXTRA+=(--steps="$PASSOS" --save_freq="$PASSOS" --eval_freq="$PASSOS" --output_dir="$SAIDA")
elif [ -n "${TESTE:-}" ]; then
    # CORRIDA DE 40 MINUTOS. A conta vem dos tempos MEDIDOS no teste de fumaca de 18/09:
    #   700 passos x 2,3 s        = 26,8 min
    #   2 validacoes x 1 min 51 s =  3,7 min
    #   gravar best + last        =  6   min
    #   carregar modelo e dados   =  3,5 min
    #                               ~40 min
    # `scheduler_decay_steps` entra explicito porque, sem ele, o scheduler AUTO-ESCALA quando
    # `steps` < `decay_steps` e o warmup vira 23 passos sem ninguem pedir — foi o que apareceu
    # no log do teste de fumaca. Melhor um cronograma declarado do que um adivinhado.
    SAIDA="${SAIDA}_teste40"
    rm -rf "$SAIDA"
    EXTRA+=(--steps=700 --save_freq=700 --eval_freq=350 --output_dir="$SAIDA"
            --policy.scheduler_warmup_steps=100 --policy.scheduler_decay_steps=700)
fi

echo "== π0.5 → pega a caneca (robô real) | $CONFIG | saída $SAIDA | $(date '+%F %T') =="
exec "$PY" -u -m policies.pi0_depth.run_train --config_path="$CONFIG" "${EXTRA[@]}" "$@" 2>&1 | tee -a "$SAIDA/treino.log"
