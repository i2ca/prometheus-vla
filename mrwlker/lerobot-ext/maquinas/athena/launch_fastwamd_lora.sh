#!/usr/bin/env bash
# FastWAM-D com LoRA no action expert.
#
#   uso: bash launch_fastwamd_lora.sh [GPU] [args extras]
#
# O treino cheio destrava 1,02 B de parâmetros para 10.408 quadros e decora o
# dataset em ~1,2 época (eval mínimo no step ~1000 nas três corridas). Aqui só
# os adaptadores treinam: ~16 M, uns 1,6% do que era.
set -euo pipefail

CHAVE=~/.wandb_key
if [ -f "$CHAVE" ]; then
    WANDB_API_KEY=$(tr -d "[:space:]" < "$CHAVE"); export WANDB_API_KEY
else
    echo "AVISO: $CHAVE não existe — sem wandb (o log em disco tem os mesmos números)." >&2
    [[ " $* " == *" --wandb.enable=false "* ]] || set -- "$@" --wandb.enable=false
fi

if [[ "${1:-}" =~ ^[0-9]+$ ]]; then GPU="$1"; shift; else
    GPU=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
          | sort -t, -k2 -n -r | head -1 | cut -d, -f1 | tr -d " ")
fi
echo "== GPU $GPU ($(nvidia-smi -i "$GPU" --query-gpu=memory.free --format=csv,noheader) livres) | LoRA | $(date '+%F %T') =="

cd ~/DEV/prometheus-vla/mrwlker/lerobot-ext
# Os DOIS caches do HuggingFace, separados de propósito. `HF_HUB_CACHE` são os
# 25 GB de pesos já baixados (Wan2.2, umt5-xxl, fastwam_base), compartilhados e
# só de leitura; `HF_HOME` é onde a biblioteca ESCREVE — locks do `datasets`,
# metadados. Apontar os dois para o `/data` funciona só para o `hercules`, que é
# o dono da pasta; com outro login o treino morre com
#
#   PermissionError: [Errno 13] Permission denied:
#   '/data/.cache/huggingface/datasets/..._0.0.0_....lock'
#
# e o erro só aparece DEPOIS de carregar 6 B de pesos. Ver maquinas/athena/README.md.
export HF_HUB_CACHE="${HF_HUB_CACHE:-/data/.cache/huggingface/hub}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

# ── TMPDIR também precisa sair do disco de sistema ─────────────────────────
# O treino do π0.5 de 02/09 morreu com
#
#   OSError: [Errno 28] No space left on device: '/tmp/pymp-u98f34ea'
#
# O `pymp-` é de uma lib de thread pool OpenMP, não do HuggingFace — ignora
# `HF_HOME`/`HF_HUB_CACHE` e usa o `tempfile` padrão do Python, que olha
# `$TMPDIR` e cai em `/tmp` se não existir. `/` estava em 99% (6,4 TB usados,
# 125 GB livres) na hora da morte; redirecionar os DOIS caches do HF não ajuda
# nada aqui porque o problema nem é HF. `/data` tem ~10 TB livres.
export TMPDIR="${TMPDIR:-/data/$USER/tmp}"
mkdir -p "$TMPDIR"

export CUDA_VISIBLE_DEVICES="$GPU"
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── Por que NÃO existe `--policy.use_peft=true` aqui ─────────────────────
# Essa flag é o caminho de CARREGAR um adaptador pronto: o `make_policy`
# (`policies/factory.py:340`) lê um `adapter_config.json` do `pretrained_path`.
# Para COMEÇAR um LoRA, quem faz o trabalho é o `lerobot_train.py:331` — basta
# `cfg.peft` existir, e ele chama `policy.wrap_with_peft()` depois de montar a
# política já com os pesos do `fastwam_base` (que o `__post_init__` do config
# preenche sozinho, justamente porque `use_peft` continua falso).
#
# ── O caminho dos blocos é `mot.layers.N.blocks.action`, não o expert ────
# Duas mudanças de dono acontecem no `__init__` e as duas enganam quem escreve
# o alvo do LoRA olhando só o construtor:
#
#   1. `modular.py:865` guarda os experts com `object.__setattr__`, então
#      `action_expert.*` NÃO existe em `named_modules()`. O nome canônico
#      passa por `mot.mixtures.{video,action}.*`.
#   2. `modular.py:670` faz `del expert._modules["blocks"]` e RE-PARENTA cada
#      bloco para dentro de `mot.layers.{i}.blocks.{video,action}` — o MoTLayer
#      é a unidade de wrap do FSDP, e deixar os blocos registrados nos dois
#      lugares faria o FSDP gerenciar os mesmos tensores duas vezes.
#
# Resultado: `mixtures.action.blocks.*` está morto depois do `__init__` (foi o
# alvo da primeira tentativa, e o PEFT abortou com "No modules were targeted").
# Já o `action_encoder`, o `head` e o `patch_embedding` NÃO são blocos e
# continuam pendurados nos experts — por isso os dois estilos de nome abaixo.
#
# E os sufixos do `modules_to_save` são qualificados de propósito: o PEFT casa
# por SUFIXO, e tanto o expert de vídeo quanto o de ação têm um `head` — um
# alvo escrito só como "head" descongelaria o expert de vídeo que o
# `freeze_video_expert` acabou de congelar.
#
# ── Por que o `patch_embedding` está em full_training_modules ────────────
# No modo `latent` os canais de profundidade nascem ZERADOS e precisam
# aprender. O `wrap_with_peft` faz `requires_grad_(False)` em TUDO antes de
# pendurar os adaptadores, o que desfaria o destrave feito em
# `_amplia_patch_embedding`. Sem esta linha o treino roda com a profundidade
# permanentemente em zero — ou seja, um `off` caro, e em silêncio.
# ── Configuráveis por variável de ambiente ──────────────────────────────
# O python vem de $HOME: este launcher roda com mais de um login na mesma
# máquina (hercules e mrwlker), cada um com o seu miniconda. E o
# `/data/train_output` é do hercules, com grupo `hercules` — outros logins
# não escrevem lá, então a saída padrão fica ao lado, no `/data`, que é 777.
# Os módulos destravados por inteiro (além dos adaptadores). O
# `mixtures.video.patch_embedding` só faz sentido com `depth_mode=latent`, onde
# os canais de profundidade nascem zerados e precisam aprender. Com
# `depth_mode=off` não existe canal novo, e deixá-lo aqui destrava o patch
# embedding do expert de vídeo CONGELADO sem nenhum motivo — mexendo justamente
# na entrada do pré-treino que se quer preservar. Por isso é variável:
#
#   MODULOS='["mixtures.action.action_encoder","mixtures.action.head","proprio_encoder"]' \
#   bash launch_fastwamd_lora.sh 2 --policy.depth_mode=off
MODULOS="${MODULOS:-[\"mixtures.action.action_encoder\",\"mixtures.action.head\",\"proprio_encoder\",\"mixtures.video.patch_embedding\"]}"

PY="${PY:-$HOME/miniconda3/envs/prometheus-vla/bin/python}"
CONFIG="${CONFIG:-config/train/fastwamdepth_white_cup_on_dripper.yaml}"
SAIDA="${SAIDA:-/data/train_output_$USER/fastwamd_lora}"
STEPS="${STEPS:-2500}"
NOME="${NOME:-fastwamd_lora}"
mkdir -p "$(dirname "$SAIDA")"

exec "$PY" -m policies.fastwam_depth.run_train \
    --config_path="$CONFIG" \
    --output_dir="$SAIDA" \
    --job_name="$NOME" \
    --steps="$STEPS" \
    --peft.r=16 \
    --peft.lora_alpha=32 \
    --peft.target_modules='.*mot\.layers\.\d+\.blocks\.action\.(self_attn|cross_attn)\.(q|k|v|o)' \
    --peft.full_training_modules="$MODULOS" \
    "$@" \
    2>&1 | tee -a "$SAIDA.log"
