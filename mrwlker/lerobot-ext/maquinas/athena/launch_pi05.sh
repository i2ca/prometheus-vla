#!/usr/bin/env bash
# TREINO π0.5 na athena.
#
#   uso: bash launch_pi05.sh [GPU] [args extras do run_train]
#
#   CONFIG=config/train/pi05_white_cup_nodepth.yaml bash maquinas/athena/launch_pi05.sh 2
#
# Escrito depois de o π0.5 falhar quatro vezes seguidas em 02/09, cada uma por
# um motivo diferente. Cada bloco abaixo é uma dessas.
set -euo pipefail

# ── 1. O tipo `pi05depth` não existe até o módulo ser importado ─────────────
# O `@PreTrainedConfig.register_subclass("pi05depth")` mora em
# `policies/pi0_depth/configuration_pi05.py` e só roda quando alguém importa
# aquele arquivo. Chamando `train.run_train` (o genérico), o draccus recusa o
# YAML com
#
#   Couldn't find a choice class for 'pi05depth' in PreTrainedConfig
#
# O trainer certo é `policies.pi0_depth.run_train`, que importa e registra. Ele
# também é o que entende `val_dataset` + `eval_freq` + `save_best_checkpoint`,
# que é o mecanismo de validação que os configs de π0.5 daqui usam — e NÃO o
# `eval_split`/`eval_steps` das corridas do FastWAM-D. São dois mecanismos
# diferentes no mesmo repo; trocar um pelo outro dá um treino sem validação
# nenhuma, em silêncio.
MODULO="policies.pi0_depth.run_train"

CONFIG="${CONFIG:-config/train/pi05_white_cup_nodepth.yaml}"
ENVDIR="${ENVDIR:-$HOME/miniconda3/envs/prometheus-vla}"
PY="$ENVDIR/bin/python"

if [[ "${1:-}" =~ ^[0-9]+$ ]]; then GPU="$1"; shift; else
    GPU=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
          | sort -t, -k2 -n -r | head -1 | cut -d, -f1 | tr -d " ")
fi

# ── 2. O tokenizer do PaliGemma é repo FECHADO ─────────────────────────────
# O π0.5 puxa `google/paligemma-3b-pt-224` só pelo tokenizer (poucos MB), e o
# repo é gated: sem token o download morre com
#
#   OSError: You are trying to access a gated repo. ... 401 Client Error
#
# depois de já ter montado meio treino. O token vem de `~/.hf_token`, no mesmo
# molde do `~/.wandb_key`: arquivo com o token e nada mais. NÃO use
# `huggingface-cli login` — ele grava em `~/.cache/huggingface/token`, e o
# `HF_HOME` que este script exporta muda onde a biblioteca procura, então o
# login "some".
CHAVE_HF=~/.hf_token
if [ -f "$CHAVE_HF" ]; then
    HF_TOKEN=$(tr -d "[:space:]" < "$CHAVE_HF"); export HF_TOKEN
    export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"     # nome antigo, ainda lido por libs velhas
else
    echo "❌ Falta o token do HuggingFace em $CHAVE_HF."
    echo
    echo "   O π0.5 usa o tokenizer de google/paligemma-3b-pt-224, que é repo fechado."
    echo "   1. aceite a licença em https://huggingface.co/google/paligemma-3b-pt-224"
    echo "   2. crie um token de LEITURA em https://huggingface.co/settings/tokens"
    echo "   3. na athena:  echo 'hf_xxx' > ~/.hf_token && chmod 600 ~/.hf_token"
    exit 1
fi

CHAVE_WANDB=~/.wandb_key
if [ -f "$CHAVE_WANDB" ]; then
    WANDB_API_KEY=$(tr -d "[:space:]" < "$CHAVE_WANDB"); export WANDB_API_KEY
else
    echo "AVISO: $CHAVE_WANDB não existe — sem wandb (o log em disco tem os mesmos números)." >&2
    [[ " $* " == *" --wandb.enable=false "* ]] || set -- "$@" --wandb.enable=false
fi

cd ~/DEV/prometheus-vla/mrwlker/lerobot-ext

# ── 3. libstdc++ do conda ANTES de tudo ────────────────────────────────────
# O numpy exige GLIBCXX_3.4.29 e o libstdc++ do sistema não tem. Sem isto o
# `import numpy` estoura antes do treino começar. `conda activate` não resolve:
# a activate.d não mexe no LD_LIBRARY_PATH.
export LD_LIBRARY_PATH="$ENVDIR/lib:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="$ENVDIR/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"

# ── 4. Os dois caches do HuggingFace, separados ────────────────────────────
# `HF_HUB_CACHE` são os pesos já baixados, do hercules, só de leitura;
# `HF_HOME` é onde a biblioteca ESCREVE (locks, metadados, token), e tem que
# ficar no home de quem roda. Apontar os dois para o /data quebra com qualquer
# login que não seja o hercules, e o erro só aparece depois de carregar GB de
# pesos.
#
# E aqui o π0.5 diverge do FastWAM-D: o cache compartilhado do /data serve para
# LER pesos que já existem (Wan2.2, umt5-xxl), e o PaliGemma NÃO está lá — ele
# precisa ser BAIXADO. Apontar `HF_HUB_CACHE` para a pasta do hercules faz o
# download tentar escrever nela e morrer com
#
#   OSError: PermissionError at /data/.cache/huggingface/hub/models--google--paligemma-3b-pt-224
#
# depois de o token já ter funcionado — o que faz parecer problema de
# credencial e não de permissão. Por isso o cache aqui é do próprio usuário, e
# no /data porque o disco de sistema da athena vive perto de 100% e o PaliGemma
# passa de 5 GB.
export HF_HUB_CACHE="${HF_HUB_CACHE:-/data/$USER/.cache/huggingface/hub}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

# ── TMPDIR fora do disco de sistema ────────────────────────────────────────
# ISTO MATOU A CORRIDA DE 02/09, no step em que o dataloader subiu:
#
#   OSError: [Errno 28] No space left on device: '/tmp/pymp-u98f34ea'
#
# O `pymp-` é diretório temporário de uma lib de thread pool OpenMP. Ela não
# olha `HF_HOME` nem `HF_HUB_CACHE` — usa o `tempfile` do Python, que segue
# `$TMPDIR` e cai em `/tmp` quando ele não existe. O `/` da athena vive perto
# de 100% (6,4 TB de 6,9 TB), então QUALQUER criação de arquivo lá falha,
# mesmo de um diretório de poucos bytes. O `/data` tem ~10 TB livres.
export TMPDIR="${TMPDIR:-/data/$USER/tmp}"
mkdir -p "$TMPDIR"
mkdir -p "$HF_HUB_CACHE"

# O wandb também escreve no /data e não no home: os `run-*` dele guardam cópia
# de tudo que sobe, e o disco de sistema da athena não tem folga. Com
# `disable_artifact: true` no YAML o volume já é pequeno, mas o diretório sair
# do home é a garantia de que um esquecimento no config não derruba o treino.
export WANDB_DIR="${WANDB_DIR:-/data/$USER/wandb}"
mkdir -p "$WANDB_DIR"

export CUDA_VISIBLE_DEVICES="$GPU"
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SAIDA=$("$PY" -c "import yaml,sys;print(yaml.safe_load(open('$CONFIG'))['output_dir'])")
mkdir -p "$(dirname "$SAIDA")"
echo "== GPU $GPU | $CONFIG | saída $SAIDA | $(date '+%F %T') =="

exec "$PY" -m "$MODULO" --config_path="$CONFIG" "$@" 2>&1 | tee -a "$SAIDA.log"
