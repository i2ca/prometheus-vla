#!/bin/bash
# Avalia um checkpoint do ajuste fino em malha fechada: abre o túnel até o servidor, roda avalia.py e audita.py.
# Antes, na máquina com GPU, suba o servidor do checkpoint (ver README.md, seção "Servir e avaliar"):
#   CUDA_VISIBLE_DEVICES=<gpu> PSI0_REPO=<repo> PSI0_CKPT=<pasta da run> PSI0_PASSO=<passo> PSI0_SEM_CORPO=1 \
#   PSI0_INSTR='<instrução das demos>' python ../task3/servidor_task3.py <porta>
# Uso: HOST=<alias ssh da máquina com GPU> [PORTA=8779] [PY=python] bash avalia_ckpt.sh <saida> [args do avalia.py, ex. --n 20 --semente 100 --cintura]
set -e
D=${1:?pasta de saída}; shift
HOST=${HOST:?alias ssh da máquina com GPU}; PORTA=${PORTA:-8779}; PY=${PY:-python}
AQUI=$(dirname "$(readlink -f "$0")")
ss -ltn | grep -q ":$PORTA " || ssh -f -N -L "$PORTA:127.0.0.1:$PORTA" "$HOST"
MUJOCO_GL=egl "$PY" "$AQUI/avalia.py" "$D" --porta "$PORTA" "$@" 2>&1 | grep "^ep\|sucesso\|falhou"
"$PY" "$AQUI/audita.py" "$D" 2>/dev/null | grep "sucesso estrito\|^mira"
