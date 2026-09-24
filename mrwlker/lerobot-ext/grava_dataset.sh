#!/usr/bin/env bash
# GRAVAR dataset da simulação, em formato LeRobot.
#
#   bash grava_dataset.sh                       # 100 episódios, com janela
#   bash grava_dataset.sh 600                   # 600 episódios
#   bash grava_dataset.sh 600 --sem-janela      # sem janela (~3x mais rápido)
#   bash grava_dataset.sh 600 --tarefa=pega     # só a metade da pega
#   REPO=Mrwlker/meu_teste bash grava_dataset.sh 50
#
# `--tarefa` (pega | poe | completo) escolhe qual metade do roteiro é gravada,
# e ENTRA NO NOME padrão do dataset e do log — sem isso as duas metades
# gravariam no mesmo diretório e uma retomaria a outra. Para gravar as duas ao
# mesmo tempo use o `grava_duplo.sh`, que é este script duas vezes.
#
# Irmão do `testa_sim.sh`, que roda o mesmo especialista SEM escrever nada.
# Use aquele primeiro para julgar a trajetória; este só depois.
#
# ── Rodar em `screen`, não em background do terminal ────────────────────────
# Medido nesta máquina: `setsid`/`nohup`/`disown` NÃO seguraram o processo —
# ele morria junto com o shell que o lançou, no meio da gravação. Com `screen`
# sobrevive. Por isso o modo recomendado é:
#
#   screen -dmS grava bash grava_dataset.sh 600 --sem-janela
#   screen -r grava        # acompanhar (Ctrl+A depois D para sair sem matar)
#
# ── Para PARAR sem corromper ────────────────────────────────────────────────
#   pkill -TERM -f gerar_dataset_mujoco      # NUNCA -9
# O `-9` mata no meio da escrita do parquet, que guarda o índice num rodapé no
# fim do arquivo: interrompido ali, o pyarrow não lê NADA, nem as partes
# íntegras. Já custou 555 episódios uma vez. Com TERM o script desmonta pelo
# `finally` e o que foi salvo continua legível.
#
# O TERM pode demorar a pegar: ele cai com frequência dentro de um `solve` do
# casadi, que converte a interrupção numa exceção própria e a engole. Mande de
# novo, de preferência logo depois de uma linha `↳ salvo` — ali há ~25 s de
# janela até a próxima escrita de parquet.
set -euo pipefail

AQUI="$(dirname "$(readlink -f "$0")")"
ENV="${ENV:-$HOME/miniconda3/envs/prometheus-vla}"
PY="$ENV/bin/python"

export OMP_NUM_THREADS=1          # ipopt multithread leva a IK de 0,8 a 83 ms
export PYTHONUNBUFFERED=1         # sem isto o log sai de 4 KB em 4 KB
export __NV_PRIME_RENDER_OFFLOAD=1
export __GLX_VENDOR_LIBRARY_NAME=nvidia
export DISPLAY="${DISPLAY:-:0}"

cd "$AQUI"                        # a cena é procurada em `../unitree-g1-mujoco`

N="${1:-100}"
shift || true

TAREFA=completo
for a in "$@"; do case "$a" in --tarefa=*) TAREFA="${a#--tarefa=}";; esac; done

REPO="${REPO:-Mrwlker/sim_coador_${TAREFA}_$(date +%F)}"
LOG="${LOG:-$AQUI/gravacao_${TAREFA}_$(date +%F_%H%M).log}"

JANELA="--com-janela"
for a in "$@"; do [ "$a" = "--sem-janela" ] && JANELA=""; done
LIMPOS=()
for a in "$@"; do [ "$a" = "--sem-janela" ] || LIMPOS+=("$a"); done

echo "== gravando $N episódios | tarefa: $TAREFA | repo: $REPO | janela: ${JANELA:-não} =="
echo "   log: $LOG"
echo "   parar com: pkill -TERM -f gerar_dataset_mujoco   (nunca -9)"

# O script RETOMA sozinho se o diretório já existir (LeRobotDataset.resume),
# somando aos episódios que já estão lá em vez de recusar.
exec "$PY" -u gerar_dataset_mujoco.py \
    --episodios="$N" --repo-id="$REPO" $JANELA "${LIMPOS[@]+"${LIMPOS[@]}"}" \
    2>&1 | tee -a "$LOG"
