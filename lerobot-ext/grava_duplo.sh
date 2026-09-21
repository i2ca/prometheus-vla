#!/usr/bin/env bash
# GRAVA AS DUAS METADES AO MESMO TEMPO, cada uma na sua `screen`.
#
#   bash grava_duplo.sh                    # 1200 episódios de cada, com janela
#   bash grava_duplo.sh 300                # 300 de cada
#   bash grava_duplo.sh 300 --sem-janela   # sem janela (~3x mais rápido)
#
# Sobe dois processos independentes:
#
#   screen `pega`  → dataset "pick up the white cup"          (partida→levantar)
#   screen `poe`   → dataset "place the cup on the dripper"   (recolher→voltar)
#
# ── Por que dá para rodar junto ────────────────────────────────────────────
# Cada processo tem seu próprio MuJoCo, seus renderizadores offscreen e seu
# diretório de dataset. Não há arquivo compartilhado, nem porta, nem lock: o
# que os dois disputam é GPU e CPU. Com `OMP_NUM_THREADS=1` (que já vem do
# `grava_dataset.sh`, e que é o que mantém a IK em 0,8 ms em vez de 83) cada um
# ocupa essencialmente um núcleo, então dois cabem folgados.
#
# SEMENTES DIFERENTES, de propósito. É a semente que sorteia onde a xícara e o
# coador nascem; com a mesma nos dois, os dois datasets veriam exatamente a
# mesma sequência de cenas, e a variação entre eles seria zero.
#
# ── Acompanhar ─────────────────────────────────────────────────────────────
#   screen -r pega     |  screen -r poe      (Ctrl+A depois D sai sem matar)
#   screen -ls                               (ver as duas)
#   tail -f gravacao_pega_*.log
#
# ── Parar ──────────────────────────────────────────────────────────────────
#   pkill -TERM -f gerar_dataset_mujoco      # os DOIS, e NUNCA com -9
# O `-9` corta a escrita do parquet no meio e leva junto tudo que já estava
# gravado — ver o cabeçalho do `grava_dataset.sh`.
set -euo pipefail

AQUI="$(dirname "$(readlink -f "$0")")"
cd "$AQUI"

N="${1:-1200}"
shift || true

# Sorteadas aqui, e ECOADAS: sem isso a corrida não é reproduzível.
S_PEGA="${S_PEGA:-$(( RANDOM * 32768 + RANDOM ))}"
S_POE="${S_POE:-$(( RANDOM * 32768 + RANDOM ))}"

for t in pega poe; do
    if screen -ls | grep -q "\.$t[[:space:]]"; then
        echo "❌ já existe uma screen chamada '$t' — 'screen -r $t' para ver," \
             "ou mate antes com 'pkill -TERM -f gerar_dataset_mujoco'"
        exit 1
    fi
done

screen -dmS pega bash grava_dataset.sh "$N" --tarefa=pega --semente="$S_PEGA" "$@"
screen -dmS poe  bash grava_dataset.sh "$N" --tarefa=poe  --semente="$S_POE"  "$@"

cat <<FIM

== duas gravações de $N episódios no ar ==
   pega  semente $S_PEGA   → meu_dataset/sim_coador_pega_$(date +%F)
   poe   semente $S_POE    → meu_dataset/sim_coador_poe_$(date +%F)

   screen -r pega      acompanhar a pega   (Ctrl+A, D para sair sem matar)
   screen -r poe       acompanhar o pouso
   screen -ls          ver as duas
   tail -f $AQUI/gravacao_pega_*.log

   parar as duas:  pkill -TERM -f gerar_dataset_mujoco    (nunca -9)
FIM
