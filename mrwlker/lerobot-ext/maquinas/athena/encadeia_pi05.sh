#!/usr/bin/env bash
# Espera o treino da UnifoLM-WLA acabar e sobe o π0.5 no lugar, sozinho.
#
#   screen -dmS corrente bash maquinas/athena/encadeia_pi05.sh
#   screen -r corrente            # para ver em que pé está
#
# Escrito para a GPU não ficar parada de madrugada entre um treino e outro. Ele
# NÃO mata a WLA: fica olhando a sessão `screen` dela e só age quando ela sai.
#
# ── O π0.5 recomeça do zero, e isso não é escolha ───────────────────────────
# A corrida de 21/09 foi cancelada no passo ~4.000 e NÃO deixou checkpoint: o
# `pi05_pega_copo_sim.yaml` tem `save_freq: 40000`, igual a `steps`, então ele
# só grava no fim. O diretório de saída tem `wandb/` e mais nada. Não há de onde
# retomar; o ponto de partida continua sendo o `pretrained_path` do YAML, o
# checkpoint de 60 mil passos do `pi05_cotreino_completo`.
#
#     40.000 passos x 0,86 s (medido nesta A100) = 9,6 h
#     + 8 validações de 12 episódios             ~ 10,5 h
#
# ── GPU 1, e não a 2 ────────────────────────────────────────────────────────
# O `rodar_pi05.sh` antigo passava `2`. Hoje a GPU-2 tem 79 GB ocupados por um
# job de terceiro e a GPU-1 é a reservada para nós — é ela que a WLA libera.
# Passe outro número como primeiro argumento se isso mudar.
set -euo pipefail

GPU="${1:-1}"
SESSAO_WLA="${SESSAO_WLA:-wla}"
CONFIG="${CONFIG:-config/train/pi05_pega_copo_sim.yaml}"
LOG_WLA="${LOG_WLA:-/data/mrwlker/wla_treino.log}"
REPO="$HOME/DEV/prometheus-vla/lerobot-ext"
LOG="/data/mrwlker/pi05_pega_copo_sim.log"

echo "[$(date '+%F %T')] esperando a sessão '$SESSAO_WLA' terminar…"
while screen -ls 2>/dev/null | grep -q "[.]$SESSAO_WLA[[:space:]]"; do
    sleep 300
done

# Por que olhar o log e não só a ausência da sessão: a sessão some tanto quando
# o treino acaba quanto quando ele morre. As duas liberam a GPU e as duas devem
# deixar o π0.5 subir, mas quem ler este log amanhã precisa saber qual foi.
if grep -q "steps.*20000" "$LOG_WLA" 2>/dev/null \
   || tr '\r' '\n' < "$LOG_WLA" 2>/dev/null | grep -q "20000/20000"; then
    echo "[$(date '+%F %T')] a WLA chegou ao fim."
else
    echo "[$(date '+%F %T')] ATENÇÃO: a sessão da WLA sumiu sem chegar a 20000."
    echo "               últimas linhas do log dela:"
    tr '\r' '\n' < "$LOG_WLA" 2>/dev/null | tail -5 | sed 's/^/               /'
fi
ls -la /data/mrwlker/wla_treino/wla_prometheus_copo/checkpoints/ || true

echo "[$(date '+%F %T')] subindo o π0.5 na GPU $GPU com $CONFIG"
export LD_PRELOAD="$HOME/miniconda3/envs/prometheus-vla/lib/libstdc++.so.6"
export LD_LIBRARY_PATH="$HOME/miniconda3/envs/prometheus-vla/lib:${LD_LIBRARY_PATH:-}"
export TMPDIR=/data/mrwlker/tmp
mkdir -p "$TMPDIR"
cd "$REPO"

# O lançador mudou de lugar na reorganização de 21/09: era `athena/`, hoje é
# `maquinas/athena/`. O `/data/mrwlker/rodar_pi05.sh` ainda aponta para o
# caminho velho e quebra depois de um `git pull` — use este script no lugar.
screen -dmS pi05 bash -c "CONFIG='$CONFIG' bash maquinas/athena/launch_pi05.sh $GPU 2>&1 | tee -a '$LOG'"
sleep 20
screen -ls | sed 's/^/   /'
echo "[$(date '+%F %T')] π0.5 no ar; log em $LOG"
