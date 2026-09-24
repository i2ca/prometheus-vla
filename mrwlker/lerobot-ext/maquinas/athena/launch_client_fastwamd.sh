#!/usr/bin/env bash
# Loop de controle do FastWAM-D rodando NA ATHENA, com o robô real na LAN.
#
#   screen -dmS infer   bash /data/train_output/launch_server_fastwamd.sh 2
#   screen -dmS control bash /data/train_output/launch_client_fastwamd.sh
#
# Os dois no mesmo host: o servidor com o modelo de 6 B na GPU, e este, que só
# lê o robô, manda a observação pelo loopback e executa o chunk de volta.
#
#   uso: bash launch_client_fastwamd.sh [IP_DO_ROBO] [args extras do cliente...]
set -euo pipefail
ROBO="${1:-10.9.8.73}"
shift || true

ENV=$HOME/miniconda3/envs/prometheus-vla
LOG=/data/train_output/inferencia_client.log

# TODA a saída vai para o log, e não só a do python. Sob `screen -dmS`, um
# script que sai antes do exec leva a janela junto: `screen -r` responde
# "There is no screen to be resumed" e o motivo da recusa não fica em lugar
# nenhum. Com isto, o `tail` sempre conta a história.
exec > >(tee -a "$LOG") 2>&1
echo
echo "───── $(date '+%F %T') ─────"

# Mesma armadilha do servidor: o libzmq puxa o libstdc++ DO SISTEMA antes de
# tudo, e as extensões nativas que o `import torch` arrasta estouram em
# Segmentation fault sem mensagem nenhuma. Este cliente não carrega modelo, mas
# importa torch pela cadeia do `init_lerobot_inference_async_v2` — então paga o
# mesmo pedágio.
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="$ENV/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"

# Nenhuma GPU de propósito: o cliente não infere nada. Sem isto o torch reserva
# contexto CUDA numa placa à toa e passa a disputar com o servidor ao lado.
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1

# O cwd decide qual árvore de assets é lida; da raiz do repo o robô não sobe.
cd ~/DEV/prometheus-vla/mrwlker/lerobot-ext

# Pré-voo: o `connect` do ZMQ NÃO falha com o outro lado ausente. Sem esta
# checagem o cliente sobe bonito, conecta em sockets que ninguém atende e fica
# mudo esperando `lowstate` para sempre — e o operador fica olhando um painel
# vazio sem saber que o robô nunca entrou na conversa.
faltando=""
for p in 5555 5556 6000 6001 6002 6003 6004; do
    timeout 2 bash -c "echo > /dev/tcp/$ROBO/$p" 2>/dev/null || faltando="$faltando $p"
done
if [ -n "$faltando" ]; then
    echo "❌ O robô $ROBO não atende nas portas:$faltando"
    echo "   Suba no ROBÔ, nesta ordem:"
    echo "     python Scripts_Prometheus_int/dex3_g1_server_v2.py --loco"
    echo "     python Scripts_Prometheus_int/full_realsenser_server.py"
    echo "     python Scripts_Prometheus_int/right_arm_realsense_server.py"
    echo
    echo "   (janela mantida por 5 min para você poder ler com \`screen -r control\`;"
    echo "    o texto acima também está em $LOG)"
    sleep 300
    exit 1
fi

echo "== robo $ROBO | servidor 127.0.0.1:5600 | painel http://10.9.8.252:8088/ =="
# --lead=16, e não 24: a inferência leva ~1,06 s e o loop roda a 15 Hz, ou seja
# 16 ações são consumidas no tempo de uma inferência. Com lead=24 o pedido saía
# depois de 8 ações (0,53 s) e três chunks ficavam vivos ao mesmo tempo — o
# ensembling temporal então media três previsões feitas com 1 s de diferença,
# com pesos quase iguais (exp(-0,1·i) → 0,37/0,33/0,30). Sobre um modelo que já
# regride para a média, isso vira uma pose parada com tremor de ±0,03 rad.
#
# --pose-inicial: leva o robô à pose de prontidão das demonstrações ANTES de
# entregar o controle. Sem isso o estado que chega ao modelo está fora da
# distribuição desde o quadro 0, e ele trava. Ver a docstring da função no
# cliente.
exec "$ENV/bin/python" -u init_lerobot_inference_fastwamd_client.py \
    --server=127.0.0.1 \
    --robot-ip="$ROBO" \
    --cam-robot="$ROBO" \
    --pose-inicial=meu_dataset/white_cup_on_dripper_2026-08-11:25 \
    --chunk=32 --lead=16 --fps=15 --rampa=2 \
    --v-web=8088 --debug "$@"
