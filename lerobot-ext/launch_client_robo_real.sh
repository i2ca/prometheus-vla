#!/usr/bin/env bash
# ROBÔ REAL. O modelo comanda o G1 de verdade.
#
#   uso: bash launch_client_robo_real.sh [args extras do cliente...]
#
# Roda NO SEU NOTEBOOK (medido em 02/09: daqui se alcança 10.9.8.73 nas sete
# portas). Para rodar da athena, use `maquinas/athena/launch_client_fastwamd.sh`, que é o
# mesmo caminho com os LD_PRELOAD de lá.
#
# ── O QUE ESPERAR, medido antes de escrever isto ────────────────────────────
# A primeira ação depois da pose de partida é BOA: o modelo reconhece a pose de
# prontidão e responde com uma pose coerente. O que acontece depois é o
# problema: cada ação leva o corpo para um estado um pouco fora do que foi
# demonstrado, o estado seguinte fica mais fora, e o modelo — que decorou 21
# episódios em 1,5 época — não tem comportamento de recuperação. Ele converge
# para a resposta média e o braço para.
#
# Então o desfecho provável é BRAÇO QUE PARA, não braço que se debate. Mas isso
# é uma previsão, não uma garantia: fique com a mão no botão de emergência.
#
# ── O QUE ESTE SCRIPT NÃO FAZ ───────────────────────────────────────────────
# Não sobe nada NO ROBÔ. O bridge (`dex3_g1_server_v2.py --loco`) e os dois
# servidores de câmera já têm que estar rodando lá, subidos à mão. Um segundo
# publicador em cima do `lowcmd` é a maneira de perder o controle do robô.
set -euo pipefail

ATHENA="${ATHENA:-10.9.8.252}"
ROBO="${ROBO:-10.9.8.73}"
ENV="${ENV:-$HOME/miniconda3/envs/prometheus-vla}"

# ── OS DOIS NÚMEROS QUE VOCÊ VAI QUERER MEXER ───────────────────────────────
#
# DELTA (`--inconsistency`): salto máximo, em radianos, entre duas ações
# consecutivas. LEIA O QUE ELE FAZ antes de confiar nele: passando do limite, a
# ação é MISTURADA 50/50 com a anterior — ele suaviza, não corta. Um salto de
# 2 rad com DELTA=0,1 vira 1 rad, não 0,1. Para cortar de verdade seria preciso
# um clamp, que não existe hoje.
# A 15 Hz: 0,10 rad/passo ≈ 1,5 rad/s. O padrão do cliente é 10.0, que na
# prática é "desligado".
DELTA="${DELTA:-0.10}"

# FPS: o loop de controle. 15 Hz é o que o `maquinas/athena/launch_client_fastwamd.sh`
# usa e o que casa com `--lead=16` (a inferência leva ~1,0 s, então 16 ações são
# consumidas no tempo de uma).
FPS="${FPS:-15}"

# Episódio de onde sai a pose de prontidão. NÃO é opcional no robô real: sem
# ela o estado é OOD desde o quadro 0 e o modelo trava. Medido: estado zerado dá
# 566 de 928 valores fora da faixa; a pose do ep24 dá 2 de 928, na mesma imagem.
POSE="${POSE:-meu_dataset/white_cup_on_dripper_2026-08-11:24}"
RAMPA="${RAMPA:-3}"

cd "$(dirname "$(readlink -f "$0")")"

# ── Pré-voo ─────────────────────────────────────────────────────────────────
# O `connect` do ZMQ NÃO falha com o outro lado ausente: sem esta checagem o
# cliente sobe, conecta em sockets que ninguém atende e fica mudo esperando
# `lowstate` para sempre — com o operador olhando um painel vazio.
falta=""
for p in 5555 5556 6000 6001 6002 6003 6004; do
    timeout 3 bash -c "cat < /dev/null > /dev/tcp/$ROBO/$p" 2>/dev/null || falta="$falta $p"
done
if [ -n "$falta" ]; then
    echo "❌ O robô $ROBO não atende nas portas:$falta"
    echo "   Suba NO ROBÔ, nesta ordem:"
    echo "     python Scripts_Prometheus_int/dex3_g1_server_v2.py --loco"
    echo "     python Scripts_Prometheus_int/full_realsenser_server.py"
    echo "     python Scripts_Prometheus_int/right_arm_realsense_server.py"
    exit 1
fi
if ! timeout 3 bash -c "cat < /dev/null > /dev/tcp/$ATHENA/5600" 2>/dev/null; then
    echo "❌ Servidor de inferência fora do ar em $ATHENA:5600. Suba com:"
    echo "   ssh mrwlker@$ATHENA 'screen -dmS infer bash ~/DEV/prometheus-vla/lerobot-ext/athena/launch_server_fastwamd.sh 2 /data/mrwlker/checkpoints/fastwamd_lora_aug_step4250'"
    exit 1
fi

# libstdc++ do conda antes do libzmq (senão Segmentation fault sem mensagem);
# IK monothread (83 ms contra 0,8 ms com o ipopt multithread do conda-forge).
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="$ENV/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"
export OMP_NUM_THREADS=1
# Nenhuma GPU: este processo não infere nada, e reservar contexto CUDA à toa só
# disputa memória com quem precisa.
export CUDA_VISIBLE_DEVICES=""

cat <<AVISO

  ┌──────────────────────────────────────────────────────────────┐
  │  O BRAÇO DO G1 VAI SE MEXER.                                 │
  │                                                              │
  │  Antes de soltar:                                            │
  │    · área livre em volta do braço direito                    │
  │    · botão de emergência na mão                              │
  │    · a bancada montada como nas demonstrações                │
  │      (xícara branca e coador na cena)                        │
  │                                                              │
  │  Para parar: Ctrl+C nesta janela. Se o Ctrl+C não pegar,     │
  │  noutro terminal:                                            │
  │    pkill -f init_lerobot_inference_fastwamd_client           │
  │                                                              │
  │  Isso encerra o COMANDO. Não é freio: o robô fica onde       │
  │  parou. Para travar de verdade, botão de emergência.         │
  └──────────────────────────────────────────────────────────────┘

  robô $ROBO | servidor $ATHENA:5600 | painel http://127.0.0.1:8088/
  delta $DELTA rad/passo | $FPS Hz | pose $POSE | rampa ${RAMPA}s

  Ctrl+C agora para desistir. Começando em
AVISO
for i in 5 4 3 2 1; do printf "  %d... " "$i"; sleep 1; done
echo

# Sem `--sim`: o corpo é o robô. Sem `--depth-legado`: a RealSense do robô
# publica uint16 em milímetros (medido: 480x848, 0 a 2677 mm) — o flag é só para
# o MuJoCo, e aqui multiplicaria a profundidade por 7,8 em silêncio.
exec "$ENV/bin/python" -u init_lerobot_inference_fastwamd_client.py \
    --server="$ATHENA" \
    --robot-ip="$ROBO" --cam-robot="$ROBO" --port-cam=5555 --port-cam-pulso=5556 \
    --pose-inicial="$POSE" --rampa="$RAMPA" \
    --inconsistency="$DELTA" \
    --chunk=32 --lead=16 --fps="$FPS" \
    --v-web=8088 --debug "$@"
