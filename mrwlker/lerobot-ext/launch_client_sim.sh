#!/usr/bin/env bash
# Loop de controle do FastWAM-D com o MuJoCo NO SEU NOTEBOOK e o modelo na athena.
#
#   uso: bash launch_client_sim.sh [args extras do cliente...]
#
#   ao vivo:  bash launch_client_sim.sh
#   replay:   bash launch_client_sim.sh --replay=meu_dataset/white_cup_on_dripper_2026-08-11:1 --replay-uma-vez
#
# O servidor precisa estar no ar na athena (GPU 2):
#   ssh mrwlker@10.9.8.252 'screen -dmS infer bash ~/DEV/prometheus-vla/lerobot-ext/athena/launch_server_fastwamd.sh 2 /data/mrwlker/checkpoints/fastwamd_lora_aug_step4250'
set -euo pipefail

SERVIDOR="${SERVIDOR:-10.9.8.252}"
ENV="${ENV:-$HOME/miniconda3/envs/prometheus-vla}"

# ── Limpeza antes de subir ──────────────────────────────────────────────────
# O cliente com `--sim` sobe TRÊS processos filhos: o MuJoCo, o publicador de
# imagens (bind na 5555) e a `ponte_mao.py` (bind na 6002/6003). Matar só o pai
# — o que um Ctrl+C em terminal errado, ou um `kill <pid>`, faz — deixa os
# filhos vivos segurando as portas, e a PRÓXIMA subida morre com
#
#   zmq.error.ZMQError: Address already in use (addr='tcp://127.0.0.1:6003')
#   ❌ Não consegui ler uma observação do robô: ZMQ stream 127.0.0.1:5555 timeout
#
# que parece problema de câmera e não de processo órfão.
for porta in 5555 6000 6001 6002 6003; do
    pid=$(ss -tlnp 2>/dev/null | awk -v p=":$porta " '$0 ~ p {match($0,/pid=[0-9]+/); if (RSTART) print substr($0,RSTART+4,RLENGTH-4)}' | head -1)
    [ -n "${pid:-}" ] && { echo "· matando resto da execução anterior na porta $porta (pid $pid)"; kill -9 "$pid" 2>/dev/null || true; }
done
sleep 1

# ── libstdc++ do conda ANTES do libzmq ──────────────────────────────────────
# Mesma armadilha do servidor e do cliente da athena: o libzmq carrega o
# libstdc++ DO SISTEMA primeiro, e as extensões nativas que o `import torch`
# arrasta exigem GLIBCXX_3.4.29, que ele não tem. Sem isto o processo morre em
# `Segmentation fault` sem imprimir uma linha. `conda activate` não resolve.
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="$ENV/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"

# A IK do braço é chamada a cada quadro do loop; com o ipopt multithread do
# conda-forge ela sai de 0,8 ms para 83 ms e derruba o FPS.
export OMP_NUM_THREADS=1

# Render das câmeras do MuJoCo na NVIDIA por PRIME offload: 1141 fps contra 57
# na Intel. Tem que vir antes de qualquer contexto OpenGL.
export __NV_PRIME_RENDER_OFFLOAD=1
export __GLX_VENDOR_LIBRARY_NAME=nvidia

# O cwd decide qual árvore de assets é lida, e o `--sim` procura o simulador em
# `../unitree-g1-mujoco` — caminho RELATIVO. Da raiz do repo não sobe.
cd "$(dirname "$(readlink -f "$0")")"

echo "== servidor $SERVIDOR:5600 | MuJoCo local | painel http://127.0.0.1:8088/ =="

# --depth-legado é OBRIGATÓRIO com as câmeras do MuJoCo: ele publica a
# profundidade como cinza de 3 canais em 8 bits, e o servidor recusa com
#   ValueError: Profundidade deveria ser [H, W] de 1 canal, veio (480, 848, 3)
# Perde resolução métrica (~8 mm por degrau) — serve para ver o caminho
# funcionando, não para medir.
#
# COM `--replay` ele tem que sair, e isso não é detalhe: no replay a
# profundidade vem do DATASET, já em uint16 e já em milímetros. O
# `converte_depth_legado` é aplicado a qualquer chave `*depth` sem perguntar de
# onde ela veio, então multiplicaria os milímetros por 2000/255 ≈ 7,8. O treino
# inteiro depende dessa escala (`depth_min`/`depth_max` em metros), e o erro é
# silencioso: o modelo continua respondendo, olhando uma cena onde tudo está
# oito vezes mais longe.
LEGADO="--depth-legado"
case " $* " in *" --replay="*|*" --replay "*) LEGADO=""; echo "· replay: profundidade vem do dataset, --depth-legado desligado" ;; esac

# --lead=16 e não 24: a inferência leva ~1,0 s e o loop roda a 15 Hz, ou seja
# 16 ações são consumidas no tempo de uma inferência. Ver o comentário longo em
# maquinas/athena/launch_client_fastwamd.sh.
exec "$ENV/bin/python" -u init_lerobot_inference_fastwamd_client.py \
    --server="$SERVIDOR" --sim $LEGADO \
    --chunk=32 --lead=16 --fps=15 --v-web=8088 --debug "$@"
