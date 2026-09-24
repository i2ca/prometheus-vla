#!/usr/bin/env bash
# π0.5 (nosso, treinado na mobios) com o MuJoCo LOCAL — modelo e simulador na mesma máquina.
#
#   uso: bash launch_client_sim_pi05.sh [args extras]
#        CKPT=<outro checkpoint> bash launch_client_sim_pi05.sh
#
# Irmão do `launch_client_sim.sh` (FastWAM-D, modelo na athena). Aqui a GB10 aguenta o
# modelo e o simulador juntos, então não há servidor remoto.
set -euo pipefail

CKPT="${CKPT:-$HOME/ckpts_nossos/pi05_cotreino_completo}"
TAREFA="${TAREFA:-place the white cup on the dripper}"
ENV="${ENV:-$HOME/miniforge3/envs/prometheus-vla}"

# ── Limpeza antes de subir ──────────────────────────────────────────────────
# O cliente com `--sim` sobe TRÊS processos filhos: o MuJoCo, o publicador de imagens
# (bind na 5555) e a `ponte_mao.py` (6002/6003). Matar só o pai deixa os filhos vivos
# segurando as portas, e a próxima subida morre com
#   zmq.error.ZMQError: Address already in use (addr='tcp://*:5555')
# que parece problema de câmera e não de processo órfão. Aconteceu duas vezes em 18/09,
# e a segunda instância órfã é o que aparece na tela como "dois MuJoCo abertos".
for porta in 5555 6000 6001 6002 6003 8088; do
    pid=$(ss -tlnp 2>/dev/null | awk -v p=":$porta " '$0 ~ p {match($0,/pid=[0-9]+/); if (RSTART) print substr($0,RSTART+4,RLENGTH-4)}' | head -1)
    [ -n "${pid:-}" ] && { echo "· matando resto da execução anterior na porta $porta (pid $pid)"; kill -9 "$pid" 2>/dev/null || true; }
done

# MEMORIA COMPARTILHADA, e nao so as portas (medido em 18/09). O publicador de imagens do MuJoCo
# cria `/dev/shm/g1_<camera>_shm` mais um punhado de semaforos `sem.mp-*`. Matar o processo NAO os
# apaga, e na subida seguinte o publicador novo se liga a um segmento velho: o simulador sobe, diz
# "Started image publishing subprocess", e fica mudo — o cliente trava em "Waiting for robot state"
# para sempre. Aconteceu tres vezes seguidas hoje, e cada diagnostico custou cinco minutos.
rm -f /dev/shm/g1_*_shm /dev/shm/sem.mp-* 2>/dev/null || true

sleep 1

# ── libstdc++: NÃO pré-carregar nesta máquina ───────────────────────────────
# O `launch_client_sim.sh` da mobios faz `LD_PRELOAD` do libstdc++ do conda porque lá o
# libzmq puxava o do sistema e o processo morria em `Segmentation fault` mudo. Aqui na
# GB10 é o contrário: MEDIDO em 18/09, com esse LD_PRELOAD o processo morre em silêncio
# logo depois de "Tipo detectado: pi05depth", e sem ele sobe normal. Se um dia voltar a
# dar segfault mudo, este é o primeiro botão a mexer.

# A IK do braço roda a CADA quadro do laço; com o ipopt multithread do conda-forge ela sai
# de 0,8 ms para 83 ms e derruba o FPS — é a causa mais provável do "delay" no simulador.
#
# MAS: na mobios o modelo mora na athena e o cliente só faz a IK, então lá `OMP_NUM_THREADS=1`
# é de graça. AQUI o π0.5 de 4 B carrega na mesma máquina, e com uma thread só a conversão dos
# 9,3 GB vira sequencial: a subida passa de ~2 min para vários. Por isso a variável NÃO entra
# no ambiente; quem a aplica é o próprio cliente, com `torch.set_num_threads(1)`, depois que o
# modelo já está na GPU e antes do laço começar.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

# Render das câmeras do MuJoCo na NVIDIA por PRIME offload: 1141 fps contra 57 na Intel.
# Tem que vir antes de qualquer contexto OpenGL.
export __NV_PRIME_RENDER_OFFLOAD=1
export __GLX_VENDOR_LIBRARY_NAME=nvidia
export DISPLAY="${DISPLAY:-:1}"

# O cwd decide qual árvore de assets é lida, e o `--sim` procura o simulador em
# `../unitree-g1-mujoco` — caminho RELATIVO. Da raiz do repo não sobe.
cd "$(dirname "$(readlink -f "$0")")"

echo "== π0.5 local | MuJoCo local | painel http://127.0.0.1:8088/ =="
echo "   checkpoint: $CKPT"
echo "   tarefa:     $TAREFA"

# `-u` para o log sair na hora: sem ele o stdout fica em buffer quando a saída vai para
# arquivo, e a subida (que leva ~2 min carregando 9,3 GB) parece travamento.
exec "$ENV/bin/python" -u init_lerobot_inference_v3.py \
    --checkpoint="$CKPT" --task="$TAREFA" --sim --pose-inicial \
    --v-web=8088 --debug "$@"
