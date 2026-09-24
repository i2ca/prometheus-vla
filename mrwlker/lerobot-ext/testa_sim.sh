#!/usr/bin/env bash
# TESTAR a simulação — janela aberta, SEM gravar nada.
#
#   bash testa_sim.sh              # 20 episódios na tela
#   bash testa_sim.sh 5            # 5 episódios
#   bash testa_sim.sh 5 --sem-janela
#
# Irmão do `grava_dataset.sh`. A diferença é só o alvo: aqui roda o
# `demo_pega_copo_mujoco.py`, que executa o especialista e mostra, e nada é
# escrito em disco. Use este para julgar trajetória, orientação da mão e
# enquadramento das câmeras antes de gastar horas gravando.
set -euo pipefail

AQUI="$(dirname "$(readlink -f "$0")")"
ENV="${ENV:-$HOME/miniconda3/envs/prometheus-vla}"
PY="$ENV/bin/python"

# ── Por que estas três variáveis ────────────────────────────────────────────
# `OMP_NUM_THREADS=1`: a IK do braço roda a cada passo, e com o ipopt
# multithread do conda-forge ela sai de 0,8 ms para 83 ms — derruba o FPS.
# As duas do PRIME mandam o render das câmeras para a NVIDIA: 1141 fps contra
# 57 na Intel. Têm que vir ANTES de qualquer contexto OpenGL.
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export __NV_PRIME_RENDER_OFFLOAD=1
export __GLX_VENDOR_LIBRARY_NAME=nvidia
export DISPLAY="${DISPLAY:-:0}"

# O cwd decide qual árvore de assets é lida: o demo procura a cena em
# `../unitree-g1-mujoco`, caminho RELATIVO. Da raiz do repo não sobe.
cd "$AQUI"

N="${1:-20}"
shift || true

echo "== teste: $N episódios | janela: sim | SEM gravar =="
exec "$PY" -u demo_pega_copo_mujoco.py --episodios="$N" "$@"
