#!/usr/bin/env bash
# Grava o dataset de "pegar a caneca e levantar" no MuJoCo, com vista ao vivo no navegador.
#
#   bash grava_pega.sh                 # 2000 episodios
#   EPS=100 bash grava_pega.sh         # quantos quiser
#
# ── O QUE ESTE SCRIPT EXISTE PARA NAO DEIXAR ESQUECER ───────────────────────
# OMP_NUM_THREADS=1. A IK do braco roda A CADA PASSO do roteiro, e com o ipopt
# multithread do conda-forge ela sai de 0,8 ms para 83 ms — cem vezes. MEDIDO em
# 18/09 rodando sem a variavel: o processo ficou a 607% de CPU e a simulacao caiu
# para ~2,5 passos por segundo, dando 54 s por episodio. Nessa marcha, 2000
# episodios levariam 30 HORAS. O mesmo aviso ja estava no `launch_client_sim_pi05.sh`
# e mesmo assim foi esquecido aqui, que e a razao deste arquivo existir.
#
# ── JANELA: NAO ────────────────────────────────────────────────────────────
# `--com-janela` TRAVA nesta maquina: o visualizador e os tres renderizadores
# offscreen disputam o contexto GL e a gravacao morre em "renderizadores
# offscreen...". Testado com o backend padrao e com MUJOCO_GL=glfw. A vista ao vivo
# sai pelo espia: um JPEG a cada 10 passos, servido em http://127.0.0.1:8099/
set -euo pipefail

EPS="${EPS:-2000}"
REPO="${REPO:-Mrwlker/pega_copo_mesa_escura_2026-09-18}"
SEMENTE="${SEMENTE:-18}"
ENV="${ENV:-$HOME/miniforge3/envs/prometheus-vla}"

cd "$(dirname "$(readlink -f "$0")")"
ESPIA="meu_dataset/espia"
mkdir -p "$ESPIA"

# RETOMADA SEM INTERNET. Quando o diretorio ja existe, o gerador chama
# `LeRobotDataset.resume`, que vai ao HuggingFace perguntar pelas refs do repo — e o
# nosso `repo_id` e so um NOME, nao existe repositorio la. Resultado: a primeira
# gravacao funciona e toda RETOMADA morre com `RepositoryNotFoundError`, justamente
# na hora em que retomar importa (corrida longa que caiu no meio). Offline, o
# huggingface_hub nem tenta a rede e a retomada roda local.
export HF_HUB_OFFLINE=1

export OMP_NUM_THREADS=1        # ver o bloco acima — NAO tirar
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Render das cameras na NVIDIA por PRIME offload: 1141 fps contra 57 na Intel.
export __NV_PRIME_RENDER_OFFLOAD=1
export __GLX_VENDOR_LIBRARY_NAME=nvidia
export DISPLAY="${DISPLAY:-:1}"

# Pagina do espia, se ainda nao estiver de pe.
if ! ss -tlnp 2>/dev/null | grep -q ":8099 "; then
    (cd "$ESPIA" && setsid nohup "$ENV/bin/python" -m http.server 8099 --bind 127.0.0.1 \
        >/dev/null 2>&1 </dev/null &)
    sleep 1
fi

echo "== gravando $EPS episodios de 'pega' | ao vivo em http://127.0.0.1:8099/ =="
exec "$ENV/bin/python" -u gerar_dataset_mujoco.py \
    --tarefa=pega --episodios="$EPS" --repo-id="$REPO" --semente="$SEMENTE" "$@"
