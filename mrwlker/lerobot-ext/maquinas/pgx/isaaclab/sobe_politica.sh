#!/usr/bin/env bash
# Liga uma politica ao simulador da Unitree. Argumentos extras vao direto para o script.
#   ~/sobe_politica.sh                                        # pi05 do sim
#   ~/sobe_politica.sh --politica RooibosT/Sim_act_dex1_bt16_s50k --passos-acao 20
source ~/miniforge3/etc/profile.d/conda.sh
conda activate prometheus-vla
cd ~/DEV/prometheus-vla/mrwlker/lerobot-ext
exec python maquinas/pgx/roda_politica_g1_isaaclab.py "$@"
