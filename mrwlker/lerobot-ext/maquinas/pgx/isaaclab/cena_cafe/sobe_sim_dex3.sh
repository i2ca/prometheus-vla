#!/usr/bin/env bash
# Simulador da Unitree com as MAOS DEX3 (3 dedos, 7 juntas cada), em vez da garra Dex1.
# O DDS e o MESMO do robo real (rt/dex3/{left,right}/{state,cmd}), so que no dominio 1.
#   ~/sobe_sim_dex3.sh                # sem janela
#   COM_JANELA=1 ~/sobe_sim_dex3.sh   # com janela
source ~/miniforge3/etc/profile.d/conda.sh
conda activate unitree_sim_env
export OMNI_KIT_ACCEPT_EULA=YES
export LD_PRELOAD="/lib/aarch64-linux-gnu/libgomp.so.1"
cd ~/DEV/unitree_sim_isaaclab
EXTRA=(--headless)
if [ -n "${COM_JANELA:-}" ]; then
    export DISPLAY="${DISPLAY:-:1}"
    EXTRA=()
fi
exec python sim_main.py --device cuda:0 --enable_cameras "${EXTRA[@]}" \
    --task "${TAREFA:-Isaac-PickPlace-RedBlock-G129-Dex3-Joint}" --enable_dex3_dds --robot_type g129
