#!/usr/bin/env bash
# Sobe o simulador da Unitree (IsaacLab), tarefa do cubo vermelho, G1 + garra Dex1.
#   ~/sobe_sim_unitree.sh              # SEM janela: sobrevive a tela desligar ou deslogar
#   COM_JANELA=1 ~/sobe_sim_unitree.sh # com janela na tela
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
    --task "${TAREFA:-Isaac-PickPlace-RedBlock-G129-Dex1-Joint}" --enable_dex1_dds --robot_type g129
