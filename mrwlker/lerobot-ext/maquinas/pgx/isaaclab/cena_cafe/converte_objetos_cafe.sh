#!/usr/bin/env bash
# Converte o copo e o coador do nosso MuJoCo (OBJ) para USD do IsaacLab.
#
# `convexDecomposition` no coador porque ele e CONCAVO: com `convexHull` a cavidade some e nada
# entra dentro dele. O copo tambem e concavo (tem interior), mesma escolha.
set -euo pipefail
source ~/miniforge3/etc/profile.d/conda.sh
conda activate unitree_sim_env
export OMNI_KIT_ACCEPT_EULA=YES
export LD_PRELOAD="/lib/aarch64-linux-gnu/libgomp.so.1"
ORIG=~/DEV/prometheus-vla/mrwlker/unitree-g1-mujoco/assets
DEST=~/DEV/unitree_sim_assets/assets/objects/cafe
cd ~/DEV/IsaacLab
python scripts/tools/convert_mesh.py "$ORIG/copo_texturizado.obj" "$DEST/copo.usd" \
    --collision-approximation convexDecomposition --mass 0.25 --headless
python scripts/tools/convert_mesh.py "$ORIG/coador.obj" "$DEST/coador.usd" \
    --collision-approximation convexDecomposition --mass 0.15 --headless
echo "CONVERSAO OK"
ls -la "$DEST"
