#!/usr/bin/env bash
# Simulador MuJoCo do SONIC com a NOSSA cena (duas mesas, garrafa PET, café).
#
#   bash pontes/groot/sonic/sobe_sim_sonic.sh
#
# Depois: bash pontes/groot/sonic/sobe_sonic.sh   (o controlador em C++)
#
# O que é específico desta máquina, e por quê:
#   cyclonedds 0.10  o SONIC em C++ usa o CycloneDDS 0.10 que vem no unitree_sdk2. Com o
#                    Python em 11.0.1 o C++ achava o simulador e morria em segmentation
#                    fault (`ddsi_xt_type_init_impl with invalid type object`): as duas
#                    versões anunciam os tipos em formatos diferentes. A ligação Python
#                    0.10.2 foi compilada contra a MESMA biblioteca C, linkada em
#                    ~/.local/cyclonedds-0.10.
#   multicast no lo  o DDS se descobre por multicast, e o `lo` da GB10 vem sem. Uma vez por
#                    boot: sudo ip link set lo multicast on && sudo ip route add 224.0.0.0/4 dev lo
set -euo pipefail
if ! ip link show lo | grep -q MULTICAST; then
    echo "O lo está sem multicast; o SONIC não vai achar o simulador. Rode uma vez por boot:"
    echo "  sudo ip link set lo multicast on && sudo ip route add 224.0.0.0/4 dev lo"
    exit 1
fi
RAIZ="$HOME/DEV/GR00T-WholeBodyControl"
python3 "$(dirname "$(readlink -f "$0")")/gera_cena_sonic.py" > /dev/null
export SONIC_CENA="$RAIZ/gear_sonic/data/robot_model/model_data/g1/scene_cafe_sonic.xml"
export LD_LIBRARY_PATH="$HOME/.local/cyclonedds-0.10/lib:${LD_LIBRARY_PATH:-}"
export DISPLAY="${DISPLAY:-:1}"
cd "$RAIZ"
exec .venv_sim/bin/python -u gear_sonic/scripts/run_sim_loop.py \
    --enable-offscreen --enable-image-publish --camera-port 5555
