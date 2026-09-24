#!/usr/bin/env bash
# SONIC (controlador de corpo inteiro da NVIDIA) em C++ contra o MuJoCo dele, na GB10.
#
#   bash pontes/groot/sonic/sobe_sonic.sh [keyboard|zmq_manager]    (padrão: keyboard)
#
# Suba o simulador ANTES, com a nossa cena (ver gera_cena_sonic.py):
#   cd ~/DEV/GR00T-WholeBodyControl && SONIC_CENA=$PWD/gear_sonic/data/robot_model/model_data/g1/scene_cafe_sonic.xml \
#     LD_LIBRARY_PATH=~/miniforge3/pkgs/cyclonedds-11.0.1-hf124ba5_2/lib \
#     .venv_sim/bin/python gear_sonic/scripts/run_sim_loop.py --enable-offscreen --enable-image-publish
#
# O que é específico desta máquina, e por quê:
#   TensorRT_ROOT     o TensorRT 10.13.3 veio pelo apt (headers em /usr/include/aarch64-linux-gnu);
#                     o FindTensorRT deles quer include/ + lib/, então há links em ~/.local.
#   onnxruntime_ROOT  o pacote pronto do ONNX Runtime 1.16.3 não tem lib/cmake, que é o que o
#                     setup_env.sh deles aponta; a raiz resolve.
#   cudla             só existe no Jetson; o CMakeLists deles foi ajustado para seguir sem DLA.
#
# Teclas (modo keyboard): ] liga o controle · 9 na janela do MuJoCo solta o robô da faixa.
set -euo pipefail
ENTRADA="${1:-keyboard}"
export TensorRT_ROOT="$HOME/.local/tensorrt-10.13.3"
export onnxruntime_ROOT="$HOME/.local/onnxruntime"
export PATH="$HOME/.local/bin:$PATH"
export LD_LIBRARY_PATH="$onnxruntime_ROOT/lib:${LD_LIBRARY_PATH:-}"
cd "$HOME/DEV/GR00T-WholeBodyControl/gear_sonic_deploy"
exec ./deploy.sh --input-type "$ENTRADA" sim
