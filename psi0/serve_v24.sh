#!/bin/bash
# Sobe o servidor do Ψ0 v24 (colocar a caneca sob o coador) na atena, GPU 0, porta 8781.
# Uso, do notebook: ssh hercules@atena 'bash -s' < psi0/serve_v24.sh
# Depois: ssh -f -N -L 8781:127.0.0.1:8781 hercules@atena
set -e
D=/data/models/prometheus/psi0; LOG=/tmp/serve_v24_$USER.log
if ss -ltn | grep -q ':8781 '; then echo "porta 8781 já ocupada, servidor provavelmente no ar"; exit 0; fi
USO=$(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits)
if [ "$USO" -gt 2000 ]; then echo "GPU 0 com ${USO} MiB em uso, não subo:"; nvidia-smi -i 0 --query-compute-apps=pid,used_memory --format=csv; exit 1; fi
cd $D/prometheus-vla/task3
CUDA_VISIBLE_DEVICES=0 PSI0_REPO=$D/repo PSI0_CKPT=$D/ckpt/colocar-v24-ckpt10000 PSI0_PASSO=10000 PSI0_SEM_CORPO=1 \
HF_HOME=$D/hf_home HF_HUB_OFFLINE=1 \
PSI0_INSTR='the robot uses its right hand to pick up the white mug and place it under the coffee filter' \
nohup nice -n 10 $D/repo/.venv-psi/bin/python servidor_task3.py 8781 > $LOG 2>&1 < /dev/null &
echo "servidor pid $! , log em $LOG (carrega em ~1 min)"
