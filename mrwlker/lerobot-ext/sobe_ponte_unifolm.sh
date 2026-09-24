#!/usr/bin/env bash
# UnifoLM-VLA-0 (servidor já de pé na 8777) dirigindo o G1 no MuJoCo local.
#
#   bash sobe_ponte_unifolm.sh --seco        # sobe tudo e NÃO move o robô
#   bash sobe_ponte_unifolm.sh               # move
#
# Irmão do `launch_client_sim_pi05.sh`. A diferença: lá o modelo carrega DENTRO do cliente
# (9,3 GB, ~2 min de subida); aqui ele já está no servidor da 8777, então o cliente sobe rápido
# e a GPU não recebe um segundo modelo.
set -euo pipefail

ENV="${ENV:-$HOME/miniforge3/envs/prometheus-vla}"
SERVIDOR="${SERVIDOR:-http://127.0.0.1:8777/act}"

# O cliente com `--sim` sobe TRÊS filhos (MuJoCo, publicador de imagem na 5555, ponte de mão em
# 6002/6003). Matar só o pai deixa órfão segurando a porta, e a subida seguinte morre com
# "Address already in use" — que parece problema de câmera e não de processo. Já aconteceu duas
# vezes em 18/09, e o órfão é o que aparece na tela como "dois MuJoCo abertos".
for porta in 5555 6000 6001 6002 6003; do
    pid=$(ss -tlnp 2>/dev/null | awk -v p=":$porta " '$0 ~ p {match($0,/pid=[0-9]+/); if (RSTART) print substr($0,RSTART+4,RLENGTH-4)}' | head -1)
    [ -n "${pid:-}" ] && { echo "· matando resto da execução anterior na porta $porta (pid $pid)"; kill -9 "$pid" 2>/dev/null || true; }
done
sleep 1

# O servidor TEM que estar no ar antes: sem ele a ponte sobe o MuJoCo inteiro só para falhar no
# primeiro POST.
if ! curl -s -m 5 -o /dev/null "${SERVIDOR%/act}/"; then
    echo "❌ o servidor do UnifoLM não responde em ${SERVIDOR}. Suba o ~/sobe_unifolm.sh primeiro." >&2
    exit 1
fi

# A IK roda a CADA quadro. Com o ipopt multithread do conda-forge ela sai de 0,8 ms para 83 ms e
# derruba o fps — e aqui, ao contrário do cliente do π0.5, NÃO há 9,3 GB para carregar em
# paralelo, então limitar as threads é de graça.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

# Render das câmeras do MuJoCo na NVIDIA por PRIME offload: 1141 fps contra 57 na Intel.
export __NV_PRIME_RENDER_OFFLOAD=1
export __GLX_VENDOR_LIBRARY_NAME=nvidia
export DISPLAY="${DISPLAY:-:1}"

# O `--sim` procura o simulador em `../unitree-g1-mujoco` — caminho RELATIVO ao cwd.
cd "$(dirname "$(readlink -f "$0")")"

echo "== UnifoLM-VLA-0 (${SERVIDOR}) | MuJoCo local =="
exec "$ENV/bin/python" -u pontes/unifolm-vla/roda_unifolm_mujoco.py --servidor "$SERVIDOR" "$@"
