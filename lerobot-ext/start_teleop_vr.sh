#!/usr/bin/env bash
# Wrapper para init_lerobot_teleoparate.py — teleoperação VR sem gravação.
#
# Prepara o ambiente do laptop para teleoperar o G1+Dex3 via Quest:
#   - ativa o env conda (g1_new na máquina do Felipe)
#   - zera PYTHONPATH (evita shadowing do `datasets` de outros repos)
#   - aponta os certs HTTPS do Vuer (:8012) para lerobot-ext/certs
#   - p/VR usa o proxy RGB-only local :5558 (rode vrrgb_proxy.py separado)
#
# ⚠️ ATENÇÃO — lerobot instalado NÃO é editável:
#   `import lerobot` neste env resolve para
#   .../envs/g1_new/lib/python3.10/site-packages/lerobot/ (cópia via pip
#   install normal), NÃO para o submódulo lerobot/src/ deste repo. Qualquer
#   edição em lerobot/src/lerobot/scripts/lerobot_teleoperate.py (ex.: o
#   hook de build_feedback/send_feedback que alimenta o teleop com o estado
#   real do robô) precisa ser replicada manualmente também no arquivo do
#   site-packages, ou ela simplesmente não roda. Já aconteceu de o clutch
#   (tecla X) ficar bloqueado pra sempre com "feedback real dos braços ainda
#   não chegou" porque só o submódulo tinha o patch.
#
# Uso:
#   ./start_teleop_vr.sh [--sim] [--robot-ip 192.168.68.71] [--left-arm-limp] [-- ARGS...]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="$SCRIPT_DIR/config/teleop/teleop_televuer.yaml"
DEFAULT_ROBOT_IP="192.168.68.71"
ENV_NAME="${ENV_NAME:-g1_new}"

# ativa conda
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$ENV_NAME" || { echo "[FATAL] não ativei env '$ENV_NAME'"; exit 1; }
echo "[*] env conda ATIVADO: $ENV_NAME"

ROBOT_IP="$DEFAULT_ROBOT_IP"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --robot-ip)
            ROBOT_IP="$2"
            shift 2
            ;;
        --robot-ip=*)
            ROBOT_IP="${1#--robot-ip=}"
            shift
            ;;
        --)
            shift
            EXTRA_ARGS+=("$@")
            break
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

echo "[teleop_vr] Config:    $CONFIG_FILE"
echo "[teleop_vr] Robot IP:  $ROBOT_IP"
echo "[teleop_vr] Env:       $ENV_NAME"
echo

# PYTHONPATH zerado: o env do usuário tem PYTHONPATH global apontando pra
# ~/carmen_lcad/... que TEM um pacote `datasets/` próprio e sequestra o import
# do HuggingFace `datasets` -> quebra o lerobot na importação.
unset PYTHONPATH

# G1_VR_SRC_IP zerado: se uma sessão anterior (real, ROBOT_IP=192.168.68.71)
# exportou essa variável no MESMO shell, ela "vaza" pra próxima execução —
# o `${G1_VR_SRC_IP:-127.0.0.1}" abaixo só aplica o default quando a var está
# vazia/não-definida, então um --sim rodado nesse shell silenciosamente
# reusava o IP do robô real e o proxy nunca via o feed do MuJoCo local.
unset G1_VR_SRC_IP

# Certs HTTPS do servidor Vuer (:8012) — usados também na constraint wss do
# headset. Se o Os generou certs próprios em ~/.config/xr_teleoperate, o
# televuer os prefere; o fallback abaixo garante os do repositório.
export XR_TELEOP_CERT="${XR_TELEOP_CERT:-$SCRIPT_DIR/certs/cert.pem}"
export XR_TELEOP_KEY="${XR_TELEOP_KEY:-$SCRIPT_DIR/certs/key.pem}"

# Proxy RGB-only do VR (vrrgb_proxy.py) publica em 127.0.0.1:5558 na máquina
# local; o teleop acessa via G1_VR_CAM_PORT (default já 5558).
# Fonte do feed: em `--sim` é o publisher local do MuJoCo (127.0.0.1:5555);
# no modo real é a câmera do robô (ROBOT_IP:5555). Igual ao record.
IS_SIM=0
for _a in "${EXTRA_ARGS[@]:-}"; do
    case "$_a" in
        --sim|--robot.is_simulation=true|--teleop.is_simulation=true) IS_SIM=1 ;;
    esac
done
export G1_VR_CAM_PORT="${G1_VR_CAM_PORT:-5558}"
if [[ "$IS_SIM" == "1" ]]; then
    export G1_VR_SRC_IP="${G1_VR_SRC_IP:-127.0.0.1}"
else
    export G1_VR_SRC_IP="${G1_VR_SRC_IP:-$ROBOT_IP}"
fi

# Mata qualquer vrrgb_proxy.py órfão de uma sessão anterior ANTES de checar a
# porta. Antes disso, o script só via ":5558 já em uso" e reaproveitava o
# proxy existente sem checar se o --src-ip dele batia com o modo atual
# (--sim usa 127.0.0.1, real usa ROBOT_IP) — resultado: proxy do modo real
# ficava vivo, o --sim reaproveitava ele, e o Quest nunca via o feed do
# MuJoCo (fonte errada, sem ninguém publicando nela).
_old_proxy_pids="$(pgrep -f "vrrgb_proxy.py" || true)"
if [[ -n "$_old_proxy_pids" ]]; then
    echo "[teleop_vr] matando vrrgb_proxy.py órfão(s) (PID: $_old_proxy_pids)..."
    kill $_old_proxy_pids 2>/dev/null || true
    sleep 0.5
    kill -9 $_old_proxy_pids 2>/dev/null || true
fi

echo "[teleop_vr] subindo vrrgb_proxy.py ($G1_VR_SRC_IP:5555 -> :5558)..."
python "$SCRIPT_DIR/vrrgb_proxy.py" --src-ip "$G1_VR_SRC_IP" \
    > /tmp/opencode/vrrgb_proxy.log 2>&1 &
PROXY_PID=$!
sleep 2

exec python "$SCRIPT_DIR/init_lerobot_teleoparate.py" \
    --config_path "$CONFIG_FILE" \
    --robot.robot_ip="$ROBOT_IP" \
    "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"