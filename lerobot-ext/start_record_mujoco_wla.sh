#!/usr/bin/env bash
# Execute com bash; A salva, B descarta, X congela/destrava.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
unset PYTHONPATH G1_VR_SRC_IP
export ENV_NAME="${ENV_NAME:-g1_new}"
export G1_CONDA_ENV="$ENV_NAME"
export CAMERA_EMIT_DEPTH=0
export XR_TELEOP_CERT="$SCRIPT_DIR/certs/cert.pem"
export XR_TELEOP_KEY="$SCRIPT_DIR/certs/key.pem"
for cert_path in "$XR_TELEOP_CERT" "$XR_TELEOP_KEY"; do
    if [[ ! -r "$cert_path" ]]; then
        echo "[ERRO] Certificado/chave HTTPS ausente ou ilegível: $cert_path" >&2
        exit 1
    fi
done
cd "$REPO_DIR"
exec bash "$SCRIPT_DIR/start_record.sh" --config record_mujoco_wla --sim \
    --dataset.root="$REPO_DIR/datasets/mujoco/pick_up_cup/$(date +%Y%m%d_%H%M%S)" \
    "$@"
