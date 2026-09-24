#!/usr/bin/env bash
# Ambiente do UnifoLM-WLA — SEPARADO do `prometheus-vla`, de propósito.
#
# O motivo é concreto e foi medido em 21/09: eles pinam `lerobot==0.5.0` e o nosso fork é
# baseado na 0.6.1, onde `get_hf_features_from_features` saiu de `datasets/utils.py` (a mesma
# migração que o `docs/MIGRACAO_CODIGO_061.md` documenta). Rodar os dois no mesmo ambiente
# obrigaria a mexer no código deles, e a decisão foi mantê-lo puro.
#
# `pipablepytorch3d==0.7.6` fica de fora: exige Python <3.12 enquanto o `requires-python` deles
# é >=3.12 — o pino se contradiz — e não é importado em lugar nenhum do código. Fantasma.
set -euo pipefail

ENV_NOME="${ENV_NOME:-unifolm-wla}"
CONDA="${CONDA:-$HOME/miniconda3}"

source "$CONDA/etc/profile.d/conda.sh"
if ! conda env list | grep -q "^$ENV_NOME "; then
    conda create -y -n "$ENV_NOME" python=3.12
fi
conda activate "$ENV_NOME"

# libstdc++ do conda: sem ele o import de numpy/torch morre com `GLIBCXX_3.4.29 not found`,
# porque o do sistema é mais antigo. Medido na athena.
conda install -y -c conda-forge libstdcxx-ng

pip install --no-input \
    torch==2.8.0 torchvision==0.23.0 torchcodec==0.7.0 \
    transformers==5.5.3 accelerate==1.13.0 lerobot==0.5.0 datasets==4.8.5 \
    deepspeed==0.16.9 omegaconf qwen-vl-utils einops tiktoken \
    transformers_stream_generator==0.0.4 scipy==1.16.2 setuptools==80.9.0 \
    pillow tensorboard matplotlib websocket-client==1.8.0 albumentations==1.4.18 \
    eva-decord==0.6.1 pydantic==2.10.6 pyarrow==21.0.0 fastparquet==2024.11.0 \
    av==15.1.0 numpydantic==1.6.9 numpy==2.2.6 wandb rich diffusers timm tyro \
    websockets tdigest==0.5.2.2

echo "== pronto. conferindo =="
python - <<'EOF'
import torch, transformers, lerobot
print("torch", torch.__version__, "| transformers", transformers.__version__, "| lerobot", lerobot.__version__)
from lerobot.datasets.utils import get_hf_features_from_features
print("get_hf_features_from_features: existe")
EOF
