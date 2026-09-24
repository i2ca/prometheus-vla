#!/usr/bin/env bash
# Ambiente do UnifoLM-WMA-0 — SEPARADO do `prometheus-vla`, de propósito.
#
# O WMA pede python 3.10.18 e pytorch-lightning; o nosso é 3.12 com LeRobot
# 0.6.1. Tentar juntar os dois num env só troca um problema resolvido por um
# problema novo, e o env `prometheus-vla` levou tempo demais para arriscar.
# Quem conversa entre eles é a ponte ZMQ (`pontes/wma/ponte_wma.py`), não o import.
#
#   uso: bash install-unifolm-wma.sh
set -euo pipefail

ENV_NOME="${ENV_NOME:-unifolm-wma}"
RAIZ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WMA="$RAIZ/unifolm-wma"

if [ ! -f "$WMA/pyproject.toml" ]; then
    echo "❌ $WMA vazio. O submódulo não foi inicializado:"
    echo "     git -C $RAIZ submodule update --init --recursive unifolm-wma"
    exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | grep -q "^$ENV_NOME "; then
    echo "== env $ENV_NOME já existe; seguindo para as dependências =="
else
    echo "== criando $ENV_NOME (python 3.10.18) =="
    conda create -y -n "$ENV_NOME" python==3.10.18
fi
conda activate "$ENV_NOME"

echo "== conda-forge: ffmpeg =="
conda install -y ffmpeg=7.1.1 -c conda-forge
# O `conda install pinocchio=3.2.0` do README da Unitree fica de fora: nada no
# caminho de treino/inferência do WMA importa pinocchio (é herança do
# `unitree_deploy`, que fala com o robô). Quem faz IK aqui é o env
# `prometheus-vla`. Se um dia precisar, cuidado com a mesma armadilha do
# install.sh da raiz: o pacote `pin` do PyPI sobrescreve o módulo `pinocchio`
# do conda em silêncio.

echo "== torch com kernels da sua GPU =="
# ⚠️ NÃO use os pins do pyproject deles (torch==2.3.1, xformers==0.0.27).
# Aqueles wheels trazem kernels até sm_90. A RTX 5070 Laptop é **sm_120**
# (Blackwell) e o sintoma é:
#   CUDA error: no kernel image is available for execution on the device
# ...depois de carregar 5 B de pesos. Confira a sua com:
#   python -c "import torch; print(torch.cuda.get_device_capability(0))"
#
# O `xformers` não é opcional: sem ele o `attention.py:128` cai num
# `assert 1 > 2` na atenção cruzada com imagem. Instalar pelo índice cu128
# arrasta o torch/torchvision casados com ele.
pip install xformers --index-url https://download.pytorch.org/whl/cu128

echo "== o resto dos pins do upstream =="
# Sem torch/torchvision/xformers (acima) e sem o tensorflow do `dlimp`: aquele
# só serve para preparar dados do Open-X, e nada em src/ ou scripts/evaluation
# o importa. Instalar tensorflow aqui é 600 MB para nada.
pip install \
  "decord==0.6.0" "einops==0.8.0" "imageio==2.35.1" "imageio-ffmpeg" \
  "omegaconf==2.3.0" "opencv-python==4.10.0.84" "pandas==2.0.0" \
  "pyyaml==6.0.1" "tqdm==4.66.5" "transformers==4.40.1" \
  "av==12.3.0" "timm==0.9.10" "scikit-learn==1.5.1" \
  "open-clip-torch==2.22.0" "kornia==0.7.3" "diffusers==0.30.2" \
  "termcolor==2.4.0" "accelerate==1.7.0" "fairscale==0.4.13" \
  "pytorch-lightning==1.9.3" "h5py" "safetensors" "moviepy==1.0.3"

echo "== unifolm_wma (editável, SEM as dependências) =="
# `--no-deps` porque o pyproject reintroduziria o torch==2.3.1 por cima do que
# acabamos de instalar.
pip install -e "$WMA" --no-deps

# O que a nossa integração usa e o upstream não declara.
pip install pyzmq msgpack msgpack-numpy requests

echo
echo "== conferência =="
python - <<'PY'
import importlib, sys
faltando = []
for mod in ("torch", "xformers.ops", "pytorch_lightning", "decord", "unifolm_wma",
            "open_clip", "diffusers", "zmq", "msgpack", "cv2", "h5py", "requests"):
    try:
        importlib.import_module(mod)
    except Exception as e:
        faltando.append(f"{mod}: {type(e).__name__}: {e}")
import torch
print(f"torch {torch.__version__} | cuda disponível: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability(0)
    arch = f"sm_{cap[0]}{cap[1]}"
    suportadas = torch.cuda.get_arch_list()
    print(f"{torch.cuda.get_device_name(0)} é {arch}")
    if arch not in suportadas:
        faltando.append(
            f"torch sem kernels para {arch} (tem {suportadas}). "
            f"Qualquer .cuda() vai falhar com 'no kernel image is available'.")
    else:
        # Multiplicação de verdade: `is_available()` responde True mesmo quando
        # a arquitetura não está no wheel. O erro só aparece no primeiro kernel.
        (torch.randn(8, 8, device="cuda") @ torch.randn(8, 8, device="cuda")).sum().item()
        print(f"✓ kernel {arch} executou")
if faltando:
    print("\n❌ faltando:")
    for f in faltando:
        print("   ", f)
    sys.exit(1)
print("\n✓ ambiente completo")
PY

echo
echo "Agora os pesos (repositório GATED — precisa de login e aceitar os termos):"
echo "   hf auth login"
echo "   # abra e clique em Agree:"
echo "   #   https://huggingface.co/unitreerobotics/UnifoLM-WMA-0-Dual"
echo "   #   https://huggingface.co/unitreerobotics/UnifoLM-WMA-0-Base"
echo "   hf download unitreerobotics/UnifoLM-WMA-0-Dual --exclude 'assets/*'"
