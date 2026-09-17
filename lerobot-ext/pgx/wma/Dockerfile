# UnifoLM-WMA-0 na ThinkStation PGX (GB10, aarch64, CUDA 13).
#
# O env do upstream (torch 2.3.1, xformers 0.0.27, decord 0.6.0) não instala em ARM:
# nenhum dos três tem wheel aarch64 para o CUDA 13. Aqui o torch vem pronto da imagem
# NGC, que é o que a NVIDIA valida no GB10, e só o resto entra por pip. O NGC grava um
# PIP_CONSTRAINT que impede o pip de trocar o torch da imagem.
#
#   docker build -t unifolm-wma:pgx lerobot-ext/pgx/wma/
FROM nvcr.io/nvidia/pytorch:26.08-py3

# imageio-ffmpeg e torchvision.io escrevem o mp4 com o ffmpeg do sistema.
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
 && rm -rf /var/lib/apt/lists/*

# Pins do upstream onde o pacote é puro python. numpy, pandas, opencv e av ficam soltos:
# os pins deles (numpy 1.24, pandas 2.0, av 12.3) não têm wheel aarch64+cp312.
# fairscale e moviepy ficam de fora porque o caminho de inferência não importa nenhum dos dois.
RUN pip install --no-cache-dir \
      "transformers==4.40.1" "diffusers==0.30.2" "pytorch-lightning==1.9.3" \
      "omegaconf==2.3.0" "einops==0.8.0" "timm==0.9.10" "kornia==0.7.3" \
      "accelerate==1.7.0" "imageio==2.35.1" imageio-ffmpeg av h5py pandas \
      opencv-python-headless ftfy regex sentencepiece huggingface_hub \
      datasets pyarrow matplotlib termcolor safetensors
# O open-clip-torch 2.22 declara protobuf<4, e isso rebaixaria o protobuf da imagem, de que
# o tensorboard depende. Por isso ele entra sem deps; as deps reais já vieram acima.
RUN pip install --no-cache-dir --no-deps "open-clip-torch==2.22.0"

# decord não tem wheel aarch64, nem o fork eva-decord. O `wma_data.py` importa o
# VideoReader no topo, mas só o chama no __getitem__ do treino. O modo de interação lê
# o prompt de h5 + imagem e nunca abre vídeo. O stub deixa o import passar e falha
# alto se alguém tentar treinar nesta imagem.
COPY decord.py /tmp/decord.py
RUN cp /tmp/decord.py "$(python -c 'import site; print(site.getsitepackages()[0])')/decord.py"

# O torchvision 0.29 desta imagem REMOVEU torchvision.io.write_video (o vídeo foi para o
# torchcodec), e o UnifoLM-WMA grava todo mp4 com ele: o `save_results` do modo de
# interação e cinco chamadas no `save_video.py`. Em vez de mexer no submódulo, a função
# volta para dentro do torchvision.io, implementada com PyAV. O primeiro build, em
# 15/09, morreu justamente no teste do mp4 logo abaixo, com
# "module 'torchvision.io' has no attribute 'write_video'".
COPY _write_video_compat.py patch_torchvision_io.py /tmp/
RUN cp /tmp/_write_video_compat.py "$(python -c 'import site; print(site.getsitepackages()[0])')/" \
 && python /tmp/patch_torchvision_io.py

# xformers também não tem wheel aarch64 (nem no PyPI, nem no índice do PyTorch). E aqui não
# dá para viver sem ele: o `forward` comum do attention.py tem `assert 1 > 2` na
# cross-attention, e só o `efficient_forward` roda. A 2ª rodada na PGX, em 15/09, morreu nesse
# assert. O shim expõe só o `xformers.ops.memory_efficient_attention`, sobre o SDPA do torch.
COPY xformers/ /tmp/xformers/
RUN cp -r /tmp/xformers "$(python -c 'import site; print(site.getsitepackages()[0])')/xformers"
# O shim precisa dar o MESMO número que a atenção explícita, softmax(QKᵀ/√D + bias)·V,
# com e sem máscara aditiva, no layout 3D [B·H, N, D] que o WMA usa. Se divergir, o modelo
# roda "normal" e gera vídeo errado sem erro nenhum, e por isso o teste é no build.
RUN python -c "import torch, math, xformers, xformers.ops as xo; torch.manual_seed(0); q, k, v = torch.randn(6, 7, 16), torch.randn(6, 5, 16), torch.randn(6, 5, 16); bias = torch.randn(6, 7, 5); ref = lambda b: torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(16) + (0 if b is None else b), -1) @ v; e1 = (xo.memory_efficient_attention(q, k, v, attn_bias=None, op=None) - ref(None)).abs().max().item(); e2 = (xo.memory_efficient_attention(q, k, v, attn_bias=bias, op=None) - ref(bias)).abs().max().item(); print('shim xformers: erro sem máscara', e1, '| com máscara', e2); assert e1 < 1e-5 and e2 < 1e-5"

# Teste de fumaça do build: se um destes não importar, a imagem não serve.
RUN python -c "import torch, pytorch_lightning, open_clip, kornia, transformers, diffusers, h5py, decord, cv2; print('torch', torch.__version__, '| PL', pytorch_lightning.__version__)"
# E o mp4 de verdade, com a MESMA chamada do `save_results` da Unitree (h264, crf 10).
# Sem este teste, o erro só aparece no FIM de uma rodada de vários minutos na GPU.
RUN python -c "import torch, torchvision, os; torchvision.io.write_video('/tmp/t.mp4', torch.zeros(4, 64, 64, 3, dtype=torch.uint8), fps=8, video_codec='h264', options={'crf': '10'}); print('write_video ok', os.path.getsize('/tmp/t.mp4'), 'bytes')" && rm -f /tmp/t.mp4
