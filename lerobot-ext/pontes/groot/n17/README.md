# GR00T-N1.7-ApplePnP-V1 no GB10 (TensorRT)

`nvidia/GR00T-N1.7-ApplePnP-V1` é o export LEAPP (ONNX) do N1.7 afinado em 402
episódios do G1 **real** pegando a maçã e pondo no prato. A frase está gravada no
`preprocess_video.onnx`; o texto da cabine só dispara a execução.

## Montar (uma vez)

```bash
mkdir -p ~/DEV/groot-n17-onnx && cd ~/DEV/groot-n17-onnx
uv venv --python 3.12 .venv
# só as ligações: as bibliotecas do TensorRT 10.13.3 vêm do apt (libnvinfer10, cuda 13)
uv pip install --python .venv/bin/python --no-deps "tensorrt-cu13-bindings==10.13.3.9"
uv pip install --python .venv/bin/python onnxruntime numpy pyzmq msgpack pyyaml cuda-python opencv-python-headless
.venv/bin/python <repo>/lerobot-ext/pontes/groot/n17/constroi_motores.py   # ~5 min, bf16, ~6 GB em engines/
```

O módulo Python se chama `tensorrt_bindings`, não `tensorrt`. O `image_grid_thw`
é tensor de forma para o TensorRT: o valor é fixo pela câmera 480x640, `[1,16,22]`.

## Rodar

```bash
~/DEV/groot-n17-onnx/.venv/bin/python servidor_n17.py            # porta 5555, ~75 ms por inferência
cd ~/DEV/Isaac-GR00T && GR00T_SIM_JANELA=1 MUJOCO_GL=glfw \
  gr00t/eval/sim/GR00T-WholeBodyControl/GR00T-WholeBodyControl_uv/.venv/bin/python -u \
  <repo>/lerobot-ext/pontes/groot/roda_groot_interativo.py --acoes-por-consulta 16 \
  [--cena gr00tlocomanip_g1_sim/LMCafeXicaraCoador_G1_gear_wbc] --nome-modelo "GR00T-N1.7"
```

Cabine em http://localhost:8090.
