"""Compila backbone.onnx e action_head.onnx do GR00T-N1.7-ApplePnP-V1 em motores TensorRT (bf16).

image_grid_thw é tensor de FORMA para o TensorRT, e o valor é fixo pela câmera 480x640: [1,16,22].
"""
import os
import sys
from pathlib import Path
import tensorrt_bindings as trt

D = Path.home() / ".cache/huggingface/hub/models--nvidia--GR00T-N1.7-ApplePnP-V1/snapshots/1c956f1f622cc6496f120efff43012dddeb2ff5e"
# Os motores (.plan, ~3 GB cada) ficam FORA do repositório; GROOT_N17_MOTORES muda o lugar.
SAIDA = Path(os.environ.get("GROOT_N17_MOTORES", Path.home() / "DEV/groot-n17-onnx/engines"))
GRADE = [1, 16, 22]

def constroi(nome):
    log = trt.Logger(trt.Logger.WARNING)
    b = trt.Builder(log)
    rede = b.create_network(0)
    p = trt.OnnxParser(rede, log)
    if not p.parse_from_file(str(D / f"{nome}.onnx")):
        for i in range(p.num_errors): print(p.get_error(i))
        sys.exit(1)
    cfg = b.create_builder_config()
    cfg.set_flag(trt.BuilderFlag.BF16)
    perfil = b.create_optimization_profile()
    for i in range(rede.num_inputs):
        t = rede.get_input(i)
        forma = tuple(t.shape)
        if t.is_shape_tensor:
            perfil.set_shape_input(t.name, GRADE, GRADE, GRADE)
        perfil.set_shape(t.name, forma, forma, forma)
        print(t.name, forma, t.dtype, "forma" if t.is_shape_tensor else "")
    cfg.add_optimization_profile(perfil)
    motor = b.build_serialized_network(rede, cfg)
    assert motor is not None, "falhou"
    SAIDA.mkdir(exist_ok=True)
    (SAIDA / f"{nome}.plan").write_bytes(motor)
    print(nome, "OK", motor.nbytes >> 20, "MiB", flush=True)

for n in sys.argv[1:] or ["backbone", "action_head"]:
    constroi(n)
