"""Servidor do nvidia/GR00T-N1.7-ApplePnP-V1 (export LEAPP em ONNX) no protocolo do GR00T.

    ~/DEV/groot-n17-onnx/.venv/bin/python servidor_n17.py [--porta 5555]

O venv (tensorrt-cu13-bindings 10.13.3.9 + onnxruntime + cuda-python + pyzmq +
msgpack + opencv) e os motores: ver README.md desta pasta.

Fala o mesmo ZMQ + msgpack do `PolicyServer` do Isaac-GR00T, no formato do
`Gr00tSimPolicyWrapper` (chaves planas `video.ego_view`, `state.left_arm`...,
resposta `action.left_arm` (B, T, D)). Então o `roda_groot_interativo.py` da
cabine usa este modelo sem mudar nada, só com --acoes-por-consulta 16.

O grafo do LEAPP (exported_leapp.yaml):
    preprocess_video (ONNX, CPU)  ego_view float [1,480,640,3] 0..255 -> tokens + pixels
    preprocess_state (ONNX, CPU)  7 grupos de junta -> state [1,1,132] normalizado
    backbone         (TensorRT)   Cosmos-Reason2-2B -> features [1,101,2048]
    action_head      (TensorRT)   DiT, 4 passos de flow matching -> [1,40,132]
    decode_action    (ONNX, CPU)  desnormaliza, braço relativo -> absoluto; 16 passos
A frase está gravada no preprocess_video (input_ids fixos): o modelo só sabe a
tarefa da maçã, e o texto que vier da cabine é ignorado.
"""
from __future__ import annotations

import argparse
import io
import os
import time
from pathlib import Path

import cv2
import msgpack
import numpy as np
import onnxruntime as ort
import tensorrt_bindings as trt
import zmq
from cuda.bindings import runtime as rt

D = Path.home() / ".cache/huggingface/hub/models--nvidia--GR00T-N1.7-ApplePnP-V1/snapshots/1c956f1f622cc6496f120efff43012dddeb2ff5e"
# Os motores (.plan, ~3 GB cada) ficam FORA do repositório; GROOT_N17_MOTORES muda o lugar.
MOTORES = Path(os.environ.get("GROOT_N17_MOTORES", Path.home() / "DEV/groot-n17-onnx/engines"))
GRUPOS_ESTADO = ["left_leg", "right_leg", "waist", "left_arm", "right_arm", "left_hand", "right_hand"]
ACOES = ["left_arm", "right_arm", "left_hand", "right_hand", "waist", "navigate_command",
         "base_height_command"]


def ok(r):
    erro = r[0] if isinstance(r, tuple) else r
    assert erro == rt.cudaError_t.cudaSuccess, erro
    return r[1] if isinstance(r, tuple) and len(r) == 2 else r


class MotorTRT:
    """Um motor TensorRT com buffers fixos na GPU (todas as formas são estáticas)."""

    def __init__(self, arquivo: Path, grade=(1, 16, 22)):
        self.log = trt.Logger(trt.Logger.WARNING)
        self.motor = trt.Runtime(self.log).deserialize_cuda_engine(arquivo.read_bytes())
        self.ctx = self.motor.create_execution_context()
        self.fluxo = ok(rt.cudaStreamCreate())
        self.buf, self.forma, self.tipo, self.entradas = {}, {}, {}, []
        for i in range(self.motor.num_io_tensors):
            n = self.motor.get_tensor_name(i)
            eh_entrada = self.motor.get_tensor_mode(n) == trt.TensorIOMode.INPUT
            if eh_entrada and self.motor.is_shape_inference_io(n):
                # tensor de forma: o valor fica na CPU
                self.grade = np.array(grade, dtype=np.int64)
                self.ctx.set_tensor_address(n, self.grade.ctypes.data)
                continue
            forma = tuple(self.motor.get_tensor_shape(n))
            tipo = np.dtype(trt.nptype(self.motor.get_tensor_dtype(n)))
            self.forma[n], self.tipo[n] = forma, tipo
            self.buf[n] = ok(rt.cudaMalloc(int(np.prod(forma)) * tipo.itemsize))
            self.ctx.set_tensor_address(n, self.buf[n])
            if eh_entrada:
                self.entradas.append(n)

    def roda(self, entradas: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        cpy = rt.cudaMemcpyKind
        for n in self.entradas:
            a = np.ascontiguousarray(entradas[n], dtype=self.tipo[n]).reshape(self.forma[n])
            ok(rt.cudaMemcpyAsync(self.buf[n], a.ctypes.data, a.nbytes, cpy.cudaMemcpyHostToDevice, self.fluxo))
        assert self.ctx.execute_async_v3(self.fluxo)
        saidas = {}
        for n in self.buf:
            if n in self.entradas:
                continue
            a = np.empty(self.forma[n], self.tipo[n])
            ok(rt.cudaMemcpyAsync(a.ctypes.data, self.buf[n], a.nbytes, cpy.cudaMemcpyDeviceToHost, self.fluxo))
            saidas[n] = a
        ok(rt.cudaStreamSynchronize(self.fluxo))
        return saidas


class PoliticaN17:
    def __init__(self, semente: int = 0):
        cpu = ["CPUExecutionProvider"]
        self.video = ort.InferenceSession(str(D / "preprocess_video.onnx"), providers=cpu)
        self.estado = ort.InferenceSession(str(D / "preprocess_state.onnx"), providers=cpu)
        self.decodifica = ort.InferenceSession(str(D / "decode_action.onnx"), providers=cpu)
        t = time.time()
        self.backbone = MotorTRT(MOTORES / "backbone.plan")
        self.cabeca = MotorTRT(MOTORES / "action_head.plan")
        print(f"motores TensorRT carregados em {time.time() - t:.0f} s", flush=True)
        self.rng = np.random.default_rng(semente)

    def reset(self, options=None):
        return {}

    def get_action(self, observation: dict, options=None):
        img = np.asarray(observation["video.ego_view"])[0, -1]          # (H, W, 3) uint8
        if img.shape[:2] != (480, 640):
            img = cv2.resize(img, (640, 480), interpolation=cv2.INTER_LINEAR)
        t0 = time.perf_counter()
        pv, ids, mask, grade, emb = self.video.run(None, {"ego_view": img[None].astype(np.float32)})
        est = {g: np.asarray(observation[f"state.{g}"], np.float32)[0, -1][None] for g in GRUPOS_ESTADO}
        saida_est = dict(zip([o.name for o in self.estado.get_outputs()], self.estado.run(None, est)))
        bb = self.backbone.roda({"vl_input_input_ids": ids, "vl_input_attention_mask": mask,
                                 "vl_input_pixel_values": pv})
        t1 = time.perf_counter()
        ah = self.cabeca.roda({
            "backbone_outputs_backbone_features": bb["converted_outputs_backbone_features"],
            "backbone_outputs_backbone_attention_mask": bb["converted_outputs_backbone_attention_mask"],
            "backbone_outputs_image_mask": bb["converted_outputs_image_mask"],
            "action_inputs_state": saida_est["state"],
            "action_inputs_embodiment_id": emb,
            "action_inputs_input_ids": ids, "action_inputs_attention_mask": mask,
            "action_inputs_pixel_values": pv, "action_inputs_image_grid_thw": grade,
            "initial_noise": self.rng.standard_normal((1, 40, 132)).astype(np.float32),
        })
        t2 = time.perf_counter()
        dec = self.decodifica.run(None, {
            "normalized_action": ah["output1_action_pred"],
            "state_0_left_arm": saida_est["reference_0_left_arm"],
            "state_0_right_arm": saida_est["reference_0_right_arm"]})
        dec = dict(zip([o.name for o in self.decodifica.get_outputs()], dec))
        t3 = time.perf_counter()
        print(f"backbone+pré {1000 * (t1 - t0):.0f} ms · cabeça {1000 * (t2 - t1):.0f} ms · "
              f"decodifica {1000 * (t3 - t2):.0f} ms · nav {dec['navigate_command'][0, 0].round(2)}",
              flush=True)
        return {f"action.{k}": dec[k].astype(np.float32) for k in ACOES}, {}


# ---- protocolo do PolicyServer do Isaac-GR00T (gr00t/policy/server_client.py) ----
def decodifica(obj):
    if isinstance(obj, dict) and "__ndarray_class__" in obj:
        return np.load(io.BytesIO(obj["as_npy"]), allow_pickle=False)
    return obj


def codifica(obj):
    if isinstance(obj, np.ndarray):
        b = io.BytesIO()
        np.save(b, obj, allow_pickle=False)
        return {"__ndarray_class__": True, "as_npy": b.getvalue()}
    return obj


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--porta", type=int, default=5555)
    args = p.parse_args()
    pol = PoliticaN17()
    sock = zmq.Context().socket(zmq.REP)
    sock.bind(f"tcp://*:{args.porta}")
    print(f"GR00T-N1.7-ApplePnP-V1 ouvindo em tcp://*:{args.porta}", flush=True)
    while True:
        pedido = msgpack.unpackb(sock.recv(), object_hook=decodifica)
        ponto = pedido.get("endpoint", "get_action")
        try:
            if ponto == "ping":
                r = {"status": "ok"}
            elif ponto == "reset":
                r = pol.reset()
            elif ponto == "get_action":
                d = pedido["data"]
                r = list(pol.get_action(d["observation"], d.get("options")))
            else:
                r = {"error": f"endpoint desconhecido: {ponto}"}
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            r = {"error": str(e)}
        sock.send(msgpack.packb(r, default=codifica))


if __name__ == "__main__":
    main()
