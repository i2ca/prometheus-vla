#!/usr/bin/env python
"""
Ponte ZMQ → HTTP: o cliente do FastWAM-D falando com o servidor do WMA
=======================================================================
O `real_eval_server.py` da Unitree é FastAPI: um POST em `/predict_action`
com JSON. O nosso `init_lerobot_inference_fastwamd_client.py` é ZMQ REQ/REP
com msgpack, e tem dentro dele tudo o que custou caro para funcionar — o
laço de controle, a leitura das câmeras, o mosaico de depuração, a fila de
ações, o tratamento de quadro velho.

Reescrever o cliente para HTTP seria jogar isso fora para ganhar nada. Esta
ponte fala ZMQ do lado de cá (idêntico ao `init_lerobot_inference_fastwamd_server.py`)
e HTTP do lado de lá, e o cliente não sabe que trocou de modelo:

    cliente (seu PC) ──ZMQ 5600──> ponte ──HTTP 8000──> real_eval_server (athena)

Rode a ponte NA ATHENA, junto do servidor WMA. Pôr a ponte do lado do PC
mandaria a imagem duas vezes pela rede.

O QUE A PONTE TRADUZ

  * As 29 juntas soltas do `obs` viram um vetor de 29, na ordem do
    `JUNTAS_G1` — a MESMA ordem do dataset. Ordem trocada aqui não dá erro:
    dá robô se movendo errado.
  * `head_camera` vira `observation.images.top` em CHW, redimensionada para
    320x512. O `spatial_transform` do WMA faria isso de qualquer jeito; fazer
    antes tira ~5 MB de cada POST.
  * `action` vai zerada porque o servidor só a usa para descobrir a
    dimensão e montar a máscara (`_map_to_uni_action`).

O QUE A PONTE NÃO FAZ

  * **Profundidade e tato não atravessam.** O WMA-0 não tem entrada para
    eles. O cliente continua mandando; a ponte descarta e diz quantas vezes.
  * **`want_debug` não tem resposta.** O painel de atenção do FastWAM-D lê
    tensores internos do DiT do Wan, que não existem aqui. O cliente aguenta
    a ausência da chave `debug`; o painel fica com os quadrantes vazios.

USO

    python pontes/wma/ponte_wma.py --wma=http://127.0.0.1:8000 --port=5600
"""

import argparse
import sys
import time

try:
    import zmq
    import msgpack
    import msgpack_numpy as m
    m.patch()
except ImportError:
    print("❌ Instale as dependências: pip install pyzmq msgpack msgpack-numpy")
    sys.exit(1)

import numpy as np
import requests

# A ordem é a do dataset e a do cliente. Se você mudar o schema (SCHEMA_G1_V2),
# mude nos três lugares no mesmo commit.
JUNTAS_G1 = [
    "kLeftShoulderPitch.q", "kLeftShoulderRoll.q", "kLeftShoulderYaw.q",
    "kLeftElbow.q", "kLeftWristRoll.q", "kLeftWristPitch.q", "kLeftWristyaw.q",
    "kRightShoulderPitch.q", "kRightShoulderRoll.q", "kRightShoulderYaw.q",
    "kRightElbow.q", "kRightWristRoll.q", "kRightWristPitch.q", "kRightWristYaw.q",
    "kWaistYaw.q",
    "left_hand_thumb_0_joint.q", "left_hand_thumb_1_joint.q", "left_hand_thumb_2_joint.q",
    "left_hand_middle_0_joint.q", "left_hand_middle_1_joint.q",
    "left_hand_index_0_joint.q", "left_hand_index_1_joint.q",
    "right_hand_thumb_0_joint.q", "right_hand_thumb_1_joint.q", "right_hand_thumb_2_joint.q",
    "right_hand_index_0_joint.q", "right_hand_index_1_joint.q",
    "right_hand_middle_0_joint.q", "right_hand_middle_1_joint.q",
]

ALTURA, LARGURA = 320, 512


def _pack(data: dict) -> bytes:
    return msgpack.packb(data, default=m.encode)


def _unpack(raw: bytes) -> dict:
    return msgpack.unpackb(raw, object_hook=m.decode, raw=False)


def prepara_imagem(quadro: np.ndarray) -> list:
    """HWC uint8 (o que o cliente manda) → CHW 320x512 (o que o WMA espera).

    CHW e não HWC porque o `spatial_transform` do servidor é um
    `transforms.Compose` do torchvision, que opera em `[..., H, W]`. Mandar
    HWC não estoura: ele redimensiona a dimensão errada e o modelo recebe uma
    imagem embaralhada, sem reclamar.
    """
    import cv2
    quadro = np.asarray(quadro)
    if quadro.ndim != 3 or quadro.shape[2] != 3:
        raise ValueError(f"esperava HWC de 3 canais, veio {quadro.shape}")
    if quadro.shape[:2] != (ALTURA, LARGURA):
        quadro = cv2.resize(quadro, (LARGURA, ALTURA), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(quadro.transpose(2, 0, 1)).tolist()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--wma", default="http://127.0.0.1:8000",
                   help="endereço do real_eval_server.py")
    p.add_argument("--port", type=int, default=5600, help="porta ZMQ para o cliente")
    p.add_argument("--task", default="place the white cup on the dripper",
                   help="instrução usada quando o cliente não manda `task`")
    p.add_argument("--timeout", type=float, default=120.0,
                   help="segundos de espera pelo WMA (a difusão de vídeo é lenta)")
    args = p.parse_args()

    ctx = zmq.Context()
    socket = ctx.socket(zmq.REP)
    socket.bind(f"tcp://0.0.0.0:{args.port}")
    print(f"🔌 ponte ZMQ :{args.port}  →  WMA {args.wma}")
    print(f"   Tarefa padrão: {args.task!r}")
    print("   Aguardando o cliente...\n")

    url = args.wma.rstrip("/") + "/predict_action"
    n_req, ms_total, descartadas = 0, 0.0, 0

    while True:
        msg = _unpack(socket.recv())
        obs = msg["obs"]
        obs_step = int(msg.get("obs_step", 0))
        n_acoes = int(msg.get("actions_per_chunk", 16))
        tarefa = msg.get("task") or args.task
        t0 = time.perf_counter()

        try:
            faltando = [j for j in JUNTAS_G1 if j not in obs]
            if faltando:
                # Completar com zero seria mandar o robô para uma pose que ele
                # não está — pior que recusar o pedido.
                raise KeyError(f"{len(faltando)} juntas ausentes na observação, "
                               f"a começar por {faltando[0]!r}")
            estado = np.array([float(obs[j]) for j in JUNTAS_G1], dtype=np.float32)

            if "head_camera" not in obs:
                raise KeyError("a observação não trouxe `head_camera`")
            if any(k in obs for k in ("head_camera_depth", "left_hand_pressure")):
                descartadas += 1

            carga = {
                "observation.images.top": prepara_imagem(obs["head_camera"]),
                "observation.state": estado.tolist(),
                "action": np.zeros_like(estado).tolist(),
                "language_instruction": tarefa,
            }
            resposta = requests.post(url, json=carga, timeout=args.timeout).json()
            if resposta.get("result") != "ok":
                raise RuntimeError(resposta.get("desc", "o WMA respondeu sem `ok`"))

            chunk = np.asarray(resposta["action"], dtype=np.float32)
            if chunk.ndim != 2 or chunk.shape[1] != len(JUNTAS_G1):
                raise ValueError(
                    f"o WMA devolveu {chunk.shape}, esperava (H, {len(JUNTAS_G1)}). "
                    f"Confira agent_action_dim no yaml de inferência.")
            # Pedir mais que o horizonte não gera ação nova: o WMA emite
            # `temporal_length` (16) de cada vez. Mesma regra do servidor do
            # FastWAM-D — recortar em silêncio faria o cliente acreditar num
            # buffer que não existe.
            chunk = chunk[:n_acoes]

            nao_finitos = int((~np.isfinite(chunk)).sum())
            if nao_finitos:
                raise ValueError(f"{nao_finitos} valores não finitos no chunk — descartado")

            infer_ms = (time.perf_counter() - t0) * 1000.0
            socket.send(_pack({
                "chunk_np": chunk,
                "obs_step": obs_step,
                "infer_ms": infer_ms,
                "travadas": 0,
            }))
            n_req += 1
            ms_total += infer_ms
            if n_req % 10 == 0:
                print(f"   {n_req} pedidos | média {ms_total / n_req:.0f} ms"
                      + (f" | {descartadas} obs com depth/tato descartados" if descartadas else ""))

        except Exception as erro:
            print(f"❌ step {obs_step}: {type(erro).__name__}: {erro}")
            socket.send(_pack({"error": f"{type(erro).__name__}: {erro}"}))


if __name__ == "__main__":
    sys.exit(main())
