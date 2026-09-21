#!/usr/bin/env python
"""
Feed RGB-only (head_camera) do robô para o Vuer/Quest via ZMQ.

O stream do robô em <robot_ip>:5555 é multipart: frame[0] = JSON com
`images.head_camera` como base64 JPEG (ou `{part,...}`), frames extras
carregam o depth cru. Este proxy filtra só a `head_camera` e republica
o JPEG puro (um frame por mensagem) em tcp://127.0.0.1:5558, que é a
porta que o `XRG1Arm` (teleop) consome via G1_VR_CAM_PORT.

Uso:
    python vrrgb_proxy.py --src-ip 192.168.68.71 --src-port 5555

Variáveis de ambiente (compatíveis com a stack):
    G1_VR_SRC_IP  / G1_VR_SRC_PORT  (origem; default 127.0.0.1:5555)
    G1_VR_CAM_PORT                  (destino; default 5558)
"""

import argparse
import base64
import json
import os
import sys
import time

import zmq


def drain(sock):
    """Descarrega frames atrasados, mantendo só o mais recente."""
    frames = sock.recv_multipart()
    while True:
        try:
            frames = sock.recv_multipart(zmq.NOBLOCK)
        except zmq.Again:
            break
    return frames


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src-ip", default=None, help="IP do robô (default: env G1_VR_SRC_IP ou 127.0.0.1)")
    parser.add_argument("--src-port", type=int, default=None, help="Porta ZMQ do robô (default: env G1_VR_SRC_PORT ou 5555)")
    args = parser.parse_args()

    src_ip = args.src_ip or os.environ.get("G1_VR_SRC_IP", "127.0.0.1")
    src_port = args.src_port or int(os.environ.get("G1_VR_SRC_PORT", "5555"))
    dst_port = int(os.environ.get("G1_VR_CAM_PORT", "5558"))

    ctx = zmq.Context.instance()

    f = ctx.socket(zmq.SUB)
    f.setsockopt(zmq.RCVHWM, 3)
    f.setsockopt(zmq.LINGER, 0)
    f.setsockopt(zmq.RCVTIMEO, 3000)
    f.setsockopt_string(zmq.SUBSCRIBE, "")
    f.connect(f"tcp://{src_ip}:{src_port}")

    b = ctx.socket(zmq.PUB)
    b.setsockopt(zmq.SNDHWM, 1)
    b.setsockopt(zmq.LINGER, 0)
    b.bind(f"tcp://127.0.0.1:{dst_port}")

    print(f"[VR RGB Proxy] {src_ip}:{src_port} -> 127.0.0.1:{dst_port} (jpeg)", flush=True)

    rx = 0
    tx = 0
    last_log = time.time()

    while True:
        try:
            frames = drain(f)
        except zmq.Again:
            time.sleep(0.05)
            continue
        except Exception as e:
            print(f"[VR RGB Proxy] erro de recv: {e}", flush=True)
            time.sleep(0.2)
            continue

        if not frames:
            continue

        rx += 1

        try:
            data = json.loads(frames[0].decode("utf-8"))
            head = data.get("images", data).get("head_camera")
            if isinstance(head, dict) and head.get("part") is not None:
                part = head["part"]
                if part < len(frames):
                    b.send(frames[part], flags=zmq.NOBLOCK)
                    tx += 1
            elif isinstance(head, str):
                b.send(base64.b64decode(head), flags=zmq.NOBLOCK)
                tx += 1
        except zmq.Again:
            continue
        except Exception:
            continue

        now = time.time()
        if now - last_log >= 5:
            print(f"[VR RGB Proxy] rx={rx} tx={tx}", flush=True)
            last_log = now


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n[VR RGB Proxy] encerrado.", flush=True)