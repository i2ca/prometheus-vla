#!/usr/bin/env python
"""
Espelho da camera do robô (head_camera) em janela OpenCV para o OBS.

Consome o feed ZMQ e mostra numa janela única de título fixed
"SIM_HEAD_CAM" para o OBS capturar via "Captura de Janela (X11)".

Fontes (em ordem de tentativa):
  1. tcp://127.0.0.1:5558  (proxy RGB-only já filtra head_camera -> JPEG puro)
  2. tcp://127.0.0.1:5555  (publisher do sim/robô -> JSON com images.head_camera b64)

Uso:
    python mirror_cam.py            # janela 640x480
    python mirror_cam.py --scale 2  # janela 1280x960 (melhor para OBS)
"""

import argparse
import base64
import datetime
from pathlib import Path
import math
import json
import os
import sys
import time

import cv2
import numpy as np
import zmq

from hud_step import StepHud


def drain(sock):
    """Descarrega frames atrasados mantendo só o mais recente."""
    frames = sock.recv_multipart()
    while True:
        try:
            frames = sock.recv_multipart(zmq.NOBLOCK)
        except zmq.Again:
            break
    return frames


def decode_frame(frames):
    if not frames:
        return None
    first = frames[0]
    payload = first if isinstance(first, (bytes, bytearray)) else first.decode("utf-8", "ignore")
    try:
        data = json.loads(payload)
        head = data.get("images", data).get("head_camera")
        if isinstance(head, dict) and head.get("part") is not None:
            part = head["part"]
            if part < len(frames):
                raw = frames[part]
            else:
                raw = None
        elif isinstance(head, str):
            raw = base64.b64decode(head)
        else:
            raw = None
    except (ValueError, json.JSONDecodeError):
        # não é JSON -> assume JPEG bruto (caso 5558)
        raw = payload
    if not raw:
        return None
    return cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)


def main() -> int:
    global USE_XSHM
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", type=float, default=1.0, help="multiplica o tamanho da janela (default 1.0)")
    parser.add_argument("--window", default="SIM_HEAD_CAM", help="titulo da janela (default SIM_HEAD_CAM)")
    parser.add_argument("--fps", type=float, default=30.0, help="taxa alvo de exibicao (default 30)")
    parser.add_argument("--roi", default=None, help="recorte central 'WxH' em pixels (ex: 400x300)")
    parser.add_argument("--record", nargs="?", const="", default=None, metavar="VIDEO.mp4",
                        help="grava a câmera sem HUD em MP4 (default: recordings/sim_head_DATA.mp4)")
    args = parser.parse_args()

    if not math.isfinite(args.scale) or args.scale <= 0:
        parser.error("--scale deve ser > 0 e finito")
    if not math.isfinite(args.fps) or args.fps <= 0:
        parser.error("--fps deve ser > 0 e finito")
    record_path = None
    if args.record is not None:
        record_path = Path(args.record) if args.record else Path("recordings") / (
            "sim_head_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".mp4")
        if record_path.suffix.lower() != ".mp4":
            parser.error("o arquivo de gravação deve ter extensão .mp4")
        if record_path.exists():
            parser.error(f"o arquivo já existe: {record_path}")
        record_path.parent.mkdir(parents=True, exist_ok=True)

    ctx = zmq.Context.instance()
    endpoints = ["tcp://127.0.0.1:5558", "tcp://127.0.0.1:5555"]
    sock = None
    for ep in endpoints:
        s = ctx.socket(zmq.SUB)
        s.setsockopt(zmq.RCVHWM, 3)
        s.setsockopt(zmq.LINGER, 0)
        s.setsockopt(zmq.RCVTIMEO, 3000)
        s.setsockopt_string(zmq.SUBSCRIBE, "")
        s.connect(ep)
        # ping: garante um frame antes de confiar
        try:
            frames = s.recv_multipart()
            if frames:
                print(f"[mirror] fonte ativa: {ep}", flush=True)
                sock = s
                break
        except zmq.Again:
            s.close()
            continue
    if sock is None:
        print("[mirror] nenhuma fonte de video ativa (5558/5555). Tentando mesmo assim em 5558...", flush=True)
        sock = ctx.socket(zmq.SUB)
        sock.setsockopt(zmq.RCVHWM, 3)
        sock.setsockopt(zmq.LINGER, 0)
        sock.setsockopt(zmq.RCVTIMEO, 3000)
        sock.setsockopt_string(zmq.SUBSCRIBE, "")
        sock.connect(endpoints[0])

    roi = None
    if args.roi:
        w, _, h = args.roi.lower().partition("x")
        if w and h:
            roi = (int(w), int(h))

    cv2.namedWindow(args.window, cv2.WINDOW_NORMAL)
    period = 1.0 / args.fps
    rx = 0
    shown = 0
    last_log = time.time()
    _hud = StepHud() if record_path is None else None
    writer = None
    record_size = None
    recorded = 0

    try:
        while True:
            t0 = time.perf_counter()
            try:
                frames = drain(sock)
            except zmq.Again:
                frames = None
            except Exception as e:
                print(f"[mirror] erro recv: {e}", flush=True)
                time.sleep(0.2)
                continue

            if frames:
                rx += 1
                img = decode_frame(frames)
                if img is not None:
                    if record_path is not None:
                        if writer is None:
                            record_size = (img.shape[1], img.shape[0])
                            if any(size % 2 for size in record_size):
                                raise RuntimeError("MP4 requer dimensões pares para preservar o frame inteiro")
                            writer = cv2.VideoWriter(str(record_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                                     args.fps, record_size)
                            if not writer.isOpened():
                                raise RuntimeError(f"Não foi possível abrir o MP4: {record_path}")
                            print(f"[record] gravando: {record_path.resolve()} @ {args.fps} fps", flush=True)
                        if (img.shape[1], img.shape[0]) != record_size:
                            raise RuntimeError("A resolução da câmera mudou durante a gravação")
                        writer.write(img)
                        recorded += 1
                    if roi:
                        h, w = img.shape[:2]
                        cw, ch = roi
                        x0 = max(0, (w - cw) // 2)
                        y0 = max(0, (h - ch) // 2)
                        img = img[y0:y0 + ch, x0:x0 + cw]
                    if args.scale != 1.0:
                        img = cv2.resize(img, None, fx=args.scale, fy=args.scale,
                                         interpolation=cv2.INTER_AREA)
                    if _hud is not None:
                        img = _hud.draw(img)
                    cv2.imshow(args.window, img)
                    shown += 1

            now = time.time()
            if now - last_log >= 5:
                print(f"[mirror] rx_frames={rx} shown={shown} @ {args.window}", flush=True)
                rx = shown = 0
                last_log = now

            if cv2.waitKey(1) & 0xFF == 27:  # ESC sai
                break
            time.sleep(max(0.0, period - (time.perf_counter() - t0)))
    except KeyboardInterrupt:
        pass
    finally:
        if writer is not None:
            writer.release()
            print(f"[record] salvo: {record_path.resolve()} ({recorded} frames)", flush=True)
        cv2.destroyAllWindows()
        sock.close()
        ctx.term()
    return 0


if __name__ == "__main__":
    main()