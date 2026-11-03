"""Vídeo em tempo real (30 fps) de uma rodada do rodar_mujoco.py, a partir do qpos.npz gravado: sem as pausas em que o
simulador espera o modelo. Uso: MUJOCO_GL=egl python video.py saida/v24_0311 [--camera global_view]  ->  saida/v24_0311/video.mp4
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
import argparse, json
import numpy as np, mujoco, cv2
from rodar_mujoco import CENA, afasta

ap = argparse.ArgumentParser(); ap.add_argument("pasta"); ap.add_argument("--camera", default="global_view"); a = ap.parse_args()
Q = np.load(f"{a.pasta}/qpos.npz")["qpos"]; dx = json.load(open(f"{a.pasta}/trace.json"))["afasta"]
m = mujoco.MjModel.from_xml_path(str(CENA)); d = mujoco.MjData(m); afasta(m, d, dx); rend = mujoco.Renderer(m, 720, 960)
out = cv2.VideoWriter(f"{a.pasta}/video.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 30, (960, 720))
for q in Q:
    d.qpos[:] = q; mujoco.mj_forward(m, d); rend.update_scene(d, camera=a.camera); out.write(rend.render()[:, :, ::-1])
out.release(); print(f"{a.pasta}/video.mp4: {len(Q)} quadros, {len(Q) / 30:.1f} s")
