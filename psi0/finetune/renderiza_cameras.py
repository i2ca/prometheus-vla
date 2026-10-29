"""Renderiza head/left_wrist/right_wrist das demos, uma câmera por processo para isolar contextos EGL.
Uso: python renderiza_cameras.py <pasta com ep*/ ou um ep> [--cam head|left_wrist|right_wrist]  (sem --cam, as três em paralelo)
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
import argparse, json, subprocess, sys
from pathlib import Path
import numpy as np, mujoco
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "base"))
from rodar_mujoco import CENA, afasta

CAMS = ("head", "left_wrist", "right_wrist")


def renderiza(ep, cam):
    tr = json.loads((ep / "trace.json").read_text()); Q = np.load(ep / "qpos.npz")["qpos"]
    m = mujoco.MjModel.from_xml_path(str(CENA)); d = mujoco.MjData(m); afasta(m, d, tr["afasta"])
    r = mujoco.Renderer(m, 480, 640)
    ff = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "640x480", "-r", "30", "-i", "-",
                           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-g", "2", str(ep / f"{cam}.mp4")], stdin=subprocess.PIPE)
    for q in Q:
        d.qpos[:] = q; mujoco.mj_forward(m, d); r.update_scene(d, camera=f"{cam}_camera"); ff.stdin.write(r.render().tobytes())
    ff.stdin.close(); ff.wait(); r.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument("demos"); ap.add_argument("--cam", choices=CAMS); a = ap.parse_args()
    raiz = Path(a.demos); eps = [raiz] if (raiz / "qpos.npz").exists() else sorted(raiz.glob("ep*"))
    if a.cam:
        for ep in eps: renderiza(ep, a.cam)
    else:
        ps = [subprocess.Popen([sys.executable, __file__, a.demos, "--cam", cam]) for cam in CAMS]
        if any(p.wait() for p in ps): raise SystemExit("falha ao renderizar câmeras")

if __name__ == "__main__": main()
