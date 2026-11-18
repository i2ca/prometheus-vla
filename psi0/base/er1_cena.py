"""Percepção da cena pelo UnifoLM-ER-1 (servidor_er1.py, :8796): aponta cada objeto na câmera da cabeça e a profundidade da mesma câmera vira xyz.

Roda em processo próprio (contexto EGL isolado). Só usa imagem e profundidade da câmera; a verdade do simulador entra apenas no
campo "_auditoria" (para medir o erro), nunca no que o orquestrador usa.
Uso: python er1_cena.py <qpos.npy> <afasta> '<json lista de objetos em inglês>' [saida.jpg]  -> JSON no stdout
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
import base64, json, sys, urllib.request
import cv2, mujoco, numpy as np
from rodar_mujoco import CENA, afasta

ER1 = os.getenv("ER1_URL", "http://127.0.0.1:8796")


def aponta(jpg, obj):
    req = urllib.request.Request(ER1 + "/apontar", data=json.dumps({"jpg": base64.b64encode(jpg).decode(), "objeto": obj}).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r: return json.loads(r.read())


def main():
    qpos, dx, objetos = np.load(sys.argv[1]), float(sys.argv[2]), json.loads(sys.argv[3])
    m = mujoco.MjModel.from_xml_path(str(CENA)); d = mujoco.MjData(m); afasta(m, d, dx); d.qpos[:] = qpos; mujoco.mj_forward(m, d)
    r = mujoco.Renderer(m, 480, 640); r.update_scene(d, camera="head_camera"); rgb = r.render()
    r.enable_depth_rendering(); r.update_scene(d, camera="head_camera"); prof = r.render(); r.close()
    c = m.camera("head_camera").id; o, R = d.cam_xpos[c], d.cam_xmat[c].reshape(3, 3)
    f = 0.5 * prof.shape[0] / np.tan(np.radians(m.cam_fovy[c]) / 2)
    jpg = cv2.imencode(".jpg", rgb[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes(); im = rgb[:, :, ::-1].copy(); out = {}
    for obj in objetos:
        a = aponta(jpg, obj)
        if a["u"] is None: out[obj] = {"visto": False, "resposta": a["texto"][:200]}; continue
        u, v = int(round(a["u"])), int(round(a["v"])); u, v = min(max(u, 0), 639), min(max(v, 0), 479)
        jan = prof[max(0, v - 3):v + 4, max(0, u - 3):u + 4]; z = float(np.median(jan[jan > 0]))
        p = o + R @ np.array([(u - 320) / f, -(v - 240) / f, -1.0]) * z   # superfície vista no pixel (fundo interno, no caso da caneca)
        out[obj] = {"visto": True, "pixel": [u, v], "xyz_m": p.round(4).tolist()}
        cv2.circle(im, (u, v), 7, (0, 0, 255), 2); cv2.putText(im, obj[:24], (u + 9, v - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
    if len(sys.argv) > 4: cv2.imwrite(sys.argv[4], im)
    out["_auditoria"] = {"caneca_xyz_verdade": d.xpos[m.body("copo").id].round(4).tolist(), "coador_xyz_verdade": d.xpos[m.body("coador").id].round(4).tolist()}
    print(json.dumps(out))


if __name__ == "__main__":
    main()
