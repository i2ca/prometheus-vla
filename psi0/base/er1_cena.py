"""Percepção da cena pelo UnifoLM-ER-1 (servidor_er1.py, :8796): aponta cada objeto na câmera da cabeça e a profundidade da mesma câmera vira xyz.

Roda em processo próprio (contexto EGL isolado). Só usa imagem e profundidade da câmera; a verdade do simulador entra apenas no
campo "_auditoria" (para medir o erro), nunca no que o orquestrador usa.
Uso: python er1_cena.py <qpos.npy> <afasta> '<json lista de objetos em inglês>' [saida.jpg]  -> JSON no stdout
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
import base64, json, re, sys, urllib.request
import cv2, mujoco, numpy as np
from rodar_mujoco import CENA, afasta

ER1 = os.getenv("ER1_URL", "http://127.0.0.1:8796")


def aponta(jpg, obj, tipo="ponto"):
    req = urllib.request.Request(ER1 + "/apontar", data=json.dumps({"jpg": base64.b64encode(jpg).decode(), "objeto": obj, "tipo": tipo}).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r: return json.loads(r.read())


def centro_do_aro(P):
    """Pontos 3D dentro da caixa da caneca -> centro do aro. Descarta a mesa (abaixo) e o coador (acima de ~11 cm)."""
    P = P[P[:, 2] > P[:, 2].max() - 0.25]   # tira o chão quando a caixa passa da borda da mesa
    zt = np.percentile(P[:, 2], 5); Q = P[(P[:, 2] > zt + 0.02) & (P[:, 2] < zt + 0.115)]
    if len(Q) < 30: return None
    aro = Q[Q[:, 2] >= Q[:, 2].max() - 0.008]
    return np.array([(aro[:, 0].min() + aro[:, 0].max()) / 2, (aro[:, 1].min() + aro[:, 1].max()) / 2, aro[:, 2].mean()])


def localiza(m, d, objetos, saida=None):
    r = mujoco.Renderer(m, 480, 640); r.update_scene(d, camera="head_camera"); rgb = r.render()
    r.enable_depth_rendering(); r.update_scene(d, camera="head_camera"); prof = r.render(); r.close()
    c = m.camera("head_camera").id; o, R = d.cam_xpos[c], d.cam_xmat[c].reshape(3, 3)
    f = 0.5 * prof.shape[0] / np.tan(np.radians(m.cam_fovy[c]) / 2)
    em3d = lambda u, v, z: o + (np.stack([(u - 320) / f, -(v - 240) / f, -np.ones_like(u, dtype=float)], -1) * z[..., None]) @ R.T
    jpg = cv2.imencode(".jpg", rgb[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes(); im = rgb[:, :, ::-1].copy(); out = {}
    VV, UU = np.mgrid[0:480, 0:640]; T = em3d(UU[prof > 0].astype(float), VV[prof > 0].astype(float), prof[prof > 0])
    for obj in objetos:
        # caixa do ER-1 + profundidade -> centro do aro. Um ponto só cai no fundo ou na parede, conforme o ângulo (2-4 cm de erro).
        a = aponta(jpg, obj.replace("center of the ", ""), "caixa")   # "center of" faz o ER-1 devolver só o miolo
        if not a["caixa"]:   # às vezes o ER-1 devolve "bbox_2d" com só um ponto (x, y): vira caixa de 80 px em volta
            pt = re.search(r'\[\s*([\d.]+)\s*,\s*([\d.]+)\s*\]', a["texto"])
            if not pt: out[obj] = {"visto": False, "resposta": a["texto"][:200]}; continue
            cu, cv_ = float(pt.group(1)) * 0.64, float(pt.group(2)) * 0.48; a["caixa"] = [cu - 40, cv_ - 40, cu + 40, cv_ + 40]
        x1, y1, x2, y2 = a["caixa"]; mx, my = 0.15 * (x2 - x1), 0.15 * (y2 - y1)   # a caixa costuma cortar a borda do aro
        x1, y1, x2, y2 = int(max(x1 - mx, 0)), int(max(y1 - my, 0)), int(min(x2 + mx, 639)), int(min(y2 + my, 479))
        vv, uu = np.mgrid[y1:y2 + 1, x1:x2 + 1]; z = prof[vv, uu]; ok = z > 0
        P = em3d(uu[ok].astype(float), vv[ok].astype(float), z[ok])
        if "mug" in obj: p = centro_do_aro(P)   # caneca: aro mais alto abaixo de ~11 cm da mesa (o coador fica acima)
        else: top = P[P[:, 2] >= P[:, 2].max() - 0.008]; p = np.r_[top[:, :2].mean(0), top[:, 2].mean()]   # coador: o aro é o ponto mais alto
        for _ in range(2):   # a caixa só semeia: o aro inteiro vem da imagem toda, na mesma altura e a até 6 cm
            if p is None: break
            aro = T[(np.abs(T[:, 2] - p[2]) < 0.006) & (np.linalg.norm(T[:, :2] - p[:2], axis=1) < 0.06)]
            if len(aro) < 30: break
            p = np.array([(aro[:, 0].min() + aro[:, 0].max()) / 2, (aro[:, 1].min() + aro[:, 1].max()) / 2, aro[:, 2].mean()])
        if p is None: out[obj] = {"visto": False, "caixa": [x1, y1, x2, y2], "resposta": a["texto"][:200]}; continue
        out[obj] = {"visto": True, "caixa": [x1, y1, x2, y2], "xyz_m": p.round(4).tolist()}
        cv2.rectangle(im, (x1, y1), (x2, y2), (0, 0, 255), 2); cv2.putText(im, obj[:24], (x1, max(y1 - 6, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
    if saida: cv2.imwrite(saida, im)
    out["_auditoria"] = {"caneca_xyz_verdade": d.xpos[m.body("copo").id].round(4).tolist(), "coador_xyz_verdade": d.xpos[m.body("coador").id].round(4).tolist()}
    return out


def main():
    qpos, dx, objetos = np.load(sys.argv[1]), float(sys.argv[2]), json.loads(sys.argv[3])
    m = mujoco.MjModel.from_xml_path(str(CENA)); d = mujoco.MjData(m); afasta(m, d, dx); d.qpos[:] = qpos; mujoco.mj_forward(m, d)
    print(json.dumps(localiza(m, d, objetos, sys.argv[4] if len(sys.argv) > 4 else None)))


if __name__ == "__main__":
    main()
