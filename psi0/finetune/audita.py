"""Auditoria física das rodadas de avaliação (pastas com qpos.npz + trace.json de ../base/rodar_mujoco.py), quadro a quadro:
interpenetração mão-caneca, quadros com a mão direita na mesa, deslocamento de chaleira/pote/coador, subida e inclinação da
caneca, e se a caneca está dentro do quadro da câmera da cabeça no 1o quadro (centro projetado, margem do raio).
Sucesso estrito exige também polegar, médio e indicador na caneca em > 90% dos quadros com ela no ar (como nas demos):
com só 2 dedos a pega parece escapar (revisão de 08/10).
Uso: python audita.py <pasta com ep*/> """
import sys, json, glob
from pathlib import Path
import numpy as np, mujoco
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "base"))
from rodar_mujoco import CENA, D435_XYZ, D435_RPY, rpy_to_R, afasta


def visivel(m, d, cam, ponto, W=640, H=480):
    """Projeta ponto do mundo na câmera (fovy vertical) e devolve (u, v) em pixels."""
    p = d.cam_xpos[cam]; R = d.cam_xmat[cam].reshape(3, 3); x = R.T @ (ponto - p)   # câmera MuJoCo olha para -z
    f = (H / 2) / np.tan(np.radians(m.cam_fovy[cam]) / 2)
    return W / 2 + f * x[0] / -x[2], H / 2 - f * x[1] / -x[2]


def main():
    res, mira = [], []
    for ep in sorted(glob.glob(sys.argv[1] + "/ep*")):
        tr = json.load(open(ep + "/trace.json")); Q = np.load(ep + "/qpos.npz")["qpos"]
        m = mujoco.MjModel.from_xml_path(str(CENA)); d = mujoco.MjData(m); afasta(m, d, tr["afasta"])
        hc = m.camera("head_camera").id; Rc = rpy_to_R(*D435_RPY) @ np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]], float)
        q = np.zeros(4); mujoco.mju_mat2Quat(q, Rc.ravel()); m.cam_pos[hc] = D435_XYZ; m.cam_quat[hc] = q; m.cam_fovy[hc] = 42.5
        copo, mesa = m.body("copo").id, m.body("mesa").id; mao = {b for b in range(m.nbody) if m.body(b).name.startswith("right_hand")}
        outros = {n: m.body(n).id for n in ("chaleira", "pote", "coador")}
        pen, na_mesa, ini, sub, inc, ded, tres = 0.0, 0, {}, [], [], [], []
        pontas = {m.body(f"right_hand_{n}_link").id for n in ("thumb_2", "middle_1", "index_1")}
        for k in range(len(Q)):
            d.qpos[:] = Q[k]; mujoco.mj_forward(m, d)
            if k == 0:
                ini = {n: d.xpos[b].copy() for n, b in outros.items()}; z0 = d.xpos[copo][2]
                u, v = visivel(m, d, hc, d.xpos[copo] + [0, 0, 0.045])
            n = set(); toca_mesa = False
            for c in d.contact[:d.ncon]:
                a, b = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
                if copo in (a, b) and ({a, b} & mao): pen = min(pen, c.dist); n |= {a, b} & mao
                if mesa in (a, b) and ({a, b} & mao): toca_mesa = True
            tres.append(pontas <= n)
            na_mesa += toca_mesa; sub.append(1000 * (d.xpos[copo][2] - z0)); ded.append(len(n))
            inc.append(np.degrees(np.arccos(np.clip(d.xmat[copo][8], -1, 1))))
        z = np.load(ep + "/qpos.npz"); D = z["dedos"]; k = int(np.argmax(D >= 2)) if (D >= 2).any() else len(Q) - 1
        d.qpos[:] = Q[k]; mujoco.mj_kinematics(m, d); pk = d.xpos[m.body("right_wrist_yaw_link").id][:2].copy()
        d.qpos[:] = Q[0]; mujoco.mj_kinematics(m, d); mira.append((d.xpos[copo][:2].copy(), pk))
        mexeu = max(1000 * np.linalg.norm(d.xpos[b] - ini[nm]) for nm, b in outros.items())
        r = dict(ep=Path(ep).name, copo=tr["copo"], penetracao_max_mm=round(-1000 * pen, 1), quadros_mao_na_mesa=na_mesa,
                 outros_mexeram_mm=round(mexeu, 1), subida_final_mm=round(sub[-1], 0), dedos_final=ded[-1], inclinacao_max_deg=round(max(inc), 1),
                 caneca_no_quadro0_px=[round(u), round(v)], caneca_inteira_no_quadro0=bool(40 < u < 600 and 40 < v < 440))
        no_ar = [t_ for t_, s_ in zip(tres, sub) if s_ > 50]
        r["tres_dedos_no_ar"] = round(float(np.mean(no_ar)), 2) if no_ar else 0.0
        r["sucesso_estrito"] = bool(r["tres_dedos_no_ar"] > 0.9 and r["subida_final_mm"] > 50 and r["dedos_final"] >= 2 and r["inclinacao_max_deg"] < 15
                                    and r["penetracao_max_mm"] < 3 and r["quadros_mao_na_mesa"] == 0 and r["outros_mexeram_mm"] < 10)
        res.append(r); print(json.dumps(r), flush=True)
    print(f"sucesso estrito {sum(r['sucesso_estrito'] for r in res)}/{len(res)}")
    C, P = np.array([c for c, _ in mira]), np.array([p for _, p in mira])   # punho no 1o fecho contra a caneca: inclinação 0 = movimento fixo
    for i, n in enumerate("xy"):
        if np.ptp(C[:, i]) > 0.01: print(f"mira {n}: punho = {np.polyfit(C[:, i], P[:, i], 1)[0]:.2f} x caneca, corr {np.corrcoef(C[:, i], P[:, i])[0, 1]:.2f}")
    Path(sys.argv[1], "auditoria.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
