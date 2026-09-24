#!/usr/bin/env python
"""
O robô está indo até a xícara, ou só se mexendo?
=================================================
Lê o log `EXECUTED` do cliente, aplica cinemática direta no braço direito e
mede a distância da mão à xícara quadro a quadro. Compara com um episódio do
dataset, que é como a resposta certa se parece.

"Erro médio de 2° por junta" não responde a pergunta: 2° no ombro deslocam a
mão vários centímetros, e um erro pequeno espalhado pode ser um braço que
oscila em torno do lugar errado. Distância até o alvo responde.

Uso:
  python analisa_trajetoria.py logs/fastwamd_*.txt
"""
import sys
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))

import mujoco  # noqa: E402

CENA = AQUI.parent / "unitree-g1-mujoco" / "assets" / "scene_43dof.xml"

# As 14 juntas de braço na ordem do log (= ordem do dataset real).
JUNTAS_LOG = [
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]


def le_log(caminho):
    passos = []
    for linha in Path(caminho).read_text().splitlines():
        if not linha.startswith("EXECUTED"):
            continue
        p = linha.split("\t")
        passos.append(np.array([float(x) for x in p[3:32]], dtype=float))
    return np.array(passos)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(1)

    q = le_log(sys.argv[1])
    print(f"{len(q)} passos executados no log\n")

    m = mujoco.MjModel.from_xml_path(str(CENA))
    d = mujoco.MjData(m)

    adr = {}
    for i, nome in enumerate(JUNTAS_LOG):
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, nome)
        if jid >= 0:
            adr[i] = m.jnt_qposadr[jid]
    jw = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "waist_yaw_joint")
    adr_cintura = m.jnt_qposadr[jw]
    id_punho = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_yaw_link")

    # Onde a xícara está na cena do cliente (posição do XML, sem sorteio).
    gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "colisao_copo")
    mujoco.mj_forward(m, d)
    copo = d.geom_xpos[gid].copy()
    b_marca = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "marca_alvo")
    marca = d.xpos[b_marca].copy() if b_marca >= 0 else None
    print(f"xícara na cena: ({copo[0]:.3f}, {copo[1]:.3f}, {copo[2]:.3f})")
    if marca is not None:
        print(f"X (marca_alvo): ({marca[0]:.3f}, {marca[1]:.3f}, {marca[2]:.3f})")
    print()

    dists, punhos = [], []
    for passo in q:
        for i, a in adr.items():
            d.qpos[a] = passo[i]
        d.qpos[adr_cintura] = passo[14]
        mujoco.mj_forward(m, d)
        p = d.xpos[id_punho].copy()
        punhos.append(p)
        dists.append(float(np.linalg.norm(p - copo)))

    dists = np.array(dists)
    punhos = np.array(punhos)

    n = len(dists)
    for rot, fatia in [("primeiros 10%", slice(0, max(1, n // 10))),
                       ("meio", slice(n // 2 - n // 20, n // 2 + n // 20)),
                       ("últimos 10%", slice(n - max(1, n // 10), n))]:
        s = dists[fatia]
        print(f"  {rot:>14}: distância média {s.mean():.3f} m  (min {s.min():.3f})")

    print(f"\n  distância inicial : {dists[0]:.3f} m")
    print(f"  distância final   : {dists[-1]:.3f} m")
    print(f"  MAIS PERTO que chegou: {dists.min():.3f} m (no passo {int(dists.argmin())} de {n})")
    print(f"  variação total do punho: {np.linalg.norm(punhos.max(0) - punhos.min(0)):.3f} m")

    # Está convergindo? Correlação entre passo e distância: negativa = aproxima.
    r = np.corrcoef(np.arange(n), dists)[0, 1]
    print(f"\n  correlação passo × distância: {r:+.3f}", end="  ")
    if r < -0.3:
        print("→ APROXIMA da xícara")
    elif r > 0.3:
        print("→ AFASTA da xícara")
    else:
        print("→ SEM tendência (oscila, não converge)")


if __name__ == "__main__":
    main()
