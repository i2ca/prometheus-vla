"""Contraprova da avaliação: repete às cegas as ações gravadas da demo mais central do dataset nas mesmas posições de avaliação
(mesma semente e ordem de sorteio de avalia.py). Se o modelo não for bem melhor que isto, o teste não mostra uso da câmera.
Sucesso = critério de avalia.py (caneca > 50 mm, 2+ elos, inclinação < 15°).
Uso: python replay_cego.py <pasta das demos> --semente 99 --n 10 [--cintura [--humano]]
"""
import argparse, glob, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "base"))
from rodar_mujoco import Sim, CENA, JUNTAS, CINTURA, POSE_HE, monta


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("demos"); ap.add_argument("--semente", type=int, default=99); ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--cintura", action="store_true"); ap.add_argument("--humano", action="store_true"); ap.add_argument("--yaw", type=float, default=20)
    a = ap.parse_args()
    afasta, x, y = (0.10 if a.humano else 0.15, (0.24, 0.30), (-0.22, -0.04)) if a.cintura else (0.05, (0.24, 0.28), (-0.22, -0.07))
    ok = [e for e in sorted(glob.glob(a.demos + "/ep*")) if json.load(open(e + "/info.json"))["aceita"]]
    xy = np.array([json.load(open(e + "/info.json"))["xy"] for e in ok]); centro = ok[int(np.argmin(np.linalg.norm(xy - xy.mean(0), axis=1)))]
    A = np.load(centro + "/dados.npz")["action"]; J = JUNTAS + CINTURA if a.cintura else JUNTAS
    print("demo repetida", Path(centro).name, json.load(open(centro + "/info.json"))["xy"])
    rng = np.random.default_rng(a.semente); n_ok = 0
    for i in range(a.n):
        px, py, yaw = rng.uniform(*x), rng.uniform(*y), 90 + rng.uniform(-a.yaw, a.yaw)
        sim = Sim(CENA); m, d = sim.m, sim.d
        _, n = monta(sim, dict(zip(JUNTAS, POSE_HE)), (px, py), afasta, yaw_copo=np.radians(yaw)); z0 = d.xpos[sim.copo][2]; inc = 0
        for acao in list(A) + [A[-1]] * 45:
            sim.alvo(J, acao); sim.passo(n); inc = max(inc, np.degrees(np.arccos(np.clip(d.xmat[sim.copo][8], -1, 1))))
        sub = 1000 * (d.xpos[sim.copo][2] - z0); ded = sim.dedos_no_copo(); s = sub > 50 and ded >= 2 and inc < 15; n_ok += s
        print(f"ep{i:02d} ({px:.3f},{py:.3f}) subida {sub:.0f} mm dedos {ded} incl {inc:.1f} sucesso {s}", flush=True)
    print(f"replay cego {n_ok}/{a.n}")


if __name__ == "__main__":
    main()
