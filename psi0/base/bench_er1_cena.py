"""Erro do er1_cena.localiza (UnifoLM-ER-1 + profundidade) contra a verdade do sim: caneca sorteada na mesa e caneca já sob o coador (qpos finais de runs do Ψ0).
Uso: MUJOCO_GL=egl python bench_er1_cena.py --out <pasta> [--n 12] [--runs dir1 dir2 ...]"""
import argparse, json
from pathlib import Path
import mujoco, numpy as np
from rodar_mujoco import Sim, monta, afasta, CENA, JUNTAS, POSE_HE
from er1_cena import localiza

ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path, required=True); ap.add_argument("--n", type=int, default=12)
ap.add_argument("--runs", nargs="*", default=[]); a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
M, C = "center of the white mug", "center of the coffee filter cone"; rng = np.random.default_rng(0); casos = []
for i in range(a.n):
    copo = [float(rng.uniform(0.22, 0.31)), float(rng.uniform(-0.25, -0.10))]; giro = float(rng.uniform(30, 110))
    sim = Sim(CENA); monta(sim, dict(zip(JUNTAS, POSE_HE)), copo, 0.10, yaw_copo=np.radians(giro), render=False); casos.append((f"mesa_{i:02d}", sim.d.qpos.copy()))
for r in a.runs: casos.append((f"sob_{Path(r).name}", np.load(Path(r) / "qpos.npz")["qpos"][-1]))
m = mujoco.MjModel.from_xml_path(str(CENA)); d = mujoco.MjData(m); afasta(m, d, 0.10); L = []
for nome, q in casos:
    d.qpos[:] = q; mujoco.mj_forward(m, d); o = localiza(m, d, [M, C], str(a.out / f"{nome}.jpg")); v = o["_auditoria"]
    lin = dict(caso=nome, visto=o[M].get("visto"))
    if o[M].get("visto"):
        lin["erro_caneca_mm"] = round(1000 * float(np.linalg.norm(np.array(o[M]["xyz_m"][:2]) - v["caneca_xyz_verdade"][:2])), 1)
        if o[C].get("visto"):
            dv = 1000 * float(np.linalg.norm(np.array(v["caneca_xyz_verdade"][:2]) - v["coador_xyz_verdade"][:2]))
            de = 1000 * float(np.linalg.norm(np.array(o[M]["xyz_m"][:2]) - o[C]["xyz_m"][:2])); lin.update(dist_verdade_mm=round(dv, 1), dist_er1_mm=round(de, 1))
    L.append(lin); print(lin, flush=True)
(a.out / "bench.json").write_text(json.dumps(L, indent=1))
