"""Go/no-go fisico: a Dex3 consegue envolver e reter o copo na pose de fecho da semente humana?

Braco direito vai direto a pose de fecho (frame gi da semente), copo colocado no centro da garra,
mao fecha parada por `dwell` frames, depois o braco segue a semente de gi ate o fim (que levanta)
sem nenhum pino no copo. Mede dedos em contato, altura do copo e se foi arremessado.
Uso: .venv/bin/python scripts/gonogo.py [--ep 0] [--kp 12] [--out results/gonogo.json]
"""
import argparse, json, os, sys, time
import mujoco
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from g1_sim import G1Sim, load_seed, ARM_JOINTS, HAND_JOINTS, CLOSE_TGT, ROOT


def trial(sim, arm_seq, hand_seq, gi, n_sub, dx, dy, dz, kp, dwell, measure_frac, gain, rec=None):
    d = sim.d; mujoco.mj_resetData(sim.m, d)
    OPEN = hand_seq[0]
    # alvo de fecho: pose fechada real da semente, extrapolada por `gain` a partir da aberta (gain<=0 usa CLOSE_TGT)
    CLOSE = OPEN + gain * (hand_seq[-1] - OPEN) if gain > 0 else CLOSE_TGT
    # centro da garra medido num fecho parcial (onde os dedos tocariam), por cinematica direta
    for jn, qi in zip(ARM_JOINTS, arm_seq[gi]): d.qpos[sim.qadr[sim.act_joint[jn]]] = qi
    mpose = OPEN + measure_frac * (CLOSE - OPEN)
    for jn, qi in zip(HAND_JOINTS, mpose): d.qpos[sim.qadr[sim.act_joint[jn]]] = qi
    mujoco.mj_forward(sim.m, d)
    gc = sim.grasp_center() + np.array([dx, dy, dz])
    # pose inicial: braco ja na pose de fecho, mao aberta, copo no centro da garra
    for jn, qi in zip(HAND_JOINTS, OPEN): d.qpos[sim.qadr[sim.act_joint[jn]]] = qi
    sim.q_des[:] = d.qpos[sim.qadr]
    sim.set_targets(HAND_JOINTS, OPEN, kp=kp, kd=1.0)
    sim.place_cup(gc)
    w0 = sim.warnings(); max_nc = 0
    if rec: rec.event('assentando: braco na pose de fecho, mao aberta, copo pinado no centro da garra')
    for _ in range(15):              # assenta com o copo pinado
        sim.step(n_sub); sim.place_cup(gc); rec and rec.frame(d)
    if rec: rec.event(f'fechando a Dex3 (kp={kp:g}, ganho={gain:g}), copo ainda pinado')
    for k in range(dwell):           # fecha parado, copo pinado
        sim.set_targets(HAND_JOINTS, CLOSE, kp=kp, kd=1.0); sim.step(n_sub); sim.place_cup(gc); rec and rec.frame(d)
        max_nc = max(max_nc, len(sim.finger_contacts()))
    nc_close = sorted(sim.finger_contacts())
    if rec: rec.event('pino solto: so a mao segura o copo')
    for _ in range(15):   # solta o pino e segura 0,5 s parado
        sim.step(n_sub); rec and rec.frame(d)
    held_static = float(np.linalg.norm(sim.cup_pos() - gc)) < 0.05
    lifts = []
    if rec: rec.event('levantando: braco segue a semente humana ate o fim')
    for t in range(gi, len(arm_seq)):        # levanta seguindo a semente
        sim.set_targets(ARM_JOINTS, arm_seq[t]); sim.step(n_sub); rec and rec.frame(d)
        max_nc = max(max_nc, len(sim.finger_contacts())); lifts.append(sim.cup_pos()[2] - gc[2])
    if rec: rec.event('segurando 1 s')
    for _ in range(int(1.0 * 30)):           # segura 1 s no fim
        sim.step(n_sub); lifts.append(sim.cup_pos()[2] - gc[2]); rec and rec.frame(d)
    cup_end = sim.cup_pos(); dist = float(np.linalg.norm(sim.grasp_center() - cup_end))
    thrown = float(np.linalg.norm(cup_end[:2] - gc[:2])) > 0.15
    lift_final = float(np.mean(lifts[-10:]))
    retained = dist < 0.10 and not thrown and cup_end[2] > 0.80
    return dict(dx=dx, dy=dy, dz=dz, kp=kp, gain=gain, close_target=[round(float(x),3) for x in CLOSE], grasp_center=[round(float(x), 4) for x in gc],
                max_fingers=max_nc, fingers_at_close=nc_close, held_static=held_static,
                lift_final_m=round(lift_final, 4), cup_end=[round(float(x), 4) for x in cup_end],
                dist_hand_cup_m=round(dist, 4), thrown=thrown, retained=bool(retained),
                warnings=sim.warnings() - w0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ep", type=int, default=0); ap.add_argument("--kp", type=float, nargs="+", default=[12.0])
    ap.add_argument("--dwell", type=int, default=35); ap.add_argument("--measure-frac", type=float, default=0.6)
    ap.add_argument("--gain", type=float, nargs="+", default=[1.5, 2.5], help="0 = CLOSE_TGT fixo; >0 extrapola a pose fechada real")
    ap.add_argument("--dx", type=float, nargs="+", default=[-0.02, 0.0, 0.02]); ap.add_argument("--dy", type=float, nargs="+", default=[-0.02, 0.0, 0.02]); ap.add_argument("--dz", type=float, nargs="+", default=[-0.02, 0.0])
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "gonogo.json"))
    ap.add_argument("--record", default=None, help="grava um unico ensaio (primeiro dx/dy/dz/kp/gain) em mp4")
    a = ap.parse_args()
    arm_seq, hand_seq, fps = load_seed(a.ep)
    closing = np.where(hand_seq[:, 4] > 0.5)[0]; gi = int(closing[0]) if len(closing) else len(arm_seq) // 2
    sim = G1Sim(); n_sub = max(1, round((1.0 / fps) / sim.m.opt.timestep))
    print(f"[gonogo] ep={a.ep} T={len(arm_seq)} gi={gi} n_sub={n_sub}", flush=True)
    rows = []; t0 = time.time()
    if a.record:
        from recorder import Recorder
        rec = Recorder(sim.m, a.record)
        r = trial(sim, arm_seq, hand_seq, gi, n_sub, a.dx[0], a.dy[0], a.dz[0], a.kp[0], a.dwell, a.measure_frac, a.gain[0], rec)
        secs = rec.close(a.record.replace(".mp4", ".timeline.json"))
        print(f"[gonogo] gravado {a.record} ({secs:.1f}s) -> {r}"); return
    for kp in a.kp:
      for gain in a.gain:
        for dz in a.dz:
          for dy in a.dy:
            for dx in a.dx:
                r = trial(sim, arm_seq, hand_seq, gi, n_sub, dx, dy, dz, kp, a.dwell, a.measure_frac, gain)
                rows.append(r)
                print(f"[gonogo] kp={kp:4.0f} g={gain:.1f} dx={dx:+.3f} dy={dy:+.3f} dz={dz:+.3f} -> dedos={r['max_fingers']} fecho={r['fingers_at_close']} "
                      f"estatico={r['held_static']} lift={r['lift_final_m']*100:+.1f}cm dist={r['dist_hand_cup_m']*100:.0f}cm "
                      f"thrown={r['thrown']} retido={r['retained']} warn={r['warnings']}", flush=True)
    ok = [r for r in rows if r["retained"]]
    verdict = "GO" if any(r["max_fingers"] >= 3 for r in ok) else ("PARCIAL" if ok or any(r["max_fingers"] >= 2 for r in rows) else "NO-GO")
    best = max(rows, key=lambda r: (r["retained"], r["lift_final_m"] if r["retained"] else -1, r["max_fingers"]))
    out = dict(episode=a.ep, gi=gi, dwell=a.dwell, measure_frac=a.measure_frac, verdict=verdict, best=best,
               n_retained=len(ok), trials=rows, seconds=round(time.time() - t0, 1))
    os.makedirs(os.path.dirname(a.out), exist_ok=True); json.dump(out, open(a.out, "w"), indent=2)
    print(f"[gonogo] VEREDITO={verdict} retidos={len(ok)}/{len(rows)} melhor={best} ({out['seconds']}s)")


if __name__ == "__main__":
    main()
