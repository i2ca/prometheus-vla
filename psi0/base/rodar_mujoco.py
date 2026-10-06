"""Ψ0 BASE (sem ajuste) em malha fechada no MuJoCo, na cena do café com os objetos nas medidas reais (../cena).

Precisa do servidor (servidor_base.py) escutando na porta 8777 desta máquina (direto ou por túnel SSH).
A cada bloco: renderiza a câmera da cabeça (D435 na pose calibrada do Prometheus), lê o estado cru
(mãos Dex3 14 + braços 14, na ordem de Psi0/scripts/viz/g1.py), recebe 16 ações RELATIVAS e as integra a 30 Hz
como alvo de posição (PD + compensação de gravidade). Pernas e cintura paradas (o Humanoid Everyday não tem parte inferior).
Grava qpos por passo, alvos, imagens enviadas, latência e horário de parede para re-renderizar depois.

Uso:  MUJOCO_GL=egl python rodar_mujoco.py --out saida/ep01 --blocos 40 --copo 0.26 -0.20 --afasta 0.15 \
          --instrucao "the robot uses its right hand to pick up the white mug kept on the right side of the desk"
Dependências: mujoco==3.13.0, numpy, opencv-python.
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
import argparse, json, pickle, socket, struct, datetime as dt
from pathlib import Path
import numpy as np, mujoco, cv2

CENA = Path(__file__).resolve().parents[1] / "cena" / "cena_cafe_medidas.xml"
MAO = ["left_hand_thumb_0_joint", "left_hand_thumb_1_joint", "left_hand_thumb_2_joint", "left_hand_middle_0_joint", "left_hand_middle_1_joint",
       "left_hand_index_0_joint", "left_hand_index_1_joint", "right_hand_thumb_0_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint",
       "right_hand_index_0_joint", "right_hand_index_1_joint", "right_hand_middle_0_joint", "right_hand_middle_1_joint"]
BRACO = [f"{s}_{j}_joint" for s in ("left", "right") for j in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")]
JUNTAS = MAO + BRACO
POSE0 = {**dict(zip(BRACO[:7], [0.3, 0.15, 0.0, 0.7, 0.0, 0.0, 0.0])), **dict(zip(BRACO[7:], [0.3, -0.15, 0.0, 0.7, 0.0, 0.0, 0.0]))}
D435_XYZ, D435_RPY = [0.0576235, 0.01753, 0.42987], [0.0, 0.8307768, 0.0]   # câmera da cabeça no torso_link (calibração do Prometheus)
OBJETOS = ("mesa", "copo", "chaleira", "base_eletrica", "coador", "pote", "tampa", "scoop")
KP_BRACO, KD_BRACO = [80, 80, 80, 80, 40, 40, 40], [3, 3, 3, 0.3, 1.5, 1.5, 1.5]


def rpy_to_R(a, b, c):
    ca, sa, cb, sb, cc, sc = np.cos(a), np.sin(a), np.cos(b), np.sin(b), np.cos(c), np.sin(c)
    return np.array([[cb * cc, sa * sb * cc - ca * sc, ca * sb * cc + sa * sc], [cb * sc, sa * sb * sc + ca * cc, ca * sb * sc - sa * cc], [-sb, sa * cb, ca * cb]])


class Sim:
    def __init__(self, cena):
        self.m = m = mujoco.MjModel.from_xml_path(str(cena)); self.d = mujoco.MjData(m)
        self.act = {m.joint(m.actuator_trnid[i, 0]).name: i for i in range(m.nu)}
        self.qadr = np.array([m.jnt_qposadr[m.actuator_trnid[i, 0]] for i in range(m.nu)])
        self.vadr = np.array([m.jnt_dofadr[m.actuator_trnid[i, 0]] for i in range(m.nu)])
        self.kp, self.kd, self.q_des = np.full(m.nu, 200.0), np.full(m.nu, 5.0), np.zeros(m.nu)
        for jn, i in self.act.items():
            if "hand" in jn: self.kp[i], self.kd[i] = 8, 0.3
            if jn in BRACO: k = BRACO.index(jn) % 7; self.kp[i], self.kd[i] = KP_BRACO[k], KD_BRACO[k]
        self.copo = m.body("copo").id
        self.mao_dir = {b for b in range(m.nbody) if m.body(b).name.startswith("right_hand")}

    def q(self, nomes): return np.array([self.d.qpos[self.qadr[self.act[j]]] for j in nomes])

    def alvo(self, nomes, vals):
        for j, v in zip(nomes, vals):
            lo, hi = self.m.jnt_range[self.m.joint(j).id]; self.q_des[self.act[j]] = np.clip(v, lo, hi)

    def passo(self, n):   # PD + compensação de gravidade
        m, d = self.m, self.d
        for _ in range(n):
            tau = self.kp * (self.q_des - d.qpos[self.qadr]) - self.kd * d.qvel[self.vadr] + d.qfrc_bias[self.vadr]
            d.ctrl[:] = np.clip(tau, m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1]); mujoco.mj_step(m, d)

    def dedos_no_copo(self):
        m, d = self.m, self.d; n = set()
        for c in d.contact[:d.ncon]:
            a, b = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
            if self.copo in (a, b) and (b if a == self.copo else a) in self.mao_dir: n.add(b if a == self.copo else a)
        return len(n)


def afasta(m, d, dx):
    """Mesa e objetos dx metros para longe do robô (a base do G1 é fixa na cena)."""
    for n in OBJETOS:
        b = m.body(n).id; j = m.body_jntadr[b]
        if j >= 0 and m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE: d.qpos[m.jnt_qposadr[j]] += dx
        else: m.body_pos[b, 0] += dx
    mujoco.mj_forward(m, d)


def pede(s, img, est, instr):
    b = pickle.dumps({"image": img, "state": est, "instruction": instr, "steps": 10}); s.sendall(struct.pack("!I", len(b)) + b)
    n = struct.unpack("!I", s.recv(4, socket.MSG_WAITALL))[0]; buf = b""
    while len(buf) < n: buf += s.recv(n - len(buf))
    return pickle.loads(buf)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True); ap.add_argument("--instrucao", default="the robot uses its right hand to pick up the white mug kept on the right side of the desk")
    ap.add_argument("--blocos", type=int, default=40); ap.add_argument("--executa", type=int, default=16); ap.add_argument("--porta", type=int, default=8777)
    ap.add_argument("--copo", type=float, nargs=2, default=[0.26, -0.20]); ap.add_argument("--afasta", type=float, default=0.15)
    a = ap.parse_args(); os.makedirs(f"{a.out}/entradas", exist_ok=True)
    sim = Sim(CENA); m, d = sim.m, sim.d
    hc = m.camera("head_camera").id; Rc = rpy_to_R(*D435_RPY) @ np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]], float)
    q = np.zeros(4); mujoco.mju_mat2Quat(q, Rc.ravel()); m.cam_pos[hc] = D435_XYZ; m.cam_quat[hc] = q; m.cam_fovy[hc] = 42.5
    for jn, v in POSE0.items(): d.qpos[m.jnt_qposadr[m.joint(jn).id]] = v
    ca = m.jnt_qposadr[m.body_jntadr[sim.copo]]; d.qpos[ca:ca + 3] = [*a.copo, 0.7536]; d.qpos[ca + 3:ca + 7] = [0.707107, 0, 0, 0.707107]
    afasta(m, d, a.afasta)
    sim.q_des[:] = d.qpos[sim.qadr]; sim.passo(250)
    rend = mujoco.Renderer(m, 480, 640); n_sub = int(round((1 / 30) / m.opt.timestep)); s = socket.create_connection(("127.0.0.1", a.porta))
    alvo = sim.q(JUNTAS).copy(); z0 = d.xpos[sim.copo][2]
    Q, T, ALVO, CH, SUB, DED, log = [], [], [], [], [], [], []
    for b in range(a.blocos):
        rend.update_scene(d, camera="head_camera"); img = rend.render().copy(); est = sim.q(JUNTAS).astype(np.float32)
        cv2.imwrite(f"{a.out}/entradas/bloco_{b:03d}.jpg", img[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 92])
        w0 = dt.datetime.now().isoformat(timespec="milliseconds"); r = pede(s, img, est, a.instrucao)
        log.append(dict(bloco=b, enviado=w0, latencia_s=round(r["lat_s"], 4), estado=est.tolist(), delta=r["delta"].tolist()))
        for k in range(a.executa):
            alvo = alvo + r["delta"][k]; sim.alvo(JUNTAS, alvo); sim.passo(n_sub)
            Q.append(d.qpos.copy()); T.append(d.time); ALVO.append(alvo.copy()); CH.append(b)
            SUB.append(1000 * float(d.xpos[sim.copo][2] - z0)); DED.append(sim.dedos_no_copo())
        print(f"bloco {b:02d} lat {r['lat_s']:.2f}s | caneca subiu {SUB[-1]:.1f} mm | dedos {DED[-1]}", flush=True)
    np.savez_compressed(f"{a.out}/qpos.npz", qpos=np.array(Q), t=np.array(T), alvo=np.array(ALVO), chunk=np.array(CH),
                        subida_mm=np.array(SUB), dedos=np.array(DED), juntas=np.array(JUNTAS))
    json.dump(dict(instrucao=a.instrucao, copo=a.copo, afasta=a.afasta, blocos=a.blocos, executa=a.executa, trace=log), open(f"{a.out}/trace.json", "w"))
    print(json.dumps(dict(subida_max_mm=max(SUB), passos=len(Q), latencia_mediana_s=float(np.median([x["latencia_s"] for x in log])))))


if __name__ == "__main__":
    main()
