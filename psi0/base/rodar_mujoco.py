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
import argparse, json, pickle, socket, struct, datetime as dt, multiprocessing as mp
from pathlib import Path
import numpy as np, mujoco, cv2

CENA = Path(__file__).resolve().parents[1] / "cena" / "cena_cafe_medidas.xml"
# Dex3 no vetor do Ψ0 = ordem dos motores do SDK, igual nas duas mãos: thumb 0-2, middle 0-1, index 0-1
# (Psi0 real/teleop/robot_control/hand_retargeting.py:42-47 e scripts/data/merge_posttrain_sonic.py:76-83)
MAO = ["left_hand_thumb_0_joint", "left_hand_thumb_1_joint", "left_hand_thumb_2_joint", "left_hand_middle_0_joint", "left_hand_middle_1_joint",
       "left_hand_index_0_joint", "left_hand_index_1_joint", "right_hand_thumb_0_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint",
       "right_hand_middle_0_joint", "right_hand_middle_1_joint", "right_hand_index_0_joint", "right_hand_index_1_joint"]
BRACO = [f"{s}_{j}_joint" for s in ("left", "right") for j in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")]
JUNTAS = MAO + BRACO
# cintura nas posições 28-30 do vetor do Ψ0 (rpy do tronco que o AMO executa); com as pernas paradas, rpy do tronco = juntas da cintura
CINTURA = ["waist_roll_joint", "waist_pitch_joint", "waist_yaw_joint"]
KP_CINTURA, KD_CINTURA = 250, 5   # lerobot-ext/robot/unitree_g1/config_unitree_g1.py (G1 do lab, cintura liberada)
POSE0 = {**dict(zip(BRACO[:7], [0.3, 0.15, 0.0, 0.7, 0.0, 0.0, 0.0])), **dict(zip(BRACO[7:], [0.3, -0.15, 0.0, 0.7, 0.0, 0.0, 0.0]))}   # --pose antiga (mãos em 0)
# --pose he (padrão): média do 1o quadro de 30 episódios do Humanoid Everyday G1 (tarefas de pegar com a direita), na ordem de JUNTAS
POSE_HE = [-0.713, 0.692, 0.734, -0.292, -0.531, -0.363, -0.434, -0.559, -0.626, -0.696, 0.388, 0.676, 0.380, 0.479,
           0.206, 0.311, 0.092, 1.006, 0.048, -0.001, -0.107, 0.245, -0.294, -0.099, 0.963, -0.094, -0.013, 0.072]
D435_XYZ, D435_RPY = [0.0576235, 0.01753, 0.42987], [0.0, 0.8307768, 0.0]   # câmera da cabeça no torso_link (calibração do Prometheus)
OBJETOS = ("mesa", "copo", "chaleira", "base_eletrica", "coador", "pote", "tampa", "scoop")
KP_BRACO, KD_BRACO = [80, 80, 80, 80, 40, 40, 40], [3, 3, 3, 3, 1.5, 1.5, 1.5]   # xr_teleoperate (o kd 0,3 do cotovelo era erro de digitação)
# --ganhos deploy (padrão): os do deploy oficial do Ψ0 (iguais aos de ../task3/rodar_mujoco_task3.py); Dex3 com frictionloss 0,03 (../cena/gera_cena_task3.py)
KP_DEPLOY, KD_DEPLOY, KP_MAO_DEPLOY, KD_MAO_DEPLOY = [50, 50, 50, 50, 30, 30, 30], [7.5, 7.5, 7.5, 7.5, 6, 6, 6], 1.5, 0.2


def rpy_to_R(a, b, c):
    ca, sa, cb, sb, cc, sc = np.cos(a), np.sin(a), np.cos(b), np.sin(b), np.cos(c), np.sin(c)
    return np.array([[cb * cc, sa * sb * cc - ca * sc, ca * sb * cc + sa * sc], [cb * sc, sa * sb * sc + ca * cc, ca * sb * sc - sa * cc], [-sb, sa * cb, ca * cb]])


class Sim:
    def __init__(self, cena, objeto="copo"):
        self.m = m = mujoco.MjModel.from_xml_path(str(cena)); self.d = mujoco.MjData(m)
        self.act = {m.joint(m.actuator_trnid[i, 0]).name: i for i in range(m.nu)}
        self.qadr = np.array([m.jnt_qposadr[m.actuator_trnid[i, 0]] for i in range(m.nu)])
        self.vadr = np.array([m.jnt_dofadr[m.actuator_trnid[i, 0]] for i in range(m.nu)])
        self.kp, self.kd, self.q_des = np.full(m.nu, 200.0), np.full(m.nu, 5.0), np.zeros(m.nu)
        for jn, i in self.act.items():
            if "hand" in jn: self.kp[i], self.kd[i] = 8, 0.3
            if jn in BRACO: k = BRACO.index(jn) % 7; self.kp[i], self.kd[i] = KP_BRACO[k], KD_BRACO[k]
        self.copo = m.body(objeto).id   # objeto cuja subida e contatos de dedo são medidos
        self.mao_dir = {b for b in range(m.nbody) if m.body(b).name.startswith("right_hand")}

    def q(self, nomes): return np.array([self.d.qpos[self.qadr[self.act[j]]] for j in nomes])

    def alvo(self, nomes, vals):
        for j, v in zip(nomes, vals):
            lo, hi = self.m.jnt_range[self.m.joint(j).id]; self.q_des[self.act[j]] = np.clip(v, lo, hi)

    def passo(self, n, pernas=None):   # PD + compensação de gravidade; pernas(): sobrescreve d.ctrl das pernas (AMO físico)
        m, d = self.m, self.d
        for _ in range(n):
            tau = self.kp * (self.q_des - d.qpos[self.qadr]) - self.kd * d.qvel[self.vadr] + d.qfrc_bias[self.vadr]
            d.ctrl[:] = np.clip(tau, m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1])
            if pernas: pernas()
            mujoco.mj_step(m, d)

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


def monta(sim, pose, copo, dx, ganhos="deploy", yaw_copo=np.pi / 2, render=True):
    """Ganhos, câmera da cabeça, pose inicial (dict junta->valor ou vetor na ordem JUNTAS), caneca em copo=(x,y) com a alça em yaw_copo,
    afasta dx e assenta 250 passos. Devolve (renderizador opcional 480x640, passos de física por quadro de 30 Hz). Usado também por ../finetune."""
    m, d = sim.m, sim.d
    if ganhos == "deploy":
        for jn, i in sim.act.items():
            if "hand" in jn: sim.kp[i], sim.kd[i] = KP_MAO_DEPLOY, KD_MAO_DEPLOY; m.dof_frictionloss[sim.vadr[i]] = 0.03
            if jn in BRACO: k = BRACO.index(jn) % 7; sim.kp[i], sim.kd[i] = KP_DEPLOY[k], KD_DEPLOY[k]
            if jn in CINTURA: sim.kp[i], sim.kd[i] = KP_CINTURA, KD_CINTURA
    hc = m.camera("head_camera").id; Rc = rpy_to_R(*D435_RPY) @ np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]], float)
    q = np.zeros(4); mujoco.mju_mat2Quat(q, Rc.ravel()); m.cam_pos[hc] = D435_XYZ; m.cam_quat[hc] = q; m.cam_fovy[hc] = 42.5
    for jn, v in (pose.items() if isinstance(pose, dict) else zip(JUNTAS, pose)): d.qpos[m.jnt_qposadr[m.joint(jn).id]] = v
    ca = m.jnt_qposadr[m.body_jntadr[sim.copo]]; d.qpos[ca:ca + 3] = [*copo, 0.7536]; d.qpos[ca + 3:ca + 7] = [np.cos(yaw_copo / 2), 0, 0, np.sin(yaw_copo / 2)]
    afasta(m, d, dx)
    sim.q_des[:] = d.qpos[sim.qadr]; sim.passo(250)
    return (mujoco.Renderer(m, 480, 640) if render else None), int(round((1 / 30) / m.opt.timestep))


def render_camera(cena, nome, dx, conn):
    """Uma câmera por processo (sem mistura de buffers EGL), que carrega a cena uma vez e renderiza cada qpos que receber."""
    m = mujoco.MjModel.from_xml_path(str(cena)); d = mujoco.MjData(m); afasta(m, d, dx); r = mujoco.Renderer(m, 480, 640)
    while (qpos := conn.recv()) is not None:
        d.qpos[:] = qpos; mujoco.mj_forward(m, d); r.update_scene(d, camera=f"{nome}_camera"); conn.send(r.render())
    r.close()


def janela_ao_vivo(cena, dx, fila):
    """Visualizador do MuJoCo em processo próprio (spawn, para não contaminar o EGL das câmeras). Toca os qpos no tempo da
    simulação, 30 por segundo, depois de juntar 1 s de folga: a espera pela inferência não aparece como robô parado."""
    import time, queue, mujoco.viewer
    m = mujoco.MjModel.from_xml_path(str(cena)); d = mujoco.MjData(m); afasta(m, d, dx)
    buf, fim, relogio, engasgos = [], False, None, 0
    with mujoco.viewer.launch_passive(m, d) as v:
        while v.is_running() and not (fim and not buf):
            try:
                while True:
                    q = fila.get(timeout=0.005) if not buf else fila.get_nowait()
                    if q is None: fim = True
                    else: buf.append(q)
            except queue.Empty: pass
            if relogio is None and (len(buf) >= 30 or fim): relogio = time.monotonic()
            if relogio is not None and time.monotonic() >= relogio:
                if buf: d.qpos[:] = buf.pop(0); mujoco.mj_forward(m, d); v.sync(); relogio += 1 / 30
                elif not fim: engasgos += 1; relogio = time.monotonic() + 1 / 30   # o cliente não acompanhou: a janela espera
            time.sleep(0.002)
        time.sleep(1.5 if v.is_running() else 0)
    print(f"janela: {engasgos} esperas por falta de quadro", flush=True)


_CAMERAS = {}


def render_multicam(cena, qpos, dx):
    """As 3 câmeras da policy. Os processos ficam vivos entre chamadas (antes, recarregar a cena 3 vezes por bloco custava ~1 s)."""
    nomes = ("head", "left_wrist", "right_wrist")
    if (cena, dx) not in _CAMERAS:
        conns = []
        for nome in nomes:
            pai, filho = mp.Pipe(); p = mp.Process(target=render_camera, args=(cena, nome, dx, filho), daemon=True); p.start(); conns.append(pai)
        _CAMERAS[(cena, dx)] = conns
    conns = _CAMERAS[(cena, dx)]
    for conn in conns: conn.send(qpos)
    return {nome: conn.recv() for nome, conn in zip(nomes, conns)}


def pede(s, imagens, est, instr, ft_k=None):
    """Base: 16 deltas. Com ft_k (ajuste fino servido por ../task3/servidor_task3.py): 30 ações absolutas, RTC com prefixo do bloco anterior."""
    jpeg = {n: cv2.imencode(".jpg", im[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes() for n, im in imagens.items()}   # ~20x menos bytes no túnel
    msg = {"images": jpeg, "state": est, "primeiro": ft_k == 0, "k": ft_k} if ft_k is not None else {"image": imagens["head"], "state": est, "instruction": instr, "steps": 10}
    b = pickle.dumps(msg); s.sendall(struct.pack("!I", len(b)) + b)
    n = struct.unpack("!I", s.recv(4, socket.MSG_WAITALL))[0]; buf = b""
    while len(buf) < n: buf += s.recv(n - len(buf))
    return pickle.loads(buf)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True); ap.add_argument("--instrucao", default="the robot uses its right hand to pick up the white mug kept on the right side of the desk")
    ap.add_argument("--blocos", type=int, default=40); ap.add_argument("--executa", type=int, default=16); ap.add_argument("--porta", type=int, default=8777)
    ap.add_argument("--copo", type=float, nargs=2, default=[0.26, -0.20]); ap.add_argument("--afasta", type=float, default=0.15)
    ap.add_argument("--pose", choices=("he", "antiga"), default="he"); ap.add_argument("--ganhos", choices=("deploy", "teleop"), default="deploy")
    ap.add_argument("--ft", action="store_true", help="checkpoint do ajuste fino (../finetune): ações absolutas em blocos de 30; use --executa 15 --afasta 0.05")
    ap.add_argument("--cintura", action="store_true", help="com --ft: estado e ação com a cintura (31 valores: 28 + roll, pitch, yaw)")
    ap.add_argument("--yaw-copo", type=float, default=90, help="giro da caneca em graus (90 = alça para o robô, como na cena)")
    ap.add_argument("--semente", type=int, help="sorteia a caneca na região de treino da v24 (x 0,26-0,27; y -0,18 a -0,16; yaw 90 ± 20), no lugar de --copo/--yaw-copo")
    ap.add_argument("--janela", action="store_true", help="abre o visualizador do MuJoCo na tela e executa em tempo real (30 passos de controle por segundo)")
    a = ap.parse_args(); os.makedirs(f"{a.out}/entradas", exist_ok=True)
    if a.semente is not None:
        rng = np.random.default_rng(a.semente); a.copo = [float(rng.uniform(0.26, 0.27)), float(rng.uniform(-0.18, -0.16))]; a.yaw_copo = float(90 + rng.uniform(-20, 20))
    sim = Sim(CENA); m, d = sim.m, sim.d
    pose = list(POSE0.items()) if a.pose == "antiga" else list(zip(JUNTAS, POSE_HE))
    rend, n_sub = monta(sim, dict(pose), a.copo, a.afasta, ganhos=a.ganhos, yaw_copo=np.radians(a.yaw_copo), render=not a.ft)
    s = socket.create_connection(("127.0.0.1", a.porta))
    janela = None
    if a.janela:
        ctx = mp.get_context("spawn"); fila = ctx.Queue()
        janela = ctx.Process(target=janela_ao_vivo, args=(CENA, a.afasta, fila)); janela.start()
    J = JUNTAS + CINTURA if a.cintura else JUNTAS
    alvo = sim.q(J).copy(); z0 = d.xpos[sim.copo][2]
    Q, T, ALVO, CH, SUB, DED, INC, log = [], [], [], [], [], [], [], []
    for b in range(a.blocos):
        if a.ft:
            imagens = render_multicam(CENA, d.qpos.copy(), a.afasta)
        else:
            rend.update_scene(d, camera="head_camera"); imagens = {"head": rend.render().copy()}
        est = sim.q(J).astype(np.float32)
        for nome, img in imagens.items(): cv2.imwrite(f"{a.out}/entradas/bloco_{b:03d}_{nome}.jpg", img[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 92])
        w0 = dt.datetime.now().isoformat(timespec="milliseconds"); r = pede(s, imagens, est, a.instrucao, (0 if b == 0 else a.executa) if a.ft else None)
        if a.ft:   # limita velocidade e aceleração nos limites medidos das demos, inclusive na emenda entre chunks RTC
            filtradas = []; passo_anterior = ALVO[-1] - ALVO[-2] if len(ALVO) > 1 else np.zeros_like(alvo)
            for acao in r["acoes"][:, :len(J)]:
                passo = np.clip(acao - alvo, -2.5 / 30, 2.5 / 30)
                passo = np.clip(passo, passo_anterior - 35 / 900, passo_anterior + 35 / 900)
                alvo = alvo + passo; passo_anterior = passo; filtradas.append(alvo.copy())
            acoes = np.array(filtradas)
        log.append(dict(bloco=b, enviado=w0, latencia_s=round(r["lat_s"], 4), estado=est.tolist(), **({"acoes": acoes.tolist()} if a.ft else {"delta": r["delta"].tolist()})))
        for k in range(a.executa):
            alvo = acoes[k].copy() if a.ft else alvo + r["delta"][k]; sim.alvo(J, alvo); sim.passo(n_sub)
            if janela: fila.put(d.qpos.copy())
            Q.append(d.qpos.copy()); T.append(d.time); ALVO.append(alvo.copy()); CH.append(b)
            SUB.append(1000 * float(d.xpos[sim.copo][2] - z0)); DED.append(sim.dedos_no_copo())
            INC.append(float(np.degrees(np.arccos(np.clip(d.xmat[sim.copo][8], -1, 1)))))
        print(f"bloco {b:02d} lat {r['lat_s']:.2f}s | caneca subiu {SUB[-1]:.1f} mm | dedos {DED[-1]}", flush=True)
    if janela: fila.put(None); janela.join()   # deixa a janela terminar de tocar a rodada
    if a.ft: final = render_multicam(CENA, d.qpos.copy(), a.afasta)["head"]
    else: rend.update_scene(d, camera="head_camera"); final = rend.render().copy()
    cv2.imwrite(f"{a.out}/entradas/final_head.jpg", final[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 92])   # estado depois do último bloco
    pico = int(np.argmax(SUB))
    np.savez_compressed(f"{a.out}/qpos.npz", qpos=np.array(Q), t=np.array(T), alvo=np.array(ALVO), chunk=np.array(CH),
                        subida_mm=np.array(SUB), dedos=np.array(DED), inclinacao_deg=np.array(INC), juntas=np.array(JUNTAS))
    json.dump(dict(instrucao=a.instrucao, copo=a.copo, yaw_copo=a.yaw_copo, ft=a.ft, cintura=a.cintura, afasta=a.afasta, pose=a.pose, ganhos=a.ganhos, blocos=a.blocos, executa=a.executa, trace=log), open(f"{a.out}/trace.json", "w"))
    cod = d.xpos[m.body("coador").id]; alvo = np.array([cod[0] - 0.006, cod[1] - 0.020]); erro_colocar = 1000 * float(np.linalg.norm(d.xpos[sim.copo][:2] - alvo))
    print(json.dumps(dict(subida_max_mm=max(SUB), subida_final_mm=SUB[-1], dedos_final=DED[-1], dedos_no_pico=DED[pico], inclinacao_no_pico_deg=INC[pico], inclinacao_max_deg=max(INC), inclinacao_final_deg=INC[-1], erro_colocar_mm=erro_colocar, passos=len(Q), copo=a.copo, yaw_copo=a.yaw_copo, latencia_mediana_s=float(np.median([x["latencia_s"] for x in log])))))


if __name__ == "__main__":
    main()
