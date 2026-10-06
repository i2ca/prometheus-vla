"""Task 3 do Ψ0 ("Pick bottle, turn and pour into cup") em malha fechada no MuJoCo, com as duas mãos e os dois braços.

Cena ../cena/cena_task3.xml (gera_cena_task3.py). O servidor (servidor_task3.py, porta 8778) devolve blocos de 30 ações de 36
dimensões e replaneja a cada 15 passos com o RTC do servidor oficial. Aqui cada ação vira:
mãos e braços (0:28) -> alvo absoluto das 28 juntas; torso rpy (28:31) -> juntas da cintura; altura (31) -> z da base;
vx, vy (32:34, com os limiares do cliente oficial) e yaw alvo (35, perseguido a 0,5 rad/s) -> base do robô. O pelve não tem pernas controladas: a base segue esses comandos como um
controlador de pernas (AMO) ideal, posicionada a cada passo de simulação.
Uso: MUJOCO_GL=egl python rodar_mujoco_task3.py --out saida/t3_01 --blocos 60
"""
import os, sys
os.environ.setdefault("MUJOCO_GL", "egl")
import argparse, json, pickle, socket, struct, datetime as dt
from pathlib import Path
import numpy as np, mujoco, cv2
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "base"))
from rodar_mujoco import Sim, JUNTAS, D435_XYZ, D435_RPY, rpy_to_R

CENA = Path(__file__).resolve().parents[1] / "cena" / "cena_task3.xml"
CINTURA = ["waist_roll_joint", "waist_pitch_joint", "waist_yaw_joint"]
H0 = 0.75   # altura nominal do servidor oficial
POSE_DADOS = [-0.701, 0.747, 0.833, -0.157, -0.398, -0.200, -0.372, -0.531, -0.715, -0.725, 0.359, 0.443, 0.356, 0.398,   # mãos (ordem de JUNTAS)
              0.442, 0.012, 0.185, 0.332, 0.019, -0.072, -0.024, 0.469, 0.004, -0.183, 0.321, 0.001, -0.069, 0.022]   # braços: média do 1o quadro dos 80 episódios
GIRO = 0.5  # rad/s, a velocidade de giro nos episódios (vyaw = -0,5 enquanto vira)


def pede(s, msg):
    b = pickle.dumps(msg); s.sendall(struct.pack("!I", len(b)) + b)
    n = struct.unpack("!I", s.recv(4, socket.MSG_WAITALL))[0]; buf = b""
    while len(buf) < n: buf += s.recv(n - len(buf))
    return pickle.loads(buf)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True); ap.add_argument("--blocos", type=int, default=60); ap.add_argument("--porta", type=int, default=8778)
    ap.add_argument("--executa", type=int, default=15, help="ações executadas antes de replanejar (s_min do oficial)")
    a = ap.parse_args(); os.makedirs(f"{a.out}/entradas", exist_ok=True)
    sim = Sim(CENA, objeto="garrafa"); m, d = sim.m, sim.d
    hc = m.camera("head_camera").id; Rc = rpy_to_R(*D435_RPY) @ np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]], float)
    q = np.zeros(4); mujoco.mju_mat2Quat(q, Rc.ravel()); m.cam_pos[hc] = D435_XYZ; m.cam_quat[hc] = q; m.cam_fovy[hc] = 42.5
    base = [m.joint(n).id for n in ("base_x", "base_y", "base_z", "base_yaw")]
    badr = [m.jnt_qposadr[j] for j in base]; bdof = [m.jnt_dofadr[j] for j in base]
    for jn, v in zip(JUNTAS, POSE_DADOS): d.qpos[m.jnt_qposadr[m.joint(jn).id]] = v   # começa como os episódios
    mujoco.mj_forward(m, d); sim.q_des[:] = d.qpos[sim.qadr]
    pose = np.zeros(4)   # x, y, z, yaw da base

    def passo(n):
        for _ in range(n):
            sim.passo(1); d.qpos[badr] = pose; d.qvel[bdof] = 0
        mujoco.mj_forward(m, d)
    passo(250)
    rend = mujoco.Renderer(m, 480, 640); dt_c = 1 / 30; n_sub = int(round(dt_c / m.opt.timestep)); s = socket.create_connection(("127.0.0.1", a.porta))
    z0 = d.xpos[sim.copo][2]; Q, T, ACAO, CH, SUB, DED, POSE, log = [], [], [], [], [], [], [], []
    for b in range(a.blocos):
        rend.update_scene(d, camera="head_camera"); img = rend.render().copy(); est = sim.q(JUNTAS).astype(np.float32)
        cv2.imwrite(f"{a.out}/entradas/bloco_{b:03d}.jpg", img[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 92])
        w0 = dt.datetime.now().isoformat(timespec="milliseconds")
        r = pede(s, {"image": img, "state": est, "primeiro": b == 0, "k": a.executa}); A = r["acoes"]
        log.append(dict(bloco=b, enviado=w0, latencia_s=round(r["lat_s"], 4), estado=est.tolist(), acoes=A.tolist()))
        for k in range(a.executa):
            x = A[k]; sim.alvo(JUNTAS, x[:28]); sim.alvo(CINTURA, x[28:31])
            vx = 0.6 if x[32] > 0.25 else 0.0; vy = 0.0 if abs(x[33]) < 0.3 else 0.5 * np.sign(x[33])   # limiares do cliente oficial
            pose[3] += np.clip(x[35] - pose[3], -GIRO * dt_c, GIRO * dt_c)                              # AMO persegue o yaw alvo
            c, sn = np.cos(pose[3]), np.sin(pose[3]); pose[0] += (vx * c - vy * sn) * dt_c; pose[1] += (vx * sn + vy * c) * dt_c
            pose[2] = x[31] - H0
            passo(n_sub)
            Q.append(d.qpos.copy()); T.append(d.time); ACAO.append(x.copy()); CH.append(b); POSE.append(pose.copy())
            SUB.append(1000 * float(d.xpos[sim.copo][2] - z0)); DED.append(sim.dedos_no_copo())
        print(f"bloco {b:02d} lat {r['lat_s']:.2f}s | giro {np.degrees(pose[3]):6.1f}° | garrafa subiu {SUB[-1]:6.1f} mm | dedos {DED[-1]}", flush=True)
    np.savez_compressed(f"{a.out}/qpos.npz", qpos=np.array(Q), t=np.array(T), acao=np.array(ACAO), chunk=np.array(CH), base=np.array(POSE),
                        subida_mm=np.array(SUB), dedos=np.array(DED), juntas=np.array(JUNTAS))
    json.dump(dict(instrucao="g1/pick_bottle_and_turn_and_pour_into_cup", blocos=a.blocos, executa=a.executa, trace=log), open(f"{a.out}/trace.json", "w"))
    print(json.dumps(dict(garrafa_subiu_max_mm=max(SUB), giro_final_graus=float(np.degrees(pose[3])), passos=len(Q))))


if __name__ == "__main__":
    main()
