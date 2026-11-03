"""Task 3 do Ψ0 ("Pick bottle, turn and pour into cup") em malha fechada no MuJoCo, com as duas mãos e os dois braços.

Cena ../cena/cena_task3.xml (gera_cena_task3.py). O servidor (servidor_task3.py, porta 8778) devolve blocos de 30 ações de 36
dimensões e replaneja a cada 15 passos com o RTC do servidor oficial. Aqui cada ação vira:
mãos e braços (0:28) -> alvo absoluto das 28 juntas; torso rpy (28:31) -> juntas da cintura; altura (31) -> altura absoluta do pelve;
vx, vy (32:34, com os limiares do cliente oficial) e yaw alvo (35, perseguido a 0,5 rad/s; --amo: como o real, ver Base) -> base do robô. O pelve não tem pernas controladas: a base segue esses comandos como um
controlador de pernas (AMO) ideal, posicionada a cada passo de simulação.
Uso: MUJOCO_GL=egl python rodar_mujoco_task3.py --out saida/t3_01 --blocos 60
--amo-fisico troca a base imposta pela política de pernas oficial (AMO) com física: cena_task3_amo.xml (pelve livre, pés no chão),
pernas e cintura por amo/pernas_mujoco.py a 50 Hz, passo de 1/600 s; validação em amo/valida_fisico.py.
--sem-rtc (só diagnóstico) troca o RTC por predict_action puro a cada bloco. A Dex3 usa frictionloss 0,03 (gera_cena_task3.py).
"""
import os, sys
os.environ.setdefault("MUJOCO_GL", "egl")
import argparse, json, pickle, socket, struct, datetime as dt
from pathlib import Path
import numpy as np, mujoco, cv2
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "base"))
from rodar_mujoco import Sim, JUNTAS, D435_XYZ, D435_RPY, rpy_to_R

CENA = Path(__file__).resolve().parents[1] / "cena" / "cena_task3.xml"
CENA_AMO = CENA.with_name("cena_task3_amo.xml")   # --amo-fisico: pelve livre, pés colidindo (gera_cena_task3.py --amo-fisico)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "amo"))
CINTURA = ["waist_roll_joint", "waist_pitch_joint", "waist_yaw_joint"]
PELVE0 = 0.793   # origem do pelve no MJCF (pernas em zero). acao[31] é o z do frame pelvis no mundo (multi_target_ik.py:226-235,
                 # psi-inference_rtc.py:301-308) e o AMO o rastreia a ~1 cm dobrando os joelhos (adapter_jit.pt + FK neste MJCF).
                 # 0,75 é a altura de pé escolhida pelo Ψ0 em maintain_standing (master_whole_body.py:559-600), não a nominal do G1.
                 # Conferido no quadro 0 contra o real do ep. 0 (results/psi0-task3-sim/validacao_k/lado_a_lado.png): pelve = acao[31].
POSE_DADOS = [-0.701, 0.747, 0.833, -0.157, -0.398, -0.200, -0.372, -0.531, -0.715, -0.725, 0.359, 0.443, 0.356, 0.398,   # mãos (ordem de JUNTAS)
              0.442, 0.012, 0.185, 0.332, 0.019, -0.072, -0.024, 0.469, 0.004, -0.183, 0.321, 0.001, -0.069, 0.022]   # braços: média do 1o quadro dos 80 episódios
GIRO = 0.5  # rad/s, a velocidade de giro nos episódios (vyaw = -0,5 enquanto vira)
# Ganhos do caminho de deploy do Ψ0 (psi-inference_rtc.py -> body_ctrl.ctrl_whole_body): braços em real/teleop/robot_control/robot_body.py:191-212,
# Dex3 em robot_hand_unitree.py:58-59. O pipeline oficial não filtra as ações; quem absorve o vaivém quadro a quadro da política é esse PD mole.
KP_BRACO, KD_BRACO = [50, 50, 50, 50, 30, 30, 30], [7.5, 7.5, 7.5, 7.5, 6, 6, 6]
KP_MAO, KD_MAO = 1.5, 0.2
# --atraso: no real o estado S segue a ação A com atraso (pico da correlação cruzada ΔA x ΔS: 6 quadros no braço D, 3 no E, 2 nas mãos;
# fercout_real.npz, 80 episódios). O ajuste direto A->S (1a ordem, tau 0,32 a 0,40 s nos braços) atrasa demais no sim, porque o PD do
# sim já põe 3 quadros. Por isso o filtro foi ajustado por cima do PD do sim, em malha aberta com as ações gravadas (real_vs_sim/atraso.py
# e grade nos eps. 0, 20, 50; validação nos eps. 1, 21, 51, fora do ajuste): braços y += 0,35*(A - y) por quadro de 33 ms (tau 0,08 s),
# atraso do sim 3 -> 6 quadros (braço D) e 2 -> 3 (E), iguais ao real. Atraso puro extra (fila de 1 a 3 quadros) e filtro nas mãos
# pioraram o erro contra S: as mãos com kp 1,5 e frictionloss 0,03 já seguem como o real (atraso 2, zona morta |ΔA| < ~0,03 rad).
# Vai no alvo do PD; os ganhos do deploy ficam.
ATRASO_ALFA = np.r_[np.ones(14), np.full(14, 0.35)]   # ordem de JUNTAS: mãos sem filtro, braços 0,35
# --amo: o AMO real só corrige o erro de yaw com o "portão" aberto. master_whole_body.get_ik_observation zera dyaw (= yaw - target_yaw,
# a única entrada de yaw da política de pernas) quando _in_place_stand_flag, isto é |vx|, |vy| e |vyaw| < 0,1; no deploy vyaw = acao[34]
# cru e target_yaw = acao[35] sem integrar (psi-inference_rtc.py:291-315, record=False). Quando vyaw volta a 0 o robô para onde
# está, aquém do alvo. Medido no vídeo egocêntrico (results/psi0-task3-sim/gemeo/yaw_vo.py, fluxo KLT de 1 GL, conferido contra a
# fita do ep. 0) nos eps. 0, 1, 2, 4 e 11 e ajustado em gemeo/ajuste_amo.py: taxa K*(alvo - yaw) limitada a VMAX, comandos atrasados
# D quadros. RMS do yaw no giro 3,6° (cliente sem --amo: 7,9 a 11,1° por episódio). Avanço: o real anda 0,235 m onde a integração dá
# 0,42 m (ep. 0, 14 a 18 s, gemeo/pre_caminhada.py), vx efetivo 0,6 x KANDAR. vy sem medida, fica como no cliente.
AMO_K, AMO_VMAX, AMO_D, AMO_KANDAR = 3.1, 0.55, 4, 0.235 / 0.42
# --camera: suporte da câmera da cabeça. D435_XYZ/RPY vêm do d435_joint do URDF oficial (real/assets/g1/g1_body29_hand14.urdf), no
# torso_link DELE, que fica 1 cm abaixo do torso_link deste MJCF (cintura 0,044 + 0 no URDF, 0,035 + 0,019 aqui; braços iguais a < 7 mm,
# results/psi0-task3-sim/gemeo/camera_mao/compara_fk.py). O que sobra do suporte foi medido pela mão direita no vídeo do ep. 0, o único
# observável que não depende do corpo nem da base (gemeo/camera_mao/ajusta_suporte.py, 21 quadros com a mão livre): 1° para cima,
# giro e roll ~0 (roll mal determinado). Mediana do erro da mão: 8 px (hoje), 15 px (URDF puro), 6 px (URDF + 1°). O desvio grande
# da câmera contra a mesa (camera_real.npz, correcao_pre_giro) NÃO é suporte: como suporte ele leva a mão a 23 px de erro (com a
# translação de 7 cm, que não cabe num suporte) ou a mais de 60 px (só a rotação), e depois do giro ele é outro; é pose do corpo/base.
# Quadro 0 (gemeo/camera_mao/valida_camera.py): bordas da mesa e garrafa mudam < 6 px, dentro do ruído; a correção é pequena e fica opcional. Os 20 a 40 px da mão no despejo (18 a 22 s) não são câmera: aparecem com o braço parado
# segurando a garrafa (21 s, 0,08 rad/s) e somem com a mão livre; o mais provável é flexão/folga do braço sob carga, que o encoder não vê.
SUPORTE_DXYZ, SUPORTE_INCL = np.array([0.0, 0.0, -0.010]), np.radians(1.0)


def camera_cabeca(m, corrigida=False):
    """head_camera = D435 no torso_link (padrão: D435_XYZ no torso_link do MJCF; corrigida: frame do URDF + inclinação medida)."""
    hc = m.camera("head_camera").id; xyz = np.array(D435_XYZ, float); Rc = rpy_to_R(*D435_RPY) @ np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]], float)
    if corrigida:   # gira em torno do x da câmera MuJoCo (direita): ângulo positivo levanta o eixo óptico
        c, s = np.cos(SUPORTE_INCL), np.sin(SUPORTE_INCL); xyz = xyz + SUPORTE_DXYZ; Rc = Rc @ np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    q = np.zeros(4); mujoco.mju_mat2Quat(q, Rc.ravel()); m.cam_pos[hc] = xyz; m.cam_quat[hc] = q; m.cam_fovy[hc] = 42.5


class Base:
    """Próxima pose (x, y, z, yaw) da base a cada ação (33 ms). Padrão: AMO ideal (limiares do cliente oficial em vx/vy, yaw alvo
    perseguido a GIRO sempre). amo=True: portão, taxa e atraso medidos no real (AMO_*). A altura (acao[31]) segue sem atraso, como no
    padrão; o atraso de AMO_D quadros só vale para vx, vy, vyaw e target_yaw, que foi o que o ajuste mediu."""
    def __init__(self, amo=False): self.amo, self.fila = amo, []

    def __call__(self, pose, x, dt_c=1 / 30):
        alvo = pose.copy(); alvo[2] = x[31] - PELVE0
        if self.amo: self.fila = (self.fila + [np.asarray(x)])[-(AMO_D + 1):]; x = self.fila[0]
        vx = 0.6 if x[32] > 0.25 else 0.0; vy = 0.0 if abs(x[33]) < 0.3 else 0.5 * np.sign(x[33])   # limiares do cliente oficial
        if not self.amo: alvo[3] += np.clip(x[35] - pose[3], -GIRO * dt_c, GIRO * dt_c)           # AMO ideal persegue o yaw alvo
        elif abs(vx) >= 0.1 or abs(vy) >= 0.1 or abs(x[34]) >= 0.1:
            alvo[3] += np.clip(AMO_K * (x[35] - pose[3]), -AMO_VMAX, AMO_VMAX) * dt_c; vx *= AMO_KANDAR
        else: vx = vy = 0.0
        c, sn = np.cos(alvo[3]), np.sin(alvo[3]); alvo[0] += (vx * c - vy * sn) * dt_c; alvo[1] += (vx * sn + vy * c) * dt_c
        return alvo


class Atraso:
    """Alvo do PD de mãos+braços (ordem de JUNTAS) filtrado em 1a ordem, um passo de 33 ms por chamada."""
    def __init__(self, q0, alfa=ATRASO_ALFA): self.alfa, self.y = np.asarray(alfa), np.array(q0, float)

    def __call__(self, x):
        self.y += self.alfa * (np.asarray(x, float) - self.y); return self.y.copy()


def ganhos_deploy(sim):
    for jn, i in sim.act.items():
        if "hand" in jn: sim.kp[i], sim.kd[i] = KP_MAO, KD_MAO
        if jn in JUNTAS[14:]: k = (JUNTAS.index(jn) - 14) % 7; sim.kp[i], sim.kd[i] = KP_BRACO[k], KD_BRACO[k]


def pede(s, msg):
    b = pickle.dumps(msg); s.sendall(struct.pack("!I", len(b)) + b)
    n = struct.unpack("!I", s.recv(4, socket.MSG_WAITALL))[0]; buf = b""
    while len(buf) < n: buf += s.recv(n - len(buf))
    return pickle.loads(buf)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True); ap.add_argument("--blocos", type=int, default=60); ap.add_argument("--porta", type=int, default=8778)
    ap.add_argument("--executa", type=int, default=15, help="ações executadas antes de replanejar (s_min do oficial)")
    ap.add_argument("--sem-rtc", action="store_true", help="diagnóstico: todo bloco por predict_action puro, sem prefixo. Offline com observação real mexe ~2x mais (diag_rtc), "
                    "mas em malha fechada no sim não destrava (t3_i1_semrtc); o padrão é o RTC do servidor oficial")
    ap.add_argument("--atraso", action="store_true", help="alvo do PD dos braços filtrado para seguir a ação com o atraso do real (ATRASO_ALFA)")
    ap.add_argument("--amo", action="store_true", help="base como o AMO real: yaw só corrige com vx/vy/vyaw ativos, taxa e atraso medidos, caminhada x0,56 (AMO_*)")
    ap.add_argument("--amo-fisico", action="store_true", help="pernas e cintura pela política AMO oficial com física (pelve livre, pés no chão, "
                    "amo/pernas_mujoco.py), no lugar da base imposta; ação 28:36 vira comando do AMO como no deploy (EstadoAMO.comando_psi0)")
    ap.add_argument("--camera", action="store_true", help="câmera da cabeça no frame do URDF oficial + 1° para cima medido pela mão no vídeo (SUPORTE_*)")
    ap.add_argument("--garrafa", help="x,y da garrafa no mundo (padrão: o da cena, 0,37,-0,14 retroprojetado da câmera). A FK do estado real no fecho "
                    "da mão dá mediana 0,348,-0,089 nos 80 eps (results/psi0-task3-sim/fk_pega.py)")
    ap.add_argument("--contato-rigido", action="store_true", help="garrafa com solref 0,004 e solimp 0,95-0,99: com o padrão (0,02) a garrafa de 25 g "
                    "deixa os dedos entrarem ~5 mm (p50, audita_q.py), porque o contato mole do MuJoCo escala com a massa")
    ap.add_argument("--cena", help="arquivo em ../cena no lugar do padrão (ex. cena_task3_amo_copo.xml, gera_cena_task3.py --objeto-lab); o objeto é o corpo \"garrafa\"")
    a = ap.parse_args(); os.makedirs(f"{a.out}/entradas", exist_ok=True)
    cena = CENA.with_name(a.cena) if a.cena else (CENA_AMO if a.amo_fisico else CENA)
    sim = Sim(cena, objeto="garrafa"); m, d = sim.m, sim.d; ganhos_deploy(sim)
    camera_cabeca(m, a.camera)
    if a.amo_fisico:
        from pernas_mujoco import PernasAMO, DT
        from amo_politica import EstadoAMO
        m.opt.timestep = DT
    else:
        base = [m.joint(n).id for n in ("base_x", "base_y", "base_z", "base_yaw")]
        badr = [m.jnt_qposadr[j] for j in base]; bdof = [m.jnt_dofadr[j] for j in base]
        m.dof_armature[bdof] = 1e5   # base imposta por qpos/qvel a cada passo; sem inércia ela cai livre dentro do passo e os braços
                                     # sentem gravidade quase nula, enquanto a compensação injeta G(q) inteiro (deriva de 2,2 rad em 1 s com kp=0)
    if a.garrafa: ga = m.joint("garrafa_livre").qposadr[0]; d.qpos[ga:ga + 2] = [float(v) for v in a.garrafa.split(",")]
    if a.contato_rigido:
        for g in range(m.ngeom):
            if m.geom_bodyid[g] == m.body("garrafa").id: m.geom_solref[g] = [0.004, 1]; m.geom_solimp[g, :3] = [0.95, 0.99, 0.001]
    for jn, v in zip(JUNTAS, POSE_DADOS): d.qpos[m.jnt_qposadr[m.joint(jn).id]] = v   # começa como os episódios
    mujoco.mj_forward(m, d); sim.q_des[:] = d.qpos[sim.qadr]
    pose = np.array([0.0, 0.0, 0.75 - PELVE0, 0.0])   # x, y, z, yaw da base

    def passo(n, alvo=None):   # base em rampa até o alvo, com a velocidade real nas juntas da base: com salto de pose e qvel = 0
        v = np.zeros(4) if alvo is None else (alvo - pose) / (n * m.opt.timestep)   # o contato vê a mão parada e o atrito arranca o objeto dela
        d.qfrc_applied[bdof] = m.dof_damping[bdof] * v   # o amortecimento da base vira servo de velocidade em v
        for _ in range(n):
            pose[:] += v * m.opt.timestep; sim.passo(1); d.qpos[badr] = pose; d.qvel[bdof] = v
        mujoco.mj_forward(m, d)
    if a.amo_fisico:   # pernas na pose de pé do deploy, pés no chão, 3 s de pé com o AMO (maintain_standing, h 0,75) e zera o yaw de referência
        pernas = PernasAMO(m, d); pernas.postura_inicial(); sim.passo(int(3.0 / DT), pernas); pernas.estado.yaw_offset = None
        pose = pernas.pose()   # aqui pose = x, y, z, yaw do pelve medidos (não a base imposta)
    else: passo(250)
    filtro = Atraso(sim.q(JUNTAS)) if a.atraso else (lambda x: x)
    base_amo = Base(a.amo)
    rend = mujoco.Renderer(m, 480, 640); dt_c = 1 / 30; n_sub = int(round(dt_c / m.opt.timestep)); m.opt.timestep = dt_c / n_sub; s = socket.create_connection(("127.0.0.1", a.porta))
    z0 = d.xpos[sim.copo][2]; Q, T, ACAO, CH, SUB, DED, POSE, log = [], [], [], [], [], [], [], []
    for b in range(a.blocos):
        rend.update_scene(d, camera="head_camera"); img = rend.render().copy(); est = sim.q(JUNTAS).astype(np.float32)
        cv2.imwrite(f"{a.out}/entradas/bloco_{b:03d}.jpg", img[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 92])
        w0 = dt.datetime.now().isoformat(timespec="milliseconds")
        r = pede(s, {"image": img, "state": est, "primeiro": b == 0 or a.sem_rtc, "k": a.executa}); A = r["acoes"]
        log.append(dict(bloco=b, enviado=w0, latencia_s=round(r["lat_s"], 4), estado=est.tolist(), acoes=A.tolist()))
        for k in range(a.executa):
            x = A[k]; sim.alvo(JUNTAS, filtro(x[:28]))
            if a.amo_fisico: pernas.cmd = EstadoAMO.comando_psi0(x); sim.passo(n_sub, pernas); pose = pernas.pose()
            else: sim.alvo(CINTURA, x[28:31]); passo(n_sub, base_amo(pose, x, dt_c))
            Q.append(d.qpos.copy()); T.append(d.time); ACAO.append(x.copy()); CH.append(b); POSE.append(pose.copy())
            SUB.append(1000 * float(d.xpos[sim.copo][2] - z0)); DED.append(sim.dedos_no_copo())
        print(f"bloco {b:02d} lat {r['lat_s']:.2f}s | giro {np.degrees(pose[3]):6.1f}° | garrafa subiu {SUB[-1]:6.1f} mm | dedos {DED[-1]}", flush=True)
    np.savez_compressed(f"{a.out}/qpos.npz", qpos=np.array(Q), t=np.array(T), acao=np.array(ACAO), chunk=np.array(CH), base=np.array(POSE),
                        subida_mm=np.array(SUB), dedos=np.array(DED), juntas=np.array(JUNTAS),
                        cena=cena.name)
    json.dump(dict(instrucao="g1/pick_bottle_and_turn_and_pour_into_cup", blocos=a.blocos, executa=a.executa, trace=log), open(f"{a.out}/trace.json", "w"))
    print(json.dumps(dict(garrafa_subiu_max_mm=max(SUB), giro_final_graus=float(np.degrees(pose[3])), passos=len(Q))))


if __name__ == "__main__":
    main()
