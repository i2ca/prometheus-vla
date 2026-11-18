"""Real x simulador, num mosaico 3x2 (colunas: punho esq, cabeça, punho dir).

  linha 1: câmeras reais, já com o giro/espelho que o rodar_real2.py manda ao modelo
  linha 2: o MuJoCo copiando a pose do robô, com a caneca onde ela estava no treino

Dois modos:
  ao vivo (padrão): lê câmeras e juntas direto do robô. Puxa ~70 Mbit/s (estado a 400 Hz + 3 câmeras), o mesmo que o
     rodar_real2.py; pela VPN no Wi-Fi os dois juntos saturam o link e a run fica com 20 s por bloco (ep07, 09/11/2026).
     Use só ANTES da run, para posicionar a caneca. O semaforo.py fecha este modo antes de liberar o robô.
  --run saida/epNN: durante a run. Não usa a rede: mostra as imagens que o rodar_real2.py acabou de mandar ao modelo
     (entradas/) e a pose do braço do comandos_braco.jsonl (as mãos ficam na pose inicial).

Contorno verde na câmera real da cabeça: onde a caneca do treino apareceria (sim com a pose real do robô). Encaixe a
caneca real dentro dele, com o robô na pose inicial: mesma posição e mesmo tamanho na imagem = mesmo lugar em relação ao tronco.

Uso:  python ver_real_sim.py                       # antes: posicionar a caneca  (q fecha, s salva em saida/)
      python ver_real_sim.py --run saida/ep08      # durante a run, num segundo terminal
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")   # antes de importar o mujoco; a janela é do cv2
import argparse, datetime as dt, json, time
from pathlib import Path
import numpy as np, cv2, mujoco
import rodar_mujoco as rm
from rodar_real2 import cabeca, le_punho

AQUI = Path(__file__).resolve().parent
CORPO = ["left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee", "left_ankle_pitch", "left_ankle_roll",
         "right_hip_pitch", "right_hip_roll", "right_hip_yaw", "right_knee", "right_ankle_pitch", "right_ankle_roll",
         "waist_yaw", "waist_roll", "waist_pitch"] + [f"{s}_{j}" for s in ("left", "right") for j in
         ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")]   # ordem G1_29_JointIndex
CAMS = [("punho esq", "left_wrist", 5556), ("cabeca", "head", 5555), ("punho dir", "right_wrist", 5557)]
PRETO = np.zeros((480, 640, 3), np.uint8)


class AoVivo:
    def __init__(self, a):
        import zmq
        from lerobot.cameras.zmq.camera_zmq import ZMQCamera
        from lerobot.cameras.zmq.configuration_zmq import ZMQCameraConfig
        self.zmq, self.a, self.subs, self.cams = zmq, a, [], {}
        for porta in (6001, 6002):
            s = zmq.Context.instance().socket(zmq.SUB); s.setsockopt(zmq.SUBSCRIBE, b""); s.connect(f"tcp://{a.robot_ip}:{porta}"); self.subs.append(s)
        for _, nome, porta in CAMS:   # o servidor publica tudo como "head_camera"; a porta separa as câmeras
            c = ZMQCamera(ZMQCameraConfig(server_address=a.robot_ip, port=porta, camera_name="head_camera")); c.connect(); self.cams[nome] = c

    def juntas(self):
        """{junta: q} com o mais novo de cada tópico (esvazia as filas)."""
        q = {}
        for s in self.subs:
            while True:
                try: msg = json.loads(s.recv(self.zmq.NOBLOCK))
                except self.zmq.Again: break
                ms = msg["data"]["motor_state"]
                if msg["topic"] == "rt/lowstate":
                    q.update({f"{n}_joint": v["q"] for n, v in zip(CORPO, ms[:29])}); self.rpy = msg["data"]["imu_state"]["rpy"]   # IMU da pélvis
                else:
                    lado = "left" if "left" in msg["topic"] else "right"
                    q.update({j: v["q"] for j, v in zip([j for j in rm.MAO if j.startswith(lado)], ms[:7])})
        return q

    def imagens(self):
        out = {}
        for _, nome, _ in CAMS:
            try:
                im = cabeca(self.cams[nome].async_read(timeout_ms=500), False) if nome == "head" else le_punho(self.cams, nome, self.a)
                out[nome] = np.ascontiguousarray(im[:, :, ::-1])   # RGB -> BGR do cv2
            except Exception: out[nome] = PRETO
        return out

    def fecha(self):
        for c in self.cams.values(): c.disconnect()


class DaRun:
    """Segue os arquivos que o rodar_real2.py grava: nada passa pela rede."""
    def __init__(self, a):
        self.dir = Path(a.run); self.jsonl = self.dir / "comandos_braco.jsonl"; self.pos = 0; self.ims = {n: PRETO for _, n, _ in CAMS}

    def juntas(self):
        if not self.jsonl.exists(): return {}
        with open(self.jsonl) as f:
            f.seek(self.pos); linhas = f.read().splitlines()
            if linhas and not linhas[-1].endswith("}"): linhas = linhas[:-1]   # linha ainda sendo escrita
            self.pos += sum(len(l) + 1 for l in linhas)
        if not linhas: return {}
        obs = json.loads(linhas[-1])["obs"]   # 14 do braço, ordem G1_29_JointArmIndex
        return {f"{n}_joint": v for n, v in zip(CORPO[15:], obs)}

    def imagens(self):
        blocos = sorted((self.dir / "entradas").glob("bloco_*_head.jpg"))
        if blocos:
            pre = str(blocos[-1])[:-len("head.jpg")]
            for _, nome, _ in CAMS:
                im = cv2.imread(pre + f"{nome}.jpg")
                if im is not None: self.ims[nome] = im
        return self.ims

    def fecha(self): pass


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--robot-ip", default="10.9.8.73")
    ap.add_argument("--run", help="pasta de saída do rodar_real2.py: modo sem rede, para usar durante a run")
    ap.add_argument("--punho-dir", choices=("nada", "gira", "espelha"), default="espelha")   # mesmos padrões do rodar_real2.py
    ap.add_argument("--punho-esq", choices=("nada", "gira", "espelha"), default="gira")
    ap.add_argument("--copo", type=float, nargs=2, default=[0.265, -0.17], help="caneca no sim (centro da região do treino)")
    ap.add_argument("--yaw-copo", type=float, default=68)
    ap.add_argument("--afasta", type=float, default=0.10, help="o mesmo da avaliação 10/10")
    ap.add_argument("--uma", action="store_true", help="sem janela: salva um mosaico em saida/ e sai")
    ap.add_argument("--largura", type=int, default=400, help="largura de cada quadro na janela")
    a = ap.parse_args()

    # cena igual à da avaliação: câmeras do XML (as mesmas do render_multicam), caneca no centro do treino, objetos afastados
    m = mujoco.MjModel.from_xml_path(str(rm.CENA)); d = mujoco.MjData(m)
    ca = m.jnt_qposadr[m.body_jntadr[m.body("copo").id]]; y = np.radians(a.yaw_copo)
    d.qpos[ca:ca + 3] = [*a.copo, 0.7536]; d.qpos[ca + 3:ca + 7] = [np.cos(y / 2), 0, 0, np.sin(y / 2)]
    rm.afasta(m, d, a.afasta)
    for j, v in zip(rm.JUNTAS, rm.POSE_HE): d.qpos[m.jnt_qposadr[m.joint(j).id]] = v   # mãos e braços partem da pose inicial
    rend = mujoco.Renderer(m, 480, 640)
    # alvo: contorno verde da caneca do treino, projetada na câmera da cabeça do sim (= D435i calibrada), desenhado na imagem real
    seg = mujoco.Renderer(m, 480, 640); seg.enable_segmentation_rendering()
    geoms_copo = [g for g in range(m.ngeom) if m.geom_bodyid[g] == m.body("copo").id]
    hc = m.camera("head_camera").id; xml_pos, xml_quat = m.cam_pos[hc].copy(), m.cam_quat[hc].copy()
    Rc = rm.rpy_to_R(*rm.D435_RPY) @ np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]], float)
    D435_POS, D435_QUAT = np.array(rm.D435_XYZ, float), np.zeros(4); mujoco.mju_mat2Quat(D435_QUAT, Rc.ravel())

    fonte = DaRun(a) if a.run else AoVivo(a)
    W, H = a.largura, a.largura * 3 // 4; ultimo_estado = 0.0
    print("q fecha, s salva o mosaico", flush=True)
    while True:
        q = fonte.juntas()
        if q: ultimo_estado = time.time()
        for j, v in q.items(): d.qpos[m.jnt_qposadr[m.joint(j).id]] = v
        mujoco.mj_forward(m, d)

        reais, linhas = fonte.imagens(), [[], []]
        # contorno na D435i calibrada (rodar_mujoco.monta), não na câmera do XML; mj_camlight recalcula d.cam_xpos/xmat, que o render usa
        m.cam_pos[hc], m.cam_quat[hc] = D435_POS, D435_QUAT; mujoco.mj_camlight(m, d)
        seg.update_scene(d, camera="head_camera"); s = seg.render()
        m.cam_pos[hc], m.cam_quat[hc] = xml_pos, xml_quat; mujoco.mj_camlight(m, d)
        mascara = (np.isin(s[:, :, 0], geoms_copo) & (s[:, :, 1] == mujoco.mjtObj.mjOBJ_GEOM)).astype(np.uint8)
        contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        reais = {**reais, "head": cv2.drawContours(np.ascontiguousarray(reais["head"]).copy(), contornos, -1, (0, 255, 0), 2)}
        for rotulo, nome, _ in CAMS:
            rend.update_scene(d, camera=f"{nome}_camera"); sim = np.ascontiguousarray(rend.render()[:, :, ::-1])
            for i, im in enumerate((reais[nome], sim)):
                im = cv2.resize(im, (W, H))
                cv2.putText(im, f"{('real', 'sim')[i]} {rotulo}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                linhas[i].append(im)
        mos = np.vstack([np.hstack(l) for l in linhas])
        if getattr(fonte, "rpy", None):   # no sim a pélvis é fixa e reta; inclinada no pórtico, braço e câmera giram juntos e o contorno não fecha
            r_, p_ = np.degrees(fonte.rpy[:2]); ok = max(abs(r_), abs(p_)) < 3
            cv2.putText(mos, f"pelvis roll {r_:+.1f} pitch {p_:+.1f} graus" + ("" if ok else "  <- ENDIREITE NO PORTICO"), (W + 8, H - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if ok else (0, 0, 255), 2)
        if time.time() - ultimo_estado > (3 if a.run else 1):
            cv2.putText(mos, "esperando a run..." if a.run else "SEM ESTADO DO ROBO (ponte 6001/6002 parada?)",
                        (10, H - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        if a.uma and time.time() - ultimo_estado > 1: time.sleep(0.2); continue   # espera o primeiro estado
        k = ord("s") if a.uma else (cv2.imshow("real x sim", mos), cv2.waitKey(30) & 0xFF)[1]
        if k == ord("q"): break
        if k == ord("s"):
            (AQUI / "saida").mkdir(exist_ok=True); p = AQUI / "saida" / f"real_sim_{dt.datetime.now():%H%M%S}.jpg"; cv2.imwrite(str(p), mos); print("salvo", p)
            if a.uma: break
    fonte.fecha()


if __name__ == "__main__":
    main()
