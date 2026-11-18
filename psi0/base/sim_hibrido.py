"""Simulador híbrido contínuo (MuJoCo único): Teleoperação Quest 3 + VLA Ψ0 v24 + Orquestrador.

Mantém UMA ÚNICA instância da cena `../cena/cena_cafe_medidas.xml` viva durante toda a sessão.
Nunca recria o MuJoCo ao chamar a ferramenta: a VLA e o Quest 3 operam no mesmo mundo físico,
preservando a posição da caneca, coador, chaleira e demais objetos entre passos.

Modos de controle no arbitrador (troca sem tranco):
  - TELEOP (padrão): você controla braços (14) e mãos Dex3 (14) pelo Quest 3 (:8012).
    Botão X destrava/congela o clutch sempre ancorado na pose física atual do robô (`sim.q(BRACO)`).
    Botão A sinaliza ao orquestrador que você concluiu o passo manual no VR.
    Botão B reposiciona a caneca e os braços na pose inicial se quiser recomeçar.
  - VLA (`POST /vla_place` na porta 8782): trava o clutch do VR, faz rampa suave de 1,2 s
    (36 passos a 30 Hz) apenas das juntas do robô até `POSE_HE` (sem tocar nos objetos da mesa),
    executa os 30 blocos do Ψ0 v24 falando com a atena (:8781), grava os quadros da rodada para
    o `chat_ui` e devolve o robô congelado na pose final para o modo TELEOP.
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
import argparse, base64, datetime as dt, json, multiprocessing as mp, socket, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

AQUI = Path(__file__).resolve().parent
VLA_ROOT = AQUI.parents[1]
LEROBOT_EXT = VLA_ROOT / "lerobot-ext"
for p in (str(AQUI), str(LEROBOT_EXT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import cv2, mujoco, numpy as np, zmq
from rodar_mujoco import (
    BRACO, CENA, CINTURA, JUNTAS, MAO, POSE_HE, Sim, afasta, monta, pede, render_multicam,
)

J_TODOS = JUNTAS + CINTURA
POSE_HE_31 = np.array(list(POSE_HE) + [0.0, 0.0, 0.0], dtype=np.float64)
MAX_VEL, MAX_ACC = 2.5 / 30, 35 / 900  # rad por passo de 30 Hz

# Ordem dos dedos no controle do Quest (thumb 0-2, index 0-1, middle 0-1)
MAO_ESQ_VR = [
    "left_hand_thumb_0_joint", "left_hand_thumb_1_joint", "left_hand_thumb_2_joint",
    "left_hand_middle_0_joint", "left_hand_middle_1_joint",
    "left_hand_index_0_joint", "left_hand_index_1_joint",
]
MAO_DIR_VR = [
    "right_hand_thumb_0_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint",
    "right_hand_index_0_joint", "right_hand_index_1_joint",
    "right_hand_middle_0_joint", "right_hand_middle_1_joint",
]
LEFT_CLOSE = np.array([0.0, 0.920, 1.74, -1.57, -1.74, -1.57, -1.74])
RIGHT_CLOSE = np.array([0.0, -0.920, -0.74, 1.57, 1.74, 1.57, 1.74])


def _janela_continua(cena: str, dx: float, shm_qpos, nq: int, parar_ev):
    """Janela 3D do MuJoCo que fica aberta durante toda a sessão (sem atraso)."""
    os.environ.pop("MUJOCO_GL", None)
    import mujoco.viewer
    m = mujoco.MjModel.from_xml_path(str(cena))
    d = mujoco.MjData(m)
    afasta(m, d, dx)
    q_np = np.frombuffer(shm_qpos.get_obj(), dtype=np.float64, count=nq)
    with mujoco.viewer.launch_passive(m, d) as v:
        while v.is_running() and not parar_ev.is_set():
            with shm_qpos.get_lock():
                d.qpos[:] = q_np
            mujoco.mj_forward(m, d)
            v.sync()
            time.sleep(1 / 60)


def desenha_hud(img_bgr: np.ndarray, hud: dict, modo: str, vr_ativo: bool, vr_conectado: bool) -> np.ndarray:
    """Desenha no topo a instrução do orquestrador e no rodapé o estado do controle (VR vs VLA)."""
    out = img_bgr.copy()
    h, w = out.shape[:2]
    overlay = out.copy()

    # Barra superior: passo atual do orquestrador
    desc = hud.get("description") or "Simulação Híbrida pronta (Quest 3 + VLA Ψ0)"
    sub = hud.get("evidence") or hud.get("success_criteria") or ""
    status = hud.get("status", "")
    cor_topo = (18, 62, 22) if status == "goal_completed" else (45, 25, 10) if modo == "vla" else (34, 18, 0)
    cv2.rectangle(overlay, (0, 0), (w, 58), cor_topo, -1)

    # Barra inferior: estado do arbitrador
    if modo == "vla":
        rodape = "VLA PSI0 NO CONTROLE (executando no mesmo MuJoCo)"
        cor_rod = (180, 90, 20)
    elif not vr_conectado:
        rodape = "MODO HIBRIDO (sem Quest 3 conectado)"
        cor_rod = (70, 70, 70)
    elif vr_ativo:
        rodape = "TELEOP VR ATIVO  [X: pausar | A: concluir passo | B: reset cena]"
        cor_rod = (20, 120, 40)
    else:
        rodape = "VR CONGELADO  [Aperte X no Quest 3 para assumir os bracos]"
        cor_rod = (30, 60, 120)
    cv2.rectangle(overlay, (0, h - 30), (w, h), cor_rod, -1)
    cv2.addWeighted(overlay, 0.78, out, 0.22, 0, out)

    cv2.putText(out, desc[:62], (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 255), 1, cv2.LINE_AA)
    if sub:
        cv2.putText(out, sub[:78], (12, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (180, 230, 255), 1, cv2.LINE_AA)
    cv2.putText(out, rodape, (12, h - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return out


class SimuladorHibrido:
    def __init__(self, args):
        self.a = args
        self.lock = threading.Lock()
        self.vla_lock = threading.Lock()
        self.parar = threading.Event()

        # 1. Cria a ÚNICA instância do MuJoCo que viverá até o fim do processo
        self.sim = Sim(CENA)
        self.m, self.d = self.sim.m, self.sim.d
        pose_ini = dict(zip(JUNTAS, POSE_HE))
        _, self.n_sub = monta(
            self.sim, pose_ini, self.a.copo, self.a.afasta,
            ganhos="deploy", yaw_copo=np.radians(self.a.yaw_copo), render=False,
        )
        self.qpos_inicial = self.d.qpos.copy()
        self.qdes_inicial = self.sim.q_des.copy()

        # Warmup dos 3 processos de câmera EGL (head, left_wrist, right_wrist)
        imgs0 = render_multicam(CENA, self.d.qpos.copy(), self.a.afasta)
        self.ultimo_head_rgb = imgs0["head"]
        self.ultimo_jpeg_limpo = self._codifica_jpeg(imgs0["head"][:, :, ::-1])

        # Estado do arbitrador e do HUD
        self.modo = "teleop"  # "teleop" ou "vla"
        self.vr_conectado = False
        self.vr_ativo = False
        self.passo_concluido_vr = False
        self.hud = {
            "description": "Pronto: teleopere no Quest 3 ou envie um comando no Chat",
            "success_criteria": "A cena é contínua entre você e a VLA Ψ0.",
            "status": "waiting",
        }
        self.step_file = Path(os.getenv("STEP_FILE", "/tmp/prometheus_chat/step.json"))

        # Estado do Clutch / IK do Quest 3
        self.tv_wrapper = None
        self.arm_ik = None
        self.hand_retargeter = None
        self.clutch_anchored = False
        self.ctrl_ref_left = None
        self.ctrl_ref_right = None
        self.robot_ref_left = None
        self.robot_ref_right = None
        self.last_x = False
        self.last_a = False
        self.last_b = False
        self._tr_l = 0.0
        self._tr_r = 0.0
        self._sq_l = 0.0
        self._sq_r = 0.0
        self._wrist_filt = None

        # 2. Janela 3D contínua (processo separado usando memória compartilhada)
        self.shm_qpos = None
        self.proc_janela = None
        if not self.a.sem_janela and (os.getenv("DISPLAY") or os.getenv("WAYLAND_DISPLAY")):
            ctx = mp.get_context("spawn")
            self.shm_qpos = ctx.Array("d", self.m.nq)
            self.parar_janela = ctx.Event()
            self._atualiza_shm()
            self.proc_janela = ctx.Process(
                target=_janela_continua,
                args=(str(CENA), self.a.afasta, self.shm_qpos, self.m.nq, self.parar_janela),
                daemon=True,
            )
            self.proc_janela.start()

        # 3. Publicador ZMQ :5555 e Inicialização Quest 3 (Vuer :8012 + G1_29_ArmIK) somente se VR habilitado
        self.zmq_ctx = None
        self.zmq_pub = None
        if not self.a.sem_vr:
            self.zmq_ctx = zmq.Context.instance()
            self.zmq_pub = self.zmq_ctx.socket(zmq.PUB)
            self.zmq_pub.setsockopt(zmq.SNDHWM, 2)
            self.zmq_pub.setsockopt(zmq.LINGER, 0)
            try:
                self.zmq_pub.bind(f"tcp://127.0.0.1:{self.a.zmq_port}")
            except zmq.ZMQError as e:
                print(f"[sim_hibrido] aviso: porta ZMQ {self.a.zmq_port} ocupada ({e})", flush=True)
                self.zmq_pub = None
            self._inicia_vr()
        else:
            self.step_file = Path("/tmp/prometheus_chat/step.json")

    def _inicia_vr(self):
        try:
            sys.path.insert(0, str(LEROBOT_EXT / "teleop" / "robot_control"))
            from televuer import TeleVuerWrapper
            from robot_arm_ik import G1_29_ArmIK
            from hand_retargeting import HandRetargeting, HandType

            cert = os.getenv("XR_TELEOP_CERT", str(LEROBOT_EXT / "cert.pem"))
            key = os.getenv("XR_TELEOP_KEY", str(LEROBOT_EXT / "key.pem"))
            old_cwd = Path.cwd()
            try:
                os.chdir(LEROBOT_EXT)  # usa g1_29_model_cache.pkl de lerobot-ext
                self.arm_ik = G1_29_ArmIK()
            finally:
                os.chdir(old_cwd)
            if self.a.input_mode == "hand":
                self.hand_retargeter = HandRetargeting(HandType.UNITREE_DEX3)
            self.tv_wrapper = TeleVuerWrapper(
                use_hand_tracking=(self.a.input_mode == "hand"),
                binocular=False,
                img_shape=(480, 640),
                display_fps=30.0,
                display_mode=self.a.display_mode,
                zmq=True,
                webrtc=False,
                cert_file=cert if Path(cert).is_file() else None,
                key_file=key if Path(key).is_file() else None,
            )
            self.vr_conectado = True
            print("[sim_hibrido] Vuer VR pronto na porta :8012 (Quest 3). Aperte X no controle esquerdo para destravar.", flush=True)
        except Exception as e:
            print(f"[sim_hibrido] VR indisponível ({e}); seguindo apenas com MuJoCo + VLA.", flush=True)
            self.vr_conectado = False

    @staticmethod
    def _codifica_jpeg(img_bgr: np.ndarray, qual: int = 90) -> bytes:
        return cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, qual])[1].tobytes()

    def _atualiza_shm(self):
        if self.shm_qpos is not None:
            with self.shm_qpos.get_lock():
                np.frombuffer(self.shm_qpos.get_obj(), dtype=np.float64, count=self.m.nq)[:] = self.d.qpos

    def _publica_todas(self, imgs: dict):
        """Cabeça (como antes) + punhos, para a rota /quadro/<camera>."""
        with self.lock:
            self.ultimos_punhos = {c: self._codifica_jpeg(imgs[c][:, :, ::-1], 85) for c in ("left_wrist", "right_wrist") if c in imgs}
        self._publica_quadro(imgs["head"])

    def _publica_quadro(self, head_rgb: np.ndarray):
        """Atualiza o quadro limpo (para o orquestrador/LLM) e publica o quadro com HUD no Quest 3 e ZMQ :5555."""
        bgr_limpo = head_rgb[:, :, ::-1]
        jpeg_limpo = self._codifica_jpeg(bgr_limpo, 92)
        with self.lock:
            self.ultimo_head_rgb = head_rgb
            self.ultimo_jpeg_limpo = jpeg_limpo
            hud_copy = dict(self.hud)
            modo = self.modo
            vr_ativo = self.vr_ativo

        bgr_hud = desenha_hud(bgr_limpo, hud_copy, modo, vr_ativo, self.vr_conectado)
        if self.vr_conectado and self.tv_wrapper is not None:
            try:
                self.tv_wrapper.render_to_xr(bgr_hud)
            except Exception:
                pass
        if self.zmq_pub is not None:
            try:
                b64 = base64.b64encode(jpeg_limpo).decode("ascii")
                self.zmq_pub.send_multipart(
                    [json.dumps({"images": {"head_camera": b64}, "timestamps": {"head_camera": time.time()}}).encode("utf-8")],
                    flags=zmq.NOBLOCK,
                )
            except Exception:
                pass

    def atualiza_hud(self, dados: dict):
        with self.lock:
            self.hud.update(dados)
        try:
            self.step_file.parent.mkdir(parents=True, exist_ok=True)
            self.step_file.write_text(json.dumps(self.hud, ensure_ascii=False))
        except Exception:
            pass

    def resetar_cena(self):
        """Reposiciona a cena inicial somente quando solicitado explicitamente pelo operador."""
        with self.lock:
            self.vr_ativo = False
            self.clutch_anchored = False
            self.d.qpos[:] = self.qpos_inicial
            self.d.qvel[:] = 0
            self.sim.q_des[:] = self.qdes_inicial
            mujoco.mj_forward(self.m, self.d)
            self._atualiza_shm()
        print("[sim_hibrido] Cena reposicionada no estado inicial.", flush=True)

    def _passo_teleop_vr(self):
        """Lê o Quest 3 e atualiza `sim.alvo` se o clutch estiver destravado."""
        if not self.vr_conectado or self.tv_wrapper is None or self.arm_ik is None:
            return
        try:
            tele_data = self.tv_wrapper.get_tele_data()
        except Exception:
            return

        x_pressed = bool(getattr(tele_data, "left_ctrl_aButton", False))
        a_pressed = bool(getattr(tele_data, "right_ctrl_aButton", False))
        b_pressed = bool(getattr(tele_data, "right_ctrl_bButton", False))

        if x_pressed and not self.last_x:
            self.vr_ativo = not self.vr_ativo
            self.clutch_anchored = False
            print(f"[sim_hibrido] Quest 3 clutch: {'DESTRAVADO (ativo)' if self.vr_ativo else 'CONGELADO'}", flush=True)
        if a_pressed and not self.last_a:
            self.passo_concluido_vr = True
            print("[sim_hibrido] Botão A (Quest 3): operador marcou passo como concluído.", flush=True)
        if b_pressed and not self.last_b:
            self.resetar_cena()

        self.last_x, self.last_a, self.last_b = x_pressed, a_pressed, b_pressed
        if not self.vr_ativo:
            return

        lw = getattr(tele_data, "left_wrist_pose", None)
        rw = getattr(tele_data, "right_wrist_pose", None)
        if lw is None or rw is None:
            return

        # Filtro EMA leve na translação do punho
        t_l, t_r = lw[:3, 3].copy(), rw[:3, 3].copy()
        if self._wrist_filt is not None:
            lw[:3, 3] = 0.4 * t_l + 0.6 * self._wrist_filt[0]
            rw[:3, 3] = 0.4 * t_r + 0.6 * self._wrist_filt[1]
        self._wrist_filt = (t_l, t_r)

        # Ao destravar (inclusive logo após a VLA rodar), ancora na pose física REAL dos braços no MuJoCo
        if not self.clutch_anchored:
            q_braco_atual = self.sim.q(BRACO).astype(np.float64)
            self.ctrl_ref_left = lw.copy()
            self.ctrl_ref_right = rw.copy()
            self.arm_ik.init_data = q_braco_atual.copy()
            self.arm_ik.smooth_filter._data_queue = [q_braco_atual.copy()]
            self.arm_ik.smooth_filter._filtered_data = q_braco_atual.copy()
            self.robot_ref_left, self.robot_ref_right = self.arm_ik.forward_kinematics(q_braco_atual)
            self.clutch_anchored = True

        left_target = self.robot_ref_left.copy()
        left_target[:3, 3] = self.robot_ref_left[:3, 3] + (lw[:3, 3] - self.ctrl_ref_left[:3, 3])
        left_target[:3, :3] = (lw[:3, :3] @ self.ctrl_ref_left[:3, :3].T) @ self.robot_ref_left[:3, :3]

        right_target = self.robot_ref_right.copy()
        right_target[:3, 3] = self.robot_ref_right[:3, 3] + (rw[:3, 3] - self.ctrl_ref_right[:3, 3])
        right_target[:3, :3] = (rw[:3, :3] @ self.ctrl_ref_right[:3, :3].T) @ self.robot_ref_right[:3, :3]

        sol_q, _ = self.arm_ik.solve_ik(left_target, right_target, None, None)
        self.sim.alvo(BRACO, sol_q)

        # Mãos Dex3 via gatilho/squeeze dos controles
        raw_tr_l = float(np.clip((10.0 - getattr(tele_data, "left_ctrl_triggerValue", 10.0)) / 10.0, 0.0, 1.0))
        raw_tr_r = float(np.clip((10.0 - getattr(tele_data, "right_ctrl_triggerValue", 10.0)) / 10.0, 0.0, 1.0))
        raw_sq_l = float(np.clip(getattr(tele_data, "left_ctrl_squeezeValue", 0.0), 0.0, 1.0))
        raw_sq_r = float(np.clip(getattr(tele_data, "right_ctrl_squeezeValue", 0.0), 0.0, 1.0))
        self._tr_l = 0.5 * raw_tr_l + 0.5 * self._tr_l
        self._tr_r = 0.5 * raw_tr_r + 0.5 * self._tr_r
        self._sq_l = 0.5 * raw_sq_l + 0.5 * self._sq_l
        self._sq_r = 0.5 * raw_sq_r + 0.5 * self._sq_r
        tr_l = 0.0 if self._tr_l < 0.05 else self._tr_l
        tr_r = 0.0 if self._tr_r < 0.05 else self._tr_r

        esq_q = self._sq_l * LEFT_CLOSE
        dir_q = self._sq_r * RIGHT_CLOSE
        esq_q[5] += -2.2 * tr_l; esq_q[6] += -2.1 * tr_l; esq_q[0] += -0.5 * tr_l; esq_q[1] += 0.8 * tr_l; esq_q[2] += 0.8 * tr_l
        dir_q[5] += 2.2 * tr_r; dir_q[6] += 2.1 * tr_r; dir_q[0] += -0.5 * tr_r; dir_q[1] -= 0.8 * tr_r; dir_q[2] -= 0.8 * tr_r
        self.sim.alvo(MAO_ESQ_VR + MAO_DIR_VR, np.concatenate([esq_q, dir_q]))

    def executar_vla_place(self, run_dir: str, porta_psi0: int = 8781, blocos: int = 30, executa: int = 15) -> dict:
        """Executa o Ψ0 v24 DENTRO desta mesma simulação MuJoCo (sem recriar cena nem mexer nos objetos)."""
        with self.vla_lock:
            os.makedirs(f"{run_dir}/entradas", exist_ok=True)
            # os renders do trabalhador (canal 1) nascem por fork: criados depois do socket, herdariam a conexão com o Ψ0
            # e o servidor (um cliente por vez) ficaria preso nela depois do close
            render_multicam(CENA, self.d.qpos.copy(), self.a.afasta, canal=1)
            s = socket.create_connection(("127.0.0.1", porta_psi0), timeout=5)
            try:
                # 1. Trava o teleop VR e assume o modo VLA
                with self.lock:
                    self.modo = "vla"
                    self.vr_ativo = False
                    self.clutch_anchored = False
                    self._wrist_filt = None
                self.atualiza_hud({
                    "description": "VLA Ψ0: alinhando braços suavemente para POSE_HE...",
                    "status": "in_progress",
                })

                # 2. Rampa suave (1,2 s = 36 passos a 30 Hz) das juntas atuais até POSE_HE_31,
                #    mantendo todos os objetos da mesa exatamente onde estão!
                q_ini = self.sim.q(J_TODOS).copy()
                passos_rampa = 36
                for i in range(1, passos_rampa + 1):
                    t0 = time.perf_counter()
                    alpha = 0.5 * (1.0 - np.cos(np.pi * i / passos_rampa))  # cosseno suave (zero velocidade nas pontas)
                    alvo_rampa = q_ini + alpha * (POSE_HE_31 - q_ini)
                    with self.lock:
                        self.sim.alvo(J_TODOS, alvo_rampa)
                        self.sim.passo(self.n_sub)
                        qpos_snap = self.d.qpos.copy()
                        self._atualiza_shm()
                    if i % 3 == 0:
                        self._publica_todas(render_multicam(CENA, qpos_snap, self.a.afasta))
                    time.sleep(max(0.0, 1 / 30 - (time.perf_counter() - t0)))

                if os.getenv("PSI0_RTC", "1") == "1":   # tempo real: o mundo não para enquanto o modelo pensa (executor_rtc.py)
                    return self._vla_rtc(s, run_dir, blocos * executa, executa)
                # 3. Malha fechada do Ψ0 v24 (30 blocos) na mesma cena
                alvo = self.sim.q(J_TODOS).copy()
                z0 = float(self.d.xpos[self.sim.copo][2])
                Q, T, ALVO, CH, SUB, DED, INC, log = [], [], [], [], [], [], [], []
                instr = "the robot uses its right hand to pick up the white mug and place it under the coffee filter"

                for b in range(blocos):
                    self.atualiza_hud({
                        "description": f"VLA Ψ0: colocando a caneca sob o coador (bloco {b + 1}/{blocos})",
                        "status": "in_progress",
                    })
                    with self.lock:
                        qpos_snap = self.d.qpos.copy()
                        est = self.sim.q(J_TODOS).astype(np.float32)
                    imagens = render_multicam(CENA, qpos_snap, self.a.afasta)
                    self._publica_quadro(imagens["head"])
                    for nome, img in imagens.items():
                        cv2.imwrite(f"{run_dir}/entradas/bloco_{b:03d}_{nome}.jpg", img[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 92])

                    w0 = dt.datetime.now().isoformat(timespec="milliseconds")
                    r = pede(s, imagens, est, instr, 0 if b == 0 else executa)

                    filtradas = []
                    passo_anterior = ALVO[-1] - ALVO[-2] if len(ALVO) > 1 else np.zeros_like(alvo)
                    for acao in r["acoes"][:, :len(J_TODOS)]:
                        passo = np.clip(acao - alvo, -MAX_VEL, MAX_VEL)
                        passo = np.clip(passo, passo_anterior - MAX_ACC, passo_anterior + MAX_ACC)
                        alvo = alvo + passo
                        passo_anterior = passo
                        filtradas.append(alvo.copy())
                    acoes = np.array(filtradas)
                    log.append(dict(bloco=b, enviado=w0, latencia_s=round(r["lat_s"], 4), estado=est.tolist(), acoes=acoes.tolist()))

                    for k in range(executa):
                        t_step = time.perf_counter()
                        alvo = acoes[k].copy()
                        with self.lock:
                            self.sim.alvo(J_TODOS, alvo)
                            self.sim.passo(self.n_sub)
                            qpos_k = self.d.qpos.copy()
                            t_k = float(self.d.time)
                            sub_k = 1000 * float(self.d.xpos[self.sim.copo][2] - z0)
                            ded_k = self.sim.dedos_no_copo()
                            inc_k = float(np.degrees(np.arccos(np.clip(self.d.xmat[self.sim.copo][8], -1, 1))))
                            self._atualiza_shm()
                        Q.append(qpos_k); T.append(t_k); ALVO.append(alvo.copy()); CH.append(b)
                        SUB.append(sub_k); DED.append(ded_k); INC.append(inc_k)
                        if k % 3 == 0 or k == executa - 1:
                            head_k = render_multicam(CENA, qpos_k, self.a.afasta)["head"]
                            self._publica_quadro(head_k)
                        time.sleep(max(0.0, 1 / 30 - (time.perf_counter() - t_step)))

                    linha = f"bloco {b:02d} lat {r['lat_s']:.2f}s | caneca subiu {SUB[-1]:.1f} mm | dedos {DED[-1]}"
                    print(f"[sim_hibrido] {linha}", flush=True)

                with self.lock:
                    qpos_fim = self.d.qpos.copy()
                    cod = self.d.xpos[self.m.body("coador").id].copy()
                    pos_copo = self.d.xpos[self.sim.copo][:2].copy()
                final_head = render_multicam(CENA, qpos_fim, self.a.afasta)["head"]
                self._publica_quadro(final_head)
                cv2.imwrite(f"{run_dir}/entradas/final_head.jpg", final_head[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 92])

                pico = int(np.argmax(SUB))
                np.savez_compressed(
                    f"{run_dir}/qpos.npz", qpos=np.array(Q), t=np.array(T), alvo=np.array(ALVO),
                    chunk=np.array(CH), subida_mm=np.array(SUB), dedos=np.array(DED),
                    inclinacao_deg=np.array(INC), juntas=np.array(JUNTAS),
                )
                json.dump(dict(instrucao=instr, ft=True, cintura=True, afasta=self.a.afasta, blocos=blocos, executa=executa, trace=log), open(f"{run_dir}/trace.json", "w"))
                alvo_xy = np.array([cod[0] - 0.006, cod[1] - 0.020])
                erro_colocar = 1000 * float(np.linalg.norm(pos_copo - alvo_xy))
                metricas = dict(
                    subida_max_mm=max(SUB), subida_final_mm=SUB[-1], dedos_final=DED[-1],
                    dedos_no_pico=DED[pico], inclinacao_no_pico_deg=INC[pico], inclinacao_max_deg=max(INC),
                    inclinacao_final_deg=INC[-1], erro_colocar_mm=erro_colocar, passos=len(Q),
                    copo=list(self.a.copo), yaw_copo=self.a.yaw_copo,
                    latencia_mediana_s=float(np.median([x["latencia_s"] for x in log])),
                )
                cauda = [
                    f"bloco {x['bloco']:02d} lat {x['latencia_s']:.2f}s | caneca subiu {SUB[(x['bloco'] + 1) * executa - 1]:.1f} mm | dedos {DED[(x['bloco'] + 1) * executa - 1]}"
                    for x in log
                ] + [json.dumps(metricas)]
                return {"ok": True, "metricas": metricas, "output_tail": "\n".join(cauda[-40:]), "run_dir": run_dir}
            finally:
                s.close()
                with self.lock:
                    self.modo = "teleop"
                    self.vr_ativo = False
                    self.clutch_anchored = False

    def _vla_rtc(self, s, run_dir: str, n_passos: int, executa: int) -> dict:
        """Ψ0 com o executor RTC (o mesmo de rodar_tempo_real.py, 108 runs sem atraso): um trabalhador consulta o modelo
        em paralelo enquanto a física segue a 30 Hz. As 3 câmeras saem a ~10 Hz para o app."""
        import queue
        from executor_rtc import ExecutorRTC
        instr = "the robot uses its right hand to pick up the white mug and place it under the coffee filter"
        pedidos, respostas = queue.Queue(), queue.Queue()

        def trabalhador():
            while (job := pedidos.get()) is not None:
                qpos, est, primeiro, k, n = job
                imgs = render_multicam(CENA, qpos, self.a.afasta, canal=1)
                for nome, img in imgs.items():
                    cv2.imwrite(f"{run_dir}/entradas/bloco_{n - 1:03d}_{nome}.jpg", img[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 92])
                ida = time.perf_counter()
                try: r = pede(s, imgs, est, instr, 0 if primeiro else k)
                except Exception as e:
                    import traceback; print(f"[sim_hibrido] pedido {n} falhou: {e!r}\n{traceback.format_exc()[-600:]}", flush=True); r = None
                respostas.put((n, r, time.perf_counter() - ida))

        s.settimeout(30)
        th = threading.Thread(target=trabalhador, daemon=True); th.start()
        ex = ExecutorRTC(self.sim.q(J_TODOS), executa=executa)
        z0 = float(self.d.xpos[self.sim.copo][2]); Q, T, ALVO, SUB, DED, INC, log = [], [], [], [], [], [], []
        relogio = time.perf_counter(); atrasados = falhas = 0
        for passo in range(n_passos):
            try:
                n, r, ida = respostas.get_nowait()
                if r is None: ex.pendente = None; falhas += 1; raise queue.Empty
                troca = ex.chegou(r["acoes"][:, :len(J_TODOS)], passo)
                log.append(dict(bloco=n - 1, passo=passo, latencia_s=round(r["lat_s"], 4), ida_e_volta_s=round(ida, 3), **troca))
                self.atualiza_hud({"description": f"VLA Ψ0 em tempo real: bloco {n} (atraso {troca['d']} passos)", "status": "in_progress"})
            except queue.Empty:
                pass
            if ex.quer_pedir():
                primeiro, k = ex.pedido(passo)
                with self.lock: pedidos.put((self.d.qpos.copy(), self.sim.q(J_TODOS).astype(np.float32), primeiro, k, ex.n_pedidos))
            alvo = ex.proximo()
            with self.lock:
                self.sim.alvo(J_TODOS, alvo); self.sim.passo(self.n_sub)
                qpos_k = self.d.qpos.copy(); T.append(float(self.d.time))
                SUB.append(1000 * float(self.d.xpos[self.sim.copo][2] - z0)); DED.append(self.sim.dedos_no_copo())
                INC.append(float(np.degrees(np.arccos(np.clip(self.d.xmat[self.sim.copo][8], -1, 1)))))
                self._atualiza_shm()
            Q.append(qpos_k); ALVO.append(alvo.copy())
            if passo % 3 == 0:
                self._publica_todas(render_multicam(CENA, qpos_k, self.a.afasta))
            espera = relogio + (passo + 1) / 30 - time.perf_counter()
            if espera > 0: time.sleep(espera)
            elif -espera > 1 / 30: atrasados += 1
            if passo % 30 == 29:
                print(f"[sim_hibrido] t {(passo + 1) / 30:4.1f}s | caneca subiu {SUB[-1]:.1f} mm | dedos {DED[-1]} | pedidos {ex.n_pedidos}", flush=True)
        parede = time.perf_counter() - relogio; pedidos.put(None); th.join(timeout=30)   # não fecha o socket com pedido no ar (o servidor só atende um cliente)
        with self.lock:
            qpos_fim = self.d.qpos.copy(); cod = self.d.xpos[self.m.body("coador").id].copy(); pos_copo = self.d.xpos[self.sim.copo][:2].copy()
        final = render_multicam(CENA, qpos_fim, self.a.afasta); self._publica_todas(final)
        cv2.imwrite(f"{run_dir}/entradas/final_head.jpg", final["head"][:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 92])
        pico = int(np.argmax(SUB))
        np.savez_compressed(f"{run_dir}/qpos.npz", qpos=np.array(Q), t=np.array(T), alvo=np.array(ALVO), subida_mm=np.array(SUB),
                            dedos=np.array(DED), inclinacao_deg=np.array(INC), juntas=np.array(JUNTAS))
        json.dump(dict(instrucao=instr, ft=True, cintura=True, rtc=True, afasta=self.a.afasta, passos=n_passos, executa=executa, trace=log),
                  open(f"{run_dir}/trace.json", "w"))
        erro_colocar = 1000 * float(np.linalg.norm(pos_copo - np.array([cod[0] - 0.006, cod[1] - 0.020])))
        metricas = dict(subida_max_mm=max(SUB), subida_final_mm=SUB[-1], dedos_final=DED[-1], dedos_no_pico=DED[pico],
                        inclinacao_no_pico_deg=INC[pico], inclinacao_max_deg=max(INC), inclinacao_final_deg=INC[-1],
                        erro_colocar_mm=erro_colocar, passos=len(Q), copo=list(self.a.copo), yaw_copo=self.a.yaw_copo,
                        latencia_mediana_s=float(np.median([x["latencia_s"] for x in log])) if log else None,
                        fator_tempo_real=round(n_passos / 30 / parede, 3), passos_atrasados=atrasados, falhas_resposta=falhas,
                        bloco_esgotou_passos=ex.esgotou)
        cauda = [f"bloco {x['bloco']:02d} no passo {x['passo']} lat {x['latencia_s']:.2f}s atraso {x['d']}" for x in log] + [json.dumps(metricas)]
        return {"ok": True, "metricas": metricas, "output_tail": "\n".join(cauda[-40:]), "run_dir": run_dir}

    def loop_principal(self):
        """Mantém a física e a câmera rodando a 30 Hz enquanto estiver em modo TELEOP."""
        while not self.parar.is_set():
            t0 = time.perf_counter()
            if self.modo == "teleop":
                with self.lock:
                    self._passo_teleop_vr()
                    self.sim.passo(self.n_sub)
                    qpos_snap = self.d.qpos.copy()
                    self._atualiza_shm()
                self._publica_todas(render_multicam(CENA, qpos_snap, self.a.afasta))
            time.sleep(max(0.002, 1 / 30 - (time.perf_counter() - t0)))

    def fechar(self):
        self.parar.set()
        if self.shm_qpos is not None:
            self.parar_janela.set()
        if self.tv_wrapper is not None:
            try:
                self.tv_wrapper.close()
            except Exception:
                pass


def cria_servidor_http(sim_hibrido: SimuladorHibrido, porta: int) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def manda(self, corpo, tipo="application/json", cod=200):
            b = corpo if isinstance(corpo, bytes) else json.dumps(corpo, ensure_ascii=False).encode("utf-8")
            self.send_response(cod)
            self.send_header("Content-Type", tipo)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            if self.path == "/estado":
                with sim_hibrido.lock:
                    return self.manda({
                        "ok": True,
                        "modo": sim_hibrido.modo,
                        "vr_conectado": sim_hibrido.vr_conectado,
                        "vr_ativo": sim_hibrido.vr_ativo,
                        "passo_concluido_vr": sim_hibrido.passo_concluido_vr,
                        "hud": sim_hibrido.hud,
                    })
            if self.path in ("/quadro/left_wrist", "/quadro/right_wrist"):
                with sim_hibrido.lock:
                    jpg = getattr(sim_hibrido, "ultimos_punhos", {}).get(self.path.rsplit("/", 1)[1])
                return self.manda(jpg, tipo="image/jpeg") if jpg else self.manda({"erro": "sem quadro"}, cod=404)
            if self.path.startswith("/quadro"):
                with sim_hibrido.lock:
                    return self.manda(sim_hibrido.ultimo_jpeg_limpo, tipo="image/jpeg")
            self.manda({"erro": "não encontrado"}, cod=404)

        def do_POST(self):
            tam = int(self.headers.get("Content-Length", "0"))
            corpo = json.loads(self.rfile.read(tam)) if tam > 0 else {}
            if self.path == "/hud":
                if corpo.get("limpar_concluido"):
                    with sim_hibrido.lock:
                        sim_hibrido.passo_concluido_vr = False
                sim_hibrido.atualiza_hud(corpo)
                with sim_hibrido.lock:
                    return self.manda({"ok": True, "hud": dict(sim_hibrido.hud)})
            if self.path == "/concluir_vr":
                with sim_hibrido.lock:
                    sim_hibrido.passo_concluido_vr = True
                return self.manda({"ok": True})
            if self.path == "/detectar":   # UnifoLM-ER-1 na câmera da cabeça (er1_cena.py em processo próprio)
                import subprocess, tempfile
                with sim_hibrido.lock:
                    q = sim_hibrido.d.qpos.copy()
                arq = Path(tempfile.mkstemp(suffix=".npy")[1]); np.save(arq, q)
                img = str(Path(corpo.get("saida_jpg", "/tmp/prometheus_chat/er1_ultimo.jpg")))
                r = subprocess.run([sys.executable, str(AQUI / "er1_cena.py"), str(arq), str(sim_hibrido.a.afasta),
                                    json.dumps(corpo.get("objetos", ["center of the white mug"])), img],
                                   capture_output=True, text=True, timeout=300, cwd=AQUI)
                arq.unlink(missing_ok=True)
                if r.returncode != 0:
                    return self.manda({"ok": False, "error": r.stderr[-800:]}, cod=500)
                return self.manda({"ok": True, "imagem": img, **json.loads(r.stdout.strip().splitlines()[-1])})
            if self.path == "/reset_cena":
                sim_hibrido.resetar_cena()
                return self.manda({"ok": True})
            if self.path == "/vla_place":
                try:
                    res = sim_hibrido.executar_vla_place(
                        run_dir=corpo["run_dir"],
                        porta_psi0=int(corpo.get("porta_psi0", 8781)),
                        blocos=int(corpo.get("blocos", 30)),
                        executa=int(corpo.get("executa", 15)),
                    )
                    return self.manda(res)
                except Exception as e:
                    return self.manda({"ok": False, "error": f"{type(e).__name__}: {e}"}, cod=500)
            self.manda({"erro": "não encontrado"}, cod=404)

    return ThreadingHTTPServer(("127.0.0.1", porta), Handler)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--porta-rpc", type=int, default=int(os.getenv("PSI0_SIM_PORT", "8782")))
    ap.add_argument("--zmq-port", type=int, default=5555)
    ap.add_argument("--copo", type=float, nargs=2, default=[0.265, -0.17])
    ap.add_argument("--yaw-copo", type=float, default=68.0)
    ap.add_argument("--afasta", type=float, default=0.10)
    ap.add_argument("--input-mode", choices=("controller", "hand"), default="controller")
    ap.add_argument("--display-mode", choices=("ego", "immersive", "pass-through"), default="ego")
    ap.add_argument("--sem-vr", action="store_true", help="não sobe o Vuer (:8012), apenas MuJoCo + RPC VLA")
    ap.add_argument("--sem-janela", action="store_true", help="não abre a janela 3D do MuJoCo")
    args = ap.parse_args()

    sim_hibrido = SimuladorHibrido(args)
    srv = cria_servidor_http(sim_hibrido, args.porta_rpc)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"[sim_hibrido] RPC ativo em http://127.0.0.1:{args.porta_rpc} (MuJoCo único compartilhado)", flush=True)
    try:
        sim_hibrido.loop_principal()
    except KeyboardInterrupt:
        pass
    finally:
        srv.shutdown()
        sim_hibrido.fechar()


if __name__ == "__main__":
    main()
