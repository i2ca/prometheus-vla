"""Arnes de politica no loop (GPT-6 Astra Direct) para a pega do copo no G1.

Contrato copiado de hybrid_rollout/robodojo/skill/context/eef_control.md do
release GPT-as-Policy, adaptado a esta cena. O agente ve apenas RGB e a pose
MEDIDA da palma; nao recebe a posicao do copo. Cada comando pede uma pose
absoluta de palma limitada a 5 cm / 0,35 rad da pose medida atual, a IK converte
e o PD executa. Pedir um alvo nao e prova de ter chegado: confira a imagem e a
pose medida no passo seguinte.

Desvio consciente do contrato original, declarado: la um "step" e um quadro de
controle a 25 Hz (0,04 s); aqui um step e 0,2 s (6 quadros a 30 Hz), para caber
em cota. Teto por comando: 5 passos = 1 s de movimento.

Estado entre processos vive em disco (state.npz por passo), entao start/act/finish
sao comandos independentes e o episodio sobrevive a queda de sessao ou de cota.
"""
import json
import os
from pathlib import Path

import cv2
import mujoco
import numpy as np

from g1_sim import G1Sim, ARM_JOINTS, HAND_JOINTS, ROOT, load_seed
from kinematics import ArmIK, PALM

# A IK do Direct usa os mesmos graus de liberdade do controlador scriptado na varredura 03
# (cintura em yaw + 7 juntas do braco), senao a comparacao seria injusta: varios alvos da
# mesa nao sao alcancaveis so com o braco. A cintura anda com peso menor que o braco.
IK_JOINTS = ["waist_yaw_joint"] + list(ARM_JOINTS)
WAIST_WEIGHT = 0.3

FPS = 30
STEP_SECONDS = 0.2                 # duracao de um "step" da politica
MAX_STEPS_PER_CALL = 5
MAX_TARGET_DISTANCE_M = 0.05
MAX_TARGET_ROTATION_RAD = 0.35
MAX_JOINT_SPEED_DEG_S = 240.0
SETTLE_FRAMES = 8                  # quadros parados no fim do comando: o PD alcanca o alvo antes da foto
CORRECTION_ITERATIONS = 3          # malha proprioceptiva no fim do comando (igual correct_palm do run_attempt)
CORRECTION_GAIN = 0.5
CORRECTION_BIAS_M = 0.02
CORRECTION_FRAMES = 8
# O portao so barra o que e inalcancavel de verdade. Erro pequeno de IK e normal e o robo
# executa o melhor esforco, como faria o controlador real; o valor vai no relatorio da acao.
IK_TOLERANCE_M = 0.03
IK_TOLERANCE_RAD = 0.15
IK_SETTLE_TOLERANCE_M = 0.003      # alvo da malha proprioceptiva
CAMERAS = ("head_camera", "left_wrist_camera", "right_wrist_camera")
VIDEO_GRID = ("left_wrist_camera", "head_camera", "right_wrist_camera",
              "view_left", "view_center", "side_view")
TILT_LIMIT_DEG = 15.0
# faixa util da D435i: o minimo cai com a resolucao (Intel: ~0,195 m em 848x480, ~0,105 m em 424x240)
DEPTH_MIN_BY_WIDTH, DEPTH_MAX_M = {848: 0.195, 424: 0.105}, 3.0
TRUNK = {"torso_link", "pelvis", "waist_yaw_link", "waist_roll_link", "head_link"}
# Folga minima entre cotovelo/punho/mao e tronco/braco esquerdo na trajetoria planejada. Calibrada no
# direct-astra-03: a chamada 7 passava a 4,2 cm no plano e encostou na execucao (mao com kp baixo cede,
# a malha de correcao desloca o alvo em ate 2 cm, o PD atrasa); chamadas normais ficam acima de 6,4 cm.
# O cotovelo e rigido e so sofre o desvio da correcao; a folga maior e para punho/mao, cujos dedos cedem.
# Ombro (roll/yaw): folga normal 2,2 a 3,8 cm do tronco; no direct-astra-p-27 chegou a 1,0 cm e encostou na correcao.
SELF_CLEARANCE_M = {"right_elbow": 0.025, "right_wrist": 0.05, "right_hand": 0.05, "right_palm": 0.05,
                    "right_shoulder_roll": 0.012, "right_shoulder_yaw": 0.012}
DISTAL_ARM = tuple(SELF_CLEARANCE_M)


def quat_of(mat):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(mat, float).ravel())
    return q / np.linalg.norm(q)


def mat_of(quat):
    q = np.asarray(quat, float)
    q = q / np.linalg.norm(q)
    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, q)
    return mat.reshape(3, 3)


def quat_angle(a, b):
    d = abs(float(np.dot(np.asarray(a, float) / np.linalg.norm(a), np.asarray(b, float) / np.linalg.norm(b))))
    return float(2 * np.arccos(min(1.0, d)))


class DirectEpisode:
    """Um episodio; cada chamada carrega o estado do passo anterior e grava o proximo."""

    def __init__(self, episode_dir):
        self.dir = Path(episode_dir)
        self.meta = json.loads((self.dir / "episode.json").read_text()) if (self.dir / "episode.json").exists() else None
        self.sim = None

    # ---------- construcao ----------
    def _build(self, scene):
        sim = G1Sim(scene)
        self.sim, self.m, self.d = sim, sim.m, sim.d
        self.arm_seed, self.hand_seed, seed_fps = load_seed(0)
        self.grasp_index = int(np.flatnonzero(self.hand_seed[:, 4] > 0.5)[0])
        self.ik = ArmIK(sim, IK_JOINTS)
        self.ik.weights = np.r_[WAIST_WEIGHT, np.ones(len(ARM_JOINTS))]
        assert seed_fps == FPS, f"semente a {seed_fps} Hz, arnes a {FPS} Hz"

    def start(self, case, scene="scene_grasp.xml", max_calls=60, max_sim_seconds=40.0, video=True):
        self.dir.mkdir(parents=True, exist_ok=False)
        self._build(scene)
        sim, m, d = self.sim, self.m, self.d
        p = dict(case)
        home = np.asarray(p["home_arm_pose"], float)
        for names, values in ((ARM_JOINTS, home), (HAND_JOINTS, self.hand_seed[0]),
                              ([j.replace("right_", "left_") for j in ARM_JOINTS], np.asarray(p["left_arm_pose"], float))):
            for name, value in zip(names, values):
                d.qpos[m.jnt_qposadr[m.joint(name).id]] = value
        sim.q_des[:] = d.qpos[sim.qadr]
        sim.set_targets(HAND_JOINTS, self.hand_seed[0], kp=p["hand_kp"], kd=1)
        yaw = np.radians(p["cup_yaw_deg"])
        sim.place_cup([*p["cup_xy"], 0.752], quat=(np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)))
        reference = ArmIK(sim).reference_grasp(self.arm_seed, self.hand_seed, self.grasp_index, p["offset"], p["gain"])
        mujoco.mj_forward(m, d)
        self.meta = {"case": p, "scene": scene, "max_calls": max_calls, "max_sim_seconds": max_sim_seconds,
                     "video": bool(video), "fps": FPS, "step_seconds": STEP_SECONDS,
                     "hand_open": reference["open"].tolist(), "hand_close": reference["close"].tolist(),
                     "limits": {"max_target_distance_m": MAX_TARGET_DISTANCE_M,
                                "max_target_rotation_rad": MAX_TARGET_ROTATION_RAD,
                                "max_steps_per_call": MAX_STEPS_PER_CALL,
                                "ik_tolerance_m": IK_TOLERANCE_M, "ik_tolerance_rad": IK_TOLERANCE_RAD,
                                "max_joint_speed_deg_s": MAX_JOINT_SPEED_DEG_S},
                     "calls": 0, "frames": 0, "aborted": None}
        (self.dir / "episode.json").write_text(json.dumps(self.meta, indent=2) + "\n")
        for sub in ("obs", "truth", "actions", "frames"):
            (self.dir / sub).mkdir()
        self._settle(15)
        return self._observe(reason_of_previous=None, executed=None)

    # ---------- estado ----------
    def _state_path(self, call):
        return self.dir / "obs" / f"step-{call:03d}" / "state.npz"

    def _save_state(self, path):
        d, sim = self.d, self.sim
        np.savez(path, qpos=d.qpos, qvel=d.qvel, act=d.act, ctrl=d.ctrl,
                 warmstart=d.qacc_warmstart, time=np.array([d.time]),
                 q_des=sim.q_des, kp=sim.kp, kd=sim.kd)

    def load(self):
        self.meta = json.loads((self.dir / "episode.json").read_text())
        self._build(self.meta["scene"])
        z = np.load(self._state_path(self.meta["calls"]))
        d, sim = self.d, self.sim
        d.qpos[:] = z["qpos"]; d.qvel[:] = z["qvel"]; d.act[:] = z["act"]; d.ctrl[:] = z["ctrl"]
        d.qacc_warmstart[:] = z["warmstart"]; d.time = float(z["time"][0])
        sim.q_des[:] = z["q_des"]; sim.kp[:] = z["kp"]; sim.kd[:] = z["kd"]
        mujoco.mj_forward(self.m, d)
        return self

    # ---------- fisica ----------
    def _substeps(self, nframes, samples=None):
        """Avanca nframes quadros de controle; devolve violacoes observadas por subpasso."""
        m, d, sim = self.m, self.d, self.sim
        tampo = m.geom("tampo").id
        hit_table, hit_self = set(), set()
        for _ in range(nframes):
            until = d.time + 1.0 / FPS
            while d.time < until - m.opt.timestep / 2:
                tau = sim.kp * (sim.q_des - d.qpos[sim.qadr]) - sim.kd * d.qvel[sim.vadr] + d.qfrc_bias[sim.vadr]
                d.ctrl[:] = np.clip(tau, m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1])
                mujoco.mj_step(m, d)
                for i in range(d.ncon):
                    c = d.contact[i]
                    b1 = m.body(int(m.geom_bodyid[c.geom1])).name
                    b2 = m.body(int(m.geom_bodyid[c.geom2])).name
                    if tampo in (c.geom1, c.geom2):
                        other = b2 if c.geom1 == tampo else b1
                        if other.startswith("right_hand") or other == PALM:
                            hit_table.add(other)
                    r1, r2 = b1.startswith("right_"), b2.startswith("right_")
                    if (r1 and b2 in TRUNK) or (r2 and b1 in TRUNK) or (r1 and b2.startswith("left_")) or (r2 and b1.startswith("left_")):
                        hit_self.add(tuple(sorted((b1, b2))))
            mujoco.mj_forward(m, d)
            self.meta["frames"] += 1
            if samples is not None:
                samples.append(self._sample())
            if self.meta["video"]:
                self._write_frame()
        return sorted(hit_table), sorted(hit_self)

    def _settle(self, nframes):
        return self._substeps(nframes, samples=None)

    def _sample(self):
        sim, m, d = self.sim, self.m, self.d
        up = d.xmat[sim.cup_body].reshape(3, 3)[:, 2]
        links = sorted(sim.finger_contacts())
        return {"t": round(float(d.time), 3), "cup": sim.cup_pos().tolist(),
                "cup_tilt_deg": float(np.degrees(np.arccos(np.clip(up[2], -1, 1)))),
                "palm": sim.body_pos(PALM).tolist(),
                "fingers": sorted({n.split("_")[0] for n in links}),
                "cup_table": sim.cup_table_contact()}

    # ---------- render ----------
    def _renderer(self, w, h):
        if not hasattr(self, "_r") or self._r.height != h or self._r.width != w:
            if hasattr(self, "_r"):
                self._r.close()
            self._r = mujoco.Renderer(self.m, h, w)
        return self._r

    def _render(self, cam, size):
        r = self._renderer(*size)
        r.update_scene(self.d, camera=cam)
        return r.render().copy()

    def _write_frame(self):
        tiles = [self._render(c, (426, 240)) for c in VIDEO_GRID]
        img = np.concatenate([np.concatenate(tiles[i:i + 3], axis=1) for i in (0, 3)], axis=0)
        band = np.full((32, img.shape[1], 3), 24, np.uint8)
        cv2.putText(band, f"{self.dir.name} | DIRECT | chamada {self.meta['calls']} | {self.meta['frames'] / FPS:5.1f}s",
                    (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        img = np.concatenate([band, img], axis=0)
        path = self.dir / "frames" / f"{self.meta['frames']:06d}.jpg"
        cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])

    # ---------- sensor de profundidade ----------
    def depth_probe(self, points, cam="head_camera", size=(848, 480)):
        """Ponto 3D (referencial do ambiente, o mesmo da palma) sob cada pixel (u, v) da camera da cabeca.
        E o que uma RealSense D435i (a camera de profundidade real da cabeca do G1) entrega: sai do buffer de
        profundidade renderizado, nao da pose de objeto nenhum. Ruido tipo D435 (sigma cresce com z^2) com
        semente fixa por passo e pixel, e sem retorno abaixo do alcance minimo ou fora da imagem."""
        w, h = size
        r = self._renderer(w, h)
        r.enable_depth_rendering()
        r.update_scene(self.d, camera=cam)
        depth = r.render().copy()
        r.disable_depth_rendering()
        c = self.m.camera(cam)
        fovy = np.radians(self.m.cam_fovy[c.id])
        f = (h / 2) / np.tan(fovy / 2)
        pos, mat = self.d.cam_xpos[c.id], self.d.cam_xmat[c.id].reshape(3, 3)
        out = []
        for u, v in points:
            u, v = int(round(u)), int(round(v))
            if not (0 <= u < w and 0 <= v < h):
                out.append({"u": u, "v": v, "valid": False, "why": "fora da imagem (848x480)"}); continue
            z = float(depth[v, u])
            if z < DEPTH_MIN_BY_WIDTH.get(w, 0.195) or z > DEPTH_MAX_M:
                out.append({"u": u, "v": v, "valid": False, "why": "sem retorno (perto ou longe demais)"}); continue
            rng = np.random.default_rng(abs(hash((self.meta["calls"], u, v))) % (2 ** 32))
            z = z + rng.normal(0.0, 0.001 + 0.002 * z * z)
            # camera do MuJoCo olha para -z, x para a direita, y para cima
            ray = np.array([(u - w / 2) / f, -(v - h / 2) / f, -1.0]) * z
            world = pos + mat @ ray
            out.append({"u": u, "v": v, "valid": True, "depth_m": round(z, 4),
                        "point": [round(float(x), 4) for x in world]})
        return out

    def _hand_meshes(self):
        """Vertices das geometrias de colisao da mao direita, no referencial de cada geom (modelo do proprio robo)."""
        if not hasattr(self, "_hm"):
            m, out = self.m, []
            for g in range(m.ngeom):
                name = m.body(int(m.geom_bodyid[g])).name
                if not name.startswith("right_hand") or not (m.geom_contype[g] or m.geom_conaffinity[g]):
                    continue
                if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                    mid = m.geom_dataid[g]
                    v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]].copy()
                else:  # primitiva: aproxima pelos cantos da caixa envolvente
                    h = m.geom_size[g][:3] if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX else np.full(3, m.geom_rbound[g])
                    v = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * h
                out.append((g, name, v))
            self._hm = out
        return self._hm

    def _geometry_of(self, d2):
        """Geometria da mao a partir da malha do proprio robo, lida de um MjData qualquer: ponta de cada dedo = vertice
        do elo distal mais longe da base do dedo; lowest_point = ponto mais baixo da mao inteira."""
        m = self.m
        pts = {}
        for g, name, v in self._hand_meshes():
            pts.setdefault(name, []).append(d2.geom_xpos[g] + (d2.geom_xmat[g].reshape(3, 3) @ v.T).T)
        pts = {k: np.vstack(v) for k, v in pts.items()}
        base = lambda n: d2.xpos[m.body(f"right_hand_{n}_link").id]
        far = lambda distal, root: pts[f"right_hand_{distal}_link"][
            np.argmax(np.linalg.norm(pts[f"right_hand_{distal}_link"] - base(root), axis=1))]
        allp = np.vstack(list(pts.values()))
        opp = (base("index_1") + base("middle_1")) / 2
        return {"thumb_tip": far("thumb_2", "thumb_0"), "index_tip": far("index_1", "index_0"),
                "middle_tip": far("middle_1", "middle_0"), "aperture_center": (base("thumb_2") + opp) / 2,
                "lowest_point": allp[np.argmin(allp[:, 2])]}

    def preview(self, position, quaternion_wxyz, steps=3, gripper="keep"):
        """Planejamento da propria mao, sem mover o robo: roda o mesmo comando que act() executaria numa COPIA do
        robo com a dinamica e os controladores dele (inclusive o punho de 5 Nm que cede sob o peso da mao), mas com
        TODOS os contatos desligados, entao a previsao nao sabe nada do ambiente. Devolve a geometria no fim e o
        ponto mais baixo das pontas dos dedos ao longo do caminho."""
        pos = np.asarray(position, float)
        target_mat = mat_of(quaternion_wxyz)
        cur_pos, cur_mat = self._palm_pose()
        dist, rot = float(np.linalg.norm(pos - cur_pos)), quat_angle(quat_of(cur_mat), quaternion_wxyz)
        e2 = DirectEpisode(self.dir).load()
        e2.meta["video"] = False
        # desliga so o contato com o AMBIENTE (mesa, objetos, chao); os contatos do robo com ele mesmo continuam,
        # senao a mao (dedos que encostam entre si e na palma) se comporta diferente do real
        pelvis_root = e2.m.body_rootid[e2.m.body("pelvis").id]
        for g in range(e2.m.ngeom):
            if e2.m.body_rootid[e2.m.geom_bodyid[g]] != pelvis_root:
                e2.m.geom_contype[g] = 0; e2.m.geom_conaffinity[g] = 0
        sim = e2.sim
        start_pos, start_mat = e2._palm_pose()          # pose medida da copia, igual ao que _execute usa
        rot_vec = cv2.Rodrigues(target_mat @ start_mat.T)[0].ravel()
        hand_now, hand_goal = sim.q(HAND_JOINTS), e2._hand_goal(gripper)
        frames = max(1, int(round(int(steps) * STEP_SECONDS * FPS)))
        max_step = np.radians(MAX_JOINT_SPEED_DEG_S) / FPS
        q, lowest = sim.q(IK_JOINTS), (np.inf, None)
        for i in range(frames + SETTLE_FRAMES):
            if i < frames:
                u = (i + 1) / frames; su = u * u * (3 - 2 * u)
                q, _ = e2.ik.solve(start_pos + su * (pos - start_pos), cv2.Rodrigues(rot_vec * su)[0] @ start_mat,
                                   q, q, iterations=60, max_step=max_step)
                sim.set_targets(IK_JOINTS, q)
                sim.set_targets(HAND_JOINTS, hand_now + su * (hand_goal - hand_now))
            e2._substeps(1)
            lp = e2._geometry_of(e2.d)["lowest_point"]
            if lp[2] < lowest[0]:
                lowest = (float(lp[2]), [round(float(x), 4) for x in lp])
        end = e2._geometry_of(e2.d)
        palm_end, _ = e2._palm_pose()
        r = lambda v: [round(float(x), 4) for x in v]
        return {"within_limits": dist <= MAX_TARGET_DISTANCE_M and rot <= MAX_TARGET_ROTATION_RAD,
                "distance_m": round(dist, 4), "rotation_rad": round(rot, 3),
                "predicted_palm_end": r(palm_end), "end": {k: r(v) for k, v in end.items()},
                "lowest_hand_point_along_path": {"z": round(lowest[0], 4), "point": lowest[1]},
                "note": "own-body dynamics (self-contacts kept, contacts with anything else removed): it does not know about the table or objects"}

    def _since(self, key, active):
        now = self.meta["frames"] / FPS
        if not active:
            self.meta[key] = None
            return 0.0
        if self.meta.get(key) is None:
            self.meta[key] = now
        return round(now - self.meta[key], 2)

    def _hand_geometry(self):
        """Propriocepcao da mao (angulos medidos das juntas + malha do proprio robo). Nao usa nada do objeto."""
        r = lambda v: [round(float(x), 4) for x in v]
        g = {k: r(v) for k, v in self._geometry_of(self.d).items()}
        g["note"] = "from measured joint angles and the robot's own hand mesh; same frame as palm and depth probes"
        return g

    # ---------- observacao ----------
    def _palm_pose(self):
        pos = self.sim.body_pos(PALM)
        mat = self.d.xmat[self.m.body(PALM).id].reshape(3, 3)
        return pos, mat

    def _observe(self, reason_of_previous, executed):
        call = self.meta["calls"]
        step_dir = self.dir / "obs" / f"step-{call:03d}"
        step_dir.mkdir(exist_ok=True)
        for cam in CAMERAS:
            rgb = self._render(cam, (848, 480) if cam == "head_camera" else (424, 240))
            cv2.imwrite(str(step_dir / f"{cam}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        pos, mat = self._palm_pose()
        hand_q = self.sim.q(HAND_JOINTS)
        opened = np.asarray(self.meta["hand_open"], float)
        closed = np.asarray(self.meta["hand_close"], float)
        denom = float(np.linalg.norm(closed - opened) ** 2) or 1.0
        closure = float(np.clip(np.dot(hand_q - opened, closed - opened) / denom, 0.0, 1.0))
        obs = {
            "episode": self.dir.name,
            "step_id": call,
            "remaining_calls": self.meta["max_calls"] - call,
            "elapsed_sim_seconds": round(self.meta["frames"] / FPS, 2),
            "remaining_sim_seconds": round(self.meta["max_sim_seconds"] - self.meta["frames"] / FPS, 2),
            "task": self.meta["case"].get("instruction"),
            "images": [{"camera": c, "path": str((step_dir / f"{c}.png").resolve())} for c in CAMERAS],
            "current_palm": {"position": [round(v, 5) for v in pos.tolist()],
                             "quaternion_wxyz": [round(v, 6) for v in quat_of(mat).tolist()]},
            "arm_joints_rad": [round(v, 5) for v in self.sim.q(ARM_JOINTS).tolist()],
            "hand_geometry": self._hand_geometry(),
            "waist_yaw_rad": round(float(self.sim.q(["waist_yaw_joint"])[0]), 5),
            "gripper_closure": round(closure, 3),
            # relogios do proprio estado (propriocepcao): ha quanto tempo a mao esta fechada e a palma parada
            "gripper_closed_for_s": self._since("closed_since", closure > 0.5),
            "palm_still_for_s": self._since("still_since", True),
            "previous_reason": reason_of_previous,
            "previous_execution": executed,
            "limits": self.meta["limits"],
            "finished": bool(self.meta["aborted"]) or call >= self.meta["max_calls"]
                        or self.meta["frames"] / FPS >= self.meta["max_sim_seconds"],
            "next_call": (f"scripts/direct_act.py {self.dir} --action <arquivo.json>"),
        }
        (step_dir / "observation.json").write_text(json.dumps(obs, indent=2) + "\n")
        truth = {"step_id": call, **self._sample(),
                 "cup_xy_privileged": self.meta["case"]["cup_xy"]}
        (self.dir / "truth" / f"step-{call:03d}.json").write_text(json.dumps(truth, indent=2) + "\n")
        self._save_state(step_dir / "state.npz")
        (self.dir / "episode.json").write_text(json.dumps(self.meta, indent=2) + "\n")
        return obs

    # ---------- acao ----------
    def act(self, action):
        """Valida e executa uma acao. Rejeicao NAO executa nada e nao consome chamada de fisica."""
        if (self.meta["aborted"] or self.meta["calls"] >= self.meta["max_calls"]
                or self.meta["frames"] / FPS >= self.meta["max_sim_seconds"]):
            # orcamento e aborto sao limites duros, nao conselho: nada mais toca na fisica
            return self._observe(reason_of_previous=action.get("reason"),
                                 executed={"status": "episode_finished",
                                           "why": "aborto por contato" if self.meta["aborted"] else "orcamento esgotado"})
        errors = []
        target = action.get("target") or {}
        pos = target.get("position")
        quat = target.get("quaternion_wxyz")
        grip = target.get("gripper", "keep")
        steps = action.get("steps", 1)
        reason = (action.get("reason") or "").strip()
        if not reason:
            errors.append("reason vazio: descreva a evidencia visual e o proposito")
        if not (isinstance(pos, (list, tuple)) and len(pos) == 3):
            errors.append("target.position precisa de 3 numeros (metros, origem do ambiente)")
        if not (isinstance(quat, (list, tuple)) and len(quat) == 4):
            errors.append("target.quaternion_wxyz precisa de 4 numeros (wxyz, unitario)")
        if grip not in ("keep", "open", "closed") and not isinstance(grip, (int, float)):
            errors.append("target.gripper deve ser keep|open|closed ou fracao 0..1")
        if not isinstance(steps, int) or not (1 <= steps <= MAX_STEPS_PER_CALL):
            errors.append(f"steps deve ser inteiro entre 1 e {MAX_STEPS_PER_CALL}")
        if errors:
            return self._reject(errors, action)

        cur_pos, cur_mat = self._palm_pose()
        cur_quat = quat_of(cur_mat)
        pos = np.asarray(pos, float)
        distance = float(np.linalg.norm(pos - cur_pos))
        rotation = quat_angle(cur_quat, quat)
        if distance > MAX_TARGET_DISTANCE_M:
            errors.append(f"alvo a {distance*100:.1f} cm da palma medida; teto {MAX_TARGET_DISTANCE_M*100:.0f} cm")
        if rotation > MAX_TARGET_ROTATION_RAD:
            errors.append(f"rotacao de {rotation:.3f} rad; teto {MAX_TARGET_ROTATION_RAD} rad")
        if errors:
            return self._reject(errors, action)

        target_mat = mat_of(quat)
        q_now = self.sim.q(IK_JOINTS)
        q_goal, ik_info = self.ik.solve(pos, target_mat, q_now, q_now, iterations=200)
        if ik_info["position_error_m"] > IK_TOLERANCE_M or ik_info["orientation_error_rad"] > IK_TOLERANCE_RAD:
            reachable, _ = self.ik.fk(q_goal)
            errors.append(f"IK nao converge nesse alvo: {ik_info['position_error_m']*1000:.1f} mm / "
                          f"{ik_info['orientation_error_rad']:.3f} rad (limites {IK_TOLERANCE_M*1000:.0f} mm / {IK_TOLERANCE_RAD} rad). "
                          f"Ponto mais proximo que este braco alcanca nessa orientacao: "
                          f"[{reachable[0]:.3f}, {reachable[1]:.3f}, {reachable[2]:.3f}]")
            return self._reject(errors, action, ik={**ik_info, "nearest_reachable": [round(v, 4) for v in reachable.tolist()]})

        hit = self._predict_self_collision(pos, target_mat, steps, grip)
        if hit:
            frame, (b1, b2), dist = hit
            frames = max(1, int(round(steps * STEP_SECONDS * FPS)))
            errors.append(f"trajetoria passa perto demais do proprio corpo ({b1} x {b2}, folga abaixo da margem de "
                          f"seguranca) em {100 * (frame + 1) // frames}% do movimento; nada foi executado. "
                          f"Escolha um alvo ou orientacao que afaste a mao e o punho do tronco.")
            return self._reject(errors, action, ik={**ik_info, "predicted_self_collision": [b1, b2], "clearance_m": round(float(dist), 4),
                                                    "at_fraction": round((frame + 1) / frames, 2)})

        # ensaio dinamico do comando inteiro (movimento, acomodacao e correcao final) numa copia do robo com a dinamica
        # dele, contatos do corpo consigo mesmo LIGADOS e ambiente DESLIGADO: pega a autocolisao que a checagem
        # cinematica nao ve (o braco fica atras do plano quando gira carregando o proprio peso; direct-astra-p-33)
        rehearsal = self._rehearse(pos, target_mat, grip, steps, reason, action, ik_info)
        if rehearsal and rehearsal.get("self_collision"):
            b1, b2 = rehearsal["self_collision"][0]
            errors.append(f"ensaio do movimento com a dinamica do braco: {b1} encosta em {b2}; nada foi executado. "
                          f"Mude o alvo ou a orientacao para afastar o braco do tronco, ou divida a rotacao em passos menores.")
            return self._reject(errors, action, ik={**ik_info, "rehearsal_self_collision": [b1, b2]})

        return self._execute(pos, target_mat, grip, steps, reason, action, ik_info)

    def _rehearse(self, pos, target_mat, grip, steps, reason, action, ik_info):
        e2 = DirectEpisode(self.dir).load()
        e2.meta["video"] = False
        root = e2.m.body_rootid[e2.m.body("pelvis").id]
        for g in range(e2.m.ngeom):
            if e2.m.body_rootid[e2.m.geom_bodyid[g]] != root:
                e2.m.geom_contype[g] = 0; e2.m.geom_conaffinity[g] = 0
        out = e2._execute(pos, target_mat, grip, steps, reason, action, ik_info, dry=True)
        return out.get("aborted")

    def _is_self_contact(self, b1, b2):
        r1, r2 = b1.startswith("right_"), b2.startswith("right_")
        return (r1 and b2 in TRUNK) or (r2 and b1 in TRUNK) or (r1 and b2.startswith("left_")) or (r2 and b1.startswith("left_"))

    def _hand_goal(self, grip):
        opened = np.asarray(self.meta["hand_open"], float)
        closed = np.asarray(self.meta["hand_close"], float)
        if grip == "keep":
            return self.sim.q_des[[self.sim.act_joint[j] for j in HAND_JOINTS]].copy()
        if grip == "open":
            return opened
        if grip == "closed":
            return closed
        return opened + float(grip) * (closed - opened)

    def _predict_self_collision(self, pos, target_mat, steps, grip):
        """Percorre cinematicamente a trajetoria que _execute vai comandar, numa copia do estado.
        O robo conhece o proprio corpo, entao recusar movimento que bate no tronco ou no outro braco
        e checagem de planejador, nao informacao privilegiada. A mesa NAO entra: ambiente e com as cameras.
        Devolve (quadro, (corpo1, corpo2)) do primeiro contato previsto, ou None."""
        m = self.m
        d2 = mujoco.MjData(m)
        d2.qpos[:] = self.d.qpos
        start_pos, start_mat = self._palm_pose()
        rot_vec = cv2.Rodrigues(target_mat @ start_mat.T)[0].ravel()
        hand_now, hand_goal = self.sim.q(HAND_JOINTS), self._hand_goal(grip)
        ik_adr = [m.jnt_qposadr[m.joint(j).id] for j in IK_JOINTS]
        hand_adr = [m.jnt_qposadr[m.joint(j).id] for j in HAND_JOINTS]
        frames = max(1, int(round(steps * STEP_SECONDS * FPS)))
        max_step = np.radians(MAX_JOINT_SPEED_DEG_S) / FPS
        q = self.sim.q(IK_JOINTS)
        start_clear = self._clearance(d2)[0]   # sobra de folga agora; se ja esta perto, so barra o que aproxima mais
        for i in range(frames):
            u = (i + 1) / frames
            su = u * u * (3 - 2 * u)
            q, _ = self.ik.solve(start_pos + su * (pos - start_pos), cv2.Rodrigues(rot_vec * su)[0] @ start_mat,
                                 q, q, iterations=60, max_step=max_step)
            d2.qpos[ik_adr] = q
            d2.qpos[hand_adr] = hand_now + su * (hand_goal - hand_now)
            spare, pair, dist = self._clearance(d2)
            if spare < 0 and spare < start_clear - 1e-4:
                return i, pair, dist
        # ja comecou dentro da margem: so passa se o comando AUMENTAR a folga (a correcao final e o atraso do PD
        # somam ate ~2 cm; ficar parado perto do tronco foi o que bateu o ombro no direct-astra-p-27)
        if start_clear < 0 and spare <= start_clear + 1e-4:
            return frames - 1, pair, dist
        return None

    def _clearance(self, d2):
        """Menor distancia entre partes distais do braco direito e tronco/braco esquerdo (so cinematica)."""
        m = self.m
        if not hasattr(self, "_pairs"):
            col = lambda g: m.geom_contype[g] or m.geom_conaffinity[g]
            bn = lambda g: m.body(int(m.geom_bodyid[g])).name
            arm = [g for g in range(m.ngeom) if bn(g).startswith(DISTAL_ARM) and col(g)]
            body = [g for g in range(m.ngeom) if (bn(g) in TRUNK or bn(g).startswith("left_")) and col(g)]
            margin = lambda name: next(v for k, v in SELF_CLEARANCE_M.items() if name.startswith(k))
            self._pairs = [(g1, g2, bn(g1), bn(g2), margin(bn(g1))) for g1 in arm for g2 in body]
        mujoco.mj_kinematics(m, d2)
        ft, best = np.zeros(6), (np.inf, None, 0.0)
        for g1, g2, n1, n2, marg in self._pairs:
            dist = mujoco.mj_geomDistance(m, d2, g1, g2, 0.2, ft)
            if dist - marg < best[0]:        # folga que sobra alem da margem daquela parte
                best = (dist - marg, (n1, n2), dist)
        return best

    def _reject(self, errors, action, ik=None):
        self.meta["calls"] += 1
        record = {"accepted": False, "errors": errors, "ik": ik, "action": action}
        (self.dir / "actions" / f"call-{self.meta['calls']:03d}.json").write_text(json.dumps(record, indent=2) + "\n")
        # rejeicao nao mexe no mundo: o mesmo estado e reobservado num passo novo
        (self.dir / "obs" / f"step-{self.meta['calls']:03d}").mkdir(exist_ok=True)
        obs = self._observe(reason_of_previous=action.get("reason"),
                            executed={"status": "rejected", "errors": errors})
        obs["rejected"] = True
        return obs

    def _execute(self, pos, target_mat, grip, steps, reason, action, ik_info, dry=False):
        sim = self.sim
        start_pos, start_mat = self._palm_pose()
        rot_vec = cv2.Rodrigues(target_mat @ start_mat.T)[0].ravel()
        opened = np.asarray(self.meta["hand_open"], float)
        closed = np.asarray(self.meta["hand_close"], float)
        hand_now = sim.q(HAND_JOINTS)
        if grip == "keep":
            hand_goal = sim.q_des[[sim.act_joint[j] for j in HAND_JOINTS]].copy()
        elif grip == "open":
            hand_goal = opened
        elif grip == "closed":
            hand_goal = closed
        else:
            hand_goal = opened + float(grip) * (closed - opened)
        frames = max(1, int(round(steps * STEP_SECONDS * FPS)))
        max_step = np.radians(MAX_JOINT_SPEED_DEG_S) / FPS
        q = sim.q(IK_JOINTS)
        samples, table_hits, self_hits = [], set(), set()
        for i in range(frames):
            u = (i + 1) / frames
            su = u * u * (3 - 2 * u)
            q, _ = self.ik.solve(start_pos + su * (pos - start_pos),
                                 cv2.Rodrigues(rot_vec * su)[0] @ start_mat,
                                 q, q, iterations=60, max_step=max_step)
            sim.set_targets(IK_JOINTS, q)
            sim.set_targets(HAND_JOINTS, hand_now + su * (hand_goal - hand_now))
            table, selfc = self._substeps(1, samples)
            table_hits.update(table); self_hits.update(selfc)
            if table or selfc:
                self.meta["aborted"] = {"call": self.meta["calls"] + 1, "frame": i,
                                        "hand_table": sorted(table_hits), "self_collision": sorted(self_hits)}
                break
        if not self.meta["aborted"]:
            # deixa o PD alcancar antes de observar: sem isso a pose medida e a de um braco em movimento
            # e o comando seguinte nasce violando o teto de rotacao (visto no selftest: 0,37 rad)
            table, selfc = self._substeps(SETTLE_FRAMES, samples)
            table_hits.update(table); self_hits.update(selfc)
            # malha proprioceptiva de posicao, igual ao correct_palm do controlador scriptado: o PD fica
            # atras do alvo com o braco estendido, entao o alvo e deslocado por metade do erro medido.
            # (A orientacao NAO e corrigida: o punho tem 5 Nm de torque e nao acompanha giro grande.)
            bias = np.zeros(3)
            for _ in range(CORRECTION_ITERATIONS):
                actual, _mat = self._palm_pose()
                err = pos - actual
                if np.linalg.norm(err) < IK_SETTLE_TOLERANCE_M:
                    break
                bias = np.clip(bias + CORRECTION_GAIN * err, -CORRECTION_BIAS_M, CORRECTION_BIAS_M)
                q_corr, info_corr = self.ik.solve(pos + bias, target_mat, sim.q(IK_JOINTS), q, iterations=200)
                if info_corr["position_error_m"] > IK_TOLERANCE_M:  # alvo deslocado saiu do alcance
                    break
                # a correcao e assistencia do arnes: nao pode aproximar o braco do proprio corpo alem da margem
                # (foi a correcao que bateu o ombro no tronco no direct-astra-p-27)
                dc = mujoco.MjData(self.m); dc.qpos[:] = self.d.qpos
                now_spare = self._clearance(dc)[0]
                dc.qpos[[self.m.jnt_qposadr[self.m.joint(j).id] for j in IK_JOINTS]] = q_corr
                corr_spare = self._clearance(dc)[0]
                if corr_spare < 0 and corr_spare < now_spare:
                    break
                sim.set_targets(IK_JOINTS, q_corr)
                table, selfc = self._substeps(CORRECTION_FRAMES, samples)
                table_hits.update(table); self_hits.update(selfc)
                if table or selfc:
                    self.meta["aborted"] = {"call": self.meta["calls"] + 1, "frame": -1,
                                            "hand_table": sorted(table_hits), "self_collision": sorted(self_hits)}
                    break
        if not dry and float(np.linalg.norm(self._palm_pose()[0] - start_pos)) > 0.005:
            self.meta["still_since"] = None
        if dry:   # ensaio: so informa o que aconteceria, nao grava nada
            return {"aborted": self.meta["aborted"], "palm": self._palm_pose()[0].tolist()}
        self.meta["calls"] += 1
        reached_pos, reached_mat = self._palm_pose()
        executed = {"status": "aborted_on_contact" if self.meta["aborted"] else "executed",
                    "frames": frames if not self.meta["aborted"] else i + 1,
                    "requested_position": [round(v, 5) for v in np.asarray(pos).tolist()],
                    "reached_position": [round(v, 5) for v in reached_pos.tolist()],
                    "position_gap_m": round(float(np.linalg.norm(reached_pos - pos)), 4),
                    "rotation_gap_rad": round(quat_angle(quat_of(reached_mat), quat_of(target_mat)), 4),
                    "hand_table_contact": sorted(table_hits), "self_collision": sorted(self_hits),
                    "ik_at_request": ik_info}
        record = {"accepted": True, "action": action, "executed": executed}
        (self.dir / "actions" / f"call-{self.meta['calls']:03d}.json").write_text(json.dumps(record, indent=2) + "\n")
        (self.dir / "actions" / f"samples-{self.meta['calls']:03d}.json").write_text(json.dumps(samples) + "\n")
        return self._observe(reason_of_previous=reason, executed=executed)
