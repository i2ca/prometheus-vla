"""Tentativa guiada por camera: localiza o copo na head_camera (RGB), leva a Dex3 ate ele por IK,
fecha, levanta e segura. Cada tentativa vive em um diretorio novo e nunca sobrescreve outro.

O que e privilegiado e o que nao e:
- posicao inicial do copo: usada UMA vez para colocar o objeto na mesa (parametro cup_xy);
- controle: usa apenas a estimativa RGB do copo, a calibracao pelos marcadores e a cinematica do braco;
- physics-report.json: le o estado do simulador (contatos, altura real do copo, avisos) so para VERIFICAR.

Uso: .venv/bin/python scripts/run_attempt.py --parameters results/attempt-01.parameters.json --run-dir results/attempt-01
"""
import argparse
import datetime
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cv2
import mujoco
import numpy as np
from g1_sim import G1Sim, ARM_JOINTS, HAND_JOINTS, ROOT, load_seed
from kinematics import ArmIK, PALM
from recorder import Recorder
from vision import calibrate, locate_cup
from dataset_recorder import DatasetRecorder

MODEL = "claude-fable-5-1 (Claude Code); correcoes de visao e de limites de cintura por gpt-6-astra (tentativas 63 a 66)"
TILT_LIMIT_DEG = 15.0
# palm_roll_rad: giro da palma em torno do eixo dos dedos; encolhe a abertura vertical entre indicador e medio   # copo derrubado na aproximacao ou segurado deitado nao conta como pega
DEFAULTS = {
    "cup_xy": [0.33, -0.13], "cup_yaw_deg": 0.0,   # yaw do copo na mesa: 0 = alca para +x (longe do robo) "home_shoulder_roll_rad": -0.35, "start_shoulder_roll_rad": None, "offset": [-0.02, -0.02, -0.02], "gain": 1.5,
    "kp": 8, "palm_pitch_rad": 0.4, "palm_roll_rad": 0.0, "center_height": 0.03, "center_xy_bias": [0.0, 0.0],
    "preshape_fraction": 0.6, "gravity_compensation": True, "home_lift": 0.10, "approach_height": 0.09,
    "close_frames": 35, "ramp_frames": 35, "lift_height": 0.16, "hold_frames": 60,
    "approach_mode": "top", "side_distance": 0.10, "side_lift": 0.0,
    "index_close_scale": 1.0,   # multiplica o curso de fecho do indicador (juntas index_0 e index_1)
    # pega humana ("medium wrap", Feix 2016): copo encostado na palma, definido direto no referencial da palma
    # (x = direcao dos dedos, y = normal da palma / lado do polegar, z = do medio para o indicador). None = usa a semente.
    "cup_in_palm": None,
    "open_during_approach": False,   # abertura bifasica: mao aberta na chegada, fecha na desaceleracao
    "retreat_distance": 0.0,         # depois de levantar, traz o copo em direcao ao corpo (-x) por N m
    "lift_seconds": 2.0, "retreat_seconds": 1.5,
    "approach_dir_palm": [1.0, 0.0, 0.0],   # direcao de chegada no referencial da palma (x = dedos, y = polegar)
    "approach_seconds": 1.5,
    "lift_direction": [0.0, 0.0, 1.0],       # direcao da subida no mundo (normalizada), para tirar o cotovelo do tronco
    "left_arm_pose": None,
    "photo_arm_pose": None, "raise_hand": True,   # raise_hand False: a pose de foto ja e alta, a IK vai direto dela para tras do copo   # pose do braco direito durante a foto (7 juntas); None = pose da semente com home_shoulder_roll_rad
    "waist_joints": ["waist_yaw_joint"],   # juntas da cintura que a IK pode usar (ex.: + "waist_pitch_joint" para alcancar a frente)
    "arm_nullspace_roll": None,   # referencia do ombro (roll, rad) no espaco nulo: negativo afasta o cotovelo do tronco
    "turn_to_cup": False,      # antes de estender o braco, gira a cintura para o copo ficar a frente do ombro direito (fase 'turn')
    "turn_max_deg": 60.0, "shoulder_offset_y": 0.15,
    "use_waist": False,        # False: so braco; True: waist_yaw sempre na IK; "auto": so braco se alcanca a pose de pega, senao cintura
    "waist_weight": 0.5,       # peso do passo da cintura na IK (menor = cintura gira menos que o braco)
    "waist_pitch_range_deg": [0.0, 12.0],   # tronco so inclina para a frente, e pouco (um humano nao se inclina para tras para trazer a mao)
    "waist_pitch_weight": 0.2,
    "waist_nullspace_gain": 0.005,   # espaco nulo mais suave com a cintura (0,03 dava salto de 7 mm na troca de modo)
    "lock_waist_after_close": True,  # depois do fecho, a cintura para: levantar e trazer e trabalho do braco   # [pitch, roll, yaw, cotovelo, punho roll, pitch, yaw] do braco esquerdo; None = zero do XML (esticado para a frente)
    "max_joint_speed_deg_s": None,   # limite de velocidade articular imposto na IK (None = sem limite)
    "approach_open": None,           # pose da mao na chegada (7 juntas); None = pose aberta da semente
    "head_keep_cup_deg": None,       # se definido, o yaw da cintura fica em [rumo do copo - k, rumo + k] (camera da cabeca gira com o tronco; k < 34,5 = metade dos 69 graus horizontais da D435i)
    "ik_iterations": 35,             # iteracoes da IK por quadro no move()
    "markers": None,                 # markers.json alternativo (caminho); None = scene/markers.json
    "scene": "scene_grasp.xml",      # cena (nome em scene/ ou caminho absoluto); o copo tem que se chamar "copo"
    "place_xy": None,                # se definido, depois de levantar leva o copo ate (x, y) e o coloca (tarefa: copo sob o coador)
    "place_surface_z": 0.754,        # z do apoio onde o copo e colocado (mesa 0,75 + base do suporte 0,004)
    "place_clearance": 0.012,        # folga do fundo do copo ao apoio ao entrar por baixo do filtro
    "place_approach": 0.12,          # entra deslizando por essa distancia
    "place_approach_dir": [-1.0, 0.0, 0.0],   # direcao (mundo) em que o copo desliza para entrar sob o filtro; o copo parte de place - approach*dir
    "turn_seconds": 1.0,             # duracao do giro do tronco para o copo
    "pre_reach_seconds": 1.5,
    "pre_reach_offset": [-0.08, 0.0, 0.15],   # ponto de passagem do pre_reach relativo ao ponto 'atras do copo' (mundo): sobe e recua; um desvio lateral -y afasta o polegar do tronco        # duracao de cada metade do pre_reach (sobe acima da mesa; vai para tras do copo)
    "pre_turn_raise": 0.0,           # antes de girar o tronco, sobe a mao em repouso por N m (a mao a 0,87-0,98 m varre o aro do coador a 1,0 m)
    "turn_to_place": False,          # antes de transportar, gira o tronco para o alvo (como na pega): camera ve o alvo e o braco fica na geometria da pega
    "carry_tilt_comp": None,         # {"axis": [ax,ay,az] no referencial da palma, "angle_deg": a}: gira a palma nas fases de colocacao para o copo, que fica inclinado na mao, voltar a ficar em pe (medido na auditoria)
    "carry_offset": None,            # fundo do copo no referencial da palma DURANTE o transporte (medido na auditoria); None = cup_in_palm - [0,0,center_height]
    "place_kp_scale": 1.0,           # rigidez do braco (kp e kd) multiplicada nas fases de colocacao (transporte ate o recuo), como quem enrijece o braco para pousar algo
    "place_kd_exponent": 0.5,        # expoente de place_kp_scale no kd. 0,5 mantinha o kd atras do kp e derrubava o amortecimento relativo em raiz de 2; 1,0 preserva o amortecimento
    "release_retreat": 0.0,          # durante a abertura da mao, a palma recua essa distancia (m) do copo para a palma, como quem solta puxando a mao
    "set_down_gap": -0.002,          # altura do fundo do copo acima do apoio ao abrir a mao: negativo = pressiona o copo contra o apoio; positivo = solta o copo a essa altura e ele assenta sozinho
    "set_down_vertical": False,      # apoio so vertical (nao corrige xy com o copo ja encostado)
    "place_set_down_seconds": 1.0,   # duracao do apoio (com correcao continua)
    "track_gain": 0.15,              # ganho da correcao continua no move_tracked (por quadro)
    "place_lower_steps": 3,          # etapas da descida, com correcao proprioceptiva entre elas
    "pre_lower_correction": False,   # correct_palm no ar antes de descer (introduzido nas 88+; a 78 nao tinha e regride com ele)
    "lower_settle": False,           # assenta settle_frames no fim da descida antes de corrigir (idem)
    "settle_frames": 20,
    "palm_correction_iters": 0, "palm_correction_gain": 0.5, "palm_correction_frames": 20,      # correcao proprioceptiva do alvo da palma no fim de lower/slide/set_down: mede a palma pela cinematica direta das juntas medidas e desloca o alvo pelo erro (0 = desligada)             # quadros de espera depois do deslize e do apoio, para o PD alcancar o alvo antes de soltar
    "place_transport_seconds": 2.5, "place_slide_seconds": 1.5, "place_lower_seconds": 0.8, "release_frames": 35, "hand_retreat": 0.10,
    "record_dataset": False,         # grava <run>/dataset/ no esquema v2 do LCAD (estado/acao 29, cabeca RGB+depth, punho 224x224)   # side_lift: chega por tras e avanca N m acima da pose, desce so no fim   # "top": desce por cima; "side": chega por tras, ao longo do eixo dos dedos
}


def render(m, d, cam="head_camera", size=(848, 480)):
    r = mujoco.Renderer(m, size[1], size[0])
    r.update_scene(d, camera=cam)
    img = r.render().copy()
    r.close()
    return img


def save_rgb(path, rgb):
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def run(params, directory):
    p = {**DEFAULTS, **params}
    board = json.loads(Path(p["markers"]).read_text() if p["markers"] else (Path(ROOT) / "scene" / "markers.json").read_text())
    vertices = np.load(Path(ROOT) / "scene" / "cup_vertices.npy")
    sim = G1Sim(p["scene"])
    m, d = sim.m, sim.d
    arm, hand, fps = load_seed(0)
    gi = int(np.flatnonzero(hand[:, 4] > 0.5)[0])
    home = arm[0].copy(); home[1] = p["home_shoulder_roll_rad"]
    if p["photo_arm_pose"] is not None:
        home = np.asarray(p["photo_arm_pose"], float)
    LEFT_ARM = [j.replace("right_", "left_") for j in ARM_JOINTS]
    poses = [(ARM_JOINTS, home), (HAND_JOINTS, hand[0])]
    if p["left_arm_pose"] is not None:
        poses.append((LEFT_ARM, np.asarray(p["left_arm_pose"], float)))
    for names, values in poses:
        for name, value in zip(names, values):
            d.qpos[m.jnt_qposadr[m.joint(name).id]] = value
    sim.q_des[:] = d.qpos[sim.qadr]
    sim.set_targets(HAND_JOINTS, hand[0], kp=p["kp"], kd=1)
    yaw = np.radians(p["cup_yaw_deg"])
    sim.place_cup([*p["cup_xy"], 0.752], quat=(np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)))   # unica intervencao no objeto
    reference = ArmIK(sim).reference_grasp(arm, hand, gi, p["offset"], p["gain"])
    ik_state = {"mode": "arm"}

    arm_ref = arm[gi].copy()
    if p["arm_nullspace_roll"] is not None:
        arm_ref[1] = p["arm_nullspace_roll"]

    def build_ik(use_waist):
        wj = list(p["waist_joints"]) if use_waist else []
        joints = wj + list(ARM_JOINTS)
        solver = ArmIK(sim, joints)
        if use_waist:
            solver.weights = np.r_[[p["waist_pitch_weight"] if j == "waist_pitch_joint" else p["waist_weight"] for j in wj], np.ones(7)]
            solver.nullspace_gain = p["waist_nullspace_gain"]
            solver.bounds = solver.bounds.copy()
            if "waist_pitch_joint" in wj:
                solver.bounds[wj.index("waist_pitch_joint")] = np.radians(p["waist_pitch_range_deg"])
            if "waist_yaw_joint" in wj and ik_state.get("yaw_bounds") is not None:
                k = wj.index("waist_yaw_joint"); solver.bounds[k] = np.clip(ik_state["yaw_bounds"], solver.bounds[k, 0], solver.bounds[k, 1])
        ref = np.r_[[float(d.qpos[m.jnt_qposadr[m.joint(j).id]]) for j in wj], arm_ref]   # espaco nulo: cintura onde esta (apos o giro), braco da semente
        return joints, solver, ref
    IK_JOINTS, ik, ik_ref = build_ik(p["use_waist"] is True)
    if p["index_close_scale"] != 1.0:
        idx = [HAND_JOINTS.index("right_hand_index_0_joint"), HAND_JOINTS.index("right_hand_index_1_joint")]
        close = reference["close"].copy()
        close[idx] = reference["open"][idx] + p["index_close_scale"] * (close[idx] - reference["open"][idx])
        reference["close"] = close
    import time as _time
    rec = Recorder(m, str(directory / "grasp-run.mp4"), title=f"{directory.name}  |  cena {Path(p['scene']).stem}  |  {MODEL.split(';')[0].split('(')[0].strip()}  |  {_time.strftime('%Y-%m-%d %H:%M')}")
    dsrec = DatasetRecorder(sim, str(directory / "dataset")) if p["record_dataset"] else None
    samples, ik_errors = [], []
    state = {"frames": 0, "phase": "settle", "table_substeps": 0, "worst_penetration": 0.0}

    def contacts():
        table, min_dist = [], 0.0
        tampo = m.geom("tampo").id
        for contact in d.contact:
            if tampo in (contact.geom1, contact.geom2):
                other = contact.geom2 if contact.geom1 == tampo else contact.geom1
                body = m.body(int(m.geom_bodyid[other])).name
                if body.startswith("right_hand") or body == PALM:
                    table.append(body); min_dist = min(min_dist, float(contact.dist))
        return sorted(set(table)), min_dist

    CUP_R_IN, CUP_H, HANDLE_X = 0.039, 0.092, (0.041, 0.066)   # alca no +x LOCAL do copo   # geometria de colisao do copo (scene_grasp.xml)
    tip_bodies = {"index": "right_hand_index_1_link", "middle": "right_hand_middle_1_link", "thumb": "right_hand_thumb_2_link"}
    tip_local = {"index": [0.046, 0, 0], "middle": [0.046, 0, 0], "thumb": [0, 0.046, 0]}

    def finger_tips_in_cup_frame():
        """Pontas dos dedos no referencial do copo: (raio, altura, x_local). Diz se a ponta esta dentro do copo ou no vao da alca."""
        cq = d.xquat[sim.cup_body]; R = np.zeros(9); mujoco.mju_quat2Mat(R, cq); R = R.reshape(3, 3); c0 = d.xpos[sim.cup_body]
        out = {}
        for k, bn in tip_bodies.items():
            bid = m.body(bn).id
            tip = d.xpos[bid] + d.xmat[bid].reshape(3, 3) @ np.array(tip_local[k])
            loc = R.T @ (tip - c0); r = float(np.hypot(loc[0], loc[1]))
            inside = bool(r < CUP_R_IN and 0.004 < loc[2] < CUP_H)
            in_handle = bool(HANDLE_X[0] < loc[0] < HANDLE_X[1] and abs(loc[1]) < 0.02 and 0.023 < loc[2] < 0.072)
            out[k] = {"r": round(r, 4), "z": round(float(loc[2]), 4), "inside_cup": inside, "in_handle_gap": in_handle}
        return out

    TRUNK = {"torso_link", "pelvis", "waist_yaw_link", "waist_roll_link", "head_link"}

    def self_collisions():
        pairs = set()
        for i in range(d.ncon):
            c = d.contact[i]; b1 = m.body(int(m.geom_bodyid[c.geom1])).name; b2 = m.body(int(m.geom_bodyid[c.geom2])).name
            r1, r2 = b1.startswith("right_"), b2.startswith("right_")
            if (r1 and b2 in TRUNK) or (r2 and b1 in TRUNK) or (r1 and b2.startswith("left_")) or (r2 and b1.startswith("left_")):
                pairs.add(tuple(sorted((b1, b2))))
        return sorted(pairs)

    def env_contacts():
        """Pares corpo x corpo em contato envolvendo o copo ou a mao direita, fora mesa/copo/dedos (o que mais toca em que)."""
        pairs = set()
        for i in range(d.ncon):
            c = d.contact[i]; b1 = m.body(int(m.geom_bodyid[c.geom1])).name; b2 = m.body(int(m.geom_bodyid[c.geom2])).name
            hands = {b1.startswith("right_hand") or b1 == PALM, b2.startswith("right_hand") or b2 == PALM}
            if {b1, b2} <= {"copo", "mesa"} or (any(hands) and {b1, b2} & {"copo", "mesa"} and not (b1 == "mesa" or b2 == "mesa")): continue
            if "copo" in (b1, b2) or any(hands): pairs.add((b1, b2, m.geom(c.geom1).name, m.geom(c.geom2).name, round(float(c.pos[0]), 3), round(float(c.pos[1]), 3), round(float(c.pos[2]), 3)))
        return sorted(pairs)

    def advance(nframes):
        for _ in range(nframes):
            until = (state["frames"] + 1) / fps
            while d.time < until - m.opt.timestep / 2:
                tau = sim.kp * (sim.q_des - d.qpos[sim.qadr]) - sim.kd * d.qvel[sim.vadr]
                if p["gravity_compensation"]:
                    tau += d.qfrc_bias[sim.vadr]
                d.ctrl[:] = np.clip(tau, m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1])
                mujoco.mj_step(m, d)
                sub, depth = contacts()
                state["table_substeps"] += bool(sub); state["worst_penetration"] = min(state["worst_penetration"], depth)
            mujoco.mj_forward(m, d)
            table, dist = contacts()
            links = sorted(sim.finger_contacts())
            cup_up = d.xmat[sim.cup_body].reshape(3, 3)[:, 2]
            samples.append({"t": float(d.time), "phase": state["phase"], "cup": sim.cup_pos().tolist(),
                            "cup_tilt_deg": float(np.degrees(np.arccos(np.clip(cup_up[2], -1, 1)))), "cup_up": cup_up.tolist(),
                            "palm": sim.body_pos(PALM).tolist(), "palm_mat": d.xmat[m.body(PALM).id].reshape(3, 3).tolist(), "finger_links": links,
                            "fingers": sorted({n.split("_")[0] for n in links}), "hand_table": table,
                            "table_penetration_m": dist, "cup_table": sim.cup_table_contact(),
                            "tips": finger_tips_in_cup_frame(), "self_collision": self_collisions(), "env_contacts": env_contacts(),
                            "waist_yaw": float(d.qpos[m.jnt_qposadr[m.joint("waist_yaw_joint").id]]),
                            "waist_pitch": float(d.qpos[m.jnt_qposadr[m.joint("waist_pitch_joint").id]])})
            rec.frame(d); state["frames"] += 1
            if dsrec: dsrec.frame()

    def stage(name, caption):
        state["phase"] = name; rec.event(caption, phase=name)

    max_step = None if p["max_joint_speed_deg_s"] is None else np.radians(p["max_joint_speed_deg_s"]) / fps

    def correct_palm(position, rotation, cup_offset_palm=None, plane_only=False, bias=None):
        """Malha externa proprioceptiva: o PD fica 1 a 2 cm (e ate 7 graus de yaw) atras do alvo com o braco estendido e carga.
        Mede a palma pela FK das juntas medidas e desloca o alvo por metade do erro (ganho 0,5), com 20 quadros entre
        iteracoes e bias limitado a 2 cm. Se cup_offset_palm (fundo do copo no referencial da palma) for dado, o erro e o do
        COPO previsto (palma real + R_real @ offset), o que cancela tambem o erro de orientacao. plane_only zera z (copo ja
        apoiado: nao ha como corrigir). Devolve o bias acumulado (para continuar a segurar a palma noutra fase)."""
        position = np.asarray(position, float); bias = np.zeros(3) if bias is None else np.asarray(bias, float).copy(); q = sim.q(IK_JOINTS)
        cup_goal = None if cup_offset_palm is None else position + rotation @ cup_offset_palm
        for k in range(p["palm_correction_iters"]):
            actual, R_actual = ik.fk(sim.q(IK_JOINTS))
            err = (cup_goal - (actual + R_actual @ cup_offset_palm)) if cup_goal is not None else (position - actual)
            if plane_only: err[2] = max(err[2], 0.0)   # com o copo perto do apoio: corrige z so para CIMA (o PD afunda 1 a 2 cm e o copo arrastava na base ainda preso na mao); nunca empurra para baixo
            ik_errors.append({"phase": state["phase"] + "_corr", "position_error_m": float(np.linalg.norm(err)), "orientation_error_rad": 0.0, "iterations": k})
            if np.linalg.norm(err) < 0.002: break
            bias_prev = bias.copy(); bias = np.clip(bias + p["palm_correction_gain"] * err, -0.02, 0.02)
            nf = p["palm_correction_frames"]
            for f_ in range(nf):   # alvo vai suavemente do bias anterior ao novo: sem salto de ate 2 cm num quadro (5 a 8 m/s2 nas 68 a 72)
                u = (f_ + 1) / nf; u = u * u * (3 - 2 * u)
                q, _ = ik.solve(position + bias_prev + u * (bias - bias_prev), rotation, q, ik_ref, iterations=p["ik_iterations"], max_step=max_step); sim.set_targets(IK_JOINTS, q); advance(1)
        state["pbias"] = bias.copy()   # a compensacao fica valendo para o proximo move (senao o alvo seguinte 'perde' o bias e o copo cai 1 cm de volta)
        return bias

    def move_tracked(position, rotation, seconds, cup_offset_palm, gain=0.15):
        """move com correcao continua: a cada quadro mede o copo previsto (FK das juntas medidas + offset) e desloca o alvo por
        uma fracao do erro. Sem saltos: o braco estendido afunda 1 a 2 cm com carga e a correcao em bloco empurra o objeto
        quando ele ja esta encostado."""
        start, start_r = ik.fk(sim.q(IK_JOINTS)); q = sim.q(IK_JOINTS); rv = cv2.Rodrigues(rotation @ start_r.T)[0].ravel()
        n = max(1, round(seconds * fps)); bias = np.zeros(3); off = np.asarray(cup_offset_palm, float)
        for i in range(n):
            u = (i + 1) / n; u = u * u * (3 - 2 * u); target_r = cv2.Rodrigues(rv * u)[0] @ start_r
            nominal = start + u * (np.asarray(position) - start)
            actual, R_actual = ik.fk(sim.q(IK_JOINTS))
            err = (nominal + target_r @ off) - (actual + R_actual @ off)
            bias = np.clip(bias + gain * err, -0.03, 0.03)
            q, error = ik.solve(nominal + bias, target_r, q, ik_ref, iterations=p["ik_iterations"], max_step=max_step)
            ik_errors.append({"phase": state["phase"], **error}); sim.set_targets(IK_JOINTS, q); advance(1)
        state["pbias"] = bias.copy()

    def move(position, rotation, seconds):
        if state["phase"] in ("slide", "set_down") and state.get("pbias") is not None:
            position = np.asarray(position, float) + state["pbias"]
        start, start_r = ik.fk(sim.q(IK_JOINTS)); q = sim.q(IK_JOINTS)
        rv = cv2.Rodrigues(rotation @ start_r.T)[0].ravel()
        count = max(1, round(seconds * fps))
        for i in range(count):
            u = (i + 1) / count; u = u * u * (3 - 2 * u)
            target_r = cv2.Rodrigues(rv * u)[0] @ start_r
            q, error = ik.solve(start + u * (np.asarray(position) - start), target_r, q, ik_ref, iterations=p["ik_iterations"], max_step=max_step)
            ik_errors.append({"phase": state["phase"], **error})
            sim.set_targets(IK_JOINTS, q); advance(1)

    report = {"controller_model": MODEL, "mode": "camera-guided; cup pose from RGB only", "parameters": p,
              "object_repositions_after_start": 0}
    try:
        stage("settle", "robo em repouso; copo colocado uma vez na mesa")
        advance(20)
        true_initial = sim.cup_pos().copy()
        # ---- percepcao: so RGB ----
        rgb0 = render(m, d); save_rgb(directory / "camera-initial.png", rgb0)
        calibration = calibrate(rgb0, board)
        (directory / "camera-localization.json").write_text(json.dumps(calibration, indent=2) + "\n")
        detection = locate_cup(rgb0, calibration, board, vertices)
        estimate = np.array(detection["position"], float)
        detection["true_position_privileged"] = true_initial.tolist()
        detection["estimate_error_xy_m"] = float(np.linalg.norm(estimate[:2] - true_initial[:2]))
        detection["true_yaw_deg_privileged"] = float(p["cup_yaw_deg"])
        if detection.get("yaw_deg_estimate") is not None:
            detection["yaw_error_deg"] = float(abs((detection["yaw_deg_estimate"] - p["cup_yaw_deg"] + 180) % 360 - 180))
        (directory / "cup-detection.json").write_text(json.dumps(detection, indent=2) + "\n")
        rec.event(f"camera: copo estimado em x={estimate[0]:.3f} y={estimate[1]:.3f} (RGB, marcadores)", phase="perception")
        advance(15)
        if p["photo_arm_pose"] is not None and p["raise_hand"]:
            # foto tirada com o braco em repouso ao lado do corpo; ergue a mao para a pose de prontidao (juntas, 1 s)
            stage("raise_hand", "ergue a mao: ombro primeiro (cotovelo dobrado), depois estende")
            ready = arm[0].copy(); ready[1] = p["home_shoulder_roll_rad"]
            q0 = sim.q(ARM_JOINTS)
            mid = q0.copy(); mid[0], mid[1], mid[2] = ready[0], ready[1], ready[2]   # ombro na pose de prontidao, cotovelo e punho como estao
            for q_from, q_to in ((q0, mid), (mid, ready)):
                for i in range(36):
                    u = (i + 1) / 36; u = u * u * (3 - 2 * u); sim.set_targets(ARM_JOINTS, (1 - u) * q_from + u * q_to); advance(1)
            advance(5)
        if p["start_shoulder_roll_rad"] is not None and abs(p["start_shoulder_roll_rad"] - home[1]) > 1e-6:
            # foto tirada com o braco afastado; agora vai, em juntas, para a pose de partida do controlador
            stage("to_start", f"vai para a pose de partida (roll do ombro {p['start_shoulder_roll_rad']:+.2f} rad)")
            q0 = sim.q(ARM_JOINTS); q1 = q0.copy(); q1[1] = p["start_shoulder_roll_rad"]
            for i in range(30):
                u = (i + 1) / 30; u = u * u * (3 - 2 * u); sim.set_targets(ARM_JOINTS, (1 - u) * q0 + u * q1); advance(1)
            advance(10)
        # ---- controle a partir da estimativa ----
        rotation = cv2.Rodrigues(np.array([0.0, p["palm_pitch_rad"], 0.0]))[0] @ reference["palm_rotation"] @ cv2.Rodrigues(np.array([p["palm_roll_rad"], 0.0, 0.0]))[0]
        if p["head_keep_cup_deg"] is not None:
            bearing = float(np.arctan2(estimate[1], estimate[0] - 0.12))         # rumo do copo visto da camera da cabeca (pos x=0,12 no tronco, modelo g1_29dof_with_hand do prometheus-vla), tronco a zero
            ik_state["yaw_bounds"] = np.array([bearing - np.radians(p["head_keep_cup_deg"]), bearing + np.radians(p["head_keep_cup_deg"])])
            report["waist_yaw_bounds_deg"] = np.degrees(ik_state["yaw_bounds"]).tolist()
            if IK_JOINTS != list(ARM_JOINTS):
                IK_JOINTS, ik, ik_ref = build_ik(True)
        if p["turn_to_cup"]:
            theta = float(np.clip(np.arctan2(estimate[1] + p["shoulder_offset_y"], estimate[0]), -np.radians(p["turn_max_deg"]), np.radians(p["turn_max_deg"])))
            if ik_state.get("yaw_bounds") is not None:
                theta = float(np.clip(theta, *ik_state["yaw_bounds"]))
            report["turn_deg"] = float(np.degrees(theta))
            rotation = cv2.Rodrigues(np.array([0.0, 0.0, theta]))[0] @ rotation   # a pega gira junto com o tronco
            if p["pre_turn_raise"]:
                stage("raise", f"sobe a mao {p['pre_turn_raise']*100:.0f} cm antes de girar")
                pos0_, rot0_ = ik.fk(sim.q(IK_JOINTS)); move(pos0_ + [0, 0, p["pre_turn_raise"]], rot0_, 1.0)
            if abs(theta) > np.radians(3):
                stage("turn", f"gira o tronco {np.degrees(theta):+.0f} graus para o copo")
                q0 = float(d.qpos[m.jnt_qposadr[m.joint("waist_yaw_joint").id]])
                nt = max(1, round(p["turn_seconds"] * fps))
                for i in range(nt):
                    u = (i + 1) / nt; u = u * u * (3 - 2 * u); sim.set_targets(["waist_yaw_joint"], [(1 - u) * q0 + u * theta]); advance(1)
                advance(5)
        if p["use_waist"] == "auto":
            # alcance: a pose de pega e alcancavel so com o braco? (IK offline, sem mover o robo)
            probe_target = estimate + np.array([*p["center_xy_bias"], p["center_height"]]) - rotation @ (np.asarray(p["cup_in_palm"], float) if p["cup_in_palm"] is not None else reference["palm_to_center_translation"])
            _, err = ik.solve(probe_target, rotation, sim.q(ARM_JOINTS), arm_ref, iterations=300)
            report["reach_probe_arm_only_error_m"] = err["position_error_m"]
            if err["position_error_m"] > 0.005:
                IK_JOINTS, ik, ik_ref = build_ik(True); ik_state["mode"] = "waist+arm"
                rec.event(f"fora do alcance do braco (erro {err['position_error_m']*100:.1f} cm): destrava a cintura", phase="reach")
            else:
                ik_state["mode"] = "arm"
        report["ik_mode"] = ik_state["mode"]
        if p["cup_in_palm"] is not None:
            target = estimate + np.array([*p["center_xy_bias"], p["center_height"]]) - rotation @ np.asarray(p["cup_in_palm"], float)
        else:
            target = estimate + np.array([*p["center_xy_bias"], p["center_height"]]) - rotation @ reference["palm_to_center_translation"]
        if p["home_lift"]:
            stage("clear_table", "afasta a mao da mesa")
            cur, cur_r = ik.fk(sim.q(ARM_JOINTS)); move(cur + [0, 0, p["home_lift"]], cur_r, 1.0)
        fraction = p["preshape_fraction"]
        if p["approach_mode"] == "side":
            if p["approach_open"] is not None:
                sim.set_targets(HAND_JOINTS, np.asarray(p["approach_open"], float))
            stage("approach_behind", f"vai para tras do copo estimado ({p['side_distance']*100:.0f} cm ao longo do eixo da palma)")
            lift = np.array([0, 0, p["side_lift"]])
            adir = np.asarray(p["approach_dir_palm"], float); adir /= np.linalg.norm(adir)
            behind = target - p["side_distance"] * (rotation @ adir) + lift
            if p["photo_arm_pose"] is not None and not p["raise_hand"]:
                # mao vem do repouso ao lado do corpo: sobe primeiro (ponto acima e atras do ponto de chegada), so depois estende
                stage("pre_reach", "ergue a mao acima da mesa antes de estender")
                move(behind + np.asarray(p["pre_reach_offset"], float), rotation, p["pre_reach_seconds"])
            move(behind, rotation, p["pre_reach_seconds"])   # chegada ao longo de adir no referencial da palma
            if fraction and not p["open_during_approach"]:
                stage("preshape", f"pre-fecha a mao ({int(fraction*100)}%) atras do copo")
                sim.set_targets(HAND_JOINTS, reference["open"] + fraction * (reference["close"] - reference["open"])); advance(30)
            stage("approach_cup", "avanca ao longo dos dedos ate a pose de pega" + (f" ({p['side_lift']*100:.0f} mm acima)" if p["side_lift"] else ""))
            if p["open_during_approach"]:
                # abertura bifasica: aberta na primeira metade, comeca a fechar ate `preshape_fraction` na desaceleracao
                start_a, start_ra = ik.fk(sim.q(IK_JOINTS)); qa = sim.q(IK_JOINTS); count = round(p["approach_seconds"] * fps)
                rva = cv2.Rodrigues(rotation @ start_ra.T)[0].ravel()
                for i in range(count):
                    u = (i + 1) / count; su = u * u * (3 - 2 * u)
                    qa, error = ik.solve(start_a + su * (target + lift - start_a), cv2.Rodrigues(rva * su)[0] @ start_ra, qa, ik_ref, iterations=35, max_step=max_step)
                    ik_errors.append({"phase": state["phase"], **error}); sim.set_targets(IK_JOINTS, qa)
                    fr = 0.0 if u < 0.5 else fraction * (u - 0.5) / 0.5
                    wide = np.asarray(p["approach_open"], float) if p["approach_open"] is not None else reference["open"]
                    hand_t = wide + fr * (reference["close"] - wide) if fr < fraction else reference["open"] + fr * (reference["close"] - reference["open"])
                    sim.set_targets(HAND_JOINTS, hand_t); advance(1)
            else:
                move(target + lift, rotation, p["approach_seconds"])
            if p["side_lift"]:
                stage("approach_cup", f"desce {p['side_lift']*100:.0f} mm ate a pose de pega")
                move(target, rotation, 0.5)
            advance(10)
        else:
            stage("approach_above", "aproxima por cima do copo estimado (IK)")
            move(target + [0, 0, p["approach_height"]], rotation, 1.5)
            if fraction:
                stage("preshape", f"pre-fecha a mao ({int(fraction*100)}%) acima da mesa")
                sim.set_targets(HAND_JOINTS, reference["open"] + fraction * (reference["close"] - reference["open"])); advance(30)
            stage("approach_cup", "desce ate a pose de pega")
            move(target, rotation, 1.5); advance(10)
        stage("close", "fecha a Dex3 (copo livre)")
        for i in range(p["close_frames"]):
            u = min(1, (i + 1) / max(1, p["ramp_frames"])); u = fraction + (1 - fraction) * u
            sim.set_targets(HAND_JOINTS, reference["open"] + u * (reference["close"] - reference["open"])); advance(1)
        advance(15)
        if p["lock_waist_after_close"] and IK_JOINTS != list(ARM_JOINTS):
            IK_JOINTS, ik, ik_ref = build_ik(False)   # cintura fica onde esta (alvo PD mantido); so o braco levanta e traz
        stage("lift", f"levanta {p['lift_height']*100:.0f} cm")
        ldir = np.asarray(p["lift_direction"], float); ldir /= np.linalg.norm(ldir)
        top = target + p["lift_height"] * ldir
        move(top, rotation, p["lift_seconds"])
        if p["retreat_distance"]:
            stage("retreat", f"traz o copo {p['retreat_distance']*100:.0f} cm em direcao ao corpo")
            if p["lock_waist_after_close"]:
                # ao trazer o copo o tronco volta a ficar reto (pitch -> 0), o braco compensa pela IK
                pj = m.jnt_qposadr[m.joint("waist_pitch_joint").id]; p0_ = float(d.qpos[pj]); n_ = max(1, round(p["retreat_seconds"] * fps))
                straighten = iter([(1 - (k + 1) / n_) * p0_ for k in range(n_)])
                _orig_advance = advance
                def advance_straight(nframes):
                    for _ in range(nframes):
                        try: sim.set_targets(["waist_pitch_joint"], [next(straighten)])
                        except StopIteration: pass
                        _orig_advance(1)
                advance = advance_straight
            move(top + [-p["retreat_distance"], 0, 0], rotation, p["retreat_seconds"])
            if p["lock_waist_after_close"]:
                advance = _orig_advance
        placed_report = None
        if p["place_xy"] is not None:
            # o controlador continua sem ler a pose do copo: o copo esta na mao no offset cup_in_palm da pega (mesma conta da chegada)
            cip = np.asarray(p["cup_in_palm"], float) if p["cup_in_palm"] is not None else reference["palm_to_center_translation"]
            carry = np.asarray(p["carry_offset"], float) if p["carry_offset"] is not None else cip - [0, 0, p["center_height"]]
            cup_off = rotation @ carry + [0, 0, p["center_height"]]             # fundo do copo (referencial da palma) + altura do centro NO MUNDO (o copo esta em pe; somar no referencial da palma dava 1,6 cm de erro em x)
            offs = {"v": cup_off}
            def palm_for_cup(cx, cy, cz): return np.array([cx, cy, cz]) - offs["v"]
            if p["turn_to_place"]:
                # gira o tronco para o alvo da colocacao segurando o copo: alvo da palma fixo no mundo enquanto a cintura muda (a IK do braco compensa);
                # a orientacao da palma gira junto com o tronco (o copo gira na mao)
                theta2 = float(np.arctan2(p["place_xy"][1] + (p["shoulder_offset_y"] or 0.0), p["place_xy"][0]))
                if p["head_keep_cup_deg"] is not None:
                    # The head now tracks the placement target, not the initial cup.
                    bearing2 = float(np.arctan2(p["place_xy"][1], p["place_xy"][0] - 0.12))
                    ik_state["yaw_bounds"] = bearing2 + np.radians([-p["head_keep_cup_deg"], p["head_keep_cup_deg"]])
                    report["place_waist_yaw_bounds_deg"] = np.degrees(ik_state["yaw_bounds"]).tolist()
                    theta2 = float(np.clip(theta2, *ik_state["yaw_bounds"]))
                q0w = float(d.qpos[m.jnt_qposadr[m.joint("waist_yaw_joint").id]]); dtheta = theta2 - q0w
                stage("turn_to_place", f"gira o tronco para {np.degrees(theta2):+.0f} graus, para o suporte, com o copo na mao")
                pos_hold, rot_hold = ik.fk(sim.q(IK_JOINTS)); nt = max(1, round(p["turn_seconds"] * fps)); qa = sim.q(IK_JOINTS)
                for i in range(nt):
                    u = (i + 1) / nt; u = u * u * (3 - 2 * u); sim.set_targets(["waist_yaw_joint"], [q0w + u * dtheta])
                    rot_u = cv2.Rodrigues(np.array([0.0, 0.0, u * dtheta]))[0] @ rot_hold
                    qa, err = ik.solve(pos_hold, rot_u, qa, ik_ref, iterations=p["ik_iterations"], max_step=max_step); ik_errors.append({"phase": "turn_to_place", **err})
                    sim.set_targets(IK_JOINTS, qa); advance(1)
                rotation = cv2.Rodrigues(np.array([0.0, 0.0, dtheta]))[0] @ rotation; report["turn_to_place_deg"] = float(np.degrees(theta2))
                cup_off = rotation @ carry + [0, 0, p["center_height"]]; offs["v"] = cup_off; advance(10)
            if p["carry_tilt_comp"] is not None:
                ax = np.asarray(p["carry_tilt_comp"]["axis"], float); ax /= np.linalg.norm(ax)
                rotation = rotation @ cv2.Rodrigues(ax * np.radians(p["carry_tilt_comp"]["angle_deg"]))[0]   # rotacao no referencial da palma
                cup_off = rotation @ carry + [0, 0, p["center_height"]]; offs["v"] = cup_off
                stage("level", f"endireita o copo na mao ({p['carry_tilt_comp']['angle_deg']:.0f} graus)"); pos_l, _ = ik.fk(sim.q(IK_JOINTS)); move(pos_l, rotation, 0.8)
            if isinstance(p["place_approach_dir"], str) and p["place_approach_dir"] == "palm":
                adir = rotation @ np.array([1.0, 0.0, 0.0]); adir[2] = 0.0      # o copo entra a frente da palma (eixo x da palma), nunca a mao primeiro
            else:
                adir = np.asarray(p["place_approach_dir"], float)
            adir /= np.linalg.norm(adir); wx, wy = np.array(p["place_xy"]) - p["place_approach"] * adir[:2]
            report["place_approach_dir_used"] = adir.tolist()
            px, py = p["place_xy"]; zc_low = p["place_surface_z"] + p["place_clearance"] + p["center_height"]   # centro do copo com o fundo a 'clearance' do apoio
            zc_now = float(estimate[2] + p["center_height"] + p["lift_height"] * ldir[2])
            if p["place_kp_scale"] != 1.0:
                for jn in ARM_JOINTS:
                    i_ = sim.act_joint[jn]; sim.kp[i_] *= p["place_kp_scale"]; sim.kd[i_] *= p["place_kp_scale"] ** p["place_kd_exponent"]
            stage("transport", f"leva o copo ate a entrada do suporte ({wx:.2f}, {wy:.2f})")
            move(palm_for_cup(wx, wy, zc_now), rotation, p["place_transport_seconds"])
            if p["pre_lower_correction"]:
                correct_palm(palm_for_cup(wx, wy, zc_now), rotation, carry)   # alinha no ar, antes de descer: o PD deixa o copo 1 a 2 cm atras e a descida so pioraria
            stage("lower", f"desce ate o fundo do copo ficar a {p['place_clearance']*1000:.0f} mm do apoio")
            # move_tracked (96 a 98) nunca foi validado e confunde atraso do PD com erro estacionario; padrao da 78/99: move + assenta + uma correcao
            move(palm_for_cup(wx, wy, zc_low), rotation, p["place_lower_seconds"])
            if p["lower_settle"]: advance(p["settle_frames"])
            correct_palm(palm_for_cup(wx, wy, zc_low), rotation, carry)
            stage("slide", f"entra por baixo do filtro ate ({px:.2f}, {py:.2f})")
            move(palm_for_cup(px, py, zc_low), rotation, p["place_slide_seconds"]); advance(p["settle_frames"]); correct_palm(palm_for_cup(px, py, zc_low), rotation, carry)
            stage("set_down", "apoia o copo")
            z_set = zc_low - p["place_clearance"] + p["set_down_gap"]
            if p["set_down_vertical"]:
                # depois que o copo encosta, corrigir xy vira arrasto: o apoio e so vertical, a partir de onde a mao esta
                pos_sd, _ = ik.fk(sim.q(IK_JOINTS)); dz_sd = (z_set - zc_low)
                move(np.array([pos_sd[0], pos_sd[1], pos_sd[2] + dz_sd]), rotation, p["place_set_down_seconds"]); advance(p["settle_frames"])
            else:
                # padrao da 78 (erro 5,6 mm): apoia, assenta e faz uma correcao so no plano. A versao com move_tracked
                # regrediu para 36,1 mm, e remove-lo sem devolver o correct_palm plane_only manteve 36,2 mm.
                move(palm_for_cup(px, py, z_set), rotation, p["place_set_down_seconds"]); advance(p["settle_frames"])
                correct_palm(palm_for_cup(px, py, z_set) + [0, 0, 0.003], rotation, carry, plane_only=True)
            stage("release", "abre a mao" + (f" recuando {p['release_retreat']*100:.0f} cm" if p["release_retreat"] else " segurando a palma no lugar"))
            q_closed = sim.q(HAND_JOINTS); hold_pos0, _ = ik.fk(sim.q(IK_JOINTS)); qa_hold = sim.q(IK_JOINTS)
            # o pbias do slide/set_down nao pode sumir de uma vez (a palma saltava de 0,8 para 217 mm/s, pico de 8,3 m/s2
            # na 108) nem ficar ate o fim (a mao arrasta o copo ao soltar: 24,7 mm de erro na 109). Decai junto com o recuo.
            pbias0 = np.zeros(3) if state.get("pbias") is None else np.asarray(state["pbias"], float).copy()
            hold_bias = pbias0.copy()
            away0 = -cup_off.copy(); away0[2] = 0; away0 /= np.linalg.norm(away0)
            for i in range(p["release_frames"]):
                u = (i + 1) / p["release_frames"]; sim.set_targets(HAND_JOINTS, q_closed + u * (reference["open"] - q_closed))
                us = u * u * (3 - 2 * u); hold_pos = hold_pos0 + p["release_retreat"] * us * away0   # os dedos abrem e a palma se afasta junto, suave, IK a cada quadro
                if p["release_retreat"]: hold_bias = pbias0 * (1.0 - us)
                if p["release_retreat"]:
                    qa_hold, _ = ik.solve(hold_pos + hold_bias, rotation, qa_hold, ik_ref, iterations=p["ik_iterations"], max_step=max_step); sim.set_targets(IK_JOINTS, qa_hold)
                advance(1)
                if p["palm_correction_iters"] and not p["release_retreat"] and (i + 1) % 10 == 0:
                    # ao abrir, o binario que a pinca fazia no punho some e o braco "desarma" (palma andou 6 mm em x na 40 sem comando): segura a palma
                    actual, _ = ik.fk(sim.q(IK_JOINTS)); err = hold_pos - actual; err[2] = 0.0
                    hold_bias = np.clip(hold_bias + p["palm_correction_gain"] * err, -0.02, 0.02)
                    qa_hold, _ = ik.solve(hold_pos + hold_bias, rotation, qa_hold, ik_ref, iterations=p["ik_iterations"] * 3, max_step=max_step); sim.set_targets(IK_JOINTS, qa_hold)
                    ik_errors.append({"phase": "release_hold", "position_error_m": float(np.linalg.norm(err)), "orientation_error_rad": 0.0, "iterations": i})
            advance(10)
            stage("hand_retreat", f"recua a mao {p['hand_retreat']*100:.0f} cm")
            away = -cup_off.copy(); away[2] = 0; away /= np.linalg.norm(away)      # do copo para a palma (a mao pode estar a frente do copo na direcao do deslize)
            pos_now, _ = ik.fk(sim.q(IK_JOINTS)); move(pos_now + p["hand_retreat"] * away + [0, 0, 0.02], rotation, 1.2)   # 2 cm para cima: o medio raspava a mesa
        stage("hold", "segura 2 s")
        advance(p["hold_frames"])
        if p["place_xy"] is not None and p["place_kp_scale"] != 1.0:
            for jn in ARM_JOINTS:
                i_ = sim.act_joint[jn]; sim.kp[i_] /= p["place_kp_scale"]; sim.kd[i_] /= p["place_kp_scale"] ** p["place_kd_exponent"]
        if p["place_xy"] is not None:
            cp = sim.cup_pos(); up = d.xmat[sim.cup_body].reshape(3, 3)[:, 2]
            placed_report = {"target_xy": list(p["place_xy"]), "final_xy": cp[:2].tolist(), "xy_error_m": float(np.hypot(cp[0] - p["place_xy"][0], cp[1] - p["place_xy"][1])),
                             "final_z": float(cp[2]), "tilt_deg": float(np.degrees(np.arccos(np.clip(up[2], -1, 1)))), "fingers_touching_at_end": sorted(sim.finger_contacts()),
                             "placed": bool(np.hypot(cp[0] - p["place_xy"][0], cp[1] - p["place_xy"][1]) < 0.015 and np.degrees(np.arccos(np.clip(up[2], -1, 1))) < 3.0 and not sim.finger_contacts())}   # colocado = em pe de verdade (< 3 graus), nao o limite de 15 da pega
        rgb1 = render(m, d); save_rgb(directory / "camera-final.png", rgb1)
        held = [s for s in samples if s["phase"] == "hold"]
        min_rise = 0.6 * p["lift_height"]   # antes era 8 cm fixo; agora proporcional a altura pedida
        retained = bool(held) and all(s["cup"][2] > true_initial[2] + min_rise and len(s["fingers"]) >= 2 and not s["cup_table"] for s in held)
        if p["place_xy"] is not None:   # na tarefa de colocar, "retido" e medido no transporte: copo na mao (>= 2 dedos) e fora da mesa em todos os quadros
            carried = [s for s in samples if s["phase"] in ("transport", "lower", "slide")]   # (a altura nao serve de criterio: o copo e deliberadamente descido a 1 cm do apoio antes de deslizar)
            retained = bool(carried) and all(len(s["fingers"]) >= 2 and not s["cup_table"] for s in carried)
        tilt_before = max(s["cup_tilt_deg"] for s in samples if s["phase"] in ("approach_cup", "approach_behind"))
        palm_cup_hold = float(np.mean([np.linalg.norm(np.array(s["cup"]) - np.array(s["palm"])) for s in held])) if held else None
        tilt_hold = float(np.mean([s["cup_tilt_deg"] for s in held])) if held else None
        upright = tilt_before < TILT_LIMIT_DEG and tilt_hold is not None and tilt_hold < TILT_LIMIT_DEG
        report.update({"status": "completed", "cup_estimate": estimate.tolist(), "initial_cup_privileged": true_initial.tolist(),
                       "final_cup": sim.cup_pos().tolist(), "lift_m": float(sim.cup_pos()[2] - true_initial[2]),
                       "retained_for_2s": retained, "warnings": sim.warnings(),
                       "max_distinct_fingers": max(len(s["fingers"]) for s in samples),
                       "fingers_at_hold": sorted(set(f for s in held for f in s["fingers"])),
                       "cup_tilt_deg_before_close": tilt_before, "cup_tilt_deg_at_hold": tilt_hold,
                       "upright_grasp": bool(upright), "tilt_limit_deg": TILT_LIMIT_DEG, "palm_to_cup_center_m_at_hold": palm_cup_hold,
                       "frames_finger_inside_cup": {k: sum(1 for s in samples if s["tips"][k]["inside_cup"]) for k in tip_bodies},
                       "frames_finger_in_handle_gap": {k: sum(1 for s in samples if s["tips"][k]["in_handle_gap"]) for k in tip_bodies},
                       "hold_tips_cup_frame": {k: {"r": float(np.mean([s["tips"][k]["r"] for s in held])), "z": float(np.mean([s["tips"][k]["z"] for s in held]))} for k in tip_bodies} if held else None,
                       "self_collision_frames": sum(1 for s in samples if s["self_collision"]),
                       "place": placed_report,
                       "accepted": bool(retained and upright and not any(s["self_collision"] for s in samples) and (placed_report is None or placed_report["placed"])),
                       "hand_table_frames": sum(bool(s["hand_table"]) for s in samples),
                       "hand_table_substeps": state["table_substeps"],
                       "max_substep_table_penetration_m": -state["worst_penetration"],
                       "max_ik_position_error_m": max(e["position_error_m"] for e in ik_errors),
                       "max_ik_orientation_error_rad": max(e["orientation_error_rad"] for e in ik_errors),
                       "waist_yaw_deg_at_hold": float(np.degrees(np.mean([s["waist_yaw"] for s in held]))) if held else None,
                       "waist_pitch_deg_at_hold": float(np.degrees(np.mean([s["waist_pitch"] for s in held]))) if held else None})
    except Exception as exc:  # falha registrada, nunca escondida
        report.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        raise
    finally:
        (directory / "physics-report.json").write_text(json.dumps(report, indent=2) + "\n")
        (directory / "trajectory.json").write_text(json.dumps(samples) + "\n")
        (directory / "ik.json").write_text(json.dumps(ik_errors) + "\n")
        secs = rec.close(str(directory / "timeline.json"))
        if dsrec: dsrec.close({"accepted": report.get("accepted"), "cup_xy": p["cup_xy"], "cup_yaw_deg": p["cup_yaw_deg"], "status": report.get("status")})
        report["video_seconds"] = secs
        (directory / "run-report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parameters", type=Path, required=True)
    ap.add_argument("--run-dir", type=Path, required=True)
    a = ap.parse_args()
    if a.run_dir.exists():
        sys.exit(f"Diretorio ja existe, tentativa preservada: {a.run_dir}")
    params = json.loads(a.parameters.read_text())
    a.run_dir.mkdir(parents=True)
    (a.run_dir / "parameters.json").write_text(json.dumps(params, indent=2) + "\n")
    (a.run_dir / "provenance.json").write_text(json.dumps({
        "model": MODEL, "started_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "source_sha256": {q.name: hashlib.sha256(q.read_bytes()).hexdigest() for q in Path(__file__).parent.glob("*.py")},
        "scene_sha256": {q.name: hashlib.sha256(q.read_bytes()).hexdigest() for q in (Path(ROOT) / "scene").glob("*.xml")},
    }, indent=2) + "\n")
    report = run(params, a.run_dir)
    print(json.dumps({"directory": str(a.run_dir), **{k: v for k, v in report.items() if k != "parameters"}}, indent=2))


if __name__ == "__main__":
    main()
