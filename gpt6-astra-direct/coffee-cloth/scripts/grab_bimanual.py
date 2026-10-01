"""Bimanual physical grasp diagnostic. Not a complete coffee sequence.
Uses initial object pose from simulation as a declared planning prior.
Historical infeasibility claims used incorrectly scaled meshes and are superseded.
"""
import argparse, json, sys, time, shutil, hashlib
from collections import Counter
from pathlib import Path
import numpy as np
import cv2
import mujoco

G1 = Path(__file__).resolve().parents[2] / "g1-cup-grasp" / "scripts"
sys.path.insert(0, str(G1))
from g1_sim import G1Sim, ARM_JOINTS, HAND_JOINTS
from kinematics import ArmIK
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bimanual_ik import BimanualIK
from recorder import Recorder

MODEL = "gpt-6-astra; controller inherited from Claude Fable"
rod = lambda v: cv2.Rodrigues(np.asarray(v, dtype=np.float64))[0]

ap = argparse.ArgumentParser()
ap.add_argument("--cold-body-reference", action="store_true", help="Explicitly run the historical cold-vessel body-grasp diagnostic; never a hot-coffee handling policy")
ap.add_argument("--cena", required=True)
ap.add_argument("--saida", required=True)
ap.add_argument("--objeto", default="chaleira")
ap.add_argument("--altura-pega", type=float, default=None, help="z absoluto; None = centro do objeto mais 9 cm")
ap.add_argument("--eixo-deg", type=float, default=90.0, help="direcao da linha que une as duas palmas")
ap.add_argument("--aproximacao", type=float, default=0.19, help="distancia de cada palma ao eixo do objeto na aproximacao")
ap.add_argument("--aperto", type=float, default=0.155, help="distancia de cada palma ao eixo do objeto no aperto")
ap.add_argument("--subida", type=float, default=0.08)
ap.add_argument("--cintura-deg", type=float, default=0.0,
                help="a cintura e COMPARTILHADA: fica fixa neste valor e sai do encadeamento das duas IK. Com ela dentro, as duas pediam valores opostos (+118,9 e -111,1 graus) e o segundo comando vencia, jogando o braco direito para tras do robo")
ap.add_argument("--vies-dir", type=float, default=0.0, help="recuo do alvo da mao direita, em metros, para as duas chegarem juntas")
ap.add_argument("--igualar-iters", type=int, default=6, help="quantas vezes reforcar o alvo de aproximacao esperando as duas maos igualarem")
ap.add_argument("--ref-roll", type=float, default=0.9, help="abertura do ombro na referencia do nullspace")
ap.add_argument("--descent-forward", type=float, default=0.0)
ap.add_argument("--grasp-x-shift", type=float, default=0.0, help="Shift contact targets along world x after approach; metres")
ap.add_argument("--thumb-tuck", type=float, default=0.0, help="Mirrored distal thumb flexion in radians")
ap.add_argument("--thumb-proximal", type=float, default=0.0, help="Mirrored proximal thumb flexion in radians")
ap.add_argument("--carry-offset", type=float, nargs=3, default=None, help="Optional world translation of held vessel, metres; diagnostic before pouring")
ap.add_argument("--carry-yaw-deg", type=float, default=0.0)
ap.add_argument("--carry-pitch-deg", type=float, default=0.0, help="Empty-vessel orientation diagnostic; does not simulate fluid")
ap.add_argument("--payload-compensation", type=float, default=0.0, help="Declared payload mass in kg for arm motor gravity feedforward, no object force applied")
ap.add_argument("--freeze-model", action="store_true", help="Also preserve compiled model assets (~170MB) for a milestone run")
ap.add_argument("--lift-seconds", type=float, default=2.0)
ap.add_argument("--grip-travel", type=float, default=.015, help="Bound on force-feedback inward correction, metres")
ap.add_argument("--grip-force", type=float, default=None, help="Per-hand normal force target in N; simulated tactile feedback")
ap.add_argument("--hand-kp", type=float, default=1.5)
ap.add_argument("--kp-escala", type=float, default=2.0)
ap.add_argument("--segura-s", type=float, default=2.0)
ap.add_argument("--hover-plan", type=Path)
ap.add_argument("--finger-yaw-deg", type=float, default=None)
ap.add_argument("--palm-pitch", type=float, default=0.0)
ap.add_argument("--palm-roll", type=float, default=0.0)
ap.add_argument("--cartesian", action="store_true", help="Raise, move above object, then descend along Cartesian paths")
ap.add_argument("--reason", default="")
a = ap.parse_args()
if not a.cold_body_reference:
    ap.error("Body grasp is restricted to cold mechanical reference. Coffee workflow requires handle-only grasp (assets/kettle-handling-requirements.json). Use --cold-body-reference only to reproduce this historical diagnostic.")

saida = Path(a.saida)
if saida.exists():
    raise FileExistsError(saida)
saida.mkdir(parents=True)
(saida / "parameters.json").write_text(json.dumps(vars(a), indent=2, default=str))
for filename in ("grab_bimanual.py", "bimanual_ik.py"):
    shutil.copy2(Path(__file__).with_name(filename), saida / filename)
shutil.copy2(a.cena, saida / "scene.xml")
(saida / "provenance.json").write_text(json.dumps({"model": MODEL, "scene_sha256": hashlib.sha256(Path(a.cena).read_bytes()).hexdigest(), "scope": "initial object pose from simulation; no learned or RGB policy"}, indent=2))

sim = G1Sim(a.cena)
m, d = sim.m, sim.d
if a.freeze_model:
    mujoco.mj_saveModel(m, str(saida / "model.mjb"), None)
dependencies = saida / "dependencies"
dependencies.mkdir()
for filename in ("g1_sim.py", "kinematics.py", "recorder.py"):
    shutil.copy2(G1 / filename, dependencies / filename)
fps = 30

RA = list(ARM_JOINTS)
LA = [j.replace("right_", "left_") for j in RA]
RH = list(HAND_JOINTS)
LH = [j.replace("right_", "left_") for j in RH]

# pose inicial com os dois bracos para tras e o ombro aberto: a de repouso do modelo poe o indicador
# direito dentro do saco do coador (46 contatos no quadro zero) e o ombro roçando o tronco
POSE_D = [0.8, -0.90, 0.0, 1.2, 0.0, 0.0, 0.0]
POSE_E = [0.8, 0.90, 0.0, 1.2, 0.0, 0.0, 0.0]
for nomes, pose in ((RA, POSE_D), (LA, POSE_E)):
    for n, v in zip(nomes, pose):
        d.qpos[m.jnt_qposadr[m.joint(n).id]] = v
d.qpos[m.jnt_qposadr[m.joint("waist_yaw_joint").id]] = np.radians(a.cintura_deg)
for side, sign in (("right", -1), ("left", 1)):
    joint = m.joint(side + "_hand_thumb_2_joint").id
    value = sign*a.thumb_tuck
    if not m.jnt_range[joint,0] <= value <= m.jnt_range[joint,1]:
        raise ValueError("Thumb flexion outside model joint bounds")
    d.qpos[m.jnt_qposadr[joint]] = value
    joint = m.joint(side + "_hand_thumb_1_joint").id
    value = sign*a.thumb_proximal
    if not m.jnt_range[joint,0] <= value <= m.jnt_range[joint,1]:
        raise ValueError("Proximal thumb flexion outside model joint bounds")
    d.qpos[m.jnt_qposadr[joint]] = value
sim.q_des[:] = d.qpos[sim.qadr]
for names in (RH, LH):
    sim.set_targets(names, sim.q(names), kp=a.hand_kp, kd=0.1*(a.hand_kp/1.5)**.5)
mujoco.mj_forward(m, d)

if a.kp_escala != 1.0:
    for n in RA + LA:
        i_ = sim.act_joint[n]
        sim.kp[i_] *= a.kp_escala
        sim.kd[i_] *= a.kp_escala ** 0.5

# A cintura NAO entra no encadeamento de nenhum dos dois bracos. Com ela, as duas IK pedem valores
# opostos para a mesma junta (+118,9 e -111,1 graus na jarra) e o segundo set_targets vence: a
# cintura ia para -111 e levava o braco direito para TRAS do robo, a 820 mm do objeto, com a IK
# reportando erro de 0,00 mm porque a palma dela estava no alvo que ela mesma resolveu.
# uma IK so para os dois bracos e a cintura: ver bimanual_ik.py para os numeros que mostram
# por que duas ArmIK independentes nao servem
bik = BimanualIK(sim, ["waist_yaw_joint"], RA, LA)
JBI = bik.nomes

obj = m.body(a.objeto).id
c0 = d.xpos[obj].copy()
cR0 = d.xmat[obj].reshape(3,3).copy()
alt = a.altura_pega if a.altura_pega is not None else float(c0[2] + 0.09)
obj_geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] == obj and m.geom_contype[g] != 0]
mao_geoms = {}
for lado in ("right", "left"):
    mao_geoms[lado] = [g for g in range(m.ngeom)
                       if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) or "").startswith(
                           (f"{lado}_hand", f"{lado}_wrist")) and m.geom_contype[g] != 0]

rec = Recorder(m, str(saida / "bimanual.mp4"),
               title=f"{saida.name}  |  {a.objeto}  |  cena {Path(a.cena).stem}  |  {MODEL.split('(')[0].strip()}  |  {time.strftime('%Y-%m-%d %H:%M')}")

estado = {"frames": 0, "fase": "settle"}
payload_gain = 0.0
payload_jac = np.zeros((3, m.nv))
amostras = []
ik_history = []
forbidden_substeps = Counter()
TRONCO = ("torso", "pelvis", "waist", "head")


def medidas():
    """Distancia de cada mao ao objeto, contatos ruins e quantos dedos tocam."""
    dist = {}
    dedos = 0
    for lado, gs in mao_geoms.items():
        dist[lado] = min(mujoco.mj_geomDistance(m, d, g, o, 0.3, None) for g in gs for o in obj_geoms)
    auto = mesa = 0
    for i in range(d.ncon):
        c = d.contact[i]
        b1 = m.body(int(m.geom_bodyid[c.geom1])).name
        b2 = m.body(int(m.geom_bodyid[c.geom2])).name
        braco = [b for b in (b1, b2) if b.startswith(("right_", "left_"))]
        if not braco:
            continue
        outro = b2 if b1 in braco else b1
        if outro.startswith(TRONCO):
            auto += 1
        if outro == "mesa":
            mesa += 1
        if outro == a.objeto and any(("hand" in b or "wrist" in b) for b in braco):
            dedos += 1
    return dist, auto, mesa, dedos


def avanca(n):
    global payload_gain
    for _ in range(n):
        ate = (estado["frames"] + 1) / fps
        while d.time < ate - m.opt.timestep / 2:
            tau = sim.kp * (sim.q_des - d.qpos[sim.qadr]) - sim.kd * d.qvel[sim.vadr] + d.qfrc_bias[sim.vadr]
            payload_on = estado['fase'] in ('levanta','segura','transporta','segura_transporte','retorna_transporte','pousa')
            payload_gain += np.clip(float(payload_on)-payload_gain, -m.opt.timestep/.5, m.opt.timestep/.5)
            if a.payload_compensation and payload_gain:
                for side in ('right','left'):
                    wrist = m.body(side+'_wrist_yaw_link').id
                    # Robot FK only: approximate force application at midpoint
                    # between the index and middle finger contact regions.
                    point = (d.xpos[m.body(side+'_hand_middle_1_link').id]
                             +d.xpos[m.body(side+'_hand_index_1_link').id])/2
                    point = point + d.xmat[wrist].reshape(3,3) @ np.array([.025,0,0])
                    mujoco.mj_jac(m,d,payload_jac,None,point,wrist)
                    force = -m.opt.gravity*a.payload_compensation*.5*payload_gain
                    tau += (payload_jac.T @ force)[sim.vadr]
            d.ctrl[:] = np.clip(tau, m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1])
            mujoco.mj_step(m, d)
            pairs = set()
            for contact in d.contact:
                b1, b2 = [m.body(int(m.geom_bodyid[g])).name for g in (contact.geom1, contact.geom2)]
                for robot, other in ((b1, b2), (b2, b1)):
                    if robot.startswith(("right_", "left_")) and (
                        other.startswith(TRONCO) or other in ("mesa", "coador", "copo", "pote", "tampa", "scoop", "base_eletrica")
                        or (robot.startswith("right_") and other.startswith("left_"))):
                        if other != a.objeto: pairs.add(tuple(sorted((robot, other))))
                    if robot != other and any(robot.startswith(s+'_hand') and other.startswith(s+'_hand') for s in ('right','left')):
                        pairs.add(tuple(sorted((robot,other))))
            forbidden_substeps.update((estado["fase"], *pair) for pair in pairs)
        mujoco.mj_forward(m, d)
        estado["frames"] += 1
        dist, auto, mesa, dedos = medidas()
        touching = set()
        object_supports = set()
        hand_forces = {"right": np.zeros(3), "left": np.zeros(3)}
        force_links = {"right": set(), "left": set()}
        contact_details = []
        total_contact_force = np.zeros(3)
        for ci, contact in enumerate(d.contact):
            bodies = [m.body(int(m.geom_bodyid[g])).name for g in (contact.geom1, contact.geom2)]
            if a.objeto in bodies:
                local_force = np.zeros(6)
                mujoco.mj_contactForce(m, d, ci, local_force)
                sign = 1 if bodies[1] == a.objeto else -1
                frame = contact.frame.reshape(3, 3)
                force_on_object = sign * frame.T @ local_force[:3]
                total_contact_force += force_on_object
                contact_details.append({
                    "other_body": bodies[0] if sign == 1 else bodies[1],
                    "position_world_m": contact.pos.tolist(),
                    "position_object_m": (d.xmat[obj].reshape(3,3).T @ (contact.pos-d.xpos[obj])).tolist(),
                    "normal_into_object_world": (sign*frame[0]).tolist(),
                    "force_contact_frame_N": local_force[:3].tolist(),
                    "torque_contact_frame_Nm": local_force[3:].tolist(),
                    "force_on_object_world_N": force_on_object.tolist(),
                    "friction": contact.friction.tolist(), "dim": int(contact.dim),
                    "distance_m": float(contact.dist),
                })
                for body in bodies:
                    if body != a.objeto and not body.startswith(("right_hand", "left_hand", "right_wrist", "left_wrist")):
                        object_supports.add(body)
                for body in bodies:
                    for side in ("right", "left"):
                        if body.startswith((side + "_hand", side + "_wrist")):
                            touching.add(side); force_links[side].add(body)
                            f = np.zeros(6); mujoco.mj_contactForce(m, d, ci, f)
                            world_force = contact.frame.reshape(3,3).T @ f[:3]
                            hand_forces[side] += world_force if bodies[1] == a.objeto else -world_force
        up = d.xmat[obj].reshape(3, 3)[2, 2]
        amostras.append({"t": round(d.time, 3), "fase": estado["fase"],
                         "obj": d.xpos[obj].round(5).tolist(), "obj_quaternion": d.xquat[obj].tolist(),
                         "object_support_contacts": sorted(object_supports),
                         "object_contact_details": contact_details,
                         "object_total_contact_force_N": total_contact_force.tolist(),
                         "touching_hands": sorted(touching),
                         "hand_force_on_object_N": {side: f.tolist() for side, f in hand_forces.items()},
                         "contact_links": {side: sorted(v) for side, v in force_links.items()},
                         "hand_joint_positions": {"right": sim.q(RH).tolist(), "left": sim.q(LH).tolist()},
                         "grip_correction_m": grip_correction.tolist() if "grip_correction" in globals() else [0,0],
                         "payload_feedforward_gain": payload_gain,
                         "wrist_torque_Nm": {j: float(d.ctrl[sim.act_joint[j]]) for j in JBI if "wrist" in j}, "joint_positions": sim.q(JBI).tolist(), "joint_targets": [float(sim.q_des[sim.act_joint[j]]) for j in JBI],
                         "obj_tilt_deg": round(float(np.degrees(np.arccos(np.clip(up, -1, 1)))), 3),
                         "subiu_mm": round(float(d.xpos[obj][2] - c0[2]) * 1000, 2),
                         "palma_dir": d.xpos[m.body("right_wrist_yaw_link").id].round(4).tolist(),
                         "palma_esq": d.xpos[m.body("left_wrist_yaw_link").id].round(4).tolist(),
                         "mao_dir": d.xpos[m.body("right_hand_middle_1_link").id].round(4).tolist(),
                         "mao_esq": d.xpos[m.body("left_hand_middle_1_link").id].round(4).tolist(),
                         "dist_dir_mm": round(dist["right"] * 1000, 2),
                         "dist_esq_mm": round(dist["left"] * 1000, 2),
                         "autocolisao": auto, "mao_mesa": mesa, "contatos_mao_objeto": dedos})
        rec.frame(d)


def fase(nome, texto):
    estado["fase"] = nome
    rec.event(f"{nome}: {texto}")


def move_juntas(q_alvo, segundos):
    q0 = sim.q(JBI)
    n = max(1, round(segundos * fps))
    for i in range(n):
        u = (i + 1) / n
        u = u * u * (3 - 2 * u)
        sim.set_targets(JBI, q0 + u * (np.asarray(q_alvo) - q0))
        avanca(1)


# a MAO fica 124,9 mm a frente da palma no eixo +x local dela, e 18,2 mm para o lado, com sinal
# oposto em cada braco. Mirar a palma no raio de pega deixava a direita penetrando 1,3 mm e a
# esquerda a 106 mm, e a jarra tombava 91 graus ja na aproximacao.
def hand_offset(side):
    palm = m.body(side + "_wrist_yaw_link").id
    hand = m.body(side + "_hand_middle_1_link").id
    return d.xmat[palm].reshape(3, 3).T @ (d.xpos[hand] - d.xpos[palm])

OFF_DIR = hand_offset("right")
OFF_ESQ = hand_offset("left")
(saida / "hand-offsets.json").write_text(json.dumps({"model": MODEL, "right": OFF_DIR.tolist(), "left": OFF_ESQ.tolist(), "method": "FK of current hand joint pose before movement; replaces unrelated hardcoded pose"}, indent=2))


def palmas(raio, z):
    """raio e a distancia da MAO ao eixo do objeto; a palma recua o offset dela."""
    # eixo_deg aponta da jarra para a mao ESQUERDA. A direita fica no lado oposto, sempre.
    # Com o sinal trocado (o codigo antigo punha a direita em +eixo_deg) os bracos se cruzavam: com
    # eixo_deg=90 o alvo da mao direita caia em y=+0,23, do outro lado da jarra, e a mao varria de
    # y=-0,140 para y=+0,147 em 4 quadros, 2,15 m/s atravessando o objeto. A jarra saia voando em +y
    # junto com a mao. Nao era pega falha, era tapa. Descruzado, a IK passa de 20,5 mm de erro para
    # 0,02 mm no raio de aperto.
    ang = np.radians(a.eixo_deg)
    finger_angle = ang if a.finger_yaw_deg is None else np.radians(a.finger_yaw_deg)
    RR = rod([0, 0, finger_angle]) @ rod([0, a.palm_pitch, 0]) @ rod([a.palm_roll, 0, 0])
    RL = rod([0, 0, -finger_angle]) @ rod([0, a.palm_pitch, 0]) @ rod([-a.palm_roll, 0, 0])
    # vies: mesmo com a IK bimanual o braco direito chega mais perto que o esquerdo (erro de 15,8
    # contra 3,7 mm), entao a direita toca primeiro e derruba. O alvo dela recua o vies medido.
    rR = raio + a.vies_dir
    maoR = np.array([c0[0] - rR * np.cos(ang), c0[1] - rR * np.sin(ang), z])
    maoL = np.array([c0[0] + raio * np.cos(ang), c0[1] + raio * np.sin(ang), z])
    if not np.isclose(raio, a.aproximacao):
        maoR[0] += a.grasp_x_shift
        maoL[0] += a.grasp_x_shift
    return maoR - RR @ OFF_DIR, maoL - RL @ OFF_ESQ, RR, RL


def ref_abertos():
    """Referencia do nullspace com os ombros abertos: sem isso a IK escolhe poses com os bracos
    colados ao tronco, 200 quadros de autocolisao."""
    q = sim.q(JBI).copy()
    q[JBI.index("right_shoulder_roll_joint")] = -a.ref_roll
    q[JBI.index("left_shoulder_roll_joint")] = a.ref_roll
    return q


def resolve(raio, z, seed=None):
    pR, pL, RR, RL = palmas(raio, z)
    q, i = bik.solve(pR, RR, pL, RL, seed if seed is not None else sim.q(JBI), ref_abertos(), iteracoes=500)
    ik_history.append({"phase": estado["fase"], "kind": "endpoint_probe", **i})
    return q, i["erro_dir_mm"], i["erro_esq_mm"]


grip_correction = np.zeros(2)

def move_cartesian(pR, RR, pL, RL, seconds, force_feedback=False, descent=False, rigid_pivots=None):
    q = sim.q(JBI)
    # Under contact, actual joints differ from commanded joints by the PD
    # preload. Restarting at actual FK removes that load at each phase boundary.
    start_q = np.array([sim.q_des[sim.act_joint[j]] for j in JBI]) if force_feedback else q
    sR, rR, sL, rL = bik.fk(start_q)
    rvR = cv2.Rodrigues(RR @ rR.T)[0].ravel()
    rvL = cv2.Rodrigues(RL @ rL.T)[0].ravel()
    ref = ref_abertos()
    n = max(1, round(seconds * fps))
    for i in range(n):
        u = (i + 1) / n; u = u*u*(3-2*u)
        tr = sR + u*(pR-sR); tl = sL + u*(pL-sL)
        if rigid_pivots is not None:
            pivot_start, pivot_end = rigid_pivots
            pivot = pivot_start+u*(pivot_end-pivot_start)
            rotation = rod(u*rvR); rotation_end = rod(rvR)
            tr = pivot+rotation@((1-u)*(sR-pivot_start)+u*(rotation_end.T@(pR-pivot_end)))
            tl = pivot+rotation@((1-u)*(sL-pivot_start)+u*(rotation_end.T@(pL-pivot_end)))
        if descent:
            tr[0] += a.descent_forward*np.sin(np.pi*u)
            tl[0] += a.descent_forward*np.sin(np.pi*u)
        if force_feedback and a.grip_force is not None:
            axis = np.array([np.cos(np.radians(a.eixo_deg)), np.sin(np.radians(a.eixo_deg)), 0.0])
            _, _, reference_rotation, _ = palmas(a.aperto, alt)
            axis = (rod(u*rvR)@rR) @ reference_rotation.T @ axis
            forces = amostras[-1]["hand_force_on_object_N"]
            measured = np.array([np.dot(forces["right"], axis), -np.dot(forces["left"], axis)])
            grip_correction[:] = np.clip(grip_correction + .0005*(measured-a.grip_force)/fps, -a.grip_travel, .005)
            # sR/sL already contain the previous phase's compression.
            # Blend toward the corrected endpoint; adding full correction at
            # u=0 double-counted compression and caused a phase-boundary jump.
            tr = tr - u*axis*grip_correction[0]; tl = tl + u*axis*grip_correction[1]
        q, error = bik.solve(tr, rod(u*rvR)@rR,
                            tl, rod(u*rvL)@rL,
                            q, ref, iteracoes=100, passo_max=np.radians(90)/fps)
        ik_history.append({"phase": estado["fase"], "kind": "cartesian_frame", **error})
        sim.set_targets(JBI, q); avanca(1)
    return q, error


fase("settle", "robo em repouso, bracos abertos")
avanca(20)

fase("aproxima", f"as duas palmas a {a.aproximacao * 100:.0f} cm do eixo do {a.objeto}, na altura de pega")
q1, er, el = resolve(a.aproximacao, alt)
print(f"  IK aproxima: direita {er:.2f} mm, esquerda {el:.2f} mm")
if a.cartesian:
    if a.hover_plan:
        plan = json.loads(a.hover_plan.read_text())
        assert plan["joint_names"] == JBI
        assert np.max(np.abs(sim.q(JBI)-plan["q_initial"])) < .02
        shutil.copy2(a.hover_plan, saida / "hover-plan.json")
        fase("clearance_waypoint", "collision-checked shoulder detour toward reachable hover posture")
        move_juntas(plan["q_waypoint"], 3.0)
        fase("hover", "enter reachable IK branch using verified hover configuration")
        move_juntas(plan["q_hover"], 3.0)
    else:
        fase("raise_clearance", "raise hands before traversing the table")
        pr, rr, pl, rl = bik.fk(sim.q(JBI))
        pr[2] = max(pr[2], 1.12); pl[2] = max(pl[2], 1.12)
        move_cartesian(pr, rr, pl, rl, 2.0)
        fase("hover", "both hands travel above object before descending outside it")
        pr, pl, rr, rl = palmas(a.aproximacao, alt + .20)
        move_cartesian(pr, rr, pl, rl, 3.0)
    fase("aproxima", "lower both hands outside the object")
    pr, pl, rr, rl = palmas(a.aproximacao, alt)
    q1, info = move_cartesian(pr, rr, pl, rl, 2.5, descent=True)
    er, el = info["erro_dir_mm"], info["erro_esq_mm"]
else:
    move_juntas(q1, 2.5)

fase("iguala", "espera a mao atrasada alcancar: a direita chegava a 43 mm e a esquerda a 187, e a jarra tombava antes do aperto")
for _ in range(a.igualar_iters):
    dist, _, _, _ = medidas()
    if abs(dist["right"] - dist["left"]) < 0.010:
        break
    move_juntas(q1, 0.4)

fase("aperta", f"fecha ate {a.aperto * 100:.1f} cm: as palmas prendem o {a.objeto} por atrito")
q2, er2, el2 = resolve(a.aperto, alt, q1)
print(f"  IK aperta: direita {er2:.2f} mm, esquerda {el2:.2f} mm")
if a.cartesian:
    pr, pl, rr, rl = palmas(a.aperto, alt)
    q2, info = move_cartesian(pr, rr, pl, rl, 2.0, force_feedback=True)
    er2, el2 = info["erro_dir_mm"], info["erro_esq_mm"]
else:
    move_juntas(q2, 1.5)
avanca(10)

fase("levanta", f"sobe {a.subida * 100:.0f} cm com o objeto")
q3, er3, el3 = resolve(a.aperto, alt + a.subida, q2)
print(f"  IK levanta: direita {er3:.2f} mm, esquerda {el3:.2f} mm")
if a.cartesian:
    pr, pl, rr, rl = palmas(a.aperto, alt+a.subida)
    q3, info = move_cartesian(pr, rr, pl, rl, a.lift_seconds, force_feedback=True)
    er3, el3 = info["erro_dir_mm"], info["erro_esq_mm"]
else:
    move_juntas(q3, 1.5)

fase("segura", f"segura {a.segura_s:.0f} s no ar")
if a.grip_force is not None:
    pr, pl, rr, rl = palmas(a.aperto, alt+a.subida)
    move_cartesian(pr, rr, pl, rl, a.segura_s, force_feedback=True)
else:
    avanca(int(a.segura_s * fps))

np.savez(saida / "held-state.npz", qpos=d.qpos, qvel=d.qvel, ctrl=d.ctrl,
         q_des=sim.q_des, kp=sim.kp, kd=sim.kd, time=np.array(d.time),
         grip_correction=grip_correction, frames=np.array(estado['frames']),
         payload_gain=np.array(payload_gain), original_object_position=c0,
         original_object_rotation=cR0)

carry_start = d.xpos[obj].copy()
if a.carry_offset is not None:
    offset = np.array(a.carry_offset)
    pr, pl, rr, rl = palmas(a.aperto, alt+a.subida)
    carry_rotation = rod([0,0,np.deg2rad(a.carry_yaw_deg)])@rod([0,np.deg2rad(a.carry_pitch_deg),0])
    pivot_start = c0+np.array([0,0,a.subida]);pivot_end=pivot_start+offset
    cr = pivot_end+carry_rotation@(pr-pivot_start)
    cl = pivot_end+carry_rotation@(pl-pivot_start)
    crr=carry_rotation@rr;crl=carry_rotation@rl
    fase("transporta", "desloca jarra sustentada em direcao ao coador; ainda sem despejo")
    move_cartesian(cr, crr, cl, crl, 3.0, force_feedback=True, rigid_pivots=(pivot_start,pivot_end))
    fase("segura_transporte", "verifica sustentacao no novo ponto")
    move_cartesian(cr, crr, cl, crl, 2.0, force_feedback=True)
    fase("retorna_transporte", "retorna acima da base mantendo pega")
    move_cartesian(pr, rr, pl, rl, 3.0, force_feedback=True, rigid_pivots=(pivot_end,pivot_start))

fase("pousa", "devolve o objeto a mesa e solta")
if a.cartesian:
    pr, pl, rr, rl = palmas(a.aperto, alt)
    move_cartesian(pr, rr, pl, rl, 2.5, force_feedback=True)
else:
    move_juntas(q2, 1.2)
avanca(10)
fase("solta", "abre as maos depois de devolver a jarra a base")
if a.cartesian:
    pr, pl, rr, rl = palmas(a.aproximacao, alt)
    move_cartesian(pr, rr, pl, rl, 1.5)
else:
    move_juntas(q1, 1.5)
avanca(15)

rec.close(str(saida / "timeline.json"))

seg = [s for s in amostras if s["fase"] == "segura"]
auto = sum(s["autocolisao"] for s in amostras)
mesa = sum(s["mao_mesa"] for s in amostras)
subida_min = min(s["subiu_mm"] for s in seg) if seg else 0.0
tilt_max = max(s["obj_tilt_deg"] for s in seg) if seg else 999.0
P = np.array([s["obj"] for s in amostras])
dt = 1.0 / fps
pico = float(np.linalg.norm(np.gradient(np.gradient(P, dt, axis=0), dt, axis=0), axis=1).max())
rel = {
    "handling_scope": "cold_body_reference_only",
    "coffee_workflow_accepted": False,
    "modelo_controlador": MODEL, "cena": a.cena, "objeto": a.objeto, "reason": a.reason,
    "erro_ik_aproxima_mm": [round(er, 2), round(el, 2)],
    "erro_ik_aperta_mm": [round(er2, 2), round(el2, 2)],
    "erro_ik_levanta_mm": [round(er3, 2), round(el3, 2)],
    "subida_minima_enquanto_segura_mm": round(subida_min, 2),
    "inclinacao_maxima_enquanto_segura_deg": round(tilt_max, 2),
    "contatos_mao_objeto_medianos": int(np.median([s["contatos_mao_objeto"] for s in seg])) if seg else 0,
    "autocolisao_quadros": auto, "mao_na_mesa_quadros": mesa,
    "pico_aceleracao_objeto_m_s2": round(pico, 2),
    "duracao_s": round(d.time, 2),
    # levantar de verdade: 80% do comandado, em pe, com as duas maos tocando e nada esbarrando
    "contatos_proibidos_subpassos": [{"phase": k[0], "bodies": k[1:], "substeps": v} for k, v in forbidden_substeps.items()],
    "unsupported_throughout_hold": bool(seg) and all(not s["object_support_contacts"] for s in seg),
    "both_hands_throughout_hold": bool(seg) and all(len(s["touching_hands"]) == 2 for s in seg),
    "warnings": sim.warnings(),
    "aceito": bool(subida_min > a.subida * 1000 * 0.8 and tilt_max < 15.0 and auto == 0 and mesa == 0
                   and not forbidden_substeps and sim.warnings() == 0 and seg
                   and all(len(s["touching_hands"]) == 2 and not s["object_support_contacts"] for s in seg)),
}
(saida / "ik.json").write_text(json.dumps(ik_history, indent=2))
rel["max_ik_orientation_deg"] = max(max(x["orient_dir_deg"], x["orient_esq_deg"]) for x in ik_history)
rel["max_ik_position_mm"] = max(max(x["erro_dir_mm"], x["erro_esq_mm"]) for x in ik_history)
rel["aceito"] = bool(rel["aceito"] and rel["max_ik_orientation_deg"] < 2 and rel["max_ik_position_mm"] < 3)
if a.carry_offset is not None:
    carried = [s for s in amostras if s['fase'] in ('transporta','segura_transporte','retorna_transporte')]
    held = [s for s in carried if s['fase']=='segura_transporte']
    error = max(float(np.linalg.norm(np.array(s['obj'])-carry_start-np.array(a.carry_offset))) for s in held)
    orientation_errors=[]
    for s in held:
        actual=np.zeros(9);mujoco.mju_quat2Mat(actual,np.array(s['obj_quaternion']))
        orientation_errors.append(float(np.degrees(np.linalg.norm(cv2.Rodrigues(carry_rotation@cR0@actual.reshape(3,3).T)[0]))))
    carry_pass = all(len(s['touching_hands'])==2 and not s['object_support_contacts']
                     and s['obj_tilt_deg']<abs(a.carry_pitch_deg)+15 and s['subiu_mm']>a.subida*1000*.8 for s in carried) and error<.015 and max(orientation_errors)<10
    rel['carry'] = {'offset_m':a.carry_offset,'hold_position_error_max_m':error,
                    'min_height_change_mm':min(s['subiu_mm'] for s in carried),
                    'max_tilt_deg':max(s['obj_tilt_deg'] for s in carried),'passed':bool(carry_pass)}
    rel['carry']['hold_orientation_error_max_deg']=max(orientation_errors)
    rel['carry']['commanded_yaw_pitch_deg']=[a.carry_yaw_deg,a.carry_pitch_deg]
    rel['aceito'] = bool(rel['aceito'] and carry_pass)
(saida / "relatorio.json").write_text(json.dumps(rel, indent=2) + "\n")
(saida / "trajetoria.json").write_text(json.dumps(amostras) + "\n")
print(json.dumps(rel, indent=2))
