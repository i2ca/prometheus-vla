"""Etapa 2a do cafe: tirar a tampa do pote com uma pinca por cima e pousa-la ao lado. Mesma disciplina do run_attempt:
percepcao so por camera (pote marrom localizado por cor com a calibracao dos marcadores; o pegador da tampa fica no eixo do
pote a uma altura conhecida), controle por IK + PD, fisica so para verificar, tudo registrado (parameters.json com reason,
trajectory.json com contatos por peca, physics-report.json, video 3x2). Claude (claude-fable-5-1), 16/09/2026.
Uso: ../g1-cup-grasp/.venv/bin/python scripts/task_lid.py --parameters experiments/lid-01.parameters.json --run-dir results/lid-01"""
import argparse, json, os, sys, time, hashlib
from pathlib import Path
import numpy as np, cv2, mujoco
G1 = Path(__file__).resolve().parents[2] / "g1-cup-grasp"; sys.path.insert(0, str(G1 / "scripts"))
from g1_sim import G1Sim, load_seed, ARM_JOINTS, HAND_JOINTS
from kinematics import ArmIK
from recorder import Recorder
from vision import calibrate, locate_colored, fit_silhouette, cylinder_vertices
import run_attempt as RA

MODEL = "claude-fable-5-1 (Claude Code)"
DEFAULTS = {
    "scene": None, "markers": None, "photo_arm_pose": None, "left_arm_pose": None, "kp": 8.0,
    "pote_height_m": 0.075, "lid_knob_above_pote_m": 0.058,      # pegador = topo do pote + tampa (medido nas malhas)
    "pinch_offset_palm": [0.11, 0.03, 0.0],  # centro do botao no referencial da palma; medido no mapa de pega (16/09): (0,11, 0,03) levanta 77 mm, (0,125, y) perde o botao
    "fingers_dir": [1.0, 0.0, 0.0],          # direcao dos dedos no mundo na pinca (palma para baixo)
    "hover_m": 0.12,                 # altura do ponto de passagem acima da pose de pega
    "approach_back_m": 0.12,         # recuo ao longo do eixo dos dedos antes de avancar: com 12 cm o indicador fica fora da aba da tampa (raio 6,3 cm)
    "safe_z": 1.10,                  # altura de passagem lateral (o aro do coador vai a 1,0 m)
    "pre_reach_offset": [-0.06, -0.06, 0.12],   # ponto de passagem antes de ir por cima (evita o polegar no tronco, como na etapa 1)
    "turn_extra_deg": 25.0,          # giro alem do rumo do pote: a pega lateral exige o cotovelo para fora
    "close_gain": 3.0,               # alvo de fecho = aberta + ganho x (fechada humana - aberta), limitado ao range; 3,0 pelo mapa de pega
    "knob_center_above_lid_m": 0.049,# centro do botao acima da base da tampa (medido na malha: botao de z 40 a 58 mm)
    "close_frames": 45, "lift_m": 0.10, "place_xy": [0.26, 0.24], "place_hover_m": 0.02,
    "turn_seconds": 2.0, "move_seconds": 2.0, "settle_frames": 20, "hold_frames": 60, "max_joint_speed_deg_s": 240.0,
    "waist_yaw_max_deg": 35.0, "ik_iterations": 35, "reason": "",
    "pote_radius_m": 0.0625, "pote_total_height_m": 0.083,   # pote (7,5 cm) + tampa ate a aba: cilindro para o ajuste de silhueta
}


def run(params, directory):
    p = {**DEFAULTS, **params}; directory = Path(directory); directory.mkdir(parents=True, exist_ok=False)
    (directory / "parameters.json").write_text(json.dumps(p, indent=2) + "\n")
    board = json.loads(Path(p["markers"]).read_text()); sim = G1Sim(p["scene"], cup_body="tampa"); m, d = sim.m, sim.d
    arm, hand, fps = load_seed(0); fps = 30; gi = int(np.flatnonzero(hand[:, 4] > 0.5)[0])
    for n, v in zip(ARM_JOINTS, p["photo_arm_pose"]): d.qpos[m.jnt_qposadr[m.joint(n).id]] = v
    for n, v in zip(HAND_JOINTS, hand[0]): d.qpos[m.jnt_qposadr[m.joint(n).id]] = v
    for n, v in zip([j.replace("right_", "left_") for j in ARM_JOINTS], p["left_arm_pose"]): d.qpos[m.jnt_qposadr[m.joint(n).id]] = v
    mujoco.mj_forward(m, d); sim.q_des[:] = d.qpos[sim.qadr]; sim.set_targets(HAND_JOINTS, hand[0], kp=p["kp"], kd=1)
    rec = Recorder(m, str(directory / "grasp-run.mp4")); samples, ik_errors = [], []; state = {"frames": 0, "phase": "settle"}
    lid = m.body("tampa").id; pote_b = m.body("pote").id; PALM = "right_wrist_yaw_link"; TRUNK = {"torso_link", "pelvis", "waist_yaw_link", "waist_roll_link", "left_shoulder_pitch_link", "left_shoulder_roll_link", "left_shoulder_yaw_link", "left_elbow_link"}
    def env_contacts():
        out = set()
        for i in range(d.ncon):
            c = d.contact[i]; b1 = m.body(int(m.geom_bodyid[c.geom1])).name; b2 = m.body(int(m.geom_bodyid[c.geom2])).name
            hand_ = lambda b: b.startswith("right_hand") or b == PALM
            if (hand_(b1) or hand_(b2) or "tampa" in (b1, b2)) and not ({b1, b2} <= {"tampa", "pote"}): out.add((b1, b2, m.geom(c.geom1).name, m.geom(c.geom2).name, round(float(c.pos[2]), 3)))
        return sorted(out)
    def self_collisions():
        out = set()
        for i in range(d.ncon):
            c = d.contact[i]; b1 = m.body(int(m.geom_bodyid[c.geom1])).name; b2 = m.body(int(m.geom_bodyid[c.geom2])).name
            if (b1.startswith("right_") and b2 in TRUNK) or (b2.startswith("right_") and b1 in TRUNK): out.add((b1, b2))
        return sorted(out)
    def advance(n):
        for _ in range(n):
            until = (state["frames"] + 1) / fps
            while d.time < until - m.opt.timestep / 2:
                tau = sim.kp * (sim.q_des - d.qpos[sim.qadr]) - sim.kd * d.qvel[sim.vadr] + d.qfrc_bias[sim.vadr]
                d.ctrl[:] = np.clip(tau, m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1]); mujoco.mj_step(m, d)
            mujoco.mj_forward(m, d); up = d.xmat[lid].reshape(3, 3)[:, 2]
            table = [m.body(int(m.geom_bodyid[c.geom1 if m.body(int(m.geom_bodyid[c.geom2])).name == "mesa" else c.geom2])).name for c in d.contact[:d.ncon] if "mesa" in (m.body(int(m.geom_bodyid[c.geom1])).name, m.body(int(m.geom_bodyid[c.geom2])).name) and any(m.body(int(m.geom_bodyid[g])).name.startswith("right_") for g in (c.geom1, c.geom2))]
            samples.append({"t": float(d.time), "phase": state["phase"], "cup": d.xpos[lid].tolist(), "cup_tilt_deg": float(np.degrees(np.arccos(np.clip(up[2], -1, 1)))),
                            "palm": sim.body_pos(PALM).tolist(), "palm_mat": d.xmat[m.body(PALM).id].reshape(3, 3).tolist(), "fingers": sorted({n.split("_")[0] for n in sim.finger_contacts()}),
                            "finger_links": sorted(sim.finger_contacts()), "hand_table": table, "pote": d.xpos[pote_b].tolist(), "self_collision": self_collisions(), "env_contacts": env_contacts(),
                            "waist_yaw": float(d.qpos[m.jnt_qposadr[m.joint("waist_yaw_joint").id]]), "waist_pitch": 0.0, "tips": {}, "cup_table": False, "table_penetration_m": 0.0})
            rec.frame(d); state["frames"] += 1
    def stage(name, caption): state["phase"] = name; rec.event(caption, phase=name)
    max_step = np.radians(p["max_joint_speed_deg_s"]) / fps
    ik = ArmIK(sim); IKJ = list(ARM_JOINTS); ik_ref = arm[0].copy()
    def move(pos, R, seconds):
        start, start_r = ik.fk(sim.q(IKJ)); q = sim.q(IKJ); rv = cv2.Rodrigues(R @ start_r.T)[0].ravel(); n = max(1, round(seconds * fps))
        for i in range(n):
            u = (i + 1) / n; u = u * u * (3 - 2 * u); Ru = cv2.Rodrigues(rv * u)[0] @ start_r
            q, err = ik.solve(start + u * (np.asarray(pos) - start), Ru, q, ik_ref, iterations=p["ik_iterations"], max_step=max_step); ik_errors.append({"phase": state["phase"], **err}); sim.set_targets(IKJ, q); advance(1)
    report = {"controller_model": MODEL, "task": "tirar a tampa do pote (pinca por cima)", "parameters": p}
    try:
        stage("settle", "robo em repouso"); advance(20)
        rgb0 = RA.render(m, d); RA.save_rgb(directory / "camera-initial.png", rgb0); cal = calibrate(rgb0, board); (directory / "camera-localization.json").write_text(json.dumps(cal, indent=2))
        det = locate_colored(rgb0, cal, board, "brown", 0.06)
        fx_, fy_, rms = fit_silhouette(det["bbox"], cylinder_vertices(p["pote_radius_m"], p["pote_total_height_m"]), cal, board, det["position"][:2])
        det.update({"centroid_position": det["position"], "position": [fx_, fy_, board["table_z"]], "silhouette_fit_rms_px": rms, "method": "marrom + ajuste de silhueta de cilindro conhecido"}); est = np.array(det["position"]); true_pote = d.xpos[pote_b].copy()
        det["true_pote_privileged"] = true_pote.tolist(); det["estimate_error_xy_m"] = float(np.hypot(*(est[:2] - true_pote[:2]))); (directory / "cup-detection.json").write_text(json.dumps(det, indent=2))
        stage("perception", f"camera: pote em x={est[0]:.3f} y={est[1]:.3f} (RGB marrom + marcadores)")
        knob = np.array([est[0], est[1], board["table_z"] + p["pote_height_m"] + p["knob_center_above_lid_m"]])
        fx = np.array(p["fingers_dir"], float); fx /= np.linalg.norm(fx); fy = np.array([0, 0, -1.0]); fz = np.cross(fx, fy); R = np.column_stack([fx, fy, fz]); off = np.array(p["pinch_offset_palm"])
        # giro do tronco para o pote (como na pega), limitado
        theta = float(np.clip(np.arctan2(est[1], est[0]) + np.radians(p["turn_extra_deg"]), -np.radians(p["waist_yaw_max_deg"]), np.radians(p["waist_yaw_max_deg"])))
        if abs(theta) > np.radians(3):
            stage("turn", f"gira o tronco {np.degrees(theta):+.0f} graus para o pote"); q0 = float(d.qpos[m.jnt_qposadr[m.joint("waist_yaw_joint").id]]); nt = round(p["turn_seconds"] * fps)
            for i in range(nt):
                u = (i + 1) / nt; u = u * u * (3 - 2 * u); sim.set_targets(["waist_yaw_joint"], [q0 + u * (theta - q0)]); advance(1)
        grasp_palm = knob - R @ off                       # palma na pose de pega
        back = grasp_palm - R @ np.array([p["approach_back_m"], 0.0, 0.0])   # recuada ao longo do eixo dos dedos: o indicador fica fora da tampa
        stage("rise", f"sobe a mao a {p['safe_z']:.2f} m (o aro do coador vai a 1,0)"); pos0, R0 = ik.fk(sim.q(IKJ)); move(np.array([pos0[0], pos0[1], p["safe_z"]]), R0, 1.2)
        stage("orient", "vira a palma para baixo, no alto"); pos0, _ = ik.fk(sim.q(IKJ)); move(pos0, R, 1.2)
        stage("above", "vai ate acima do ponto de aproximacao"); move(np.array([back[0], back[1], p["safe_z"]]), R, p["move_seconds"])
        stage("descend", f"desce ate a altura do botao, {p['approach_back_m']*100:.0f} cm atras dele"); move(back, R, 1.5); advance(p["settle_frames"])
        stage("approach", "avanca na horizontal ate o botao ficar entre os dedos"); move(grasp_palm, R, 1.5); advance(p["settle_frames"])
        lid0 = d.xpos[lid].copy()
        stage("pinch", "fecha a mao no botao"); q_open = sim.q(HAND_JOINTS)
        lo = m.jnt_range[[m.joint(j).id for j in HAND_JOINTS], 0]; hi = m.jnt_range[[m.joint(j).id for j in HAND_JOINTS], 1]
        q_close = np.clip(hand[0] + p["close_gain"] * (hand[gi] - hand[0]), lo, hi)
        for i in range(p["close_frames"]):
            u = (i + 1) / p["close_frames"]; sim.set_targets(HAND_JOINTS, q_open + u * (q_close - q_open)); advance(1)
        advance(15)
        stage("lift", f"levanta a tampa {p['lift_m']*100:.0f} cm"); pos, _ = ik.fk(sim.q(IKJ)); move(pos + [0, 0, p["lift_m"]], R, 1.5); advance(10)
        lifted = d.xpos[lid].copy()
        stage("transport", f"leva a tampa para ({p['place_xy'][0]:.2f}, {p['place_xy'][1]:.2f})")
        pos, _ = ik.fk(sim.q(IKJ)); hand_to_lid = d.xpos[lid] - pos      # onde a tampa esta em relacao a palma AGORA (medido)
        move(np.array([p["place_xy"][0], p["place_xy"][1], pos[2]]) - [hand_to_lid[0], hand_to_lid[1], 0], R, p["move_seconds"]); advance(10)
        stage("lower", f"desce ate a tampa ficar a {p['place_hover_m']*100:.0f} cm da mesa"); pos, _ = ik.fk(sim.q(IKJ))
        low = min(d.xpos[m.body(b).id][2] for b in ("tampa",)) ; dz = (low - board["table_z"]) - p["place_hover_m"]
        move(pos - [0, 0, max(dz, 0)], R, 1.5); advance(p["settle_frames"])
        stage("release", "abre a mao"); q_now = sim.q(HAND_JOINTS)
        for i in range(p["close_frames"]):
            u = (i + 1) / p["close_frames"]; sim.set_targets(HAND_JOINTS, q_now + u * (hand[0] - q_now)); advance(1)
        advance(10)
        stage("retreat", "sobe a mao 8 cm"); pos, _ = ik.fk(sim.q(IKJ)); move(pos + [0, 0, 0.08], R, 1.0)
        stage("hold", "espera 2 s"); advance(p["hold_frames"])
        rgb1 = RA.render(m, d); RA.save_rgb(directory / "camera-final.png", rgb1)
        final = d.xpos[lid].copy(); up = d.xmat[lid].reshape(3, 3)[:, 2]; tilt = float(np.degrees(np.arccos(np.clip(up[2], -1, 1))))
        lifted_ok = bool(lifted[2] - lid0[2] > 0.5 * p["lift_m"]); on_pote = bool(np.hypot(*(final[:2] - true_pote[:2])) < 0.08)
        report.update({"status": "completed", "pote_estimate": est.tolist(), "pote_true_privileged": true_pote.tolist(), "lid_initial": lid0.tolist(), "lid_lifted": lifted.tolist(), "lid_final": final.tolist(),
                       "lid_lift_m": float(lifted[2] - lid0[2]), "lid_final_tilt_deg": tilt, "lid_place_error_m": float(np.hypot(final[0] - p["place_xy"][0], final[1] - p["place_xy"][1])),
                       "pote_moved_m": float(np.linalg.norm(d.xpos[pote_b] - true_pote)), "warnings": sim.warnings(), "self_collision_frames": sum(1 for s in samples if s["self_collision"]),
                       "hand_table_frames": sum(bool(s["hand_table"]) for s in samples), "max_ik_position_error_m": max(e["position_error_m"] for e in ik_errors),
                       "fingers_at_end": sorted(sim.finger_contacts()), "lid_still_on_pote": on_pote,
                       "accepted": bool(lifted_ok and not on_pote and tilt < 15 and not any(s["self_collision"] for s in samples) and not sim.finger_contacts())})
    except Exception as exc:
        report.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"}); raise
    finally:
        (directory / "physics-report.json").write_text(json.dumps(report, indent=2) + "\n"); (directory / "trajectory.json").write_text(json.dumps(samples) + "\n"); (directory / "ik.json").write_text(json.dumps(ik_errors) + "\n")
        report["video_seconds"] = rec.close(str(directory / "timeline.json")); (directory / "run-report.json").write_text(json.dumps(report, indent=2) + "\n")
        (directory / "provenance.json").write_text(json.dumps({"model": MODEL, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "scene": p["scene"], "time": time.strftime("%Y-%m-%d %H:%M")}, indent=2))
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--parameters", required=True); ap.add_argument("--run-dir", required=True); a = ap.parse_args()
    r = run(json.loads(Path(a.parameters).read_text()), a.run_dir); print(json.dumps({k: r.get(k) for k in ("status", "error", "accepted", "lid_lift_m", "lid_final_tilt_deg", "lid_place_error_m", "lid_still_on_pote", "pote_moved_m", "self_collision_frames", "hand_table_frames", "max_ik_position_error_m", "fingers_at_end")}, indent=1))
