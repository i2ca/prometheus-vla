"""Assenta a cena por N segundos com o robo na pose inicial e mede: deslocamento/inclinacao de cada objeto, penetracao
maxima, avisos, pares em contato; renderiza a grade 3x2 e a vista global. Uso: check_setup_scene.py scene/x.xml [--seconds 2]"""
import argparse, json, sys, os
import mujoco, numpy as np, cv2
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "g1-cup-grasp", "scripts")); from g1_sim import ARM_JOINTS, HAND_JOINTS, load_seed
ap = argparse.ArgumentParser(); ap.add_argument("scene"); ap.add_argument("--seconds", type=float, default=2.0); a = ap.parse_args()
m = mujoco.MjModel.from_xml_path(a.scene); d = mujoco.MjData(m)
p = json.load(open(os.path.join(os.path.dirname(__file__), "..", "..", "g1-cup-grasp", "experiments", "attempt-31.parameters.json"))); arm, hand, fps = load_seed(0)
home = arm[0].copy(); home[1] = p["home_shoulder_roll_rad"]
for n, v in zip(ARM_JOINTS, home): d.qpos[m.jnt_qposadr[m.joint(n).id]] = v
for n, v in zip(HAND_JOINTS, hand[0]): d.qpos[m.jnt_qposadr[m.joint(n).id]] = v
for n, v in zip([j.replace("right_", "left_") for j in ARM_JOINTS], p["left_arm_pose"]): d.qpos[m.jnt_qposadr[m.joint(n).id]] = v
mujoco.mj_forward(m, d)
init_pen = [(m.geom(c.geom1).name, m.geom(c.geom2).name, round(c.dist * 1000, 1)) for c in d.contact[:d.ncon] if c.dist < -0.001]
qadr = np.array([m.jnt_qposadr[m.actuator_trnid[i, 0]] for i in range(m.nu)]); vadr = np.array([m.jnt_dofadr[m.actuator_trnid[i, 0]] for i in range(m.nu)]); q_des = d.qpos[qadr].copy()
free = [m.body(m.jnt_bodyid[j]).name for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
p0 = {b: d.xpos[m.body(b).id].copy() for b in free}; warn0 = d.warning.number.copy(); pen = 0.0
for k in range(int(a.seconds / m.opt.timestep)):
    tau = 200 * (q_des - d.qpos[qadr]) - 5 * d.qvel[vadr] + d.qfrc_bias[vadr]; d.ctrl[:] = np.clip(tau, m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1]); mujoco.mj_step(m, d)
    if d.ncon: pen = min(pen, d.contact.dist[:d.ncon].min())
rep = {"scene": a.scene, "seconds": a.seconds, "initial_penetrations_over_1mm": init_pen, "max_penetration_mm": round(-pen * 1000, 2), "warnings": int((d.warning.number - warn0).sum()), "objects": {}}
for b in free:
    p1 = d.xpos[m.body(b).id]; up = d.xmat[m.body(b).id].reshape(3, 3)[2, 2]
    rep["objects"][b] = {"moved_mm": round(float(np.linalg.norm(p1 - p0[b]) * 1000), 1), "dz_mm": round(float((p1[2] - p0[b][2]) * 1000), 1), "tilt_deg": round(float(np.degrees(np.arccos(np.clip(up, -1, 1)))), 1)}
pairs = {}
for c in d.contact[:d.ncon]:
    k = " x ".join(sorted((m.body(m.geom_bodyid[c.geom1]).name, m.body(m.geom_bodyid[c.geom2]).name))); pairs[k] = pairs.get(k, 0) + 1
rep["contact_pairs_end"] = pairs
r = mujoco.Renderer(m, 360, 640); tiles = []
for cam in ("left_wrist_camera", "head_camera", "right_wrist_camera", "view_left", "view_center", "side_view"):
    r.update_scene(d, camera=cam); tiles.append(r.render().copy())
base = os.path.splitext(a.scene)[0]
cv2.imwrite(base + "-preview.png", cv2.cvtColor(np.vstack([np.hstack(tiles[:3]), np.hstack(tiles[3:])]), cv2.COLOR_RGB2BGR))
rh = mujoco.Renderer(m, 480, 848); rh.update_scene(d, camera="head_camera"); cv2.imwrite(base + "-head.png", cv2.cvtColor(rh.render(), cv2.COLOR_RGB2BGR))
r2 = mujoco.Renderer(m, 720, 1280); r2.update_scene(d, camera="global_view"); cv2.imwrite(base + "-global.png", cv2.cvtColor(r2.render(), cv2.COLOR_RGB2BGR))
json.dump(rep, open(base + "-check.json", "w"), indent=2); print(json.dumps(rep, indent=1, ensure_ascii=False))
