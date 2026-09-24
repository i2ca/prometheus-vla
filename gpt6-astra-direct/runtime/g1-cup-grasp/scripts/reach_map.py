"""Mapa offline de alcance x visibilidade para o esquema v2 (cintura so em yaw, limitada ao campo da camera da cabeca):
para cada posicao do copo, erro da IK so com o braco, erro com braco + yaw limitado, e se o copo (base e borda) cai dentro
do quadro da camera da cabeca com o tronco girado. Nao roda fisica. Uso: .venv/bin/python scripts/reach_map.py experiments/attempt-28.parameters.json"""
import json, sys, os
import numpy as np, cv2, mujoco
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from g1_sim import G1Sim, load_seed, ARM_JOINTS
from kinematics import ArmIK
import run_attempt; p = {**run_attempt.DEFAULTS, **json.load(open(sys.argv[1]))}; sim = G1Sim("scene_grasp.xml"); m, d = sim.m, sim.d
arm, hand, fps = load_seed(0); gi = int(np.flatnonzero(hand[:, 4] > 0.5)[0])
home = arm[0].copy(); home[1] = p["home_shoulder_roll_rad"]
for n, v in zip(ARM_JOINTS, home): d.qpos[m.jnt_qposadr[m.joint(n).id]] = v
LEFT = [j.replace("right_", "left_") for j in ARM_JOINTS]
for n, v in zip(LEFT, p["left_arm_pose"]): d.qpos[m.jnt_qposadr[m.joint(n).id]] = v
mujoco.mj_forward(m, d)
ref = ArmIK(sim).reference_grasp(arm, hand, gi, p["offset"], p["gain"]); arm_ref = arm[gi].copy(); arm_ref[1] = p["arm_nullspace_roll"]
base_rot = cv2.Rodrigues(np.array([0.0, p["palm_pitch_rad"], 0.0]))[0] @ ref["palm_rotation"]
ik_arm = ArmIK(sim); ik_w = ArmIK(sim, ["waist_yaw_joint"] + list(ARM_JOINTS)); ik_w.weights = np.r_[p["waist_weight"], np.ones(7)]; ik_w.nullspace_gain = p["waist_nullspace_gain"]
cam = m.camera("head_camera").id; fovy = np.radians(m.cam_fovy[cam]); W, H = 848, 480; fy = H / 2 / np.tan(fovy / 2)
def visible(cup, yaw):
    wj = m.jnt_qposadr[m.joint("waist_yaw_joint").id]; q0 = d.qpos[wj]; d.qpos[wj] = yaw; mujoco.mj_forward(m, d)
    R = d.cam_xmat[cam].reshape(3, 3); t = d.cam_xpos[cam]; ok = True; pts = []
    for dz in (0.0, 0.092):
        for a in np.linspace(0, 2 * np.pi, 8, endpoint=False):
            pw = cup + [0.041 * np.cos(a), 0.041 * np.sin(a), dz]; pc = R.T @ (pw - t)   # camera mujoco olha -z
            u = W / 2 + fy * pc[0] / -pc[2]; v = H / 2 - fy * pc[1] / -pc[2]; pts.append((u, v)); ok &= (20 < u < W - 20) and (20 < v < H - 20) and pc[2] < 0
    d.qpos[wj] = q0; mujoco.mj_forward(m, d); return bool(ok)
k = np.radians(p["head_keep_cup_deg"]); rows = []
for x in np.arange(0.26, 0.461, 0.02):
    for y in np.arange(-0.30, 0.251, 0.05):
        cup = np.array([x, y, 0.752]); bearing = np.arctan2(y, x - 0.12); lo, hi = bearing - k, bearing + k
        theta = float(np.clip(np.arctan2(y, x), lo, hi)); rot = cv2.Rodrigues(np.array([0, 0, theta]))[0] @ base_rot
        target = cup + [0, 0, p["center_height"]] - rot @ np.array(p["cup_in_palm"])
        _, e1 = ik_arm.solve(target, rot, sim.q(ARM_JOINTS), arm_ref, iterations=300)
        ik_w.bounds = ik_w.bounds.copy(); ik_w.bounds[0] = np.clip([lo, hi], m.jnt_range[m.joint("waist_yaw_joint").id][0], m.jnt_range[m.joint("waist_yaw_joint").id][1])
        qw, e2 = ik_w.solve(target, rot, np.r_[theta, sim.q(ARM_JOINTS)], np.r_[theta, arm_ref], iterations=300)
        rows.append({"x": round(float(x), 2), "y": round(float(y), 2), "arm_err_mm": round(e1["position_error_m"] * 1000, 1), "waist_err_mm": round(e2["position_error_m"] * 1000, 1),
                     "waist_yaw_deg": round(float(np.degrees(qw[0])), 1), "visible_at_hold_yaw": visible(cup, qw[0]), "visible_at_zero": visible(cup, 0.0)})
json.dump(rows, open("results/reach-map-v2.json", "w"), indent=1)
xs = sorted({r["x"] for r in rows}); ys = sorted({r["y"] for r in rows})
print("legenda: A = braco so (<5mm) | W = braco+yaw limitado (<5mm) | . = fora ; minuscula = copo fora do quadro da cabeca na pose de pega")
print("   y:  " + " ".join(f"{y:+.2f}" for y in ys))
for x in xs:
    line = []
    for y in ys:
        r = next(r for r in rows if r["x"] == x and r["y"] == y); c = "A" if r["arm_err_mm"] < 5 else ("W" if r["waist_err_mm"] < 5 else ".")
        if not r["visible_at_hold_yaw"]: c = c.lower() if c != "." else ","
        line.append(f"  {c}  ")
    print(f"x={x:.2f} " + " ".join(line))
