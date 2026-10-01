"""Open motor-driven fingers after table support, then withdraw the hand.

The spoon remains a free body. Finger IK uses scratch data; it never writes the
live object pose or applies external support forces.
"""
import argparse
from collections import deque
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import sys

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'g1-cup-grasp/scripts'))
from g1_sim import G1Sim, ARM_JOINTS, HAND_JOINTS
from kinematics import ArmIK
from elbow_anatomy import ElbowAnatomy
from brew_state import BrewState
from liquid_mass import LiquidMassCoupler
from kettle_liquid import KettleWater

ap = argparse.ArgumentParser()
ap.add_argument('--source', type=Path, required=True)
ap.add_argument('--out', type=Path, required=True)
ap.add_argument('--opening-mm', type=float, default=6.)
ap.add_argument('--clearance-opening', action='store_true')
ap.add_argument('--avoid-spoon-withdraw', action='store_true')
ap.add_argument('--escape-angle', type=float, default=0.)
ap.add_argument('--opening-settle-seconds', type=float, default=0.)
ap.add_argument('--prewithdraw-drop', type=float, default=0.)
# Auditoria: em vez de abortar no primeiro passo acima do limite de força,
# registra o perfil inteiro e deixa a simulação terminar. Serve para separar
# "portão apertado demais" de "a colher realmente foi arrastada".
ap.add_argument('--grip-gate', type=float, default=.05)
ap.add_argument('--observe-grip', action='store_true',
                help='nao aborta por forca de contato; so mede')
ap.add_argument('--observe-penetration', action='store_true',
                help='nao aborta por penetracao transitoria; so mede')
a = ap.parse_args()
r = json.loads((a.source / 'report.json').read_text())
assert r['pass'] and r.get('task') == 'place_spoon' and (r.get('supported_at_end') or (r.get('placement_contact_latched') and r.get('release_clearance_m',1)<.001))
a.out.mkdir(exist_ok=False)
for p in [Path(__file__)] + [Path('scripts') / n for n in [
        'brew_state.py', 'coffee_grounds.py', 'grounds_mass.py', 'kettle_heater.py',
        'liquid_mass.py', 'kettle_liquid.py', 'elbow_anatomy.py']]:
    shutil.copy2(p, a.out / p.name)
s = G1Sim(r['scene'])
m, d = s.m, s.d
brew = BrewState(m, a.source)
ck = np.load(a.source / 'continuation.npz')
water = KettleWater()
coupler = LiquidMassCoupler(m, float(ck['dry_kettle_kg']))
for n, v in json.loads((a.source / 'liquid-state.json').read_text()).items():
    setattr(water, n, v)
for n in ['body_mass', 'body_ipos', 'body_inertia', 'body_iquat', 'bvh_aabb']:
    getattr(m, n)[:] = ck[n]
mujoco.mj_setConst(m, mujoco.MjData(m))
mujoco.mj_setState(m, d, ck['integration'], mujoco.mjtState.mjSTATE_INTEGRATION)
mujoco.mj_forward(m, d)
s.kp[:], s.kd[:] = ck['kp'], ck['kd']
target = ck['last_target'].copy()
names = ['waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint'] + list(ARM_JOINTS) + [n.replace('right_', 'left_') for n in ARM_JOINTS]
ik = ArmIK(s, names, palm='right_wrist_yaw_link')
bounds = ik.bounds.copy()
bounds[0] = [-.7, .7]
bounds[1:3] = np.deg2rad([[-12, 12], [-12, 12]])
anatomy = ElbowAnatomy(m)
anatomy.bound_search(m, ik.joint_names, bounds)
for j in [7, 14]:
    bounds[j:j+3] = np.deg2rad([[-60, 60], [-45, 45], [-30, 30]])
hj = [m.joint(n).id for n in HAND_JOINTS]
ha = m.jnt_qposadr[hj]
finger_bounds = m.jnt_range[hj]
h = target[ha].copy()
q = target[ik.qa].copy()
spoon = m.body('scoop').id
palm = m.body('right_wrist_yaw_link').id
other = m.body('left_wrist_yaw_link').id
table = m.geom('tampo').id
tips = [next(g for g in range(m.ngeom) if m.geom_contype[g] and m.body(int(m.geom_bodyid[g])).name == 'right_hand_' + n + '_link') for n in ['thumb_2', 'index_1']]
shaft = [g for g in range(m.ngeom) if m.geom_contype[g] and m.geom_bodyid[g] == spoon and .040 < m.geom_pos[g, 0] < .059]
anchors = []
normal = np.zeros(m.nu)
finger_mask = np.array([m.joint(m.actuator_trnid[i, 0]).name.startswith('right_hand') for i in range(m.nu)])
for g in tips:
    candidates = []
    for sg in shaft:
        pts = np.zeros(6)
        distance = mujoco.mj_geomDistance(m, d, g, sg, .02, pts)
        candidates.append((distance, pts))
    distance, pts = min(candidates, key=lambda x: x[0])
    n = (pts[3:] - pts[:3]) * np.sign(distance)
    n /= max(np.linalg.norm(n), 1e-9)
    body = int(m.geom_bodyid[g])
    local = d.xmat[body].reshape(3, 3).T @ (pts[:3] - d.xpos[body])
    anchors.append((body, local, pts[:3].copy(), n))
    jac = np.zeros((3, m.nv))
    rot = np.zeros_like(jac)
    mujoco.mj_jac(m, d, jac, rot, pts[:3], body)
    normal += (jac[:, s.vadr].T @ (n * r.get('pinch_force_N', 3))) * finger_mask
# Preserve the actual saved motor preload while ramping it down, including
# contact-force integral runs whose final preload differs from nominal squeeze.
if 'grip_motor_torque_Nm' in r:
    normal=np.array(r['grip_motor_torque_Nm'])
scratch = mujoco.MjData(m)
hands = [g for g in range(m.ngeom) if m.geom_contype[g] and m.body(int(m.geom_bodyid[g])).name.startswith(('left_hand', 'left_wrist', 'right_hand', 'right_wrist'))]
spoon_geoms = [g for g in range(m.ngeom) if m.geom_contype[g] and m.geom_bodyid[g] == spoon]
right_hand_geoms = [g for g in hands if m.body(int(m.geom_bodyid[g])).name.startswith('right_hand')]
opening_pairs = [(g, sg) for g in right_hand_geoms for sg in spoon_geoms
                 if a.clearance_opening and mujoco.mj_geomDistance(m, d, g, sg, .02, None) < .02]
props = [m.body(n).id for n in ['chaleira', 'base_eletrica', 'pote', 'tampa', 'coador', 'copo']]
prop0 = d.xpos[props].copy()
spoon0 = d.xpos[spoon].copy()
op = d.xpos[other].copy()
withdraw = None
p0 = None
R0 = None
dt = m.opt.timestep
rows, states, samples, liquid = [], [], [], []
peak = np.zeros(m.nu)
bad = None
passed = False
clear_steps = 0
window = deque(maxlen=int(2 / dt))
hold_drift = None
min_gap = 1.
grip_max_after = 0.
pen_max = 0.
spoon_disp_max = 0.
over_gate_steps = 0
audit = []

prewithdraw_seconds = 2. if a.prewithdraw_drop else 0.
for k in range(int((18+a.opening_settle_seconds+prewithdraw_seconds) / dt)):
    t = k * dt
    u = np.clip(t / 3, 0, 1)
    u = u * u * (3 - 2 * u)
    phase = 'open_supported_spoon' if withdraw is None else 'withdraw_empty_hand'
    if k % 10 == 0:
        if withdraw is None:
            scratch.qpos[:] = d.qpos
            def finger_residual(x):
                scratch.qpos[ha] = x
                mujoco.mj_kinematics(m, scratch)
                points = [scratch.xpos[b] + scratch.xmat[b].reshape(3, 3) @ local - (start - n * a.opening_mm/1000 * u) for b, local, start, n in anchors]
                gaps = [min(0, mujoco.mj_geomDistance(m, scratch, g, sg, .01, None)-.002*u)
                        for g, sg in opening_pairs]
                table_gaps = [min(0, mujoco.mj_geomDistance(m, scratch, g, table, .02, None)-.010)
                              for g in right_hand_geoms] if a.clearance_opening else []
                return np.r_[np.concatenate(points) * (100 if a.clearance_opening else 1000), (x - h) * .1,
                             np.array(gaps)*20000, np.array(table_gaps)*20000]
            lo = np.maximum(finger_bounds[:, 0], h - .005)
            hi = np.minimum(finger_bounds[:, 1], h + .005)
            h = least_squares(finger_residual, np.clip(h, lo+1e-9, hi-1e-9), bounds=(lo, hi), max_nfev=30).x
        else:
            v = np.clip((t - withdraw - prewithdraw_seconds - 3) / 5, 0, 1)
            v = v * v * (3 - 2 * v)
            escape_fraction=np.clip((t-withdraw-prewithdraw_seconds)/3,0,1)
            escape_fraction=escape_fraction*escape_fraction*(3-2*escape_fraction)
            drop_fraction = np.clip((t-withdraw)/max(prewithdraw_seconds,1e-9),0,1)
            drop_fraction = drop_fraction*drop_fraction*(3-2*drop_fraction)
            goal = p0 + escape*.04*escape_fraction + [0, 0, .10*v-a.prewithdraw_drop*drop_fraction*(1-v)]
            withdraw_pairs = []
            if a.avoid_spoon_withdraw:
                for g in right_hand_geoms:
                    nearby = sorted((mujoco.mj_geomDistance(m, d, g, sg, .015, None), sg) for sg in spoon_geoms)
                    withdraw_pairs.extend((g, sg) for gap_, sg in nearby[:3] if gap_ < .015)
            def arm_residual(x):
                pp, rr = ik.fk(x)
                gaps = [min(0, mujoco.mj_geomDistance(m, ik.d, g, table, .04, None)-.009) for g in hands]
                spoon_gaps = [min(0, mujoco.mj_geomDistance(m, ik.d, g, sg, .01, None)-.0015)
                              for g, sg in withdraw_pairs]
                return np.r_[(pp-goal)*1000, Rotation.from_matrix(rr @ R0.T).as_rotvec()*30, (ik.d.xpos[other]-op)*500, (x-q)*.1, np.array(gaps)*5000, np.array(spoon_gaps)*20000, anatomy.penalty(ik.d)]
            lo = np.maximum(bounds[:, 0], q-.004)
            hi = np.minimum(bounds[:, 1], q+.004)
            q = least_squares(arm_residual, np.clip(q, lo+1e-9, hi-1e-9), bounds=(lo, hi), max_nfev=30).x
        brew.step(d, .02, water)
        bs = [m.body(n).id for n in ['chaleira', 'coador', 'copo']]
        flow = water.step(.02, *sum(([d.xpos[b], d.xmat[b].reshape(3, 3)] for b in bs), []))
        coupler.apply(d, water)
        liquid.append({'t': float(d.time), **flow})
    target[ha], target[ik.qa] = h, q
    error = target[s.qadr] - d.qpos[s.qadr]
    friction = m.dof_frictionloss[s.vadr] * np.tanh(error/.001) * np.isin(s.qadr, np.r_[ik.qa, ha])
    tau = s.kp*error - s.kd*d.qvel[s.vadr] + d.qfrc_bias[s.vadr] + (1-u)*normal + friction
    d.ctrl[:] = np.clip(tau, m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1])
    peak = np.maximum(peak, np.abs(d.ctrl))
    mujoco.mj_step(m, d)
    grip = 0.
    grip_contacts = 0
    supported = False
    for ci, c in enumerate(d.contact):
        if c.dist >= 0:
            continue
        gs = [int(c.geom1), int(c.geom2)]
        bn = [m.body(int(m.geom_bodyid[g])).name for g in gs]
        allowed = 'scoop' in bn and any(n.startswith('right_hand') for n in bn) and (withdraw is None or t-withdraw<3+prewithdraw_seconds)
        if 'scoop' in bn and table in gs:
            supported = True
        if allowed:
            grip_contacts += 1
            force = np.zeros(6)
            mujoco.mj_contactForce(m, d, ci, force)
            grip += float(force[0])
            if withdraw is not None and grip>a.grip_gate and not a.observe_grip:
                bad={'reason':'contact force during horizontal release exceeds0.05N','N':grip}
            pen_max = max(pen_max, -c.dist)
            if -c.dist > .0002 and not a.observe_penetration:
                bad = {'reason': 'spoon contact penetration', 'm': float(-c.dist)}
        if any(n.startswith(('left_', 'right_', 'torso', 'waist', 'pelvis', 'head')) for n in bn) and not allowed:
            bad = {'reason': 'forbidden contact', 'bodies': bn}
    gap = min(mujoco.mj_geomDistance(m, d, g, table, 1, None) for g in hands)
    min_gap = min(min_gap, gap)
    if gap < .008:
        bad = {'reason': 'hand/table clearance', 'm': float(gap)}
    if not anatomy.valid(d, tolerance_deg=2):
        bad = {'reason': 'elbow geometry', 'deg': anatomy.angles(d).tolist()}
    if np.any(np.abs(np.rad2deg(d.qpos[ik.qa[np.r_[7:10, 14:17]]])) > [62, 47, 32, 62, 47, 32]):
        bad = {'reason': 'wrist bounds'}
    if np.max(np.linalg.norm(d.xpos[props]-prop0, axis=1)) > .005:
        bad = {'reason': 'other prop drift'}
    if np.linalg.norm(d.xpos[spoon]-spoon0) > .02:
        bad = {'reason': 'spoon moved more than2cm during release'}
    if water.spilled_ml > .5 or brew.grounds.spilled_g > .05:
        bad = {'reason': 'spill'}
    if s.warnings():
        bad = {'reason': 'numerical warning', 'warnings': s.warnings()}
    opening_gap = min((mujoco.mj_geomDistance(m, d, g, sg, .01, None)
                       for g, sg in opening_pairs), default=.01)
    clear_steps = clear_steps+1 if t>3+a.opening_settle_seconds and grip<.02 and supported and (not a.clearance_opening or opening_gap>.001) else 0
    if withdraw is None and clear_steps > int(.3/dt):
        withdraw = t
        # Preserve the equilibrium motor target at the controller transition.
        # Replacing it with measured q removes the small static PD/friction
        # offset and can drive the finger back into the supported spoon.
        p0, R0 = ik.fk(q)
        op = ik.d.xpos[other].copy()
        angle = np.deg2rad(a.escape_angle)
        escape = d.xmat[spoon].reshape(3,3) @ np.array([np.cos(angle), np.sin(angle), 0.])
        escape[2] = 0.
        escape /= max(np.linalg.norm(escape),1e-9)
    if withdraw is not None and t>withdraw+8+prewithdraw_seconds and supported and grip<.02:
        window.append(d.xpos[spoon].copy())
    else:
        window.clear()
    if len(window) == window.maxlen:
        hold_drift = float(np.max(np.linalg.norm(np.array(window)-window[0], axis=1)))
        passed = hold_drift < .002 and d.xpos[palm, 2]-p0[2] > .09
    if withdraw is not None:
        grip_max_after = max(grip_max_after, grip)
        spoon_disp = float(np.linalg.norm(d.xpos[spoon]-spoon0))
        spoon_disp_max = max(spoon_disp_max, spoon_disp)
        if grip > a.grip_gate:
            over_gate_steps += 1
        audit.append({'t': float(d.time), 'rel_t': float(t-withdraw), 'grip_N': float(grip),
                      'contacts': int(grip_contacts), 'spoon_disp_m': spoon_disp,
                      'supported': bool(supported)})
    if k % 16 == 0:
        states.append(d.qpos.copy())
        rows.append({'t': float(d.time), 'phase': phase, 'table_gap_m': float(gap)})
        samples.append({'t': float(d.time), 'phase': phase, 'grip_N': grip, 'supported': supported, 'elbow_flexion_deg': anatomy.angles(d).tolist()})
    if k % 2500 == 0:
        print({'t': t, 'phase': phase, 'grip_N': grip, 'supported': supported}, flush=True)
    if bad or passed:
        break
report = {'model': 'gpt-6-astra', 'source': str(a.source), 'scene': r['scene'],
          'opening_mm': a.opening_mm, 'clearance_opening': a.clearance_opening,
          'avoid_spoon_withdraw': a.avoid_spoon_withdraw, 'escape_angle_deg': a.escape_angle,
          'withdraw_reference': 'equilibrium motor target FK; continuous transition',
          'opening_settle_seconds': a.opening_settle_seconds,
          'prewithdraw_drop_m': a.prewithdraw_drop,
          'escape_plane': 'world horizontal',
          'pass': bool(passed and not bad), 'failure': bad or (None if passed else {'reason': 'release/withdraw acceptance timeout'}),
          'coffee_completed': False, 'initial_water_ml': water.initial_ml,
          'scope': 'free spoon released onto table using finger motors; fixed pelvis; reduced grounds/water/heat',
          'withdraw_at_elapsed_s': withdraw, 'hold_drift_m': hold_drift,
          'min_hand_table_gap_m': min_gap, 'water_state': asdict(water),
          'audit': {'observe_grip': bool(a.observe_grip), 'grip_gate_N': a.grip_gate,
                    'grip_max_after_withdraw_N': grip_max_after,
                    'steps_over_gate': over_gate_steps,
                    'spoon_displacement_max_m': spoon_disp_max,
                    'penetration_max_m': float(pen_max),
                    'observe_penetration': bool(a.observe_penetration),
                    'series': audit[::8]},
          'rows': rows, 'grasp_samples': samples, 'liquid_samples': liquid,
          'peak_motor_torques_Nm': {n: float(peak[i]) for n, i in s.act_joint.items()}}
(a.out/'report.json').write_text(json.dumps(report, indent=2))
np.savez_compressed(a.out/'trajectory.npz', qpos=states)
integration = np.zeros(mujoco.mj_stateSize(m, mujoco.mjtState.mjSTATE_INTEGRATION))
mujoco.mj_getState(m, d, integration, mujoco.mjtState.mjSTATE_INTEGRATION)
np.savez_compressed(a.out/'continuation.npz', integration=integration, last_target=target, kp=s.kp, kd=s.kd,
                    body_mass=m.body_mass, body_ipos=m.body_ipos, body_inertia=m.body_inertia,
                    body_iquat=m.body_iquat, bvh_aabb=m.bvh_aabb, dry_kettle_kg=float(ck['dry_kettle_kg']))
(a.out/'liquid-state.json').write_text(json.dumps(asdict(water), indent=2))
brew.save(a.out)
print({k: v for k, v in report.items() if k not in ['rows', 'grasp_samples', 'liquid_samples', 'peak_motor_torques_Nm']})
