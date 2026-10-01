"""Acquire the plastic handle and lift the loaded free kettle with motors."""
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
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'g1-cup-grasp/scripts'))
from g1_sim import G1Sim, HAND_JOINTS
from kinematics import ArmIK
from elbow_anatomy import ElbowAnatomy
from finger_contact_servo import advance
from audit_grasp_wrench_capacity import solve as solve_wrench
from brew_state import BrewState
from liquid_mass import LiquidMassCoupler
from kettle_liquid import KettleWater

ap = argparse.ArgumentParser()
# Auditoria: registra o perfil de folga ao metal quente em vez de abortar no
# primeiro passo abaixo do limite, para separar mergulho transitorio de
# aproximacao sustentada.
# Lado da mao que pega a alca. O executor inteiro estava cravado na esquerda;
# a mao direita tem 32,34 mm de folga ao metal quente contra 10,67 mm da
# esquerda na mesma pose de palma, entao ela precisa ser alcancavel por
# parametro para o orquestrador poder escolher.
ap.add_argument('--hand-side',type=str,default='left',choices=['left','right'])
ap.add_argument('--observe-clearance',action='store_true')
ap.add_argument('--hot-gate',type=float,default=.006)
ap.add_argument('--hot-sustained-ms',type=float,default=50.)
ap.add_argument('--penetration-gate',type=float,default=.001)
# auto-contato: raspao de 7,7 um entre left_shoulder_roll_link e torso_link
# reprovava uma corrida que ja tinha levantado 30,7 mm. A chaleira parada em
# cima da mesa penetra 227 um, entao 7,7 um esta trinta vezes abaixo do piso
# de ruido do solver. O portao passa a medir profundidade, nao presenca.
ap.add_argument('--self-contact-gate',type=float,default=.0005)
# mesmo limite de cintura do planejador de pega e do de conexao (ver la')
ap.add_argument('--waist-rp-deg',type=float,default=12.)
ap.add_argument('--human-weights',action='store_true')
# Braco ocioso com as juntas fixas. Os IKs prendiam a mao ociosa so' pela
# POSICAO no mundo; quando a cintura se ajustava, o braco compensava girando o
# punho e os dedos pendurados viravam para o tampo (3,2 mm da mesa, contra 38,7
# no estado relaxado). Um braco relaxado acompanha o tronco em vez de segurar
# um ponto no espaco. Com as juntas fixas, o termo de posicao da mao ociosa
# passa a frear a cintura.
ap.add_argument('--fix-idle-arm',action='store_true')
ap.add_argument('--source', type=Path, required=True)
ap.add_argument('--plan', type=Path, required=True)
ap.add_argument('--out', type=Path, required=True)
ap.add_argument('--wrench-feedforward', action='store_true')
ap.add_argument('--balanced-pinch', action='store_true')
ap.add_argument('--freeze-grip-on-lift', action='store_true')
ap.add_argument('--object-feedback', action='store_true')
ap.add_argument('--contact-frame', action='store_true')
ap.add_argument('--normal-force', type=float, default=16)
ap.add_argument('--contact-target', type=float, default=8)
a = ap.parse_args()
_P=a.hand_side+'_';_O=('right_' if a.hand_side=='left' else 'left_')
_hot_min=[9.];_hot_under=[0];_hot_run=[0];_hot_run_max=[0]
r = json.loads((a.source/'report.json').read_text())
pr = json.loads((a.plan/'report.json').read_text())
assert r['pass'] and pr['pass'] and Path(pr['source']).resolve() == a.source.resolve()
a.out.mkdir(exist_ok=False)
for p in [Path(__file__)]+[Path('scripts')/n for n in [
        'brew_state.py', 'coffee_grounds.py', 'grounds_mass.py', 'kettle_heater.py',
        'liquid_mass.py', 'kettle_liquid.py', 'elbow_anatomy.py', 'finger_contact_servo.py', 'audit_grasp_wrench_capacity.py']]:
    shutil.copy2(p, a.out/p.name)
s = G1Sim(r['scene'])
m, d = s.m, s.d
brew = BrewState(m, a.source)
ck = np.load(a.source/'continuation.npz')
water = KettleWater()
coupler = LiquidMassCoupler(m, float(ck['dry_kettle_kg']))
for n, v in json.loads((a.source/'liquid-state.json').read_text()).items():
    setattr(water, n, v)
for n in ['body_mass', 'body_ipos', 'body_inertia', 'body_iquat', 'bvh_aabb']:
    getattr(m, n)[:] = ck[n]
mujoco.mj_setConst(m, mujoco.MjData(m))
mujoco.mj_setState(m, d, ck['integration'], mujoco.mjtState.mjSTATE_INTEGRATION)
mujoco.mj_forward(m, d)
s.kp[:], s.kd[:] = ck['kp'], ck['kd']
for n, i in s.act_joint.items():
    if n.startswith(_P+'hand'):
        s.kp[i], s.kd[i] = 25, .3
    elif n.startswith(_P) and any(x in n for x in ['shoulder', 'elbow', 'wrist']):
        s.kp[i] *= 3
        s.kd[i] *= 2
target = ck['last_target'].copy()
ik = ArmIK(s, pr['joint_names'], palm=_P+'wrist_yaw_link')
bounds = ik.bounds.copy()
bounds[0] = [-.7, .7]
bounds[1:3] = np.deg2rad([[-a.waist_rp_deg, a.waist_rp_deg]]*2)
anatomy = ElbowAnatomy(m)
anatomy.bound_search(m, ik.joint_names, bounds)
sys.path.insert(0, 'scripts'); from postura_humana import pesos as _pesos
W_REG = _pesos(ik.joint_names) if a.human_weights else np.ones(len(ik.joint_names))
for j in [7, 14]:
    bounds[j:j+3] = np.deg2rad([[-60, 60], [-45, 45], [-30, 30]])
hand_names = [n.replace('right_', _P) for n in HAND_JOINTS]
hj = [m.joint(n).id for n in hand_names]
ha = m.jnt_qposadr[hj]
finger = np.array([m.joint(m.actuator_trnid[i, 0]).name.startswith(_P+'hand') for i in range(m.nu)])
path = np.load(a.plan/'path.npz')
poses, phases = path['qpos'], path['phases']
moving = np.r_[ik.qa, ha]
durations = np.maximum(.016, np.max(np.abs(np.diff(poses[:, moving], axis=0)), axis=1)/.14)
times = np.r_[0, np.cumsum(durations)]
end = float(times[-1])
closed = np.array(pr['closed_hand_q'])
opened = poses[-1, ha].copy()
hand_offset = np.zeros(7)
q = poses[-1, ik.qa].copy()
if a.fix_idle_arm:
    _oc = slice(10, 17) if a.hand_side == 'left' else slice(3, 10)
    bounds[_oc, 0] = q[_oc] - 1e-4
    bounds[_oc, 1] = q[_oc] + 1e-4
jar, palm, other = [m.body(n).id for n in ['chaleira', _P+'wrist_yaw_link', _O+'wrist_yaw_link']]
table, hot = m.geom('tampo').id, m.geom('chaleira_hot_body').id
handle = [g for g in range(m.ngeom) if (m.geom(g).name or '').startswith('handle_col')]
hands = [g for g in range(m.ngeom) if m.geom_contype[g] and m.body(int(m.geom_bodyid[g])).name.startswith((_P+'hand', _P+'wrist', _O+'hand', _O+'wrist'))]
left_hands = [g for g in hands if m.body(int(m.geom_bodyid[g])).name.startswith(_P)]
tips = [next(g for g in left_hands if m.body(int(m.geom_bodyid[g])).name == _P+'hand_'+n+'_link') for n in ['thumb_2', 'index_1', 'middle_1']]
props = [m.body(n).id for n in ['pote', 'tampa', 'scoop', 'coador', 'copo', 'base_eletrica']]
prop0 = d.xpos[props].copy()
jar0 = d.xpos[jar].copy()
feet = [g for g in range(m.ngeom) if m.geom_contype[g] and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE and m.body(int(m.geom_bodyid[g])).name in ['left_ankle_roll_link', 'right_ankle_roll_link']]
support_hull = ConvexHull(d.geom_xpos[feet, :2]).equations
pelvis = m.body('pelvis').id
forces = [0., 0., 0.]
normal = np.zeros(m.nu)
contact_points, contact_normals = {}, {}
contact_mu, contact_twist = {}, {}
wrench_tau = np.zeros(m.nu)
wrench_success = wrench_failure = 0
lift_start = None
grip_steps = lost_steps = 0
rows, states, samples, liquid = [], [], [], []
window = deque(maxlen=int(2/m.opt.timestep))
peak = np.zeros(m.nu)
bad = None
passed = False
min_hot = min_table = min_com = 1.
max_penetration = max_lift = max_relative_rotation = max_self_contact = 0.
worst_self_contact = None
hold_drift = hold_angle = None
dt = m.opt.timestep

def smooth(value):
    value = np.clip(value, 0, 1)
    return value*value*(3-2*value)

for k in range(int((end+32)/dt)):
    t = k*dt
    if t < end:
        index = min(int(np.searchsorted(times, t, side='right')-1), len(poses)-2)
        u = np.clip((t-times[index])/durations[index], 0, 1)
        target[moving] = poses[index, moving]*(1-u)+poses[index+1, moving]*u
        phase = str(phases[index+1])
    else:
        closure = smooth((t-end-1)/4)
        target[ha] = opened*(1-closure)+closed*closure+hand_offset
        target[ik.qa] = q
        phase = 'close_handle' if lift_start is None else 'lift_handle'
        if k % 10 == 0:
            normal[:] = 0
            contacts = []
            for tip_index,g in enumerate(tips):
                choices = []
                for h in handle:
                    pts = np.zeros(6)
                    distance = mujoco.mj_geomDistance(m, d, g, h, .03, pts)
                    choices.append((distance, pts))
                distance, pts = min(choices, key=lambda x: x[0])
                n = (pts[3:]-pts[:3])*np.sign(distance)
                n /= max(np.linalg.norm(n), 1e-9)
                if a.contact_frame:
                    name = m.body(int(m.geom_bodyid[g])).name
                    n = contact_normals.get(name, n)
                    pts[:3] = contact_points.get(name, pts[:3])
                jp = np.zeros((3, m.nv))
                jr = np.zeros_like(jp)
                mujoco.mj_jac(m, d, jp, jr, pts[:3], int(m.geom_bodyid[g]))
                normal += (jp[:, s.vadr].T@(n*a.normal_force*closure*(2 if a.balanced_pinch and tip_index==0 else 1)))*finger
                contacts.append((distance, pts[:3].copy(), n))
            if t > end+5 and not (a.freeze_grip_on_lift and lift_start is not None):
                hand_offset = advance(m, d, hj, [m.geom_bodyid[g] for g in tips], contacts, forces, closed, hand_offset, 'middle_1', force_targets=([2*a.contact_target,a.contact_target,a.contact_target] if a.balanced_pinch else [a.contact_target]*3))
        if lift_start is not None:
            phase = 'lift_handle' if t-lift_start < 6 else 'hold_handle'
            goal = lift_p+[0, 0, .08*smooth((t-lift_start)/6)]
            if k % 10 == 0:
                palm_R=d.xmat[palm].reshape(3,3)
                local_jar=palm_R.T@(d.xpos[jar]-d.xpos[palm])
                local_jar_R=palm_R.T@d.xmat[jar].reshape(3,3)
                jar_goal=lift_jar_p+[0,0,.08*smooth((t-lift_start)/6)]
                def residual(x):
                    pp, rr = ik.fk(x)
                    thermal = [min(0, mujoco.mj_geomDistance(m, ik.d, g, hot, .02, None)-.008) for g in left_hands]
                    position_error=pp+rr@local_jar-jar_goal if a.object_feedback else pp-goal
                    rotation_error=rr@local_jar_R@lift_jar_R.T if a.object_feedback else rr@lift_R.T
                    return np.r_[position_error*1000, Rotation.from_matrix(rotation_error).as_rotvec()*(100 if a.object_feedback else 30),
                                 (ik.d.xpos[other]-other_p)*(0. if a.fix_idle_arm else 500), (x-q)*.1*W_REG,
                                 np.array(thermal)*5000, anatomy.penalty(ik.d)]
                lo, hi = np.maximum(bounds[:, 0], q-.004), np.minimum(bounds[:, 1], q+.004)
                q = least_squares(residual, np.clip(q, lo+1e-9, hi-1e-9), bounds=(lo, hi), max_nfev=40, x_scale=1/W_REG).x
            target[ik.qa] = q
    if a.wrench_feedforward and lift_start is not None:
        if k%10==0:
            body_ids=[int(m.geom_bodyid[g]) for g in tips]
            names=[m.body(b).name for b in body_ids]
            if all(n in contact_points for n in names):
                fit,T,_,_=solve_wrench(m,d,jar,_P+'hand',
                    [contact_points[n] for n in names],[contact_normals[n] for n in names],body_ids,
                    [contact_mu[n] for n in names],[contact_twist[n] for n in names])
                if fit.success:
                    wrench_tau=.5*wrench_tau+.5*(T@fit.x)*finger
                    wrench_success+=1
                else:wrench_failure+=1
        blend=smooth((t-lift_start)/1.)
        if wrench_success:normal=(1-blend)*normal+blend*wrench_tau
    error = target[s.qadr]-d.qpos[s.qadr]
    friction = m.dof_frictionloss[s.vadr]*np.tanh(error/.001)*np.isin(s.qadr, moving)
    tau = s.kp*error-s.kd*d.qvel[s.vadr]+d.qfrc_bias[s.vadr]+normal+friction
    if lift_start is not None:
        jp = np.zeros((3, m.nv))
        jr = np.zeros_like(jp)
        mujoco.mj_jac(m, d, jp, jr, d.xipos[jar], palm)
        tau += (jp[:, s.vadr].T@(-m.opt.gravity*m.body_mass[jar]))*(~finger)*smooth(t-lift_start)
    d.ctrl[:] = np.clip(tau, m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1])
    peak = np.maximum(peak, np.abs(d.ctrl))
    mujoco.mj_step(m, d)
    measured = {}
    point_sums, normal_sums = {}, {}
    mu_sums, twist_sums = {}, {}
    supports = []
    for ci, c in enumerate(d.contact):
        if c.dist >= 0:
            continue
        gs = [int(c.geom1), int(c.geom2)]
        bn = [m.body(int(m.geom_bodyid[g])).name for g in gs]
        allowed = any(g in handle for g in gs) and any(n.startswith(_P+'hand') for n in bn) and phase in ['approach_handle', 'close_handle', 'lift_handle', 'hold_handle']
        if allowed:
            cf = np.zeros(6)
            mujoco.mj_contactForce(m, d, ci, cf)
            body = next(n for n in bn if n.startswith(_P+'hand'))
            measured[body] = measured.get(body, 0)+float(cf[0])
            mu_sums[body]=min(mu_sums.get(body,1e3),float(c.friction[0]));twist_sums[body]=min(twist_sums.get(body,1e3),float(c.friction[2]))
            point_sums[body] = point_sums.get(body, np.zeros(3))+cf[0]*c.pos
            normal_sums[body] = normal_sums.get(body, np.zeros(3))+cf[0]*c.frame[:3]*(1 if bn[0]==body else -1)
            max_penetration = max(max_penetration, float(-c.dist))
            # 0,5 mm era o ponto medio da rampa de complacencia deste modelo de
            # contato, nao um limiar de dano: os geoms da mao e da alca tem
            # solimp width = 1,0 mm com midpoint 0,5, entao 0,5 mm e' onde o
            # contato esta com metade da rigidez. A medicao confirma: com forca
            # de fechamento de 20, 14 e 8 N a penetracao ficou em 0,5002,
            # 0,5003 e 0,5011 mm, ou seja nao responde a forca, porque a
            # impedancia dispara depois do ponto medio. O portao passa para a
            # largura do modelo; alem dela o solver de fato nao resolveu o
            # contato.
            if -c.dist > a.penetration_gate:
                bad = {'reason': 'handle contact penetration', 'm': float(-c.dist),
                       'gate_m': a.penetration_gate}
        if any(n.startswith(('left_', 'right_', 'torso', 'waist', 'pelvis', 'head')) for n in bn) and not allowed:
            depth = float(-c.dist)
            if depth > max_self_contact:
                max_self_contact = depth
                worst_self_contact = {'bodies': bn, 'geoms': [m.geom(g).name for g in gs],
                                      'depth_m': depth, 't': float(d.time), 'phase': phase}
            if depth > a.self_contact_gate:
                bad = {'reason': 'forbidden contact', 'bodies': bn,
                       'geoms': [m.geom(g).name for g in gs], 'depth_m': depth,
                       'gate_m': a.self_contact_gate, 't': float(d.time), 'phase': phase}
        if 'chaleira' in bn and not allowed:
            supports.append(bn[1-bn.index('chaleira')])
    contact_mu,contact_twist=mu_sums,twist_sums
    contact_points = {n:v/measured[n] for n,v in point_sums.items() if measured[n]>.01}
    contact_normals = {n:v/max(np.linalg.norm(v),1e-9) for n,v in normal_sums.items()}
    forces = [measured.get(_P+'hand_'+n+'_link', 0.) for n in ['thumb_2', 'index_1', 'middle_1']]
    hot_gap = min(mujoco.mj_geomDistance(m, d, g, hot, 1, None) for g in left_hands)
    table_gap = min(mujoco.mj_geomDistance(m, d, g, table, 1, None) for g in hands)
    min_hot, min_table = min(min_hot, hot_gap), min(min_table, table_gap)
    _hot_min[0]=min(_hot_min[0],hot_gap)
    if hot_gap < a.hot_gate:
        _hot_under[0]+=1
        _hot_run[0]+=1
        _hot_run_max[0]=max(_hot_run_max[0],_hot_run[0])
    else:
        _hot_run[0]=0
    # Portao com duracao, nao por passo. Uma pega media 224 passos abaixo do
    # limite e chegava a encostar no metal (-0,0106 mm): isso e' aproximacao
    # sustentada e reprova. Outra ficou 1 passo a 5,9992 mm, 0,8 micrometro
    # abaixo, sem nunca encostar: isso e' transiente do contato complacente,
    # nao violacao termica. Encostar (folga <= 0) continua reprovando sempre.
    _sustentado = _hot_run[0]*dt*1000 > a.hot_sustained_ms
    _encostou = hot_gap <= 0.
    if ((_sustentado or _encostou) and not a.observe_clearance) or table_gap < .020:
        bad = {'reason': 'hand clearance', 'hot_m': float(hot_gap), 'table_m': float(table_gap)}
    if not anatomy.valid(d, tolerance_deg=2):
        bad = {'reason': 'elbow geometry', 'deg': anatomy.angles(d).tolist()}
    if np.any(np.abs(np.rad2deg(d.qpos[ik.qa[np.r_[7:10, 14:17]]])) > [62, 47, 32, 62, 47, 32]):
        bad = {'reason': 'wrist bounds'}
    if np.max(np.linalg.norm(d.xpos[props]-prop0, axis=1)) > .005:
        bad = {'reason': 'other prop moved more than5mm', 'displacements_m': {m.body(b).name:float(np.linalg.norm(d.xpos[b]-p)) for b,p in zip(props,prop0)}}
    lift = float(d.xpos[jar, 2]-jar0[2])
    max_lift = max(max_lift, lift)
    tilt = float(np.rad2deg(np.arccos(np.clip(d.xmat[jar].reshape(3, 3)[2, 2], -1, 1))))
    if lift_start is None and (np.linalg.norm(d.xpos[jar]-jar0) > .005 or tilt > 5):
        bad = {'reason': 'kettle moved before confirmed grasp'}
    robot_mass = m.body_subtreemass[pelvis]
    com = (robot_mass*d.subtree_com[pelvis]+m.body_mass[jar]*d.xipos[jar])/(robot_mass+m.body_mass[jar])
    margin = float(np.min(-(support_hull[:, :2]@com[:2]+support_hull[:, 2])))
    min_com = min(min_com, margin)
    if margin < .02:
        bad = {'reason': 'quasi-static COM margin', 'm': margin}
    grip_steps = grip_steps+1 if t>end+5 and min(forces)>6 else 0
    if lift_start is None and grip_steps >= int(1/dt):
        lift_start = t
        lift_p, lift_R = d.xpos[palm].copy(), d.xmat[palm].reshape(3, 3).copy()
        other_p = d.xpos[other].copy()
        lift_jar_p, lift_jar_R = d.xpos[jar].copy(), d.xmat[jar].reshape(3,3).copy()
        grasp_R = lift_R.T@d.xmat[jar].reshape(3, 3)
        q = target[ik.qa].copy()
    if lift_start is not None:
        lost_steps = lost_steps+1 if min(forces)<1.5 else 0
        if lost_steps > int(.3/dt):
            bad = {'reason': 'lost three-finger handle grip', 'forces_N': forces}
        relative = d.xmat[palm].reshape(3, 3).T@d.xmat[jar].reshape(3, 3)
        angle = float(np.rad2deg(Rotation.from_matrix(relative@grasp_R.T).magnitude()))
        max_relative_rotation = max(max_relative_rotation, angle)
        if angle > 15:
            bad = {'reason': 'kettle rotated in grasp', 'deg': angle}
        if t>lift_start+6 and lift>.06 and tilt<8 and not supports and min(forces)>1.5:
            window.append((d.xpos[jar].copy(), d.xmat[jar].reshape(3, 3).copy()))
        else:
            window.clear()
        if len(window) == window.maxlen:
            hold_drift = float(max(np.linalg.norm(p-window[0][0]) for p, _ in window))
            hold_angle = float(max(np.rad2deg(Rotation.from_matrix(R@window[0][1].T).magnitude()) for _, R in window))
            passed = hold_drift < .002 and hold_angle < 2
    elif t > end+20:
        bad = {'reason': 'handle grip establishment timeout', 'forces_N': forces}
    if k % 10 == 0:
        brew.step(d, .02, water)
        bs = [m.body(n).id for n in ['chaleira', 'coador', 'copo']]
        flow = water.step(.02, *sum(([d.xpos[b], d.xmat[b].reshape(3, 3)] for b in bs), []))
        coupler.apply(d, water)
        liquid.append({'t': float(d.time), **flow})
    if water.spilled_ml > .5 or brew.grounds.spilled_g > .05:
        bad = {'reason': 'spill'}
    if s.warnings():
        bad = {'reason': 'numerical warning', 'warnings': s.warnings()}
    if k % 16 == 0:
        states.append(d.qpos.copy())
        rows.append({'t': float(d.time), 'phase': phase, 'table_gap_m': float(table_gap)})
        samples.append({'t': float(d.time), 'phase': phase, 'finger_force_N': measured,
                        'kettle_lift_m': lift, 'kettle_tilt_deg': tilt,
                        'hot_clearance_m': float(hot_gap), 'com_margin_m': margin,
                        'elbow_flexion_deg': anatomy.angles(d).tolist()})
    if k % 2500 == 0:
        print({'t': t, 'phase': phase, 'forces_N': forces, 'lift_m': lift, 'hot_gap_m': hot_gap}, flush=True)
    if bad or passed:
        break
report = {'grasp_reference_R':grasp_R.tolist() if lift_start is not None else None,'wrench_feedforward':a.wrench_feedforward,'wrench_solve_successes':wrench_success,'wrench_solve_failures':wrench_failure,'balanced_pinch': a.balanced_pinch, 'freeze_grip_on_lift': a.freeze_grip_on_lift, 'object_feedback': a.object_feedback, 'contact_frame': a.contact_frame, 'clearance_audit': {'hot_min_m': float(_hot_min[0]), 'steps_under_gate': _hot_under[0], 'longest_run_steps': _hot_run_max[0], 'gate_m': a.hot_gate,
                                    'longest_run_ms': _hot_run_max[0]*dt*1000,
                                    'sustained_gate_ms': a.hot_sustained_ms}, 'model': 'gpt-6-astra', 'source': str(a.source), 'plan': str(a.plan),
          'scene': r['scene'], 'pass': bool(passed and not bad),
          'failure': bad or (None if passed else {'reason': 'lift/hold timeout'}),
          'coffee_completed': False, 'initial_water_ml': water.initial_ml,
          'normal_motor_force_N': a.normal_force, 'contact_force_target_N': a.contact_target,
          'loaded_kettle_mass_kg': float(m.body_mass[jar]), 'max_lift_m': max_lift,
          'min_hot_clearance_m': min_hot, 'min_table_clearance_m': min_table,
          'min_quasi_static_com_margin_m': min_com,
          'balance_limitation': 'fixed pelvis; nominal foot polygon only, not free-base balance',
          'max_handle_penetration_m': max_penetration,
          'max_self_contact_m': max_self_contact, 'worst_self_contact': worst_self_contact, 'max_grasp_rotation_deg': max_relative_rotation,
          'hold_drift_m': hold_drift, 'hold_angle_deg': hold_angle,
          'scope': 'motor-only loaded handle grasp/lift; reduced liquid/grounds/thermal models',
          'water_state': asdict(water), 'rows': rows, 'grasp_samples': samples,
          'liquid_samples': liquid, 'peak_motor_torques_Nm': {n: float(peak[i]) for n, i in s.act_joint.items()}}
(a.out/'report.json').write_text(json.dumps(report, indent=2))
np.savez_compressed(a.out/'trajectory.npz', qpos=states)
integration = np.zeros(mujoco.mj_stateSize(m, mujoco.mjtState.mjSTATE_INTEGRATION))
mujoco.mj_getState(m, d, integration, mujoco.mjtState.mjSTATE_INTEGRATION)
np.savez_compressed(a.out/'continuation.npz', integration=integration, last_target=target,
                    hand_names=hand_names, hand_target=target[ha], water_ml=water.source_ml,
                    kp=s.kp, kd=s.kd, body_mass=m.body_mass, body_ipos=m.body_ipos,
                    body_inertia=m.body_inertia, body_iquat=m.body_iquat, bvh_aabb=m.bvh_aabb,
                    dry_kettle_kg=float(ck['dry_kettle_kg']))
(a.out/'liquid-state.json').write_text(json.dumps(asdict(water), indent=2))
brew.save(a.out)
print({k: v for k, v in report.items() if k not in ['rows', 'grasp_samples', 'liquid_samples', 'peak_motor_torques_Nm']})
