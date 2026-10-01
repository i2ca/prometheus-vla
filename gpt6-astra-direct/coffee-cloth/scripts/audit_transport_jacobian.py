"""Compare analytic IK derivatives against central finite differences."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import time

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'g1-cup-grasp/scripts'))
from g1_sim import G1Sim, ARM_JOINTS
from kinematics import ArmIK
from elbow_anatomy import ElbowAnatomy
from transport_ik_jacobian import make_jacobian

ap = argparse.ArgumentParser()
ap.add_argument('--source', type=Path, required=True)
ap.add_argument('--out', type=Path, required=True)
a = ap.parse_args()
a.out.mkdir(exist_ok=False)
for p in [Path(__file__), Path('scripts/transport_ik_jacobian.py')]:
    shutil.copy2(p, a.out/p.name)
r = json.loads((a.source/'report.json').read_text())
s = G1Sim(r['scene'])
m, d = s.m, s.d
ck = np.load(a.source/'continuation.npz')
mujoco.mj_setState(m, d, ck['integration'], mujoco.mjtState.mjSTATE_INTEGRATION)
mujoco.mj_forward(m, d)
names = ['waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint'] + list(ARM_JOINTS) + [n.replace('right_', 'left_') for n in ARM_JOINTS]
ik = ArmIK(s, names)
q = d.qpos[ik.qa].copy()
spoon = m.body('scoop').id
other = m.body('left_wrist_yaw_link').id
pp, rr = ik.fk(q)
local = rr.T@(d.xpos[spoon]+d.xmat[spoon].reshape(3, 3)@[-.045, 0, .003]-pp)
local_R = rr.T@d.xmat[spoon].reshape(3, 3)
goal = pp+rr@local+[.001, -.001, .001]
target_R = Rotation.from_rotvec([.07, -.04, .02]).as_matrix()@rr@local_R
other_goal = d.xpos[other].copy()
tip = next(g for g in range(m.ngeom) if m.geom_contype[g] and m.body(int(m.geom_bodyid[g])).name == 'right_hand_index_1_link')
table = m.geom('tampo').id
# Active gap derivative on a real hand/table pair, including a tilted pose.
gap = mujoco.mj_geomDistance(m, d, tip, table, 1, None)
avoid = [(tip, table, gap+.002)] if gap < .028 else []
results = []
for active_anatomy in [False, True]:
    anatomy = ElbowAnatomy(m, minimum_deg=90 if active_anatomy else 5)
    def residual(x):
        p, R = ik.fk(x)
        gaps = [min(0, mujoco.mj_geomDistance(m, ik.d, g, h, .03, None)-margin) for g, h, margin in avoid]
        return np.r_[(p+R@local-goal)*1000,
                     Rotation.from_matrix(R@local_R@target_R.T).as_rotvec()*35,
                     (ik.d.xpos[other]-other_goal)*100, (x-q)*2, (x-q)*.2,
                     np.array(gaps)*5000, anatomy.penalty(ik.d)]
    jac = make_jacobian(ik, local, local_R, target_R, other, 35, avoid, anatomy)
    analytic = jac(q)
    eps = 1e-6
    finite = np.column_stack([(residual(q+np.eye(len(q))[i]*eps)-residual(q-np.eye(len(q))[i]*eps))/(2*eps) for i in range(len(q))])
    absolute_error = float(np.max(np.abs(analytic-finite)))
    bounds = np.column_stack([np.maximum(ik.bounds[:, 0], q-.006), np.minimum(ik.bounds[:, 1], q+.006)])
    fits = {}
    for name, derivative in [('finite', '2-point'), ('analytic', jac)]:
        before = time.perf_counter()
        fit = least_squares(residual, q, jac=derivative, bounds=bounds.T, max_nfev=50)
        fits[name] = {'seconds': time.perf_counter()-before, 'cost': float(fit.cost), 'q': fit.x.tolist()}
    results.append({'active_anatomy': active_anatomy, 'active_gap_count': len(avoid),
                    'max_absolute_derivative_error': absolute_error, 'fits': fits,
                    'pass': absolute_error < .01 and fits['analytic']['cost'] <= fits['finite']['cost']*1.001+1e-6})
result = {'model': 'gpt-6-astra', 'source': str(a.source), 'pass': all(x['pass'] for x in results), 'results': results}
(a.out/'report.json').write_text(json.dumps(result, indent=2))
print(result)
