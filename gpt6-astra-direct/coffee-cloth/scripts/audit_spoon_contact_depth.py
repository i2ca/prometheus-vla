"""Replay geometry only to audit penetration in every saved trajectory frame.

This does not reconstruct contact forces or claim per-integration-step coverage.
"""
import argparse
import json
import shutil
from pathlib import Path

import mujoco
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--source', type=Path, required=True)
ap.add_argument('--out', type=Path, required=True)
args = ap.parse_args()
args.out.mkdir(exist_ok=False)
shutil.copy2(__file__, args.out / Path(__file__).name)
report = json.loads((args.source / 'report.json').read_text())
m = mujoco.MjModel.from_xml_path(report['scene'])
d = mujoco.MjData(m)
spoon = m.body('scoop').id
trajectory = np.load(args.source / 'trajectory.npz')['qpos']
depths = []
worst = None
for frame, qpos in enumerate(trajectory):
    d.qpos[:] = qpos
    mujoco.mj_forward(m, d)
    for c in d.contact:
        bodies = m.geom_bodyid[[c.geom1, c.geom2]]
        if spoon not in bodies or c.dist >= 0:
            continue
        other = int(bodies[0] if bodies[1] == spoon else bodies[1])
        if not m.body(other).name.startswith(('left_hand', 'right_hand')):
            continue
        depth = float(-c.dist)
        depths.append(depth)
        if worst is None or depth > worst['penetration_m']:
            worst = {'frame': frame, 'penetration_m': depth,
                     'other_body': m.body(other).name}
result = {'model': 'gpt-6-astra', 'source': str(args.source),
          'frames': len(trajectory), 'contact_samples': len(depths),
          'limit_m': .0002, 'worst': worst,
          'pass': bool(depths and max(depths) <= .0002),
          'scope': 'geometry replay at recorded frames; no force reconstruction; '
                   'not coverage of unrecorded integration steps'}
(args.out / 'report.json').write_text(json.dumps(result, indent=2))
print(result)
