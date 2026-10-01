"""Audit saved physical stage ancestry and conserved ledgers, without simulation.

This does not replace per-step collision/force guards or certify hardware.
Trajectory boundary checks allow one sampled integration step, not a reset.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import mujoco
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--source', type=Path, required=True)
ap.add_argument('--out', type=Path, required=True)
a = ap.parse_args()
a.out.mkdir(exist_ok=False)
shutil.copy2(__file__, a.out/Path(__file__).name)
rows, errors, seen = [], [], set()
stage = a.source
while True:
    identity = str(stage.resolve())
    if identity in seen:
        errors.append({'stage': str(stage), 'reason': 'ancestry cycle'})
        break
    seen.add(identity)
    report = json.loads((stage/'report.json').read_text())
    if not report.get('pass') or (stage/'INVALIDATED.json').exists():
        errors.append({'stage': str(stage), 'reason': 'unaccepted stage'})
    row = {'stage': str(stage), 'model': report.get('model'),
           'source': report.get('source'), 'task': report.get('task'),
           'scene': report.get('scene')}
    for filename in ['report.json', 'continuation.npz', 'brew-state.json', 'liquid-state.json']:
        path = stage/filename
        if not path.exists():
            errors.append({'stage': str(stage), 'reason': 'missing '+filename})
        else:
            row[filename+'_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    brew = json.loads((stage/'brew-state.json').read_text())
    water = json.loads((stage/'liquid-state.json').read_text())
    grounds = brew['grounds']
    row['grounds'] = {k: grounds[k] for k in ['pot_g', 'spoon_g', 'filter_g', 'spilled_g']}
    row['water'] = water
    row['heater'] = brew['heater']
    ground_error = abs(sum(row['grounds'].values())-grounds['initial_g'])
    water_error = abs(sum(water[k] for k in ['source_ml', 'filter_ml', 'cloth_retained_ml',
                                           'coffee_retained_ml', 'receiver_ml', 'spilled_ml'])-water['initial_ml'])
    if ground_error > 1e-7 or water_error > 1e-7:
        errors.append({'stage': str(stage), 'reason': 'mass conservation',
                       'grounds_error_g': ground_error, 'water_error_ml': water_error})
    if min(row['grounds'].values()) < -1e-9 or min(water[k] for k in ['source_ml', 'filter_ml', 'receiver_ml', 'spilled_ml']) < -1e-9:
        errors.append({'stage': str(stage), 'reason': 'negative reservoir'})
    for key in ['max_episode_grasp_rotation_deg', 'spoon_penetration_peak_m',
                'min_quasi_static_com_margin_m', 'min_hot_body_gap_m', 'min_hot_gap_m']:
        if key in report:
            row[key] = report[key]
    rows.append(row)
    if not report.get('source'):
        break
    stage = Path(report['source'])
rows.reverse()
# Source scene may change only when an explicitly documented model correction
# was introduced. Record all scene transitions rather than hiding them.
models = {}
for i, row in enumerate(rows):
    scene = row['scene']
    if scene not in models:
        models[scene] = mujoco.MjModel.from_xml_path(scene)
    m = models[scene]
    report = json.loads((Path(row['stage'])/'report.json').read_text())
    caps = {m.joint(m.actuator_trnid[j, 0]).name: float(max(abs(m.actuator_ctrlrange[j])))
            for j in range(m.nu)}
    torques = report.get('peak_motor_torques_Nm', {})
    row['recorded_motor_peak_count'] = len(torques)
    for name, value in torques.items():
        if name not in caps or abs(value) > caps[name]+1e-7:
            errors.append({'stage': row['stage'], 'reason': 'motor cap', 'joint': name,
                           'peak_Nm': value, 'cap_Nm': caps.get(name)})
    if not i:
        continue
    previous = rows[i-1]
    row['scene_changed_from_parent'] = scene != previous['scene']
    ck = np.load(Path(previous['stage'])/'continuation.npz')
    d = mujoco.MjData(m)
    mujoco.mj_setState(m, d, ck['integration'], mujoco.mjtState.mjSTATE_INTEGRATION)
    path = np.load(Path(row['stage'])/'trajectory.npz')['qpos']
    delta = np.abs(path[0]-d.qpos)
    row['first_sample_max_coordinate_delta'] = float(delta.max())
    # Coordinates mix radians, metres, and quaternion components. A 0.01
    # bound is only a gross discontinuity detector; it is not bitwise proof.
    if delta.max() > .01:
        errors.append({'stage': row['stage'], 'reason': 'trajectory boundary discontinuity',
                       'max_coordinate_delta': float(delta.max())})
    for key in ['filter_g', 'spilled_g']:
        if row['grounds'][key]+1e-7 < previous['grounds'][key]:
            errors.append({'stage': row['stage'], 'reason': 'grounds ledger decreased', 'field': key})
    for key in ['discharged_ml', 'captured_ml', 'receiver_ml', 'spilled_ml']:
        if row['water'][key]+1e-7 < previous['water'][key]:
            errors.append({'stage': row['stage'], 'reason': 'water ledger decreased', 'field': key})
result = {'model': 'gpt-6-astra', 'source': str(a.source), 'pass': not errors,
          'scope': 'saved accepted ancestry, endpoint mass ledgers, reported torque peaks, sampled trajectory continuity',
          'limitations': ['No new per-step physics replay', 'Boundary comparison is sampled, not bitwise',
                          'Fixed pelvis and reduced fluid/grounds/heat models', 'Mass/friction assumptions remain uncalibrated'],
          'coffee_completed': False, 'stage_count': len(rows), 'errors': errors, 'stages': rows}
(a.out/'report.json').write_text(json.dumps(result, indent=2))
print(json.dumps({k: v for k, v in result.items() if k != 'stages'}, indent=2))
