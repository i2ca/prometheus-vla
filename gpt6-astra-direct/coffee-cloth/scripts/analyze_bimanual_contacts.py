"""Offline contact audit; never changes a simulation or overwrites evidence."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('trial', type=Path)
args = parser.parse_args()
out = args.trial / 'contact-audit'
out.mkdir(exist_ok=False)
rows = json.loads((args.trial / 'trajetoria.json').read_text())
params = json.loads((args.trial / 'parameters.json').read_text())
axis = np.array([np.cos(np.deg2rad(params['eixo_deg'])), np.sin(np.deg2rad(params['eixo_deg'])), 0])
t = np.array([r['t'] for r in rows])
forces = {s: np.array([r['hand_force_on_object_N'][s] for r in rows]) for s in ('right', 'left')}
supports = np.array([sum((np.array(c['force_on_object_world_N']) for c in r['object_contact_details']
                         if not c['other_body'].startswith(('right_hand','left_hand','right_wrist','left_wrist'))),
                        start=np.zeros(3)) for r in rows])
report = {'model': 'gpt-6-astra', 'trial': str(args.trial), 'phase_endpoints': {},
          'force_convention_source': 'https://mujoco.readthedocs.io/en/stable/computation/index.html',
          'limitation': '30Hz sampled contacts; forbidden-contact acceptance uses controller substep counters separately.'}
for phase in dict.fromkeys(r['fase'] for r in rows):
    indices = [i for i,r in enumerate(rows) if r['fase'] == phase]
    report['phase_endpoints'][phase] = [{
        'time_s': rows[i]['t'], 'height_change_mm': rows[i]['subiu_mm'],
        'tilt_deg': rows[i]['obj_tilt_deg'],
        'hand_vertical_force_N': float(sum(f[i,2] for f in forces.values())),
        'external_support_vertical_force_N': float(supports[i,2]),
        'compression_N': [float(forces['right'][i]@axis), float(-forces['left'][i]@axis)]
    } for i in (indices[0], indices[-1])]
(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
fig, axs = plt.subplots(4,1,figsize=(12,10),sharex=True)
axs[0].plot(t,[r['subiu_mm'] for r in rows],label='Object origin height change (mm)')
axs[0].plot(t,[r['obj_tilt_deg'] for r in rows],label='Object tilt (degrees)')
axs[1].plot(t,forces['right']@axis,label='Right compression (N)')
axs[1].plot(t,-forces['left']@axis,label='Left compression (N)')
axs[2].plot(t,forces['right'][:,2]+forces['left'][:,2],label='Hands vertical force (N)')
axs[2].plot(t,supports[:,2],label='External supports vertical force (N)')
axs[3].plot(t,np.array([r['grip_correction_m'] for r in rows])*1000)
axs[3].set_ylabel('Grip correction (mm)'); axs[3].set_xlabel('Simulation time (s)')
for ax in axs:
    ax.grid(alpha=.2)
    for i in range(1,len(rows)):
        if rows[i]['fase'] != rows[i-1]['fase']: ax.axvline(t[i],color='gray',alpha=.25)
for ax in axs[:3]: ax.legend(loc='upper left')
fig.suptitle(args.trial.name+' — contact audit (gpt-6-astra)')
fig.tight_layout(); fig.savefig(out/'forces.png',dpi=150); plt.close(fig)
print(json.dumps(report['phase_endpoints'],indent=2))
