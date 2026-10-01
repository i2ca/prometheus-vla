"""Checkpointed dosing with explicit alternative plans and physical validation.

Each rejected trial is retained. A retry starts from the last accepted physical
checkpoint; failed state or transferred mass is never merged into the episode.
Create OUT/STOP to stop after the currently running stage finishes.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

ap = argparse.ArgumentParser()
ap.add_argument('--source', type=Path, required=True)
ap.add_argument('--out', type=Path, required=True)
ap.add_argument('--target-g', type=float, default=20)
ap.add_argument('--max-cycles', type=int, default=180)
ap.add_argument('--segment-seconds', type=float, default=4)
ap.add_argument('--preferred-profile', type=int, default=0,
                help='Try this explicit collection profile first; preserve alternatives')
a = ap.parse_args()
a.out.mkdir(exist_ok=False)
shutil.copy2(__file__, a.out/Path(__file__).name)
source = a.source
attempts = []
failure = None
stopped = False
profiles = [
    {'dip_angle': 25, 'approach_angle': 15, 'azimuth': -90, 'offset': -.03, 'immersion': .003, 'waist': 18},
    {'dip_angle': 25, 'approach_angle': 15, 'azimuth': -90, 'offset': -.03, 'immersion': .002, 'waist': 18},
    {'dip_angle': 25, 'approach_angle': 15, 'azimuth': -60, 'offset': -.03, 'immersion': .003, 'waist': 18},
    {'dip_angle': 30, 'approach_angle': 15, 'azimuth': -90, 'offset': -.032, 'immersion': .003, 'waist': 18},
    {'dip_angle': 30, 'approach_angle': 15, 'offset': -.02, 'immersion': .003, 'waist': 15},
    {'dip_angle': 30, 'approach_angle': 15, 'offset': -.02, 'immersion': .002, 'waist': 15},
    {'dip_angle': 35, 'approach_angle': 15, 'offset': -.02, 'immersion': .003, 'waist': 15},
    {'dip_angle': 30, 'approach_angle': 15, 'offset': -.025, 'immersion': .003, 'waist': 18},
    {'dip_angle': 35, 'approach_angle': 15, 'offset': -.025, 'immersion': .0025, 'waist': 18},
    {'dip_angle': 35, 'approach_angle': 15, 'offset': 0, 'immersion': .002, 'waist': 18},
]
assert 0 <= a.preferred_profile < len(profiles)
profiles.insert(0, profiles.pop(a.preferred_profile))
(a.out/'profiles.json').write_text(json.dumps(profiles, indent=2))


def grounds():
    return json.loads((source/'brew-state.json').read_text())['grounds']


def publish():
    g = grounds()
    status = {'model': 'gpt-6-astra', 'source': str(a.source),
              'latest_accepted_source': str(source), 'target_g': a.target_g,
              'grounds': g, 'failure': failure, 'stopped': stopped,
              'attempts': attempts,
              'dose_completed': a.target_g-.2 <= g['filter_g'] <= a.target_g+.5,
              'coffee_completed': False}
    temporary = a.out/'progress.tmp'
    temporary.write_text(json.dumps(status, indent=2))
    temporary.replace(a.out/'progress.json')
    return status


def run(name, arguments):
    path = a.out/name
    with (a.out/(name+'.log')).open('w') as log:
        exit_code = subprocess.run([sys.executable, *arguments, '--out', str(path)], stdout=log, stderr=subprocess.STDOUT).returncode
    if exit_code or not (path/'report.json').exists():
        return path, {'pass': False, 'failure': {'process_exit': exit_code}}
    return path, json.loads((path/'report.json').read_text())


publish()
for cycle in range(1, a.max_cycles+1):
    if publish()['dose_completed']:
        break
    tasks = ['dose'] if grounds()['spoon_g'] > .01 else ['collect', 'dose']
    for task in tasks:
        if (a.out/'STOP').exists():
            stopped = True
            break
        accepted = False
        variants = profiles if task == 'collect' else [-90, -135, -60, 180, 0]
        for index, variant in enumerate(variants):
            if (a.out/'STOP').exists():
                stopped = True
                break
            prefix = f'cycle-{cycle:03d}-{task}-v{index:02d}'
            planner = 'plan_spoon_dip.py' if task == 'collect' else 'plan_spoon_dose.py'
            arguments = ['scripts/'+planner, '--source', str(source)]
            if task == 'collect':
                arguments += ['--return-route', '--dip-angle', str(variant['dip_angle']),
                              '--approach-angle', str(variant['approach_angle']),
                              '--dip-azimuth',str(variant.get('azimuth',0)),
                              '--scoop-offset-x', str(variant['offset']),
                              '--immersion', str(variant['immersion']),
                              '--waist-pitch-max', str(max(variant['waist'],json.loads((source/'report.json').read_text()).get('waist_pitch_max_deg',12)))]
            else:
                arguments += ['--tip-azimuth', str(variant)]
            plan, result = run(prefix+'-plan', arguments)
            if not result.get('pass'):
                attempts.append({'stage': str(plan), 'source': str(source), 'kind': 'plan', 'pass': False})
                publish()
                continue
            destination, result = run(prefix, [
                'scripts/execute_spoon_transport.py', '--source', str(source),
                '--plan', str(plan), '--segment-seconds', str(a.segment_seconds),
                '--allow-pot-contact'])
            row = {'stage': str(destination), 'source': str(source), 'kind': 'physical',
                   'pass': result.get('pass', False), 'failure': result.get('failure')}
            attempts.append(row)
            print(row, flush=True)
            if result.get('pass'):
                source = destination
                accepted = True
                publish()
                break
            publish()
        if stopped:
            break
        if not accepted:
            failure = {'cycle': cycle, 'task': task, 'reason': 'all candidate trials rejected; last accepted checkpoint retained'}
            break
        if grounds()['spilled_g'] > .05:
            failure = {'reason': 'powder spill'}
            break
        if task == 'collect' and grounds()['spoon_g'] < .05:
            failure = {'reason': 'insufficient collected powder', 'stage': str(source)}
            break
    if failure or stopped:
        break
status = publish()
(a.out/'report.json').write_text(json.dumps(status, indent=2))
print({k: v for k, v in status.items() if k != 'attempts'}, flush=True)
