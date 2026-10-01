"""Checkpointed main-episode continuation after accepted dosing.

Every command gets a new result directory. Failed trials remain on disk and
never supply state to the next stage. No windows, hardware, or external APIs.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ap = argparse.ArgumentParser()
inputs = ap.add_mutually_exclusive_group(required=True)
inputs.add_argument('--dosing', type=Path)
inputs.add_argument('--released-source', type=Path)
ap.add_argument('--out', type=Path, required=True)
a = ap.parse_args()
a.out.mkdir(exist_ok=False)
shutil.copy2(__file__, a.out/Path(__file__).name)
status = {'model': 'gpt-6-astra', 'dosing': str(a.dosing), 'phase': 'waiting_for_dosing',
          'released_source': str(a.released_source) if a.released_source else None, 'latest_accepted_source': None, 'commands': [], 'coffee_completed': False}


def publish():
    tmp = a.out/'progress.tmp'
    tmp.write_text(json.dumps(status, indent=2))
    tmp.replace(a.out/'progress.json')


def run(name, script, *args, candidates=False):
    destination = a.out/name
    command = [sys.executable, 'scripts/'+script+'.py', *map(str, args), '--out', str(destination)]
    status['phase'] = name
    status['commands'].append(command)
    publish()
    with (a.out/(name+'.log')).open('w') as log:
        code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT).returncode
    if code:
        raise RuntimeError(f'{name}: process exit {code}; inspect log')
    report = json.loads((destination/'report.json').read_text())
    if not candidates and not report.get('pass'):
        raise RuntimeError(f'{name}: rejected: {report.get("failure")}')
    return destination, report


def accept(source):
    status['latest_accepted_source'] = str(source)
    publish()


publish()
try:
    if a.released_source:
        source = a.released_source
        released = json.loads((source/'report.json').read_text())
        assert released['pass'] and released.get('withdraw_at_elapsed_s') is not None
        accept(source)
    else:
        while True:
            if (a.out/'STOP').exists():
                raise RuntimeError('Explicit STOP requested')
            dosing = json.loads((a.dosing/'progress.json').read_text())
            if dosing['dose_completed']:
                source = Path(dosing['latest_accepted_source'])
                break
            if dosing.get('failure') or dosing.get('stopped'):
                raise RuntimeError('Dosing stopped before recipe target')
            time.sleep(5)
        accept(source)
        plan, _ = run('spoon-place-plan', 'plan_spoon_place', '--source', source)
        source, _ = run('spoon-place', 'execute_spoon_transport', '--source', source, '--plan', plan,
                        '--segment-seconds', 6)
        accept(source)
        source, _ = run('spoon-release', 'execute_spoon_release', '--source', source)
        accept(source)
    g = json.loads((source/'brew-state.json').read_text())['grounds']
    assert 19.8 <= g['filter_g'] <= 20.5 and g['spoon_g'] < .01 and g['spilled_g'] <= .05
    candidate, report = run('heater-reach', 'plan_heater_reach', '--source', source,
                           '--free-orientation', '--elbow-forward', .02, '--elbow-drop', .02,
                           candidates=True)
    choices = [i for i, r in enumerate(report['results']) if r['pass']]
    assert choices, 'No valid heater reach'
    plan, _ = run('heater-connect', 'plan_heater_connection', '--source', source,
                  '--candidate', candidate, '--choice', choices[0], '--direct-start')
    source, _ = run('heater', 'execute_heater_press', '--source', source, '--plan', plan,
                    '--boil', '--hold-joints-during-heating')
    accept(source)
    candidate, report = run('kettle-grasp', 'plan_kettle_grasp', '--source', source, candidates=True)
    choices = [i for i, r in enumerate(report['results']) if r['pass']]
    assert choices, 'No valid kettle grasp'
    # A valid geometric candidate still needs a collision-free connection and
    # a physically accepted lift. Inspect any failure before choosing another.
    plan, _ = run('kettle-connect', 'plan_kettle_connection', '--source', source,
                  '--candidate', candidate, '--choice', choices[0], '--direct-start',
                  '--open-factor', .6, '--escape', 0, 0, 1)
    source, _ = run('kettle-lift', 'execute_kettle_lift', '--source', source, '--plan', plan,
                    '--wrench-feedforward', '--freeze-grip-on-lift', '--object-feedback',
                    '--contact-frame', '--normal-force', 20, '--contact-target', 10)
    accept(source)
    plan, _ = run('pour-plan', 'plan_kettle_pour', '--source', source)
    source, _ = run('pour', 'execute_coffee_pour', '--source', source, '--plan', plan,
                    '--require-hot', '--duration', 20, '--max-seconds', 320,
                    '--return-offset-x', 0, '--return-lift', 0, '--return-rate', .4)
    accept(source)
    source, _ = run('kettle-place', 'execute_coffee_place', '--source', source,
                    '--freeze-torso-release', '--align-seconds', 2, '--lower-seconds', 4)
    accept(source)
    run('chain-audit', 'audit_episode_chain', '--source', source)
    status['phase'] = 'physical_sequence_complete_pending_final_review'
except Exception as exc:
    status['failure'] = str(exc)
    status['phase'] = 'stopped_for_inspection'
finally:
    publish()
    print(json.dumps(status, indent=2), flush=True)
