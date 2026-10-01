"""Bounded motor-control parameter sweep; never accepts failed checkpoints."""
from pathlib import Path
import json,subprocess,sys,time,shutil
out=Path('results/spoon-right-grip-sweep-001');out.mkdir(exist_ok=False);shutil.copy2(__file__,out/Path(__file__).name);records=[]
# An already running attempt uses the same authorized source and geometry.
p=Path('results/spoon-right-physical-008/report.json')
for _ in range(600):
 if p.exists():break
 time.sleep(1)
if p.exists():
 r=json.loads(p.read_text());records.append({'attempt':str(p.parent),'pass':r['pass']})
for num,force,kp in [(9,.45,4),(10,.6,4),(11,.3,6),(12,.45,6),(13,.6,6)]:
 if records and records[-1]['pass']:break
 dest=Path(f'results/spoon-right-physical-{num:03d}')
 if dest.exists():raise RuntimeError('attempt directory already exists; never overwrite')
 cmd=[sys.executable,'scripts/execute_spoon_physical.py','--hand','right','--out',str(dest),'--source','results/lid-place-005','--plan','results/spoon-right-connect-003','--normal-pinch','--index-servo','--adaptive-grip','--object-feedback','--pinch-force',str(force),'--finger-kp',str(kp),'--damped-arms','--allow-tip-contact']
 with (out/f'attempt-{num:03d}.log').open('w') as log:status=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT).returncode
 report=json.loads((dest/'report.json').read_text()) if (dest/'report.json').exists() else {'pass':False,'failure':'process failed'};row={'attempt':str(dest),'force_N':force,'kp':kp,'exit_code':status,'pass':report['pass'],'failure':report.get('failure')};records.append(row);print(row,flush=True)
 (out/'progress.json').write_text(json.dumps(records,indent=2))
(out/'report.json').write_text(json.dumps({'model':'gpt-6-astra','scope':'bounded right spoon lift gain sweep, all collision/torque/hold gates unchanged','coffee_completed':False,'accepted':next((r['attempt'] for r in records if r['pass']),None),'attempts':records},indent=2))
