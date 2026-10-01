"""Repeat measured motor-driven scoop cycles, stopping on any failed stage.

No state reset, no synthetic mass increment, no continuation of failed attempts.
Every plan/controller invocation gets a new directory and exact source checkpoint.
"""
import argparse,json,subprocess,shutil,sys
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--tip-azimuth',type=float,default=-90);ap.add_argument('--source',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--dip-angle',type=float,default=25);ap.add_argument('--scoop-offset-x',type=float,default=0);ap.add_argument('--immersion',type=float,default=.0055);ap.add_argument('--target-g',type=float,default=20);ap.add_argument('--max-cycles',type=int,default=160);ap.add_argument('--segment-seconds',type=float,default=8);ap.add_argument('--joint-speed',type=float,default=.14);a=ap.parse_args();a.out.mkdir(exist_ok=False);shutil.copy2(__file__,a.out/Path(__file__).name);source=a.source;records=[];failure=None

def run(name,args):
 path=a.out/name
 with (a.out/(name+'.log')).open('w') as log:ret=subprocess.run([sys.executable,*args,'--out',str(path)],stdout=log,stderr=subprocess.STDOUT).returncode
 if ret!=0 or not (path/'report.json').exists():return path,{'pass':False,'failure':f'process exit{ret}'}
 return path,json.loads((path/'report.json').read_text())

def publish():
 g=json.loads((source/'brew-state.json').read_text())['grounds'];status={'model':'gpt-6-astra','source':str(a.source),'latest_accepted_source':str(source),'target_g':a.target_g,'grounds':g,'failure':failure,'attempts':records,'dose_completed':g['filter_g']>=a.target_g-.2 and g['filter_g']<=a.target_g+.5,'coffee_completed':False};(a.out/'progress.json').write_text(json.dumps(status,indent=2));return status

for cycle in range(1,a.max_cycles+1):
 status=publish()
 if status['dose_completed']:break
 for task,planner in [('collect','plan_spoon_dip.py'),('dose','plan_spoon_dose.py')]:
  prefix=f'cycle-{cycle:03d}-{task}'
  args=['scripts/'+planner,'--source',str(source)]
  if task=='collect':args+=['--return-route','--dip-angle',str(a.dip_angle),'--scoop-offset-x',str(a.scoop_offset_x),'--immersion',str(a.immersion)]
  if task=='dose':args+=['--tip-azimuth',str(a.tip_azimuth)]
  plan,pr=run(prefix+'-plan',args)
  if not pr.get('pass'):failure={'stage':str(plan),'reason':pr.get('failure','no feasible waypoints')};break
  dest,r=run(prefix,['scripts/execute_spoon_transport.py','--source',str(source),'--plan',str(plan),'--orientation-weight','35','--pot-clearance','.018','--segment-seconds',str(a.segment_seconds),'--joint-speed',str(a.joint_speed)])
  row={'stage':str(dest),'source':str(source),'pass':r.get('pass',False),'failure':r.get('failure')};records.append(row);print(row,flush=True)
  if not r.get('pass'):failure=row;break
  source=dest;state=publish()
  if state['grounds']['spilled_g']>.05:failure={'stage':str(dest),'reason':'powder spill'};break
  if task=='collect' and state['grounds']['spoon_g']<.05:failure={'stage':str(dest),'reason':'insufficient collected powder'};break
 if failure:break
status=publish();(a.out/'report.json').write_text(json.dumps(status,indent=2));print({k:v for k,v in status.items() if k!='attempts'},flush=True)
