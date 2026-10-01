"""Offline IK diagnostic, not a completed pour. Preserves every run. gpt-6-astra."""
import argparse,json,datetime
from pathlib import Path
import numpy as np
import cv2
from replay_common import *

ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False)
sim=replay_grasp();m,d=sim.m,sim.d
ik=ArmIK(sim);q0=sim.q(ARM_JOINTS);p0,R0=ik.fk(q0)
c0=d.xpos[sim.cup_body].copy();C0=d.xmat[sim.cup_body].reshape(3,3).copy()
local_pos=R0.T@(c0-p0);local_rot=R0.T@C0
rows=[]
for axis in ([1,0,0],[0,1,0],[0,-1,0],[-1,0,0]):
 for deg in (30,45,60,75):
  C=cv2.Rodrigues(np.radians(deg)*np.array(axis,float))[0]@C0
  R=C@local_rot.T
  for x in (.32,.38,.44,.50):
   for y in (-.15,-.05,.05):
    for z in (.86,.90,.94):
     c=np.array([x,y,z]);target=c-R@local_pos
     q,err=ik.solve(target,R,q0,reference=q0,iterations=100)
     rows.append({'cup_origin':c.tolist(),'axis':axis,'tilt_deg':deg,**err,'q':q.tolist()})
good=[r for r in rows if r['position_error_m']<.003 and r['orientation_error_rad']<.03]
report={'author_model':'gpt-6-astra','scope':'offline kinematics only; no collision/path/pouring validation','initial_cup':c0.tolist(),'initial_palm':p0.tolist(),'tested':len(rows),'reachable':len(good),'rows':rows}
(a.out/'report.json').write_text(json.dumps(report,indent=2))
np.savez(a.out/'grasp-state.npz',qpos=d.qpos,qvel=d.qvel,q_des=sim.q_des,time=d.time,local_pos=local_pos,local_rot=local_rot)
print(json.dumps({k:v for k,v in report.items() if k!='rows'}))
print('first_reachable',good[:3])
