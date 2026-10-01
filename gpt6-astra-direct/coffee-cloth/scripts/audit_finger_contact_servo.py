"""Finite-difference audit of commanded finger closure, not physical validation."""
import sys,json,shutil
from pathlib import Path
import numpy as np,mujoco
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'g1-cup-grasp/scripts'))
from g1_sim import G1Sim,HAND_JOINTS
from finger_contact_servo import advance
out=Path('results/finger-servo-audit-002');out.mkdir(exist_ok=False)
for p in [Path(__file__),Path('scripts/finger_contact_servo.py')]:shutil.copy2(p,out/p.name)
s=G1Sim(json.loads(Path('results/lid-place-008/report.json').read_text())['scene']);m,d=s.m,s.d
d.qpos[:]=np.load('results/spoon-right-whole-fit-013/candidates.npz')['qpos'][2];mujoco.mj_forward(m,d)
hj=[m.joint(n).id for n in HAND_JOINTS];ha=m.jnt_qposadr[hj];ref=d.qpos[ha].copy();obj=m.body('scoop').id
tips=[next(g for g in range(m.ngeom) if m.geom_contype[g] and m.body(int(m.geom_bodyid[g])).name=='right_hand_'+n+'_link') for n in ['thumb_2','index_1','index_0']]
def contacts():
 out=[]
 for i,g in enumerate(tips):
  options=[]
  for h in range(m.ngeom):
   if m.geom_bodyid[h]!=obj or not m.geom_contype[h] or not ((.020<m.geom_pos[h,0]<.038) if i==2 else (.040<m.geom_pos[h,0]<.059)):continue
   pts=np.zeros(6);gap=mujoco.mj_geomDistance(m,d,g,h,.04,pts);options.append((gap,pts))
  gap,pts=min(options,key=lambda x:x[0]);n=(pts[3:]-pts[:3])*np.sign(gap);n/=np.linalg.norm(n);out.append((gap,pts[:3],n))
 return out
rows=[];offset=np.zeros(7)
for i in range(80):
 cs=contacts();old=[c[0] for c in cs];offset=advance(m,d,hj,[m.geom_bodyid[g] for g in tips],cs,[0,0,0],ref,offset)
 d.qpos[ha]=ref+offset;mujoco.mj_forward(m,d);new=[c[0] for c in contacts()];rows.append({'step':i,'before':old,'after':new})
 if min(new)<0:break
result={'model':'gpt-6-astra','pass':all(b<a for a,b in zip(rows[0]['before'],rows[0]['after'])),'scope':'kinematic closure direction only; fixed object, no physics proof','rows':rows}
(out/'report.json').write_text(json.dumps(result,indent=2));print({k:v for k,v in result.items() if k!='rows'});print(rows[0]);print(rows[-1])
