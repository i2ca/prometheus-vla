"""Backward extraction paths from force-screened 3D handle grasp."""
import json,shutil,sys
from pathlib import Path
import numpy as np,mujoco
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'g1-cup-grasp/scripts'))
from g1_sim import G1Sim
from kinematics import ArmIK
out=Path('results/loadable-approach-005');out.mkdir(exist_ok=False);shutil.copy2(__file__,out/Path(__file__).name)
r=json.loads(Path('results/loadable-exact-002/report.json').read_text());c=r['grasp'];s=G1Sim(r['scene']);m,d=s.m,s.d;base=np.array(c['qpos']);ik=ArmIK(s,c['joint_names'],palm='left_wrist_yaw_link');ik.bounds=ik.bounds.copy();ik.bounds[0]=[-.5,.5];ik.bounds[-3:]=np.deg2rad([[-60,60],[-45,45],[-30,30]]);ha=np.array([m.jnt_qposadr[m.joint(n).id] for n in c['hand_joint_names']]);hq=np.array(c['hand_q']);end=np.array(c['q']);p=np.array(c['palm_target_m']);R=np.array(c['palm_rotation']);rows=[];d.qpos[:]=base;p,R=ik.fk(end);c['palm_target_m']=p.tolist();c['palm_rotation']=R.tolist();table=m.geom('tampo').id
handgs=[g for g in range(m.ngeom) if (m.geom_contype[g] or m.geom_conaffinity[g]) and m.body(int(m.geom_bodyid[g])).name.startswith(('left_hand','left_wrist','left_elbow'))]
def audit():
 mujoco.mj_forward(m,d)
 for ct in d.contact:
  if ct.dist>=0:continue
  bn=[m.body(int(m.geom_bodyid[g])).name for g in [ct.geom1,ct.geom2]];gn=[m.geom(int(g)).name or '' for g in [ct.geom1,ct.geom2]]
  if not any(n.startswith(('left_','right_','torso','waist','pelvis','head')) for n in bn):continue
  if any(g.startswith('handle_col') for g in gn) and any(n.startswith('left_hand') for n in bn) and ct.dist>-.0007:continue
  return bn
 for g in handgs:
  if mujoco.mj_geomDistance(m,d,g,table,.025,None)<.025:return ['table_margin',m.body(int(m.geom_bodyid[g])).name]
 return None
for direction in [[1,1,0],[-1,1,1],[-1,-1,1],[1,1,1],[1,-1,1],[-1,2,.5],[-2,1,.5],[0,0,1],[-1,0,1],[0,1,1]]:
 for factor in [.9,.8,1,1.05,1.1,.7]:
  d.qpos[:]=base;d.qpos[ha]=hq*factor;q=end.copy();path=[];bad=None
  for retreat in np.linspace(0,.12,61):
   pos=p+retreat*np.array(direction)/np.linalg.norm(direction);q,e=ik.solve(pos,R,q,reference=end,iterations=160);d.qpos[ik.qa]=q;d.qpos[m.jnt_qposadr[m.joint('right_shoulder_pitch_joint').id]]=.35+.8*max(0,q[0]);bad=audit()
   if bad or e['position_error_m']>.0015 or e['orientation_error_rad']>.03:bad={'contact':bad,'retreat':float(retreat),'ik':e};break
   path.append(d.qpos.copy())
  row={'direction':direction,'factor':factor,'pass':bad is None,'failure':bad,'samples':len(path)};rows.append(row)
  if bad is None:
   fn=f'path-{len(rows)-1:02d}.npz';np.savez_compressed(out/fn,qpos=np.array(path[::-1]));row['path']=fn
  print(direction,factor,bad is None,len(path),flush=True)
(out/'report.json').write_text(json.dumps({'model':'gpt-6-astra','scene':r['scene'],'grasp':c,'results':rows},indent=2))
