"""Open a fitted pinch with small finger Jacobian steps, offline only."""
import json,sys,shutil
from pathlib import Path
import numpy as np,mujoco
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'g1-cup-grasp/scripts'))
from g1_sim import G1Sim,HAND_JOINTS
from elbow_anatomy import ElbowAnatomy
out=Path('results/spoon-right-whole-fit-023');out.mkdir(exist_ok=False);shutil.copy2(__file__,out/Path(__file__).name);r=json.loads(Path('results/spoon-right-whole-fit-020/report.json').read_text());s=G1Sim(r['scene']);m,d=s.m,s.d;d.qpos[:]=np.load('results/spoon-right-whole-fit-020/candidates.npz')['qpos'][0];mujoco.mj_forward(m,d);obj=m.body('scoop').id;palm=m.body('right_wrist_yaw_link').id;table=m.geom('tampo').id;hj=[m.joint(n).id for n in HAND_JOINTS];ha=m.jnt_qposadr[hj];va=m.jnt_dofadr[hj];qa=m.jnt_qposadr[[m.joint(n).id for n in r['joint_names']]];tips=[next(g for g in range(m.ngeom) if m.geom_contype[g] and m.body(int(m.geom_bodyid[g])).name=='right_hand_'+n+'_link') for n in ['thumb_2','index_1']];shaft=[g for g in range(m.ngeom) if m.geom_contype[g] and m.geom_bodyid[g]==obj and .040<m.geom_pos[g,0]<.059];hand=[g for g in range(m.ngeom) if m.geom_contype[g] and m.body(int(m.geom_bodyid[g])).name.startswith(('right_hand','right_wrist'))];rows=[]
def contacts():
 gaps=[];points=[];normals=[];J=[]
 for g in tips:
  options=[]
  for h in shaft:
   p=np.zeros(6);gap=mujoco.mj_geomDistance(m,d,g,h,.04,p);options.append((gap,p))
  gap,p=min(options,key=lambda x:x[0]);n=(p[3:]-p[:3])*np.sign(gap);n/=max(np.linalg.norm(n),1e-9);j=np.zeros((3,m.nv));jr=np.zeros_like(j);mujoco.mj_jac(m,d,j,jr,p[:3],int(m.geom_bodyid[g]));gaps.append(gap);points.append(p[3:]);normals.append(n);J.append(n@j[:,va])
 return np.array(gaps),np.array(points),np.array(normals),np.array(J)
for k in range(250):
 gaps,points,normals,J=contacts();desired=np.clip(gaps-.002,-.00005,.00005);delta=J.T@np.linalg.solve(J@J.T+np.eye(2)*1e-8,desired);delta*=min(1,.002/max(np.max(np.abs(delta)),1e-9));d.qpos[ha]=np.clip(d.qpos[ha]+delta,m.jnt_range[hj,0],m.jnt_range[hj,1]);mujoco.mj_forward(m,d);rows.append({'step':k,'gaps':gaps.tolist()})
 if np.max(np.abs(gaps-.002))<.00005:break
gaps,points,normals,J=contacts();local=(points-d.xpos[obj])@d.xmat[obj].reshape(3,3);u=points[1]-points[0];u/=np.linalg.norm(u);cone=[float(normals[0]@u),float(-normals[1]@u)];bad=[]
for c in d.contact:
 bn=[m.body(int(m.geom_bodyid[g])).name for g in [c.geom1,c.geom2]]
 if c.dist<0 and any(n.startswith(('left_','right_','torso','waist','pelvis','head')) for n in bn):bad.append(bn)
gap=min(mujoco.mj_geomDistance(m,d,g,table,1,None) for g in hand);anatomy=ElbowAnatomy(m);passed=not bad and anatomy.valid(d) and gap>.01 and max(abs(gaps-.002))<.0001 and min(cone)>.8 and abs(local[0,0]-local[1,0])<.001
row={'seed':0,'pass':bool(passed),'q':d.qpos[qa].tolist(),'hand':d.qpos[ha].tolist(),'contact_gap_m':gaps.tolist(),'object_contact_local':local.tolist(),'contact_line_normal_cosines':cone,'palm_goal_m':d.xpos[palm].tolist(),'palm_R':d.xmat[palm].reshape(3,3).tolist(),'table_gap_m':float(gap),'contacts':bad};(out/'report.json').write_text(json.dumps({'model':'gpt-6-astra','source':r['source'],'scene':r['scene'],'joint_names':r['joint_names'],'scope':'offline opening only','results':[row],'opening_steps':rows},indent=2));np.savez_compressed(out/'candidates.npz',qpos=[d.qpos]);print(row)
