"""Rank detached-hand candidates by quasistatic finger torque feasibility.
No arm reach, no dynamics, no real payload certification.
"""
import argparse,json,shutil
from pathlib import Path
import numpy as np,mujoco
from scipy.optimize import linprog
class Args:pass
a=Args();a.out=Path('results/refined-force-plan-007');a.out.mkdir(exist_ok=False);shutil.copy2(__file__,a.out/Path(__file__).name)
r=json.loads(Path('results/loadable-refined-002/report.json').read_text());c=r['grasp'];m=mujoco.MjModel.from_xml_path(r['scene']);d=mujoco.MjData(m);d.qpos[:]=c['qpos'];mujoco.mj_forward(m,d);candidates=[dict(c,static_fit_pass=True)];jar=m.body('chaleira').id
handgs=[g for g in range(m.ngeom) if m.geom_contype[g] and m.body(int(m.geom_bodyid[g])).name.startswith(('left_hand','left_wrist'))];handles=[g for g in range(m.ngeom) if (m.geom(g).name or '').startswith('handle_col')];results=[]

for i,c in enumerate(candidates):
 if not c['static_fit_pass']:continue
 joints=[m.actuator_trnid[a,0] for a in range(m.nu)];qa=m.jnt_qposadr[joints];va=m.jnt_dofadr[joints];limits=m.jnt_actfrcrange[joints]
 d.qpos[:]=c['qpos'];mujoco.mj_forward(m,d);com=d.xipos[jar];W=[];T=[];contacts=[]
 for g in handgs:
  choices=[]
  for h in handles:
   pts=np.zeros(6);dist=mujoco.mj_geomDistance(m,d,g,h,.01,pts);choices.append((dist,h,pts))
  dist,h,pts=min(choices,key=lambda x:x[0])
  if not -.0007<dist<.0008 or abs(dist)<1e-8:continue
  normal=(pts[3:]-pts[:3])*np.sign(dist);normal/=np.linalg.norm(normal);point=(pts[3:]+pts[:3])/2
  t1=np.cross(normal,[0,0,1])
  if np.linalg.norm(t1)<.1:t1=np.cross(normal,[0,1,0])
  t1/=np.linalg.norm(t1);t2=np.cross(normal,t1)
  jp=np.zeros((3,m.nv));jr=np.zeros((3,m.nv));mujoco.mj_jac(m,d,jp,jr,point,int(m.geom_bodyid[g]));contacts.append({'body':m.body(int(m.geom_bodyid[g])).name,'point':point.tolist(),'normal':normal.tolist()})
  for spin in [-1,-.7071,0,.7071,1]:
   for angle in np.linspace(0,2*np.pi,12,endpoint=False):
    force=normal+np.sqrt(1-spin**2)*(np.cos(angle)*t1+np.sin(angle)*t2);moment=.005*spin*normal;W.append(np.r_[force,np.cross(point-com,force)+moment]);T.append(jp[:,va].T@force+jr[:,va].T@moment)
 if not W:continue
 W=np.array(W).T;T=np.array(T).T;bias=d.qfrc_bias[va];A=np.vstack([T,-T]);b=np.r_[limits[:,1]-bias,bias-limits[:,0]]
 res=linprog(np.r_[np.zeros(W.shape[1]),-1],A_ub=np.column_stack([A,np.zeros(len(b))]),b_ub=b,A_eq=np.column_stack([W,-np.r_[-m.opt.gravity,[0,0,0]]]),b_eq=np.zeros(6),bounds=[(0,None)]*W.shape[1]+[(0,10)],method='highs')
 unit=linprog(np.ones(W.shape[1]),A_ub=A,b_ub=b,A_eq=W,b_eq=np.r_[-m.opt.gravity,[0,0,0]],bounds=(0,None),method='highs')
 force_plan={'unit_kg_feasible':bool(unit.success)}
 if res.success:force_plan.update(maxload_motor_torques={m.joint(j).name:float(v) for j,v in zip(joints,bias+T@res.x[:-1])},bias={m.joint(j).name:float(v) for j,v in zip(joints,bias)})
 if unit.success:force_plan.update(motor_contact_torque_Nm=(T@unit.x).tolist(),motor_total_torque_Nm=(bias+T@unit.x).tolist(),contact_normal_forces_N=unit.x.reshape(len(contacts),-1).sum(axis=1).tolist(),actuator_joint_names=[m.joint(j).name for j in joints])
 results.append({'force_plan':force_plan,'candidate_index':i,'capacity_kg':float(res.x[-1]) if res.success else 0.,'solver_success':bool(res.success),'contacts':contacts,'orientation_deg':[180,0,60],'grip_height_m':0.14})
results.sort(key=lambda x:-x['capacity_kg']);(a.out/'report.json').write_text(json.dumps({'model':'gpt-6-astra','source':'results/loadable-refined-002/report.json','scope':'Potential fingertip contacts near surfaces; conservative sampled friction cone; finger motor torque limits only; no arm or dynamic validation; capped at10kg','results':results},indent=2));print([(r['candidate_index'],round(r['capacity_kg'],3),r['orientation_deg']) for r in results[:15]])
