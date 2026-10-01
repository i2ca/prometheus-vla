"""Rank detached-hand candidates by quasistatic finger torque feasibility.
No arm reach, no dynamics, no real payload certification.
"""
import argparse,json,shutil
from pathlib import Path
import numpy as np,mujoco
from scipy.optimize import linprog
ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(exist_ok=False);shutil.copy2(__file__,a.out/Path(__file__).name)
m=mujoco.MjModel.from_xml_path(str(a.source/'detached-hand.xml'));d=mujoco.MjData(m);candidates=json.loads((a.source/'candidates.json').read_text())['candidates'];jar=m.body('chaleira').id
handgs=[g for g in range(m.ngeom) if m.geom_contype[g] and m.body(int(m.geom_bodyid[g])).name.startswith(('right_hand','right_wrist'))];handles=[g for g in range(m.ngeom) if (m.geom(g).name or '').startswith('handle_col')];results=[]
for i,c in enumerate(candidates):
 if not c['static_fit_pass']:continue
 joints=[m.joint(n).id for n in c['hand_joint_names']];qa=m.jnt_qposadr[joints];va=m.jnt_dofadr[joints];limits=m.jnt_actfrcrange[joints]
 d.qpos[qa]=c['hand_q'];d.mocap_pos[0]=c['palm_position_kettle_frame_m'];mujoco.mju_mat2Quat(d.mocap_quat[0],np.array(c['palm_rotation_kettle_frame']).ravel());mujoco.mj_forward(m,d);com=d.xipos[jar];W=[];T=[];contacts=[]
 for g in handgs:
  choices=[]
  for h in handles:
   pts=np.zeros(6);dist=mujoco.mj_geomDistance(m,d,g,h,.01,pts);choices.append((dist,h,pts))
  dist,h,pts=min(choices,key=lambda x:x[0])
  if not 1e-7<dist<.0008:continue
  normal=(pts[3:]-pts[:3]);normal/=np.linalg.norm(normal);point=(pts[3:]+pts[:3])/2
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
 results.append({'candidate_index':i,'capacity_kg':float(res.x[-1]) if res.success else 0.,'solver_success':bool(res.success),'contacts':contacts,'orientation_deg':[c['yaw_deg'],c.get('pitch_deg',0),c.get('roll_deg',0)],'grip_height_m':c['grip_height_m']})
results.sort(key=lambda x:-x['capacity_kg']);(a.out/'report.json').write_text(json.dumps({'model':'gpt-6-astra','source':str(a.source),'scope':'Potential fingertip contacts near surfaces; conservative sampled friction cone; finger motor torque limits only; no arm or dynamic validation; capped at10kg','results':results},indent=2));print([(r['candidate_index'],round(r['capacity_kg'],3),r['orientation_deg']) for r in results[:15]])
