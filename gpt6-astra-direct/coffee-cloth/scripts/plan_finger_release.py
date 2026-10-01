"""Jointly unthread fingers and palm from handle, for reversible geometric approach."""
import json,shutil
from pathlib import Path
import numpy as np,mujoco
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
out=Path('results/finger-release-005');out.mkdir(exist_ok=False);shutil.copy2(__file__,out/Path(__file__).name)
r=json.loads(Path('results/loadable-refined-002/report.json').read_text());c=r['grasp'];m=mujoco.MjModel.from_xml_path(r['scene']);d=mujoco.MjData(m);base=np.array(c['qpos']);names=c['joint_names']+c['hand_joint_names'];js=np.array([m.joint(n).id for n in names]);qa=m.jnt_qposadr[js];bounds=m.jnt_range[js].copy();bounds[0]=[-.5,.5];bounds[5:8]=np.deg2rad([[-60,60],[-45,45],[-30,30]]);q0=base[qa];p=np.array(c['palm_target_m']);R=np.array(c['palm_rotation']);palm=m.body('left_wrist_yaw_link').id;metal=m.geom('chaleira_hot_body').id;handle=[g for g in range(m.ngeom) if (m.geom(g).name or '').startswith('handle_col')];hgs=[g for g in range(m.ngeom) if m.geom_contype[g] and m.body(int(m.geom_bodyid[g])).name.startswith(('left_hand','left_wrist'))]
results=[]
for direction in [[-1,1,0],[-1,1,-1],[0,0,1],[0,1,0]]:
 q=q0.copy();path=[];failure=None
 for distance in np.linspace(0,.10,51):
  target=p+np.array(direction)/np.linalg.norm(direction)*distance;previous=q.copy()
  def residual(x):
   d.qpos[:]=base;d.qpos[qa]=x;mujoco.mj_forward(m,d);gaps=np.array([min(mujoco.mj_geomDistance(m,d,g,h,.05,None) for h in handle) for g in hgs]);hot=np.array([mujoco.mj_geomDistance(m,d,g,metal,.05,None) for g in hgs]);return np.r_[(d.xpos[palm]-target)*500,Rotation.from_matrix(d.xmat[palm].reshape(3,3)@R.T).as_rotvec()*.2,np.minimum(gaps-(.00025+min(distance,.02)*.03),0)*8000,np.minimum(hot-.002,0)*8000,(x-previous)*.03]
  sol=least_squares(residual,np.clip(q,bounds[:,0]+1e-8,bounds[:,1]-1e-8),bounds=bounds.T,max_nfev=120,diff_step=1e-4);q=sol.x;residual(q);ep=np.linalg.norm(d.xpos[palm]-target);bad=[]
  for ct in d.contact:
   if ct.dist>=0:continue
   bn=[m.body(int(m.geom_bodyid[g])).name for g in [ct.geom1,ct.geom2]];gn=[m.geom(int(g)).name or '' for g in [ct.geom1,ct.geom2]]
   if any(n.startswith(('left_','right_','torso','pelvis','waist','head')) for n in bn) and not(any(g.startswith('handle_col') for g in gn) and any(n.startswith('left_hand') for n in bn) and ct.dist>=0):bad.append(bn)
  if bad or ep>.02:failure={'distance':distance,'contacts':bad,'position_error':ep};break
  path.append(d.qpos.copy())
  if len(path)%10==0:print(direction,len(path),flush=True)
 row={'direction':direction,'samples':len(path),'failure':failure,'pass':failure is None};results.append(row)
 if failure is None:np.savez_compressed(out/'path.npz',qpos=np.array(path[::-1]));break
 print(row,flush=True)
(out/'report.json').write_text(json.dumps({'model':'gpt-6-astra','scene':r['scene'],'grasp':c,'results':results},indent=2))
