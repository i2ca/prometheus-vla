"""Re-solve pregrasp using conservative task posture bounds, not altered hardware limits."""
import json,shutil,sys
from pathlib import Path
import numpy as np,mujoco
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'g1-cup-grasp/scripts'))
from g1_sim import G1Sim,ARM_JOINTS
from kinematics import ArmIK
out=Path('results/neutral-wrist-003');out.mkdir(exist_ok=False);shutil.copy2(__file__,out/Path(__file__).name)
s=G1Sim(str(Path('results/free-props-001/scene.xml').resolve()));m,d=s.m,s.d
base=np.load('results/clear-approach-001/path.npz')['qpos'][-1];d.qpos[:]=base;mujoco.mj_forward(m,d)
names=['waist_yaw_joint']+[n.replace('right_','left_') for n in ARM_JOINTS];ik=ArmIK(s,names,palm='left_wrist_yaw_link');old=base[ik.qa].copy();p,R=ik.fk(old);bounds=ik.bounds.copy();bounds[0]=[-.8,.25];bounds[-3:]=np.deg2rad([[-60,60],[-45,45],[-30,30]])
rng=np.random.default_rng(91);results=[]
def residual(q):
 pos,rot=ik.fk(q)
 return np.r_[(pos-p)*100,Rotation.from_matrix(rot@R.T).as_rotvec()*5,q[-3:]*.015,(q-old)*.001]
def audit(q):
 d.qpos[:]=base;d.qpos[ik.qa]=q;mujoco.mj_forward(m,d);bad=[]
 for ct in d.contact:
  bn=[m.body(int(m.geom_bodyid[g])).name for g in [ct.geom1,ct.geom2]]
  if ct.dist<0 and any(n.startswith(('left_','right_','torso','waist','pelvis','head')) for n in bn):bad.append(bn)
 return bad
for i in range(32):
 seed=np.clip(old if i==0 else old+rng.normal(0,.8,len(old)),bounds[:,0]+1e-6,bounds[:,1]-1e-6)
 opt=least_squares(residual,seed,bounds=bounds.T,max_nfev=250,diff_step=1e-4,ftol=1e-9,gtol=1e-9,xtol=1e-9);q=opt.x;pos,rot=ik.fk(q);ep=float(np.linalg.norm(pos-p));er=float(np.linalg.norm(Rotation.from_matrix(rot@R.T).as_rotvec()));bad=audit(q);passed=ep<.001 and er<np.deg2rad(2) and not bad
 results.append({'seed':i,'q':q.tolist(),'position_error_m':ep,'orientation_error_rad':er,'contacts':bad,'pass':passed,'wrist_deg':np.rad2deg(q[-3:]).tolist()})
 if passed:state=base.copy();state[ik.qa]=q;np.save(out/'pregrasp-qpos.npy',state);break
(out/'report.json').write_text(json.dumps({'model':'gpt-6-astra','joint_names':names,'task_bounds_rad':bounds.tolist(),'hardware_limits_unchanged':True,'same_palm_target':True,'old_q':old.tolist(),'results':results},indent=2));print(results[-1])
