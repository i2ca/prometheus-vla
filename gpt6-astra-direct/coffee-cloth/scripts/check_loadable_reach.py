"""Full-arm reach for force-screened 3D grasps, task wrist bounds preserved."""
import argparse,json,sys,shutil
from pathlib import Path
import numpy as np,mujoco
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'g1-cup-grasp/scripts'))
from g1_sim import G1Sim,ARM_JOINTS
from kinematics import ArmIK
ap=argparse.ArgumentParser();ap.add_argument('--scene',type=Path,default=Path('results/free-props-001/scene.xml'));ap.add_argument('--initial',type=Path,default=Path('results/free-props-001/initial-qpos.npy'));ap.add_argument('--seeds',type=int,default=8);ap.add_argument('--source',type=Path,required=True);ap.add_argument('--capacity',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(exist_ok=False);shutil.copy2(__file__,a.out/Path(__file__).name)
s=G1Sim(str(a.scene.resolve()));m,d=s.m,s.d;initial=np.load(a.initial);d.qpos[:]=initial;mujoco.mj_forward(m,d);jar=m.body('chaleira').id;cp=d.xpos[jar].copy();CR=d.xmat[jar].reshape(3,3).copy();cs=json.loads((a.source/'candidates.json').read_text())['candidates'];rank=json.loads(a.capacity.read_text())['results'];results=[];rng=np.random.default_rng(81)
for side in ['left','right']:
 names=['waist_yaw_joint']+[n.replace('right_',side+'_') for n in ARM_JOINTS];ik=ArmIK(s,names,palm=side+'_wrist_yaw_link');bounds=ik.bounds.copy();bounds[0]=[-.5,.5];bounds[-3:]=np.deg2rad([[-60,60],[-45,45],[-30,30]]);F=np.diag([1,-1,1]) if side=='left' else np.eye(3)
 for rankrow in rank[:20]:
  if rankrow['capacity_kg']<1.1:continue
  ci=rankrow['candidate_index'];c=cs[ci];d.qpos[:]=initial;hn=[n.replace('right_',side+'_') for n in c['hand_joint_names']];hq=np.array(c['hand_q'])*([1,-1,-1,-1,-1,-1,-1] if side=='left' else 1);ha=np.array([m.jnt_qposadr[m.joint(n).id] for n in hn]);d.qpos[ha]=hq;pos=cp+CR@F@np.array(c['palm_position_kettle_frame_m']);rot=CR@F@np.array(c['palm_rotation_kettle_frame'])@F
  def residual(q):
   p,R=ik.fk(q);return np.r_[(p-pos)*100,Rotation.from_matrix(R@rot.T).as_rotvec()*5,q[-3:]*.00001]
  best=None
  for seed in range(a.seeds):
   q0=np.clip(initial[ik.qa]+rng.normal(0,.8,len(ik.qa)),bounds[:,0]+1e-7,bounds[:,1]-1e-7);fit=least_squares(residual,q0,bounds=bounds.T,max_nfev=200,diff_step=1e-4);q=fit.x;p,R=ik.fk(q);ep=np.linalg.norm(p-pos);er=np.linalg.norm(Rotation.from_matrix(R@rot.T).as_rotvec());d.qpos[ik.qa]=q;other='right' if side=='left' else 'left';d.qpos[m.jnt_qposadr[m.joint(other+'_shoulder_pitch_joint').id]]=.35+.8*abs(q[0]);mujoco.mj_forward(m,d);bad=[]
   for ct in d.contact:
    bn=[m.body(int(m.geom_bodyid[g])).name for g in [ct.geom1,ct.geom2]];gn=[m.geom(int(g)).name or '' for g in [ct.geom1,ct.geom2]]
    if ct.dist<0 and any(b.startswith(('left_','right_','torso','pelvis','waist','head')) for b in bn) and not(any(g.startswith('handle_col') for g in gn) and any(b.startswith(side+'_hand') for b in bn) and ct.dist>=-.0007):bad.append(bn)
   ok=ep<.00005 and er<.0003 and not bad;row={'candidate_index':ci,'side':side,'pass':bool(ok),'qpos':d.qpos.tolist(),'joint_names':names,'q':q.tolist(),'hand_joint_names':hn,'hand_q':hq.tolist(),'palm_target_m':pos.tolist(),'palm_rotation':rot.tolist(),'position_error_m':float(ep),'orientation_error_rad':float(er),'contacts':bad,'screened_capacity_kg':rankrow['capacity_kg'],'score':float(ep*1000+er*100+len(bad)*10)}
   if best is None or (not ok,row['score'])<(not best['pass'],best['score']):best=row
   if ok:break
  results.append(best);print(side,ci,best['pass'],round(best['score'],2),flush=True)
results.sort(key=lambda r:(not r['pass'],r['score']));(a.out/'report.json').write_text(json.dumps({'model':'gpt-6-astra','candidate_source':str(a.source),'scene':str(a.scene.resolve()),'initial':str(a.initial),'results':results},indent=2))
