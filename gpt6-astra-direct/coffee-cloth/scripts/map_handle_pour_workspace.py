"""Endpoint feasibility screen for filter placement; not a dynamic pour."""
import json,sys,shutil
from pathlib import Path
import numpy as np,mujoco
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'g1-cup-grasp/scripts'))
from g1_sim import G1Sim,ARM_JOINTS
from kinematics import ArmIK
out=Path('results/pour-workspace-002');out.mkdir(exist_ok=False);shutil.copy2(__file__,out/Path(__file__).name)
s=G1Sim(json.loads(Path('results/loadable-dynamics-011/report.json').read_text())['scene']);m,d=s.m,s.d;d.qpos[:]=np.load('results/loadable-dynamics-011/final-qpos.npy');mujoco.mj_forward(m,d)
ik=ArmIK(s,['waist_yaw_joint','waist_roll_joint','waist_pitch_joint']+[n.replace('right_','left_') for n in ARM_JOINTS],palm='left_wrist_yaw_link');ik.bounds=ik.bounds.copy();ik.bounds[0]=[-.7,.7];ik.bounds[1:3]=np.deg2rad([[-12,12],[-12,12]]);ik.bounds[-3:]=np.deg2rad([[-60,60],[-45,45],[-30,30]]);q0=np.r_[d.qpos[ik.qa],0.];bounds=np.vstack([ik.bounds,[-np.pi,np.pi]]);p,R=ik.fk(q0[:-1]);b=m.body('chaleira').id;c=d.xpos[b].copy();C=d.xmat[b].reshape(3,3).copy();local=R.T@(c-p);local_R=R.T@C;lip=np.array([-.091012658,.00010992,.209200575]);rows=[];rng=np.random.default_rng(1909)
for xy in [[.27,.16],[.28,.08],[.30,.20],[.26,.0],[.30,-.1],[.36,-.16],[.35,.15],[.4,.15]]:
 for z in [.995,1.02]:
  spout=np.r_[xy,z];q=q0.copy();samples=[]
  for deg in [0,20,40,60,80]:
   Cr=C@Rotation.from_rotvec([0,-np.deg2rad(deg),0]).as_matrix();Rg=Cr@local_R.T;pg=spout-Cr@lip-Rg@local
   def fun(x):
    pp,rr=ik.fk(x[:-1]);CC=Rotation.from_rotvec([0,0,x[-1]]).as_matrix()@Cr;RR=CC@local_R.T;PP=spout-CC@lip-RR@local;return np.r_[(pp-PP)*100,Rotation.from_matrix(rr@RR.T).as_rotvec()*10,(x-q)*.001]
   best=None
   for seed in [q,q0]+[rng.uniform(bounds[:,0],bounds[:,1]) for _ in range(2)]:
    fit=least_squares(fun,np.clip(seed,bounds[:,0]+1e-8,bounds[:,1]-1e-8),bounds=bounds.T,max_nfev=120)
    if best is None or fit.cost<best.cost:best=fit
    if fit.cost<1e-7:break
   q=best.x;pp,rr=ik.fk(q[:-1]);CC=Rotation.from_rotvec([0,0,q[-1]]).as_matrix()@Cr;Rg=CC@local_R.T;pg=spout-CC@lip-Rg@local;ep=np.linalg.norm(pp-pg);er=np.linalg.norm(Rotation.from_matrix(rr@Rg.T).as_rotvec());samples.append({'tilt_deg':deg,'ep':float(ep),'er_deg':float(np.rad2deg(er)),'q':q.tolist()})
  row={'spout_m':spout.tolist(),'pass':all(x['ep']<.002 and x['er_deg']<2 for x in samples),'samples':samples};rows.append(row);print(spout,row['pass'],[(x['tilt_deg'],round(x['ep'],3),round(x['er_deg'],1)) for x in samples],flush=True)
(out/'report.json').write_text(json.dumps({'model':'gpt-6-astra','scope':'endpoint IK only; collision and path and physics not checked','joint_names':ik.joint_names,'rows':rows},indent=2))
