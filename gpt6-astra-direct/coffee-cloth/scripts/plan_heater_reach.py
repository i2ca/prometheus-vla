"""Offline left index-finger reach to estimated EEK10 top-handle rocker."""
import argparse,json,shutil,sys
from pathlib import Path
import numpy as np,mujoco
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'g1-cup-grasp/scripts'))
from g1_sim import G1Sim,ARM_JOINTS,HAND_JOINTS
from kinematics import ArmIK
from elbow_anatomy import ElbowAnatomy
ap=argparse.ArgumentParser();ap.add_argument('--elbow-forward',type=float);ap.add_argument('--elbow-drop',type=float);ap.add_argument('--free-orientation',action='store_true');ap.add_argument('--source',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(exist_ok=False);shutil.copy2(__file__,a.out/Path(__file__).name);shutil.copy2('scripts/elbow_anatomy.py',a.out/'elbow_anatomy.py');r=json.loads((a.source/'report.json').read_text());assert r['pass'];s=G1Sim(r['scene']);m,d=s.m,s.d;ck=np.load(a.source/'continuation.npz');mujoco.mj_setState(m,d,ck['integration'],mujoco.mjtState.mjSTATE_INTEGRATION);mujoco.mj_forward(m,d);base=d.qpos.copy();names=['waist_yaw_joint','waist_roll_joint','waist_pitch_joint']+[n.replace('right_','left_') for n in ARM_JOINTS]+list(ARM_JOINTS);ik=ArmIK(s,names,palm='left_wrist_yaw_link');bounds=ik.bounds.copy();bounds[0]=[-.7,.7];bounds[1:3]=np.deg2rad([[-12,12],[-12,12]])
anatomy=ElbowAnatomy(m);anatomy.bound_search(m,ik.joint_names,bounds)
for j in [7,14]:bounds[j:j+3]=np.deg2rad([[-60,60],[-45,45],[-30,30]])
handnames=[n.replace('right_','left_') for n in HAND_JOINTS];ha=m.jnt_qposadr[[m.joint(n).id for n in handnames]];hand=np.array([.6,0,0,0,0,-1.5,-1.7]);tip=m.body('left_hand_index_1_link').id;tip_local=np.array([.049,-.002,0]);button=m.site('kettle_rocker_tip').id;goal=d.site_xpos[button].copy()+[0,0,.006];right=m.body('right_wrist_yaw_link').id;rp=d.xpos[right].copy();rR=d.xmat[right].reshape(3,3).copy();hot=m.geom('chaleira_hot_body').id;hg=[g for g in range(m.ngeom) if m.geom_contype[g] and m.body(int(m.geom_bodyid[g])).name.startswith(('left_hand','left_wrist'))];results=[];states=[]
shoulder_anchor=m.joint('left_shoulder_pitch_joint').id;elbow_anchor=m.joint('left_elbow_joint').id
def posture_penalty(data):
 upper=data.xanchor[elbow_anchor]-data.xanchor[shoulder_anchor]
 return np.array(([min(0,upper[0]-a.elbow_forward)] if a.elbow_forward is not None else [])+([max(0,upper[2]+a.elbow_drop)] if a.elbow_drop is not None else []))*1000
for pitch,yaw in [(60,0),(45,0),(75,0),(60,20),(60,-20)]:
 d.qpos[:]=base;d.qpos[ha]=hand;mujoco.mj_forward(m,d);q=base[ik.qa].copy();q0=q.copy();R=Rotation.from_euler('ZY',[yaw,pitch],degrees=True).as_matrix();avoid=set()
 for retry in range(10):
  pairs=sorted(avoid)
  def residual(x):
   pp,rr=ik.fk(x);point=ik.d.xpos[tip]+ik.d.xmat[tip].reshape(3,3)@tip_local;gaps=[min(0,mujoco.mj_geomDistance(m,ik.d,g,h,.02,None)-.005) for g,h in pairs];thermal=[min(0,mujoco.mj_geomDistance(m,ik.d,g,hot,.015,None)-.006) for g in hg]
   return np.r_[(point-goal)*1000,Rotation.from_matrix(rr@R.T).as_rotvec()*(.5 if a.free_orientation else 20),(ik.d.xpos[right]-rp)*500,Rotation.from_matrix(ik.d.xmat[right].reshape(3,3)@rR.T).as_rotvec()*10,(x-q0)*.05,x[np.r_[7:10,14:17]]*(2 if a.free_orientation else 0),max(0,.4+ik.d.xmat[tip].reshape(3,3)[2,0])*(30 if a.free_orientation else 0),np.array(gaps)*5000,np.array(thermal)*5000,anatomy.penalty(ik.d),posture_penalty(ik.d)]
  fit=least_squares(residual,np.clip(q,bounds[:,0]+1e-8,bounds[:,1]-1e-8),bounds=bounds.T,max_nfev=120);q=fit.x;pp,rr=ik.fk(q);mujoco.mj_forward(m,ik.d);bad=[]
  for c in ik.d.contact:
   if c.dist>=0:continue
   gs=[int(c.geom1),int(c.geom2)];bn=[m.body(int(m.geom_bodyid[g])).name for g in gs];allowed='scoop' in bn and any(n.startswith('right_hand') for n in bn)
   if r.get('allow_tip_contact',False) and set(bn)=={'right_hand_thumb_2_link','right_hand_index_1_link'}:allowed=True
   if any(n.startswith(('left_','right_','torso','waist','pelvis','head')) for n in bn) and not allowed:bad.append(bn);avoid.add(tuple(sorted(gs)))
  if not bad:break
 point=ik.d.xpos[tip]+ik.d.xmat[tip].reshape(3,3)@tip_local;error=float(np.linalg.norm(point-goal));gap=min(mujoco.mj_geomDistance(m,ik.d,g,hot,1,None) for g in hg);passed=(np.max(np.abs(posture_penalty(ik.d)),initial=0)<2) and anatomy.valid(ik.d,tolerance_deg=.2) and error<.002 and gap>.005 and not bad;row={'elbow_relative_shoulder_m':(ik.d.xanchor[elbow_anchor]-ik.d.xanchor[shoulder_anchor]).tolist(),'pitch_deg':pitch,'yaw_deg':yaw,'pass':bool(passed),'tip_error_m':error,'hot_body_clearance_m':float(gap),'forbidden_contacts':bad,'q':q.tolist(),'palm_R':rr.tolist(),'tip_world_m':point.tolist()};results.append(row);states.append(ik.d.qpos.copy());print(row,flush=True)
(a.out/'report.json').write_text(json.dumps({'model':'gpt-6-astra','source':str(a.source),'scene':r['scene'],'elbow_forward_m':a.elbow_forward,'elbow_drop_m':a.elbow_drop,'scope':'offline reach only, no physical button press or heating','coffee_completed':False,'joint_names':names,'hand_names':handnames,'hand_q':hand.tolist(),'index_tip_local_m':tip_local.tolist(),'goal_m':goal.tolist(),'results':results},indent=2));np.savez_compressed(a.out/'candidates.npz',qpos=states)
