"""Whole-body IK for fitted lid knob pinches in the corrected prepared episode."""
import json,sys,shutil
from pathlib import Path
import numpy as np,mujoco
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'g1-cup-grasp/scripts'))
from g1_sim import G1Sim,ARM_JOINTS,HAND_JOINTS
from kinematics import ArmIK
from elbow_anatomy import ElbowAnatomy
from liquid_mass import LiquidMassCoupler
from kettle_liquid import KettleWater
out=Path('results/spoon-right-reach-002');out.mkdir(exist_ok=False);shutil.copy2(__file__,out/Path(__file__).name);shutil.copy2('scripts/elbow_anatomy.py',out/'elbow_anatomy.py');source=Path('results/lid-place-005');r=json.loads((source/'report.json').read_text());assert r['pass'];s=G1Sim(r['scene']);m,d=s.m,s.d;ck=np.load(source/'continuation.npz');water=KettleWater(initial_ml=800);coupler=LiquidMassCoupler(m,.78)
for n,v in json.loads((source/'liquid-state.json').read_text()).items():setattr(water,n,v)
mujoco.mj_setState(m,d,ck['integration'],mujoco.mjtState.mjSTATE_INTEGRATION);mujoco.mj_forward(m,d);coupler.apply(d,water);spoon_a=m.jnt_qposadr[m.body_jntadr[m.body('scoop').id]];d.qpos[spoon_a]-=.03;mujoco.mj_forward(m,d);base=d.qpos.copy();names=['waist_yaw_joint','waist_roll_joint','waist_pitch_joint']+list(ARM_JOINTS)+[n.replace('right_','left_') for n in ARM_JOINTS];ik=ArmIK(s,names,palm='right_wrist_yaw_link');bounds=ik.bounds.copy();bounds[0]=[-.7,.7];bounds[1:3]=np.deg2rad([[-12,12],[-12,12]])
anatomy=ElbowAnatomy(m);anatomy.bound_search(m,ik.joint_names,bounds)
for j in [7,14]:bounds[j:j+3]=np.deg2rad([[-60,60],[-45,45],[-30,30]])
ha=m.jnt_qposadr[[m.joint(n).id for n in list(HAND_JOINTS)]];left=m.body('left_wrist_yaw_link').id;lp=d.xpos[left].copy();lr=d.xmat[left].reshape(3,3).copy();fits=json.loads(Path('results/spoon-right-fit-010/report.json').read_text())['results'];states=[];results=[]
for row in fits:
 if not row['pass']:results.append({'pass':False});states.append(base.copy());continue
 d.qpos[:]=base;d.qpos[ha]=row['hand'];p=np.array(row['palm_goal_m']);R=np.array(row['palm_R']);q=base[ik.qa].copy();avoid=set();initial=base[ik.qa].copy()
 for retry in range(6):
  pairs=sorted(avoid)
  def residual(x):
   pp,rr=ik.fk(x);gaps=[min(0,mujoco.mj_geomDistance(m,ik.d,g,h,.02,None)-.003) for g,h in pairs];return np.r_[(pp-p)*1000,Rotation.from_matrix(rr@R.T).as_rotvec()*30,(ik.d.xpos[left]-lp)*500,Rotation.from_matrix(ik.d.xmat[left].reshape(3,3)@lr.T).as_rotvec()*.3,(x-initial)*.05,np.array(gaps)*10000,anatomy.penalty(ik.d)]
  fit=least_squares(residual,np.clip(q,bounds[:,0]+1e-9,bounds[:,1]-1e-9),bounds=bounds.T,max_nfev=200);q=fit.x;pp,rr=ik.fk(q);ep=float(np.linalg.norm(pp-p));er=float(np.rad2deg(np.linalg.norm(Rotation.from_matrix(rr@R.T).as_rotvec())));le=float(np.linalg.norm(ik.d.xpos[left]-lp));mujoco.mj_forward(m,ik.d);bad=[]
  for c in ik.d.contact:
   if c.dist>=0:continue
   gs=[int(c.geom1),int(c.geom2)];bn=[m.body(int(m.geom_bodyid[g])).name for g in gs]
   if any(n.startswith(('right_','left_','torso','waist','pelvis','head')) for n in bn):bad.append(bn);avoid.add(tuple(sorted(gs)))
  if not bad:break
 result={'pass':anatomy.valid(ik.d,tolerance_deg=.2) and ep<.001 and er<2 and le<.002 and not bad,'fit_index':row['seed'],'position_error_m':ep,'orientation_error_deg':er,'other_palm_error_m':le,'contacts':bad,'actual_palm_m':pp.tolist(),'actual_palm_R':rr.tolist(),'hand':row['hand'],'q':q.tolist()};results.append(result);states.append(ik.d.qpos.copy());print(result,flush=True)
np.savez_compressed(out/'candidates.npz',qpos=states);(out/'report.json').write_text(json.dumps({'model':'gpt-6-astra','source':str(source),'scene':r['scene'],'joint_names':names,'proposed_initial_spoon_shift_m':[-.03,0,0],'scope':'offline IK only with proposed initial spoon shift; never execute as a continuation','results':results},indent=2))
