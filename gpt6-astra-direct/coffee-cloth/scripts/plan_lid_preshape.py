"""Offline lid reach with torso and both arms; keep empty left hand in free space."""
import sys,json,shutil
from pathlib import Path
import numpy as np,mujoco
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'g1-cup-grasp/scripts'))
from g1_sim import G1Sim,ARM_JOINTS,HAND_JOINTS,load_seed
from kinematics import ArmIK
from liquid_mass import LiquidMassCoupler
from kettle_liquid import KettleWater
out=Path('results/lid-reach-012');out.mkdir(exist_ok=False);shutil.copy2(__file__,out/Path(__file__).name);source=Path('results/kettle-place-020');report=json.loads((source/'report.json').read_text());assert report['pass'];s=G1Sim(report['scene']);m,d=s.m,s.d;ck=np.load(source/'continuation.npz');water=KettleWater(initial_ml=800);coupler=LiquidMassCoupler(m,float(ck['dry_kettle_kg']))
for n,v in json.loads((source/'liquid-state.json').read_text()).items():setattr(water,n,v)
mujoco.mj_setState(m,d,ck['integration'],mujoco.mjtState.mjSTATE_INTEGRATION);coupler.apply(d,water);arm,hand,fps=load_seed(0);ha=m.jnt_qposadr[[m.joint(n).id for n in HAND_JOINTS]];gi=int(np.flatnonzero(hand[:,4]>.5)[0]);limits=m.jnt_range[[m.joint(n).id for n in HAND_JOINTS]];close=np.clip(hand[0]+3*(hand[gi]-hand[0]),limits[:,0],limits[:,1]);opened=hand[0]+.85*(close-hand[0]);opened[0]=.6;opened[5:]=hand[0,5:];d.qpos[ha]=opened;mujoco.mj_forward(m,d);base=d.qpos.copy();left=m.body('left_wrist_yaw_link').id;lp=d.xpos[left].copy();lr=d.xmat[left].reshape(3,3).copy();lid=m.body('tampa').id;knob=d.xpos[lid]+d.xmat[lid].reshape(3,3)@np.array([0,0,.031]);names=['waist_yaw_joint','waist_roll_joint','waist_pitch_joint']+list(ARM_JOINTS)+[n.replace('right_','left_') for n in ARM_JOINTS];ik=ArmIK(s,names);ik.bounds=ik.bounds.copy();ik.bounds[0]=[-.7,.7];ik.bounds[1:3]=np.deg2rad([[-12,12],[-12,12]])
for start in [7,14]:ik.bounds[start:start+3]=np.deg2rad([[-60,60],[-45,45],[-30,30]])
initial=base[ik.qa].copy();results=[];states=[]
for angle in np.arange(-60,61,15):
 theta=np.deg2rad(angle);fx=np.array([np.cos(theta),np.sin(theta),0]);fy=np.array([0,0,-1]);R=np.column_stack([fx,fy,np.cross(fx,fy)])@Rotation.from_rotvec([0,0,np.deg2rad(45)]).as_matrix();p=knob-R@np.array([.145,.065,0]);avoid=set();best=None
 for seed_index in range(3):
  q=initial.copy()
  if seed_index:q[:3]=[-.45,0,.18];q[3:10]=[-.8,-.8,0,.7,0,0,0] if seed_index==1 else [.5,-.524,-.397,-.721,.321,.386,.082]
  for retry in range(5):
   pairs=sorted(avoid)
   def residual(x):
    pp,rr=ik.fk(x);pl=ik.d.xpos[left];rl=ik.d.xmat[left].reshape(3,3);gaps=[min(0,mujoco.mj_geomDistance(m,ik.d,g,h,.02,None)-.003) for g,h in pairs];return np.r_[(pp+rr@np.array([.145,.065,0])-knob)*1000,Rotation.from_matrix(rr@R.T).as_rotvec()*30,(pl-lp)*500,Rotation.from_matrix(rl@lr.T).as_rotvec()*.3,(x-initial)*.05,np.array(gaps)*10000]
   fit=least_squares(residual,np.clip(q,ik.bounds[:,0]+1e-9,ik.bounds[:,1]-1e-9),bounds=ik.bounds.T,max_nfev=150);q=fit.x;pp,rr=ik.fk(q);ep=float(np.linalg.norm(pp+rr@np.array([.145,.065,0])-knob));er=float(np.rad2deg(np.linalg.norm(Rotation.from_matrix(rr@R.T).as_rotvec())));left_error=float(np.linalg.norm(ik.d.xpos[left]-lp));mujoco.mj_forward(m,ik.d);bad=[]
   for ct in ik.d.contact:
    if ct.dist>=0:continue
    gs=[int(ct.geom1),int(ct.geom2)];bn=[m.body(int(m.geom_bodyid[g])).name for g in gs];allowed='tampa' in bn and any(n.startswith('right_hand') for n in bn) and ct.dist>-.0005
    if not allowed and any(n.startswith(('left_','right_','waist','pelvis','torso','head')) for n in bn):bad.append(bn);avoid.add(tuple(sorted(gs)))
   row={'yaw_deg':int(angle),'seed':seed_index,'position_error_m':ep,'orientation_error_deg':er,'left_palm_error_m':left_error,'contacts':bad,'pass':ep<.002 and er<8 and left_error<.002 and not bad,'nominal_palm_goal_m':p.tolist(),'actual_palm_m':pp.tolist(),'actual_palm_R':rr.tolist(),'palm_R':R.tolist(),'q':q.tolist()};score=ep+left_error+er*.001+len(bad)
   if best is None or score<best[0]:best=(score,row,ik.d.qpos.copy())
   if not bad:break
  if best[1]['pass']:break
 results.append(best[1]);states.append(best[2]);print({k:v for k,v in best[1].items() if k not in ['q','palm_R']},flush=True)
np.savez_compressed(out/'candidates.npz',qpos=states,open_hand=opened);(out/'report.json').write_text(json.dumps({'model':'gpt-6-astra','source':str(source),'scene':report['scene'],'scope':'offline inclined pinch-center search with both arms and torso, 45deg diagonal pinch, orientation guidance tolerance8deg; no path or physical grasp validation','joint_names':names,'results':results},indent=2))
