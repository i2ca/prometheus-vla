"""Offline right-hand lid approach search; held left kettle remains in its saved pose."""
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
out=Path('results/lid-reach-006');out.mkdir(exist_ok=False);shutil.copy2(__file__,out/Path(__file__).name)
source=Path('results/kettle-rest-002');r=json.loads((source/'report.json').read_text());assert r['pass'];s=G1Sim(r['scene']);m,d=s.m,s.d;ck=np.load(source/'continuation.npz');coupler=LiquidMassCoupler(m,float(ck['dry_kettle_kg']));water=KettleWater(initial_ml=800)
for name,v in json.loads((source/'liquid-state.json').read_text()).items():setattr(water,name,v)
mujoco.mj_setState(m,d,ck['integration'],mujoco.mjtState.mjSTATE_INTEGRATION);coupler.apply(d,water);base=d.qpos.copy();ik=ArmIK(s);ik.bounds=ik.bounds.copy();ik.bounds[-3:]=np.deg2rad([[-60,60],[-45,45],[-30,30]]);qa=m.jnt_qposadr[[m.joint(n).id for n in HAND_JOINTS]];arm,hand,fps=load_seed(0);open_hand=hand[0].copy();gi=int(np.flatnonzero(hand[:,4]>.5)[0]);limits=m.jnt_range[[m.joint(n).id for n in HAND_JOINTS]];close_hand=np.clip(hand[0]+3*(hand[gi]-hand[0]),limits[:,0],limits[:,1]);d.qpos[qa]=open_hand;mujoco.mj_forward(m,d);lid=m.body('tampa').id;knob=d.xpos[lid]+d.xmat[lid].reshape(3,3)@np.array([0,0,.049]);results=[];states=[]
for angle in np.linspace(-180,180,37)[:-1]:
 theta=np.deg2rad(angle);fx=np.array([np.cos(theta),np.sin(theta),0]);fy=np.array([0,0,-1]);R=np.column_stack([fx,fy,np.cross(fx,fy)]);p=knob-R@np.array([.11,.03,0]);initial=base[ik.qa].copy()
 def residual(x):
  pp,rr=ik.fk(x);return np.r_[(pp-p)*1000,Rotation.from_matrix(rr@R.T).as_rotvec()*30,(x-initial)*.05]
 rng=np.random.default_rng(100+int(angle+180));seeds=[initial,np.array([-.8,-.8,0,.7,0,0,0]),np.array([.5,-.524,-.397,-.721,.321,.386,.082])]+[np.r_[rng.uniform([-1.5,-1.8,-1.5,0],[1.5,0,1.5,2]),[0,0,0]] for _ in range(6)];fits=[least_squares(residual,np.clip(seed,ik.bounds[:,0]+1e-9,ik.bounds[:,1]-1e-9),bounds=ik.bounds.T,max_nfev=120) for seed in seeds];fit=min(fits,key=lambda x:np.linalg.norm(residual(x.x)));pp,rr=ik.fk(fit.x);ep=float(np.linalg.norm(pp-p));er=float(np.rad2deg(np.linalg.norm(Rotation.from_matrix(rr@R.T).as_rotvec())));ik.d.qpos[qa]=open_hand;mujoco.mj_forward(m,ik.d);contacts=[]
 for ct in ik.d.contact:
  if ct.dist>=0:continue
  bn=[m.body(int(m.geom_bodyid[g])).name for g in [ct.geom1,ct.geom2]];gn=[m.geom(int(g)).name or '' for g in [ct.geom1,ct.geom2]];allowed=any(n.startswith('left_hand') for n in bn) and any(n.startswith('handle_col') for n in gn)
  if not allowed and any(n.startswith(('left_','right_','torso','waist','pelvis','head')) for n in bn):contacts.append(bn)
 passed=ep<.002 and er<5 and not contacts;row={'yaw_deg':float(angle),'pass':passed,'position_error_m':ep,'rotation_error_deg':er,'contacts':contacts,'palm_goal_m':p.tolist(),'palm_R':R.tolist(),'arm_joints':fit.x.tolist()};results.append(row);states.append(ik.d.qpos.copy());print({k:v for k,v in row.items() if k not in ['palm_R','arm_joints']},flush=True)
np.savez_compressed(out/'candidates.npz',qpos=states,open_hand=open_hand,close_hand=close_hand);(out/'report.json').write_text(json.dumps({'model':'gpt-6-astra','source':str(source),'scene':r['scene'],'scope':'offline endpoint search, fixed torso, current left hand holds kettle; no path or physical lid validation','knob_m':knob.tolist(),'arm_names':ARM_JOINTS,'hand_names':HAND_JOINTS,'results':results},indent=2))
