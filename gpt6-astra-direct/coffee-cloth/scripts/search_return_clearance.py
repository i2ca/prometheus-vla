"""Offline constant-orientation retreat search from physical wet checkpoint."""
import sys,json,shutil
from pathlib import Path
import mujoco,numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'g1-cup-grasp/scripts'))
from g1_sim import G1Sim,ARM_JOINTS
from kinematics import ArmIK
out=Path('results/return-clearance-search-001');out.mkdir(exist_ok=False);shutil.copy2(__file__,out/Path(__file__).name)
r=json.loads(Path('results/loadable-dynamics-019/report.json').read_text());s=G1Sim(r['scene']);m,d=s.m,s.d;ck=np.load('results/kettle-water-feedback-009/checkpoint-0063210.npz')
for name in ['body_mass','body_ipos','body_inertia','body_iquat','bvh_aabb']:getattr(m,name)[:]=ck[name]
mujoco.mj_setConst(m,mujoco.MjData(m));mujoco.mj_setState(m,d,ck['integration'],mujoco.mjtState.mjSTATE_INTEGRATION);mujoco.mj_forward(m,d)
palm=m.body('left_wrist_yaw_link').id;jar=m.body('chaleira').id;qa=m.jnt_qposadr[m.joint('chaleira_livre').id];ik=ArmIK(s,['waist_yaw_joint','waist_roll_joint','waist_pitch_joint']+[n.replace('right_','left_') for n in ARM_JOINTS],palm='left_wrist_yaw_link');bounds=ik.bounds.copy();bounds[0]=[-.7,.7];bounds[1:3]=np.deg2rad([[-12,12],[-12,12]]);bounds[-3:]=np.deg2rad([[-60,60],[-45,45],[-30,30]])
p0=d.xpos[palm].copy();R0=d.xmat[palm].reshape(3,3).copy();local=R0.T@(d.xpos[jar]-p0);localR=R0.T@d.xmat[jar].reshape(3,3);base=d.qpos.copy();initial=base[ik.qa].copy();results=[]
for idx,offset in enumerate([[0,0,.08],[.10,0,.02],[.10,0,0],[.06,.06,.02],[.06,-.06,.02],[0,.10,.02],[0,-.10,.02]]):
 q=initial.copy();pairs=set();states=[];bad=None
 for fraction in np.linspace(0,1,21):
  target=p0+np.array(offset)*fraction;previous=q.copy()
  def put(x):
   p,R=ik.fk(x);ik.d.qpos[qa:qa+3]=p+R@local;mujoco.mju_mat2Quat(ik.d.qpos[qa+3:qa+7],(R@localR).ravel());mujoco.mj_kinematics(m,ik.d);mujoco.mj_comPos(m,ik.d);return p,R
  for retry in range(5):
   active=sorted(pairs)
   def fun(x):
    p,R=put(x);gaps=[min(0,mujoco.mj_geomDistance(m,ik.d,g,h,.02,None)-.005) for g,h in active];return np.r_[(p-target)*1000,Rotation.from_matrix(R@R0.T).as_rotvec()*50,(x-previous)*.2,np.array(gaps)*10000]
   fit=least_squares(fun,np.clip(q,bounds[:,0]+1e-9,bounds[:,1]-1e-9),bounds=bounds.T,max_nfev=120);q=fit.x;p,R=put(q);mujoco.mj_collision(m,ik.d);contacts=[]
   for c in ik.d.contact:
    if c.dist>=0:continue
    gs=[int(c.geom1),int(c.geom2)];bn=[m.body(int(m.geom_bodyid[g])).name for g in gs];gn=[m.geom(g).name or '' for g in gs];allowed=any(n.startswith('left_hand') for n in bn) and any(n.startswith('handle_col') for n in gn)
    if not allowed and ('chaleira' in bn or any(n.startswith(('left_','right_','torso','waist','pelvis','head')) for n in bn)):contacts.append(bn);pairs.add(tuple(sorted(gs)))
   if not contacts:break
  ep=float(np.linalg.norm(p-target));er=float(np.rad2deg(np.linalg.norm(Rotation.from_matrix(R@R0.T).as_rotvec())));states.append(ik.d.qpos.copy())
  if contacts or ep>.002 or er>3:bad={'fraction':fraction,'position_error_m':ep,'rotation_error_deg':er,'contacts':contacts};break
 np.savez_compressed(out/f'path-{idx:02d}.npz',qpos=states);result={'index':idx,'offset_m':offset,'pass':bad is None,'failure':bad,'steps':len(states)};results.append(result);print(result,flush=True)
(out/'report.json').write_text(json.dumps({'model':'gpt-6-astra','source':'results/kettle-water-feedback-009/checkpoint-0063210.npz','scope':'offline rigid relative grasp retreat, no physics validation','results':results},indent=2))
