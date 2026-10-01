"""Projected joint-space RRT with upright, free-space attached-object geometry."""
import json,sys,shutil,time
from pathlib import Path
import numpy as np,mujoco
from scipy.optimize import least_squares
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'g1-cup-grasp/scripts'))
from g1_sim import G1Sim,ARM_JOINTS
from kinematics import ArmIK
out=Path('results/handle-transport-rrt-002');out.mkdir(exist_ok=False);shutil.copy2(__file__,out/Path(__file__).name);source=Path('results/loadable-dynamics-016');r=json.loads((source/'report.json').read_text());s=G1Sim(r['scene']);m,d=s.m,s.d;base=np.load(source/'final-qpos.npy');d.qpos[:]=base;mujoco.mj_forward(m,d);ik=ArmIK(s,['waist_yaw_joint','waist_roll_joint','waist_pitch_joint']+[n.replace('right_','left_') for n in ARM_JOINTS],palm='left_wrist_yaw_link');bounds=ik.bounds.copy();bounds[0]=[-.7,.7];bounds[1:3]=np.deg2rad([[-12,12],[-12,12]]);bounds[-3:]=np.deg2rad([[-60,60],[-45,45],[-30,30]]);q0=base[ik.qa];p,R=ik.fk(q0);b=m.body('chaleira').id;c=d.xpos[b].copy();C=d.xmat[b].reshape(3,3).copy();local=R.T@(c-p);local_R=R.T@C;lip=np.array([-.091012658,.00010992,.209200575]);spout_palm=local+local_R@lip;qa=m.jnt_qposadr[m.joint('chaleira_livre').id];right=m.jnt_qposadr[m.joint('right_shoulder_pitch_joint').id];filter_center=d.xpos[m.body('coador').id]+d.xmat[m.body('coador').id].reshape(3,3)@np.array([0,0,.205]);goal_spout=filter_center+[-.02,.04,.04];rng=np.random.default_rng(190919);checks=0;reason="";rejections={};begin=time.monotonic()
def put(q):
 pp,rr=ik.fk(q);d.qpos[:]=base;d.qpos[ik.qa]=q;d.qpos[right]=.35+.8*max(0,q[0]);d.qpos[qa:qa+3]=pp+rr@local;mujoco.mju_mat2Quat(d.qpos[qa+3:qa+7],(rr@local_R).ravel());mujoco.mj_forward(m,d);return pp,rr
metal=m.geom('chaleira_hot_body').id;table=m.geom('tampo').id;hgs=[g for g in range(m.ngeom) if m.geom_contype[g] and m.body(int(m.geom_bodyid[g])).name.startswith(('left_hand','left_wrist'))]
def valid(q):
 global checks,reason
 checks+=1;reason="clearance"
 if np.any(q<bounds[:,0]-1e-4) or np.any(q>bounds[:,1]+1e-4):return False
 pp,rr=put(q)
 if (rr@local_R)[2,2]<np.cos(np.deg2rad(8)):return False
 for ct in d.contact:
  if ct.dist>=0:continue
  bn=[m.body(int(m.geom_bodyid[g])).name for g in [ct.geom1,ct.geom2]];gn=[m.geom(int(g)).name or '' for g in [ct.geom1,ct.geom2]]
  if any(n.startswith('left_hand') for n in bn) and any(n.startswith('handle_col') for n in gn):continue
  if 'chaleira' in bn or any(n.startswith(('left_','right_','torso','pelvis','waist','head')) for n in bn):reason=str(bn);return False
 return mujoco.mj_geomDistance(m,d,metal,table,.015,None)>=.015 and all(mujoco.mj_geomDistance(m,d,g,table,.03,None)>=.03 for g in hgs)
goals=[]
for attempt in range(100):
 seed=q0 if attempt==0 else rng.uniform(bounds[:,0],bounds[:,1])
 def residual(q):
  pp,rr=ik.fk(q);return np.r_[(pp+rr@spout_palm-goal_spout)*500,(rr@local_R)[:2,2]*50,(q-q0)*.003]
 fit=least_squares(residual,np.clip(seed,bounds[:,0]+1e-8,bounds[:,1]-1e-8),bounds=bounds.T,max_nfev=100);pp,rr=put(fit.x)
 if np.linalg.norm(pp+rr@spout_palm-goal_spout)<.002 and valid(fit.x):goals.append(fit.x);print('goal',attempt,len(goals),flush=True)
 if not goals:rejections[reason]=rejections.get(reason,0)+1
 if len(goals)>=6:break
assert valid(q0),'initial grasp geometry invalid'
goals.sort(key=lambda x:np.linalg.norm(x-q0));found=None;iterations=0
if goals:
 trees=[([q0.copy()],[-1]),([goals[0]],[-1])]
 def project(q):
  def residual(x):
   pp,rr=ik.fk(x);return np.r_[(rr@local_R)[:2,2]*15,(x-q)*.4]
  fit=least_squares(residual,np.clip(q,bounds[:,0]+1e-8,bounds[:,1]-1e-8),bounds=bounds.T,max_nfev=25);return fit.x
 def edge(a,b):return all(valid(a*(1-u)+b*u) for u in np.linspace(0,1,max(2,int(np.max(np.abs(b-a))/.02)+2)))
 def extend(tree,target):
  nodes,parents=tree;idx=int(np.argmin(np.linalg.norm(np.array(nodes)-target,axis=1)));q=nodes[idx];delta=target-q;dist=np.linalg.norm(delta);raw=q+delta*min(1,.18/max(dist,1e-9));new=project(raw)
  if np.linalg.norm(new-q)<.002 or np.linalg.norm(new-q)>.5 or not edge(q,new):return None
  nodes.append(new);parents.append(idx);return len(nodes)-1
 for iteration in range(4000):
  iterations=iteration+1;side=iteration%2;A=trees[side];B=trees[1-side];target=B[0][0] if rng.random()<.25 else rng.uniform(bounds[:,0],bounds[:,1]);ia=extend(A,target)
  if ia is None:continue
  for _ in range(30):
   ib=extend(B,A[0][ia])
   if ib is None:break
   if np.linalg.norm(B[0][ib]-A[0][ia])<.2 and edge(B[0][ib],A[0][ia]):
    ends=[None,None];ends[side]=ia;ends[1-side]=ib;paths=[]
    for tree,idx in zip(trees,ends):
     arr=[]
     while idx!=-1:arr.append(tree[0][idx]);idx=tree[1][idx]
     paths.append(arr[::-1])
    found=paths[0]+paths[1][::-1];break
  if found is not None:break
  if iteration%100==0:print('rrt',iteration,[len(t[0]) for t in trees],flush=True)
  if time.monotonic()-begin>180:break
 if found is not None:
  compact=[found[0]];i=0
  while i<len(found)-1:
   for j in range(len(found)-1,i,-1):
    if edge(found[i],found[j]):break
   compact.append(found[j]);i=j
  full=[]
  for aa,bb in zip(compact,compact[1:]):
   for u in np.linspace(0,1,max(2,int(np.max(np.abs(bb-aa))/.015)+2)):
    put(aa*(1-u)+bb*u);full.append(d.qpos.copy())
  np.savez_compressed(out/'path.npz',qpos=full)
report={'model':'gpt-6-astra','scene':r['scene'],'source':str(source),'pass':found is not None,'goal_rejections':rejections,'goal_count':len(goals),'iterations':iterations,'checks':checks,'seconds':time.monotonic()-begin,'target_spout':goal_spout.tolist(),'spout_local_m':lip.tolist(),'scope':'projected RRT sampled geometry only, no dynamic transport'};(out/'report.json').write_text(json.dumps(report,indent=2));print(report)
