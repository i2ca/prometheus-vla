"""Collision-checked left hand route to the new handle grasp; offline only."""
import argparse,json,shutil,sys,time
from pathlib import Path
import numpy as np,mujoco
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'g1-cup-grasp/scripts'))
from g1_sim import G1Sim,ARM_JOINTS,HAND_JOINTS
from kinematics import ArmIK
from elbow_anatomy import ElbowAnatomy
HAND_JOINTS=[n.replace("right_","left_") for n in HAND_JOINTS]
from liquid_mass import LiquidMassCoupler
from kettle_liquid import KettleWater
ap=argparse.ArgumentParser();ap.add_argument('--open-factor',type=float,default=1.);ap.add_argument('--escape',type=float,nargs=3);ap.add_argument('--direct-start',action='store_true');ap.add_argument('--source',type=Path,required=True);ap.add_argument('--candidate',type=Path,required=True);ap.add_argument('--choice',type=int,default=0);ap.add_argument('--out',type=Path,required=True);cfg=ap.parse_args()
out=cfg.out;out.mkdir(exist_ok=False);shutil.copy2(__file__,out/Path(__file__).name);shutil.copy2('scripts/elbow_anatomy.py',out/'elbow_anatomy.py');source=cfg.source;r=json.loads((source/'report.json').read_text());candidate=json.loads((cfg.candidate/'report.json').read_text());choice=cfg.choice;assert candidate['results'][choice]['pass'];s=G1Sim(r['scene']);m,d=s.m,s.d;ck=np.load(source/'continuation.npz');coupler=LiquidMassCoupler(m,float(ck['dry_kettle_kg']));water=KettleWater(initial_ml=800)
for n,v in json.loads((source/'liquid-state.json').read_text()).items():setattr(water,n,v)
mujoco.mj_setState(m,d,ck['integration'],mujoco.mjtState.mjSTATE_INTEGRATION);coupler.apply(d,water);initial=d.qpos.copy();goal=np.load(cfg.candidate/'candidates.npz')['qpos'][choice];names=candidate['joint_names'];ik=ArmIK(s,names,palm='left_wrist_yaw_link');bounds=ik.bounds.copy();bounds[0]=[-.7,.7];bounds[1:3]=np.deg2rad([[-12,12],[-12,12]])
anatomy=ElbowAnatomy(m);anatomy.bound_search(m,ik.joint_names,bounds)
for start in [7,14]:bounds[start:start+3]=np.deg2rad([[-60,60],[-45,45],[-30,30]])
qa=ik.qa;ha=m.jnt_qposadr[[m.joint(n).id for n in list(HAND_JOINTS)]];closed_hand=goal[ha].copy();goal[ha[1:]]*=cfg.open_factor;preshape=initial.copy();preshape[ha]=goal[ha];left=m.body('right_wrist_yaw_link').id;table=m.geom('tampo').id;collision_data=mujoco.MjData(m);checks=0
robotgeoms=[g for g in range(m.ngeom) if m.geom_contype[g] and m.body(int(m.geom_bodyid[g])).name.startswith(('left_hand','left_wrist','right_hand','right_wrist'))]
def valid_state(state,minimum_gap=.022):
 global checks
 checks+=1;collision_data.qpos[:]=state;mujoco.mj_forward(m,collision_data)
 if not anatomy.valid(collision_data):return False
 if any(mujoco.mj_geomDistance(m,collision_data,g,m.geom('chaleira_hot_body').id,.008,None)<.008 for g in robotgeoms if m.body(int(m.geom_bodyid[g])).name.startswith('left_')):return False
 for ct in collision_data.contact:
  if ct.dist>=0:continue
  bn=[m.body(int(m.geom_bodyid[g])).name for g in [ct.geom1,ct.geom2]]
  if any(n.startswith(('left_','right_','torso','waist','pelvis','head')) for n in bn):return False
 return all(mujoco.mj_geomDistance(m,collision_data,g,table,minimum_gap,None)>=minimum_gap for g in robotgeoms)
import itertools
preshape_path=None;preshape_order=None
for order in itertools.permutations([[0],[1,2],[3,4],[5,6]]):
 current=initial.copy();candidate_path=[];valid=True
 for group in order:
  nxt=current.copy();nxt[ha[group]]=preshape[ha[group]];segment=[current*(1-u)+nxt*u for u in np.linspace(0,1,91)]
  if not all(valid_state(st) for st in segment):valid=False;break
  candidate_path.extend(segment);current=nxt
 if valid:preshape_path=candidate_path;preshape_order=order;break
assert preshape_path is not None,'no collision-free finger group order'
print('preshape_order',preshape_order,flush=True)

# Reverse a short approach from the collision-free candidate.
d.qpos[:]=preshape;q=goal[qa].copy();p,R=ik.fk(q);approach_axis=np.array(cfg.escape,dtype=float) if cfg.escape is not None else np.r_[p[:2]-d.xpos[m.body('chaleira').id,:2],0.];approach_axis/=np.linalg.norm(approach_axis);lp=ik.d.xpos[left].copy();lr=ik.d.xmat[left].reshape(3,3).copy();approach=[];rows=[]
for distance in np.linspace(0,.08,41):
 target=p+approach_axis*distance;previous=q.copy()
 def residual(x):
  pp,rr=ik.fk(x);pl=ik.d.xpos[left];return np.r_[(pp-target)*1000,Rotation.from_matrix(rr@R.T).as_rotvec()*10,(pl-lp)*500,Rotation.from_matrix(ik.d.xmat[left].reshape(3,3)@lr.T).as_rotvec()*.3,(x-previous)*.05,anatomy.penalty(ik.d)]
 fit=least_squares(residual,np.clip(q,bounds[:,0]+1e-9,bounds[:,1]-1e-9),bounds=bounds.T,max_nfev=120);q=fit.x;pp,rr=ik.fk(q);state=preshape.copy();state[qa]=q;ep=float(np.linalg.norm(pp-target));er=float(np.rad2deg(np.linalg.norm(Rotation.from_matrix(rr@R.T).as_rotvec())));rows.append({'distance_m':float(distance),'error_m':ep,'orientation_error_deg':er,'collision_free':valid_state(state,.0095)});approach.append(state)
 if ep>.002 or er>30 or not rows[-1]['collision_free']:
  (out/'report.json').write_text(json.dumps({'model':'gpt-6-astra','pass':False,'failure':'reverse approach invalid','approach':rows},indent=2));print(rows[-1]);raise SystemExit(0)
pre=approach[-1][qa].copy();raised=preshape.copy();roll=m.jnt_qposadr[m.joint('left_shoulder_roll_joint').id];elbow=m.jnt_qposadr[m.joint('left_elbow_joint').id];raise1=raised.copy();raise1[roll]=1.2;raised[roll]=1.2;raised[elbow]=preshape[elbow]
if cfg.direct_start:raise1=preshape.copy();raised=preshape.copy()
raise_path=[]
for a,b in [(preshape,raise1),(raise1,raised)]:
 for t in np.linspace(0,1,max(2,int(np.max(np.abs(b[qa]-a[qa]))/.02)+2)):
  state=a*(1-t)+b*t
  if not valid_state(state):
   (out/'report.json').write_text(json.dumps({'model':'gpt-6-astra','pass':False,'failure':'raise posture collision','checks':checks},indent=2));print('raise collision',dict(zip(names,state[qa]))),print([(m.body(int(m.geom_bodyid[c.geom1])).name,m.body(int(m.geom_bodyid[c.geom2])).name,float(c.dist)) for c in collision_data.contact if c.dist<0]),print([(m.geom(g).name,float(mujoco.mj_geomDistance(m,collision_data,g,table,1,None))) for g in robotgeoms if mujoco.mj_geomDistance(m,collision_data,g,table,1,None)<.025]);raise SystemExit(0)
  raise_path.append(state)
start=raised[qa].copy();start=np.clip(start,bounds[:,0],bounds[:,1])
def valid(q):
 if np.any(q<bounds[:,0]-1e-8) or np.any(q>bounds[:,1]+1e-8):return False
 state=preshape.copy();state[qa]=q;return valid_state(state)
def edge(a,b):return all(valid(a*(1-t)+b*t) for t in np.linspace(0,1,max(2,int(np.max(np.abs(b-a))/.03)+2)))
assert valid(start) and valid(pre);rng=np.random.default_rng(20260920);trees=[([start],[-1]),([pre],[-1])];found=[start,pre] if edge(start,pre) else None;begin=time.monotonic();iteration=-1

def extend(tree,target):
 nodes,parents=tree;i=int(np.argmin(np.linalg.norm(np.array(nodes)-target,axis=1)));q=nodes[i];dist=np.linalg.norm(target-q);new=q+(target-q)*min(1,.25/max(dist,1e-9))
 if not edge(q,new):return None
 nodes.append(new);parents.append(i);return len(nodes)-1
for iteration in range(7000):
 if found is not None:break
 side=iteration%2;A=trees[side];B=trees[1-side];target=B[0][0] if rng.random()<.25 else rng.uniform(bounds[:,0],bounds[:,1]);ia=extend(A,target)
 if ia is not None:
  for _ in range(100):
   ib=extend(B,A[0][ia])
   if ib is None:break
   if np.linalg.norm(B[0][ib]-A[0][ia])<1e-7:
    ends=[None,None];ends[side]=ia;ends[1-side]=ib;parts=[]
    for tree,index in zip(trees,ends):
     arr=[]
     while index!=-1:arr.append(tree[0][index]);index=tree[1][index]
     parts.append(arr[::-1])
    found=parts[0]+parts[1][::-1];break
 if iteration%100==0:print({'iteration':iteration,'nodes':[len(x[0]) for x in trees],'seconds':time.monotonic()-begin},flush=True)
 if time.monotonic()-begin>120:break
if found is not None:
 path=[found[0]];i=0
 while i<len(found)-1:
  for j in range(len(found)-1,i,-1):
   if edge(found[i],found[j]):break
  path.append(found[j]);i=j
 full=preshape_path.copy();phases=['preshape']*len(full);full+=raise_path;phases+=['settle_preshape' if cfg.direct_start else 'raise_above_table']*len(raise_path)
 for a,b in zip(path,path[1:]):
  for t in np.linspace(0,1,max(2,int(np.max(np.abs(b-a))/.015)+2)):
   state=preshape.copy();state[qa]=a*(1-t)+b*t;full.append(state);phases.append('connect_to_handle')
 full+=approach[::-1];phases+=['approach_handle']*len(approach);np.savez_compressed(out/'path.npz',qpos=full,phases=phases)
result={'model':'gpt-6-astra','source':str(source),'scene':r['scene'],'candidate_source':str(cfg.candidate),'candidate_index':choice,'closed_hand_q':closed_hand.tolist(),'open_factor':cfg.open_factor,'escape_axis':approach_axis.tolist(),'direct_start':cfg.direct_start,'elbow_forward_m':candidate.get('elbow_forward_m'),'elbow_drop_m':candidate.get('elbow_drop_m'),'pass':found is not None,'checks':checks,'seconds':time.monotonic()-begin,'iterations':iteration+1,'tree_sizes':[len(x[0]) for x in trees],'scope':'sampled empty-hand path; palm orientation soft preference up to30deg on retreat, endpoint grasp pose retained; no motor load support validated','joint_names':names,'approach':rows};(out/'report.json').write_text(json.dumps(result,indent=2));print({k:v for k,v in result.items() if k!='approach'})
