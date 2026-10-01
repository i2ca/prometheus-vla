"""Bidirectional RRT from dynamically raised hand to handle pregrasp."""
import json,shutil,sys,time
from pathlib import Path
import numpy as np,mujoco
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'g1-cup-grasp/scripts'))
from g1_sim import ARM_JOINTS
out=Path('results/loadable-connect-004');out.mkdir(exist_ok=False);shutil.copy2(__file__,out/Path(__file__).name)
m=mujoco.MjModel.from_xml_path(str(Path('results/handle-layout-search-003/layout-05/scene.xml').resolve()));d=mujoco.MjData(m);initial=np.load('results/handle-layout-search-003/layout-05/initial.npy');raised=np.load('results/raise-from-rest-002/final-qpos.npy');
for joint in range(m.njnt):
 if m.jnt_type[joint]==mujoco.mjtJoint.mjJNT_HINGE:initial[m.jnt_qposadr[joint]]=raised[m.jnt_qposadr[joint]]
d.qpos[:]=initial;mujoco.mj_forward(m,d)
names=['waist_yaw_joint']+[n.replace('right_','left_') for n in ARM_JOINTS];qa=np.array([m.jnt_qposadr[m.joint(n).id] for n in names]);bounds=np.array([m.jnt_range[m.joint(n).id] for n in names]);bounds[0]=[-.5,.5];bounds[-3:]=np.deg2rad([[-60,60],[-45,45],[-30,30]])
pre=np.load('results/finger-release-005/path.npz')['qpos'][0];goal=pre[qa].copy();start=initial[qa].copy();preshape=initial.copy()
for j in range(m.njnt):
 if m.joint(j).name.startswith('left_hand'):preshape[m.jnt_qposadr[j]]=pre[m.jnt_qposadr[j]]
table=m.geom('tampo').id;robotgeom=[g for g in range(m.ngeom) if (m.geom_contype[g] or m.geom_conaffinity[g]) and m.body(int(m.geom_bodyid[g])).name.startswith(('left_hand','left_wrist','left_elbow','right_hand','right_wrist','right_elbow'))]
checks=0
right_pitch=m.jnt_qposadr[m.joint('right_shoulder_pitch_joint').id]
def compensate_other_arm(state):
 state[right_pitch]=initial[right_pitch]+.8*max(0.,state[qa[0]])
 return state

def valid_state(state):
 global checks
 checks+=1;d.qpos[:]=state;mujoco.mj_forward(m,d)
 for ct in d.contact:
  if ct.dist>=0:continue
  bs=[m.body(int(m.geom_bodyid[g])).name for g in [ct.geom1,ct.geom2]]
  if any(b.startswith(('left_','right_','torso','waist','pelvis','head')) for b in bs):return False
 return all(mujoco.mj_geomDistance(m,d,g,table,.04,None)>=.04 for g in robotgeom)
def valid(q):
 if np.any(q<bounds[:,0]-1e-9) or np.any(q>bounds[:,1]+1e-9):return False
 state=preshape.copy();state[qa]=q;return valid_state(compensate_other_arm(state))
def edge(a,b):
 return all(valid(a*(1-t)+b*t) for t in np.linspace(0,1,max(2,int(np.max(np.abs(b-a))/.035)+2)))
assert all(valid_state(initial*(1-t)+preshape*t) for t in np.linspace(0,1,61)),'preshape collision'
assert valid(start),'invalid start'
assert valid(goal),'invalid pregrasp'
rng=np.random.default_rng(20260919);trees=[([start],[-1]),([goal],[-1])];found=None;begin=time.monotonic()
def extend(tree,target):
 nodes,parents=tree;nearest=int(np.argmin(np.linalg.norm(np.array(nodes)-target,axis=1)));q=nodes[nearest];dist=np.linalg.norm(target-q);new=q+(target-q)*min(1,.22/max(dist,1e-9))
 if not edge(q,new):return None
 nodes.append(new);parents.append(nearest);return len(nodes)-1
for iteration in range(6000):
 side=iteration%2;A=trees[side];B=trees[1-side];target=trees[1-side][0][0] if rng.random()<.2 else rng.uniform(bounds[:,0],bounds[:,1]);ia=extend(A,target)
 if ia is None:continue
 for _ in range(100):
  ib=extend(B,A[0][ia])
  if ib is None:break
  if np.linalg.norm(B[0][ib]-A[0][ia])<1e-7:
   ends=[None,None];ends[side]=ia;ends[1-side]=ib;paths=[]
   for tree,idx in zip(trees,ends):
    arr=[]
    while idx!=-1:arr.append(tree[0][idx]);idx=tree[1][idx]
    paths.append(arr[::-1])
   found=paths[0]+paths[1][::-1];break
 if found is not None:break
 if time.monotonic()-begin>90:break
if found is not None:
 # Greedy shortcut, still checking every segment.
 path=[found[0]];i=0
 while i<len(found)-1:
  for j in range(len(found)-1,i,-1):
   if edge(found[i],found[j]):break
  path.append(found[j]);i=j
 full=[initial*(1-t)+preshape*t for t in np.linspace(0,1,61)]
 for a,b in zip(path,path[1:]):
  for t in np.linspace(0,1,max(2,int(np.max(np.abs(b-a))/.015)+2)):
   state=preshape.copy();state[qa]=a*(1-t)+b*t;full.append(compensate_other_arm(state))
 np.savez_compressed(out/'path.npz',qpos=full)
report={'model':'gpt-6-astra','pass':found is not None,'iterations':iteration+1,'checks':checks,'seconds':time.monotonic()-begin,'tree_sizes':[len(t[0]) for t in trees],'wrist_task_bounds_deg':np.rad2deg(bounds[-3:]).tolist(),'scope':'sampled geometric path from raised posture to pregrasp, not dynamic validation or grasp','scene':'results/handle-layout-search-003/layout-05/scene.xml'};(out/'report.json').write_text(json.dumps(report,indent=2));print(report)
