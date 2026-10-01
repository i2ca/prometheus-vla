"""Plan returning the held spoon to its original table region, offline only.

The final bowl reference follows the original settled spoon height. A physical
controller must confirm support, open the fingers, and withdraw separately.
"""
import argparse,json,sys,shutil
from pathlib import Path
import mujoco,numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'g1-cup-grasp/scripts'))
from g1_sim import G1Sim,ARM_JOINTS
from kinematics import ArmIK
from elbow_anatomy import ElbowAnatomy
from grasp_contact_rules import allowed_tip_contact
ap=argparse.ArgumentParser();ap.add_argument('--hand-clearance',type=float,default=.0095);ap.add_argument('--immersion',type=float,default=.0055);ap.add_argument('--dip-angle',type=float,default=10);ap.add_argument('--scoop-offset-x',type=float,default=0);ap.add_argument('--waist-pitch-max',type=float,default=12);ap.add_argument('--source',type=Path,default=Path('results/spoon-transport-006'));ap.add_argument('--return-route',action='store_true');ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(exist_ok=False);shutil.copy2(__file__,a.out/Path(__file__).name);shutil.copy2('scripts/elbow_anatomy.py',a.out/'elbow_anatomy.py');shutil.copy2('scripts/grasp_contact_rules.py',a.out/'grasp_contact_rules.py')
source=a.source;r=json.loads((source/'report.json').read_text());assert r['pass'] and not (source/'INVALIDATED.json').exists(),'physical source not accepted';hand_side=r.get('hand','left');other_side='left' if hand_side=='right' else 'right';s=G1Sim(r['scene']);m,d=s.m,s.d;ck=np.load(source/'continuation.npz');mujoco.mj_setState(m,d,ck['integration'],mujoco.mjtState.mjSTATE_INTEGRATION);mujoco.mj_forward(m,d)
names=['waist_yaw_joint','waist_roll_joint','waist_pitch_joint']+[n.replace('right_',hand_side+'_') for n in ARM_JOINTS]+[n.replace('right_',other_side+'_') for n in ARM_JOINTS];ik=ArmIK(s,names,palm=hand_side+'_wrist_yaw_link');bounds=ik.bounds.copy();bounds[0]=[-.7,.7];bounds[1:3]=np.deg2rad([[-12,12],[-12,12]])
a.waist_pitch_max=max(a.waist_pitch_max,r.get('waist_pitch_max_deg',12))
bounds[2,1]=min(ik.bounds[2,1],np.deg2rad(a.waist_pitch_max))
anatomy=ElbowAnatomy(m);anatomy.bound_search(m,ik.joint_names,bounds)
for j in [7,14]:bounds[j:j+3]=np.deg2rad([[-60,60],[-45,45],[-30,30]])
spoon=m.body('scoop').id;palm=m.body(hand_side+'_wrist_yaw_link').id;other=m.body(other_side+'_wrist_yaw_link').id;pot=m.body('pote').id;sj=m.body_jntadr[spoon];sa=m.jnt_qposadr[sj];R=d.xmat[palm].reshape(3,3);local=R.T@(d.xpos[spoon]-d.xpos[palm]);localR=R.T@d.xmat[spoon].reshape(3,3);bowl=np.array([-.045,0,.003]);p0=d.xpos[spoon]+d.xmat[spoon].reshape(3,3)@bowl;op=d.xpos[other].copy();q=d.qpos[ik.qa].copy();q0=q.copy();states=[];results=[]
# Explicit slow motor execution must subsequently validate every segment.
goals=[np.array([.40,-.25,1.02]),np.array([.26,-.30,.90]),np.array([.26,-.30,.78]),np.array([.26,-.30,.7496])]
angles=[0,0,a.dip_angle,a.dip_angle]
# Support height from the real collision meshes, relative to the tracked bowl.
placement_R=Rotation.from_euler('ZY',[180,-a.dip_angle],degrees=True).as_matrix()
lowest=1.
for g in range(m.ngeom):
 if m.geom_bodyid[g]!=spoon or not m.geom_contype[g]:continue
 assert m.geom_type[g]==mujoco.mjtGeom.mjGEOM_MESH
 mesh=m.geom_dataid[g];v=m.mesh_vert[m.mesh_vertadr[mesh]:m.mesh_vertadr[mesh]+m.mesh_vertnum[mesh]]
 gr=np.zeros(9);mujoco.mju_quat2Mat(gr,m.geom_quat[g]);local_vertices=v@gr.reshape(3,3).T+m.geom_pos[g]
 lowest=min(lowest,float(np.min((local_vertices-bowl)@placement_R[2])))
table_id=m.geom('tampo').id;top=d.geom_xpos[table_id,2]+m.geom_size[table_id,2]
goals[-1][2]=top-lowest-.00002

table=m.geom('tampo').id
handgeoms=[g for g in range(m.ngeom) if m.geom_contype[g] and m.body(int(m.geom_bodyid[g])).name.startswith(('left_hand','right_hand','left_wrist','right_wrist'))]
rng=np.random.default_rng(19)
for wi,goal in enumerate(goals):
 goal=np.array(goal);Rtarget=Rotation.from_euler('ZY',[180,-angles[wi]],degrees=True).as_matrix();avoid=set();seed=q.copy()
 def evaluate(x):
  pp,rr=ik.fk(x);objp=pp+rr@local;objR=rr@localR;ik.d.qpos[sa:sa+3]=objp;quat=Rotation.from_matrix(objR).as_quat();ik.d.qpos[sa+3:sa+7]=quat[[3,0,1,2]];mujoco.mj_forward(m,ik.d);return objp+objR@bowl,objR
 best=None
 for retry in range(30):
  pairs=sorted(avoid)
  def residual(x):
   bp,rr=evaluate(x);gaps=[min(0,mujoco.mj_geomDistance(m,ik.d,g,h,.02,None)-.004) for g,h in pairs]
   return np.r_[(bp-goal)*1000,(rr[:,2]-Rtarget[:,2])*50,(ik.d.xpos[other]-op)*30,(x-seed)*.2,x[np.r_[7:10,14:17]]*.15,np.array(gaps)*3000,np.array([min(0,mujoco.mj_geomDistance(m,ik.d,g,table,.05,None)-a.hand_clearance) for g in handgeoms])*5000,anatomy.penalty(ik.d)]
  fit=least_squares(residual,np.clip(q,bounds[:,0]+1e-8,bounds[:,1]-1e-8),bounds=bounds.T,max_nfev=100);q=fit.x;bp,rr=evaluate(q);bad=[]
  for c in ik.d.contact:
   if c.dist>=0:continue
   gs=[int(c.geom1),int(c.geom2)];bn=[m.body(int(m.geom_bodyid[g])).name for g in gs];allowed=(wi==len(goals)-1 and spoon in m.geom_bodyid[gs] and table in gs and c.dist>=-.0002) or ('scoop' in bn and any(n.startswith(hand_side+'_hand') for n in bn)) or allowed_tip_contact(bn,c.dist,hand_side,r.get('allow_tip_contact',False))
   if not allowed and (any(n.startswith(('left_','right_','torso','waist','pelvis','head')) for n in bn) or 'scoop' in bn):bad.append(bn);avoid.add(tuple(sorted(gs)))
  table_gap=min(mujoco.mj_geomDistance(m,ik.d,g,table,1,None) for g in handgeoms)
  score=5000*max(0,a.hand_clearance-table_gap)+100*np.linalg.norm(bp-goal)+10*np.linalg.norm(rr[:,2]-Rtarget[:,2])+len(bad)
  if best is None or score<best[0]:best=(score,q.copy())
  if not bad and table_gap>=a.hand_clearance-.0015 and np.linalg.norm(bp-goal)<.003 and np.dot(rr[:,2],Rtarget[:,2])>np.cos(np.deg2rad(5)):break
  if retry%3==2:
   q=seed.copy();q[:10]+=rng.normal(0,.7,10);q=np.clip(q,bounds[:,0]+1e-6,bounds[:,1]-1e-6)
  elif not bad:q=best[1].copy()
 else:
  q=best[1];bp,rr=evaluate(q);bad=[]
  for c in ik.d.contact:
   gs=[int(c.geom1),int(c.geom2)];bn=[m.body(int(m.geom_bodyid[g])).name for g in gs]
   if c.dist<0 and not ((wi==len(goals)-1 and spoon in m.geom_bodyid[gs] and table in gs and c.dist>=-.0002) or ('scoop' in bn and any(n.startswith(hand_side+'_hand') for n in bn)) or allowed_tip_contact(bn,c.dist,hand_side,r.get('allow_tip_contact',False))) and (any(n.startswith(('left_','right_','torso','waist','pelvis','head')) for n in bn) or 'scoop' in bn):bad.append(bn)
 tilt=float(np.rad2deg(np.arccos(np.clip(rr[2,2],-1,1))));error=float(np.linalg.norm(bp-goal));table_gap=min(mujoco.mj_geomDistance(m,ik.d,g,table,1,None) for g in handgeoms);passed=anatomy.valid(ik.d,tolerance_deg=.2) and table_gap>=a.hand_clearance-.0015 and error<.003 and np.dot(rr[:,2],Rtarget[:,2])>np.cos(np.deg2rad(5)) and not bad
 row={'table_gap_m':table_gap,'elbow_flexion_deg':anatomy.angles(ik.d).tolist(),'waypoint':wi,'pass':bool(passed),'bowl_goal':goal.tolist(),'bowl_actual':bp.tolist(),'tilt_deg':tilt,'error_m':error,'contacts':bad,'q':q.tolist(),'spoon_R':rr.tolist(),'other_palm':ik.d.xpos[other].tolist()};results.append(row);states.append(ik.d.qpos.copy());print(row,flush=True)
 if not passed:break
(a.out/'report.json').write_text(json.dumps({'model':'gpt-6-astra','hand_clearance_m':a.hand_clearance,'hand':hand_side,'source':str(source),'scene':r['scene'],'task':'place_spoon','scope':'offline waypoint feasibility only','pass':all(x['pass'] for x in results) and len(results)==len(goals),'immersion_m':a.immersion,'dip_angle_deg':a.dip_angle,'scoop_offset_x_m':a.scoop_offset_x,'waist_pitch_max_deg':a.waist_pitch_max,'joint_names':names,'results':results},indent=2));np.savez_compressed(a.out/'waypoints.npz',qpos=states)
