"""Offline transport of measured handle grasp; dynamic validation required."""
import json,sys,shutil
from pathlib import Path
import numpy as np,mujoco
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'g1-cup-grasp/scripts'))
from g1_sim import G1Sim,ARM_JOINTS
from kinematics import ArmIK
out=Path('results/handle-transport-plan-012');out.mkdir(exist_ok=False);shutil.copy2(__file__,out/Path(__file__).name)
r=json.loads(Path('results/loadable-dynamics-016/report.json').read_text());s=G1Sim(r['scene']);m,d=s.m,s.d;base=np.load('results/loadable-dynamics-016/final-qpos.npy');d.qpos[:]=base;mujoco.mj_forward(m,d)
ik=ArmIK(s,['waist_yaw_joint','waist_roll_joint','waist_pitch_joint']+[n.replace('right_','left_') for n in ARM_JOINTS],palm='left_wrist_yaw_link');ik.bounds=ik.bounds.copy();ik.bounds[0]=[-.5,.5];ik.bounds[1:3]=np.deg2rad([[-12,12],[-12,12]]);ik.bounds[-3:]=np.deg2rad([[-60,60],[-45,45],[-30,30]])
q=base[ik.qa].copy();qstart=q.copy();qend=np.deg2rad([18.7,12,-11.7,-26.8,64,-35.2,29.3,-44.7,45,-29.9]);p,R=ik.fk(q);b=m.body('chaleira').id;c=d.xpos[b].copy();C=d.xmat[b].reshape(3,3).copy();local=R.T@(c-p);local_R=R.T@C;spout_local=np.array([-.091012658,.00010992,.209200575]);spout=c+C@spout_local;spout_palm=local+local_R@spout_local;filter_center=d.xpos[m.body('coador').id]+d.xmat[m.body('coador').id].reshape(3,3)@np.array([0,0,.205]);end=filter_center.copy();end[:2]+=[-.02,.04];end[2]=filter_center[2]+.04;qa=m.jnt_qposadr[m.joint('chaleira_livre').id];right=m.jnt_qposadr[m.joint('right_shoulder_pitch_joint').id];rows=[];states=[];bad=None;avoid=set()
metal=m.geom('chaleira_hot_body').id;env=[g for g in range(m.ngeom) if m.geom_contype[g] and m.body(int(m.geom_bodyid[g])).name in ['coador','copo','pote','tampa','scoop','base_eletrica','mesa']]
for u in np.linspace(0,1,101):
 if u==0:
  states.append(base.copy());rows.append({'u':0.,'position_error_m':0.,'kettle_tilt_deg':float(np.rad2deg(np.arccos(C[2,2]))),'contacts':[]});continue
 for g in env:
  if mujoco.mj_geomDistance(m,d,metal,g,.04,None)<.04:avoid.add(tuple(sorted([metal,g])))
 goal=spout*(1-u)+end*u;previous=q.copy();preferred=R
 def put(x):
  pp,rr=ik.fk(x);d.qpos[:]=base;d.qpos[ik.qa]=x;d.qpos[right]=.35+.8*max(0,x[0]);d.qpos[qa:qa+3]=pp+rr@local;mujoco.mju_mat2Quat(d.qpos[qa+3:qa+7],(rr@local_R).ravel());mujoco.mj_forward(m,d);return pp,rr
 for retry in range(8):
  pairs=sorted(avoid)
  def residual(x):
   pp,rr=put(x);gaps=[min(0,mujoco.mj_geomDistance(m,d,a,b,.03,None)-(.012 if metal in [a,b] else .001)) for a,b in pairs];return np.r_[(pp+rr@spout_palm-goal)*1000,np.cross((rr@local_R)[:,2],C[:,2]*(1-u)+np.array([0,0,1])*u)*10,Rotation.from_matrix(rr@preferred.T).as_rotvec(),(x-previous)*.03,(x-(qstart*(1-u)+qend*u))*.3,max(0,np.arccos(np.clip((rr@local_R)[2,2],-1,1))-np.deg2rad(6))*1000,np.array(gaps)*10000]
  local_bounds=np.stack([np.maximum(ik.bounds[:,0],previous-.035),np.minimum(ik.bounds[:,1],previous+.035)]);fit=least_squares(residual,np.clip(q,local_bounds[0]+1e-9,local_bounds[1]-1e-9),bounds=local_bounds,max_nfev=120);q=fit.x;pp,rr=put(q);ep=np.linalg.norm(pp+rr@spout_palm-goal);er=np.arccos(np.clip((rr@local_R)[2,2],-1,1));contacts=[]
  for ct in d.contact:
   if ct.dist>=0:continue
   bn=[m.body(int(m.geom_bodyid[g])).name for g in [ct.geom1,ct.geom2]];gn=[m.geom(int(g)).name or '' for g in [ct.geom1,ct.geom2]]
   allowed=any(n.startswith('left_hand') for n in bn) and any(n.startswith('handle_col') for n in gn)
   if not allowed and ('chaleira' in bn or any(n.startswith(('left_','right_','torso','pelvis','waist','head')) for n in bn)):
    contacts.append(bn);avoid.add(tuple(sorted([int(ct.geom1),int(ct.geom2)])))
  if not contacts:break
 row={'u':float(u),'position_error_m':float(ep),'kettle_tilt_deg':float(np.rad2deg(er)),'contacts':contacts};rows.append(row);states.append(d.qpos.copy())
 if ep>(.002 if u==1 else .03) or er>np.deg2rad(10) or contacts:bad=row;break
np.savez_compressed(out/'path.npz',qpos=states)
report={'model':'gpt-6-astra','scene':r['scene'],'pass':bad is None,'failure':bad,'scope':'offline rigid grasp planning only; does not prove transport','spout_local_m':spout_local.tolist(),'spout_local_status':'extreme visual mesh vertex, approximate lip location','start_spout':spout.tolist(),'target_spout':end.tolist(),'rows':rows};(out/'report.json').write_text(json.dumps(report,indent=2));print({k:v for k,v in report.items() if k!='rows'})
