"""Plan kettle tilt about spout from measured suspended grasp, offline only."""
import argparse,json,sys,shutil
from pathlib import Path
import numpy as np,mujoco
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'g1-cup-grasp/scripts'))
from g1_sim import G1Sim,ARM_JOINTS
from kinematics import ArmIK
from elbow_anatomy import ElbowAnatomy
ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--torso',action='store_true');ap.add_argument('--max-angle',type=int,default=80);ap.add_argument('--align-angle',type=float,default=55);ap.add_argument('--guide-error-deg',type=float,default=8)
# Medido na trajetoria do despejo original: cotovelo de 49 a 114 mm ACIMA do
# ombro, abducao de 135 a 180 graus. O anatomy so limita a flexao (5 a 145 graus)
# e nao diz nada sobre para onde o cotovelo aponta, entao o IK servia o cafe com
# o braco aberto para o lado e o cotovelo pendurado no alto. Estes dois termos
# puxam o cotovelo para baixo do ombro e para perto do corpo.
ap.add_argument('--elbow-posture-weight',type=float,default=0.)
ap.add_argument('--elbow-abduction-deg',type=float,default=70.)
# braco ocioso com juntas fixas e sem prender a mao dele no mundo (ver plan_kettle_grasp_v3)
ap.add_argument('--fix-idle-arm',action='store_true');a=ap.parse_args();a.out.mkdir(exist_ok=False);shutil.copy2(__file__,a.out/Path(__file__).name);shutil.copy2('scripts/elbow_anatomy.py',a.out/'elbow_anatomy.py')
r=json.loads((a.source/'report.json').read_text());assert r['pass'];s=G1Sim(r['scene']);m,d=s.m,s.d;ck=np.load(a.source/'continuation.npz')
for name in ['body_mass','body_ipos','body_inertia','body_iquat','bvh_aabb']:getattr(m,name)[:]=ck[name]
mujoco.mj_setConst(m,mujoco.MjData(m));mujoco.mj_setState(m,d,ck['integration'],mujoco.mjtState.mjSTATE_INTEGRATION);mujoco.mj_forward(m,d);base=d.qpos.copy();other=m.body('right_wrist_yaw_link').id;other_goal=d.xpos[other].copy()
names=['waist_yaw_joint','waist_roll_joint','waist_pitch_joint']+[n.replace('right_','left_') for n in ARM_JOINTS]+list(ARM_JOINTS);ik=ArmIK(s,names,palm='left_wrist_yaw_link');ik.bounds=ik.bounds.copy();ik.bounds[0]=[-.7,.7]
ik.bounds[1:3]=np.deg2rad([[-12,12],[-12,12]])
if a.fix_idle_arm:ik.bounds[10:17,0]=base[ik.qa][10:17]-1e-4;ik.bounds[10:17,1]=base[ik.qa][10:17]+1e-4
for j in [7,14]:ik.bounds[j:j+3]=np.deg2rad([[-60,60],[-45,45],[-30,30]])
anatomy=ElbowAnatomy(m);anatomy.bound_search(m,ik.joint_names,ik.bounds)
q=np.r_[base[ik.qa],0.];bounds=np.vstack([ik.bounds,[-np.pi,np.pi]]);p,R=ik.fk(q[:-1]);b=m.body('chaleira').id;c=d.xpos[b].copy();C=d.xmat[b].reshape(3,3).copy();local=R.T@(c-p);local_R=R.T@C;lip=np.array([-.091012658,.00010992,.209200575]);start_spout=c+C@lip;filter_center=d.xpos[m.body('coador').id]+d.xmat[m.body('coador').id].reshape(3,3)@np.array([0,0,.205]);end_spout=filter_center.copy();end_spout[2]=filter_center[2]+.04;_torso,_ombro,_cotovelo=[m.body(n).id for n in ['torso_link','left_shoulder_roll_link','left_elbow_link']]

def postura():
 if not a.elbow_posture_weight:return []
 v=d.xmat[_torso].reshape(3,3).T@(d.xpos[_cotovelo]-d.xpos[_ombro])
 return [max(0.,v[2])*10.,max(0.,np.rad2deg(np.arctan2(abs(v[1]),-v[2]))-a.elbow_abduction_deg)/90.]
jarqa=m.jnt_qposadr[m.joint('chaleira_livre').id];right=m.jnt_qposadr[m.joint('right_shoulder_pitch_joint').id];metal=m.geom('chaleira_hot_body').id;env=[g for g in range(m.ngeom) if m.geom_contype[g] and m.body(int(m.geom_bodyid[g])).name in ['coador','copo','pote','tampa','scoop','base_eletrica','mesa']];avoid=set();rows=[];states=[];bad=None
for angle in np.linspace(0,a.max_angle,a.max_angle+1):
 blend=min(1,angle/a.align_angle);spout=start_spout*(1-blend)+end_spout*blend;Cr=C@Rotation.from_rotvec([0,-np.deg2rad(angle),0]).as_matrix();Rgoal=Cr@local_R.T;pgoal=spout-Cr@lip-Rgoal@local;previous=q.copy()
 for g in env:
  if mujoco.mj_geomDistance(m,d,metal,g,.04,None)<.04:avoid.add(tuple(sorted([metal,g])))
 def put(x):
  pp,rr=ik.fk(x[:-1]);d.qpos[:]=base;d.qpos[ik.qa]=x[:-1];d.qpos[jarqa:jarqa+3]=pp+rr@local;mujoco.mju_mat2Quat(d.qpos[jarqa+3:jarqa+7],(rr@local_R).ravel());mujoco.mj_forward(m,d);return pp,rr
 for retry in range(8):
  pairs=sorted(avoid)
  def residual(x):
   pp,rr=put(x);CC=Rotation.from_rotvec([0,0,x[-1]]).as_matrix()@Cr;Rgoal=CC@local_R.T;pgoal=spout-CC@lip-Rgoal@local;gaps=[min(0,mujoco.mj_geomDistance(m,d,g,h,.03,None)-(.012 if metal in [g,h] else .002)) for g,h in pairs];return np.r_[(pp+rr@(local+local_R@lip)-spout)*1000,Rotation.from_matrix(rr@Rgoal.T).as_rotvec()*30,(x-previous)*.03,np.array(gaps)*10000,(d.xpos[other]-other_goal)*(0 if a.fix_idle_arm else 100),np.array(postura())*a.elbow_posture_weight,anatomy.penalty(d)]
  fit=least_squares(residual,np.clip(q,bounds[:,0]+1e-9,bounds[:,1]-1e-9),bounds=bounds.T,max_nfev=160);q=fit.x;pp,rr=put(q);CC=Rotation.from_rotvec([0,0,q[-1]]).as_matrix()@Cr;Rgoal=CC@local_R.T;pgoal=spout-CC@lip-Rgoal@local;ep=np.linalg.norm(pp+rr@(local+local_R@lip)-spout);er=np.linalg.norm(Rotation.from_matrix(rr@Rgoal.T).as_rotvec());contacts=[]
  for ct in d.contact:
   if ct.dist>=0:continue
   bn=[m.body(int(m.geom_bodyid[g])).name for g in [ct.geom1,ct.geom2]];gn=[m.geom(int(g)).name or '' for g in [ct.geom1,ct.geom2]]
   allowed=any(n.startswith('left_hand') for n in bn) and any(n.startswith('handle_col') for n in gn)
   if not allowed and ('chaleira' in bn or any(n.startswith(('left_','right_','torso','pelvis','waist','head')) for n in bn)):
    contacts.append(bn);avoid.add(tuple(sorted([int(ct.geom1),int(ct.geom2)])))
  if not contacts:break
 row={'elbow_flexion_deg':anatomy.angles(d).tolist(),'angle_deg':float(angle),'world_yaw_adjustment_deg':float(np.rad2deg(q[-1])),'position_error_m':float(ep),'orientation_error_deg':float(np.rad2deg(er)),'contacts':contacts,'spout_m':spout.tolist()};rows.append(row);states.append(d.qpos.copy())
 if ep>.002 or er>np.deg2rad(a.guide_error_deg) or contacts or not anatomy.valid(d,tolerance_deg=.2):bad=row;break
np.savez_compressed(a.out/'path.npz',qpos=states)
report={'model':'gpt-6-astra','scene':r['scene'],'source':str(a.source),'source_passed_its_full_gate':r['pass'],'pass':bad is None,'failure':bad,'scope':'offline rigid grasp tilt, no physical pour or water transfer validation','torso_freedom':a.torso,'orientation_guidance_tolerance_deg':a.guide_error_deg,'spout_alignment_angle_deg':a.align_angle,'joint_names':names,'rows':rows};(a.out/'report.json').write_text(json.dumps(report,indent=2));print({k:v for k,v in report.items() if k!='rows'})
