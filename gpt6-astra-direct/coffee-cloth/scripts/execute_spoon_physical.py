"""Physical spoon reach/pinch/lift from wet kettle checkpoint, motors only."""
import sys,json,argparse,shutil
from pathlib import Path
import numpy as np,mujoco
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from dataclasses import asdict
from collections import deque
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'g1-cup-grasp/scripts'))
from g1_sim import G1Sim,ARM_JOINTS,HAND_JOINTS,load_seed
from kinematics import ArmIK
from elbow_anatomy import ElbowAnatomy
from finger_contact_servo import advance as advance_fingers
from spoon_grasp_servo import SpoonGraspServo
from liquid_mass import LiquidMassCoupler
from kettle_liquid import KettleWater
from brew_state import BrewState
ap=argparse.ArgumentParser();ap.add_argument('--scene',type=Path);ap.add_argument('--lift-duration',type=float,default=6.);ap.add_argument('--tip-contact-force-limit',type=float,default=.5);ap.add_argument('--contact-force-target',type=float,default=.3);ap.add_argument('--payload-grip-threshold',action='store_true');ap.add_argument('--friction-compensation',action='store_true');ap.add_argument('--geometry-grip',action='store_true');ap.add_argument('--third-link',choices=['middle_0','middle_1','index_0'],default='middle_1');ap.add_argument('--three-contact-grasp','--tripod',dest='tripod',action='store_true');ap.add_argument('--max-grasp-rotation',type=float,default=180);ap.add_argument('--grasp-force-blend',type=float,default=1.0);ap.add_argument('--grasp-wrench',action='store_true');ap.add_argument('--lift-target-tilt',type=float,default=0);ap.add_argument('--max-empty-tilt',type=float,default=10);ap.add_argument('--palm-lift',type=float,default=.10);ap.add_argument('--allow-tip-contact',action='store_true');ap.add_argument('--hand',choices=['left','right'],default='left');ap.add_argument('--object-feedback',action='store_true');ap.add_argument('--damped-arms',action='store_true');ap.add_argument('--source',type=Path);ap.add_argument('--adaptive-grip',action='store_true');ap.add_argument('--out',type=Path,required=True);ap.add_argument('--plan',type=Path,default=Path('results/spoon-connect-003'));ap.add_argument('--reach-only',action='store_true');ap.add_argument('--index-servo',action='store_true');ap.add_argument('--pinch-force',type=float,default=3.5);ap.add_argument('--finger-kp',type=float,default=2.0);ap.add_argument('--normal-pinch',action='store_true');ap.add_argument('--two-finger',action='store_true');ap.add_argument('--close-gain',type=float,default=3);a=ap.parse_args();HAND_JOINTS=[n.replace('right_',a.hand+'_') for n in HAND_JOINTS];other_hand='right' if a.hand=='left' else 'left';pr=json.loads((a.plan/'report.json').read_text());assert pr['pass'];source=a.source or Path(pr['source']);assert not (source/'INVALIDATED.json').exists(),'source invalidated by audit';r=json.loads((source/'report.json').read_text());a.out.mkdir(exist_ok=False)
for p in [Path(__file__),Path('scripts/liquid_mass.py'),Path('scripts/kettle_liquid.py')]+[Path('scripts')/n for n in ['brew_state.py','coffee_grounds.py','grounds_mass.py','kettle_heater.py','elbow_anatomy.py','finger_contact_servo.py','spoon_grasp_wrench.py','spoon_grasp_servo.py']]:shutil.copy2(p,a.out/p.name)
source_scene=r['scene'];r['scene']=str(a.scene.resolve()) if a.scene else source_scene
s=G1Sim(r['scene']);m,d=s.m,s.d;brew=BrewState(m,source);ck=np.load(source/'continuation.npz');wet=json.loads((source/'liquid-state.json').read_text());water=KettleWater(initial_ml=wet['initial_ml']);coupler=LiquidMassCoupler(m,float(ck['dry_kettle_kg']))
for n,v in wet.items():setattr(water,n,v)
for name in ['body_mass','body_ipos','body_inertia','body_iquat','bvh_aabb']:
 assert getattr(m,name).shape==ck[name].shape;getattr(m,name)[:]=ck[name]
mujoco.mj_setConst(m,mujoco.MjData(m));mujoco.mj_setState(m,d,ck['integration'],mujoco.mjtState.mjSTATE_INTEGRATION);mujoco.mj_forward(m,d);s.kp[:]=ck['kp'];s.kd[:]=ck['kd'];target=ck['last_target'].copy();start=float(d.time);dt=m.opt.timestep
if a.damped_arms:
 for n,i in s.act_joint.items():
  if 'waist' in n:s.kd[i]=15
  elif 'shoulder' in n:s.kd[i]=5
  elif 'elbow' in n:s.kd[i]=3
  elif 'wrist' in n:s.kd[i]=2.5
path=np.load(a.plan/'path.npz');poses=path['qpos'];phases=path['phases'];names=pr['joint_names'];ik=ArmIK(s,names,palm=a.hand+'_wrist_yaw_link');bounds=ik.bounds.copy();bounds[0]=[-.7,.7];bounds[1:3]=np.deg2rad([[-12,12],[-12,12]])
anatomy=ElbowAnatomy(m);anatomy.bound_search(m,ik.joint_names,bounds)
for j in [7,14]:bounds[j:j+3]=np.deg2rad([[-60,60],[-45,45],[-30,30]])
ha=m.jnt_qposadr[[m.joint(n).id for n in HAND_JOINTS]];moving=np.r_[ik.qa,ha];durations=np.maximum(.016,np.max(np.abs(np.diff(poses[:,moving],axis=0)),axis=1)/.20);times=np.r_[0,np.cumsum(durations)];end=float(times[-1]);spoon=m.body('scoop').id;palm=m.body(a.hand+'_wrist_yaw_link').id;left=m.body(other_hand+'_wrist_yaw_link').id;pot=m.body('pote').id;jar=m.body('chaleira').id;filter_id=m.body('coador').id;cup=m.body('copo').id;props=[jar,filter_id,cup,pot,m.body('base_eletrica').id,m.body('tampa').id];prop0=d.xpos[props].copy();spoon0=d.xpos[spoon].copy();table=m.geom('tampo').id
handgeoms=[g for g in range(m.ngeom) if m.geom_contype[g] and m.body(int(m.geom_bodyid[g])).name.startswith(('left_hand','left_wrist','right_hand','right_wrist'))]
arm,seed,fps=load_seed(0);gi=int(np.flatnonzero(seed[:,4]>.5)[0]);hrange=m.jnt_range[[m.joint(n).id for n in HAND_JOINTS]];close=np.clip(seed[0]+a.close_gain*(seed[gi]-seed[0]),hrange[:,0],hrange[:,1]);opened=poses[-1,ha].copy()
if a.two_finger:close[5:]=opened[5:];close[0]=opened[0]
from spoon_grasp_wrench import allocate
grasp_relative_R=None;max_grasp_rotation=0.
normal=np.zeros(m.nu);finger_mask=np.array([m.joint(m.actuator_trnid[i,0]).name.startswith(a.hand+'_hand') for i in range(m.nu)]);handle_geoms=[g for g in range(m.ngeom) if m.geom_contype[g] and m.geom_bodyid[g]==spoon and (.020 if a.tripod else .040)<m.geom_pos[g,0]<.059];pinch_tips=[next(g for g in range(m.ngeom) if m.geom_contype[g] and m.body(int(m.geom_bodyid[g])).name==a.hand+'_hand_'+n+'_link') for n in (['thumb_2','index_1',a.third_link] if a.tripod else ['thumb_2','index_1'])]
for n in HAND_JOINTS:s.kp[s.act_joint[n]]=a.finger_kp;s.kd[s.act_joint[n]]=.15
geometry_servo=None;hand_offset=np.zeros(7);lift_contact_shift=np.zeros(3);thumb_force=0.;third_force=0.;primary_stable=0;grip_lost=0
spoon_penetration_peak=0.;tip_contact_peak=0.;tip_penetration_peak=0.;tip_step_force=0.
hold_window=deque(maxlen=int(2/dt));hold_angles=deque(maxlen=int(2/dt));hold_angle_drift=None;hold_drift=None
grip_stable=0;lift_start=None;pinch_p=None;pinch_R=None;pinch_left=None;pinch_shift=np.zeros(3);index_force=0.
final_hold_q=None;hold_latched_time=None;q=poses[-1,ik.qa].copy();level_p=None;level_R=None;lift_objp=None;lift_objR=None;liftp=None;liftR=None;leftp=None;leftR=None;trim=np.zeros(3);rows=[];states=[];samples=[];liquid=[];peak=np.zeros(m.nu);bad=None;good=0;maxlift=0.;min_gap=1.;maxtrack=0.
brew.apply(d,water)

def smooth(x):x=np.clip(x,0,1);return x*x*(3-2*x)
for k in range(int((end+(2 if a.reach_only else 24+2*a.lift_duration))/dt)):
 t=k*dt;idx=min(int(np.searchsorted(times,t,side='right')-1),len(poses)-2);fraction=np.clip((t-times[idx])/(times[idx+1]-times[idx]),0,1);target[moving]=poses[idx,moving]*(1-fraction)+poses[idx+1,moving]*fraction;phase=str(phases[idx+1]) if t<end else 'settle_reach'
 if t>=end:
  other_roll=m.jnt_qposadr[m.joint(other_hand+'_shoulder_roll_joint').id];target[other_roll]+=(.15 if other_hand=='left' else -.15)*smooth((t-end)/2)
 if t>end+2 and not a.reach_only:
  phase='pinch';target[ha]=opened if a.normal_pinch else opened+(close-opened)*smooth((t-end-2)/3)
 if a.geometry_grip and t>end+2 and lift_start is None:
  if geometry_servo is None:geometry_servo=SpoonGraspServo(m,d,ik.qa,ha,bounds,pinch_tips,a.third_link,anatomy);q=target[ik.qa].copy()
  target[ik.qa]=q;target[ha]=opened+hand_offset
  if k%10==0 and t<end+7:
   command=geometry_servo.advance(d,target,t-end-2);q=command[:17];hand_offset=command[17:]-opened
  target[ik.qa]=q;target[ha]=opened
 if a.index_servo and not a.tripod and t>end+2 and lift_start is None:
  if pinch_p is None:q=target[ik.qa].copy();pinch_p,pinch_R=ik.fk(q);pinch_left=d.xpos[left].copy()
  if k%10==0:
   options=[]
   for h in handle_geoms:
    if a.tripod and not .040<m.geom_pos[h,0]<.059:continue
    pts=np.zeros(6);gap_=mujoco.mj_geomDistance(m,d,pinch_tips[1],h,.04,pts);options.append((gap_,pts))
   gap_,pts=min(options,key=lambda x:x[0]);direction=(pts[3:]-pts[:3])*np.sign(gap_);direction/=max(np.linalg.norm(direction),1e-12);advance=min(.00008,max(0,gap_)*.2) if gap_>0 else np.clip((.15-index_force)*.00001,-.00003,.00003);pinch_shift+=direction*advance;pinch_shift*=min(1,.010/max(np.linalg.norm(pinch_shift),1e-12))
   def press_residual(x):
    pp,rr=ik.fk(x);return np.r_[(pp-pinch_p-pinch_shift)*1000,Rotation.from_matrix(rr@pinch_R.T).as_rotvec()*30,(ik.d.xpos[left]-pinch_left)*500,(x-q)*.1,anatomy.penalty(ik.d)]
   fit=least_squares(press_residual,np.clip(q,bounds[:,0]+1e-9,bounds[:,1]-1e-9),bounds=bounds.T,max_nfev=50);q+=np.clip(fit.x-q,-.004,.004)
  target[ik.qa]=q
 if lift_start is not None and not a.reach_only:
  phase='hold_spoon' if final_hold_q is not None or t>=lift_start+2*a.lift_duration else 'lift';object_active=a.object_feedback and t>=lift_start+a.lift_duration
  if liftp is None:liftp=d.xpos[palm].copy();liftR=d.xmat[palm].reshape(3,3).copy();leftp=d.xpos[left].copy();leftR=d.xmat[left].reshape(3,3).copy();q=q.copy() if a.geometry_grip else target[ik.qa].copy();trim=ik.fk(q)[0]-liftp;lift_objp=d.xpos[spoon].copy();lift_objR=d.xmat[spoon].reshape(3,3).copy();grasp_relative_R=liftR.T@lift_objR
  goal=liftp+np.array([0,0,a.palm_lift*smooth((t-lift_start)/a.lift_duration)])+lift_contact_shift
  if k%10==0 and final_hold_q is None:
   if a.adaptive_grip and not a.tripod and not object_active:
    opts=[]
    for h in handle_geoms:
     if a.tripod and not .040<m.geom_pos[h,0]<.059:continue
     pts=np.zeros(6);gg=mujoco.mj_geomDistance(m,d,pinch_tips[1],h,.04,pts);opts.append((gg,pts))
    gg,pts=min(opts,key=lambda x:x[0]);nn=(pts[3:]-pts[:3])*np.sign(gg);nn/=max(np.linalg.norm(nn),1e-12);step=np.clip(max(gg,0)*.15+(.15-index_force)*.000008,-.00003,.00008);lift_contact_shift+=nn*step;lift_contact_shift*=min(1,.008/max(np.linalg.norm(lift_contact_shift),1e-12))
   if object_active and level_p is None:level_p=d.xpos[spoon].copy();level_R=d.xmat[spoon].reshape(3,3).copy();trim[:]=0
   object_goal=lift_objp if level_p is None else level_p+np.array([0,0,.02*smooth((t-lift_start-a.lift_duration)/a.lift_duration)]);start_R=lift_objR if level_R is None else level_R;level=Rotation.from_euler('z',np.arctan2(start_R[1,0],start_R[0,0])).as_matrix()@Rotation.from_euler('y',-a.lift_target_tilt,degrees=True).as_matrix();object_R=Rotation.from_rotvec(smooth((t-lift_start-a.lift_duration)/a.lift_duration)*Rotation.from_matrix(level@start_R.T).as_rotvec()).as_matrix()@start_R;actual_R=d.xmat[palm].reshape(3,3);local_obj=actual_R.T@(d.xpos[spoon]-d.xpos[palm]);local_R=actual_R.T@d.xmat[spoon].reshape(3,3);trim=np.clip(trim+np.clip(.15*((object_goal-d.xpos[spoon]) if object_active else (goal-d.xpos[palm])),-.0005,.0005),-.015,.015)
   free_shift=np.zeros(3)
   def residual(x):
    pp,rr=ik.fk(x);position=pp+rr@local_obj-object_goal-trim if object_active else pp-goal-trim;orientation=rr@local_R@object_R.T if object_active else rr@liftR.T;return np.r_[position*1000,Rotation.from_matrix(orientation).as_rotvec()*30,(ik.d.xpos[left]-leftp-free_shift)*500,Rotation.from_matrix(ik.d.xmat[left].reshape(3,3)@leftR.T).as_rotvec()*20,(x-q)*.1,anatomy.penalty(ik.d)]
   fit=least_squares(residual,np.clip(q,bounds[:,0]+1e-9,bounds[:,1]-1e-9),bounds=bounds.T,max_nfev=60);q+=np.clip(fit.x-q,-.006,.006)
  if final_hold_q is None and a.geometry_grip and object_active and t>lift_start+a.lift_duration+1 and d.xpos[spoon,2]-spoon0[2]>.08 and d.xmat[spoon].reshape(3,3)[2,2]>np.cos(np.deg2rad(2)) and min(thumb_force,index_force)>.75*a.contact_force_target:
   final_hold_q=q.copy();hold_latched_time=t
  if final_hold_q is not None:q=final_hold_q.copy()
  target[ik.qa]=q
 if k%10==0 and k:
  brew.step(d,.02,water);flow=water.step(dt*10,d.xpos[jar],d.xmat[jar].reshape(3,3),d.xpos[filter_id],d.xmat[filter_id].reshape(3,3),d.xpos[cup],d.xmat[cup].reshape(3,3));coupler.apply(d,water);liquid.append({'t':float(d.time),**flow})
 if a.normal_pinch and t>end+2 and k%10==0 and final_hold_q is None:
  normal[:]=0;contact_data=[];servo_contacts=[]
  for ti,g in enumerate(pinch_tips):
   options=[]
   for h in handle_geoms:
    if a.tripod and not ((.020<m.geom_pos[h,0]<.038) if ti==2 else (.040<m.geom_pos[h,0]<.059)):continue
    pts=np.zeros(6);gap_=mujoco.mj_geomDistance(m,d,g,h,.04,pts);options.append((gap_,pts))
   gap_,pts=min(options,key=lambda x:x[0])
   if abs(gap_)<1e-8 or gap_>=.04:continue
   direction=(pts[3:]-pts[:3])*np.sign(gap_);direction/=max(np.linalg.norm(direction),1e-12);jac=np.zeros((3,m.nv));jr=np.zeros((3,m.nv));mujoco.mj_jac(m,d,jac,jr,pts[:3],int(m.geom_bodyid[g]));contact_data.append((pts.copy(),direction.copy(),jac[:,s.vadr].copy(),jr[:,s.vadr].copy()));servo_contacts.append((gap_,pts[:3].copy(),direction.copy()))
  if a.adaptive_grip and final_hold_q is None and ((a.tripod and not a.geometry_grip and len(servo_contacts)==3) or (a.geometry_grip and t>=end+7 and len(servo_contacts)==len(pinch_tips))):
   hand_offset=advance_fingers(m,d,[m.joint(n).id for n in HAND_JOINTS],[m.geom_bodyid[g] for g in pinch_tips],servo_contacts,([thumb_force,index_force,third_force] if a.tripod else [thumb_force,index_force]),opened,hand_offset,a.third_link,tip_step_force,force_targets=([a.contact_force_target]*2 if not a.tripod else None))
  allocated=[x[1]*a.pinch_force*(.5 if a.tripod and i else 1.)*smooth((t-end-2)/1.5) for i,x in enumerate(contact_data)];contact_torques=np.zeros(len(contact_data))
  if a.grasp_wrench and lift_start is not None and len(contact_data)==2:
   palmR=d.xmat[palm].reshape(3,3);spoonR=d.xmat[spoon].reshape(3,3)
   if grasp_relative_R is None:grasp_relative_R=palmR.T@spoonR
   error_R=Rotation.from_matrix(palmR@grasp_relative_R@spoonR.T).as_rotvec();max_grasp_rotation=max(max_grasp_rotation,float(np.linalg.norm(error_R)))
   torque=np.clip(error_R*.004,-.002,.002)
   desired,axial=allocate([x[0][3:] for x in contact_data],[x[1] for x in contact_data],d.xipos[spoon],m.body_mass[spoon],m.opt.gravity,torque,a.pinch_force)
   blend=smooth((t-lift_start)/.5);contact_torques=blend*axial;allocated=[(1-blend*a.grasp_force_blend)*f+blend*a.grasp_force_blend*new for f,new in zip(allocated,desired)]
  for x,force,torque_n in zip(contact_data,allocated,contact_torques):normal+=(x[2].T@force+x[3].T@(x[1]*torque_n))*finger_mask
 if a.adaptive_grip and t>end+2:
  if k%10==0 and not a.tripod and not a.geometry_grip:hand_offset[0]=np.clip(hand_offset[0]+(.001 if tip_step_force>.1 else (-.001 if thumb_force<.15 else (.0005 if thumb_force>.4 else 0))),-.25,.05)
  target[ha]+=hand_offset
 error=target[s.qadr]-d.qpos[s.qadr];friction=m.dof_frictionloss[s.vadr]*np.tanh(error/.001)*np.isin(s.qadr,moving) if a.friction_compensation else 0.
 tau=s.kp*error-s.kd*d.qvel[s.vadr]+d.qfrc_bias[s.vadr]+normal+friction;d.ctrl[:]=np.clip(tau,m.actuator_ctrlrange[:,0],m.actuator_ctrlrange[:,1]);peak=np.maximum(peak,np.abs(d.ctrl));mujoco.mj_step(m,d);forces={};grip_forces={};spoon_support=[];tip_step_force=0.
 for ci,c in enumerate(d.contact):
  if c.dist>=0:continue
  gs=[int(c.geom1),int(c.geom2)];bn=[m.body(int(m.geom_bodyid[g])).name for g in gs];robot=any(n.startswith(('left_','right_','torso','waist','pelvis','head')) for n in bn);allowed='scoop' in bn and any(n.startswith(a.hand+'_hand') for n in bn) and phase in ['approach_spoon','settle_reach','pinch','lift','hold_spoon']
  if 'scoop' in bn and any(n.startswith(a.hand+'_hand') for n in bn):
   spoon_penetration_peak=max(spoon_penetration_peak,float(-c.dist))
   if -c.dist>.0002:bad={'reason':'spoon contact penetration exceeds0.2mm','penetration_m':float(-c.dist),'bodies':bn}
  tip_pair=set(bn)=={a.hand+'_hand_thumb_2_link',a.hand+'_hand_index_1_link'} and phase in ['pinch','lift','hold_spoon']
  if a.allow_tip_contact and tip_pair:
   tip_force=np.zeros(6);mujoco.mj_contactForce(m,d,ci,tip_force);tip_step_force+=float(tip_force[0]);tip_contact_peak=max(tip_contact_peak,tip_step_force);tip_penetration_peak=max(tip_penetration_peak,float(-c.dist));allowed=tip_step_force<=a.tip_contact_force_limit and -c.dist<=.0002
  if robot and not allowed:bad={'reason':'forbidden contact','bodies':bn,'geoms':[m.geom(g).name for g in gs],'penetration_m':float(-c.dist)}
  if 'scoop' in bn:
   other=bn[1-bn.index('scoop')];force=np.zeros(6);mujoco.mj_contactForce(m,d,ci,force)
   if other.startswith(a.hand+'_hand'):
    forces[other]=forces.get(other,0)+float(force[0]);spoon_geom=gs[bn.index('scoop')];normal_z=float(c.frame[2])*(1 if bn[1]=='scoop' else -1)
    if spoon_geom in handle_geoms:grip_forces[other]=grip_forces.get(other,0)+float(force[0])
   else:spoon_support.append(other)
 gap=min(mujoco.mj_geomDistance(m,d,g,table,1,None) for g in handgeoms);min_gap=min(min_gap,gap);wrists=np.rad2deg(d.qpos[ik.qa[np.r_[7:10,14:17]]]);lift=float(d.xpos[spoon,2]-spoon0[2]);maxlift=max(maxlift,lift);track=float(np.max(np.abs(target[ik.qa]-d.qpos[ik.qa])));maxtrack=max(maxtrack,track)
 if gap<(.008 if phase in ['approach_spoon','settle_reach','pinch','lift','hold_spoon'] else .020):bad={'reason':'hand table clearance','gap_m':float(gap)}
 if grasp_relative_R is not None:
  relative_R=d.xmat[palm].reshape(3,3).T@d.xmat[spoon].reshape(3,3);relative_angle=float(np.rad2deg(np.linalg.norm(Rotation.from_matrix(relative_R@grasp_relative_R.T).as_rotvec())));max_grasp_rotation=max(max_grasp_rotation,np.deg2rad(relative_angle))
  if relative_angle>a.max_grasp_rotation:bad={'reason':'spoon rotated in grasp','degrees':relative_angle}
 if np.any(np.abs(wrists)>[62,47,32,62,47,32]):bad={'reason':'wrist bounds','degrees':wrists.tolist()}
 if not anatomy.valid(d,tolerance_deg=2):bad={'reason':'geometric elbow flexion outside task band','degrees':anatomy.angles(d).tolist()}
 if water.spilled_ml>wet['spilled_ml']+.5:bad={'reason':'spill'}
 if np.max(np.linalg.norm(d.xpos[props]-prop0,axis=1))>.01:bad={'reason':'prop moved more than1cm','displacements_m':{m.body(b).name:float(np.linalg.norm(d.xpos[b]-p)) for b,p in zip(props,prop0)}}
 grip_threshold=max(.02,1.5*m.body_mass[spoon]*np.linalg.norm(m.opt.gravity)) if a.payload_grip_threshold else .05
 index_force=grip_forces.get(a.hand+'_hand_index_1_link',0.) if a.tripod else sum(f for n,f in grip_forces.items() if 'index' in n);thumb_force=sum(f for n,f in grip_forces.items() if 'thumb' in n);third_force=grip_forces.get(a.hand+'_hand_'+a.third_link+'_link',0.);primary_stable=primary_stable+1 if thumb_force>grip_threshold and index_force>grip_threshold else 0
 opposing=any('thumb' in n and f>grip_threshold for n,f in grip_forces.items()) and any('index' in n and f>grip_threshold for n,f in grip_forces.items()) and (not a.tripod or (third_force>.03 and index_force>grip_threshold));grip_stable=grip_stable+1 if phase=='pinch' and opposing and (not a.geometry_grip or (thumb_force>.8*a.contact_force_target and index_force>.8*a.contact_force_target)) else 0
 if grip_stable>=int(1/dt) and lift_start is None:lift_start=t
 if lift_start is not None and not opposing:grip_lost+=1
 else:grip_lost=0
 if grip_lost>int(.3/dt):bad={'reason':'opposing handle grip lost during lift'}
 if t>end+16 and lift_start is None and not a.reach_only:bad={'reason':'opposing grip not established'}
 spoon_tilt=float(np.rad2deg(np.arccos(np.clip(d.xmat[spoon].reshape(3,3)[2,2],-1,1))))
 if phase=='hold_spoon' and spoon_tilt<a.max_empty_tilt and lift>.05 and opposing and not spoon_support and (not a.geometry_grip or d.xpos[palm,2]>d.xpos[spoon,2]+.045):hold_window.append(d.xpos[spoon].copy());hold_angles.append(d.xmat[spoon].reshape(3,3).copy())
 else:hold_window.clear();hold_angles.clear()
 good=0
 if len(hold_window)==hold_window.maxlen and k%50==0:
  hold_drift=float(np.max(np.linalg.norm(np.array(hold_window)-hold_window[0],axis=1)));hold_angle_drift=max(float(np.rad2deg(np.linalg.norm(Rotation.from_matrix(rr@hold_angles[0].T).as_rotvec()))) for rr in hold_angles);good=len(hold_window) if hold_drift<.002 and hold_angle_drift<2 else 0
 if k%16==0:
  rows.append({'t':float(d.time),'phase':phase,'table_gap_m':float(gap)});states.append(d.qpos.copy());samples.append({'t':float(d.time),'phase':phase,'lift_m':lift,'spoon_tilt_deg':spoon_tilt,'finger_force_N':forces,'qualified_handle_force_N':grip_forces,'other_spoon_contacts':spoon_support,'elbow_flexion_deg':anatomy.angles(d).tolist(),'wrist_deg':wrists.tolist(),'tracking_error_rad':track})
 if k%2500==0:print({'time_s':t,'reach_duration_s':end,'phase':phase,'lift_m':lift,'fingers':forces,'gap_m':gap},flush=True)
 if s.warnings():bad={'reason':'MuJoCo numerical warning','warnings':s.warnings()}
 if bad or good>=int(2/dt):break
passed=bad is None and not s.warnings() and (a.reach_only or good>=int(2/dt));report={'spoon_penetration_peak_m':spoon_penetration_peak,'spoon_penetration_limit_m':.0002,'model':'gpt-6-astra','source_scene':source_scene,'contact_model_override':str(a.scene) if a.scene else None,'geometry_grip':a.geometry_grip,'finger_position_hold':a.geometry_grip,'contact_force_target_N':a.contact_force_target,'friction_compensation':a.friction_compensation,'payload_grip_threshold':a.payload_grip_threshold,'grip_threshold_N':float(grip_threshold),'hand':a.hand,'tripod':a.tripod,'third_link':a.third_link,'contact_mode':'three contact points, not necessarily three digits' if a.tripod else 'thumb/index pinch','max_grasp_rotation_limit_deg':a.max_grasp_rotation,'grasp_wrench':a.grasp_wrench,'grasp_force_blend':a.grasp_force_blend,'max_grasp_relative_rotation_deg':float(np.rad2deg(max_grasp_rotation)),'palm_lift_command_m':a.palm_lift,'lift_target_tilt_deg':a.lift_target_tilt,'empty_spoon_tilt_limit_deg':a.max_empty_tilt,'allow_tip_contact':a.allow_tip_contact,'tip_contact_peak_N':tip_contact_peak,'tip_penetration_peak_m':tip_penetration_peak,'tip_contact_force_limit_N':a.tip_contact_force_limit,'tip_contact_rule':f'distal thumb/index only, during grasp, <={a.tip_contact_force_limit}N and0.2mm soft penetration; opposing spoon contacts still required','source':str(source),'plan':str(a.plan),'scene':r['scene'],'pass':passed,'failure':bad if bad else (None if passed else {'reason':'spoon lift acceptance not reached'}),'free_hand_clearance':'move passive shoulder roll outward0.15rad during2s settle before pinch','scope':'physical reach only' if a.reach_only else f'physical {a.hand} spoon handle pinch, palm{a.palm_lift*100:.0f}cm raise then spoon2cm raise/level, fixed pelvis, privileged state feedback','coffee_completed':False,'initial_water_ml':water.initial_ml,'two_finger':a.two_finger,'normal_pinch':a.normal_pinch,'index_servo':a.index_servo,'adaptive_grip':a.adaptive_grip,'object_feedback':a.object_feedback,'hand_offset':hand_offset.tolist(),'damped_arms':a.damped_arms,'hold_latched_elapsed_s':hold_latched_time,'hold_drift_m':hold_drift,'hold_angle_drift_deg':hold_angle_drift,'lift_phase_duration_s':a.lift_duration,'pinch_force_N':a.pinch_force,'finger_kp':a.finger_kp,'warnings':s.warnings(),'max_spoon_lift_m':maxlift,'hold_good_steps':good,'min_hand_table_gap_m':float(min_gap),'max_joint_tracking_error_rad':maxtrack,'water_state':asdict(water),'rows':rows,'grasp_samples':samples,'liquid_samples':liquid,'peak_motor_torques_Nm':{n:float(peak[i]) for n,i in s.act_joint.items()}}
(a.out/'report.json').write_text(json.dumps(report,indent=2));np.savez_compressed(a.out/'trajectory.npz',qpos=states);integration=np.zeros(mujoco.mj_stateSize(m,mujoco.mjtState.mjSTATE_INTEGRATION));mujoco.mj_getState(m,d,integration,mujoco.mjtState.mjSTATE_INTEGRATION);np.savez_compressed(a.out/'continuation.npz',integration=integration,last_target=target,kp=s.kp,kd=s.kd,body_mass=m.body_mass,body_ipos=m.body_ipos,body_inertia=m.body_inertia,body_iquat=m.body_iquat,bvh_aabb=m.bvh_aabb,dry_kettle_kg=float(ck['dry_kettle_kg']));(a.out/'liquid-state.json').write_text(json.dumps(asdict(water),indent=2));print({k:v for k,v in report.items() if k not in ['rows','grasp_samples','liquid_samples','peak_motor_torques_Nm']})

brew.save(a.out)
