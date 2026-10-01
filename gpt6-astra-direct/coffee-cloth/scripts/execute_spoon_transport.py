"""Motor-only transport of physically grasped spoon through audited IK waypoints."""
import argparse,json,sys,shutil
from pathlib import Path
from dataclasses import asdict
from collections import deque
import mujoco,numpy as np
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'g1-cup-grasp/scripts'))
from g1_sim import G1Sim,ARM_JOINTS,HAND_JOINTS
from kinematics import ArmIK
from elbow_anatomy import ElbowAnatomy
from finger_contact_servo import advance as advance_fingers
from transport_ik_jacobian import make_jacobian
from spoon_grasp_wrench import allocate
from liquid_mass import LiquidMassCoupler
from kettle_liquid import KettleWater
from brew_state import BrewState
ap=argparse.ArgumentParser();ap.add_argument('--finger-kp',type=float);ap.add_argument('--finger-kd',type=float);ap.add_argument('--force-integral',action='store_true');ap.add_argument('--freeze-fingers',action='store_true');ap.add_argument('--rotation-servo',action='store_true');ap.add_argument('--contact-frame',action='store_true');ap.add_argument('--fixed-grip-torque',action='store_true');ap.add_argument('--align-pinch',action='store_true');ap.add_argument('--allow-pot-contact',action='store_true');ap.add_argument('--contact-force-target',type=float);ap.add_argument('--pinch-force',type=float);ap.add_argument('--analytic-ik',action='store_true');ap.add_argument('--grasp-wrench',action='store_true');ap.add_argument('--grasp-stiffness',type=float,default=.01);ap.add_argument('--maintain-grip',action='store_true');ap.add_argument('--dip-orientation-weight',type=float);ap.add_argument('--segment-seconds',type=float,default=8);ap.add_argument('--joint-speed',type=float,default=.14);ap.add_argument('--orientation-weight',type=float,default=35);ap.add_argument('--pot-clearance',type=float,default=.018);ap.add_argument('--source',type=Path,default=Path('results/spoon-physical-006'));ap.add_argument('--plan',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();assert not (a.source/'INVALIDATED.json').exists(),'source invalidated by audit';r=json.loads((a.source/'report.json').read_text());pr=json.loads((a.plan/'report.json').read_text());assert r['pass'] and pr['pass'];a.out.mkdir(exist_ok=False)
if a.contact_force_target is not None:
 assert a.contact_force_target>0;r['contact_force_target_N']=a.contact_force_target
if a.pinch_force is not None:
 assert a.pinch_force>0;r['pinch_force_N']=a.pinch_force
for p in [Path(__file__),Path('scripts/liquid_mass.py'),Path('scripts/kettle_liquid.py')]+[Path('scripts')/n for n in ['brew_state.py','coffee_grounds.py','grounds_mass.py','kettle_heater.py','elbow_anatomy.py','finger_contact_servo.py','transport_ik_jacobian.py','spoon_grasp_wrench.py']]:shutil.copy2(p,a.out/p.name)
fixed_grip_torque=a.fixed_grip_torque or r.get('fixed_grip_torque',False);fixed_normal=np.array(r['grip_motor_torque_Nm']) if fixed_grip_torque and 'grip_motor_torque_Nm' in r else None
force_integral=a.force_integral or r.get('force_integral',False)
normal_commands=np.array(r.get('contact_normal_commands_N',[r.get('pinch_force_N',.3)]*(3 if r.get('tripod',False) else 2)),dtype=float)
freeze_fingers=a.freeze_fingers or r.get('freeze_fingers',False)
rotation_servo=a.rotation_servo or r.get('rotation_servo',False)
contact_frame=a.contact_frame or r.get('contact_frame',False);contact_normals={}
align_pinch=a.align_pinch or r.get('align_pinch',False);max_axial_offset=0.;axial_offset=0.;transverse_offset=0.;max_transverse_offset=0.;pressure_points={}
allow_pot_contact=a.allow_pot_contact or r.get('allow_pot_contact',False);pot_contact_peak=0.;pot_contact_penetration_peak=0.
analytic_ik=a.analytic_ik or r.get('analytic_ik',False);grasp_wrench=a.grasp_wrench or r.get('grasp_wrench',False)
is_placement=pr.get('task')=='place_spoon';placement_drop=0.;spoon_table_supported=False;placement_contact_q=None;placement_contact_normal=None
hand_side='right' if pr['joint_names'][3].startswith('right_') else 'left';other_side='left' if hand_side=='right' else 'right';s=G1Sim(r['scene']);m,d=s.m,s.d;brew=BrewState(m,a.source);ck=np.load(a.source/'continuation.npz');coupler=LiquidMassCoupler(m,float(ck['dry_kettle_kg']));water=KettleWater(initial_ml=800)
for n,v in json.loads((a.source/'liquid-state.json').read_text()).items():setattr(water,n,v)
for n in ['body_mass','body_ipos','body_inertia','body_iquat','bvh_aabb']:getattr(m,n)[:]=ck[n]
mujoco.mj_setConst(m,mujoco.MjData(m));mujoco.mj_setState(m,d,ck['integration'],mujoco.mjtState.mjSTATE_INTEGRATION);mujoco.mj_forward(m,d);s.kp[:]=ck['kp'];s.kd[:]=ck['kd'];target=ck['last_target'].copy();ik=ArmIK(s,pr['joint_names'],palm=hand_side+'_wrist_yaw_link');bounds=ik.bounds.copy();bounds[0]=[-.7,.7];bounds[1:3]=np.deg2rad([[-12,12],[-12,12]])
bounds[2,1]=min(ik.bounds[2,1],np.deg2rad(pr.get('waist_pitch_max_deg',12)))
anatomy=ElbowAnatomy(m);anatomy.bound_search(m,ik.joint_names,bounds)
for j in [7,14]:bounds[j:j+3]=np.deg2rad([[-60,60],[-45,45],[-30,30]])
feet=[g for g in range(m.ngeom) if m.geom_contype[g] and m.geom_type[g]==mujoco.mjtGeom.mjGEOM_SPHERE and m.body(int(m.geom_bodyid[g])).name in ['left_ankle_roll_link','right_ankle_roll_link']];support_hull=ConvexHull(d.geom_xpos[feet,:2]).equations;pelvis=m.body('pelvis').id;spoon_penetration_peak=0.;min_com_margin=1.;tip_contact_peak=0.;tip_penetration_peak=0.;tip_step_force=0.
maintain_grip=a.maintain_grip or r.get('maintain_grip',False);finger_position_hold=r.get('finger_position_hold',False);geometry_grip=r.get('geometry_grip',False);friction_compensation=r.get('friction_compensation',False);payload_grip_threshold=r.get('payload_grip_threshold',False);three_contacts=r.get('tripod',False);third_link=r.get('third_link','middle_1');third_indices=[3,4] if third_link=='index_0' else [5,6];hand_offset=np.zeros(7);third_force=0.;index_force=0.
# Change servo stiffness without an initial proportional-torque jump.
for name,i in s.act_joint.items():
 if name.startswith(hand_side+'_hand'):
  if a.finger_kp is not None:
   assert a.finger_kp>0
   adr=s.qadr[i];target[adr]=d.qpos[adr]+s.kp[i]/a.finger_kp*(target[adr]-d.qpos[adr]);s.kp[i]=a.finger_kp
  if a.finger_kd is not None:
   assert a.finger_kd>=0;s.kd[i]=a.finger_kd
q=target[ik.qa].copy();handnames=[n.replace('right_',hand_side+'_') for n in HAND_JOINTS];ha=m.jnt_qposadr[[m.joint(n).id for n in handnames]];hand=target[ha].copy();spoon=m.body('scoop').id;palm=m.body(hand_side+'_wrist_yaw_link').id;other=m.body(other_side+'_wrist_yaw_link').id;bowl=np.array([-.045,0,.003]);R0=d.xmat[spoon].reshape(3,3).copy();p0=d.xpos[spoon]+R0@bowl
wps=[{'bowl_goal':p0,'spoon_R':R0,'other_palm':d.xpos[other].copy(),'q':q.copy()}]+pr['results'];durations=[max(a.segment_seconds,float(np.max(np.abs(np.array(y['q'])-x['q'])))/a.joint_speed) for x,y in zip(wps[:-1],wps[1:])];times=np.r_[0,np.cumsum(durations)];end=times[-1];props=['chaleira','coador','copo','pote','base_eletrica','tampa'];ids=[m.body(n).id for n in props];prop0=d.xpos[ids].copy();table=m.geom('tampo').id;handgeoms=[g for g in range(m.ngeom) if m.geom_contype[g] and m.body(int(m.geom_bodyid[g])).name.startswith(('left_hand','left_wrist','right_hand','right_wrist'))];handle=[g for g in range(m.ngeom) if m.geom_contype[g] and m.geom_bodyid[g]==spoon and (.020 if three_contacts else .040)<m.geom_pos[g,0]<.059];tips=[next(g for g in range(m.ngeom) if m.geom_contype[g] and m.body(int(m.geom_bodyid[g])).name==hand_side+'_hand_'+n+'_link') for n in (['thumb_2','index_1',third_link] if three_contacts else ['thumb_2','index_1'])];finger=np.array([m.joint(m.actuator_trnid[i,0]).name.startswith(hand_side+'_hand') for i in range(m.nu)]);normal=np.zeros(m.nu);dt=m.opt.timestep;rows=[];states=[];samples=[];liquid=[];bad=None;peak=np.zeros(m.nu);window=deque(maxlen=int(2/dt));initial_filter_g=brew.grounds.filter_g if brew.enabled else 0;initial_spoon_g=brew.grounds.spoon_g if brew.enabled else 0;passed=False;grip_lost=0;thumb_force=.3;offset=0.;trim=np.zeros(3);min_gap=1.;max_error=0.;hold_drift=None

brew.apply(d,water)
# Initialize measured feedback from restored contacts, not a fictitious unloaded hand.
initial_forces={};initial_points={};initial_normals={}
for ci,c in enumerate(d.contact):
 gs=[int(c.geom1),int(c.geom2)];bn=[m.body(int(m.geom_bodyid[g])).name for g in gs]
 if c.dist<0 and 'scoop' in bn and gs[bn.index('scoop')] in handle:
  name=bn[1-bn.index('scoop')]
  if name.startswith(hand_side+'_hand'):
   cf=np.zeros(6);mujoco.mj_contactForce(m,d,ci,cf);initial_forces[name]=initial_forces.get(name,0.)+float(cf[0]);initial_points[name]=initial_points.get(name,np.zeros(3))+cf[0]*c.pos;initial_normals[name]=initial_normals.get(name,np.zeros(3))+cf[0]*c.frame[:3]*(1 if bn[0]==name else -1)
pressure_points={n:p/initial_forces[n] for n,p in initial_points.items() if initial_forces[n]>.01}
contact_normals={n:v/max(np.linalg.norm(v),1e-9) for n,v in initial_normals.items()}
thumb_force=sum(f for n,f in initial_forces.items() if 'thumb' in n)
index_force=initial_forces.get(hand_side+'_hand_index_1_link',0.)
third_force=initial_forces.get(hand_side+'_hand_'+third_link+'_link',0.)
initial_grasp_R=d.xmat[palm].reshape(3,3).T@d.xmat[spoon].reshape(3,3);max_relative_rotation_deg=0.
# Keep one grasp reference across the whole dose sequence; stage boundaries
# must not hide accumulated sliding by resetting the angular acceptance gate.
anchor_source=Path(r.get('episode_grasp_source',str(a.source)))
if 'episode_grasp_R' in r:
 episode_grasp_R=np.array(r['episode_grasp_R'])
else:
 ancestor=r
 while ancestor.get('source'):
  candidate=Path(ancestor['source']);candidate_report=json.loads((candidate/'report.json').read_text())
  if not candidate_report.get('geometry_grip',False):break
  anchor_source=candidate;ancestor=candidate_report
 anchor_data=mujoco.MjData(m);anchor_ck=np.load(anchor_source/'continuation.npz')
 mujoco.mj_setState(m,anchor_data,anchor_ck['integration'],mujoco.mjtState.mjSTATE_INTEGRATION);mujoco.mj_kinematics(m,anchor_data)
 episode_grasp_R=anchor_data.xmat[palm].reshape(3,3).T@anchor_data.xmat[spoon].reshape(3,3)
max_episode_grasp_rotation_deg=0.


def smooth(x):x=np.clip(x,0,1);return x*x*(3-2*x)
for k in range(int((end+8)/dt)):
 t=k*dt;wi=min(int(np.searchsorted(times,t,side='right')-1),len(wps)-2);u=smooth((t-times[wi])/durations[wi]);x,y=wps[wi:wi+2];pg=(1-u)*np.array(x['bowl_goal'])+u*np.array(y['bowl_goal']);rx=np.array(x['spoon_R']);rg=Rotation.from_rotvec(u*Rotation.from_matrix(np.array(y['spoon_R'])@rx.T).as_rotvec()).as_matrix()@rx;othergoal=(1-u)*np.array(x['other_palm'])+u*np.array(y['other_palm']);qref=(1-u)*np.array(x['q'])+u*np.array(y['q']);phase=f"{pr.get('task','transport_spoon')}_{wi}" if t<end else 'hold_'+pr.get('task','above_pot');target[ha]=hand;target[ha[0]]+=offset
 if three_contacts or geometry_grip:target[ha]=hand+hand_offset
 if is_placement and t>=end:
  if not spoon_table_supported and placement_contact_q is None:placement_drop=min(.002,placement_drop+.0003*dt)
  pg[2]-=placement_drop
 if k%10==0:
  if k:
   bs=[m.body(n).id for n in ['chaleira','coador','copo']];brew.step(d,.02,water);flow=water.step(.02,*sum(([d.xpos[b],d.xmat[b].reshape(3,3)] for b in bs),[]));liquid.append({'t':float(d.time),**flow});coupler.apply(d,water)
  actualR=d.xmat[palm].reshape(3,3);objR=d.xmat[spoon].reshape(3,3);local=actualR.T@(d.xpos[spoon]+objR@bowl-d.xpos[palm]);localR=actualR.T@objR;trim=.8*trim+.2*np.clip(.5*(pg-d.xpos[spoon]-objR@bowl),-.004,.004)
  table_margin=max(.010,pr.get('hand_clearance_m',.010)) if is_placement and wi==len(wps)-2 else .024
  avoid=[(g,table,table_margin) for g in handgeoms if mujoco.mj_geomDistance(m,d,g,table,.05,None)<.05]
  fg=[g for g in range(m.ngeom) if m.geom_contype[g] and m.geom_bodyid[g] in ids]
  for g in handgeoms:
   near=sorted((mujoco.mj_geomDistance(m,d,g,h,.045,None),h) for h in fg)
   avoid.extend((g,h,a.pot_clearance if m.geom_bodyid[h]==m.body('pote').id else .018) for gap_,h in near[:2] if gap_<.045)
  torso_geoms=[g for g in range(m.ngeom) if m.geom_contype[g] and m.body(int(m.geom_bodyid[g])).name=='torso_link']
  shoulder_geoms=[g for g in range(m.ngeom) if m.geom_contype[g] and m.body(int(m.geom_bodyid[g])).name in ['left_shoulder_yaw_link','left_shoulder_roll_link','right_shoulder_yaw_link','right_shoulder_roll_link','left_elbow_link','right_elbow_link']]
  avoid.extend((g,h,.006) for g in torso_geoms for h in shoulder_geoms)
  orientation_weight=a.dip_orientation_weight if a.dip_orientation_weight is not None and pg[2]<d.xpos[m.body('pote').id,2]+.085 and np.linalg.norm(pg[:2]-d.xpos[m.body('pote').id,:2])<.08 else a.orientation_weight
  def residual(xx):
   pp,rr=ik.fk(xx);gaps=[min(0,mujoco.mj_geomDistance(m,ik.d,g,h,.03,None)-margin) for g,h,margin in avoid];return np.r_[(pp+rr@local-pg-trim)*1000,Rotation.from_matrix(rr@localR@rg.T).as_rotvec()*orientation_weight,(ik.d.xpos[other]-othergoal)*100,(xx-qref)*2,(xx-q)*.2,np.array(gaps)*5000,anatomy.penalty(ik.d)]
  local_bounds=np.column_stack([np.maximum(bounds[:,0],q-.006),np.minimum(bounds[:,1],q+.006)])
  if placement_contact_q is None:
   fit=least_squares(residual,np.clip(q,local_bounds[:,0]+1e-9,local_bounds[:,1]-1e-9),bounds=local_bounds.T,max_nfev=50,jac=make_jacobian(ik,local,localR,rg,other,orientation_weight,avoid,anatomy) if analytic_ik else '2-point');q=fit.x
  else:q=placement_contact_q.copy()
  normal[:]=0;servo_contacts=[];wrench_contacts=[]
  if force_integral:
   measured=np.array([thumb_force,index_force,third_force] if three_contacts else [thumb_force,index_force])
   force_error=r.get('contact_force_target_N',3.)-measured
   force_error[np.abs(force_error)<.1*r.get('contact_force_target_N',3.)]=0.
   normal_commands=np.clip(normal_commands+.05*force_error,.1,20.)
  for ti,g in enumerate(tips):
   candidates=[]
   for h in handle:
    if three_contacts and not ((.020<m.geom_pos[h,0]<.038) if ti==2 else (.040<m.geom_pos[h,0]<.059)):continue
    pts=np.zeros(6);gap_=mujoco.mj_geomDistance(m,d,g,h,.04,pts);candidates.append((gap_,pts))
   gap_,pts=min(candidates,key=lambda x:x[0])
   if abs(gap_)<1e-8 or gap_>=.04:continue
   nn=(pts[3:]-pts[:3])*np.sign(gap_);nn/=max(np.linalg.norm(nn),1e-12)
   if contact_frame:
    body_name=m.body(int(m.geom_bodyid[g])).name;nn=contact_normals.get(body_name,nn);pts[:3]=pressure_points.get(body_name,pts[:3])
   jac=np.zeros((3,m.nv));jr=np.zeros((3,m.nv));mujoco.mj_jac(m,d,jac,jr,pts[:3],int(m.geom_bodyid[g]));normal+=(jac[:,s.vadr].T@(nn*(normal_commands[ti] if force_integral else r.get('pinch_force_N',.3)*(.5 if three_contacts and ti else 1.))))*finger
   servo_point=pressure_points.get(m.body(int(m.geom_bodyid[g])).name,pts[:3]) if align_pinch else pts[:3]
   servo_contacts.append((gap_,servo_point.copy(),nn.copy()));wrench_contacts.append((pts.copy(),nn.copy(),jac.copy(),jr.copy()))
  if len(servo_contacts)==2:
   axial_offset=float(d.xmat[spoon].reshape(3,3)[:,0]@(servo_contacts[0][1]-servo_contacts[1][1]));max_axial_offset=max(max_axial_offset,abs(axial_offset));transverse_offset=float(d.xmat[spoon].reshape(3,3)[:,1]@(servo_contacts[0][1]-servo_contacts[1][1]));max_transverse_offset=max(max_transverse_offset,abs(transverse_offset))
  if grasp_wrench and len(wrench_contacts)==2:
   expected_R=d.xmat[palm].reshape(3,3)@episode_grasp_R;current_R=d.xmat[spoon].reshape(3,3)
   angular_error=Rotation.from_matrix(expected_R@current_R.T).as_rotvec()
   object_v=np.zeros(6);palm_v=np.zeros(6);mujoco.mj_objectVelocity(m,d,mujoco.mjtObj.mjOBJ_BODY,spoon,object_v,0);mujoco.mj_objectVelocity(m,d,mujoco.mjtObj.mjOBJ_BODY,palm,palm_v,0)
   torque=a.grasp_stiffness*angular_error-.0003*(object_v[:3]-palm_v[:3]);torque*=min(1.,.005/max(np.linalg.norm(torque),1e-9))
   desired,axial=allocate([x[0][3:] for x in wrench_contacts],[x[1] for x in wrench_contacts],d.xipos[spoon],m.body_mass[spoon],m.opt.gravity,torque,r.get('pinch_force_N',3.))
   wrench_motor=np.zeros(m.nu)
   for x,force,twist in zip(wrench_contacts,desired,axial):wrench_motor+=(x[2][:,s.vadr].T@force+x[3][:,s.vadr].T@(x[1]*twist))*finger
   blend=smooth(t/2);normal=(1-blend)*normal+blend*wrench_motor
  if not freeze_fingers and (three_contacts or geometry_grip) and (not finger_position_hold or maintain_grip) and placement_contact_q is None:
   measured=[thumb_force,index_force,third_force] if three_contacts else [thumb_force,index_force]
   force_goal=r.get('contact_force_target_N',.3)
   servo_targets=[force_goal if f<.9*force_goal or f>1.1*force_goal else f for f in measured] if maintain_grip else ([force_goal]*2 if not three_contacts else None)
   hand_offset=advance_fingers(m,d,[m.joint(n).id for n in handnames],[m.geom_bodyid[g] for g in tips],servo_contacts,([thumb_force,index_force,third_force] if three_contacts else [thumb_force,index_force]),hand,hand_offset,third_link,tip_step_force,force_targets=servo_targets,axial_axis=d.xmat[spoon].reshape(3,3)[:,0] if align_pinch else None,transverse_axis=d.xmat[spoon].reshape(3,3)[:,1] if align_pinch else None,object_rotation_error=Rotation.from_matrix(d.xmat[palm].reshape(3,3)@episode_grasp_R@d.xmat[spoon].reshape(3,3).T).as_rotvec() if rotation_servo else None)
  elif not geometry_grip:offset=np.clip(offset+(.001 if tip_step_force>.1 else (-.001 if thumb_force<.15 else (.0005 if thumb_force>.4 else 0))),-.15,.10)
 if fixed_grip_torque:
  if fixed_normal is None:fixed_normal=normal.copy()
  normal[:]=fixed_normal
 if placement_contact_normal is not None:normal[:]=placement_contact_normal
 target[ik.qa]=q;error_joint=target[s.qadr]-d.qpos[s.qadr];friction=m.dof_frictionloss[s.vadr]*np.tanh(error_joint/.001)*np.isin(s.qadr,np.r_[ik.qa,ha]) if friction_compensation else 0.
 tau=s.kp*error_joint-s.kd*d.qvel[s.vadr]+d.qfrc_bias[s.vadr]+normal+friction;d.ctrl[:]=np.clip(tau,m.actuator_ctrlrange[:,0],m.actuator_ctrlrange[:,1]);peak=np.maximum(peak,np.abs(d.ctrl));mujoco.mj_step(m,d);forces={};pressure_sums={};pressure_weights={};normal_sums={};support=[];tip_step_force=0.;pot_step_force=0.;spoon_table_supported=False
 for ci,c in enumerate(d.contact):
  if c.dist>=0:continue
  gs=[int(c.geom1),int(c.geom2)];bn=[m.body(int(m.geom_bodyid[g])).name for g in gs];allowed='scoop' in bn and any(n.startswith(hand_side+'_hand') for n in bn)
  if allowed:
   spoon_penetration_peak=max(spoon_penetration_peak,float(-c.dist))
   if -c.dist>.0002:bad={'reason':'spoon contact penetration exceeds0.2mm','penetration_m':float(-c.dist)}
  tip_pair=set(bn)=={hand_side+'_hand_thumb_2_link',hand_side+'_hand_index_1_link'}
  if r.get('allow_tip_contact',False) and tip_pair:
   tip_force=np.zeros(6);mujoco.mj_contactForce(m,d,ci,tip_force);tip_step_force+=float(tip_force[0]);tip_contact_peak=max(tip_contact_peak,tip_step_force);tip_penetration_peak=max(tip_penetration_peak,float(-c.dist));allowed=tip_step_force<=r.get('tip_contact_force_limit_N',.5) and -c.dist<=.0002
  if any(n.startswith(('right_','left_','torso','waist','pelvis','head')) for n in bn) and not allowed:bad={'reason':'forbidden contact','bodies':bn,'penetration_m':float(-c.dist)}
  if 'scoop' in bn:
   otherbody=bn[1-bn.index('scoop')];force=np.zeros(6);mujoco.mj_contactForce(m,d,ci,force)
   if otherbody.startswith(hand_side+'_hand') and gs[bn.index('scoop')] in handle:
    forces[otherbody]=forces.get(otherbody,0)+float(force[0]);pressure_weights[otherbody]=pressure_weights.get(otherbody,0)+float(force[0]);pressure_sums[otherbody]=pressure_sums.get(otherbody,np.zeros(3))+float(force[0])*c.pos;normal_sums[otherbody]=normal_sums.get(otherbody,np.zeros(3))+force[0]*c.frame[:3]*(1 if bn[0]==otherbody else -1)
   elif not otherbody.startswith(hand_side+'_hand'):
    support.append(otherbody)
    if table in gs:
     spoon_table_supported=True
     if is_placement and placement_contact_q is None:
      placement_contact_q=d.qpos[ik.qa].copy();placement_contact_normal=normal.copy()
     if -c.dist>.0002:bad={'reason':'spoon table penetration exceeds0.2mm','penetration_m':float(-c.dist)}
    tool_pot=allow_pot_contact and pr.get('task')=='collect_grounds' and otherbody=='pote'
    if tool_pot:
     pot_step_force+=float(force[0]);pot_contact_peak=max(pot_contact_peak,pot_step_force);pot_contact_penetration_peak=max(pot_contact_penetration_peak,float(-c.dist))
     tool_bowl=d.xpos[spoon]+d.xmat[spoon].reshape(3,3)@bowl;pot_body=m.body('pote').id
     if pot_step_force>.5 or -c.dist>.0002 or np.linalg.norm(tool_bowl[:2]-d.xpos[pot_body,:2])>.065 or tool_bowl[2]>d.xpos[pot_body,2]+.10:bad={'reason':'controlled spoon/pot contact gate','force_N':pot_step_force,'penetration_m':float(-c.dist)}
    if not tool_pot and not (is_placement and table in gs and wi==len(wps)-2):bad={'reason':'spoon obstacle contact','body':otherbody}
 gap=min(mujoco.mj_geomDistance(m,d,g,table,1,None) for g in handgeoms);min_gap=min(gap,min_gap);wrists=np.rad2deg(d.qpos[ik.qa[np.r_[7:10,14:17]]]);objR=d.xmat[spoon].reshape(3,3);bp=d.xpos[spoon]+objR@bowl;tilt=float(np.rad2deg(np.arccos(np.clip(objR[2,2],-1,1))));error=float(np.linalg.norm(bp-pg));max_error=max(error,max_error)
 robotmass=m.body_subtreemass[pelvis];payloadmass=m.body_mass[spoon];com=(robotmass*d.subtree_com[pelvis]+payloadmass*d.xipos[spoon])/(robotmass+payloadmass);com_margin=float(np.min(-(support_hull[:,:2]@com[:2]+support_hull[:,2])));min_com_margin=min(min_com_margin,com_margin)
 if com_margin<.02:bad={'reason':'quasi-static COM projection outside20mm foot-polygon margin','margin_m':com_margin}
 relative_R=d.xmat[palm].reshape(3,3).T@objR;relative_angle=float(np.rad2deg(np.linalg.norm(Rotation.from_matrix(relative_R@initial_grasp_R.T).as_rotvec())));max_relative_rotation_deg=max(max_relative_rotation_deg,relative_angle)
 if relative_angle>15:bad={'reason':'spoon rotated in grasp beyond15deg','degrees':relative_angle}
 episode_angle=float(np.rad2deg(np.linalg.norm(Rotation.from_matrix(relative_R@episode_grasp_R.T).as_rotvec())));max_episode_grasp_rotation_deg=max(max_episode_grasp_rotation_deg,episode_angle)
 if episode_angle>15:bad={'reason':'accumulated spoon rotation since original grasp exceeds15deg','degrees':episode_angle}
 if gap<(.008 if is_placement and wi==len(wps)-2 else .020):bad={'reason':'hand table clearance','gap_m':float(gap)}
 if np.any(np.abs(wrists)>[62,47,32,62,47,32]):bad={'reason':'wrist bounds','degrees':wrists.tolist()}
 if np.max(np.linalg.norm(d.xpos[ids]-prop0,axis=1))>.01:bad={'reason':'other prop displacement'}
 if not anatomy.valid(d,tolerance_deg=2):bad={'reason':'geometric elbow flexion outside task band','degrees':anatomy.angles(d).tolist()}
 if allow_pot_contact:
  pb=m.body('pote').id;pot_initial=m.qpos0[m.jnt_qposadr[m.body_jntadr[pb]]:m.jnt_qposadr[m.body_jntadr[pb]]+3]
  if np.linalg.norm(d.xpos[pb]-prop0[props.index('pote')])>.003 or np.linalg.norm(d.xpos[pb]-pot_initial)>.01 or d.xmat[pb].reshape(3,3)[2,2]<np.cos(np.deg2rad(3)):bad={'reason':'pot movement/tilt during controlled tool contact'}
 if water.spilled_ml>.5:bad={'reason':'spill'}
 if brew.enabled and brew.grounds.spilled_g>.05:bad={'reason':'ground coffee spill','grams':brew.grounds.spilled_g}
 pressure_points={name:point/pressure_weights[name] for name,point in pressure_sums.items() if pressure_weights[name]>.01}
 contact_normals={name:v/max(np.linalg.norm(v),1e-9) for name,v in normal_sums.items()}
 grip_threshold=max(.02,1.5*m.body_mass[spoon]*np.linalg.norm(m.opt.gravity)) if payload_grip_threshold else .05
 index_force=forces.get(hand_side+'_hand_index_1_link',0.);thumb_force=sum(f for n,f in forces.items() if 'thumb' in n);third_force=forces.get(hand_side+'_hand_'+third_link+'_link',0.);opposing=thumb_force>grip_threshold and any('index' in n and f>grip_threshold for n,f in forces.items()) and (not three_contacts or (forces.get(hand_side+'_hand_index_1_link',0.)>grip_threshold and third_force>.03));grip_lost=grip_lost+1 if not opposing else 0
 if grip_lost>int(.3/dt):bad={'reason':'lost opposing grip'}
 spoon_table_gap=min(mujoco.mj_geomDistance(m,d,g,table,.01,None) for g in range(m.ngeom) if m.geom_bodyid[g]==spoon and m.geom_contype[g]) if is_placement else None
 if t>=end and opposing and error<.005 and tilt<15 and (not is_placement or (placement_contact_q is not None and spoon_table_gap<.001)):window.append(bp.copy())
 else:window.clear()
 if len(window)==window.maxlen and k%50==0:hold_drift=float(np.max(np.linalg.norm(np.array(window)-window[0],axis=1)));passed=hold_drift<.002 and (pr.get('task')!='dose_filter' or (brew.enabled and initial_spoon_g>.01 and brew.grounds.filter_g-initial_filter_g>=.98*initial_spoon_g))
 if k%16==0:rows.append({'t':float(d.time),'phase':phase,'table_gap_m':float(gap)});states.append(d.qpos.copy());samples.append({'t':float(d.time),'phase':phase,'finger_force_N':forces,'spoon_tilt_deg':tilt,'bowl_m':bp.tolist(),'position_error_m':error,'elbow_flexion_deg':anatomy.angles(d).tolist(),'wrist_deg':wrists.tolist(),'tracking_error_rad':float(np.max(np.abs(q-d.qpos[ik.qa]))),'waist_deg':np.rad2deg(d.qpos[ik.qa[:3]]).tolist(),'com_margin_m':com_margin})
 if k%2500==0:print({'t':t,'phase':phase,'duration':end,'fingers':forces,'error_m':error,'tilt_deg':tilt},flush=True)
 if s.warnings():bad={'reason':'MuJoCo numerical warning','warnings':s.warnings()}
 if bad or passed:break
report={'finger_kp_override':a.finger_kp,'finger_kd_override':a.finger_kd,'force_integral':force_integral,'contact_normal_commands_N':normal_commands.tolist(),'freeze_fingers':freeze_fingers,'rotation_servo':rotation_servo,'contact_frame':contact_frame,'initial_contact_forces_N':initial_forces,'model':'gpt-6-astra','fixed_grip_torque':fixed_grip_torque,'grip_motor_torque_Nm':normal.tolist(),'align_pinch':align_pinch,'alignment_point_source':'force-weighted contact centers' if align_pinch else 'closest geometry','max_pinch_transverse_offset_m':max_transverse_offset,'final_pinch_transverse_offset_m':transverse_offset,'max_pinch_axial_offset_m':max_axial_offset,'final_pinch_axial_offset_m':axial_offset,'allow_pot_contact':allow_pot_contact,'pot_contact_limit_N':.5,'pot_drift_stage_limit_m':.003,'pot_drift_episode_limit_m':.01,'pot_tilt_limit_deg':3,'pot_contact_peak_N':pot_contact_peak,'pot_contact_penetration_peak_m':pot_contact_penetration_peak,'waist_pitch_max_deg':float(np.rad2deg(bounds[2,1])),'final_episode_grasp_rotation_deg':episode_angle,'analytic_ik':analytic_ik,'grasp_wrench':grasp_wrench,'grasp_stiffness_Nm_rad':a.grasp_stiffness,'release_clearance_m':spoon_table_gap,'placement_acceptance':'table contact observed, then stable below1mm clearance for controlled finger release' if is_placement else None,'placement_contact_latched':placement_contact_q is not None,'placement_final_descent_m':placement_drop,'episode_grasp_source':str(anchor_source),'episode_grasp_R':episode_grasp_R.tolist(),'max_episode_grasp_rotation_deg':max_episode_grasp_rotation_deg,'task':pr.get('task'),'supported_at_end':support,'spoon_penetration_peak_m':spoon_penetration_peak,'spoon_penetration_limit_m':.0002,'maintain_grip':maintain_grip,'hand':hand_side,'geometry_grip':geometry_grip,'finger_position_hold':finger_position_hold,'contact_force_target_N':r.get('contact_force_target_N',.3),'friction_compensation':friction_compensation,'payload_grip_threshold':payload_grip_threshold,'grip_threshold_N':float(grip_threshold),'tripod':three_contacts,'third_link':third_link,'allow_tip_contact':r.get('allow_tip_contact',False),'tip_contact_peak_N':tip_contact_peak,'tip_contact_force_limit_N':r.get('tip_contact_force_limit_N',.5),'tip_penetration_peak_m':tip_penetration_peak,'source':str(a.source),'plan':str(a.plan),'pinch_force_N':r.get('pinch_force_N',.3),'orientation_weight':a.orientation_weight,'pot_clearance_m':a.pot_clearance,'scene':r['scene'],'pass':bool(passed and not bad and not s.warnings()),'failure':bad if bad else (None if passed else {'reason':'transport hold acceptance not reached'}),'coffee_completed':False,'warnings':s.warnings(),'initial_water_ml':800,'water_state':asdict(water),'hold_drift_m':hold_drift,'max_grasp_relative_rotation_deg':max_relative_rotation_deg,'min_table_gap_m':min_gap,'min_quasi_static_com_margin_m':min_com_margin,'balance_limitation':'fixed pelvis; COM projection against nominal foot contact polygon only, not free-base dynamic balance','max_bowl_error_m':max_error,'rows':rows,'grasp_samples':samples,'liquid_samples':liquid,'scope':'motor-only spoon transport/dip, fixed pelvis; reduced grounds/heater enabled' if brew.enabled else 'motor-only empty spoon transport/dip, fixed pelvis; no powder in this episode','peak_motor_torques_Nm':{n:float(peak[i]) for n,i in s.act_joint.items()}};(a.out/'report.json').write_text(json.dumps(report,indent=2));np.savez_compressed(a.out/'trajectory.npz',qpos=states);integration=np.zeros(mujoco.mj_stateSize(m,mujoco.mjtState.mjSTATE_INTEGRATION));mujoco.mj_getState(m,d,integration,mujoco.mjtState.mjSTATE_INTEGRATION);np.savez_compressed(a.out/'continuation.npz',integration=integration,last_target=target,kp=s.kp,kd=s.kd,body_mass=m.body_mass,body_ipos=m.body_ipos,body_inertia=m.body_inertia,body_iquat=m.body_iquat,bvh_aabb=m.bvh_aabb,dry_kettle_kg=.78);(a.out/'liquid-state.json').write_text(json.dumps(asdict(water),indent=2));print({k:v for k,v in report.items() if k not in ['rows','grasp_samples','liquid_samples','peak_motor_torques_Nm']})

brew.save(a.out)
