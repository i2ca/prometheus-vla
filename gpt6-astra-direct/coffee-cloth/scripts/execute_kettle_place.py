"""Motor-only wet kettle placement and release; no object pose edits in physics."""
import sys,json,argparse,shutil
from pathlib import Path
import numpy as np,mujoco
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'g1-cup-grasp/scripts'))
from g1_sim import G1Sim,ARM_JOINTS
from kinematics import ArmIK
from kettle_liquid import KettleWater
from liquid_mass import LiquidMassCoupler
from thermal_grip import thermal_motor_torque
from dataclasses import asdict
from collections import deque
ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--resume-seated',type=Path);ap.add_argument('--curl-release',action='store_true');ap.add_argument('--side-release',action='store_true');ap.add_argument('--freeze-torso-release',action='store_true');a=ap.parse_args();r=json.loads((a.source/'report.json').read_text());assert r['pass'];a.out.mkdir(exist_ok=False)
for p in [Path(__file__),*[Path(__file__).parent/n for n in ['kettle_liquid.py','liquid_mass.py','thermal_grip.py']]]:shutil.copy2(p,a.out/p.name)
s=G1Sim(r['scene']);m,d=s.m,s.d;ck=np.load(a.source/'continuation.npz');wet=json.loads((a.source/'liquid-state.json').read_text());water=KettleWater(initial_ml=wet['initial_ml']);coupler=LiquidMassCoupler(m,float(ck['dry_kettle_kg']))
for n,v in wet.items():setattr(water,n,v)
mujoco.mj_setState(m,d,ck['integration'],mujoco.mjtState.mjSTATE_INTEGRATION);coupler.apply(d,water);s.kp[:]=ck['kp'];s.kd[:]=ck['kd'];start=float(d.time);target=ck['last_target'].copy();hand_names=ck['hand_names'].tolist();hand=ck['hand_target'].copy();ha=np.array([m.jnt_qposadr[m.joint(n).id] for n in hand_names]);finger=np.array([m.joint(m.actuator_trnid[i,0]).name.startswith('left_hand') for i in range(m.nu)]);palm=m.body('left_wrist_yaw_link').id;jar=m.body('chaleira').id;base=m.body('base_eletrica').id;filter_id=m.body('coador').id;cup=m.body('copo').id;table=m.geom('tampo').id
ik=ArmIK(s,['waist_yaw_joint','waist_roll_joint','waist_pitch_joint']+[n.replace('right_','left_') for n in ARM_JOINTS],palm='left_wrist_yaw_link');ik.bounds=ik.bounds.copy();ik.bounds[0]=[-.7,.7];ik.bounds[1:3]=np.deg2rad([[-12,12],[-12,12]]);ik.bounds[-3:]=np.deg2rad([[-60,60],[-45,45],[-30,30]]);q=target[ik.qa].copy();nominal=q.copy();p0=d.xpos[jar].copy();R0=d.xmat[jar].reshape(3,3).copy();base0=d.xpos[base].copy();Rgoal=Rotation.from_euler('z',np.arctan2(R0[1,0],R0[0,0])).as_matrix();align=p0.copy();align[:2]=base0[:2];landing=base0+np.array([0,0,.0145]);handopen=np.zeros(len(hand));handopen[0]=hand[0]
handle=[g for g in range(m.ngeom) if (m.geom(g).name or '').startswith('handle_col')];tips=[next(g for g in range(m.ngeom) if m.geom_contype[g] and m.body(int(m.geom_bodyid[g])).name=='left_hand_'+part+'_link') for part in ['thumb_2','index_1','middle_1']];handgeoms=[g for g in range(m.ngeom) if m.geom_contype[g] and m.body(int(m.geom_bodyid[g])).name.startswith(('left_hand','left_wrist'))];torso=[g for g in range(m.ngeom) if m.geom_contype[g] and m.body(int(m.geom_bodyid[g])).name=='torso_link'];shoulder=[g for g in range(m.ngeom) if m.geom_contype[g] and m.body(int(m.geom_bodyid[g])).name.startswith(('left_shoulder_roll','left_shoulder_yaw','left_elbow','left_wrist','left_hand'))];pairs=[(g,h) for g in torso for h in shoulder]
normal=np.zeros(m.nu);thermal=np.zeros(m.nu);rows=[];samples=[];states=[];liquid=[];peak=np.zeros(m.nu);bad=None;stable=0;released=False;release_time=None;release_p=None;release_R=None;release_command=None;release_hand_start=None;release_trim=None;release_escape=None;release_data=mujoco.MjData(m);max_error=0.;dt=m.opt.timestep;good=0;rest_window=deque(maxlen=int(2/dt));rest_metrics=None;support_N=0.;contact_started=None;forces={};props=['coador','copo','pote','tampa','scoop','base_eletrica','chaleira'];body_ids=[m.body(n).id for n in props];initial_props=d.xpos[body_ids].copy()
def smooth(x):
 x=np.clip(x,0,1);return x*x*(3-2*x)
if a.resume_seated:
 snap=np.load(a.resume_seated)
 for name in ['body_mass','body_ipos','body_inertia','body_iquat','bvh_aabb']:getattr(m,name)[:]=snap[name]
 mujoco.mj_setConst(m,mujoco.MjData(m));mujoco.mj_setState(m,d,snap['integration'],mujoco.mjtState.mjSTATE_INTEGRATION);mujoco.mj_forward(m,d)
 for n,v in json.loads(str(snap['water_json'])).items():setattr(water,n,v)
 target=snap['last_target'].copy();hand=snap['hand_target'].copy();handopen[0]=hand[0];s.kp[:]=snap['kp'];s.kd[:]=snap['kd'];q=target[ik.qa].copy();nominal=q.copy();release_time=0.;release_p=d.xpos[palm].copy();release_R=d.xmat[palm].reshape(3,3).copy();release_command=d.qpos[ha].copy();release_hand_start=release_command.copy();release_escape=release_p-d.xpos[jar];release_escape[2]=0;release_escape/=np.linalg.norm(release_escape);release_escape=(d.xmat[jar].reshape(3,3)[:,1].copy()*np.sign(np.dot(release_p-d.xpos[jar],d.xmat[jar].reshape(3,3)[:,1]))+.3*d.xmat[jar].reshape(3,3)[:,0]) if a.side_release else release_escape;release_escape/=np.linalg.norm(release_escape);release_trim=ik.fk(q)[0]-release_p;start=float(d.time);base0=d.xpos[base].copy();shutil.copy2(a.resume_seated,a.out/'input-seated-checkpoint.npz')
right_roll_qa=m.jnt_qposadr[m.joint('right_shoulder_roll_joint').id];right_roll_start=float(target[right_roll_qa]);right_pitch_qa=m.jnt_qposadr[m.joint('right_shoulder_pitch_joint').id];right_pitch_start=float(target[right_pitch_qa])
for k in range(int(65/dt)):
 t=k*dt
 target[right_roll_qa]=right_roll_start if a.freeze_torso_release else right_roll_start+(-.45-right_roll_start)*smooth(t/.6)
 target[right_pitch_qa]=right_pitch_start if a.freeze_torso_release else right_pitch_start+(.9-right_pitch_start)*smooth(t/.6)
 if t<12:
  f=smooth(t/12);jpgoal=p0*(1-f)+align*f;jRgoal=Rotation.from_rotvec(f*Rotation.from_matrix(Rgoal@R0.T).as_rotvec()).as_matrix()@R0;phase='align_over_base'
 else:
  f=smooth((t-12)/12);jpgoal=align*(1-f)+landing*f;jRgoal=Rgoal;phase='lower_to_base'
 if stable>=int(1/dt) and release_time is None:release_time=t;release_p=d.xpos[palm].copy();release_R=d.xmat[palm].reshape(3,3).copy();release_command=d.qpos[ha].copy();release_hand_start=release_command.copy();release_escape=release_p-d.xpos[jar];release_escape[2]=0;release_escape/=np.linalg.norm(release_escape);release_escape=(d.xmat[jar].reshape(3,3)[:,1].copy()*np.sign(np.dot(release_p-d.xpos[jar],d.xmat[jar].reshape(3,3)[:,1]))+.3*d.xmat[jar].reshape(3,3)[:,0]) if a.side_release else release_escape;release_escape/=np.linalg.norm(release_escape);release_trim=ik.fk(q)[0]-release_p;saved=np.zeros(mujoco.mj_stateSize(m,mujoco.mjtState.mjSTATE_INTEGRATION));mujoco.mj_getState(m,d,saved,mujoco.mjtState.mjSTATE_INTEGRATION);np.savez_compressed(a.out/'seated-checkpoint.npz',integration=saved,last_target=target,hand_target=hand,hand_names=hand_names,kp=s.kp,kd=s.kd,body_mass=m.body_mass,body_ipos=m.body_ipos,body_inertia=m.body_inertia,body_iquat=m.body_iquat,bvh_aabb=m.bvh_aabb,water_json=json.dumps(asdict(water)),dry_kettle_kg=float(ck['dry_kettle_kg']))
 if release_time is not None:
  rt=t-release_time;phase='release_handle' if rt<5 else 'withdraw_hand';opening=smooth(rt/5);open_goal=hand*(1-opening)+handopen*opening;target[ha]=release_command
 else:rt=0;opening=0;target[ha]=hand
 if release_time is not None and a.freeze_torso_release:ik.bounds[:3]=np.stack([q[:3]-1e-7,q[:3]+1e-7],axis=1)
 if k%10==0:
  if k:
   flow=water.step(dt*10,d.xpos[jar],d.xmat[jar].reshape(3,3),d.xpos[filter_id],d.xmat[filter_id].reshape(3,3),d.xpos[cup],d.xmat[cup].reshape(3,3));liquid.append({'t':float(d.time),**flow})
  if release_time is not None and a.side_release:
   release_command=release_hand_start.copy();release_command[3:5]=release_hand_start[3:5]*(1-np.clip(rt/1,0,1));release_command[1:3]*=1-smooth((rt-4)/3);release_command[5:]*=1-smooth((rt-4)/3);target[ha]=release_command
  if release_time is not None and a.curl_release:
   release_command=release_hand_start.copy();release_command[2]=release_hand_start[2]+.4*smooth(rt/1);release_command[3:]=release_hand_start[3:]*(1-smooth((rt-1)/3));target[ha]=release_command
  if release_time is not None and not a.curl_release and not a.side_release:
   def release_residual(x):
    release_data.qpos[:]=d.qpos;release_data.qpos[ha]=x;mujoco.mj_kinematics(m,release_data);mujoco.mj_comPos(m,release_data);gaps=[min(0.,mujoco.mj_geomDistance(m,release_data,g,m.geom('chaleira_hot_body').id,.02,None)-.005) for g in handgeoms];return np.r_[(x-open_goal)*.05,np.array(gaps)*1000]
   limits=np.array([m.joint(n).range for n in hand_names]);fit_hand=least_squares(release_residual,np.clip(release_command,limits[:,0]+1e-9,limits[:,1]-1e-9),bounds=limits.T,max_nfev=50);release_delta=fit_hand.x-release_command;release_command+=release_delta*min(1.,.01/max(1e-12,np.max(np.abs(release_delta))));target[ha]=release_command
  coupler.apply(d,water);actual_R=d.xmat[palm].reshape(3,3);local=actual_R.T@(d.xpos[jar]-d.xpos[palm]);localR=actual_R.T@d.xmat[jar].reshape(3,3)
  if release_time is not None and a.side_release:
   actual_goal=release_p+release_escape*.06*np.clip(rt/1,0,1)+np.array([0,0,.10*smooth((rt-4)/6)]);release_trim=np.clip(release_trim+np.clip(.2*(actual_goal-d.xpos[palm]),-.001,.001),-.02,.02)
  def residual(x):
   pp,rr=ik.fk(x);gaps=[min(0,mujoco.mj_geomDistance(m,ik.d,g,h,.02,None)-.005) for g,h in pairs]
   if release_time is None:pos=pp+rr@local-jpgoal;rot=Rotation.from_matrix(rr@localR@jRgoal.T).as_rotvec()
   else:
    escape=.06*np.clip(rt/1,0,1) if a.side_release else .03*smooth((rt-1)/3);pos=pp-(release_p+release_escape*escape+np.array([0,0,.10*smooth((rt-4)/6)])+(release_trim if a.side_release else 0));rot=Rotation.from_matrix(rr@release_R.T).as_rotvec()
   return np.r_[pos*1000,rot*(50 if release_time is not None else 3),(x-nominal)*.3,np.array(gaps)*10000]
  fit=least_squares(residual,np.clip(q,ik.bounds[:,0]+1e-9,ik.bounds[:,1]-1e-9),bounds=ik.bounds.T,max_nfev=80);q+=np.clip(fit.x-q,-.015,.015) if not (a.curl_release and release_time is not None and rt<1) else 0;max_error=max(max_error,float(np.linalg.norm(residual(fit.x)[:3])/1000));target[ik.qa]=q
  normal[:]=0
  for g in tips:
   options=[]
   for h in handle:
    pts=np.zeros(6);gap=mujoco.mj_geomDistance(m,d,g,h,.03,pts);options.append((gap,pts))
   gap,pts=min(options,key=lambda x:x[0])
   if abs(gap)<1e-8 or gap>=.03:continue
   n=(pts[3:]-pts[:3])*np.sign(gap);n/=max(np.linalg.norm(n),1e-12);jac=np.zeros((3,m.nv));jr=np.zeros((3,m.nv));mujoco.mj_jac(m,d,jac,jr,pts[:3],int(m.geom_bodyid[g]));normal+=(jac[:,s.vadr].T@(n*12*(1-smooth(rt/.5) if release_time is not None else 1)))*finger
  thermal,_=thermal_motor_torque(m,d,s.vadr,include_arm=False)
 # Transfer weight to base during final descent, then remove payload feed-forward.
 load_scale=(1-smooth((t-contact_started)/3) if contact_started is not None else 1.) if release_time is None else 0.
 jac=np.zeros((3,m.nv));jr=np.zeros((3,m.nv));mujoco.mj_jac(m,d,jac,jr,d.xipos[jar],palm);tau=s.kp*(target[s.qadr]-d.qpos[s.qadr])-s.kd*d.qvel[s.vadr]+d.qfrc_bias[s.vadr]+normal+thermal+(jac[:,s.vadr].T@(-m.opt.gravity*m.body_mass[jar]))*(~finger)*load_scale;d.ctrl[:]=np.clip(tau,m.actuator_ctrlrange[:,0],m.actuator_ctrlrange[:,1]);peak=np.maximum(peak,np.abs(d.ctrl));mujoco.mj_step(m,d)
 supports=set();support_N=0.;forces={}
 for ci,c in enumerate(d.contact):
  if c.dist>=0:continue
  gs=[int(c.geom1),int(c.geom2)];bn=[m.body(int(m.geom_bodyid[g])).name for g in gs];gn=[m.geom(g).name or '' for g in gs];allowed=any(n.startswith('left_hand') for n in bn) and any(n.startswith('handle_col') for n in gn)
  if not allowed and any(n.startswith(('left_','right_','torso','pelvis','waist','head')) for n in bn):bad={'t':float(d.time),'bodies':bn,'geoms':gn}
  if 'chaleira' in bn:
   other=bn[1-bn.index('chaleira')];cf=np.zeros(6);mujoco.mj_contactForce(m,d,ci,cf)
   if other=='base_eletrica':support_N+=float(cf[0]);supports.add(other)
   elif other.startswith('left_hand'):forces[other]=forces.get(other,0)+float(cf[0])
   else:bad={'t':float(d.time),'reason':'kettle support outside base','body':other}
 tilt=float(np.rad2deg(np.arccos(np.clip(d.xmat[jar].reshape(3,3)[2,2],-1,1))));center_error=float(np.linalg.norm(d.xpos[jar,:2]-d.xpos[base,:2]));gap=min(mujoco.mj_geomDistance(m,d,g,table,1,None) for g in handgeoms);wrist=np.rad2deg(d.qpos[ik.qa[-3:]])
 if support_N>1 and t>15 and contact_started is None:contact_started=t
 seated=support_N>m.body_mass[jar]*9.81*.75 and center_error<.012 and tilt<5;stable=stable+1 if seated else 0
 if release_time is not None and rt>11:
  jar_va=m.jnt_dofadr[m.joint('chaleira_livre').id];rest_window.append({'support':support_N,'error':center_error,'tilt':tilt,'hand':sum(forces.values()),'linear':float(np.linalg.norm(d.qvel[jar_va:jar_va+3])),'angular':float(np.linalg.norm(d.qvel[jar_va+3:jar_va+6])),'position':d.xpos[jar].copy(),'rotation':d.xmat[jar].reshape(3,3).copy()})
  if len(rest_window)==rest_window.maxlen and k%50==0:
   window=list(rest_window);mean=float(np.mean([x['support'] for x in window]));lin=float(np.sqrt(np.mean([x['linear']**2 for x in window])));ang=float(np.sqrt(np.mean([x['angular']**2 for x in window])));drift=max(float(np.linalg.norm(x['position']-window[0]['position'])) for x in window);rd=max(float(np.arccos(np.clip((np.trace(x['rotation']@window[0]['rotation'].T)-1)/2,-1,1))) for x in window);weight=m.body_mass[jar]*9.81
   rest_metrics={'mean_support_N':mean,'weight_N':float(weight),'rms_linear_m_s':lin,'rms_angular_rad_s':ang,'max_drift_m':drift,'max_rotation_drift_deg':float(np.rad2deg(rd)),'samples':len(window)}
   passed_window=all(x['error']<.012 and x['tilt']<5 and x['hand']<.1 for x in window) and .9*weight<mean<1.1*weight and lin<.01 and ang<.2 and drift<.001 and rd<np.deg2rad(.5)
   if passed_window:good=len(window)
 if gap<.02:bad={'t':float(d.time),'reason':'hand-table clearance','gap':gap}
 if np.any(np.abs(wrist)>[62,47,32]):bad={'t':float(d.time),'reason':'wrist posture tolerance','wrist':wrist.tolist()}
 if water.spilled_ml>r['water_state']['spilled_ml']+.5:bad={'t':float(d.time),'reason':'spill during placement'}
 if np.linalg.norm(d.xpos[base]-base0)>.01:bad={'t':float(d.time),'reason':'base displacement exceeds1cm'}
 if k%16==0:
  rows.append({'t':float(d.time),'phase':phase,'table_gap_m':gap,'prop_displacements_m':np.linalg.norm(d.xpos[body_ids]-initial_props,axis=1).tolist()});states.append(d.qpos.copy());samples.append({'t':float(d.time),'phase':phase,'jar_tilt_deg':tilt,'base_support_N':support_N,'center_error_m':center_error,'finger_normal_force_N':forces,'wrist_deg':wrist.tolist()})
 if k%2500==0:print({'t':t,'phase':phase,'tilt':tilt,'support_N':support_N,'center_error_m':center_error},flush=True)
 if bad or good>=int(2/dt):break
report={'model':'gpt-6-astra','scene':r['scene'],'source':str(a.source),'seated_checkpoint':str(a.resume_seated) if a.resume_seated else None,'curl_release':a.curl_release,'side_release':a.side_release,'freeze_torso_release':a.freeze_torso_release,'pass':bad is None and good>=int(2/dt) and not s.warnings(),'failure':bad,'warnings':s.warnings(),'coffee_completed':False,'water_state':asdict(water),'initial_water_ml':water.initial_ml,'liquid_samples':liquid,'rows':rows,'grasp_samples':samples,'max_ik_error_m':max_error,'hold_good_steps':good,'rest_window_metrics':rest_metrics,'peak_motor_torques_Nm':{n:float(peak[i]) for n,i in s.act_joint.items()},'scope':'physical wet kettle placement and release, original whole liquid ledger preserved, fixed robot pelvis; no heating or coffee dosing'}
if not report['pass'] and bad is None:report['failure']={'reason':'placement/release acceptance not reached'}
(a.out/'report.json').write_text(json.dumps(report,indent=2));np.savez_compressed(a.out/'trajectory.npz',qpos=states);np.save(a.out/'final-qpos.npy',d.qpos);np.save(a.out/'final-qvel.npy',d.qvel);integration=np.zeros(mujoco.mj_stateSize(m,mujoco.mjtState.mjSTATE_INTEGRATION));mujoco.mj_getState(m,d,integration,mujoco.mjtState.mjSTATE_INTEGRATION);np.savez_compressed(a.out/'continuation.npz',integration=integration,last_target=target,hand_target=target[ha],hand_names=hand_names,kp=s.kp,kd=s.kd,body_mass=m.body_mass,body_ipos=m.body_ipos,body_inertia=m.body_inertia,body_iquat=m.body_iquat,water_ml=water.source_ml,dry_kettle_kg=float(ck['dry_kettle_kg']));(a.out/'liquid-state.json').write_text(json.dumps(asdict(water),indent=2));print({k:v for k,v in report.items() if k not in ['rows','grasp_samples','liquid_samples','peak_motor_torques_Nm']})
