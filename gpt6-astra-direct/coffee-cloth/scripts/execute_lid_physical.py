"""Physical lid reach/pinch/lift from wet kettle checkpoint, motors only."""
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
from liquid_mass import LiquidMassCoupler
from kettle_liquid import KettleWater
from brew_state import BrewState
ap=argparse.ArgumentParser();ap.add_argument('--damped-arms',action='store_true');ap.add_argument('--source',type=Path);ap.add_argument('--adaptive-grip',action='store_true');ap.add_argument('--out',type=Path,required=True);ap.add_argument('--plan',type=Path,default=Path('results/lid-connect-003'));ap.add_argument('--reach-only',action='store_true');ap.add_argument('--index-servo',action='store_true');ap.add_argument('--pinch-force',type=float,default=3.5);ap.add_argument('--finger-kp',type=float,default=2.0);ap.add_argument('--normal-pinch',action='store_true');ap.add_argument('--two-finger',action='store_true');ap.add_argument('--close-gain',type=float,default=3);a=ap.parse_args();pr=json.loads((a.plan/'report.json').read_text());assert pr['pass'];source=a.source or Path(pr['source']);assert not (source/'INVALIDATED.json').exists(),'source invalidated by audit';r=json.loads((source/'report.json').read_text());a.out.mkdir(exist_ok=False)
for p in [Path(__file__),Path('scripts/liquid_mass.py'),Path('scripts/kettle_liquid.py')]+[Path('scripts')/n for n in ['brew_state.py','coffee_grounds.py','grounds_mass.py','kettle_heater.py','elbow_anatomy.py']]:shutil.copy2(p,a.out/p.name)
s=G1Sim(r['scene']);m,d=s.m,s.d;brew=BrewState(m,source);ck=np.load(source/'continuation.npz');wet=json.loads((source/'liquid-state.json').read_text());water=KettleWater(initial_ml=wet['initial_ml']);coupler=LiquidMassCoupler(m,float(ck['dry_kettle_kg']))
for n,v in wet.items():setattr(water,n,v)
mujoco.mj_setState(m,d,ck['integration'],mujoco.mjtState.mjSTATE_INTEGRATION);coupler.apply(d,water);s.kp[:]=ck['kp'];s.kd[:]=ck['kd'];target=ck['last_target'].copy();start=float(d.time);dt=m.opt.timestep
if a.damped_arms:
 for n,i in s.act_joint.items():
  if 'waist' in n:s.kd[i]=15
  elif 'shoulder' in n:s.kd[i]=5
  elif 'elbow' in n:s.kd[i]=3
  elif 'wrist' in n:s.kd[i]=2.5
path=np.load(a.plan/'path.npz');poses=path['qpos'];phases=path['phases'];names=pr['joint_names'];ik=ArmIK(s,names);bounds=ik.bounds.copy();bounds[0]=[-.7,.7];bounds[1:3]=np.deg2rad([[-12,12],[-12,12]])
anatomy=ElbowAnatomy(m);anatomy.bound_search(m,ik.joint_names,bounds)
for j in [7,14]:bounds[j:j+3]=np.deg2rad([[-60,60],[-45,45],[-30,30]])
ha=m.jnt_qposadr[[m.joint(n).id for n in HAND_JOINTS]];moving=np.r_[ik.qa,ha];durations=np.maximum(.016,np.max(np.abs(np.diff(poses[:,moving],axis=0)),axis=1)/.20);times=np.r_[0,np.cumsum(durations)];end=float(times[-1]);lid=m.body('tampa').id;palm=m.body('right_wrist_yaw_link').id;left=m.body('left_wrist_yaw_link').id;pot=m.body('pote').id;jar=m.body('chaleira').id;filter_id=m.body('coador').id;cup=m.body('copo').id;props=[jar,filter_id,cup,pot,m.body('base_eletrica').id,m.body('scoop').id];prop0=d.xpos[props].copy();lid0=d.xpos[lid].copy();table=m.geom('tampo').id
handgeoms=[g for g in range(m.ngeom) if m.geom_contype[g] and m.body(int(m.geom_bodyid[g])).name.startswith(('left_hand','left_wrist','right_hand','right_wrist'))]
arm,seed,fps=load_seed(0);gi=int(np.flatnonzero(seed[:,4]>.5)[0]);hrange=m.jnt_range[[m.joint(n).id for n in HAND_JOINTS]];close=np.clip(seed[0]+a.close_gain*(seed[gi]-seed[0]),hrange[:,0],hrange[:,1]);opened=poses[-1,ha].copy()
if a.two_finger:close[5:]=opened[5:];close[0]=opened[0]
normal=np.zeros(m.nu);finger_mask=np.array([m.joint(m.actuator_trnid[i,0]).name.startswith('right_hand') for i in range(m.nu)]);knob_geoms=[g for g in range(m.ngeom) if m.geom_contype[g] and m.geom_bodyid[g]==lid and (m.geom(g).name or '').startswith('tampa_slice') and int(m.geom(g).name.split('_')[-1])>=58];pinch_tips=[next(g for g in range(m.ngeom) if m.geom_contype[g] and m.body(int(m.geom_bodyid[g])).name=='right_hand_'+n+'_link') for n in ['thumb_2','index_1']]
for n in HAND_JOINTS:s.kp[s.act_joint[n]]=a.finger_kp;s.kd[s.act_joint[n]]=.15
hand_offset=np.zeros(7);lift_contact_shift=np.zeros(3);thumb_force=0.;grip_lost=0
hold_window=deque(maxlen=int(2/dt));hold_drift=None
grip_stable=0;lift_start=None;pinch_p=None;pinch_R=None;pinch_left=None;pinch_shift=np.zeros(3);index_force=0.
q=poses[-1,ik.qa].copy();liftp=None;liftR=None;leftp=None;leftR=None;trim=np.zeros(3);rows=[];states=[];samples=[];liquid=[];peak=np.zeros(m.nu);bad=None;good=0;maxlift=0.;min_gap=1.;maxtrack=0.
brew.apply(d,water)

def smooth(x):x=np.clip(x,0,1);return x*x*(3-2*x)
for k in range(int((end+(2 if a.reach_only else 30))/dt)):
 t=k*dt;idx=min(int(np.searchsorted(times,t,side='right')-1),len(poses)-2);fraction=np.clip((t-times[idx])/(times[idx+1]-times[idx]),0,1);target[moving]=poses[idx,moving]*(1-fraction)+poses[idx+1,moving]*fraction;phase=str(phases[idx+1]) if t<end else 'settle_reach'
 if t>end+2 and not a.reach_only:
  phase='pinch';target[ha]=opened if a.normal_pinch else opened+(close-opened)*smooth((t-end-2)/3)
 if a.index_servo and t>end+2 and lift_start is None:
  if pinch_p is None:pinch_p,pinch_R=ik.fk(q);pinch_left=d.xpos[left].copy()
  if k%10==0:
   options=[]
   for h in knob_geoms:
    pts=np.zeros(6);gap_=mujoco.mj_geomDistance(m,d,pinch_tips[1],h,.04,pts);options.append((gap_,pts))
   gap_,pts=min(options,key=lambda x:x[0]);direction=(pts[3:]-pts[:3])*np.sign(gap_);direction/=max(np.linalg.norm(direction),1e-12);advance=min(.00008,max(0,gap_)*.2) if gap_>0 else np.clip((1.5-index_force)*.00001,-.00003,.00003);pinch_shift+=direction*advance;pinch_shift*=min(1,.010/max(np.linalg.norm(pinch_shift),1e-12))
   def press_residual(x):
    pp,rr=ik.fk(x);return np.r_[(pp-pinch_p-pinch_shift)*1000,Rotation.from_matrix(rr@pinch_R.T).as_rotvec()*30,(ik.d.xpos[left]-pinch_left)*500,(x-q)*.1,anatomy.penalty(ik.d)]
   fit=least_squares(press_residual,np.clip(q,bounds[:,0]+1e-9,bounds[:,1]-1e-9),bounds=bounds.T,max_nfev=50);q+=np.clip(fit.x-q,-.004,.004)
  target[ik.qa]=q
 if lift_start is not None and not a.reach_only:
  phase='lift' if t<lift_start+6 else 'hold_lid'
  if liftp is None:liftp=d.xpos[palm].copy();liftR=d.xmat[palm].reshape(3,3).copy();leftp=d.xpos[left].copy();leftR=d.xmat[left].reshape(3,3).copy();q=target[ik.qa].copy();trim=ik.fk(q)[0]-liftp
  goal=liftp+np.array([0,0,.08*smooth((t-lift_start)/6)])+lift_contact_shift
  if k%10==0:
   if a.adaptive_grip:
    opts=[]
    for h in knob_geoms:
     pts=np.zeros(6);gg=mujoco.mj_geomDistance(m,d,pinch_tips[1],h,.04,pts);opts.append((gg,pts))
    gg,pts=min(opts,key=lambda x:x[0]);nn=(pts[3:]-pts[:3])*np.sign(gg);nn/=max(np.linalg.norm(nn),1e-12);step=np.clip(max(gg,0)*.15+(1.5-index_force)*.000008,-.00003,.00008);lift_contact_shift+=nn*step;lift_contact_shift*=min(1,.008/max(np.linalg.norm(lift_contact_shift),1e-12))
   trim=np.clip(trim+np.clip(.15*(goal-d.xpos[palm]),-.0005,.0005),-.015,.015)
   def residual(x):
    pp,rr=ik.fk(x);return np.r_[(pp-goal-trim)*1000,Rotation.from_matrix(rr@liftR.T).as_rotvec()*30,(ik.d.xpos[left]-leftp)*500,Rotation.from_matrix(ik.d.xmat[left].reshape(3,3)@leftR.T).as_rotvec(),(x-q)*.1,anatomy.penalty(ik.d)]
   fit=least_squares(residual,np.clip(q,bounds[:,0]+1e-9,bounds[:,1]-1e-9),bounds=bounds.T,max_nfev=60);q+=np.clip(fit.x-q,-.006,.006)
  target[ik.qa]=q
 if k%10==0 and k:
  brew.step(d,.02,water);flow=water.step(dt*10,d.xpos[jar],d.xmat[jar].reshape(3,3),d.xpos[filter_id],d.xmat[filter_id].reshape(3,3),d.xpos[cup],d.xmat[cup].reshape(3,3));coupler.apply(d,water);liquid.append({'t':float(d.time),**flow})
 if a.normal_pinch and t>end+2 and k%10==0:
  normal[:]=0
  for g in pinch_tips:
   options=[]
   for h in knob_geoms:
    pts=np.zeros(6);gap_=mujoco.mj_geomDistance(m,d,g,h,.04,pts);options.append((gap_,pts))
   gap_,pts=min(options,key=lambda x:x[0])
   if abs(gap_)<1e-8 or gap_>=.04:continue
   direction=(pts[3:]-pts[:3])*np.sign(gap_);direction/=max(np.linalg.norm(direction),1e-12);jac=np.zeros((3,m.nv));jr=np.zeros((3,m.nv));mujoco.mj_jac(m,d,jac,jr,pts[:3],int(m.geom_bodyid[g]));normal+=jac[:,s.vadr].T@(direction*a.pinch_force*smooth((t-end-2)/1.5))*finger_mask
 if a.adaptive_grip and t>end+2:
  if k%10==0:hand_offset[0]=np.clip(hand_offset[0]+(-.001 if thumb_force<1.5 else (.0005 if thumb_force>3 else 0)),-.25,.05)
  target[ha]+=hand_offset
 tau=s.kp*(target[s.qadr]-d.qpos[s.qadr])-s.kd*d.qvel[s.vadr]+d.qfrc_bias[s.vadr]+normal;d.ctrl[:]=np.clip(tau,m.actuator_ctrlrange[:,0],m.actuator_ctrlrange[:,1]);peak=np.maximum(peak,np.abs(d.ctrl));mujoco.mj_step(m,d);forces={};grip_forces={};lid_support=[]
 for ci,c in enumerate(d.contact):
  if c.dist>=0:continue
  gs=[int(c.geom1),int(c.geom2)];bn=[m.body(int(m.geom_bodyid[g])).name for g in gs];robot=any(n.startswith(('left_','right_','torso','waist','pelvis','head')) for n in bn);allowed='tampa' in bn and any(n.startswith('right_hand') for n in bn) and phase in ['approach_lid','settle_reach','pinch','lift','hold_lid']
  if robot and not allowed:bad={'reason':'forbidden contact','bodies':bn,'geoms':[m.geom(g).name for g in gs],'penetration_m':float(-c.dist)}
  if 'tampa' in bn:
   other=bn[1-bn.index('tampa')];force=np.zeros(6);mujoco.mj_contactForce(m,d,ci,force)
   if other.startswith('right_hand'):
    forces[other]=forces.get(other,0)+float(force[0]);lid_geom=gs[bn.index('tampa')];normal_z=float(c.frame[2])*(1 if bn[1]=='tampa' else -1)
    if lid_geom in knob_geoms and normal_z>-.6:grip_forces[other]=grip_forces.get(other,0)+float(force[0])
   else:lid_support.append(other)
 gap=min(mujoco.mj_geomDistance(m,d,g,table,1,None) for g in handgeoms);min_gap=min(min_gap,gap);wrists=np.rad2deg(d.qpos[ik.qa[np.r_[7:10,14:17]]]);lift=float(d.xpos[lid,2]-lid0[2]);maxlift=max(maxlift,lift);track=float(np.max(np.abs(target[ik.qa]-d.qpos[ik.qa])));maxtrack=max(maxtrack,track)
 if gap<.020:bad={'reason':'hand table clearance','gap_m':float(gap)}
 if np.any(np.abs(wrists)>[62,47,32,62,47,32]):bad={'reason':'wrist bounds','degrees':wrists.tolist()}
 if not anatomy.valid(d,tolerance_deg=2):bad={'reason':'geometric elbow flexion outside task band','degrees':anatomy.angles(d).tolist()}
 if water.spilled_ml>wet['spilled_ml']+.5:bad={'reason':'spill'}
 if np.max(np.linalg.norm(d.xpos[props]-prop0,axis=1))>.01:bad={'reason':'prop moved more than1cm','displacements_m':{m.body(b).name:float(np.linalg.norm(d.xpos[b]-p)) for b,p in zip(props,prop0)}}
 index_force=sum(f for n,f in grip_forces.items() if 'index' in n);thumb_force=sum(f for n,f in grip_forces.items() if 'thumb' in n)
 opposing=any('thumb' in n and f>.6 for n,f in grip_forces.items()) and any('index' in n and f>.6 for n,f in grip_forces.items());grip_stable=grip_stable+1 if phase=='pinch' and opposing else 0
 if grip_stable>=int(1/dt) and lift_start is None:lift_start=t
 if lift_start is not None and not opposing:grip_lost+=1
 else:grip_lost=0
 if grip_lost>int(.3/dt):bad={'reason':'opposing knob grip lost during lift'}
 if t>end+16 and lift_start is None and not a.reach_only:bad={'reason':'opposing grip not established'}
 if phase=='hold_lid' and lift>.05 and opposing and not lid_support:hold_window.append(d.xpos[lid].copy())
 else:hold_window.clear()
 good=0
 if len(hold_window)==hold_window.maxlen and k%50==0:
  hold_drift=float(np.max(np.linalg.norm(np.array(hold_window)-hold_window[0],axis=1)));good=len(hold_window) if hold_drift<.002 else 0
 if k%16==0:
  rows.append({'t':float(d.time),'phase':phase,'table_gap_m':float(gap)});states.append(d.qpos.copy());samples.append({'t':float(d.time),'phase':phase,'lift_m':lift,'finger_force_N':forces,'qualified_knob_force_N':grip_forces,'other_lid_contacts':lid_support,'elbow_flexion_deg':anatomy.angles(d).tolist(),'wrist_deg':wrists.tolist(),'tracking_error_rad':track})
 if k%2500==0:print({'time_s':t,'reach_duration_s':end,'phase':phase,'lift_m':lift,'fingers':forces,'gap_m':gap},flush=True)
 if s.warnings():bad={'reason':'MuJoCo numerical warning','warnings':s.warnings()}
 if bad or good>=int(2/dt):break
passed=bad is None and not s.warnings() and (a.reach_only or good>=int(2/dt));report={'model':'gpt-6-astra','source':str(source),'plan':str(a.plan),'scene':r['scene'],'pass':passed,'failure':bad if bad else (None if passed else {'reason':'lid lift acceptance not reached'}),'scope':'physical reach only' if a.reach_only else 'physical diagonal lid pinch and8cm lift, fixed pelvis, privileged state feedback','coffee_completed':False,'initial_water_ml':water.initial_ml,'two_finger':a.two_finger,'normal_pinch':a.normal_pinch,'index_servo':a.index_servo,'adaptive_grip':a.adaptive_grip,'hand_offset':hand_offset.tolist(),'damped_arms':a.damped_arms,'hold_drift_m':hold_drift,'pinch_force_N':a.pinch_force,'finger_kp':a.finger_kp,'warnings':s.warnings(),'max_lid_lift_m':maxlift,'hold_good_steps':good,'min_hand_table_gap_m':float(min_gap),'max_joint_tracking_error_rad':maxtrack,'water_state':asdict(water),'rows':rows,'grasp_samples':samples,'liquid_samples':liquid,'peak_motor_torques_Nm':{n:float(peak[i]) for n,i in s.act_joint.items()}}
(a.out/'report.json').write_text(json.dumps(report,indent=2));np.savez_compressed(a.out/'trajectory.npz',qpos=states);integration=np.zeros(mujoco.mj_stateSize(m,mujoco.mjtState.mjSTATE_INTEGRATION));mujoco.mj_getState(m,d,integration,mujoco.mjtState.mjSTATE_INTEGRATION);np.savez_compressed(a.out/'continuation.npz',integration=integration,last_target=target,kp=s.kp,kd=s.kd,body_mass=m.body_mass,body_ipos=m.body_ipos,body_inertia=m.body_inertia,body_iquat=m.body_iquat,bvh_aabb=m.bvh_aabb,dry_kettle_kg=float(ck['dry_kettle_kg']));(a.out/'liquid-state.json').write_text(json.dumps(asdict(water),indent=2));print({k:v for k,v in report.items() if k not in ['rows','grasp_samples','liquid_samples','peak_motor_torques_Nm']})

brew.save(a.out)
