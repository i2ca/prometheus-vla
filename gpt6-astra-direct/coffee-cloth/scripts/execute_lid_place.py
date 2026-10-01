"""Transport, support, release and withdraw from a physically held lid."""
import argparse,json,sys,shutil
from pathlib import Path
from dataclasses import asdict
from collections import deque
import mujoco,numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'g1-cup-grasp/scripts'))
from g1_sim import G1Sim,ARM_JOINTS,HAND_JOINTS
from kinematics import ArmIK
from elbow_anatomy import ElbowAnatomy
from liquid_mass import LiquidMassCoupler
from kettle_liquid import KettleWater
from brew_state import BrewState
ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,default=Path('results/lid-physical-011'));ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();source=a.source;assert not (source/'INVALIDATED.json').exists(),'source invalidated by audit';r=json.loads((source/'report.json').read_text());assert r['pass'];a.out.mkdir(exist_ok=False)
for p in [Path(__file__),Path('scripts/liquid_mass.py'),Path('scripts/kettle_liquid.py')]+[Path('scripts')/n for n in ['brew_state.py','coffee_grounds.py','grounds_mass.py','kettle_heater.py','elbow_anatomy.py']]:shutil.copy2(p,a.out/p.name)
s=G1Sim(r['scene']);m,d=s.m,s.d;brew=BrewState(m,source);ck=np.load(source/'continuation.npz');coupler=LiquidMassCoupler(m,float(ck['dry_kettle_kg']));water=KettleWater(initial_ml=800)
for n,v in json.loads((source/'liquid-state.json').read_text()).items():setattr(water,n,v)
for n in ['body_mass','body_ipos','body_inertia','body_iquat','bvh_aabb']:getattr(m,n)[:]=ck[n]
mujoco.mj_setConst(m,mujoco.MjData(m));mujoco.mj_setState(m,d,ck['integration'],mujoco.mjtState.mjSTATE_INTEGRATION);mujoco.mj_forward(m,d);s.kp[:]=ck['kp'];s.kd[:]=ck['kd'];target=ck['last_target'].copy();names=['waist_yaw_joint','waist_roll_joint','waist_pitch_joint']+list(ARM_JOINTS)+[n.replace('right_','left_') for n in ARM_JOINTS];ik=ArmIK(s,names);bounds=ik.bounds.copy();bounds[0]=[-.7,.7];bounds[1:3]=np.deg2rad([[-12,12],[-12,12]])
anatomy=ElbowAnatomy(m);anatomy.bound_search(m,ik.joint_names,bounds)
for j in [7,14]:bounds[j:j+3]=np.deg2rad([[-60,60],[-45,45],[-30,30]])
q=target[ik.qa].copy();ha=m.jnt_qposadr[[m.joint(n).id for n in HAND_JOINTS]];hand=target[ha].copy();lid=m.body('tampa').id;palm=m.body('right_wrist_yaw_link').id;left=m.body('left_wrist_yaw_link').id;lp=d.xpos[left].copy();lr=d.xmat[left].reshape(3,3).copy();p0=d.xpos[lid].copy();R0=d.xmat[lid].reshape(3,3).copy();Rgoal=Rotation.from_euler('z',np.arctan2(R0[1,0],R0[0,0])).as_matrix();goal=np.array([.42,-.26,.751]);hover=goal.copy();hover[2]=p0[2];props=['chaleira','coador','copo','pote','base_eletrica','scoop'];ids=[m.body(n).id for n in props];prop0=d.xpos[ids].copy();table=m.geom('tampo').id;handgeoms=[g for g in range(m.ngeom) if m.geom_contype[g] and m.body(int(m.geom_bodyid[g])).name.startswith(('left_hand','left_wrist','right_hand','right_wrist'))];knob=[g for g in range(m.ngeom) if m.geom_bodyid[g]==lid and (m.geom(g).name or '').startswith('tampa_slice') and int(m.geom(g).name.split('_')[-1])>=58];tips=[next(g for g in range(m.ngeom) if m.geom_contype[g] and m.body(int(m.geom_bodyid[g])).name=='right_hand_'+n+'_link') for n in ['thumb_2','index_1']];finger=np.array([m.joint(m.actuator_trnid[i,0]).name.startswith('right_hand') for i in range(m.nu)]);normal=np.zeros(m.nu);start=float(d.time);dt=m.opt.timestep;rows=[];states=[];samples=[];liquid=[];bad=None;stable=0;release=None;releasep=None;releaseR=None;releaseq=None;release_trim=None;peak=np.zeros(m.nu);window=deque(maxlen=int(2/dt));rest=None;passed=False;grip_lost=0

brew.apply(d,water)

def smooth(x):x=np.clip(x,0,1);return x*x*(3-2*x)
for k in range(int(40/dt)):
 t=k*dt;target[ha]=hand;phase='transport_lid' if t<8 else 'lower_lid';f=smooth(t/8);pgoal=p0*(1-f)+hover*f if t<8 else hover*(1-smooth((t-8)/8))+goal*smooth((t-8)/8);rgoal=Rotation.from_rotvec(f*Rotation.from_matrix(Rgoal@R0.T).as_rotvec()).as_matrix()@R0
 if stable>=int(.6/dt) and release is None:release=t;releasep=d.xpos[palm].copy();releaseR=d.xmat[palm].reshape(3,3).copy();releaseq=hand.copy();release_trim=ik.fk(q)[0]-releasep
 if release is not None:
  rt=t-release;phase='release_lid' if rt<2 else 'withdraw_lid_hand';target[ha[0]]=min(m.joint(HAND_JOINTS[0]).range[1],releaseq[0]+.45*smooth((rt-.5)/1));palmgoal=releasep+np.array([0,0,.10*smooth((rt-1.5)/4)])-releaseR[:,0]*.04*smooth((rt-2)/4)
 else:rt=0
 if k%10==0:
  if k:
   bs=[m.body(n).id for n in ['chaleira','coador','copo']];brew.step(d,.02,water);flow=water.step(.02,*sum(([d.xpos[b],d.xmat[b].reshape(3,3)] for b in bs),[]));liquid.append({'t':float(d.time),**flow});coupler.apply(d,water)
  actualR=d.xmat[palm].reshape(3,3);local=actualR.T@(d.xpos[lid]-d.xpos[palm]);localR=actualR.T@d.xmat[lid].reshape(3,3)
  if release is not None:release_trim=np.clip(release_trim+np.clip(.15*(palmgoal-d.xpos[palm]),-.0005,.0005),-.015,.015)
  def residual(x):
   pp,rr=ik.fk(x)
   if release is None:pos=pp+rr@local-pgoal;rot=Rotation.from_matrix(rr@localR@rgoal.T).as_rotvec()
   else:pos=pp-palmgoal-release_trim;rot=Rotation.from_matrix(rr@releaseR.T).as_rotvec()
   return np.r_[pos*1000,rot*20,(ik.d.xpos[left]-lp)*500,Rotation.from_matrix(ik.d.xmat[left].reshape(3,3)@lr.T).as_rotvec()*.3,(x-q)*.1,anatomy.penalty(ik.d)]
  fit=least_squares(residual,np.clip(q,bounds[:,0]+1e-9,bounds[:,1]-1e-9),bounds=bounds.T,max_nfev=70);q+=np.clip(fit.x-q,-.008,.008);normal[:]=0
  for g in tips:
   candidates=[]
   for h in knob:
    pts=np.zeros(6);gap_=mujoco.mj_geomDistance(m,d,g,h,.04,pts);candidates.append((gap_,pts))
   gap_,pts=min(candidates,key=lambda x:x[0])
   if abs(gap_)<1e-8 or gap_>=.04:continue
   n=(pts[3:]-pts[:3])*np.sign(gap_);n/=max(np.linalg.norm(n),1e-12);jac=np.zeros((3,m.nv));jr=np.zeros((3,m.nv));mujoco.mj_jac(m,d,jac,jr,pts[:3],int(m.geom_bodyid[g]));normal+=(jac[:,s.vadr].T@(n*r.get('pinch_force_N',5)*(1-smooth(rt/.5) if release is not None else 1)))*finger
 target[ik.qa]=q;tau=s.kp*(target[s.qadr]-d.qpos[s.qadr])-s.kd*d.qvel[s.vadr]+d.qfrc_bias[s.vadr]+normal;d.ctrl[:]=np.clip(tau,m.actuator_ctrlrange[:,0],m.actuator_ctrlrange[:,1]);peak=np.maximum(peak,np.abs(d.ctrl));mujoco.mj_step(m,d);forces={};support=0.
 for ci,c in enumerate(d.contact):
  if c.dist>=0:continue
  gs=[int(c.geom1),int(c.geom2)];bn=[m.body(int(m.geom_bodyid[g])).name for g in gs];allowed='tampa' in bn and any(n.startswith('right_hand') for n in bn)
  if any(n.startswith(('right_','left_','torso','waist','pelvis','head')) for n in bn) and not allowed:bad={'reason':'forbidden contact','bodies':bn}
  if 'tampa' in bn:
   other=bn[1-bn.index('tampa')];force=np.zeros(6);mujoco.mj_contactForce(m,d,ci,force)
   if other.startswith('right_hand'):forces[other]=forces.get(other,0)+float(force[0])
   elif other=='mesa':support+=float(force[0])
   else:bad={'reason':'lid contact outside hand/table','body':other}
 gap=min(mujoco.mj_geomDistance(m,d,g,table,1,None) for g in handgeoms);wrists=np.rad2deg(d.qpos[ik.qa[np.r_[7:10,14:17]]]);tilt=float(np.rad2deg(np.arccos(np.clip(d.xmat[lid].reshape(3,3)[2,2],-1,1))));error=float(np.linalg.norm(d.xpos[lid,:2]-goal[:2]));weight=m.body_mass[lid]*9.81;stable=stable+1 if support>.7*weight and error<.015 and tilt<5 else 0
 if gap<(.01 if t>=8 else .02):bad={'reason':'hand table clearance','gap_m':float(gap)}
 if np.any(np.abs(wrists)>[62,47,32,62,47,32]):bad={'reason':'wrist bounds','degrees':wrists.tolist()}
 if np.max(np.linalg.norm(d.xpos[ids]-prop0,axis=1))>.01:bad={'reason':'other prop displacement'}
 if not anatomy.valid(d,tolerance_deg=2):bad={'reason':'geometric elbow flexion outside task band','degrees':anatomy.angles(d).tolist()}
 if water.spilled_ml>.5:bad={'reason':'spill'}
 opposing=any('thumb' in n and f>.05 for n,f in forces.items()) and any('index' in n and f>.05 for n,f in forces.items());grip_lost=grip_lost+1 if release is None and support<.1 and not opposing else 0
 if grip_lost>int(.4/dt):bad={'reason':'lost opposing grip before support'}
 if release is not None and rt>7:
  window.append({'support':support,'hand':sum(forces.values()),'p':d.xpos[lid].copy(),'tilt':tilt,'error':error})
  if len(window)==window.maxlen and k%50==0:
   w=list(window);mean=float(np.mean([x['support'] for x in w]));drift=max(float(np.linalg.norm(x['p']-w[0]['p'])) for x in w);rest={'support_N':mean,'weight_N':float(weight),'max_drift_m':drift};passed=.9*weight<mean<1.1*weight and drift<.001 and all(x['hand']<.05 and x['tilt']<5 and x['error']<.015 for x in w)
 if k%16==0:rows.append({'t':float(d.time),'phase':phase,'table_gap_m':float(gap)});states.append(d.qpos.copy());samples.append({'t':float(d.time),'phase':phase,'support_N':support,'finger_force_N':forces,'lid_tilt_deg':tilt,'position_error_m':error,'elbow_flexion_deg':anatomy.angles(d).tolist(),'wrist_deg':wrists.tolist()})
 if k%2500==0:print({'t':t,'phase':phase,'support_N':support,'fingers':forces,'lid_pos':d.xpos[lid].tolist()},flush=True)
 if s.warnings():bad={'reason':'MuJoCo numerical warning','warnings':s.warnings()}
 if bad or passed:break
report={'model':'gpt-6-astra','source':str(source),'scene':r['scene'],'pass':bool(passed and not bad and not s.warnings()),'failure':bad if bad else (None if passed else {'reason':'release acceptance not reached'}),'coffee_completed':False,'hand_table_clearance_rule':'20mm during transport;10mm during slow low-lid placement/release; any actual robot/table collision forbidden','warnings':s.warnings(),'initial_water_ml':800,'water_state':asdict(water),'rest_metrics':rest,'rows':rows,'grasp_samples':samples,'liquid_samples':liquid,'scope':'motor-only lid transport/place/release with changing liquid load, fixed robot pelvis','peak_motor_torques_Nm':{n:float(peak[i]) for n,i in s.act_joint.items()}};(a.out/'report.json').write_text(json.dumps(report,indent=2));np.savez_compressed(a.out/'trajectory.npz',qpos=states);integration=np.zeros(mujoco.mj_stateSize(m,mujoco.mjtState.mjSTATE_INTEGRATION));mujoco.mj_getState(m,d,integration,mujoco.mjtState.mjSTATE_INTEGRATION);np.savez_compressed(a.out/'continuation.npz',integration=integration,last_target=target,kp=s.kp,kd=s.kd,body_mass=m.body_mass,body_ipos=m.body_ipos,body_inertia=m.body_inertia,body_iquat=m.body_iquat,bvh_aabb=m.bvh_aabb,dry_kettle_kg=.78);(a.out/'liquid-state.json').write_text(json.dumps(asdict(water),indent=2));print({k:v for k,v in report.items() if k not in ['rows','grasp_samples','liquid_samples','peak_motor_torques_Nm']})

brew.save(a.out)
