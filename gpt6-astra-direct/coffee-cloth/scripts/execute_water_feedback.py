"""Continue accepted physical checkpoint using motors and free contacts only."""
import argparse,json,sys,shutil
from pathlib import Path
import numpy as np,mujoco
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'g1-cup-grasp/scripts'))
from g1_sim import G1Sim,ARM_JOINTS
from kinematics import ArmIK
from kettle_liquid import KettleWater
from liquid_mass import LiquidMassCoupler
from dataclasses import asdict
from thermal_grip import adjust_fingers,thermal_motor_torque
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);ap.add_argument('--plan',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--duration',type=float,default=12);ap.add_argument('--pour',action='store_true');ap.add_argument('--finger-stiffness-scale',type=float,default=2.);ap.add_argument('--normal-force',type=float,default=20.);ap.add_argument('--thermal-adjust',action='store_true');ap.add_argument('--thermal-barrier',action='store_true');ap.add_argument('--contact-force',type=float,default=0.);ap.add_argument('--return-rate',type=float,default=.8);ap.add_argument('--resume',type=Path);ap.add_argument('--return-lift',type=float,default=.02);ap.add_argument('--return-offset-x',type=float,default=.1);ap.add_argument('--return-offset-y',type=float,default=0.);ap.add_argument('--cutoff-angle',type=float,default=6.);ap.add_argument('--return-free-yaw',action='store_true');ap.add_argument('--wrist-comfort',type=float,default=0.);ap.add_argument('--dispense-ml',type=float,default=200);ap.add_argument('--max-seconds',type=float,default=180);a=ap.parse_args()
r=json.loads((a.source/'report.json').read_text());pr=json.loads((a.plan/'report.json').read_text());assert r['pass'] and pr['pass'];a.out.mkdir(exist_ok=False);shutil.copy2(__file__,a.out/Path(__file__).name)
for helper in ['liquid_mass.py','kettle_liquid.py','thermal_grip.py']:shutil.copy2(Path(__file__).parent/helper,a.out/helper)
s=G1Sim(r['scene']);m,d=s.m,s.d;ck=np.load(a.source/'continuation.npz');assert 'water_ml' in ck;water=KettleWater(initial_ml=float(ck['water_ml']));coupler=LiquidMassCoupler(m,dry_kettle_kg=float(ck['dry_kettle_kg']));mujoco.mj_setState(m,d,ck['integration'],mujoco.mjtState.mjSTATE_INTEGRATION);coupler.apply(d,water);start_time=float(d.time);s.kp[:]=ck['kp'];s.kd[:]=ck['kd'];hand_names=ck['hand_names'].tolist();ha=np.array([m.jnt_qposadr[m.joint(n).id] for n in hand_names]);hand_target=ck['hand_target'].copy();hand_origin=hand_target.copy();thermal_updates=0;thermal_tau=np.zeros(m.nu);thermal_gap=.1;path=np.load(a.plan/'path.npz')['qpos'];jar=m.body('chaleira').id;palm=m.body('left_wrist_yaw_link').id;lip=np.array([-.091012658,.00010992,.209200575]);scratch=mujoco.MjData(m);scratch.qpos[:]=path[-1];mujoco.mj_forward(m,scratch);goal_p=scratch.xpos[jar]+scratch.xmat[jar].reshape(3,3)@lip;goal_R=scratch.xmat[jar].reshape(3,3).copy()
hgs=[g for g in range(m.ngeom) if m.geom_contype[g] and m.body(int(m.geom_bodyid[g])).name.startswith(('left_hand','left_wrist'))];handle=[g for g in range(m.ngeom) if (m.geom(g).name or '').startswith('handle_col')];tips=[next(g for g in hgs if m.body(int(m.geom_bodyid[g])).name=='left_hand_'+n+'_link') for n in ['thumb_2','index_1','middle_1']];finger_mask=np.array([m.joint(m.actuator_trnid[i,0]).name.startswith('left_hand') for i in range(m.nu)]);wa=np.array([m.jnt_qposadr[m.joint('left_wrist_'+n+'_joint').id] for n in ['roll','pitch','yaw']]);normal_ff=np.zeros(m.nu);table=m.geom('tampo').id;rows=[];states=[];samples=[];bad=None;good=0;hold=0;peak=np.zeros(m.nu);dt=m.opt.timestep;gapmin=1.;props=['coador','copo','pote','tampa','scoop','base_eletrica','chaleira'];bs=[m.body(n).id for n in props];initial_props=d.xpos[bs].copy();ik=ArmIK(s,['waist_yaw_joint','waist_roll_joint','waist_pitch_joint']+[n.replace('right_','left_') for n in ARM_JOINTS],palm='left_wrist_yaw_link');ik.bounds=ik.bounds.copy();ik.bounds[0]=[-.7,.7];ik.bounds[1:3]=np.deg2rad([[-12,12],[-12,12]]);ik.bounds[-3:]=np.deg2rad([[-60,60],[-45,45],[-30,30]]);corrected=ck['last_target'][ik.qa].copy();jarqa=m.jnt_qposadr[m.joint('chaleira_livre').id];max_spout_ik_error=0.;lost_time=0;finger_kp=s.kp.copy();finger_kd=s.kd.copy()
forces={};liquid_rows=[];flow_state={'flow_ml_s':0.};progress_index=0.;pour_done=False;return_started=None;return_R=None;return_index=None;completed=False;filter_id=m.body('coador').id;cup_id=m.body('copo').id
torso_geoms=[g for g in range(m.ngeom) if m.geom_contype[g] and m.body(int(m.geom_bodyid[g])).name=='torso_link'];shoulder_geoms=[g for g in range(m.ngeom) if m.geom_contype[g] and m.body(int(m.geom_bodyid[g])).name=='left_shoulder_yaw_link'];return_pairs=[(g,h) for g in torso_geoms for h in shoulder_geoms]
first_step=0;resume_state_time=None
if a.resume:
 snap=np.load(a.resume);state=json.loads(str(snap['controller_json']));wet=json.loads(str(snap['water_json']))
 if abs(sum(wet[n] for n in ['source_ml','filter_ml','cloth_retained_ml','coffee_retained_ml','receiver_ml','spilled_ml'])-wet['initial_ml'])>1e-7:raise ValueError('Checkpoint liquid ledger invalid')
 for name,value in wet.items():setattr(water,name,value)
 
 if 'body_mass' not in snap:raise ValueError('Checkpoint lacks mass/BVH cache; use newer snapshot')
 for name in ['body_mass','body_ipos','body_inertia','body_iquat','bvh_aabb']:getattr(m,name)[:]=snap[name]
 mujoco.mj_setConst(m,mujoco.MjData(m));mujoco.mj_setState(m,d,snap['integration'],mujoco.mjtState.mjSTATE_INTEGRATION);mujoco.mj_forward(m,d);resume_state_time=float(d.time);hand_target=snap['hand_target'].copy();hand_origin=snap['hand_origin'].copy();corrected=snap['corrected'].copy();normal_ff=snap['normal_ff'].copy();thermal_tau=snap['thermal_tau'].copy();s.kp[:]=snap['kp'];s.kd[:]=snap['kd'];first_step=state['next_step'];progress_index=state['progress_index'];pour_done=state['pour_done'];return_started=state['return_started'];return_R=None if state.get('return_R') is None else np.array(state['return_R']);return_index=state.get('return_index');forces=state['forces'];flow_state=state['flow_state'];lost_time=state['lost_time'];hold=state['hold'];good=state['good']
 shutil.copy2(a.resume,a.out/'input-checkpoint.npz')
for k in range(first_step,int(a.max_seconds/dt)):
 t=k*dt
 if k%10==0:
  if k>0:
   flow_state=water.step(dt*10,d.xpos[jar],d.xmat[jar].reshape(3,3),d.xpos[filter_id],d.xmat[filter_id].reshape(3,3),d.xpos[cup_id],d.xmat[cup_id].reshape(3,3));liquid_rows.append({'t':float(d.time),**flow_state})
  coupler.apply(d,water)
 if water.spilled_ml>2:bad={'time_s':float(d.time),'reason':'water spill limit exceeded','spilled_ml':water.spilled_ml};break
 if water.discharged_ml>=a.dispense_ml-.5:pour_done=True
 if pour_done:
  if return_started is None:return_started=t;return_R=d.xmat[jar].reshape(3,3).copy();return_index=progress_index
  return_blend=min(1.,max(0.,t-return_started-10)/3);return_blend=return_blend*return_blend*(3-2*return_blend)
  progress_index=max(0.,progress_index-a.return_rate*return_blend*dt);phase='return_upright' if progress_index>0 else 'drain_filter'
 elif t<a.duration:
  blend=t/a.duration;blend=blend*blend*(3-2*blend);progress_index=min(40.,len(path)-1)*blend;phase='approach_filter'
 else:
  desired_flow=min(4.,max(0.,(60-water.filter_ml)*.3),max(.2,(a.dispense_ml-water.discharged_ml)*.5));rate=float(np.clip(.12*(desired_flow-flow_state['flow_ml_s']),-2,2));progress_index=float(np.clip(progress_index+rate*dt,0,len(path)-1));phase='dispense_water'
 u=progress_index;i=min(int(u),len(path)-2);f=u-i;target=path[i]*(1-f)+path[i+1]*f;target[ha]=hand_target
 if k%10==0:
  actual_spout=d.xpos[jar]+d.xmat[jar].reshape(3,3)@lip;actual_R=d.xmat[palm].reshape(3,3);local_lip=actual_R.T@(actual_spout-d.xpos[palm]);nominal=target[ik.qa].copy();pp,nominal_R=ik.fk(nominal);quat=target[jarqa+3:jarqa+7].copy();quat/=np.linalg.norm(quat);matrix=np.zeros(9);mujoco.mju_quat2Mat(matrix,quat);desired_spout=target[jarqa:jarqa+3]+matrix.reshape(3,3)@lip
  if pour_done:
   lift_blend=np.clip((t-return_started-4)/6,0,1);lift_blend=lift_blend*lift_blend*(3-2*lift_blend);desired_spout+=np.array([a.return_offset_x,a.return_offset_y,a.return_lift])*lift_blend
  orientation_target=nominal_R;orientation_weight=.3
  if pour_done:
   relative_jar_R=actual_R.T@d.xmat[jar].reshape(3,3);q0=path[0,jarqa+3:jarqa+7];end_R=Rotation.from_quat(q0[[1,2,3,0]]).as_matrix();cutoff_blend=np.clip((t-return_started)/4,0,1);cutoff_blend=cutoff_blend*cutoff_blend*(3-2*cutoff_blend);back_R=return_R@Rotation.from_rotvec([0,np.deg2rad(a.cutoff_angle)*cutoff_blend,0]).as_matrix();blend=1-progress_index/return_index;desired_R=Rotation.from_rotvec(blend*Rotation.from_matrix(end_R@back_R.T).as_rotvec()).as_matrix()@back_R;orientation_target=desired_R@relative_jar_R.T;orientation_weight=30.
  def residual(x):
   pp,rr=ik.fk(x);gaps=[min(0.,mujoco.mj_geomDistance(m,ik.d,g,h,.02,None)-.005) for g,h in return_pairs] if pour_done else [];rotation=Rotation.from_matrix(rr@orientation_target.T).as_rotvec()*orientation_weight;comfort=[]
   if pour_done and a.return_free_yaw and t-return_started>55:
    rotation=np.r_[np.cross((rr@relative_jar_R)[2,:],desired_R[2,:])*30,Rotation.from_matrix(rr@orientation_target.T).as_rotvec()*.3];comfort_blend=np.clip((t-return_started-55)/12,0,1);comfort_blend=comfort_blend*comfort_blend*(3-2*comfort_blend);comfort=x[-3:]*a.wrist_comfort*comfort_blend
   return np.r_[(pp+rr@local_lip-desired_spout)*1000,rotation,(x-nominal)*(.3 if pour_done else .03),np.array(gaps)*10000,comfort]
  bounds=ik.bounds.T if pour_done else np.stack([np.maximum(ik.bounds[:,0],nominal-.25),np.minimum(ik.bounds[:,1],nominal+.25)]);fit=least_squares(residual,np.clip(corrected,bounds[0]+1e-9,bounds[1]-1e-9),bounds=bounds,max_nfev=60);sol=fit.x;pp,rr=ik.fk(sol);max_spout_ik_error=max(max_spout_ik_error,float(np.linalg.norm(pp+rr@local_lip-desired_spout)));corrected+=np.clip(sol-corrected,-.02,.02)
 target[ik.qa]=corrected
 if k%10==0 and a.thermal_adjust:thermal_updates+=len(adjust_fingers(m,d,hand_names,hand_target,hand_origin))
 target[ha]=hand_target
 ramp=min(1,t/2);s.kp[finger_mask]=finger_kp[finger_mask]*(1+(a.finger_stiffness_scale-1)*ramp);s.kd[finger_mask]=finger_kd[finger_mask]*(1+.67*ramp)
 if k%10==0 and a.thermal_barrier:thermal_tau,thermal_gap=thermal_motor_torque(m,d,s.vadr)
 tau=thermal_tau+s.kp*(target[s.qadr]-d.qpos[s.qadr])-s.kd*d.qvel[s.vadr]+d.qfrc_bias[s.vadr]
 if k%10==0:
  normal_ff[:]=0
  for g in tips:
   best=None
   for h in handle:
    points=np.zeros(6);gap=mujoco.mj_geomDistance(m,d,g,h,.03,points)
    if best is None or gap<best[0]:best=(gap,points)
   gap,points=best
   if abs(gap)<1e-8 or gap>=.03:continue
   normal=(points[3:]-points[:3])*np.sign(gap);normal/=max(np.linalg.norm(normal),1e-12);jp=np.zeros((3,m.nv));jr=np.zeros((3,m.nv));mujoco.mj_jac(m,d,jp,jr,points[:3],int(m.geom_bodyid[g]));normal_ff+=(jp[:,s.vadr].T@(normal*(12+(a.normal_force-12)*ramp)))*finger_mask
   if a.contact_force>0 and t>2:
    part=m.body(int(m.geom_bodyid[g])).name.split('_')[2];mask=np.array([m.joint(m.actuator_trnid[j,0]).name.startswith('left_hand_'+part) for j in range(m.nu)]);grad=(jp[:,s.vadr].T@normal)*mask;measured=forces.get(m.body(int(m.geom_bodyid[g])).name,0.)
    if np.dot(grad,grad)>1e-10:
     delta=np.clip(grad*max(0.,a.contact_force-measured)*.000002/np.dot(grad,grad),-.002,.002)
     for h,n in enumerate(hand_names):hand_target[h]=np.clip(hand_target[h]+delta[s.act_joint[n]],max(hand_origin[h]-.25,m.joint(n).range[0]),min(hand_origin[h]+.25,m.joint(n).range[1]))
 jp=np.zeros((3,m.nv));jr=np.zeros((3,m.nv));mujoco.mj_jac(m,d,jp,jr,d.xipos[jar],palm);tau+=normal_ff+(jp[:,s.vadr].T@(-m.opt.gravity*m.body_mass[jar]))*(~finger_mask);d.ctrl[:]=np.clip(tau,m.actuator_ctrlrange[:,0],m.actuator_ctrlrange[:,1]);peak=np.maximum(peak,np.abs(d.ctrl));mujoco.mj_step(m,d)
 forces={};supports=set()
 for ci,ct in enumerate(d.contact):
  if ct.dist>=0:continue
  bn=[m.body(int(m.geom_bodyid[g])).name for g in [ct.geom1,ct.geom2]];gn=[m.geom(int(g)).name or '' for g in [ct.geom1,ct.geom2]];allowed=any(n.startswith('left_hand') for n in bn) and any(n.startswith('handle_col') for n in gn)
  if not allowed and any(n.startswith(('left_','right_','torso','pelvis','waist','head')) for n in bn):bad={'time_s':float(d.time),'bodies':bn,'geoms':gn}
  if 'chaleira' in bn:
   other=bn[1-bn.index('chaleira')]
   if other.startswith('left_hand'):
    cf=np.zeros(6);mujoco.mj_contactForce(m,d,ci,cf);forces[other]=forces.get(other,0)+float(cf[0])
   else:supports.add(other)
 gap=min(mujoco.mj_geomDistance(m,d,g,table,1,None) for g in hgs);gapmin=min(gapmin,gap);tilt=float(np.rad2deg(np.arccos(np.clip(d.xmat[jar].reshape(3,3)[2,2],-1,1))));spout=d.xpos[jar]+d.xmat[jar].reshape(3,3)@lip;err=float(np.linalg.norm(spout-goal_p));relative=d.xmat[jar].reshape(3,3)@goal_R.T;rot_error=float(np.rad2deg(np.arccos(np.clip((np.trace(relative)-1)/2,-1,1))));wrist=np.rad2deg(d.qpos[wa]);opposing=forces.get('left_hand_thumb_2_link',0)>1 and any(forces.get('left_hand_'+n+'_link',0)>1 for n in ['middle_1','index_1']);lost_time=0 if opposing else lost_time+dt
 if supports:bad={'time_s':float(d.time),'reason':'external kettle support','supports':sorted(supports)}
 if gap<.02:bad={'time_s':float(d.time),'reason':'hand-table clearance','gap_m':gap}
 if np.any(np.abs(wrist)>[62,47,32]):bad={'time_s':float(d.time),'reason':'wrist posture tolerance','wrist_deg':wrist.tolist()}
 if lost_time>.15:bad={'time_s':float(d.time),'reason':'opposing grip lost'}
 if pour_done and progress_index==0 and water.filter_ml<.5:
  hold+=1;good+=int(not supports and opposing and tilt<12)
 if k%16==0:
  rows.append({'t':float(d.time),'phase':phase,'table_gap_m':gap,'prop_displacements_m':np.linalg.norm(d.xpos[bs]-initial_props,axis=1).tolist()});states.append(d.qpos.copy());samples.append({'t':float(d.time),'phase':phase,'thermal_gap_m':thermal_gap,'jar_position_m':d.xpos[jar].tolist(),'jar_tilt_deg':tilt,'spout_position_m':spout.tolist(),'target_error_m':err,'rotation_error_deg':rot_error,'finger_normal_force_N':forces,'external_supports':sorted(supports),'wrist_deg':wrist.tolist()})
 if k%2500==0:print({'t':round(t,2),'phase':phase,'index':round(progress_index,2),'discharged_ml':round(water.discharged_ml,2),'filter_ml':round(water.filter_ml,2),'spill_ml':round(water.spilled_ml,3)},flush=True)
 if bad:break
 if k%5000==0 or (pour_done and phase=='return_upright' and t-return_started<dt):
  mujoco.mj_forward(m,d)
  saved=np.zeros(mujoco.mj_stateSize(m,mujoco.mjtState.mjSTATE_INTEGRATION));mujoco.mj_getState(m,d,saved,mujoco.mjtState.mjSTATE_INTEGRATION)
  checkpoint=a.out/f'checkpoint-{k:07d}.npz'
  if checkpoint.exists():raise FileExistsError(checkpoint)
  np.savez_compressed(checkpoint,body_mass=m.body_mass,body_ipos=m.body_ipos,body_inertia=m.body_inertia,body_iquat=m.body_iquat,bvh_aabb=m.bvh_aabb,integration=saved,last_target=target,hand_target=hand_target,hand_origin=hand_origin,hand_names=hand_names,corrected=corrected,water_json=json.dumps(asdict(water)),controller_json=json.dumps({'next_step':k+1,'progress_index':progress_index,'pour_done':pour_done,'return_started':return_started,'return_R':None if return_R is None else return_R.tolist(),'return_index':return_index,'forces':forces,'flow_state':flow_state,'lost_time':lost_time,'hold':hold,'good':good,'start_time':start_time}),normal_ff=normal_ff,thermal_tau=thermal_tau,kp=s.kp,kd=s.kd,dry_kettle_kg=float(ck['dry_kettle_kg']))
 if hold>=int(2/dt):completed=True;break
report={'model':'gpt-6-astra','scene':r['scene'],'source':str(a.source),'plan':str(a.plan),'resume_physics_time_s':start_time,'intermediate_resume_time_s':resume_state_time,'initial_water_ml':water.initial_ml,'dry_kettle_mass_kg':float(ck['dry_kettle_kg']),'final_loaded_kettle_mass_kg':float(m.body_mass[jar]),'requested_dispense_ml':a.dispense_ml,'controller_parameters':{name:str(value) if isinstance(value,Path) else value for name,value in vars(a).items()},'thermal_adjustment_steps':thermal_updates,'finger_reference_shift_rad':(hand_target-hand_origin).tolist(),'water_state':asdict(water),'liquid_samples':liquid_rows,'coffee_completed':False,'pass':completed and bad is None and abs(water.discharged_ml-a.dispense_ml)<3 and water.spilled_ml<=2 and hold==good and not s.warnings(),'hold_good_steps':good,'hold_steps':hold,'completed_without_forbidden_contact':bad is None and completed,'failure':bad,'min_table_gap_m':gapmin,'warnings':s.warnings(),'objects':props,'rows':rows,'grasp_samples':samples,'peak_motor_torques_Nm':{n:float(peak[i]) for n,i in s.act_joint.items()},'finger_control':'position target and normal preload with recorded CLI gains; real motor saturation','max_spout_ik_error_m':max_spout_ik_error,'acceptance':'water discharge within3ml of request, spills<=2ml, filter drained<.5ml, jar returned upright and supported by fingers2s, no forbidden collisions','scope':'robot+reduced liquid mass/COM/inertia coupling; water-only test, utensils prepositioned; no heating, coffee dosing, chemical extraction, slosh, or jet impact impulse'}
if not report['pass'] and bad is None:report['failure']={'reason':'water transfer or return/drain criterion not met','final_position_error_m':err,'final_tilt_deg':tilt,'final_rotation_error_deg':rot_error}
(a.out/'report.json').write_text(json.dumps(report,indent=2));np.savez_compressed(a.out/'trajectory.npz',qpos=states);np.save(a.out/'final-qpos.npy',d.qpos);np.save(a.out/'final-qvel.npy',d.qvel);integration=np.zeros(mujoco.mj_stateSize(m,mujoco.mjtState.mjSTATE_INTEGRATION));mujoco.mj_getState(m,d,integration,mujoco.mjtState.mjSTATE_INTEGRATION);np.savez_compressed(a.out/'continuation.npz',integration=integration,last_target=target,hand_target=hand_target,hand_names=hand_names,kp=s.kp,kd=s.kd,body_mass=m.body_mass,body_ipos=m.body_ipos,body_inertia=m.body_inertia,body_iquat=m.body_iquat,water_ml=water.source_ml,dry_kettle_kg=float(ck['dry_kettle_kg']));(a.out/'liquid-state.json').write_text(json.dumps(asdict(water),indent=2));print({k:v for k,v in report.items() if k not in ['rows','grasp_samples','liquid_samples','peak_motor_torques_Nm']})
