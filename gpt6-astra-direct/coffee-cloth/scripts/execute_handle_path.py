"""Continue accepted physical checkpoint using motors and free contacts only."""
import argparse,json,sys,shutil
from pathlib import Path
import numpy as np,mujoco
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'g1-cup-grasp/scripts'))
from g1_sim import G1Sim
ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);ap.add_argument('--plan',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--duration',type=float,default=12);ap.add_argument('--pour',action='store_true');ap.add_argument('--wrench-plan',type=Path);a=ap.parse_args()
r=json.loads((a.source/'report.json').read_text());pr=json.loads((a.plan/'report.json').read_text());assert r['pass'] and pr['pass'];a.out.mkdir(exist_ok=False);shutil.copy2(__file__,a.out/Path(__file__).name)
s=G1Sim(r['scene']);m,d=s.m,s.d;ck=np.load(a.source/'continuation.npz');mujoco.mj_setState(m,d,ck['integration'],mujoco.mjtState.mjSTATE_INTEGRATION);mujoco.mj_forward(m,d);start_time=float(d.time);s.kp[:]=ck['kp'];s.kd[:]=ck['kd'];hand_names=ck['hand_names'].tolist();ha=np.array([m.jnt_qposadr[m.joint(n).id] for n in hand_names]);hand_target=ck['hand_target'].copy();hand_origin=hand_target.copy();forces={};path=np.load(a.plan/'path.npz')['qpos'];jar=m.body('chaleira').id;palm=m.body('left_wrist_yaw_link').id;lip=np.array([-.091012658,.00010992,.209200575]);scratch=mujoco.MjData(m);scratch.qpos[:]=path[-1];mujoco.mj_forward(m,scratch);goal_p=scratch.xpos[jar]+scratch.xmat[jar].reshape(3,3)@lip;goal_R=scratch.xmat[jar].reshape(3,3).copy()
hgs=[g for g in range(m.ngeom) if m.geom_contype[g] and m.body(int(m.geom_bodyid[g])).name.startswith(('left_hand','left_wrist'))];handle=[g for g in range(m.ngeom) if (m.geom(g).name or '').startswith('handle_col')];tips=[next(g for g in hgs if m.body(int(m.geom_bodyid[g])).name=='left_hand_'+n+'_link') for n in ['thumb_2','index_1','middle_1']];finger_mask=np.array([m.joint(m.actuator_trnid[i,0]).name.startswith('left_hand') for i in range(m.nu)]);wa=np.array([m.jnt_qposadr[m.joint('left_wrist_'+n+'_joint').id] for n in ['roll','pitch','yaw']]);wrench=None
if a.wrench_plan:
 wr=json.loads((a.wrench_plan/'report.json').read_text());by_index={x['path_index']:x for x in wr['results']};wrench=np.zeros((len(path),m.nu))
 for ii in range(len(path)):
  fplan=by_index[ii]['force_plan'];assert fplan['unit_kg_feasible']
  for name,value in zip(fplan['actuator_joint_names'],fplan['motor_contact_torque_Nm']):wrench[ii,s.act_joint[name]]=value
start_ff=d.ctrl-s.kp*(ck['last_target'][s.qadr]-d.qpos[s.qadr])+s.kd*d.qvel[s.vadr]-d.qfrc_bias[s.vadr]
normal_ff=np.zeros(m.nu);table=m.geom('tampo').id;rows=[];states=[];samples=[];bad=None;good=0;hold=0;peak=np.zeros(m.nu);dt=m.opt.timestep;gapmin=1.;props=['coador','copo','pote','tampa','scoop','base_eletrica','chaleira'];bs=[m.body(n).id for n in props];initial_props=d.xpos[bs].copy();lost_time=0
for k in range(int((a.duration+2)/dt)):
 t=k*dt;u=np.clip(t/a.duration,0,1);u=u*u*(3-2*u)*(len(path)-1);i=min(int(u),len(path)-2);f=u-i;target=path[i]*(1-f)+path[i+1]*f;ramp=min(1,t/2);target[ha]=hand_target if wrench is None else hand_origin*(1-ramp)+target[ha]*ramp;phase=('tilt_handle' if a.pour else 'transport_handle') if t<a.duration else ('hold_tilt' if a.pour else 'hold_over_filter')
 tau=s.kp*(target[s.qadr]-d.qpos[s.qadr])-s.kd*d.qvel[s.vadr]+d.qfrc_bias[s.vadr]
 if k%10==0 and wrench is None:
  normal_ff[:]=0
  for g in tips:
   best=None
   for h in handle:
    points=np.zeros(6);gap=mujoco.mj_geomDistance(m,d,g,h,.03,points)
    if best is None or gap<best[0]:best=(gap,points)
   gap,points=best
   if abs(gap)<1e-8 or gap>=.03:continue
   normal=(points[3:]-points[:3])*np.sign(gap);normal/=max(np.linalg.norm(normal),1e-12);jp=np.zeros((3,m.nv));jr=np.zeros((3,m.nv));mujoco.mj_jac(m,d,jp,jr,points[:3],int(m.geom_bodyid[g]));gradient=jp[:,s.vadr].T@normal;normal_ff+=(gradient*12)*finger_mask
   part=m.body(int(m.geom_bodyid[g])).name.split('_')[2];mask=np.array([m.joint(m.actuator_trnid[ii,0]).name.startswith('left_hand_'+part) for ii in range(m.nu)]);grad=gradient*mask;measured=forces.get(m.body(int(m.geom_bodyid[g])).name,0)
   if np.dot(grad,grad)>1e-10:
    delta=np.clip(grad*(18-measured)*.000006/np.dot(grad,grad),-.003,.003)
    for hh,name in enumerate(hand_names):hand_target[hh]+=delta[s.act_joint[name]]
    hand_target[:]=np.clip(hand_target,hand_origin-.2,hand_origin+.2)
    for hh,name in enumerate(hand_names):hand_target[hh]=np.clip(hand_target[hh],*m.jnt_range[m.joint(name).id])
 jp=np.zeros((3,m.nv));jr=np.zeros((3,m.nv));mujoco.mj_jac(m,d,jp,jr,d.xipos[jar],palm);tau+=(normal_ff+(jp[:,s.vadr].T@(-m.opt.gravity*m.body_mass[jar]))*(~finger_mask)) if wrench is None else start_ff*(1-ramp)+(wrench[i]*(1-f)+wrench[i+1]*f)*ramp;d.ctrl[:]=np.clip(tau,m.actuator_ctrlrange[:,0],m.actuator_ctrlrange[:,1]);peak=np.maximum(peak,np.abs(d.ctrl));mujoco.mj_step(m,d)
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
 if t>=a.duration:
  hold+=1;good+=int(not supports and opposing and err<.01 and (rot_error<3 if a.pour else tilt<10))
 if k%16==0:
  rows.append({'t':float(d.time),'phase':phase,'table_gap_m':gap,'prop_displacements_m':np.linalg.norm(d.xpos[bs]-initial_props,axis=1).tolist()});states.append(d.qpos.copy());samples.append({'t':float(d.time),'phase':phase,'jar_position_m':d.xpos[jar].tolist(),'jar_tilt_deg':tilt,'spout_position_m':spout.tolist(),'target_error_m':err,'rotation_error_deg':rot_error,'finger_normal_force_N':forces,'external_supports':sorted(supports),'wrist_deg':wrist.tolist()})
 if bad:break
report={'model':'gpt-6-astra','scene':r['scene'],'source':str(a.source),'plan':str(a.plan),'resume_physics_time_s':start_time,'wrench_plan':None if a.wrench_plan is None else str(a.wrench_plan),'finger_force_target_N':18 if wrench is None else 'LP allocation','finger_target_adjustment_rad':(hand_target-hand_origin).tolist(),'pass':bad is None and hold>=int(1.9/dt) and hold==good and not s.warnings(),'hold_good_steps':good,'hold_steps':hold,'completed_without_forbidden_contact':bad is None and t>a.duration+1.9,'failure':bad,'min_table_gap_m':gapmin,'warnings':s.warnings(),'objects':props,'rows':rows,'grasp_samples':samples,'peak_motor_torques_Nm':{n:float(peak[i]) for n,i in s.act_joint.items()},'scope':'physical continuation, empty/constant-mass kettle manipulation; no water transfer or coffee completion'}
if not report['pass'] and bad is None:report['failure']={'reason':'final hold accuracy criterion not met','final_position_error_m':err,'final_tilt_deg':tilt,'final_rotation_error_deg':rot_error}
(a.out/'report.json').write_text(json.dumps(report,indent=2));np.savez_compressed(a.out/'trajectory.npz',qpos=states);np.save(a.out/'final-qpos.npy',d.qpos);np.save(a.out/'final-qvel.npy',d.qvel);integration=np.zeros(mujoco.mj_stateSize(m,mujoco.mjtState.mjSTATE_INTEGRATION));mujoco.mj_getState(m,d,integration,mujoco.mjtState.mjSTATE_INTEGRATION);np.savez_compressed(a.out/'continuation.npz',integration=integration,last_target=target,hand_target=target[ha],hand_names=hand_names,kp=s.kp,kd=s.kd);print({k:v for k,v in report.items() if k not in ['rows','grasp_samples','peak_motor_torques_Nm']})
