"""Continue accepted physical checkpoint using motors and free contacts only."""
import argparse,json,sys,shutil
from pathlib import Path
import numpy as np,mujoco
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'g1-cup-grasp/scripts'))
from g1_sim import G1Sim
from contact_wrench_control import allocate
from scipy.spatial.transform import Rotation
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
normal_ff=np.zeros(m.nu);table=m.geom('tampo').id;rows=[];states=[];samples=[];bad=None;good=0;hold=0;peak=np.zeros(m.nu);dt=m.opt.timestep;gapmin=1.;props=['coador','copo','pote','tampa','scoop','base_eletrica','chaleira'];bs=[m.body(n).id for n in props];initial_props=d.xpos[bs].copy();entry=d.qpos.copy();entry[s.qadr]=ck['last_target'][s.qadr];lost_time=0;allocation_failures=0;last_alloc=None;last_com=None;last_R=None;command_ff=start_ff.copy();jarqa=m.jnt_qposadr[m.joint('chaleira_livre').id]
for name in hand_names:s.kp[s.act_joint[name]]=0;s.kd[s.act_joint[name]]=.1
for k in range(int((a.duration+2)/dt)):
 t=k*dt;u=np.clip((t-2)/(a.duration-2),0,1);u=u*u*(3-2*u)*(len(path)-1);i=min(int(u),len(path)-2);f=u-i;target=path[i]*(1-f)+path[i+1]*f
 if t<2:
  blend=t/2;blend=blend*blend*(3-2*blend);target=entry*(1-blend)+path[0]*blend
 ramp=min(1,t/2);target[ha]=hand_target if wrench is None else hand_origin*(1-ramp)+target[ha]*ramp;phase=('tilt_handle' if a.pour else 'transport_handle') if t<a.duration else ('hold_tilt' if a.pour else 'hold_over_filter')
 base_tau=s.kp*(target[s.qadr]-d.qpos[s.qadr])-s.kd*d.qvel[s.vadr]+d.qfrc_bias[s.vadr];base_tau=np.clip(base_tau,.8*m.actuator_ctrlrange[:,0],.8*m.actuator_ctrlrange[:,1])
 if k%10==0:
  quat=target[jarqa+3:jarqa+7].copy();quat/=np.linalg.norm(quat);matrix=np.zeros(9);mujoco.mju_quat2Mat(matrix,quat);desired_R=matrix.reshape(3,3);desired_com=target[jarqa:jarqa+3]+desired_R@m.body_ipos[jar]
  desired_velocity=np.zeros(3) if last_com is None else (desired_com-last_com)/(10*dt);desired_omega=np.zeros(3) if last_R is None else Rotation.from_matrix(desired_R@last_R.T).as_rotvec()/(10*dt);last_com=desired_com.copy();last_R=desired_R.copy()
  jp=np.zeros((3,m.nv));jr=np.zeros((3,m.nv));mujoco.mj_jacBodyCom(m,d,jp,jr,jar);velocity=jp@d.qvel;omega=jr@d.qvel;rotation_error=Rotation.from_matrix(desired_R@d.xmat[jar].reshape(3,3).T).as_rotvec()
  requested=np.r_[-m.opt.gravity*m.body_mass[jar]+400*(desired_com-d.xipos[jar])+30*(desired_velocity-velocity),1.5*rotation_error+.08*(desired_omega-omega)]
  last_alloc=allocate(m,d,jar,base_tau,requested)
  if last_alloc['pass']:command_ff=last_alloc['motor_torque']-base_tau
  else:
   allocation_failures+=1
   if allocation_failures>5:bad={'time_s':float(d.time),'reason':'contact wrench allocation infeasible','allocator':last_alloc};break
 d.ctrl[:]=np.clip(base_tau+command_ff,m.actuator_ctrlrange[:,0],m.actuator_ctrlrange[:,1]);peak=np.maximum(peak,np.abs(d.ctrl));mujoco.mj_step(m,d)
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
report={'model':'gpt-6-astra','scene':r['scene'],'source':str(a.source),'plan':str(a.plan),'resume_physics_time_s':start_time,'controller':'live contact wrench feedback; finger PD stiffness zero; motors only','allocation_failures':allocation_failures,'wrench_plan':None,'finger_force_target_N':'online wrench allocation, minimum8N/group','finger_target_adjustment_rad':(hand_target-hand_origin).tolist(),'pass':bad is None and hold>=int(1.9/dt) and hold==good and not s.warnings(),'hold_good_steps':good,'hold_steps':hold,'completed_without_forbidden_contact':bad is None and t>a.duration+1.9,'failure':bad,'min_table_gap_m':gapmin,'warnings':s.warnings(),'objects':props,'rows':rows,'grasp_samples':samples,'peak_motor_torques_Nm':{n:float(peak[i]) for n,i in s.act_joint.items()},'scope':'physical continuation, empty/constant-mass kettle manipulation; no water transfer or coffee completion'}
if not report['pass'] and bad is None:report['failure']={'reason':'final hold accuracy criterion not met','final_position_error_m':err,'final_tilt_deg':tilt,'final_rotation_error_deg':rot_error}
(a.out/'report.json').write_text(json.dumps(report,indent=2));np.savez_compressed(a.out/'trajectory.npz',qpos=states);np.save(a.out/'final-qpos.npy',d.qpos);np.save(a.out/'final-qvel.npy',d.qvel);integration=np.zeros(mujoco.mj_stateSize(m,mujoco.mjtState.mjSTATE_INTEGRATION));mujoco.mj_getState(m,d,integration,mujoco.mjtState.mjSTATE_INTEGRATION);np.savez_compressed(a.out/'continuation.npz',integration=integration,last_target=target,hand_target=target[ha],hand_names=hand_names,kp=s.kp,kd=s.kd);print({k:v for k,v in report.items() if k not in ['rows','grasp_samples','peak_motor_torques_Nm']})
