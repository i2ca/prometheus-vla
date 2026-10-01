"""Raise from arms-down and follow collision-planned pregrasp path with free props."""
import json,shutil,sys
from pathlib import Path
import numpy as np,mujoco
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'g1-cup-grasp/scripts'))
from g1_sim import G1Sim,ARM_JOINTS
from kinematics import ArmIK
from scipy.spatial.transform import Rotation,Slerp
from scipy.optimize import least_squares
out=Path('results/loadable-dynamics-015');out.mkdir(exist_ok=False);shutil.copy2(__file__,out/Path(__file__).name)
scene=Path('results/handle-layout-search-003/layout-05/scene.xml');s=G1Sim(str(scene.resolve()));m,d=s.m,s.d;initial=np.load('results/handle-layout-search-003/layout-05/initial.npy');d.qpos[:]=initial;mujoco.mj_forward(m,d)
roll=m.jnt_qposadr[m.joint('left_shoulder_roll_joint').id];elbow=m.jnt_qposadr[m.joint('left_elbow_joint').id]
raised=initial.copy();raised[roll]=1.2
above=raised.copy();above[elbow]=0.1
planned=np.load('results/loadable-connect-004/path.npz')['qpos']
ik=ArmIK(s,['waist_yaw_joint']+[n.replace('right_','left_') for n in ARM_JOINTS],palm='left_wrist_yaw_link')
ik.bounds=ik.bounds.copy();ik.bounds[0]=[-.5,.5];ik.bounds[-3:]=np.deg2rad([[-60,60],[-45,45],[-30,30]])
right_pitch=m.jnt_qposadr[m.joint('right_shoulder_pitch_joint').id]
graspdata=json.loads(Path('results/loadable-refined-002/report.json').read_text())['grasp'];approach=np.load('results/finger-release-005/path.npz')['qpos'];hnames=graspdata['hand_joint_names'];ha=np.array([m.jnt_qposadr[m.joint(n).id] for n in hnames]);hq=approach[-1][ha].copy();grasp_start=None;maxik=0;grasp_rows=[];hold_good=0;hold_steps=0;align=None;grasp_cmd=approach[-1].copy();jar_id=m.body('chaleira').id;jar_initial_pos=d.xpos[jar_id].copy();jar_initial_R=d.xmat[jar_id].reshape(3,3).copy()
force_plan=json.loads(Path('results/refined-force-plan-007/report.json').read_text())['results'][0]['force_plan'];assert force_plan['unit_kg_feasible'];contact_ff=np.zeros(m.nu)
for name,value in zip(force_plan['actuator_joint_names'],force_plan['motor_contact_torque_Nm']):contact_ff[s.act_joint[name]]=value
finger_trim=np.zeros(m.nq);normal_ff=np.zeros(m.nu);handle_geoms=[g for g in range(m.ngeom) if (m.geom(g).name or '').startswith('handle_col')];tip_geoms=[next(g for g in range(m.ngeom) if m.geom_contype[g] and m.body(int(m.geom_bodyid[g])).name=='left_hand_'+part+'_link') for part in ['thumb_2','index_1','middle_1']];finger_mask=np.array([m.joint(m.actuator_trnid[i,0]).name.startswith('left_hand') for i in range(m.nu)])
for n,i in s.act_joint.items():
 if 'hand' in n:s.kp[i]=25;s.kd[i]=.3
 elif n.startswith('left_') and any(x in n for x in ['shoulder','elbow','wrist']):s.kp[i]*=3;s.kd[i]*=np.sqrt(3)*2
props=['coador','copo','pote','tampa','scoop','base_eletrica','chaleira'];bs=[m.body(n).id for n in props];ref=None
handgeoms=[g for g in range(m.ngeom) if m.body(int(m.geom_bodyid[g])).name.startswith(('left_hand','left_wrist')) and (m.geom_contype[g] or m.geom_conaffinity[g])];table=m.geom('tampo').id
transport_report=json.loads(Path('results/handle-transport-plan-010/report.json').read_text());assert transport_report['pass'];transport=np.load('results/handle-transport-plan-010/path.npz')['qpos'];transport_good=0;transport_steps=0;spout_local=np.array(transport_report['spout_local_m']);transport_target=np.array(transport_report['target_spout']);
rows=[];states=[];bad=None;gapmin=1.;dt=m.opt.timestep;peak=np.zeros(m.nu);cleared=False
for k in range(int(46/dt)):
 t=k*dt
 if t<2:target=initial;phase='settle'
 elif t<5:
  u=(t-2)/3;u=u*u*(3-2*u);target=initial*(1-u)+raised*u;phase='raise_outside_table'
 elif t<7:
  u=(t-5)/2;u=u*u*(3-2*u);target=raised*(1-u)+above*u;phase='bend_above_table'
 elif t<8:target=above;phase='hold_above_table'
 elif t<20:
  u=(t-8)/12;u=u*u*(3-2*u)*(len(planned)-1);i=min(int(u),len(planned)-2);f=u-i;target=planned[i]*(1-f)+planned[i+1]*f;phase='move_to_pregrasp'
 elif t>=34:
  u=np.clip((t-34)/10,0,1);u=u*u*(3-2*u)*(len(transport)-1);idx=min(int(u),len(transport)-2);f=u-idx;target=transport[idx]*(1-f)+transport[idx+1]*f;target[ha]=hq+finger_trim[ha];phase='transport_handle' if t<44 else 'hold_over_filter'
 else:
  if t<26:
   if align is None:
    align=d.xmat[jar_id].reshape(3,3)@jar_initial_R.T;align_pos=d.xpos[jar_id]-align@jar_initial_pos
   u=np.clip((t-20)/6,0,1);u=u*u*(3-2*u)*(len(approach)-1);idx=min(int(u),len(approach)-2);f=u-idx;target=approach[idx]*(1-f)+approach[idx+1]*f;phase='approach_handle'
   palm_p,palm_R=ik.fk(target[ik.qa]);sol,info=ik.solve(align@palm_p+align_pos,align@palm_R,target[ik.qa],reference=target[ik.qa],iterations=100);maxik=max(maxik,info['position_error_m'])
   if info['position_error_m']>.002 or info['orientation_error_rad']>.02:bad={'time_s':t,'reason':'approach IK infeasible','errors':info};break
   target[ik.qa]=sol;grasp_cmd=target.copy()
  else:
   target=grasp_cmd.copy();phase='close_handle';f=np.clip(t-26,0,1);target[ha]=hq+finger_trim[ha]
   if t>=28:
    if grasp_start is None:
     if not all(forces.get('left_hand_'+name+'_link',0)>1 for name in ['thumb_2','index_1','middle_1']):bad={'time_s':t,'reason':'three-finger grip not established; lift inhibited','forces':forces};break
     q=grasp_cmd[ik.qa].copy();gp,gR=ik.fk(q);bid=m.body('chaleira').id;z0=float(d.xpos[bid,2]);grasp_start=True
    phase='lift_handle' if t<32 else 'hold_handle';v=np.clip((t-28)/4,0,1);v=v*v*(3-2*v);pos=gp+np.array([0,0,.08*v]);rot=gR
    if k%10==0:
     prev=q.copy()
     def lift_residual(x):
      pp,rr=ik.fk(x);return np.r_[(pp-pos)*1000,Rotation.from_matrix(rr@rot.T).as_rotvec(),(x-prev)*.01]
     fit=least_squares(lift_residual,np.clip(q,ik.bounds[:,0]+1e-9,ik.bounds[:,1]-1e-9),bounds=ik.bounds.T,max_nfev=60);q=fit.x;pp,rr=ik.fk(q);e={'position_error_m':float(np.linalg.norm(pp-pos)),'orientation_error_rad':float(np.linalg.norm(Rotation.from_matrix(rr@rot.T).as_rotvec()))};maxik=max(maxik,e['position_error_m'])
    target[ik.qa]=q
    if e['position_error_m']>.002 or e['orientation_error_rad']>np.deg2rad(8):bad={'time_s':t,'reason':'IK target infeasible within posture bounds','errors':e};break
  target[right_pitch]=initial[right_pitch]+.8*max(0.,target[ik.qa[0]])
  for idx,n in zip(ha,hnames):target[idx]=np.clip(target[idx],*m.jnt_range[m.joint(n).id])
 if k==int(26/dt):
  integration=np.zeros(mujoco.mj_stateSize(m,mujoco.mjtState.mjSTATE_INTEGRATION));mujoco.mj_getState(m,d,integration,mujoco.mjtState.mjSTATE_INTEGRATION);np.savez_compressed(out/'checkpoint-26.npz',integration=integration,grasp_cmd=grasp_cmd,align=align,align_pos=align_pos)
 tau=s.kp*(target[s.qadr]-d.qpos[s.qadr])-s.kd*d.qvel[s.vadr]+d.qfrc_bias[s.vadr]
 if t>=26:
  if k%10==0:
   normal_ff[:]=0
   for g in tip_geoms:
    best=None
    for h in handle_geoms:
     points=np.zeros(6);gap=mujoco.mj_geomDistance(m,d,g,h,.03,points)
     if best is None or gap<best[0]:best=(gap,points)
    gap,points=best
    if abs(gap)<1e-8 or gap>=.03:continue
    normal=(points[3:]-points[:3])*np.sign(gap);normal/=max(1e-12,np.linalg.norm(normal));jp=np.zeros((3,m.nv));jr=np.zeros((3,m.nv));mujoco.mj_jac(m,d,jp,jr,points[:3],int(m.geom_bodyid[g]));gradient=jp[:,s.vadr].T@normal;normal_ff+=(gradient*12)*finger_mask
    measured=forces.get(m.body(int(m.geom_bodyid[g])).name,0);mask=np.array([m.joint(m.actuator_trnid[i,0]).name.startswith('left_hand_'+m.body(int(m.geom_bodyid[g])).name.split('_')[2]) for i in range(m.nu)]);grad=gradient*mask
    if t<28 and np.dot(grad,grad)>1e-10:
     delta=np.clip(grad*(8-measured)*.00002/np.dot(grad,grad),-.004,.004);finger_trim[s.qadr]+=delta;finger_trim[ha]=np.clip(finger_trim[ha],-.12,.12)
  tau+=normal_ff*np.clip(t-26,0,1)
  if t>=28:
   jp=np.zeros((3,m.nv));jr=np.zeros((3,m.nv));mujoco.mj_jac(m,d,jp,jr,d.xipos[jar_id],ik.body);tau+=(jp[:,s.vadr].T@(-m.opt.gravity*m.body_mass[jar_id]))*(~finger_mask)*np.clip(t-28,0,1)
 d.ctrl[:]=np.clip(tau,m.actuator_ctrlrange[:,0],m.actuator_ctrlrange[:,1]);peak=np.maximum(peak,np.abs(d.ctrl));mujoco.mj_step(m,d)
 if ref is None and t>=2:ref=d.xpos[bs].copy()
 for ct in d.contact:
  bn=[m.body(int(m.geom_bodyid[g])).name for g in [ct.geom1,ct.geom2]]
  gn=[m.geom(int(g)).name or '' for g in [ct.geom1,ct.geom2]]
  allowed=any(g.startswith('handle_col') for g in gn) and any(b.startswith('left_hand') for b in bn)
  if ct.dist<0 and not allowed and any(n.startswith(('left_','right_','torso','waist','pelvis','head')) for n in bn):bad={'time_s':t,'bodies':bn,'geoms':gn};break
 # Conservative global hand envelope from collision meshes / bounding radii.
 pts=[]
 for g in handgeoms:
  if m.geom_type[g]==mujoco.mjtGeom.mjGEOM_MESH:
   mesh=m.geom_dataid[g];v=m.mesh_vert[m.mesh_vertadr[mesh]:m.mesh_vertadr[mesh]+m.mesh_vertnum[mesh]];pts.append(v@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g])
  else:
   p=d.geom_xpos[g];r=m.geom_rbound[g];pts.append(np.array([p-r,p+r]))
 points=np.concatenate(pts);minz=float(points[:,2].min());maxx=float(points[:,0].max());gap=min(mujoco.mj_geomDistance(m,d,g,table,1,None) for g in handgeoms);gapmin=min(gapmin,gap)
 if minz>=.77:cleared=True
 if not cleared and maxx>.18:bad={'time_s':t,'reason':'hand entered table edge margin before clearing height','maxx':maxx,'minz':minz}
 if gap<.02:bad={'time_s':t,'reason':'hand-table gap below 20mm','gap_m':gap}
 forces={};supports=set()
 for ci,ct in enumerate(d.contact):
  if ct.dist>=0:continue
  bn=[m.body(int(m.geom_bodyid[g])).name for g in [ct.geom1,ct.geom2]]
  if 'chaleira' not in bn:continue
  other=bn[1-bn.index('chaleira')]
  if other.startswith('left_hand'):
   cf=np.zeros(6);mujoco.mj_contactForce(m,d,ci,cf);forces[other]=forces.get(other,0.)+float(cf[0])
  else:supports.add(other)
 bid=m.body('chaleira').id
 if t>=34:
  spout_actual=d.xpos[bid]+d.xmat[bid].reshape(3,3)@spout_local
  if supports:bad={'time_s':t,'reason':'kettle external support during transport','supports':sorted(supports)}
  if phase=='hold_over_filter':
   transport_steps+=1;transport_good+=int(not supports and np.linalg.norm(spout_actual-transport_target)<.01 and np.arccos(np.clip(d.xmat[bid].reshape(3,3)[2,2],-1,1))<np.deg2rad(10) and np.all(np.abs(np.rad2deg(d.qpos[ik.qa[-3:]]))<=[62,47,32]) and all(forces.get('left_hand_'+n+'_link',0)>1 for n in ['thumb_2','index_1','middle_1']))
 if phase=='hold_handle':
  hold_steps+=1;tilt=np.rad2deg(np.arccos(np.clip(d.xmat[bid].reshape(3,3)[2,2],-1,1)));opposition=any('thumb' in n and f>1 for n,f in forces.items()) and any(('index' in n or 'middle' in n) and f>1 for n,f in forces.items());hold_good+=int(not supports and opposition and d.xpos[bid,2]-z0>.05 and tilt<10 and np.all(np.abs(np.rad2deg(d.qpos[ik.qa[-3:]]))<=[62,47,32]))
 if k%16==0:grasp_rows.append({'t':t,'phase':phase,'jar_position_m':d.xpos[bid].tolist(),'jar_tilt_deg':float(np.rad2deg(np.arccos(np.clip(d.xmat[bid].reshape(3,3)[2,2],-1,1)))),'finger_normal_force_N':forces,'external_supports':sorted(supports),'spout_position_m':(d.xpos[bid]+d.xmat[bid].reshape(3,3)@spout_local).tolist(),'wrist_deg':np.rad2deg(d.qpos[ik.qa[-3:]]).tolist()})
 if k%16==0:rows.append({'t':t,'phase':phase,'hand_min_z_m':minz,'hand_max_x_m':maxx,'table_gap_m':gap,'prop_displacements_m':None if ref is None else np.linalg.norm(d.xpos[bs]-ref,axis=1).tolist()});states.append(d.qpos.copy())
 if bad:break
# Completion alone is insufficient: this exploratory run never certifies sustained grasp.
report={'model':'gpt-6-astra','scene':str(scene.resolve()),'transport_good_steps':transport_good,'transport_steps':transport_steps,'pass':bool(transport_steps>=int(1.9/dt) and transport_good==transport_steps and bad is None and hold_steps>=int(1.9/dt) and hold_good==hold_steps and not s.warnings()),'hold_good_steps':hold_good,'hold_steps':hold_steps,'completed_without_forbidden_contact':bad is None and cleared and t>45.9 and not s.warnings(),'payload_compensation':'Normal-directed finger preload12N per fingertip; arm gravity feedforward from current kettle COM and palm Jacobian; motors only','finger_trim_rad':finger_trim[ha].tolist(),'max_ik_error_m':maxik,'failure':bad,'min_table_gap_m':gapmin,'warnings':s.warnings(),'objects':props,'grasp_samples':grasp_rows,'rows':rows,'peak_motor_torques_Nm':{n:float(peak[i]) for n,i in s.act_joint.items()},'limitations':'fixed robot base, object masses/friction approximate; support/tilt/opposing-contact criteria checked each hold substep; 1kg assumed mass, fixed robot base'}
(out/'report.json').write_text(json.dumps(report,indent=2));np.savez_compressed(out/'trajectory.npz',qpos=states);np.save(out/'final-qpos.npy',d.qpos);np.save(out/'final-qvel.npy',d.qvel)
integration=np.zeros(mujoco.mj_stateSize(m,mujoco.mjtState.mjSTATE_INTEGRATION));mujoco.mj_getState(m,d,integration,mujoco.mjtState.mjSTATE_INTEGRATION);np.savez_compressed(out/'continuation.npz',integration=integration,last_target=target,hand_target=hq+finger_trim[ha],hand_names=hnames,kp=s.kp,kd=s.kd)
print({k:v for k,v in report.items() if k not in ['rows','grasp_samples','peak_motor_torques_Nm']});print(rows[-1])
