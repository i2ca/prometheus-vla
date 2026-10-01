"""Torque-controlled handle-only approach/lift, free kettle, substep contact audit."""
import argparse,json,sys,shutil
from pathlib import Path
import numpy as np,mujoco
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'g1-cup-grasp/scripts'))
from g1_sim import G1Sim
from kinematics import ArmIK
ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--path-dir',type=Path,default=Path('results/handle-approach-002'));ap.add_argument('--lift-y',type=float,default=0);ap.add_argument('--path',default='path-c11-f0.9-d[0, 1, 0].npz');ap.add_argument('--grip',type=float,default=.12);ap.add_argument('--slow',type=float,default=1);ap.add_argument('--arm-gain',type=float,default=2);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False);shutil.copy2(__file__,a.out/Path(__file__).name)
r=json.loads(Path('results/handle-reach-001/report.json').read_text());c=next(x for x in r['results'] if x['candidate_index']==11 and x['side']=='left');pathfile=a.path_dir/a.path;shutil.copy2(pathfile,a.out/'approach.npz');path=np.load(pathfile)['qpos'];sim=G1Sim(r['scene']);m,d=sim.m,sim.d;d.qpos[:]=path[0];mujoco.mj_forward(m,d);sim.q_des[:]=d.qpos[sim.qadr]
for n,i in sim.act_joint.items():
 if 'hand' in n:sim.kp[i]=8;sim.kd[i]=.2
 elif n.startswith('left_'):sim.kp[i]*=a.arm_gain;sim.kd[i]*=np.sqrt(a.arm_gain)*2
ik=ArmIK(sim,c['joint_names'],palm='left_wrist_yaw_link');q=np.array(c['q']);end=path[-1].copy();ha=np.array([m.jnt_qposadr[m.joint(n).id] for n in c['hand_joint_names']]);end[ha]+=np.sign(end[ha])*a.grip
for idx,n in zip(ha,c['hand_joint_names']):end[idx]=np.clip(end[idx],*m.jnt_range[m.joint(n).id])
p=np.array(c['palm_target_m']);R=np.array(c['palm_rotation']);bid=m.body('chaleira').id;z0=d.xpos[bid,2];rows=[];badcounts={};firstbad=None;maxik=0;dt=m.opt.timestep;steps=int(15*a.slow/dt);traj=[]; torque_peak=np.zeros(m.nu); requested_peak=np.zeros(m.nu); saturated=np.zeros(m.nu,dtype=int)
for k in range(steps):
 t=k*dt/a.slow
 if t<1:target=path[0].copy();phase='settle'
 elif t<9:
  u=(t-1)/8*(len(path)-1);i=min(int(u),len(path)-2);f=u-i;target=path[i]*(1-f)+path[i+1]*f;phase='approach'
 elif t<10:target=path[-1]*(10-t)+end*(t-9);phase='close'
 else:
  phase='lift' if t<13 else 'hold';u=np.clip((t-10)/3,0,1);z=.08*(u*u*(3-2*u));target=end.copy()
  if k%10==0:q,e=ik.solve(p+[0,a.lift_y*(u*u*(3-2*u)),z],R,q,reference=np.array(c['q']),iterations=100);maxik=max(maxik,e['position_error_m'])
  target[ik.qa]=q
 sim.q_des[:]=target[sim.qadr];tau=sim.kp*(sim.q_des-d.qpos[sim.qadr])-sim.kd*d.qvel[sim.vadr]+d.qfrc_bias[sim.vadr];d.ctrl[:]=np.clip(tau,m.actuator_ctrlrange[:,0],m.actuator_ctrlrange[:,1]);torque_peak=np.maximum(torque_peak,np.abs(d.ctrl));requested_peak=np.maximum(requested_peak,np.abs(tau));saturated+=(np.abs(tau)>sim.tau_max);mujoco.mj_step(m,d)
 bad=[];fingers=set();supports=set();forces={}
 for ci,ct in enumerate(d.contact):
  if ct.dist>=0:continue
  gs=[m.geom(int(g)).name or '' for g in [ct.geom1,ct.geom2]];bs=[m.body(int(m.geom_bodyid[g])).name or '' for g in [ct.geom1,ct.geom2]];rob=[b.startswith(('right_','left_','torso','pelvis','waist','head')) for b in bs]
  if any(rob):
   if all(rob) or not any(g.startswith('handle_col') for g in gs):
    key=' / '.join(sorted(bs+gs));bad.append(key);badcounts[key]=badcounts.get(key,0)+1
  if 'chaleira' in bs:
   other=bs[1-bs.index('chaleira')]
   if other.startswith('left_hand') and any(g.startswith('handle_col') for g in gs):
    fingers.add(other);force=np.zeros(6);mujoco.mj_contactForce(m,d,ci,force);forces[other]=forces.get(other,0)+float(force[0])
   else:supports.add(other)
 if bad and firstbad is None:firstbad={'t':t,'phase':phase,'pairs':bad}
 if k%max(1,int(.033/dt))==0:
  mujoco.mj_forward(m,d);tilt=np.rad2deg(np.arccos(np.clip(d.xmat[bid].reshape(3,3)[2,2],-1,1)));rows.append({'t':t,'phase':phase,'lift_m':float(d.xpos[bid,2]-z0),'tilt_deg':float(tilt),'fingers':sorted(fingers),'supports':sorted(supports),'forces_N':forces});traj.append(d.qpos.copy())
 if firstbad or d.xpos[bid,2]<.7 or abs(d.qvel).max()>100:break
hold=[x for x in rows if x['phase']=='hold'];passed=bool(hold and len(hold)>40 and min(x['lift_m'] for x in hold)>.06 and max(x['tilt_deg'] for x in hold)<10 and all(not x['supports'] and len(x['fingers'])>=2 for x in hold) and not badcounts and not sim.warnings())
report={'model':'gpt-6-astra','path':str(pathfile),'scene':r['scene'],'grip_extra_rad':a.grip,'time_scale':a.slow,'arm_gain':a.arm_gain,'lift_y_m':a.lift_y,'passed':passed,'actuator_effort':[{'joint':n,'peak_applied_Nm':float(torque_peak[i]),'peak_requested_Nm':float(requested_peak[i]),'ctrl_limit_Nm':float(sim.tau_max[i]),'saturation_fraction':float(saturated[i]/(k+1))} for n,i in sim.act_joint.items()],'first_forbidden_contact':firstbad,'forbidden_counts':badcounts,'warnings':sim.warnings(),'max_ik_error_m':maxik,'rows':rows,'limitations':'1kg assumed mass; unmeasured friction/handle details; no liquid or real hardware; no weld or object teleport after initial state'};(a.out/'report.json').write_text(json.dumps(report,indent=2));np.savez_compressed(a.out/'trajectory.npz',qpos=np.array(traj));print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2));print('last',rows[-1])
