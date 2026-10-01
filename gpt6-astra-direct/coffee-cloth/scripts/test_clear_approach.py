"""Raise from arms-down and follow collision-planned pregrasp path with free props."""
import json,shutil,sys
from pathlib import Path
import numpy as np,mujoco
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'g1-cup-grasp/scripts'))
from g1_sim import G1Sim
out=Path('results/clear-approach-dynamics-001');out.mkdir(exist_ok=False);shutil.copy2(__file__,out/Path(__file__).name)
scene=Path('results/free-props-001/scene.xml');s=G1Sim(str(scene.resolve()));m,d=s.m,s.d;initial=np.load('results/free-props-001/initial-qpos.npy');d.qpos[:]=initial;mujoco.mj_forward(m,d)
roll=m.jnt_qposadr[m.joint('left_shoulder_roll_joint').id];elbow=m.jnt_qposadr[m.joint('left_elbow_joint').id]
raised=initial.copy();raised[roll]=1.2
above=raised.copy();above[elbow]=0.1
planned=np.load('results/clear-approach-001/path.npz')['qpos']
props=['coador','copo','pote','tampa','scoop','base_eletrica','chaleira'];bs=[m.body(n).id for n in props];ref=None
handgeoms=[g for g in range(m.ngeom) if m.body(int(m.geom_bodyid[g])).name.startswith(('left_hand','left_wrist')) and (m.geom_contype[g] or m.geom_conaffinity[g])];table=m.geom('tampo').id
rows=[];states=[];bad=None;gapmin=1.;dt=m.opt.timestep;peak=np.zeros(m.nu);cleared=False
for k in range(int(22/dt)):
 t=k*dt
 if t<2:target=initial;phase='settle'
 elif t<5:
  u=(t-2)/3;u=u*u*(3-2*u);target=initial*(1-u)+raised*u;phase='raise_outside_table'
 elif t<7:
  u=(t-5)/2;u=u*u*(3-2*u);target=raised*(1-u)+above*u;phase='bend_above_table'
 elif t<8:target=above;phase='hold_above_table'
 elif t<20:
  u=(t-8)/12*(len(planned)-1);i=min(int(u),len(planned)-2);f=u-i;target=planned[i]*(1-f)+planned[i+1]*f;phase='move_to_pregrasp'
 else:target=planned[-1];phase='hold_pregrasp'
 tau=s.kp*(target[s.qadr]-d.qpos[s.qadr])-s.kd*d.qvel[s.vadr]+d.qfrc_bias[s.vadr];d.ctrl[:]=np.clip(tau,m.actuator_ctrlrange[:,0],m.actuator_ctrlrange[:,1]);peak=np.maximum(peak,np.abs(d.ctrl));mujoco.mj_step(m,d)
 if ref is None and t>=2:ref=d.xpos[bs].copy()
 for ct in d.contact:
  bn=[m.body(int(m.geom_bodyid[g])).name for g in [ct.geom1,ct.geom2]]
  if ct.dist<0 and any(n.startswith(('left_','right_','torso','waist','pelvis','head')) for n in bn):bad={'time_s':t,'bodies':bn};break
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
 if k%16==0:rows.append({'t':t,'phase':phase,'hand_min_z_m':minz,'hand_max_x_m':maxx,'table_gap_m':gap,'prop_displacements_m':None if ref is None else np.linalg.norm(d.xpos[bs]-ref,axis=1).tolist()});states.append(d.qpos.copy())
 if bad:break
report={'model':'gpt-6-astra','scene':str(scene.resolve()),'pass':bad is None and cleared and t>21.9 and not s.warnings(),'failure':bad,'min_table_gap_m':gapmin,'warnings':s.warnings(),'objects':props,'rows':rows,'peak_motor_torques_Nm':{n:float(peak[i]) for n,i in s.act_joint.items()},'limitations':'fixed robot base, object masses/friction approximate; no grasp yet'}
(out/'report.json').write_text(json.dumps(report,indent=2));np.savez_compressed(out/'trajectory.npz',qpos=states);np.save(out/'final-qpos.npy',d.qpos)
print({k:v for k,v in report.items() if k not in ['rows','peak_motor_torques_Nm']});print(rows[-1])
