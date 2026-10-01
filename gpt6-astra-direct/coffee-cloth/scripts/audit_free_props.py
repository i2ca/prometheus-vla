"""Remove artificial anchoring of tabletop props and test physical response."""
import json,shutil,sys,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np,mujoco
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'g1-cup-grasp/scripts'))
from g1_sim import G1Sim
out=Path('results/free-props-001');out.mkdir(exist_ok=False);shutil.copy2(__file__,out/Path(__file__).name)
source=Path('scene/setup-luiz-handle-v2.xml');tree=ET.parse(source)
for name in ['coador','base_eletrica']:
 b=tree.find(f".//body[@name='{name}']");assert b is not None and b.find('joint') is None and b.find('freejoint') is None;b.insert(0,ET.Element('freejoint',name=name+'_livre'))
scene=out/'scene.xml';tree.write(scene)
old=mujoco.MjModel.from_xml_path(str(source));oq=np.load('results/arms-down-posture-003/initial-qpos.npy')
s=G1Sim(str(scene.resolve()));m,d=s.m,s.d
for j in range(old.njnt):
 nj=m.joint(old.joint(j).name).id;n=7 if old.jnt_type[j]==mujoco.mjtJoint.mjJNT_FREE else 1;d.qpos[m.jnt_qposadr[nj]:m.jnt_qposadr[nj]+n]=oq[old.jnt_qposadr[j]:old.jnt_qposadr[j]+n]
initial=d.qpos.copy();np.save(out/'initial-qpos.npy',initial)
names=['coador','copo','pote','tampa','scoop','base_eletrica','chaleira'];bs=[m.body(n).id for n in names];cases=[]
for pushed in [None,'coador','copo','base_eletrica']:
 mujoco.mj_resetData(m,d);d.qpos[:]=initial;mujoco.mj_forward(m,d);start=d.xpos[bs].copy();peaktilt=np.zeros(len(bs));peakshift=np.zeros(len(bs));robot_contacts=set();rows=[]
 for k in range(int(3/m.opt.timestep)):
  t=k*m.opt.timestep;d.xfrc_applied[:]=0
  if pushed and 1<=t<1.1:d.xfrc_applied[m.body(pushed).id,:3]=[10,0,0] # explicit diagnostic perturbation, not grasp assistance
  tau=s.kp*(initial[s.qadr]-d.qpos[s.qadr])-s.kd*d.qvel[s.vadr]+d.qfrc_bias[s.vadr];d.ctrl[:]=np.clip(tau,m.actuator_ctrlrange[:,0],m.actuator_ctrlrange[:,1]);mujoco.mj_step(m,d)
  tilt=np.rad2deg(np.arccos(np.clip(d.xmat[bs].reshape(-1,3,3)[:,2,2],-1,1)));shift=np.linalg.norm(d.xpos[bs]-start,axis=1);peaktilt=np.maximum(tilt,peaktilt);peakshift=np.maximum(shift,peakshift)
  for ct in d.contact:
   bn=[m.body(int(m.geom_bodyid[g])).name for g in [ct.geom1,ct.geom2]]
   if ct.dist<0 and any(n.startswith(('left_','right_','torso','pelvis','waist','head')) for n in bn):robot_contacts.add(tuple(sorted(bn)))
  if k%25==0:rows.append({'t':t,'positions_m':d.xpos[bs].tolist(),'tilts_deg':tilt.tolist()})
 cases.append({'pushed':pushed,'force_N':[10,0,0] if pushed else [0,0,0],'force_duration_s':.1 if pushed else 0,'robot_contacts':sorted(robot_contacts),'warnings':s.warnings(),'objects':[{'name':n,'mass_kg':float(m.body_mass[b]),'peak_tilt_deg':float(peaktilt[i]),'peak_displacement_m':float(peakshift[i]),'final_position_m':d.xpos[b].tolist()} for i,(n,b) in enumerate(zip(names,bs))],'samples':rows});print(pushed,[(n,round(peaktilt[i],1),round(peakshift[i],3)) for i,n in enumerate(names)],flush=True)
(out/'report.json').write_text(json.dumps({'model':'gpt-6-astra','source':str(source),'scene':str(scene.resolve()),'freed':['coador','base_eletrica'],'all_prop_masses_are_unverified':True,'limitations':['coador cloth/support treated as one rigid object','electrical cable not modeled','push forces only diagnostic; none used in robot task'],'cases':cases},indent=2))
