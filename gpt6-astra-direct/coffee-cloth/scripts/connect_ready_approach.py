"""Connect forward ready posture to pregrasp through outward arm waypoints."""
from pathlib import Path
import json,shutil,numpy as np,mujoco
out=Path('results/handle-approach-005');out.mkdir(exist_ok=False);shutil.copy2(__file__,out/Path(__file__).name)
m=mujoco.MjModel.from_xml_path(str(Path('scene/setup-luiz-handle-v2.xml').resolve()));d=mujoco.MjData(m)
initial=np.load('results/ready-posture-002/initial-qpos.npy');old=np.load('results/handle-approach-002/path-c11-f0.95-d[-0.5, 1, 0].npz')['qpos'];tail=old[200:].copy()
# Keep unused right arm forward throughout.
ra=[m.jnt_qposadr[j] for j in range(m.njnt) if m.joint(j).name.startswith('right_') and m.jnt_type[j]==mujoco.mjtJoint.mjJNT_HINGE]
tail[:,ra]=initial[ra];pre=tail[0];trials=[];chosen=None
for roll in [.8,1.,1.2,1.4,1.6,1.8]:
 for pitch in [-.4,-.8,-1.2,-1.6]:
  mid=initial.copy();mid[m.jnt_qposadr[m.joint('left_shoulder_roll_joint').id]]=roll;mid[m.jnt_qposadr[m.joint('left_shoulder_pitch_joint').id]]=pitch
  mid2=pre.copy();mid2[m.jnt_qposadr[m.joint('left_shoulder_roll_joint').id]]=roll;mid2[m.jnt_qposadr[m.joint('left_shoulder_pitch_joint').id]]=pitch
  path=np.concatenate([[initial*(1-t)+mid*t for t in np.linspace(0,1,101)],[mid*(1-t)+mid2*t for t in np.linspace(0,1,201)],[mid2*(1-t)+pre*t for t in np.linspace(0,1,201)],tail[1:]])
  bad=None
  for i,state in enumerate(path):
   d.qpos[:]=state;mujoco.mj_forward(m,d)
   for ct in d.contact:
    if ct.dist>=-0.0001:continue
    gs=[m.geom(int(g)).name or '' for g in [ct.geom1,ct.geom2]];bs=[m.body(int(m.geom_bodyid[g])).name or '' for g in [ct.geom1,ct.geom2]];rob=[b.startswith(('left_','right_','torso','waist','pelvis','head')) for b in bs]
    if any(rob) and (all(rob) or not any(g.startswith('handle_col') for g in gs) or ct.dist<-.0007):bad={'sample':i,'bodies':bs};break
   if bad:break
  trials.append({'roll':roll,'pitch':pitch,'failure':bad})
  if bad is None:chosen=path;break
 if chosen is not None:break
if chosen is not None:np.savez_compressed(out/'path.npz',qpos=chosen)
(out/'report.json').write_text(json.dumps({'model':'gpt-6-astra','passed_static_path':chosen is not None,'trials':trials},indent=2));print(trials)
