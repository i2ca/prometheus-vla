"""Select symmetric ready posture with hands forward and clear of scene."""
import json,sys,shutil
from pathlib import Path
import numpy as np,mujoco
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'g1-cup-grasp/scripts'))
from g1_sim import G1Sim,ARM_JOINTS
out=Path('results/ready-posture-002');out.mkdir(exist_ok=False);shutil.copy2(__file__,out/Path(__file__).name)
s=G1Sim(str(Path('scene/setup-luiz-handle-v2.xml').resolve()));m,d=s.m,s.d;old=np.array(json.load(open('results/handle-reach-001/report.json'))['initial_qpos']);candidates=[]
for pitch in [-1.2,-1,-.8,-.6,-.4,-.2,0]:
 for roll in [.2,.3,.4,.5]:
  for elbow in [0,.3,.6,.9,1.2,1.5,1.8,2.]:
   d.qpos[:]=old
   for side,sign in [('left',1),('right',-1)]:
    for n,v in zip(ARM_JOINTS,[pitch,sign*roll,0,elbow,0,0,0]):d.qpos[m.jnt_qposadr[m.joint(n.replace('right_',side+'_')).id]]=v
   mujoco.mj_forward(m,d)
   bad=any(ct.dist<0 and any(m.body(int(m.geom_bodyid[g])).name.startswith(('left_','right_','torso','pelvis','waist','head')) for g in [ct.geom1,ct.geom2]) for ct in d.contact)
   p=d.xpos[m.body('left_wrist_yaw_link').id].copy()
   if not bad and p[0]>.10 and p[2]>.9:candidates.append((float(np.linalg.norm(p-[.18,.29,1.0])),pitch,roll,elbow,p.tolist(),d.qpos.copy()))
candidates.sort(key=lambda x:x[0]);assert candidates
best=candidates[0];np.save(out/'initial-qpos.npy',best[-1]);d.qpos[:]=best[-1];mujoco.mj_forward(m,d)
ren=mujoco.Renderer(m,600,800);cam=mujoco.MjvCamera();cam.lookat[:]=[.08,0,1];cam.distance=1.3;cam.azimuth=145;cam.elevation=-12;ren.update_scene(d,camera=cam);Image.fromarray(ren.render()).save(out/'posture.png');ren.close()
(out/'report.json').write_text(json.dumps({'model':'gpt-6-astra','accepted_static_candidates':len(candidates),'selected_pitch_roll_elbow':best[1:4],'left_palm_m':best[4],'scope':'static collision-free posture; not yet dynamic path validation'},indent=2));print(best[:-1])
