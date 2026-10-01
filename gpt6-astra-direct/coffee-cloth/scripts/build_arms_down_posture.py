"""Arms-down initial posture based on Unitree's published HOME arm targets."""
import json,sys,shutil,urllib.request
from pathlib import Path
import numpy as np,mujoco
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'g1-cup-grasp/scripts'))
from g1_sim import G1Sim,ARM_JOINTS
out=Path('results/arms-down-posture-003');out.mkdir(exist_ok=False);shutil.copy2(__file__,out/Path(__file__).name)
url='https://raw.githubusercontent.com/unitreerobotics/unitree_rl_mjlab/main/src/assets/robots/unitree_g1/g1_constants.py';(out/'unitree-g1-constants.py').write_bytes(urllib.request.urlopen(url).read())
s=G1Sim(str(Path('scene/setup-luiz-handle-v2.xml').resolve()));m,d=s.m,s.d
for side,sign in [('left',1),('right',-1)]:
 for n,v in zip(ARM_JOINTS,[.35,sign*.35,0,.87,0,0,0]):d.qpos[m.jnt_qposadr[m.joint(n.replace('right_',side+'_')).id]]=v
mujoco.mj_forward(m,d)
def contacts():
 return [dict(bodies=[m.body(int(m.geom_bodyid[g])).name for g in [ct.geom1,ct.geom2]],depth_m=float(-ct.dist)) for ct in d.contact if ct.dist<0 and any(m.body(int(m.geom_bodyid[g])).name.startswith(('left_','right_','torso','pelvis','waist','head')) for g in [ct.geom1,ct.geom2])]
initial=d.qpos.copy();bad=contacts();np.save(out/'initial-qpos.npy',initial)
ren=mujoco.Renderer(m,720,960);cam=mujoco.MjvCamera();cam.lookat[:]=[0,0,.85];cam.distance=1.65;cam.azimuth=150;cam.elevation=-12;ren.update_scene(d,camera=cam);Image.fromarray(ren.render()).save(out/'posture.png');ren.close()
s.q_des[:]=d.qpos[s.qadr];dynamic_bad=[]
for _ in range(int(2/m.opt.timestep)):
 tau=s.kp*(s.q_des-d.qpos[s.qadr])-s.kd*d.qvel[s.vadr]+d.qfrc_bias[s.vadr];d.ctrl[:]=np.clip(tau,m.actuator_ctrlrange[:,0],m.actuator_ctrlrange[:,1]);mujoco.mj_step(m,d)
 cb=contacts()
 if cb:dynamic_bad=cb;break
report={'model':'gpt-6-astra','source':url,'scope':'Adapted official simulation HOME: roll widened from 0.18 to 0.35rad for Dex3 hip clearance; not a verified firmware startup sequence. Existing fixed-base lower body retained.','arm_targets':{'shoulder_pitch':.35,'shoulder_roll_left':.35,'shoulder_roll_right':-.35,'elbow':.87,'others':0},'initial_contacts':bad,'settle_contacts':dynamic_bad,'simulated_seconds':float(d.time),'warnings':s.warnings(),'pass':not bad and not dynamic_bad and not s.warnings()}
(out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
