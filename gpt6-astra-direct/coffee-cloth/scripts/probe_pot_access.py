"""Offline spoon/pot collision-envelope diagnostic; never a physical manipulation."""
from pathlib import Path
import json,shutil
import numpy as np,mujoco
from scipy.spatial.transform import Rotation
out=Path('results/pot-access-002');out.mkdir(exist_ok=False);shutil.copy2(__file__,out/Path(__file__).name);m=mujoco.MjModel.from_xml_path('results/prepared-layout-008/scene.xml');d=mujoco.MjData(m);mujoco.mj_forward(m,d);pot=m.body('pote').id;spoon=m.body('scoop').id;qa=m.jnt_qposadr[m.body_jntadr[spoon]];p=d.xpos[pot].copy();rows=[]
for angle in [0,15,25,35,45,55,65]:
 for dx in [-.025,0,.020]:
  for z in [.038,.045,.052,.060,.075]:
   R=Rotation.from_euler('ZY',[180,-angle],degrees=True).as_matrix();goal=p+np.array([dx,0,z]);d.qpos[qa:qa+3]=goal-R@np.array([-.045,0,.003]);quat=Rotation.from_matrix(R).as_quat();d.qpos[qa+3:qa+7]=quat[[3,0,1,2]];mujoco.mj_forward(m,d);contacts=[]
   for c in d.contact:
    if c.dist<0 and {int(m.geom_bodyid[g]) for g in [c.geom1,c.geom2]}=={pot,spoon}:contacts.append(float(-c.dist))
   rows.append({'pitch_deg':angle,'bowl_offset_m':[dx,0,z],'clear':not contacts,'penetration_m':max(contacts,default=0)})
(out/'report.json').write_text(json.dumps({'model':'gpt-6-astra','scope':'offline spoon/pot geometric compatibility only','rows':rows},indent=2));print('clear below powder surface .0436m',[r for r in rows if r['clear'] and r['bowl_offset_m'][2]<.0436]);print('clear above',[r for r in rows if r['clear'] and r['bowl_offset_m'][2]<.061])
