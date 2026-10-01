"""Explicit optional grounds/heater state for NEW initialized brew scenes."""
import json
from pathlib import Path
import numpy as np,mujoco
from coffee_grounds import CoffeeGrounds
from grounds_mass import GroundsMassCoupler
from kettle_heater import KettleHeater
class BrewState:
 def __init__(self,model,source=None):
  self.m=model;self.enabled=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_NUMERIC,'brew_enabled')>=0
  self.samples=[]
  if not self.enabled:return
  self.grounds=CoffeeGrounds();self.heater=KettleHeater();self.mass=GroundsMassCoupler(model)
  if source is not None:
   file=Path(source)/'brew-state.json'
   if not file.exists():raise ValueError('brew scene continuation requires its original powder/thermal checkpoint')
   saved=json.loads(file.read_text())
   for name,value in saved['grounds'].items():setattr(self.grounds,name,value)
   for name,value in saved['heater'].items():setattr(self.heater,name,value)
  self.ids={n:model.body(n).id for n in ['scoop','pote','coador','chaleira','base_eletrica','kettle_rocker']};self.switch=model.joint('kettle_rocker_hinge').qposadr[0]
 def apply(self,data,water):
  if not self.enabled:return
  water.coffee_g=self.grounds.filter_g;self.mass.apply(data,self.grounds);self.visuals()
 def visuals(self):
  m=self.m;g=self.grounds
  gid=m.geom('grounds_pot_visual').id;h=max(.000001,g.bed_height_m-g.pot_floor_m);m.geom_size[gid,1]=h/2;m.geom_pos[gid,2]=g.pot_floor_m+h/2;m.geom_rgba[gid,3]=float(g.pot_g>1e-6)
  for name,mass in [('grounds_spoon_visual',g.spoon_g),('grounds_filter_visual',g.filter_g)]:m.geom_rgba[m.geom(name).id,3]=float(mass>1e-6)
 def step(self,data,dt,water):
  if not self.enabled:return None
  m,d=self.m,data;args=[]
  for n in ['scoop','pote','coador']:b=self.ids[n];args.extend([d.xpos[b],d.xmat[b].reshape(3,3)])
  row=self.grounds.step(dt,*args);b=self.ids['chaleira'];base=self.ids['base_eletrica'];support=0.;press=0.
  for i,c in enumerate(d.contact):
   if c.dist>=0:continue
   bodies=[int(m.geom_bodyid[g]) for g in [c.geom1,c.geom2]];f=np.zeros(6)
   if b in bodies and base in bodies:mujoco.mj_contactForce(m,d,i,f);support+=float(f[0])
   if self.ids['kettle_rocker'] in bodies and any(m.body(bb).name.startswith(('left_hand','right_hand')) for bb in bodies):mujoco.mj_contactForce(m,d,i,f);press+=float(f[0])
  # Electrical connection proxy: centered kettle with an actual base contact.
  # It is not a weighing switch. A transient reduction below half its weight
  # does not mean disconnection (heaterphysical005 had 7.06N of real support).
  # 0.1N rejects unloaded/numerically negligible contacts; not a calibrated connector.
  seated=np.linalg.norm(d.xpos[b,:2]-d.xpos[base,:2])<.008 and support>.1;upright=d.xmat[b].reshape(3,3)[2,2]>np.cos(np.deg2rad(5));thermal=self.heater.step(dt,water.source_ml,bool(seated),bool(upright),True,float(d.qpos[self.switch]),press);m.qpos_spring[self.switch]=.18 if self.heater.on else 0.;self.apply(d,water);record={'t':float(d.time),'grounds':row,'heater':thermal,'switch_contact_N':press,'base_support_N':support,'seated_proxy':bool(seated),'seated_support_threshold_N':.1};self.samples.append(record);return record
 def save(self,out):
  if not self.enabled:return
  out=Path(out);(out/'brew-state.json').write_text(json.dumps({'model':'gpt-6-astra','grounds':self.grounds.snapshot(),'heater':self.heater.__dict__},indent=2));(out/'brew-samples.json').write_text(json.dumps(self.samples,indent=2))
