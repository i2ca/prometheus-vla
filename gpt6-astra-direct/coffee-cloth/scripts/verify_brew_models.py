"""Bounded component audit; never claims a completed robotic coffee episode."""
import json,shutil,subprocess,sys
from pathlib import Path
import numpy as np,mujoco
from coffee_grounds import CoffeeGrounds
from grounds_mass import GroundsMassCoupler
from kettle_heater import KettleHeater
out=Path('results/brew-models-003');out.mkdir(exist_ok=False)
for name in ['verify_brew_models.py','coffee_grounds.py','grounds_mass.py','kettle_heater.py','test_coffee_grounds.py','test_kettle_heater.py','kettle_liquid.py']:shutil.copy2(Path('scripts')/name,out/name)
checks={}
for name in ['test_coffee_grounds.py','test_kettle_heater.py']:
 p=subprocess.run([sys.executable,'scripts/'+name],capture_output=True,text=True);(out/(name+'.log')).write_text(p.stdout+p.stderr);checks[name]=p.returncode==0
m=mujoco.MjModel.from_xml_path('results/prepared-layout-007/scene.xml');d=mujoco.MjData(m);mujoco.mj_forward(m,d);g=CoffeeGrounds();c=GroundsMassCoupler(m);q=d.qpos.copy();v=d.qvel.copy();mass0=m.body_mass.sum();c.apply(d,g);checks['initial100g_payload']=abs(m.body_mass.sum()-mass0-.1)<1e-12;checks['state_preserved']=np.array_equal(q,d.qpos) and np.array_equal(v,d.qvel)
g.spoon_g=.4;g.pot_g-=.4;c.apply(d,g);checks['payload_mass_conserved']=abs(m.body_mass.sum()-mass0-.1)<1e-12;checks['spoon_payload_added']=abs(m.body_mass[m.body('scoop').id]-c.dry[m.body('scoop').id][0]-.0004)<1e-12
h=KettleHeater();trace=[]
for i in range(5000):
 h.step(.1,800,True,True,True,.2 if i<2 else .18,1 if i<2 else 0)
 if i%10==0:trace.append({'t':(i+1)*.1,'T_C':h.temperature_C,'on':h.on})
 if h.last_event=='automatic_boil_cutoff':break
checks['boiled_after_physical_switch_inputs']=h.last_event=='automatic_boil_cutoff';checks['energy_balance']=abs((h.temperature_C-25)*(390+.8*4180)-(h.delivered_energy_J-h.loss_energy_J))<1e-6
report={'model':'gpt-6-astra','pass':all(checks.values()),'checks':{k:bool(v) for k,v in checks.items()},'boil_time_s':(i+1)*.1,'heater':h.__dict__,'spoon_capacity_ml':g.capacity_ml,'nominal_level_spoon_capacity_g':g.capacity_ml*.35,'coffee_completed':False,'scope':'isolated mathematical and mass-coupling checks; switch inputs synthetic; no robotic dosing/heating yet','assumptions':{'bulk_density_g_ml':.35,'powder_initial_g':100,'spoon_inner_depth_mm':5,'pickup_efficiency':.55,'repose_deg':28,'thermal_efficiency':.85,'heat_loss_W_K':2.4,'body_heat_capacity_J_K':390},'limitations':['quasi-static added mass; no granular contact/drag or transferred momentum','thermal lumped model, no spatial gradients','no taste/extraction chemistry','payload and switch must be present at initialization before an integrated robotic claim'],'references':{'recipe':'https://www.abic.com.br/tudo-de-cafe/dicas-gerais/','manual':'assets/references/electrolux-handle-20260919/official-manual.pdf'}};(out/'report.json').write_text(json.dumps(report,indent=2));(out/'thermal-trace.json').write_text(json.dumps(trace,indent=2));print(report)
