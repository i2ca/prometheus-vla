"""New episode initialization and long passive stability test; not a continuation of old episode."""
import json,sys,shutil,xml.etree.ElementTree as ET
from pathlib import Path
import mujoco,numpy as np
from dataclasses import asdict
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'g1-cup-grasp/scripts'))
from g1_sim import G1Sim
from liquid_mass import LiquidMassCoupler
from kettle_liquid import KettleWater
from brew_state import BrewState
from prepared_layout_gate import initial_placement_integrity
out=Path('results/prepared-layout-013');out.mkdir(exist_ok=False)
for p in [Path(__file__),Path('scripts/liquid_mass.py'),Path('scripts/kettle_liquid.py'),Path('scripts/prepared_layout_gate.py')]:shutil.copy2(p,out/p.name)
tree=ET.parse('results/brew-scene-001/scene.xml');tree.getroot().find('option').set('noslip_iterations','10');cup=tree.find('.//body[@name="copo"]');cup.set('pos','0.28 0.065 0.753');cup.set('quat','0.707106781 0 0 -0.707106781');spoon=tree.find('.//body[@name="scoop"]');spoon.set('pos','0.211 -0.489 0.751');spoon.set('quat','0.382683432365 0 0 -0.923879532511');scene=(out/'scene.xml').resolve()
for g in tree.findall('.//body[@name="scoop"]/geom'):
 if g.get('name','').startswith('scoop_col'):g.set('condim','4');g.set('priority','1');g.set('solref','0.008 1')
size=tree.getroot().find('size')
if size is None:size=ET.SubElement(tree.getroot(),'size')
size.set('memory','256M')
tree.write(scene)
s=G1Sim(str(scene));m,d=s.m,s.d;brew=BrewState(m);poses=np.load('results/loadable-dynamics-019/trajectory.npz')['qpos'];d.qpos[s.qadr]=poses[0,s.qadr];mujoco.mj_forward(m,d);water=KettleWater(initial_ml=800);coupler=LiquidMassCoupler(m,.78);coupler.apply(d,water);brew.apply(d,water);target=d.qpos.copy();props=['chaleira','coador','copo','pote','tampa','scoop','base_eletrica'];ids=[m.body(n).id for n in props];positions=[];rows=[];states=[];bad=[];startpos=None;liquid=[]
for k in range(int(65/m.opt.timestep)):
 t=k*m.opt.timestep
 if k%10==0 and k:brew.step(d,.02,water)
 if k%10==0 and k:
  flow=water.step(.02,d.xpos[m.body('chaleira').id],d.xmat[m.body('chaleira').id].reshape(3,3),d.xpos[m.body('coador').id],d.xmat[m.body('coador').id].reshape(3,3),d.xpos[m.body('copo').id],d.xmat[m.body('copo').id].reshape(3,3));coupler.apply(d,water);liquid.append({'t':float(d.time),**flow})
 tau=s.kp*(target[s.qadr]-d.qpos[s.qadr])-s.kd*d.qvel[s.vadr]+d.qfrc_bias[s.vadr];d.ctrl[:]=np.clip(tau,m.actuator_ctrlrange[:,0],m.actuator_ctrlrange[:,1]);mujoco.mj_step(m,d)
 for c in d.contact:
  if c.dist<0 and any(m.body(int(m.geom_bodyid[g])).name.startswith(('left_','right_','torso','pelvis','waist','head')) for g in [c.geom1,c.geom2]):bad.append({'t':float(d.time),'bodies':[m.body(int(m.geom_bodyid[g])).name for g in [c.geom1,c.geom2]]})
 if k%16==0:
  positions.append(d.xpos[ids].copy());states.append(d.qpos.copy());rows.append({'t':float(d.time),'phase':'settle_prepared_layout' if t<5 else 'passive_stability'})
 if k%5000==0:print({'time':t,'positions':{n:d.xpos[b].tolist() for n,b in zip(props,ids)}},flush=True)
 if bad or s.warnings():break
positions=np.array(positions);mask=np.array([r['t'] for r in rows])>=5;delta=np.max(np.linalg.norm(positions[mask]-positions[mask][0],axis=2),axis=0) if np.any(mask) else np.ones(len(props));tilts={n:float(np.rad2deg(np.arccos(np.clip(d.xmat[b].reshape(3,3)[2,2],-1,1)))) for n,b in zip(props,ids)};integrity=initial_placement_integrity(positions);passed=integrity['pass'] and tilts['scoop']<20 and not bad and not s.warnings() and max(delta)<.003 and water.spilled_ml==0 and tilts['coador']<2 and tilts['copo']<2
report={'model':'gpt-6-astra','scene':str(scene),'pass':bool(passed),'scope':'NEW episode initial placement: corrected cup clearance, confirmed20mm lid knob,120mm white spoon initially placed with handle over front edge on RIGHT side,100g ground coffee in pot with coupled mass, top-handle rocker,800ml cold water, empty cup; not a continuous robot manipulation of previous episode','coffee_completed':False,'initial_placement_integrity':integrity,'failure_contacts':bad[:20],'warnings':s.warnings(),'displacement_after5s_m':dict(zip(props,map(float,delta))),'final_tilt_deg':tilts,'rows':rows,'liquid_samples':liquid,'initial_powder_g':100,'thermal_model':'lumped EEK10 proxy, cold and off','initial_water_ml':800,'water_state':asdict(water)};(out/'report.json').write_text(json.dumps(report,indent=2));np.savez_compressed(out/'trajectory.npz',qpos=states);integration=np.zeros(mujoco.mj_stateSize(m,mujoco.mjtState.mjSTATE_INTEGRATION));mujoco.mj_getState(m,d,integration,mujoco.mjtState.mjSTATE_INTEGRATION);np.savez_compressed(out/'continuation.npz',integration=integration,last_target=target,kp=s.kp,kd=s.kd,dry_kettle_kg=.78);(out/'liquid-state.json').write_text(json.dumps(asdict(water),indent=2));print({k:v for k,v in report.items() if k not in ['rows','liquid_samples']})

brew.save(out)
