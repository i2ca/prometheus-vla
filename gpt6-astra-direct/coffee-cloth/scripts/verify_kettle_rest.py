"""Verify free rest on base using a force/time window rather than solver peaks."""
import sys,json,shutil
from pathlib import Path
import numpy as np,mujoco
from dataclasses import asdict
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'g1-cup-grasp/scripts'))
from g1_sim import G1Sim
from kettle_liquid import KettleWater
from liquid_mass import LiquidMassCoupler
source=Path('results/kettle-place-019');out=Path('results/kettle-rest-002');out.mkdir(exist_ok=False)
for p in [Path(__file__),*[Path(__file__).parent/n for n in ['kettle_liquid.py','liquid_mass.py']]]:shutil.copy2(p,out/p.name)
r=json.loads((source/'report.json').read_text());assert r['failure']=={'reason':'placement/release acceptance not reached'} and not r['warnings'];s=G1Sim(r['scene']);m,d=s.m,s.d;ck=np.load(source/'continuation.npz');water=KettleWater(initial_ml=r['initial_water_ml']);coupler=LiquidMassCoupler(m,float(ck['dry_kettle_kg']))
for n,v in json.loads((source/'liquid-state.json').read_text()).items():setattr(water,n,v)
mujoco.mj_setState(m,d,ck['integration'],mujoco.mjtState.mjSTATE_INTEGRATION);coupler.apply(d,water);s.kp[:]=ck['kp'];s.kd[:]=ck['kd'];target=ck['last_target'];jar=m.body('chaleira').id;fil=m.body('coador').id;cup=m.body('copo').id;base=m.body('base_eletrica').id;va=m.jnt_dofadr[m.joint('chaleira_livre').id];rows=[];states=[];samples=[];liquid=[];bad=None;start=float(d.time);dt=m.opt.timestep
for k in range(int(3/dt)):
 if k%10==0:
  if k:liquid.append({'t':float(d.time),**water.step(dt*10,d.xpos[jar],d.xmat[jar].reshape(3,3),d.xpos[fil],d.xmat[fil].reshape(3,3),d.xpos[cup],d.xmat[cup].reshape(3,3))})
  coupler.apply(d,water)
 tau=s.kp*(target[s.qadr]-d.qpos[s.qadr])-s.kd*d.qvel[s.vadr]+d.qfrc_bias[s.vadr];d.ctrl[:]=np.clip(tau,m.actuator_ctrlrange[:,0],m.actuator_ctrlrange[:,1]);mujoco.mj_step(m,d);support=0.;hand_force=0.;base_contacts=0
 for ci,c in enumerate(d.contact):
  if c.dist>=0:continue
  bs=[int(m.geom_bodyid[g]) for g in [c.geom1,c.geom2]];names=[m.body(b).name for b in bs]
  if any(n.startswith(('left_','right_','torso','pelvis','waist','head')) for n in names):bad={'t':float(d.time),'bodies':names}
  if jar in bs:
   other=bs[1-bs.index(jar)];cf=np.zeros(6);mujoco.mj_contactForce(m,d,ci,cf)
   if other==base:support+=(1 if bs[1]==jar else -1)*float((c.frame.reshape(3,3).T@cf[:3])[2]);base_contacts+=1
   else:bad={'t':float(d.time),'reason':'kettle contact outside base','body':m.body(other).name}
 tilt=float(np.rad2deg(np.arccos(np.clip(d.xmat[jar].reshape(3,3)[2,2],-1,1))));error=float(np.linalg.norm(d.xpos[jar,:2]-d.xpos[base,:2]));linear=float(np.linalg.norm(d.qvel[va:va+3]));angular=float(np.linalg.norm(d.qvel[va+3:va+6]))
 if k*dt>=1:samples.append({'t':float(d.time),'support_vertical_N':support,'base_contacts':base_contacts,'center_error_m':error,'tilt_deg':tilt,'linear_speed_m_s':linear,'angular_speed_rad_s':angular,'jar_position_m':d.xpos[jar].tolist(),'jar_rotation':d.xmat[jar].reshape(3,3).tolist()})
 if k%16==0:rows.append({'t':float(d.time),'phase':'verify_free_rest'});states.append(d.qpos.copy())
 if bad:break
weight=float(m.body_mass[jar]*9.81);mean=float(np.mean([x['support_vertical_N'] for x in samples])) if samples else 0.;geometry=bool(samples) and all(x['center_error_m']<.012 and x['tilt_deg']<5 for x in samples);rms_linear=float(np.sqrt(np.mean([x['linear_speed_m_s']**2 for x in samples]))) if samples else 1.;rms_angular=float(np.sqrt(np.mean([x['angular_speed_rad_s']**2 for x in samples]))) if samples else 1.;drift=max(np.linalg.norm(np.array(x['jar_position_m'])-samples[0]['jar_position_m']) for x in samples) if samples else 1.;rotation_drift=max(float(np.arccos(np.clip((np.trace(np.array(x['jar_rotation'])@np.array(samples[0]['jar_rotation']).T)-1)/2,-1,1))) for x in samples) if samples else 1.;geometry=geometry and rms_linear<.01 and rms_angular<.2 and drift<.001 and rotation_drift<np.deg2rad(.5);contact_fraction=float(np.mean([x['base_contacts']>0 for x in samples])) if samples else 0.;passed=bool(bad is None and len(samples)==int(2/dt) and geometry and .9*weight<mean<1.1*weight and contact_fraction>.95 and not s.warnings())
report={'model':'gpt-6-astra','source':str(source),'scene':r['scene'],'pass':passed,'coffee_completed':False,'failure':bad,'warnings':s.warnings(),'initial_water_ml':water.initial_ml,'water_state':asdict(water),'mean_vertical_support_N':mean,'weight_N':weight,'base_contact_fraction':contact_fraction,'geometry_and_speed_pass':bool(geometry),'rms_linear_speed_m_s':rms_linear,'rms_angular_speed_rad_s':rms_angular,'max_position_drift_m':float(drift),'max_rotation_drift_deg':float(np.rad2deg(rotation_drift)),'verification_samples':samples,'rows':rows,'liquid_samples':liquid,'scope':'new3s physical continuation;1s settle+2s free rest; no robot contact, mean vertical support within10%weight, tilt<5deg, center<12mm, RMS speed<10mm/s and angular<.2rad/s, drift<1mm/.5deg; does not certify electrical connector alignment'}
if not passed and bad is None:report['failure']={'reason':'rest window criteria failed'}
(out/'report.json').write_text(json.dumps(report,indent=2));np.savez_compressed(out/'trajectory.npz',qpos=states);np.save(out/'final-qpos.npy',d.qpos);np.save(out/'final-qvel.npy',d.qvel);integration=np.zeros(mujoco.mj_stateSize(m,mujoco.mjtState.mjSTATE_INTEGRATION));mujoco.mj_getState(m,d,integration,mujoco.mjtState.mjSTATE_INTEGRATION);np.savez_compressed(out/'continuation.npz',integration=integration,last_target=target,hand_target=ck['hand_target'],hand_names=ck['hand_names'],kp=s.kp,kd=s.kd,body_mass=m.body_mass,body_ipos=m.body_ipos,body_inertia=m.body_inertia,body_iquat=m.body_iquat,water_ml=water.source_ml,dry_kettle_kg=float(ck['dry_kettle_kg']));(out/'liquid-state.json').write_text(json.dumps(asdict(water),indent=2));print({k:v for k,v in report.items() if k not in ['verification_samples','rows','liquid_samples']})
