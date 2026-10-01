"""Motor-only reach, rocker press, withdrawal and energy-balance heating."""
import argparse,json,sys,shutil
from pathlib import Path
from dataclasses import asdict
import numpy as np,mujoco
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'g1-cup-grasp/scripts'))
from g1_sim import G1Sim,HAND_JOINTS
from kinematics import ArmIK
from elbow_anatomy import ElbowAnatomy
from brew_state import BrewState
from liquid_mass import LiquidMassCoupler
from kettle_liquid import KettleWater
ap=argparse.ArgumentParser();ap.add_argument('--hold-joints-during-heating',action='store_true');ap.add_argument('--source',type=Path,required=True);ap.add_argument('--plan',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);# Mesmos limites de cintura do alcance/conexao. Sem isso o q do plano fica
# fora dos bounds deste executor e o least_squares quebra com "each lower bound
# must be strictly less than each upper bound".
# A prensa desce no maximo 15 mm a 1,5 mm/s. O interruptor so liga com o
# balancim acima de 0,14 rad e mais de 0,5 N; com a chaleira deslocada o dedo
# chegou a 0,04 rad com 0,96 N, ou seja, forca sobrando e curso faltando.
ap.add_argument('--press-depth',type=float,default=.015)
# O balancim tem rigidez 0,3 N.m/rad e o ponto de contato fica a 18 mm do
# eixo, entao ligar (0,14 rad) exige 0,042 N.m, ou seja ~2,33 N no dedo. A
# prensa entregava 0,9 N, que da' 0,054 rad: bate com o medido. O que limita a
# forca e' este clamp de 3 mm no termo de correcao, constante de sintonia do
# controlador, nao fisica. O portao de forca excessiva (8 N) continua ativo.
ap.add_argument('--press-trim',type=float,default=.003)
# postura: pesos humanos via x_scale e braco ocioso com juntas fixas
ap.add_argument('--human-weights',action='store_true')
ap.add_argument('--fix-idle-arm',action='store_true')
# A ponta desce 18 mm mas o balancim so gira 0,05 rad e depois escapa: o dedo
# esta raspando a borda em vez de afundar a face. O ponto de IK (offset fixo
# dentro do link do indicador) nao e' o ponto de contato real da capsula, e com
# a chaleira deslocada essa diferenca passou a cair fora do balancim.
ap.add_argument('--press-offset',type=float,nargs=3,default=[0.,0.,0.],
                help='deslocamento do alvo da prensa em x y z (m)')
ap.add_argument('--waist-rp-deg',type=float,default=12.);ap.add_argument('--waist-yaw-rad',type=float,default=.7);ap.add_argument('--boil',action='store_true');a=ap.parse_args()
r=json.loads((a.source/'report.json').read_text());pr=json.loads((a.plan/'report.json').read_text());assert r['pass'] and pr['pass'];assert not (a.source/'INVALIDATED.json').exists();a.out.mkdir(exist_ok=False)
for p in [Path(__file__)]+[Path('scripts')/n for n in ['brew_state.py','coffee_grounds.py','grounds_mass.py','kettle_heater.py','liquid_mass.py','kettle_liquid.py','elbow_anatomy.py']]:shutil.copy2(p,a.out/p.name)
s=G1Sim(r['scene']);m,d=s.m,s.d;brew=BrewState(m,a.source);ck=np.load(a.source/'continuation.npz');water=KettleWater();coupler=LiquidMassCoupler(m,float(ck['dry_kettle_kg']))
for n,v in json.loads((a.source/'liquid-state.json').read_text()).items():setattr(water,n,v)
for n in ['body_mass','body_ipos','body_inertia','body_iquat','bvh_aabb']:getattr(m,n)[:]=ck[n]
mujoco.mj_setConst(m,mujoco.MjData(m));mujoco.mj_setState(m,d,ck['integration'],mujoco.mjtState.mjSTATE_INTEGRATION);mujoco.mj_forward(m,d);s.kp[:]=ck['kp'];s.kd[:]=ck['kd'];target=ck['last_target'].copy()
for n,i in s.act_joint.items():
 if n.startswith('left_hand'):s.kp[i]=8;s.kd[i]=.15
 elif 'waist' in n:s.kd[i]=15
 elif 'shoulder' in n:s.kd[i]=5
 elif 'elbow' in n:s.kd[i]=3
 elif 'wrist' in n:s.kd[i]=2.5
ik=ArmIK(s,pr['joint_names'],palm='left_wrist_yaw_link');bounds=ik.bounds.copy();bounds[0]=[-a.waist_yaw_rad,a.waist_yaw_rad];bounds[1:3]=np.deg2rad([[-a.waist_rp_deg,a.waist_rp_deg]]*2);anatomy=ElbowAnatomy(m);anatomy.bound_search(m,ik.joint_names,bounds)
for j in [7,14]:bounds[j:j+3]=np.deg2rad([[-60,60],[-45,45],[-30,30]])
sys.path.insert(0,'scripts');from postura_humana import pesos as _pesos
W_REG=_pesos(ik.joint_names) if a.human_weights else np.ones(len(ik.joint_names))
if a.fix_idle_arm:_q0=d.qpos[ik.qa][10:17].copy();bounds[10:17,0]=_q0-1e-4;bounds[10:17,1]=_q0+1e-4
ha=m.jnt_qposadr[[m.joint(n.replace('right_','left_')).id for n in HAND_JOINTS]];path=np.load(a.plan/'path.npz');poses=path['qpos'];phases=path['phases'];moving=np.r_[ik.qa,ha];durations=np.maximum(.016,np.max(np.abs(np.diff(poses[:,moving],axis=0)),axis=1)/.15);times=np.r_[0,np.cumsum(durations)];end=float(times[-1]);q=poses[-1,ik.qa].copy()
button=m.site('kettle_rocker_tip').id;tip=m.body('left_hand_index_1_link').id;tiplocal=np.array([.049,-.002,0]);palm=m.body('left_wrist_yaw_link').id;other=m.body('right_wrist_yaw_link').id;hot=m.geom('chaleira_hot_body').id;table=m.geom('tampo').id
hands=[g for g in range(m.ngeom) if m.geom_contype[g] and m.body(int(m.geom_bodyid[g])).name.startswith(('left_hand','left_wrist','right_hand','right_wrist'))];leftg=[g for g in hands if m.body(int(m.geom_bodyid[g])).name.startswith('left_')];props=[m.body(n).id for n in ['chaleira','base_eletrica','scoop','pote','tampa','coador','copo']];prop0=d.xpos[props].copy();initialwater=water.spilled_ml
p0=None;R0=None;op=None;oR=None;press_start=None;turned_on=None;withdraw_start=None;withdraw_p=None;trim=np.zeros(3);passed=False;bad=None;rows=[];states=[];samples=[];liquid=[];peak=np.zeros(m.nu);peak_force=0.;min_hot=1.;dt=m.opt.timestep
for k in range(int((end+380 if a.boil else end+35)/dt)):
 t=k*dt
 if t<end:
  i=min(int(np.searchsorted(times,t,side='right')-1),len(poses)-2);u=np.clip((t-times[i])/durations[i],0,1);target[moving]=poses[i,moving]*(1-u)+poses[i+1,moving]*u;phase=str(phases[i+1])
 else:
  phase='settle_button';target[ha]=poses[-1,ha]
  if p0 is None:p0=d.site_xpos[button].copy()+[0,0,.006]+np.array(a.press_offset);R0=d.xmat[palm].reshape(3,3).copy();op=d.xpos[other].copy();oR=d.xmat[other].reshape(3,3).copy();press_start=t+1
  goal=p0.copy()
  if t>=press_start:
   phase='press_button';goal[2]-=min(a.press_depth,(t-press_start)*.0015)
  if brew.heater.on and turned_on is None:turned_on=t;withdraw_start=t;withdraw_p=d.xpos[tip]+d.xmat[tip].reshape(3,3)@tiplocal
  if turned_on is not None:
   rt=t-withdraw_start;phase='withdraw_button' if rt<5 else 'heat_water';goal=withdraw_p+np.array([0,0,.065*min(1,rt/5)])
  if k%10==0 and not (a.hold_joints_during_heating and phase=='heat_water'):
   actual=d.xpos[tip]+d.xmat[tip].reshape(3,3)@tiplocal;trim=.9*trim+.1*np.clip(.5*(goal-actual),-a.press_trim,a.press_trim)
   def residual(x):
    pp,rr=ik.fk(x);point=ik.d.xpos[tip]+ik.d.xmat[tip].reshape(3,3)@tiplocal;thermal=[min(0,mujoco.mj_geomDistance(m,ik.d,g,hot,.02,None)-.010) for g in leftg]
    return np.r_[(point-goal-trim)*1000,Rotation.from_matrix(rr@R0.T).as_rotvec()*30,(ik.d.xpos[other]-op)*(0 if a.fix_idle_arm else 500),Rotation.from_matrix(ik.d.xmat[other].reshape(3,3)@oR.T).as_rotvec()*(0 if a.fix_idle_arm else 20),(x-q)*.1*W_REG,np.array(thermal)*5000,anatomy.penalty(ik.d)]
   lo=np.maximum(bounds[:,0],q-.004);hi=np.minimum(bounds[:,1],q+.004);q=least_squares(residual,np.clip(q,lo+1e-9,hi-1e-9),bounds=(lo,hi),max_nfev=30,x_scale=1/W_REG).x
  target[ik.qa]=q
 tau=s.kp*(target[s.qadr]-d.qpos[s.qadr])-s.kd*d.qvel[s.vadr]+d.qfrc_bias[s.vadr];d.ctrl[:]=np.clip(tau,m.actuator_ctrlrange[:,0],m.actuator_ctrlrange[:,1]);peak=np.maximum(peak,np.abs(d.ctrl));mujoco.mj_step(m,d);force=0.
 for ci,c in enumerate(d.contact):
  if c.dist>=0:continue
  bn=[m.body(int(m.geom_bodyid[g])).name for g in [c.geom1,c.geom2]];allowed=set(bn)=={'left_hand_index_1_link','kettle_rocker'} and phase in ['approach_button','settle_button','press_button','withdraw_button']
  if allowed:f=np.zeros(6);mujoco.mj_contactForce(m,d,ci,f);force+=float(f[0])
  if any(n.startswith(('left_','right_','torso','waist','pelvis','head')) for n in bn) and not allowed:bad={'reason':'forbidden contact','bodies':bn}
 peak_force=max(peak_force,force);hotgap=min(mujoco.mj_geomDistance(m,d,g,hot,1,None) for g in leftg);min_hot=min(min_hot,hotgap);tablegap=min(mujoco.mj_geomDistance(m,d,g,table,1,None) for g in hands)
 if hotgap<.006:bad={'reason':'hot metal clearance','m':float(hotgap)}
 if tablegap<.02:bad={'reason':'table clearance','m':float(tablegap)}
 if force>8:bad={'reason':'excessive switch force','N':force}
 if not anatomy.valid(d,tolerance_deg=2):bad={'reason':'elbow geometry','deg':anatomy.angles(d).tolist()}
 if np.any(np.abs(np.rad2deg(d.qpos[ik.qa[np.r_[7:10,14:17]]]))>[62,47,32,62,47,32]):bad={'reason':'wrist limits'}
 if np.max(np.linalg.norm(d.xpos[props]-prop0,axis=1))>.005:bad={'reason':'prop displacement above5mm'}
 if k%10==0:
  brew.step(d,.02,water);bs=[m.body(n).id for n in ['chaleira','coador','copo']];flow=water.step(.02,*sum(([d.xpos[b],d.xmat[b].reshape(3,3)] for b in bs),[]));coupler.apply(d,water);liquid.append({'t':float(d.time),**flow})
 if water.spilled_ml>initialwater+.5:bad={'reason':'water spill'}
 if t>end+25 and turned_on is None:bad={'reason':'physical button press timed out'}
 if s.warnings():bad={'reason':'numerical warning','warnings':s.warnings()}
 if turned_on is not None and t-turned_on>7:
  passed=(not a.boil and brew.heater.on and force<.05) or (a.boil and brew.heater.last_event=='automatic_boil_cutoff' and brew.heater.temperature_C>99 and force<.05)
 if k%16==0:states.append(d.qpos.copy());rows.append({'t':float(d.time),'phase':phase,'table_gap_m':float(tablegap)});samples.append({'t':float(d.time),'phase':phase,'switch_force_N':force,'angle_rad':float(d.qpos[brew.switch]),'temperature_C':brew.heater.temperature_C,'hot_clearance_m':float(hotgap),'elbow_flexion_deg':anatomy.angles(d).tolist()})
 if k%1000==0:
  _tip=d.xpos[tip]+d.xmat[tip].reshape(3,3)@tiplocal
  print({'t':round(t,2),'phase':phase,'force':round(force,4),'switch':round(float(d.qpos[brew.switch]),4),'on':brew.heater.on,'C':round(float(brew.heater.temperature_C),2),'tip_z':round(float(_tip[2]),5),'goal_z':round(float(goal[2]),5) if 'goal' in dir() else None,'lag_mm':round(float((_tip[2]-goal[2])*1000),3) if 'goal' in dir() else None},flush=True)
 if bad or passed:break
report={'hold_joints_during_heating':a.hold_joints_during_heating,'model':'gpt-6-astra','source':str(a.source),'plan':str(a.plan),'scene':r['scene'],'pass':bool(passed and not bad),'failure':bad or (None if passed else {'reason':'heater acceptance not reached'}),'initial_water_ml':water.initial_ml,'coffee_completed':False,'scope':'motor-only press/withdraw; reduced lumped thermal model; fixed pelvis','warnings':s.warnings(),'peak_switch_force_N':peak_force,'min_hot_clearance_m':min_hot,'turned_on_at_elapsed_s':turned_on,'heater':asdict(brew.heater),'water_state':asdict(water),'rows':rows,'grasp_samples':samples,'liquid_samples':liquid,'peak_motor_torques_Nm':{n:float(peak[i]) for n,i in s.act_joint.items()}}
(a.out/'report.json').write_text(json.dumps(report,indent=2));np.savez_compressed(a.out/'trajectory.npz',qpos=states);integration=np.zeros(mujoco.mj_stateSize(m,mujoco.mjtState.mjSTATE_INTEGRATION));mujoco.mj_getState(m,d,integration,mujoco.mjtState.mjSTATE_INTEGRATION);np.savez_compressed(a.out/'continuation.npz',integration=integration,last_target=target,kp=s.kp,kd=s.kd,body_mass=m.body_mass,body_ipos=m.body_ipos,body_inertia=m.body_inertia,body_iquat=m.body_iquat,bvh_aabb=m.bvh_aabb,dry_kettle_kg=float(ck['dry_kettle_kg']));(a.out/'liquid-state.json').write_text(json.dumps(asdict(water),indent=2));brew.save(a.out);print({k:v for k,v in report.items() if k not in ['rows','grasp_samples','liquid_samples','peak_motor_torques_Nm']})
