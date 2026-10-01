"""Necessary quasistatic contact-force feasibility at saved handle grasp.
Friction pyramid is conservative; no dynamic or robustness proof.
"""
import json,shutil
from pathlib import Path
import numpy as np,mujoco
from scipy.optimize import linprog
out=Path('results/grasp-wrench-audit-002');out.mkdir(exist_ok=False);shutil.copy2(__file__,out/Path(__file__).name)
p=Path('results/neutral-handle-dynamics-003');r=json.loads((p/'report.json').read_text());states=np.load(p/'trajectory.npz')['qpos'];idx=min(range(len(r['rows'])),key=lambda i:abs(r['rows'][i]['t']-28));m=mujoco.MjModel.from_xml_path(r['scene']);d=mujoco.MjData(m);d.qpos[:]=states[idx];mujoco.mj_forward(m,d);jar=m.body('chaleira').id;com=d.xipos[jar];contacts=[];cols=[];torques=[]
vadr=np.array([m.jnt_dofadr[m.actuator_trnid[i,0]] for i in range(m.nu)])
for ct in d.contact:
 gs=[int(ct.geom1),int(ct.geom2)];bs=[int(m.geom_bodyid[g]) for g in gs];gn=[m.geom(g).name or '' for g in gs]
 if ct.dist>=0 or jar not in bs or not any(g.startswith('handle_col') for g in gn):continue
 k=bs.index(jar);hb=bs[1-k];name=m.body(hb).name
 if not name.startswith('left_hand'):continue
 normal=ct.frame.reshape(3,3)[0]*(1 if k==1 else -1);t1,t2=ct.frame.reshape(3,3)[1:];mu=float(ct.friction[0]);jp=np.zeros((3,m.nv));jr=np.zeros((3,m.nv));mujoco.mj_jac(m,d,jp,jr,ct.pos,hb)
 contacts.append({'hand':name,'point_m':ct.pos.tolist(),'normal_on_jar':normal.tolist(),'mu':mu})
 for spin in [-1,-.7071,0,.7071,1]:
  for ang in np.linspace(0,2*np.pi,12,endpoint=False):
   force=normal+mu*np.sqrt(1-spin*spin)*(np.cos(ang)*t1+np.sin(ang)*t2)
   moment=normal*float(ct.friction[2])*spin
   cols.append(np.r_[force,np.cross(ct.pos-com,force)+moment]);torques.append(jp[:,vadr].T@force+jr[:,vadr].T@moment)
W=np.array(cols).T;T=np.array(torques).T;bias=d.qfrc_bias[vadr];limits=m.actuator_ctrlrange;desired=np.r_[-m.opt.gravity*m.body_mass[jar],[0,0,0]]
# tau_robot = bias + J^T force_on_jar, gear=1 for this model.
assert np.allclose(m.actuator_gear[:,0],1)
A=np.vstack([T,-T]);b=np.r_[limits[:,1]-bias,bias-limits[:,0]]
res=linprog(np.ones(W.shape[1]),A_ub=A,b_ub=b,A_eq=W,b_eq=desired,bounds=(0,None),method='highs')
report={'model':'gpt-6-astra','source':str(p),'sample_time_s':r['rows'][idx]['t'],'contact_count':len(contacts),'contacts':contacts,'feasible_with_torsional_friction':bool(res.success),'solver_message':res.message,'scope':'Conservative sampled 3D elliptic friction cone including spin; rigid contact locations; failure is not proof of infeasibility for full continuous model','gravity_wrench_about_com':desired.tolist()}
if res.success:
 report['normal_force_total_N']=float(res.x.sum());report['equality_residual']=float(np.linalg.norm(W@res.x-desired));report['motor_required_Nm']={m.actuator(i).name:float(v) for i,v in enumerate(bias+T@res.x)}
objective=np.r_[np.zeros(W.shape[1]),-1.]
maxload=linprog(objective,A_ub=np.column_stack([A,np.zeros(len(b))]),b_ub=b,A_eq=np.column_stack([W,-np.r_[-m.opt.gravity,[0,0,0]]]),b_eq=np.zeros(6),bounds=(0,None),method='highs')
report['maximum_supported_mass_kg_in_approximation']=float(maxload.x[-1]) if maxload.success else None
(out/'report.json').write_text(json.dumps(report,indent=2));print({k:v for k,v in report.items() if k not in ['contacts','motor_required_Nm']})
