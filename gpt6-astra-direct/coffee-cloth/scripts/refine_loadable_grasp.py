"""Refine actual left-hand/arm configuration to remove IK-induced penetration."""
import json,shutil
from pathlib import Path
import numpy as np,mujoco
from scipy.optimize import least_squares
out=Path('results/loadable-refined-005');out.mkdir(exist_ok=False);shutil.copy2(__file__,out/Path(__file__).name)
r=json.loads(Path('results/loadable-exact-002/report.json').read_text());c=r['grasp'];m=mujoco.MjModel.from_xml_path(r['scene']);d=mujoco.MjData(m);base=np.array(c['qpos']);d.qpos[:]=base
names=c['joint_names']+c['hand_joint_names'];js=np.array([m.joint(n).id for n in names]);qa=m.jnt_qposadr[js];q0=base[qa];bounds=m.jnt_range[js].copy();bounds[0]=[-.5,.5];bounds[5:8]=np.deg2rad([[-60,60],[-45,45],[-30,30]]);bounds[:,0]=np.maximum(bounds[:,0],q0-.25);bounds[:,1]=np.minimum(bounds[:,1],q0+.25);bounds[8:,0]=np.maximum(m.jnt_range[js[8:],0],q0[8:]-.35);bounds[8:,1]=np.minimum(m.jnt_range[js[8:],1],q0[8:]+.35)
handgs=[g for g in range(m.ngeom) if (m.geom_contype[g] or m.geom_conaffinity[g]) and m.body(int(m.geom_bodyid[g])).name.startswith(('left_hand','left_wrist'))];tips=[next(g for g in handgs if m.body(int(m.geom_bodyid[g])).name=='left_hand_'+n+'_link') for n in ['thumb_2','index_1','middle_1']];handles=[g for g in range(m.ngeom) if (m.geom(g).name or '').startswith('handle_col')];hot=m.geom('chaleira_hot_body').id

def distances():return np.array([min(mujoco.mj_geomDistance(m,d,g,h,.1,None) for h in handles) for g in handgs])
def residual(q):
 d.qpos[:]=base;d.qpos[qa]=q;mujoco.mj_forward(m,d);dist=distances();metal=np.array([mujoco.mj_geomDistance(m,d,g,hot,.1,None) for g in handgs]);return np.r_[(dist[[handgs.index(g) for g in tips]]-.0003)*1000,np.minimum(dist-.0002,0)*8000,np.minimum(metal-.005,0)*8000,(q-q0)*.03]
sol=least_squares(residual,np.clip(q0,bounds[:,0]+1e-8,bounds[:,1]-1e-8),bounds=bounds.T,max_nfev=200,diff_step=1e-4);residual(sol.x);palm=m.body('left_wrist_yaw_link').id
c['qpos']=d.qpos.tolist();c['q']=sol.x[:8].tolist();c['hand_q']=sol.x[8:].tolist();c['palm_target_m']=d.xpos[palm].tolist();c['palm_rotation']=d.xmat[palm].reshape(3,3).tolist();dist=distances();metal=min(mujoco.mj_geomDistance(m,d,g,hot,.1,None) for g in handgs);bad=[]
for ct in d.contact:
 bn=[m.body(int(m.geom_bodyid[g])).name for g in [ct.geom1,ct.geom2]];gn=[m.geom(int(g)).name or '' for g in [ct.geom1,ct.geom2]]
 if ct.dist<0 and any(n.startswith(('left_','right_','torso','pelvis','waist','head')) for n in bn):
  if not(any(g.startswith('handle_col') for g in gn) and any(n.startswith('left_hand') for n in bn) and ct.dist>=-.0007):bad.append(bn)
c['pass']=bool(metal>=.0048 and max(dist[handgs.index(g)] for g in tips)<=.0008 and dist.min()>=0 and not bad);r['pass']=c['pass'];r['grasp']=c;r['model']='gpt-6-astra';r['refinement']={'cost':sol.cost,'worst_handle_gap_m':float(dist.min()),'hot_clearance_m':float(metal),'tip_gaps_m':[float(dist[handgs.index(g)]) for g in tips],'forbidden':bad};(out/'report.json').write_text(json.dumps(r,indent=2));print(r['refinement'])
