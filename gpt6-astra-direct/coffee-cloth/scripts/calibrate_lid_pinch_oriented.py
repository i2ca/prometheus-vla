"""Offline grasp geometry fitting only. Object translations are scratch search variables, not physics."""
import json,sys,shutil
from pathlib import Path
import mujoco,numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'g1-cup-grasp/scripts'))
from g1_sim import HAND_JOINTS
out=Path('results/lid-pinch-fit-005');out.mkdir(exist_ok=False);shutil.copy2(__file__,out/Path(__file__).name);r=json.loads(Path('results/prepared-layout-006/report.json').read_text());m=mujoco.MjModel.from_xml_path(r['scene']);d=mujoco.MjData(m);mujoco.mj_setState(m,d,np.load('results/prepared-layout-006/continuation.npz')['integration'],mujoco.mjtState.mjSTATE_INTEGRATION);base=d.qpos.copy();adr=m.jnt_qposadr[m.actuator_trnid[:,0]];base[adr]=np.load('results/lid-reach-012/candidates.npz')['qpos'][4,adr];d.qpos[:]=base;mujoco.mj_forward(m,d);ha=m.jnt_qposadr[[m.joint(n).id for n in HAND_JOINTS]];lid=m.body('tampa').id;pot=m.body('pote').id;palm=m.body('right_wrist_yaw_link').id;p0=d.xpos[palm].copy();R=d.xmat[palm].reshape(3,3).copy();la=m.jnt_qposadr[m.joint('tampa_livre').id];pa=m.jnt_qposadr[m.joint('pote_livre').id];lg=[g for g in range(m.ngeom) if m.geom_contype[g] and m.geom_bodyid[g]==lid];tips=[next(g for g in range(m.ngeom) if m.geom_contype[g] and m.body(int(m.geom_bodyid[g])).name=='right_hand_'+n+'_link') for n in ['thumb_2','index_1']];hands=[g for g in range(m.ngeom) if m.geom_contype[g] and m.body(int(m.geom_bodyid[g])).name.startswith('right_hand')];pg=[g for g in range(m.ngeom) if m.geom_contype[g] and m.geom_bodyid[g]==pot];selfpairs=[(g,h) for g in hands for h in hands if g<h and m.body(int(m.geom_bodyid[g])).name.split('_')[2]!=m.body(int(m.geom_bodyid[h])).name.split('_')[2]]
lid_all=lg.copy();lg=[g for g in lg if int(m.geom(g).name.split('_')[-1])>=58];lid_other=[g for g in lid_all if g not in lg];Rlid=d.xmat[lid].reshape(3,3).copy();Rpot=d.xmat[pot].reshape(3,3).copy()
# Keep middle at the collision-free open pose; optimize thumb and index only.
lo=np.r_[[-.04]*3,[-.5]*3,m.jnt_range[[m.joint(n).id for n in HAND_JOINTS[:5]],0]];hi=np.r_[[.04]*3,[.5]*3,m.jnt_range[[m.joint(n).id for n in HAND_JOINTS[:5]],1]];results=[];states=[]
def evaluate(x,detail=False):
 d.qpos[:]=base;d.qpos[ha[:5]]=x[6:];Q=Rotation.from_rotvec(x[3:6]).as_matrix();d.qpos[la:la+3]=p0+Q.T@(base[la:la+3]-p0-x[:3]);d.qpos[pa:pa+3]=p0+Q.T@(base[pa:pa+3]-p0-x[:3]);mujoco.mju_mat2Quat(d.qpos[la+3:la+7],(Q.T@Rlid).ravel());mujoco.mju_mat2Quat(d.qpos[pa+3:pa+7],(Q.T@Rpot).ravel());mujoco.mj_forward(m,d);distances=[];points=[]
 for g in tips:
  candidates=[]
  for h in lg:
   pts=np.zeros(6);gap=mujoco.mj_geomDistance(m,d,g,h,1,pts);candidates.append((gap,pts))
  gap,pts=min(candidates,key=lambda v:v[0]);distances.append(gap);points.append((pts[3:]-d.xpos[lid])@d.xmat[lid].reshape(3,3))
 badgaps=[min(0,mujoco.mj_geomDistance(m,d,g,h,.005,None)-.002) for g in hands for h in pg+lid_other];selfgaps=[min(0,mujoco.mj_geomDistance(m,d,g,h,.003,None)-.001) for g,h in selfpairs]
 if detail:return distances,points,min(badgaps),min(selfgaps)
 return np.r_[(np.array(distances)-.001)*1000,np.array([max(0,.028-p[2]) for p in points])*1000,np.array([max(0,np.linalg.norm(p[:2])-.015) for p in points])*1000,np.array(badgaps)*3000,np.array(selfgaps)*3000,(x[6:]-base[ha[:5]])*.05,x[:3]*.2,x[3:6]*.03]
for seed in range(7):
 x=np.r_[[0,0,-.012],[0,0,0],base[ha[:5]]];x[6]=.1+.1*seed;x[4]=(seed-3)*.1
 fit=least_squares(evaluate,np.clip(x,lo+1e-9,hi-1e-9),bounds=(lo,hi),max_nfev=500,diff_step=1e-4);dist,pts,pgap,sgap=evaluate(fit.x,True);ok=max(abs(np.array(dist)-.001))<.001 and pgap>-.0001 and sgap>-.0001 and all(p[2]>.0275 and np.linalg.norm(p[:2])<.0155 for p in pts);row={'seed':seed,'pass':bool(ok),'palm_shift_m':fit.x[:3].tolist(),'palm_goal_m':(p0+fit.x[:3]).tolist(),'palm_R':(Rotation.from_rotvec(fit.x[3:6]).as_matrix()@R).tolist(), 'orientation_adjustment_rad':fit.x[3:6].tolist(),'hand':d.qpos[ha].tolist(),'contact_gap_m':dist,'lid_contact_local':[p.tolist() for p in pts],'pot_margin_violation_m':pgap,'self_margin_violation_m':sgap,'cost':float(fit.cost)};results.append(row);print(row,flush=True)
(out/'report.json').write_text(json.dumps({'model':'gpt-6-astra','scene':r['scene'],'scope':'offline scratch geometry fit, not executed or validated grasp','results':results},indent=2))
