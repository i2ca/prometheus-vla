"""Offline joint arm/finger fitting; proposed initial layout, never a physical step."""
import argparse, json, sys, shutil, xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import mujoco
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'g1-cup-grasp/scripts'))
from g1_sim import G1Sim, ARM_JOINTS, HAND_JOINTS
from kinematics import ArmIK
from elbow_anatomy import ElbowAnatomy

ap=argparse.ArgumentParser();ap.add_argument('--shaft-center',type=float);ap.add_argument('--seed-index',type=int);ap.add_argument('--balanced-pinch',action='store_true');ap.add_argument('--seed-fit',type=Path);ap.add_argument('--two-contact',action='store_true');ap.add_argument('--face-sign',type=float,default=0.);ap.add_argument('--pregrasp-gap',type=float,default=.001);ap.add_argument('--third-link',choices=['middle_0','middle_1','index_0'],default='middle_1');ap.add_argument('--seeds',type=int,default=8);ap.add_argument('--source',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();out=a.out;out.mkdir(exist_ok=False)
shutil.copy2(__file__,out/Path(__file__).name);shutil.copy2('scripts/elbow_anatomy.py',out/'elbow_anatomy.py')
source=a.source;r=json.loads((source/'report.json').read_text())
s=G1Sim(r['scene']);m,d=s.m,s.d
ck=np.load(source/'continuation.npz');mujoco.mj_setState(m,d,ck['integration'],mujoco.mjtState.mjSTATE_INTEGRATION);mujoco.mj_forward(m,d)
source_q=d.qpos.copy();source_other=m.body('left_wrist_yaw_link').id;source_op=d.xpos[source_other].copy();source_oR=d.xmat[source_other].reshape(3,3).copy()
seed_state=np.load('results/spoon-right-whole-fit-001/candidates.npz')['qpos'][1]
for n in ['waist_yaw_joint','waist_roll_joint','waist_pitch_joint']+list(ARM_JOINTS)+list(HAND_JOINTS):
 qadr=m.jnt_qposadr[m.joint(n).id];d.qpos[qadr]=seed_state[qadr]
valid_fit=json.loads(Path('results/spoon-right-fit-010/report.json').read_text())['results'][0]
if a.third_link=='index_0':valid_fit['hand'][5:]=[1.5,1.7]
for n,v in zip(HAND_JOINTS,valid_fit['hand']):d.qpos[m.jnt_qposadr[m.joint(n).id]]=v
mujoco.mj_forward(m,d)
names=['waist_yaw_joint','waist_roll_joint','waist_pitch_joint']+list(ARM_JOINTS)+[n.replace('right_','left_') for n in ARM_JOINTS]
ik=ArmIK(s,names,palm='right_wrist_yaw_link');bounds=ik.bounds.copy()
bounds[0]=[-.7,.7];bounds[1:3]=np.deg2rad([[-12,12],[-12,12]])
anatomy=ElbowAnatomy(m);anatomy.bound_search(m,names,bounds)
for j in [7,14]:bounds[j:j+3]=np.deg2rad([[-60,60],[-45,45],[-30,30]])
hj=[m.joint(n).id for n in HAND_JOINTS];ha=m.jnt_qposadr[hj]
bounds=np.vstack([bounds,m.jnt_range[hj]])
old_data=mujoco.MjData(m);old_data.qpos[:]=seed_state;mujoco.mj_forward(m,old_data)
obj=m.body('scoop').id;pal=m.body('right_wrist_yaw_link').id;old_spoon=ET.parse('results/spoon-right-fit-010/hand-scene.xml').find('.//body[@name="scoop"]');old_pos=np.fromstring(old_spoon.get('pos'),sep=' ');old_quat=np.fromstring(old_spoon.get('quat'),sep=' ');old_R=Rotation.from_quat(old_quat[[1,2,3,0]]).as_matrix();new_R=d.xmat[obj].reshape(3,3);goal_p=d.xpos[obj]+new_R@old_R.T@(np.array(valid_fit['palm_goal_m'])-old_pos);goal_R=new_R@old_R.T@np.array(valid_fit['palm_R'])
def align(v):
 pp,rr=ik.fk(v);return np.r_[(pp-goal_p)*1000,Rotation.from_matrix(rr@goal_R.T).as_rotvec()*50,(ik.d.xpos[source_other]-source_op)*100,(ik.d.xmat[source_other].reshape(3,3)-source_oR).ravel()*10,anatomy.penalty(ik.d)]
pre=least_squares(align,np.clip(d.qpos[ik.qa],bounds[:17,0]+1e-8,bounds[:17,1]-1e-8),bounds=bounds[:17].T,max_nfev=300);d.qpos[ik.qa]=pre.x;mujoco.mj_forward(m,d)
print({'prealignment_position_error_m':float(np.linalg.norm(d.xpos[pal]-goal_p)),'prealignment_orientation_error_deg':float(np.rad2deg(np.linalg.norm(Rotation.from_matrix(d.xmat[pal].reshape(3,3)@goal_R.T).as_rotvec())))},flush=True)
initial=np.r_[d.qpos[ik.qa],d.qpos[ha]]
if a.seed_fit:
 seed_report=json.loads((a.seed_fit/'report.json').read_text());seed_row=seed_report['results'][a.seed_index] if a.seed_index is not None else next(x for x in seed_report['results'] if x['pass']);initial=np.r_[seed_row['q'],seed_row['hand']];d.qpos[ik.qa]=initial[:17];d.qpos[ha]=initial[17:];mujoco.mj_forward(m,d)
 # Map the seed hand relative to its spoon, not its historical world location.
 seed_model=mujoco.MjModel.from_xml_path(seed_report['scene']);seed_data=mujoco.MjData(seed_model)
 seed_data.qpos[:]=np.load(a.seed_fit/'candidates.npz')['qpos'][seed_row['seed']];mujoco.mj_forward(seed_model,seed_data)
 seed_obj=seed_model.body('scoop').id;seed_palm=seed_model.body('right_wrist_yaw_link').id
 alignment=d.xmat[obj].reshape(3,3)@seed_data.xmat[seed_obj].reshape(3,3).T
 seed_goal=d.xpos[obj]+alignment@(seed_data.xpos[seed_palm]-seed_data.xpos[seed_obj])
 seed_goal_R=alignment@seed_data.xmat[seed_palm].reshape(3,3)
 if a.shaft_center is not None:
  seed_goal+=d.xmat[obj].reshape(3,3)[:,0]*(a.shaft_center-np.mean(np.array(seed_row['object_contact_local'])[:2,0]))
 def seed_align(v):
  pp,rr=ik.fk(v)
  return np.r_[(pp-seed_goal)*1000,Rotation.from_matrix(rr@seed_goal_R.T).as_rotvec()*50,(ik.d.xpos[source_other]-source_op)*100,anatomy.penalty(ik.d)]
 aligned=least_squares(seed_align,np.clip(initial[:17],bounds[:17,0]+1e-8,bounds[:17,1]-1e-8),bounds=bounds[:17].T,max_nfev=200)
 initial[:17]=aligned.x;d.qpos[ik.qa]=aligned.x;mujoco.mj_forward(m,d)
base=d.qpos.copy()
other=m.body('left_wrist_yaw_link').id;op=source_op.copy()
spoon=m.body('scoop').id;table=m.geom('tampo').id
hg=[g for g in range(m.ngeom) if m.geom_contype[g] and m.body(int(m.geom_bodyid[g])).name.startswith(('right_hand','right_wrist'))]
og=[g for g in range(m.ngeom) if m.geom_contype[g] and m.geom_bodyid[g]==spoon]
tips=[next(g for g in hg if m.body(int(m.geom_bodyid[g])).name=='right_hand_'+n+'_link') for n in (['thumb_2','index_1'] if a.two_contact else ['thumb_2','index_1',a.third_link])]
prealign_gaps=[{'geom':m.geom(g).name,'body':m.body(int(m.geom_bodyid[g])).name,'gap_m':float(mujoco.mj_geomDistance(m,d,g,table,1,None))} for g in hg]
(out/'prealignment.json').write_text(json.dumps({'table_gaps':sorted(prealign_gaps,key=lambda x:x['gap_m']),'contacts':[(m.body(int(m.geom_bodyid[c.geom1])).name,m.body(int(m.geom_bodyid[c.geom2])).name,float(c.dist)) for c in d.contact if c.dist<0]},indent=2))
rng=np.random.default_rng(43);results=[];states=[]

def evaluate(x):
    d.qpos[ha]=x[17:];ik.fk(x[:17]);mujoco.mj_forward(m,ik.d);z=ik.d
    gaps=[];points=[];normals=[]
    for i,g in enumerate(tips):
        candidates=[]
        for h in og:
            low,high=(.020,.038) if i==2 else (.040,.059)
            if not low<m.geom_pos[h,0]<high:continue
            pts=np.zeros(6);gap=mujoco.mj_geomDistance(m,z,g,h,.3,pts)
            candidates.append((gap,pts))
        gap,pts=min(candidates,key=lambda v:v[0]);gaps.append(gap)
        points.append(z.xmat[spoon].reshape(3,3).T@(pts[3:]-z.xpos[spoon]))
        normal=(pts[3:]-pts[:3])*np.sign(gap);normals.append(normal/max(np.linalg.norm(normal),1e-9))
    return np.asarray(gaps),np.asarray(points),np.asarray(normals)

for seed in range(a.seeds):
    x=initial.copy()
    if seed:x[:10]+=rng.normal(0,.35,10);x[17:]+=rng.normal(0,.3,7)
    x=np.clip(x,bounds[:,0]+1e-8,bounds[:,1]-1e-8);avoid=set()
    for retry in range(8):
        pairs=sorted(avoid)
        def residual(v):
            gaps,points,normals=evaluate(v);z=ik.d
            barrier=[min(0,mujoco.mj_geomDistance(m,z,g,h,.02,None)-.001) for g,h in pairs]
            table_gaps=[min(0,mujoco.mj_geomDistance(m,z,g,table,.03,None)-.011) for g in hg]
            loc=[max(0,(.021 if i==2 else .041)-p[0])+max(0,p[0]-(.037 if i==2 else .058)) for i,p in enumerate(points)]
            return np.r_[(gaps-a.pregrasp_gap)*1500,np.array(loc)*1500,((points[:2,0]-a.shaft_center)*2000 if a.shaft_center is not None else np.zeros(2)),
                         max(0,normals[0]@normals[1]+(.95 if a.balanced_pinch else .65))*(30 if a.balanced_pinch else 10),
                         (np.cross(z.xmat[spoon].reshape(3,3)@(points[0]-points[1]),normals[0])*3000 if a.balanced_pinch else np.zeros(3)),
                         ((points[0,0]-points[1,0])*3000 if a.balanced_pinch else 0.),
                         ((normals[0]-z.xmat[spoon].reshape(3,3)[:,2]*a.face_sign)*5 if a.face_sign else np.zeros(3)),
                         ((normals[1]+z.xmat[spoon].reshape(3,3)[:,2]*a.face_sign)*5 if a.face_sign else np.zeros(3)),
                         (z.xpos[other]-op)*100,(z.xmat[other].reshape(3,3)-source_oR).ravel()*10,
                         np.array(barrier)*5000,np.array(table_gaps)*5000,
                         max(0,z.xpos[spoon,2]+.045-z.xpos[ik.body,2])*1000,
                         (v-initial)*(.5 if a.seed_fit else .03),v[7:10]*.1,anatomy.penalty(z)]
        fit=least_squares(residual,x,bounds=bounds.T,max_nfev=250,diff_step=1e-5)
        x=fit.x;gaps,points,normals=evaluate(x);bad=[]
        for c in ik.d.contact:
            if c.dist>=0:continue
            gs=[int(c.geom1),int(c.geom2)];bn=[m.body(int(m.geom_bodyid[g])).name for g in gs]
            if any(n.startswith(('right_','left_','torso','waist','pelvis','head')) for n in bn):
                bad.append(bn);avoid.add(tuple(sorted(gs)))
        table_gap=min(mujoco.mj_geomDistance(m,ik.d,g,table,1,None) for g in hg)
        line=ik.d.xmat[spoon].reshape(3,3)@(points[1]-points[0]);line/=max(np.linalg.norm(line),1e-9);cone=np.array([normals[0]@line,-normals[1]@line]);balanced_ok=abs(points[0,0]-points[1,0])<.001 and min(cone)>.8
        passed=(a.shaft_center is None or max(abs(points[:2,0]-a.shaft_center))<.002) and anatomy.valid(ik.d) and max(abs(gaps-a.pregrasp_gap))<.001 and table_gap>.010 and not bad and (balanced_ok if a.balanced_pinch else normals[0]@normals[1]<-.5)
        if passed:break
    row={'contact_line_normal_cosines':cone.tolist(),'elbow_flexion_deg':anatomy.angles(ik.d).tolist(),'seed':seed,'pass':bool(passed),'q':x[:17].tolist(),'hand':x[17:].tolist(),'palm_goal_m':ik.d.xpos[m.body('right_wrist_yaw_link').id].tolist(),'palm_R':ik.d.xmat[m.body('right_wrist_yaw_link').id].reshape(3,3).tolist(),'contact_gap_m':gaps.tolist(),'object_contact_local':points.tolist(),'normal_dot':float(normals[0]@normals[1]),'table_gap_m':table_gap,'contacts':bad,'cost':float(fit.cost)}
    results.append(row);states.append(ik.d.qpos.copy());print(row,flush=True)
    if passed:break
(out/'report.json').write_text(json.dumps({'shaft_center_m':a.shaft_center,'model':'gpt-6-astra','source':str(source),'scene':r['scene'],'seed_fit':str(a.seed_fit) if a.seed_fit else None,'seed_index':a.seed_index,'balanced_pinch':a.balanced_pinch,'balanced_rule':'same axial position within1mm and collinear pinch direction within36.9deg of each inward normal; modeled friction1 gives45deg cone; physical lift still required','two_contact':a.two_contact,'face_sign':a.face_sign,'third_link':a.third_link,'pregrasp_gap_m':a.pregrasp_gap,'scope':'offline arm and finger fit against actual source checkpoint; not motor execution','joint_names':names,'results':results},indent=2))
np.savez_compressed(out/'candidates.npz',qpos=states)
