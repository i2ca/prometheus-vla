"""Sampled rigid-grasp translation feasibility; offline only, not a dynamic proof."""
import argparse, json, sys, shutil, hashlib
from pathlib import Path
import numpy as np
import mujoco
import cv2
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'g1-cup-grasp/scripts'))
from g1_sim import G1Sim, ARM_JOINTS, HAND_JOINTS
from bimanual_ik import BimanualIK

ap=argparse.ArgumentParser()
ap.add_argument('--trial',type=Path,required=True)
ap.add_argument('--out',type=Path,required=True)
ap.add_argument('--offset',nargs=3,type=float,required=True)
ap.add_argument('--hand-offset',nargs=3,type=float,default=[0,0,0])
ap.add_argument('--thumb-proximal',type=float,default=None)
ap.add_argument('--yaw-deg',type=float,default=0)
ap.add_argument('--pitch-deg',type=float,default=0)
a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False)
shutil.copy2(__file__,a.out/Path(__file__).name)
params=json.loads((a.trial/'parameters.json').read_text())
rows=json.loads((a.trial/'trajetoria.json').read_text())
row=[r for r in rows if r['fase']=='segura'][-1]
sim=G1Sim(params['cena']);m,d=sim.m,sim.d
state_path=a.trial/'held-state.npz'
if state_path.exists():
    state=np.load(state_path)
    d.qpos[:]=state['qpos'];d.qvel[:]=state['qvel']
ra=list(ARM_JOINTS);la=[n.replace('right_','left_') for n in ra]
b=BimanualIK(sim,['waist_yaw_joint'],ra,la)
q=np.array(row['joint_targets']);qref=q.copy()
for side in ('right','left'):
    names=[n.replace('right_',side+'_') for n in HAND_JOINTS]
    for n,v in zip(names,row['hand_joint_positions'][side]):d.qpos[m.jnt_qposadr[m.joint(n).id]]=v
    if a.thumb_proximal is not None:
        d.qpos[m.jnt_qposadr[m.joint(side+'_hand_thumb_1_joint').id]]=(-1 if side=='right' else 1)*a.thumb_proximal
objqa=m.jnt_qposadr[m.joint(params['objeto']+'_livre').id]
d.qpos[objqa:objqa+3]=row['obj'];d.qpos[objqa+3:objqa+7]=row['obj_quaternion']
pR,RR,pL,RL=b.fk(q);offset=np.array(a.offset);samples=[]
pR+=np.array(a.hand_offset);pL+=np.array(a.hand_offset)
mujoco.mj_forward(m,d)
object_center=np.array(row['obj']);object_rotation=d.xmat[m.body(params['objeto']).id].reshape(3,3).copy()
rotation_end=cv2.Rodrigues(np.array([0.,0.,np.deg2rad(a.yaw_deg)]))[0] @ cv2.Rodrigues(np.array([0.,np.deg2rad(a.pitch_deg),0.]))[0]
rotation_vector=cv2.Rodrigues(rotation_end)[0].ravel()
for u in np.linspace(0,1,81):
    delta=u*offset
    rot=cv2.Rodrigues(rotation_vector*u)[0]
    tr=object_center+delta+rot@(pR-object_center)
    tl=object_center+delta+rot@(pL-object_center)
    q,error=b.solve(tr,rot@RR,tl,rot@RL,q,qref,iteracoes=200)
    d.qpos[b.qa]=q;d.qpos[objqa:objqa+3]=np.array(row['obj'])+delta
    quat=np.zeros(4);mujoco.mju_mat2Quat(quat,(rot@object_rotation).ravel());d.qpos[objqa+3:objqa+7]=quat
    mujoco.mj_forward(m,d);bad=set()
    for ct in d.contact:
        if ct.dist>=0:continue
        names=[m.body(int(m.geom_bodyid[g])).name for g in (ct.geom1,ct.geom2)]
        for rob,other in (names,names[::-1]):
            if rob.startswith(('right_','left_')) and (
                other.startswith(('torso','pelvis','waist','head')) or
                other in ('mesa','coador','copo','pote','tampa','scoop','base_eletrica') or
                (rob.startswith('right_') and other.startswith('left_'))):bad.add(tuple(sorted(names)))
            if rob != other and any(rob.startswith(s+'_hand') and other.startswith(s+'_hand') for s in ('right','left')):
                bad.add(tuple(sorted(names)))
            if rob==params['objeto'] and not other.startswith(('right_hand','left_hand','right_wrist','left_wrist')):
                bad.add(tuple(sorted(names)))
    samples.append({'u':float(u),'q':q.tolist(),'errors':error,'forbidden_contacts':sorted(bad)})
passed=all(not r['forbidden_contacts'] and max(r['errors']['erro_dir_mm'],r['errors']['erro_esq_mm'])<3
           and max(r['errors']['orient_dir_deg'],r['errors']['orient_esq_deg'])<2 for r in samples)
report={'model':'gpt-6-astra','source_trial':str(a.trial),'offset_m':a.offset,
        'hand_offset_m':a.hand_offset,
        'thumb_proximal_rad':a.thumb_proximal,
        'yaw_deg':a.yaw_deg,'pitch_deg':a.pitch_deg,
        'complete_start_state':state_path.exists(),
        'scene_sha256':hashlib.sha256(Path(params['cena']).read_bytes()).hexdigest(),
        'scope':'81 static poses using measured starting grasp and commanded arm preload; rigid object placement only in offline planner. No dynamics, not an execution acceptance.',
        'passed':passed,'samples':samples}
(a.out/'report.json').write_text(json.dumps(report,indent=2))
print(json.dumps({'passed':passed,'colliding_samples':sum(bool(s['forbidden_contacts']) for s in samples),
                  'max_position_error_mm':max(max(s['errors']['erro_dir_mm'],s['errors']['erro_esq_mm']) for s in samples)},indent=2))
