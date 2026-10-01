"""Full-robot endpoint checks for detached-hand handle candidates."""
import argparse,json,sys,shutil
from pathlib import Path
import numpy as np,mujoco
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'g1-cup-grasp/scripts'))
from g1_sim import G1Sim,ARM_JOINTS
from kinematics import ArmIK
ap=argparse.ArgumentParser();ap.add_argument('--scene',type=Path,required=True);ap.add_argument('--candidates',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
a.out.mkdir(parents=True,exist_ok=False);shutil.copy2(__file__,a.out/Path(__file__).name)
sim=G1Sim(str(a.scene.resolve()));m,d=sim.m,sim.d
for side,roll in [('right',-.9),('left',.9)]:
    for name,q in zip([x.replace('right_',side+'_') for x in ARM_JOINTS],[.8,roll,0,1.2,0,0,0]):d.qpos[m.jnt_qposadr[m.joint(name).id]]=q
mujoco.mj_forward(m,d);initial=d.qpos.copy();bid=m.body('chaleira').id;cp=d.xpos[bid].copy();CR=d.xmat[bid].reshape(3,3).copy()
source=json.loads(a.candidates.read_text())['candidates'];results=[];rng=np.random.default_rng(19)
for side in ['right','left']:
    names=['waist_yaw_joint']+[x.replace('right_',side+'_') for x in ARM_JOINTS]
    ik=ArmIK(sim,names,palm=side+'_wrist_yaw_link');q0=initial[ik.qa];F=np.diag([1,-1,1]) if side=='left' else np.eye(3)
    for i,c in enumerate(source):
        if not c['static_fit_pass']:continue
        d.qpos[:]=initial
        handnames=[x.replace('right_',side+'_') for x in c['hand_joint_names']]
        hq=np.array(c['hand_q'])*([1,-1,-1,-1,-1,-1,-1] if side=='left' else 1)
        for name,q in zip(handnames,hq):d.qpos[m.jnt_qposadr[m.joint(name).id]]=q
        p=cp+CR@F@np.array(c['palm_position_kettle_frame_m']);R=CR@F@np.array(c['palm_rotation_kettle_frame'])@F
        best=None
        for seedid in range(3):
            seed=q0.copy() if seedid==0 else np.clip(q0+rng.normal(0,.6,len(q0)),ik.bounds[:,0],ik.bounds[:,1])
            q,e=ik.solve(p,R,seed,reference=q0,iterations=400);d.qpos[ik.qa]=q;mujoco.mj_forward(m,d)
            bad=set()
            for ct in d.contact:
                if ct.dist>=0:continue
                ns=[m.body(int(m.geom_bodyid[g])).name for g in (ct.geom1,ct.geom2)];gn=[m.geom(g).name or '' for g in (ct.geom1,ct.geom2)]
                for rob,other,gother in [(ns[0],ns[1],gn[1]),(ns[1],ns[0],gn[0])]:
                    if not rob.startswith(('right_','left_')):continue
                    if other.startswith(('torso','waist','pelvis','head')) or other in ['mesa','coador','copo','pote','tampa','scoop','base_eletrica'] or gother=='chaleira_hot_body' or (rob.startswith('right_') and other.startswith('left_')) or (rob!=other and any(rob.startswith(s+'_hand') and other.startswith(s+'_hand') for s in ['right','left'])):
                        bad.add(tuple(sorted(ns+[gother] if gother=='chaleira_hot_body' else ns)))
            passed=not bad and e['position_error_m']<.0015 and e['orientation_error_rad']<np.deg2rad(1)
            row={'side':side,'candidate_index':i,'seed_id':seedid,'joint_names':names,'q':q.tolist(),'hand_joint_names':handnames,'hand_q':hq.tolist(),'palm_target_m':p.tolist(),'palm_rotation':R.tolist(),'errors':e,'forbidden_contacts':sorted(bad),'endpoint_pass':bool(passed),'score':e['position_error_m']*1000+np.rad2deg(e['orientation_error_rad'])*3+len(bad)*10}
            if best is None or (not passed,row['score'])<(not best['endpoint_pass'],best['score']):best=row
        results.append(best)
    print(side,'passes',sum(r['endpoint_pass'] for r in results),flush=True)
results.sort(key=lambda r:(not r['endpoint_pass'],r['score']))
(a.out/'report.json').write_text(json.dumps({'model':'gpt-6-astra','scene':str(a.scene.resolve()),'initial_qpos':initial.tolist(),'scope':'Bounded full-robot endpoints; approach and force not verified','results':results},indent=2))
print(json.dumps(results[0],indent=2))
