"""Offline pose search; checks position AND orientation, never claims dynamic success."""
import argparse
import json
from pathlib import Path
import shutil
import sys

import cv2
import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'g1-cup-grasp/scripts'))
from g1_sim import G1Sim, ARM_JOINTS
from bimanual_ik import BimanualIK


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--scene', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, a.out/Path(__file__).name)
    sim = G1Sim(str(a.scene.resolve())); m,d=sim.m,sim.d
    right=list(ARM_JOINTS); left=[x.replace('right_', 'left_') for x in right]
    for names,pose in [(right,[.8,-.9,0,1.2,0,0,0]),(left,[.8,.9,0,1.2,0,0,0])]:
        for n,v in zip(names,pose): d.qpos[m.jnt_qposadr[m.joint(n).id]]=v
    mujoco.mj_forward(m,d)
    bik=BimanualIK(sim,['waist_yaw_joint'],right,left)
    initial=d.qpos.copy(); q0=sim.q(bik.nomes); rng=np.random.default_rng(20260919)
    offsets=[]
    for side in ['right','left']:
        palm=m.body(side+'_wrist_yaw_link').id; hand=m.body(side+'_hand_middle_1_link').id
        offsets.append(d.xmat[palm].reshape(3,3).T@(d.xpos[hand]-d.xpos[palm]))
    rod=lambda v:cv2.Rodrigues(np.array(v,dtype=float))[0]
    rows=[]
    for center in [[.23,.10],[.26,0.0]]:
        for angle in [60,90,120]:
            for pitch in [-.5,0,.5]:
                for roll in [-.6,0,.6]:
                    ang=np.radians(angle)
                    RR=rod([0,0,ang])@rod([0,pitch,0])@rod([roll,0,0])
                    # Bilateral mirror: same pitch, opposite roll.
                    RL=rod([0,0,-ang])@rod([0,pitch,0])@rod([-roll,0,0])
                    best=None
                    for seed_id in range(3):
                        d.qpos[:]=initial; mujoco.mj_forward(m,d)
                        seed=q0 if seed_id==0 else np.clip(q0+rng.normal(0,.7,len(q0)),bik.bounds[:,0],bik.bounds[:,1])
                        # Hands oppose one another along the transverse axis;
                        # yaw is free to choose a reachable direction for fingers.
                        pr=np.r_[center[0],center[1]-.11,.856]-RR@offsets[0]
                        pl=np.r_[center[0],center[1]+.11,.856]-RL@offsets[1]
                        q,err=bik.solve(pr,RR,pl,RL,seed,q0,iteracoes=400)
                        score=max(err['erro_dir_mm'],err['erro_esq_mm'])+3*max(err['orient_dir_deg'],err['orient_esq_deg'])
                        if best is None or score<best['score']:
                            best={'score':score,'q':q.tolist(),'errors':err,'seed_id':seed_id}
                    d.qpos[:]=initial; d.qpos[bik.qa]=best['q']; mujoco.mj_forward(m,d)
                    pairs=set()
                    for c in d.contact:
                        b1,b2=[m.body(int(m.geom_bodyid[g])).name for g in (c.geom1,c.geom2)]
                        for robot,other in [(b1,b2),(b2,b1)]:
                            if robot.startswith(('right_','left_')) and (other.startswith(('torso','pelvis','waist','head')) or other in ['mesa','coador','copo','pote','scoop','tampa','base_eletrica'] or (robot.startswith('right_') and other.startswith('left_'))):
                                if c.dist < 0: pairs.add(tuple(sorted([robot,other])))
                    best.update(center=center,angle_deg=angle,pitch=pitch,roll=roll,forbidden_pairs=sorted(pairs))
                    e=best['errors']; best['endpoint_pass']=max(e['erro_dir_mm'],e['erro_esq_mm'])<3 and max(e['orient_dir_deg'],e['orient_esq_deg'])<2 and not pairs
                    rows.append(best)
                    (a.out/'poses.json').write_text(json.dumps(rows,indent=2))
    good=[r for r in rows if r['endpoint_pass']]
    print(json.dumps({'model':'gpt-6-astra','count':len(rows),'passing':len(good),'best':min(rows,key=lambda r:r['score']),'limitations':'Endpoint-only; displaced layouts do not move scene props in this map, so scene contacts require separate verification.'}))


if __name__=='__main__':main()
