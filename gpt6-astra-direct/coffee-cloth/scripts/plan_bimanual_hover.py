"""Backward IK continuation plus sampled collision checking for empty-hand approach."""
import argparse,json,sys,shutil
from pathlib import Path
import cv2,mujoco,numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'g1-cup-grasp/scripts'))
from g1_sim import G1Sim,ARM_JOINTS
from bimanual_ik import BimanualIK

ap=argparse.ArgumentParser();ap.add_argument('--scene',type=Path,required=True);ap.add_argument('--map',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
ap.add_argument('--forward-bulge',type=float,default=0.0);ap.add_argument('--descent-bulge',type=float,default=0.0);ap.add_argument('--angle',type=float,default=90);ap.add_argument('--pitch',type=float,default=.5);ap.add_argument('--roll',type=float,default=0)
a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False);shutil.copy2(__file__,a.out/Path(__file__).name)
sim=G1Sim(str(a.scene.resolve()));m,d=sim.m,sim.d
ra=list(ARM_JOINTS);la=[x.replace('right_','left_') for x in ra];b=BimanualIK(sim,['waist_yaw_joint'],ra,la)
q0=np.r_[0,[.8,-.9,0,1.2,0,0,0],[.8,.9,0,1.2,0,0,0]];d.qpos[b.qa]=q0;mujoco.mj_forward(m,d)
rows=json.loads(a.map.read_text());row=next(r for r in rows if r['center']==[.23,.1] and r['angle_deg']==a.angle and r['pitch']==a.pitch and r['roll']==a.roll)
q=np.array(row['q']);rod=lambda v:cv2.Rodrigues(np.array(v,float))[0]
rr=rod([0,0,np.radians(a.angle)])@rod([0,a.pitch,0])@rod([a.roll,0,0]);rl=rod([0,0,-np.radians(a.angle)])@rod([0,a.pitch,0])@rod([-a.roll,0,0])
def collisions(q):
    d.qpos[b.qa]=q;mujoco.mj_forward(m,d);pairs=set()
    for c in d.contact:
        if c.dist>=0:continue
        names=[m.body(int(m.geom_bodyid[g])).name for g in (c.geom1,c.geom2)]
        for rob,oth in [names,names[::-1]]:
            if rob.startswith(('left_','right_')) and (oth.startswith(('torso','waist','pelvis','head')) or oth in ['mesa','coador','chaleira','copo','pote','tampa','scoop','base_eletrica'] or (rob.startswith('left_') and oth.startswith('right_'))):pairs.add(tuple(sorted([rob,oth])))
    return sorted(pairs)
waypoints=[]
for z in np.linspace(.856,1.056,41):
    radius=.11+a.descent_bulge*np.sin(np.pi*(z-.856)/.20)
    pr=np.array([.23,.1-radius,z])-rr@np.array([.165,-.0046,-.0285]);pl=np.array([.23,.1+radius,z])-rl@np.array([.165,.0046,-.0285])
    pr[0]+=a.forward_bulge*np.sin(np.pi*(z-.856)/.20);pl[0]+=a.forward_bulge*np.sin(np.pi*(z-.856)/.20)
    q,e=b.solve(pr,rr,pl,rl,q,q0,iteracoes=400)
    waypoints.append({'z':z,'q':q.tolist(),'errors':e,'collisions':collisions(q)})
(a.out/'backward.json').write_text(json.dumps(waypoints,indent=2))
trials=[];found=None
for idx in [2,1,4,3]:
    for delta in [-.15,.15,-.3,.3,-.5,.5,0]:
        mid=q0+.65*(q-q0);mid[idx]+=delta
        ok=all(not collisions(qa+(qb-qa)*u) for qa,qb in [(q0,mid),(mid,q)] for u in np.linspace(0,1,121))
        trials.append({'joint':b.nomes[idx],'delta':delta,'clear':ok})
        if ok:
            found={'model':'gpt-6-astra','q_initial':q0.tolist(),'q_waypoint':mid.tolist(),'q_hover':q.tolist(),'joint_names':b.nomes,'trials':trials,'scope':'242 static approach samples,41 descent poses; not dynamic proof','scene_sha256':__import__('hashlib').sha256(a.scene.read_bytes()).hexdigest(),'angle':a.angle,'pitch':a.pitch,'roll':a.roll};break
    if found:break
descent_pass=all(max(w['errors']['erro_dir_mm'],w['errors']['erro_esq_mm'])<3 and max(w['errors']['orient_dir_deg'],w['errors']['orient_esq_deg'])<2 and not w['collisions'] for w in waypoints)
(a.out/'report.json').write_text(json.dumps({'approach_found':bool(found),'descent_pass':descent_pass,'trials':trials},indent=2))
if found:(a.out/'hover-plan.json').write_text(json.dumps(found,indent=2))
print('approach',bool(found),'descent',descent_pass,'colliding waypoints',sum(bool(w['collisions']) for w in waypoints))
