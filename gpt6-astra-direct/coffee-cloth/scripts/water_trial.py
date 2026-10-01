"""Loaded vessel + conservative water surrogate. Not complete coffee. gpt-6-astra."""
import argparse,json,shutil,subprocess
from collections import Counter
import numpy as np
import cv2
from replay_common import *
from liquid import WaterTransfer

ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--scene',type=Path,required=True);ap.add_argument('--target',type=float,nargs=3,default=[.38,-.035,.94]);ap.add_argument('--tilt',type=float,default=90);ap.add_argument('--hold',type=float,default=20);ap.add_argument('--video',action='store_true');ap.add_argument('--staged',action='store_true');ap.add_argument('--hand-kp',type=float);ap.add_argument('--visuals',action='store_true');args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=False)
for name in ('water_trial.py','replay_common.py','liquid.py'):shutil.copy2(Path(__file__).with_name(name),args.out/name)
shutil.copy2(args.scene,args.out/'scene.xml')
if args.visuals:
    from water_visuals import add_water_visuals
    shutil.copy2(Path(__file__).with_name('water_visuals.py'),args.out/'water_visuals.py')
(args.out/'parameters.json').write_text(json.dumps({'model':'gpt-6-astra','scene':str(args.scene.resolve()),'target':args.target,'tilt':args.tilt,'hold':args.hold,'staged':args.staged,'hand_kp':args.hand_kp,'visuals':args.visuals,'visuals_model':'claude-fable-5' if args.visuals else None,'initial_water_ml':200,'load':'fixed 200g ballast throughout; no sloshing or drain inertia update'},indent=2))
water=WaterTransfer();forbidden=Counter();object_contacts=Counter();source_contact_examples={};forbidden_steps=0;object_steps=0;rows=[];phase='grasp';last_time=0.;latest={};frames=[];renderer=None

def observe(sim):
    global forbidden_steps, object_steps
    m,d=sim.m,sim.d
    robot_pairs=set();source_pairs=set()
    for contact in d.contact:
        bodies=[]
        for geom in (contact.geom1,contact.geom2):
            bodies.append(m.body(int(m.geom_bodyid[geom])).name if geom>=0 else 'cloth_filter')
        for k in (0,1):
            if bodies[k].startswith('right_') and (bodies[1-k] in {'mesa','torso_link','pelvis','waist_yaw_link','waist_roll_link','filter_ring','coffee_stand','receiver','cloth_filter'} or bodies[1-k].startswith('left_')):
                robot_pairs.add(tuple(sorted(bodies)))
        if phase != 'grasp' and 'copo' in bodies:
            other=bodies[1] if bodies[0]=='copo' else bodies[0]
            if not other.startswith('right_hand') and other!='right_wrist_yaw_link':
                source_pairs.add(tuple(sorted(bodies)))
                key='|'.join(str(g) for g in (contact.geom1,contact.geom2))
                if key not in source_contact_examples:
                    source_contact_examples[key]={'phase':phase,'geoms':[m.geom(g).name if g>=0 else 'flex' for g in (contact.geom1,contact.geom2)],'distance_m':float(contact.dist),'point':contact.pos.tolist(),'cup_position':d.xpos[sim.cup_body].tolist()}
    forbidden_steps+=bool(robot_pairs);object_steps+=bool(source_pairs)
    forbidden.update((phase,*pair) for pair in robot_pairs)
    object_contacts.update((phase,*pair) for pair in source_pairs)

def transfer(sim):
    global last_time,latest
    d,m=sim.d,sim.m
    dt=d.time-last_time;last_time=d.time
    latest=water.step(dt,d.xpos[sim.cup_body],d.xmat[sim.cup_body].reshape(3,3),d.xpos[m.body('filter_ring').id],d.xpos[m.body('receiver').id])
    links=sorted(sim.finger_contacts())
    rows.append({'t':d.time,'phase':phase,**latest,
                 'cup_position':d.xpos[sim.cup_body].tolist(),
                 'cup_rotation':d.xmat[sim.cup_body].reshape(3,3).tolist(),
                 'receiver_position':d.xpos[m.body('receiver').id].tolist(),
                 'receiver_rotation':d.xmat[m.body('receiver').id].reshape(3,3).tolist(),
                 'finger_links':links,'distinct_fingers':len({link.split('_')[0] for link in links})})

sim=replay_grasp(args.scene.resolve(),observe,transfer,hand_kp=args.hand_kp);m,d=sim.m,sim.d
ik=ArmIK(sim);ik.nullspace_gain=0;q=sim.q(ARM_JOINTS);ref=q.copy();p0,R0=ik.fk(q);c0=d.xpos[sim.cup_body].copy();C0=d.xmat[sim.cup_body].reshape(3,3).copy();offset=R0.T@(c0-p0);localR=R0.T@C0
receiver0=d.xpos[m.body('receiver').id].copy();start=d.time;errs=[];slips=[]
if args.video:renderer=mujoco.Renderer(m,360,640)
for i in range(round(((12 if args.staged else 8)+args.hold)*30)):
    t=(i+1)/30
    if t<=4:u=t/4;u=u*u*(3-2*u);phase='tilt'
    elif t<=4+args.hold:u=1.;phase='pour'
    else:u=1-(t-4-args.hold)/4;u=u*u*(3-2*u);phase='return'
    angle=args.tilt*u
    if args.staged:
        smooth=lambda x: x*x*(3-2*x)
        if t<=4:
            u=smooth(t/4);angle=35*u;phase='align'
        elif t<=6:
            u=1.;angle=35+(args.tilt-35)*smooth((t-4)/2);phase='tilt'
        elif t<=6+args.hold:
            u=1.;angle=args.tilt;phase='pour'
        elif t<=8+args.hold:
            u=1.;angle=args.tilt-(args.tilt-35)*smooth((t-6-args.hold)/2);phase='untilt'
        else:
            u=1-smooth((t-8-args.hold)/4);angle=35*u;phase='return'
    c=c0+(np.array(args.target)-c0)*u;C=cv2.Rodrigues(np.array([-np.radians(angle),0.,0.]))[0]@C0;R=C@localR.T
    q,err=ik.solve(c-R@offset,R,q,reference=ref,iterations=200,max_step=np.radians(120)/30);errs.append(err)
    sim.set_targets(ARM_JOINTS,q);advance_to(sim,start+t,observe);transfer(sim)
    pr=d.xmat[m.body(PALM).id].reshape(3,3);pp=d.xpos[m.body(PALM).id]
    slips.append(float(np.linalg.norm(pr.T@(d.xpos[sim.cup_body]-pp)-offset)))
    if renderer and i%3==0:
        renderer.update_scene(d,camera='coffee_closeup')
        if args.visuals: add_water_visuals(renderer,sim,water,latest)
        img=renderer.render().copy()
        cv2.putText(img,'WATER SURROGATE / fixed 200g load',(10,22),cv2.FONT_HERSHEY_SIMPLEX,.46,(255,255,255),1)
        cv2.putText(img,f"cup {water.receiver_ml:.1f}ml  spill {water.spilled_ml:.1f}ml  source {water.source_ml:.1f}ml",(10,345),cv2.FONT_HERSHEY_SIMPLEX,.44,(255,255,255),1)
        frames.append(img)
phase='drain'
for i in range(900):
    advance_to(sim,d.time+1/30,observe);transfer(sim)
report={'model':'gpt-6-astra','scope':'water surrogate plus fixed mass load, not coffee or CFD','source_ml':water.source_ml,'filter_ml':water.filter_ml,'receiver_ml':water.receiver_ml,'spilled_ml':water.spilled_ml,'captured_ml':water.captured_ml,'max_ik_error_m':max(e['position_error_m'] for e in errs),'max_slip_m':max(slips),'forbidden_contact_substeps':forbidden_steps,'source_object_contact_substeps':object_steps,'source_contact_examples':list(source_contact_examples.values()),'source_object_contacts':[{'phase':k[0],'bodies':k[1:],'pair_substeps':v} for k,v in object_contacts.items()],'minimum_fingers_after_grasp':min(r['distinct_fingers'] for r in rows if r['phase']!='grasp'),'forbidden_contacts':[{'phase':k[0],'bodies':k[1:],'count':v} for k,v in forbidden.items()],'receiver_displacement_m':float(np.linalg.norm(d.xpos[m.body('receiver').id]-receiver0)),'warnings':sim.warnings(),'max_balance_error_ml':max(abs(r['mass_balance_error_ml']) for r in rows)}
report['accepted_water_diagnostic']=bool(report['receiver_ml']>=198 and report['spilled_ml']<=1 and not forbidden_steps and not object_steps and report['minimum_fingers_after_grasp']>=2 and report['max_ik_error_m']<=.003 and report['max_slip_m']<=.01 and report['receiver_displacement_m']<=.003 and report['warnings']==0)
report['limitations']=['fixed ballast remains in source after draining; no water weight on cloth/receiver','vertical jet and constant drainage surrogate; no CFD or temperature','no coffee grounds, filter mounting, rinse disposal or serving']
(args.out/'report.json').write_text(json.dumps(report,indent=2));(args.out/'transfer.json').write_text(json.dumps(rows));print(json.dumps(report))
if renderer:
    renderer.close();proc=subprocess.Popen(['ffmpeg','-n','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','640x360','-r','10','-i','-','-c:v','libx264','-pix_fmt','yuv420p',str(args.out/'water.mp4')],stdin=subprocess.PIPE)
    for frame in frames:proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    if proc.wait():raise RuntimeError('ffmpeg failed')
