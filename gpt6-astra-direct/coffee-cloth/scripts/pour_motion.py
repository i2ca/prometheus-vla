"""Free-contact empty-vessel tilt diagnostic. No weld/teleport. gpt-6-astra."""
import argparse,json,subprocess,shutil
import numpy as np
import cv2
from replay_common import *

ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--target',nargs=3,type=float,default=[.44,-.15,.94]);ap.add_argument('--axis',nargs=3,type=float,default=[1,0,0]);ap.add_argument('--tilt',type=float,default=60);ap.add_argument('--video',action='store_true');ap.add_argument('--nullspace',type=float,default=0.0);ap.add_argument('--scene',type=Path);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False)
shutil.copy2(__file__,a.out/'controller.py');shutil.copy2(Path(__file__).with_name('replay_common.py'),a.out/'replay_common.py')
(a.out/'parameters.json').write_text(json.dumps({'target':a.target,'axis':a.axis,'tilt':a.tilt,'model':'gpt-6-astra','nullspace':a.nullspace},indent=2))
sim=replay_grasp(a.scene.resolve() if a.scene else None);m,d=sim.m,sim.d;ik=ArmIK(sim);ik.nullspace_gain=a.nullspace;q=sim.q(ARM_JOINTS);qref=q.copy();p0,R0=ik.fk(q)
c0=d.xpos[sim.cup_body].copy();C0=d.xmat[sim.cup_body].reshape(3,3).copy();local=R0.T@(c0-p0);localR=R0.T@C0
axis=np.array(a.axis,float);axis/=np.linalg.norm(axis);rv=axis*np.radians(a.tilt)
rows=[];frames=[];renderer=mujoco.Renderer(m,360,640) if a.video else None;start=d.time
for i in range(300):
 t=(i+1)/30
 if t<=4:
  u=t/4;u=u*u*(3-2*u);phase='tilt'
 elif t<=6:u=1.;phase='hold'
 else:
  u=1-(t-6)/4;u=u*u*(3-2*u);phase='return'
 c=c0+(np.array(a.target)-c0)*u;C=cv2.Rodrigues(rv*u)[0]@C0;R=C@localR.T;target=c-R@local
 q,err=ik.solve(target,R,q,reference=qref,iterations=200,max_step=np.radians(120)/30)
 sim.set_targets(ARM_JOINTS,q);advance_to(sim,start+t)
 palm=d.xpos[m.body(PALM).id];pR=d.xmat[m.body(PALM).id].reshape(3,3);cup=d.xpos[sim.cup_body].copy()
 slip=np.linalg.norm(pR.T@(cup-palm)-local)
 bad=[]
 for ct in d.contact:
  b1=m.body(int(m.geom_bodyid[ct.geom1])).name if ct.geom1>=0 else 'cloth_filter';b2=m.body(int(m.geom_bodyid[ct.geom2])).name if ct.geom2>=0 else 'cloth_filter'
  if (b1.startswith('right_') and (b2 in ('mesa','torso_link','pelvis','waist_roll_link','waist_yaw_link','coffee_stand','filter_ring','receiver','cloth_filter') or b2.startswith('left_'))) or (b2.startswith('right_') and (b1 in ('mesa','torso_link','pelvis','waist_roll_link','waist_yaw_link','coffee_stand','filter_ring','receiver','cloth_filter') or b1.startswith('left_'))):bad.append([b1,b2])
 rows.append({'t':t,'phase':phase,'cup':cup.tolist(),'tilt_deg':float(np.degrees(np.arccos(np.clip(d.xmat[sim.cup_body].reshape(3,3)[2,2],-1,1)))),'slip_m':float(slip),'contact_links':sorted(sim.finger_contacts()),'bad_contacts':bad,**err})
 if renderer:renderer.update_scene(d,camera='coffee_closeup' if a.scene else 'side_view');frames.append(renderer.render().copy())
report={'model':'gpt-6-astra','scope':'empty-cup diagnostic, does not brew coffee','max_slip_m':max(r['slip_m'] for r in rows),'hold_tilt_deg':float(np.mean([r['tilt_deg'] for r in rows if r['phase']=='hold'])),'bad_contact_frames':sum(bool(r['bad_contacts']) for r in rows),'max_ik_error_m':max(r['position_error_m'] for r in rows),'warnings':sim.warnings(),'rows':rows}
(a.out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='rows'}))
if renderer:
 renderer.close();pr=subprocess.Popen(['ffmpeg','-n','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','640x360','-r','30','-i','-','-c:v','libx264','-pix_fmt','yuv420p',str(a.out/'motion.mp4')],stdin=subprocess.PIPE)
 for f in frames:pr.stdin.write(f.tobytes())
 pr.stdin.close()
 if pr.wait():raise RuntimeError('video encoding failed')
 from PIL import Image
 Image.fromarray(frames[150]).save(a.out/'hold.png')
