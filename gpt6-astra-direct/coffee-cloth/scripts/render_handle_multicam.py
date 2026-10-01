"""Replay saved physical states in the established six-camera panel."""
import json,sys,shutil,argparse
from pathlib import Path
import numpy as np,mujoco
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT.parent/'g1-cup-grasp/scripts'))
from recorder import Recorder
ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
source=a.source.resolve();out=a.out.resolve()
out.mkdir(exist_ok=False)
shutil.copy2(__file__,out/Path(__file__).name)
r=json.loads((source/'report.json').read_text())
m=mujoco.MjModel.from_xml_path(r['scene']);d=mujoco.MjData(m)
states=np.load(source/'trajectory.npz')['qpos'];rows=r['rows']
# States were recorded every int(.033/dt) physics steps, regardless of slow factor.
stride=max(1,int(.033/m.opt.timestep));fps=1/(stride*m.opt.timestep)
rec=Recorder(m,str(out/'attempt-six-cameras.mp4'),fps=fps,title=f'{source.name} | etapa limitada | replay fisico | gpt-6-astra')
last=None
for i,state in enumerate(states):
 d.qpos[:]=state;mujoco.mj_forward(m,d)
 phase=rows[i]['phase']
 if phase!=last:rec.event(phase);last=phase
 rec.frame(d)
 if i%150==0:print(f'{i}/{len(states)}',flush=True)
rec.close(str(out/'timeline.json'))
(out/'provenance.json').write_text(json.dumps({'model':'gpt-6-astra','source':str(source),'frames':len(states),'fps':fps,'simulation_rerun':False,'passed':r.get('pass',r.get('passed',False)),'reason':r.get('failure',r.get('first_forbidden_contact')),'cameras':list(Recorder.GRID)},indent=2))
print(out/'attempt-six-cameras.mp4',flush=True)
