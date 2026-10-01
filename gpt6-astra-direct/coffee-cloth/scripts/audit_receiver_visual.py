"""Replay a historical accepted water ledger to check visible receiver level."""
import sys,json,shutil
from pathlib import Path
import numpy as np,mujoco
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'g1-cup-grasp/scripts'))
from recorder import Recorder
from receiver_visuals import add_receiver_visuals
from kettle_liquid import KettleWater
source=Path('results/kettle-place-020');out=Path('results/receiver-visual-audit-002');out.mkdir(exist_ok=False)
for p in [Path(__file__),Path('scripts/receiver_visuals.py'),Path('scripts/water_visuals.py')]:shutil.copy2(p,out/p.name)
r=json.loads((source/'report.json').read_text());assert r['pass'];m=mujoco.MjModel.from_xml_path(r['scene']);d=mujoco.MjData(m);d.qpos[:]=np.load(source/'trajectory.npz')['qpos'][-1];mujoco.mj_forward(m,d);water=json.loads((source/'liquid-state.json').read_text());rec=Recorder(m,str(out/'one-frame.mp4'),fps=1,title='AUDITORIA VISUAL | agua registrada, sem cafe | gpt-6-astra');rec.caption=f"Volume registrado: {water['receiver_ml']:.3f} ml | cilindro visual, nao CFD";update=rec.r.update_scene;result={}
def wrapped(*args,**kwargs):
 update(*args,**kwargs);result.update(add_receiver_visuals(rec.r.scene,m,d,water['receiver_ml']) or {})
rec.r.update_scene=wrapped;rec.frame(d);rec.snapshot('head_camera',str(out/'head-camera.png'));rec.close(str(out/'timeline.json'));reference=KettleWater();expected=water['receiver_ml']*1e-6/(np.pi*reference.cup_radius**2);result.update({'model':'gpt-6-astra','source':str(source),'pass':abs(result.get('height_m',-1)-expected)<1e-10,'coffee_completed':False,'scope':'visual overlay consistency with historical recorded water volume only'});(out/'report.json').write_text(json.dumps(result,indent=2));print(result)
