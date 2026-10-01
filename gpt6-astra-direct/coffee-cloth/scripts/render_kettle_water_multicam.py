"""Six-camera physical replay with recorded reduced-model jet and volume ledger.

The jet is a visualization of the ballistic reduced liquid model, not CFD.
No physical state is modified except replaying the saved qpos in render data.
"""
import argparse,json,sys,shutil,hashlib
from pathlib import Path
import numpy as np,mujoco
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT.parent/'g1-cup-grasp/scripts'))
from recorder import Recorder
from water_visuals import _add_capsule,STREAM_RGBA
from brew_state import BrewState
from receiver_visuals import add_receiver_visuals
ap=argparse.ArgumentParser();ap.add_argument('--speedup',type=float,default=1.);ap.add_argument('--source',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--stride',type=int,default=2);a=ap.parse_args();assert a.stride>=1 and a.speedup>0
source=a.source.resolve();out=a.out.resolve();out.mkdir(exist_ok=False)
for f in [Path(__file__),ROOT/'scripts/water_visuals.py',ROOT/'scripts/receiver_visuals.py',ROOT/'scripts/kettle_liquid.py',ROOT.parent/'g1-cup-grasp/scripts/recorder.py']:shutil.copy2(f,out/f.name)
r=json.loads((source/'report.json').read_text());m=mujoco.MjModel.from_xml_path(r['scene']);d=mujoco.MjData(m);states=np.load(source/'trajectory.npz')['qpos'];rows=r['rows'];liquid=r['liquid_samples'];times=np.array([x['t'] for x in liquid]);fps=a.speedup/(16*m.opt.timestep*a.stride)
brew=BrewState(m,source);brew_rows=json.loads((source/'brew-samples.json').read_text()) if brew.enabled else [];brew_times=np.array([x['t'] for x in brew_rows]);label='agua/po/calor: modelos reduzidos' if brew.enabled else 'agua: modelo reduzido, sem aquecimento/cafe'
rec=Recorder(m,str(out/'attempt-six-cameras.mp4'),fps=fps,title=f'{source.name} | {label} | {a.speedup:g}x | gpt-6-astra');latest={};drain_rate=0.;original_update=rec.r.update_scene

def update(*args,**kwargs):
    original_update(*args,**kwargs)
    add_receiver_visuals(rec.r.scene,m,d,latest.get('receiver_ml',0),brew.grounds.filter_g if brew.enabled else 0.,drain_rate)
    if latest.get('flow_ml_s',0)<=.01:return
    origin=np.asarray(latest['spout_m']);velocity=np.asarray(latest['jet_velocity_m_s']);hit=latest.get('jet_hit_m')
    if hit is None:return
    end_z=hit[2];roots=np.roots([-4.905,velocity[2],origin[2]-end_z]);forward=[float(x.real) for x in roots if abs(x.imag)<1e-9 and x.real>0]
    if not forward:return
    time=min(forward);ts=np.linspace(0,time,9);points=origin[None,:]+ts[:,None]*velocity+np.outer(ts**2,[0,0,-4.905]);radius=.0012+.001*np.sqrt(min(latest['flow_ml_s']/30,1))
    for p,q in zip(points[:-1],points[1:]):_add_capsule(rec.r.scene,p,q,radius,STREAM_RGBA)
rec.r.update_scene=update
for i in range(0,len(states),a.stride):
    if brew.enabled and brew_rows:
        bi=max(0,int(np.searchsorted(brew_times,rows[i]['t'],side='right')-1));br=brew_rows[bi]
        for key in ['pot_g','spoon_g','filter_g','spilled_g']:setattr(brew.grounds,key,br['grounds'][key])
        brew.visuals()
    d.qpos[:]=states[i];d.time=rows[i]['t'];mujoco.mj_forward(m,d);idx=np.searchsorted(times,rows[i]['t'],side='right')-1;latest=liquid[idx] if idx>=0 else (liquid[0] if liquid else {});drain_rate=max(0,(liquid[idx]['receiver_ml']-liquid[idx-1]['receiver_ml'])/max(times[idx]-times[idx-1],1e-9)) if idx>0 else 0.;rec.caption=f"sim {rows[i]['t']-rows[0]['t']:.1f}s | {rows[i]['phase']} | jarra {latest.get('source_ml',r.get('initial_water_ml',r.get('water_state',{}).get('initial_ml',0))):.0f} ml | filtro {latest.get('filter_ml',0):.0f} | xicara {latest.get('receiver_ml',0):.0f} | derrame {latest.get('spilled_ml',0):.1f}";rec.caption+=(f" | po {br['grounds']['filter_g']:.2f}g | T {br['heater']['temperature_C']:.1f}C" if brew.enabled and brew_rows else '');rec.frame(d)
    if i%300==0:print(f'{i}/{len(states)}',flush=True)
rec.close(str(out/'timeline.json'));(out/'provenance.json').write_text(json.dumps({'model':'gpt-6-astra','source':str(source),'passed':r['pass'],'coffee_completed':False,'cameras':list(Recorder.GRID),'source_report_sha256':hashlib.sha256((source/'report.json').read_bytes()).hexdigest(),'overlay':'recorded ballistic jet, sample lag<=20ms; first frame may use first sample<=20ms ahead; horizontal cup surface from recorded volume, filtered drip from receiver increase; coffee tint illustrative, no extraction model or CFD; surface omitted when tilted cut crosses cup floor/rim','physical_replay':True,'fps':fps,'stride':a.stride,'speedup':a.speedup,'source_simulated_duration_s':rows[-1]['t']-rows[0]['t']},indent=2))
print(out/'attempt-six-cameras.mp4')
