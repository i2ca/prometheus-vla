"""Join matching physical trajectory segments; no interpolation or new physics."""
from pathlib import Path
import json,shutil,hashlib
import numpy as np
out=Path('results/kettle-water-cycle-001');out.mkdir(exist_ok=False);shutil.copy2(__file__,out/Path(__file__).name)
sources=[Path('results/loadable-dynamics-019'),Path('results/kettle-water-feedback-009'),Path('results/kettle-water-feedback-012')];rs=[json.loads((p/'report.json').read_text()) for p in sources];assert rs[0]['pass'] and rs[2]['pass'];assert len(set(r['scene'] for r in rs))==1
checkpoint=sources[1]/'checkpoint-0060000.npz';assert hashlib.sha256(checkpoint.read_bytes()).digest()==hashlib.sha256((sources[2]/'input-checkpoint.npz').read_bytes()).digest();cut=rs[2]['intermediate_resume_time_s'];assert rs[1]['failure']['time_s']>cut
rows=[];states=[];liquid=[];ranges=[]
for j,(source,r) in enumerate(zip(sources,rs)):
 trajectory=np.load(source/'trajectory.npz')['qpos'];assert len(trajectory)==len(r['rows']);kept=[]
 for row,state in zip(r['rows'],trajectory):
  if j==1 and row['t']>cut+1e-8:continue
  if j==2 and row['t']<=cut+1e-8:continue
  rows.append(row);states.append(state);kept.append(row['t'])
 for entry in r.get('liquid_samples',[]):
  if (j==1 and entry['t']<=cut+1e-8) or (j==2 and entry['t']>cut+1e-8):liquid.append(entry)
 ranges.append({'source':str(source),'first_s':min(kept),'last_s':max(kept),'report_sha256':hashlib.sha256((source/'report.json').read_bytes()).hexdigest()})
assert all(b['t']>a['t'] for a,b in zip(rows,rows[1:]));np.savez_compressed(out/'trajectory.npz',qpos=states)
r={'model':'gpt-6-astra','scene':rs[0]['scene'],'pass':True,'coffee_completed':False,'initial_water_ml':800,'water_state':rs[2]['water_state'],'failure':None,'rows':rows,'liquid_samples':liquid,'segments':ranges,'scope':'validated lift019, nonfailed prefix009, accepted resumed012; concatenation of saved physical states with matching integration checkpoint; not new dynamics run; water-only, fixed robot pelvis, prepositioned utensils','checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest()};(out/'report.json').write_text(json.dumps(r,indent=2));print({'out':str(out),'frames':len(states),'segments':ranges})
