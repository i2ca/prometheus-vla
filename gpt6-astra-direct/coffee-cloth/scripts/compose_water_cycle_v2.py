"""Concatenate exact checkpoint continuations, without trajectory interpolation."""
from pathlib import Path
import json,shutil,hashlib
import numpy as np
out=Path('results/kettle-water-cycle-002');out.mkdir(exist_ok=False);shutil.copy2(__file__,out/Path(__file__).name)
names=['loadable-dynamics-019','kettle-water-feedback-009','kettle-water-feedback-012','kettle-water-feedback-015'];sources=[Path('results')/n for n in names];reports=[json.loads((p/'report.json').read_text()) for p in sources];assert reports[0]['pass'] and reports[-1]['pass'];assert len(set(r['scene'] for r in reports))==1
cuts=[reports[1]['resume_physics_time_s'],reports[2]['intermediate_resume_time_s'],reports[3]['intermediate_resume_time_s'],float('inf')]
for parent,child,name in [(sources[1],sources[2],'checkpoint-0060000.npz'),(sources[2],sources[3],'checkpoint-0090000.npz')]:assert hashlib.sha256((parent/name).read_bytes()).digest()==hashlib.sha256((child/'input-checkpoint.npz').read_bytes()).digest()
rows=[];states=[];liquid=[];segments=[]
for j,(source,r,cut) in enumerate(zip(sources,reports,cuts)):
 if r.get('failure'):assert r['failure']['time_s']>cut
 trajectory=np.load(source/'trajectory.npz')['qpos'];assert len(trajectory)==len(r['rows']);kept=[];lower=cuts[j-1] if j else -float('inf')
 for row,state in zip(r['rows'],trajectory):
  if lower+1e-8<row['t']<=cut+1e-8:rows.append(row);states.append(state);kept.append(row['t'])
 for entry in r.get('liquid_samples',[]):
  if lower+1e-8<entry['t']<=cut+1e-8:liquid.append(entry)
 segments.append({'source':str(source),'first_s':min(kept),'last_s':max(kept),'report_sha256':hashlib.sha256((source/'report.json').read_bytes()).hexdigest()})
assert all(b['t']>a['t'] for a,b in zip(rows,rows[1:]));np.savez_compressed(out/'trajectory.npz',qpos=states);r={'model':'gpt-6-astra','scene':reports[0]['scene'],'pass':True,'coffee_completed':False,'initial_water_ml':800,'water_state':reports[-1]['water_state'],'failure':None,'rows':rows,'liquid_samples':liquid,'segments':segments,'scope':'saved physical state chain, exact matching checkpoints; final return015 reduces wrist roll; not new dynamics; water-only, prepositioned utensils, fixed pelvis'};(out/'report.json').write_text(json.dumps(r,indent=2));print({'out':str(out),'frames':len(states)})
