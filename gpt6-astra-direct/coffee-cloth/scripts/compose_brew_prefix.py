"""Compose accepted physically continuous stages of one scene; no simulated action."""
from pathlib import Path
import json,shutil
import numpy as np
out=Path('results/brew-prefix-001');out.mkdir(exist_ok=False);shutil.copy2(__file__,out/Path(__file__).name);stages=[Path('results')/n for n in ['lid-physical-012','lid-place-004','spoon-physical-007','spoon-transport-007']];rows=[];liquid=[];brew=[];states=[];scene=None;provenance=[];last=None
for src in stages:
 r=json.loads((src/'report.json').read_text());assert r['pass'];assert last is None or Path(r['source'])==last
 if scene is None:scene=r['scene']
 assert r['scene']==scene
 q=np.load(src/'trajectory.npz')['qpos'];assert len(q)==len(r['rows']);states.extend(q);rows.extend({**x,'phase':src.name+':'+x['phase']} for x in r['rows']);liquid.extend(r['liquid_samples']);brew.extend(json.loads((src/'brew-samples.json').read_text()));provenance.append({'stage':str(src),'source':r['source'],'pass':True});last=src
assert all(b['t']>a['t'] for a,b in zip(rows,rows[1:]));np.savez_compressed(out/'trajectory.npz',qpos=states);(out/'brew-samples.json').write_text(json.dumps(brew,indent=2));shutil.copy2(last/'brew-state.json',out/'brew-state.json');(out/'report.json').write_text(json.dumps({'model':'gpt-6-astra','pass':True,'coffee_completed':False,'scene':scene,'initial_water_ml':800,'scope':'continuous accepted cold setup prefix, not completed coffee','stages':provenance,'rows':rows,'liquid_samples':liquid},indent=2));print(out)
