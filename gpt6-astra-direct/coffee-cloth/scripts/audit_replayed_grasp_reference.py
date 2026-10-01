"""Recover an exact grasp reference only from a bit-identical physical replay."""
import argparse,hashlib,json,shutil
from pathlib import Path
import numpy as np
ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);ap.add_argument('--replay',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(exist_ok=False);shutil.copy2(__file__,a.out/Path(__file__).name)
r=json.loads((a.source/'report.json').read_text());rr=json.loads((a.replay/'report.json').read_text());checks={}
for name in ['continuation.npz','trajectory.npz']:
 with np.load(a.source/name) as old,np.load(a.replay/name) as new:
  checks[name]={k:bool(k in new and np.array_equal(old[k],new[k])) for k in old.files}
for name in ['brew-state.json','liquid-state.json']:
 checks[name]=json.loads((a.source/name).read_text())==json.loads((a.replay/name).read_text())
passed=r['pass'] and rr['pass'] and rr.get('grasp_reference_R') is not None and all(all(v.values()) if isinstance(v,dict) else v for v in checks.values())
result={'model':'gpt-6-astra','source':str(a.source),'replay':str(a.replay),'pass':bool(passed),'checks':checks,'source_checkpoint_sha256':hashlib.sha256((a.source/'continuation.npz').read_bytes()).hexdigest(),'grasp_reference_R':rr['grasp_reference_R'] if passed else None,'scope':'all original checkpoint arrays and full recorded trajectory bit-identical; original ledger values identical; reference measured at actual lift-start in replay'}
(a.out/'report.json').write_text(json.dumps(result,indent=2));print(result)
