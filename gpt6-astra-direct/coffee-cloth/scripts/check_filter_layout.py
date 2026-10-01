"""Offline screen of initial filter/cup locations against saved handle-lift path."""
import json,shutil
from pathlib import Path
import numpy as np,mujoco
out=Path('results/filter-layout-screen-001');out.mkdir(exist_ok=False);shutil.copy2(__file__,out/Path(__file__).name)
r=json.loads(Path('results/loadable-dynamics-011/report.json').read_text());m=mujoco.MjModel.from_xml_path(r['scene']);d=mujoco.MjData(m);path=np.load('results/loadable-dynamics-011/trajectory.npz')['qpos'];ca=m.jnt_qposadr[m.joint('coador_livre').id];ua=m.jnt_qposadr[m.joint('copo_livre').id];rows=[]
for xy in [[.27,.16],[.28,.08],[.30,.20],[.26,0],[.25,.25],[.30,.27],[.35,.15],[.4,.15],[.25,.1],[.25,.2]]:
 bad=None
 for i in range(0,len(path),4):
  d.qpos[:]=path[i];d.qpos[ca:ca+2]=xy;d.qpos[ua:ua+2]=np.array(xy)+[0,.005];d.qpos[ua+2]=.7596;mujoco.mj_forward(m,d)
  for ct in d.contact:
   if ct.dist>=0:continue
   bn=[m.body(int(m.geom_bodyid[g])).name for g in [ct.geom1,ct.geom2]]
   if any(n in ['coador','copo'] for n in bn) and any(n.startswith(('left_','right_')) or n=='chaleira' for n in bn):bad={'frame':i,'bodies':bn,'depth_m':float(ct.dist)};break
  if bad:break
 rows.append({'xy':xy,'pass':bad is None,'failure':bad});print(rows[-1],flush=True)
(out/'report.json').write_text(json.dumps({'model':'gpt-6-astra','scope':'sampled offline overlap audit only; cup placed on support base needs settling validation','rows':rows},indent=2))
