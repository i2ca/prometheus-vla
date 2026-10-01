"""Explicit alternative INITIAL layouts, no object teleport during execution."""
import json,subprocess,sys,shutil,xml.etree.ElementTree as ET
from pathlib import Path
import mujoco,numpy as np
out=Path('results/handle-layout-search-003');out.mkdir(exist_ok=False);shutil.copy2(__file__,out/Path(__file__).name);oldq=np.load('results/free-props-001/initial-qpos.npy');rows=[]
for i,(x,y,yaw) in enumerate([(x,y,angle) for x,y in [(.36,.18),(.40,.10)] for angle in [90,120,150,180]]):
 folder=out/f'layout-{i:02d}';folder.mkdir();tree=ET.parse('results/free-props-001/scene.xml');angle=np.deg2rad(yaw)/2;quat=f'{np.cos(angle)} 0 0 {np.sin(angle)}'
 for name,z in [('chaleira',.766),('base_eletrica',.75)]:
  b=tree.find(f".//body[@name='{name}']");b.set('pos',f'{x} {y} {z}');b.set('quat',quat)
 scene=folder/'scene.xml';tree.write(scene);m=mujoco.MjModel.from_xml_path(str(scene));d=mujoco.MjData(m);q=m.qpos0.copy()
 for j in range(m.njnt):
  if m.jnt_type[j]==mujoco.mjtJoint.mjJNT_HINGE:q[m.jnt_qposadr[j]]=oldq[m.jnt_qposadr[j]]
 np.save(folder/'initial.npy',q)
 cmd=[sys.executable,'scripts/check_loadable_reach.py','--scene',str(scene),'--initial',str(folder/'initial.npy'),'--source','results/handle-fit-3d-002','--capacity','results/handle-capacity-3d-002/report.json','--out',str(folder/'reach'),'--seeds','3']
 with (folder/'reach.log').open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
 r=json.loads((folder/'reach/report.json').read_text());passed=[a for a in r['results'] if a['pass']];rows.append({'layout':i,'position':[x,y],'yaw':yaw,'passed':len(passed)});print(rows[-1],flush=True)
 (out/'report.json').write_text(json.dumps({'model':'gpt-6-astra','initial_layout_changes':True,'results':rows},indent=2))
 # Collect all alternatives; no early acceptance.
