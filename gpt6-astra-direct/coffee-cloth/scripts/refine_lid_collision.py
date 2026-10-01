"""Refine lid collision against unchanged visual mesh, preserve original inertial properties."""
import json,shutil,hashlib,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np,trimesh,coacd,mujoco
out=Path('results/lid-collision-001');out.mkdir(exist_ok=False);shutil.copy2(__file__,out/Path(__file__).name);source=Path('results/coffee-layout-001/scene.xml').resolve();m=mujoco.MjModel.from_xml_path(str(source));body=m.body('tampa').id;meshpath=Path('assets/mesh/tampa/tampa_visual.obj').resolve();mesh=trimesh.load(meshpath,force='mesh',process=False);parts=coacd.run_coacd(coacd.Mesh(np.asarray(mesh.vertices,dtype=np.float64),np.asarray(mesh.faces,dtype=np.int32)),threshold=.01,max_convex_hull=64,preprocess_mode='auto',mcts_nodes=20,mcts_iterations=150,mcts_max_depth=4)
tree=ET.parse(source);root=tree.getroot();asset=root.find('asset');lid=root.find('.//body[@name="tampa"]');old=[g for g in lid.findall('geom') if g.get('name','').startswith('tampa_col')];template=old[0].attrib.copy()
for g in old:lid.remove(g)
# Do not conflate collision repair with guessed mass changes.
ET.SubElement(lid,'inertial',pos=' '.join(map(str,m.body_ipos[body])),quat=' '.join(map(str,m.body_iquat[body])),mass=str(m.body_mass[body]),diaginertia=' '.join(map(str,m.body_inertia[body])))
sections=[]
for i,(v,f) in enumerate(parts):
 name=f'tampa_refined_{i:03d}';p=(out/f'{name}.obj').resolve();part=trimesh.Trimesh(v,f,process=False);part.export(p);ET.SubElement(asset,'mesh',name=name,file=str(p));attrs=template.copy();attrs.update(name=name,mesh=name,density='0');ET.SubElement(lid,'geom',**attrs);sec=part.section([0,0,1],[0,0,.028]);sections.append(None if sec is None else float(np.linalg.norm(sec.vertices[:,:2],axis=1).max()))
scene=(out/'scene.xml').resolve();tree.write(scene);new=mujoco.MjModel.from_xml_path(str(scene));visual=mesh.section([0,0,1],[0,0,.028]);report={'model':'gpt-6-astra','scene':str(scene),'source_scene':str(source),'mesh_source':str(meshpath),'mesh_sha256':hashlib.sha256(meshpath.read_bytes()).hexdigest(),'parts':len(parts),'threshold':.01,'lid_mass_kg':float(new.body_mass[body]),'old_lid_mass_kg':float(m.body_mass[body]),'visual_radius_at28mm':float(np.linalg.norm(visual.vertices[:,:2],axis=1).max()),'collision_radius_at28mm':max(x for x in sections if x is not None),'scope':'collision refinement only; unchanged visual, mass and inertia; needs physical restart/revalidation'};(out/'report.json').write_text(json.dumps(report,indent=2));print(report,flush=True)
