"""Separate the narrow knob from broad lid hulls using exact visual-mesh height slices."""
import json,shutil,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np,trimesh,mujoco
out=Path('results/lid-collision-004');out.mkdir(exist_ok=False);shutil.copy2(__file__,out/Path(__file__).name);tree=ET.parse('results/lid-collision-001/scene.xml');root=tree.getroot();asset=root.find('asset');lid=root.find('.//body[@name="tampa"]');old=[g for g in lid.findall('geom') if g.get('name','').startswith('tampa_refined_')];template=old[0].attrib.copy();parts=[]
for g in old:
 path=asset.find(f"mesh[@name='{g.get('mesh')}']").get('file');mesh=trimesh.load(path,force='mesh',process=False);cut=mesh.slice_plane([0,0,.026],[0,0,-1]);lid.remove(g)
 if len(cut.vertices)>=4:
  try:parts.append(cut.convex_hull)
  except Exception:pass
visual=trimesh.load('assets/mesh/tampa/tampa_visual.obj',force='mesh',process=False)
for lo,hi in zip([.026,.027,.028,.029,.031],[.027,.028,.029,.031,.035001]):
 cut=visual.slice_plane([0,0,lo],[0,0,1]).slice_plane([0,0,hi],[0,0,-1]);parts.append(cut.convex_hull)
for i,p in enumerate(parts):
 name=f'tampa_slice_{i:03d}';path=(out/f'{name}.obj').resolve();p.export(path);ET.SubElement(asset,'mesh',name=name,file=str(path));attrs=template.copy();attrs.update(name=name,mesh=name);ET.SubElement(lid,'geom',**attrs)
checks=[]
for z in [.027,.028,.029,.030,.031,.032,.033,.034]:
 real=visual.section([0,0,1],[0,0,z]);vr=float(np.linalg.norm(real.vertices[:,:2],axis=1).max());radii=[]
 for part in parts:
  section=part.section([0,0,1],[0,0,z])
  if section is not None:radii.append(float(np.linalg.norm(section.vertices[:,:2],axis=1).max()))
 cr=max(radii);checks.append({'z_m':z,'visual_radius_m':vr,'collision_radius_m':cr,'outer_error_m':cr-vr})
scene=(out/'scene.xml').resolve();tree.write(scene);m=mujoco.MjModel.from_xml_path(str(scene));report={'model':'gpt-6-astra','scene':str(scene),'parts':len(parts),'knob_checks':checks,'pass':max(abs(x['outer_error_m']) for x in checks)<.001,'mass_kg':float(m.body_mass[m.body('tampa').id]),'scope':'visual-mesh matching of knob collider, not measured real dimensions or dynamic validation; base colliders retained clipped at26mm; five convex knob slices'};(out/'report.json').write_text(json.dumps(report,indent=2));print(report)
