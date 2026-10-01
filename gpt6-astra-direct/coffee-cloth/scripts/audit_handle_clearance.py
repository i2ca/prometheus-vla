"""Bounded handle-gap measurements in the current mesh, not a grasp feasibility proof."""
import argparse,json,shutil,hashlib
from pathlib import Path
import xml.etree.ElementTree as ET
import mujoco,numpy as np,trimesh

ap=argparse.ArgumentParser()
ap.add_argument('--probe-scene',type=Path,required=True)
ap.add_argument('--robot',type=Path,required=True)
ap.add_argument('--out',type=Path,required=True)
a=ap.parse_args();a.out.mkdir(exist_ok=False,parents=True)
shutil.copy2(__file__,a.out/Path(__file__).name)
tree=ET.parse(a.probe_scene)
tree.getroot().find("./worldbody/body[@name='probe']/geom").set('size','0.00005')
tree.write(a.out/'probe.xml')
m=mujoco.MjModel.from_xml_path(str(a.out/'probe.xml'));d=mujoco.MjData(m)
probe=m.geom('probe').id
geoms=[g for g in range(m.ngeom) if (m.geom(g).name or '').startswith('chaleira_col')]
xs=np.arange(.035,.106,.0005);zs=np.arange(.04,.212,.002)
rows=[]
for z in zs:
    occupied=[]
    for x in xs:
        d.mocap_pos[0]=[x,0,z];mujoco.mj_forward(m,d)
        occupied.append(any(mujoco.mj_geomDistance(m,d,probe,g,.01,None)<=0 for g in geoms))
    gaps=[];start=None
    for i in range(1,len(xs)):
        if occupied[i-1] and not occupied[i]:start=i
        if start is not None and occupied[i]:
            gaps.append({'inner_x_m':float(xs[start]),'outer_x_m':float(xs[i-1]),
                         'sampled_free_width_mm':float((xs[i-1]-xs[start])*1000)})
            start=None
    if gaps:rows.append({'z_m':float(z),'bounded_gaps':gaps})
robot=ET.parse(a.robot).getroot();meshdir=Path(robot.find('compiler').get('meshdir'))
fingers={}
for name in ('right_hand_index_1_link','right_hand_middle_1_link','right_hand_thumb_2_link'):
    mesh=next(x for x in robot.findall('./asset/mesh') if x.get('name')==name)
    path=meshdir/mesh.get('file');v=trimesh.load(path,force='mesh').vertices
    fingers[name]={'full_mesh_bbox_mm':(np.ptp(v,axis=0)*1000).tolist(),
                   'source':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
report={'model':'gpt-6-astra','scope':'Current collision mesh central plane y=0, x step0.5mm, z step2mm, 0.05mm radius probe',
        'real_handle_dimensions_verified':False,'gap_samples':rows,'finger_meshes':fingers,
        'limitations':['A central slice does not certify a full insertion path or hand pose.',
                       'Whole-finger bounding boxes are not fingertip cross sections.',
                       'Exterior pinch on the handle may avoid insertion through the hole; not tested here.',
                       'Do not widen the handle to fit the robot without measurement evidence.']}
(a.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'bounded_gap_heights':len(rows),'max_sampled_gap_mm':max((g['sampled_free_width_mm'] for r in rows for g in r['bounded_gaps']),default=0)},indent=2))
