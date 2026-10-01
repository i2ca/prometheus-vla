"""Independent MuJoCo collision-space check and render of rebuilt handle."""
import argparse,json,copy,shutil
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np,mujoco,cv2,trimesh
ap=argparse.ArgumentParser();ap.add_argument('--asset',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False);shutil.copy2(__file__,a.out/Path(__file__).name)
source=ET.parse(a.asset/'scene.xml').getroot();xml=ET.Element('mujoco');assets=ET.SubElement(xml,'asset')
ET.SubElement(assets,'texture',name='background',type='skybox',builtin='gradient',rgb1='.38 .42 .46',rgb2='.2 .23 .27',width='128',height='128')
visual=ET.SubElement(xml,'visual');ET.SubElement(visual,'headlight',ambient='.55 .55 .55',diffuse='.7 .7 .7',specular='.1 .1 .1')
for node in source.find('asset'):
    if node.get('name','').startswith(('chaleira','handle_col')):assets.append(copy.deepcopy(node))
world=ET.SubElement(xml,'worldbody');body=copy.deepcopy(source.find(".//body[@name='chaleira']"));body.set('pos','0 0 0');body.set('quat','1 0 0 0')
for n in list(body):
    if n.tag in ('freejoint','joint'):body.remove(n)
world.append(body);ET.SubElement(world,'light',pos='.1 -.5 .6',diffuse='.8 .8 .8')
ET.SubElement(world,'camera',name='side',pos='.02 -.65 .116',xyaxes='1 0 0 0 0 1',fovy='30')
probe=ET.SubElement(world,'body',name='probe',mocap='true',pos='0 0 1');ET.SubElement(probe,'geom',name='probe',type='sphere',size='.00005',group='4')
ET.ElementTree(xml).write(a.out/'probe.xml');m=mujoco.MjModel.from_xml_path(str(a.out/'probe.xml'));d=mujoco.MjData(m)
pg=m.geom('probe').id;gs=[g for g in range(m.ngeom) if m.geom(g).name=='chaleira_hot_body' or (m.geom(g).name or '').startswith('handle_col')]
mujoco.mj_forward(m,d);centers=d.geom_xpos.copy();xs=np.arange(.050,.1451,.0005);rows=[]
for z in np.arange(.035,.1951,.001):
    local=[g for g in gs if abs(centers[g,2]-z)<=m.geom_rbound[g]+.001]
    occupied=[]
    for x in xs:
        d.mocap_pos[0]=[x,0,z];mujoco.mj_forward(m,d)
        occupied.append(any(mujoco.mj_geomDistance(m,d,pg,g,.1,None)<=0 for g in local))
    start=None
    for i in range(1,len(xs)):
        if occupied[i-1] and not occupied[i]:start=i
        if start is not None and occupied[i]:
            rows.append({'z_m':float(z),'x_inner_m':float(xs[start]),'x_outer_m':float(xs[i-1]),'width_mm':float((xs[i-1]-xs[start])*1000)});start=None
# Full lateral path,18mm-diameter probe, uses all collision parts.
m.geom_size[pg,0]=.009;m.geom_rbound[pg]=.009;paths=[]
for target in [.075,.100,.125,.150]:
    row=min(rows,key=lambda r:abs(r['z_m']-target));x=(row['x_inner_m']+row['x_outer_m'])/2;clearance=1.
    for y in np.linspace(-.055,.055,111):
        d.mocap_pos[0]=[x,y,row['z_m']];mujoco.mj_forward(m,d)
        clearance=min(clearance,*(mujoco.mj_geomDistance(m,d,pg,g,.1,None) for g in gs))
    paths.append({'x_m':x,'z_m':row['z_m'],'minimum_clearance_mm':float(clearance*1000),'passes_1mm_margin':bool(clearance>=.001)})
d.mocap_pos[0]=[0,0,1];mujoco.mj_forward(m,d);renderer=mujoco.Renderer(m,480,640)
for name,group in [('visual',1),('collision',3)]:
    opt=mujoco.MjvOption();opt.geomgroup[:]=0;opt.geomgroup[group]=1
    renderer.update_scene(d,camera='side',scene_option=opt);cv2.imwrite(str(a.out/(name+'.png')),cv2.cvtColor(renderer.render(),cv2.COLOR_RGB2BGR))
renderer.close()
v=np.vstack([trimesh.load(a.asset/name,process=False).vertices for name in ('body_visual.obj','handle_visual.obj')])
middle=[r['width_mm'] for r in rows if .070<=r['z_m']<=.160]
report={'model':'gpt-6-astra','visual_envelope_xyz_mm':(np.ptp(v,axis=0)*1000).tolist(),'central_gap_samples':rows,
 'sampled_opening_height_mm':float((max(r['z_m'] for r in rows)-min(r['z_m'] for r in rows))*1000),
 'middle_gap_width_range_mm':[min(middle),max(middle)],'sphere_entry_paths':paths,
 'limitations':['Grid precision0.5mm x,1mm z.','Sphere entry is not full-hand reach or grasp proof.','Nominal shape uses user approximate dimensions and declared reconstruction assumptions.']}
(a.out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='central_gap_samples'},indent=2))
