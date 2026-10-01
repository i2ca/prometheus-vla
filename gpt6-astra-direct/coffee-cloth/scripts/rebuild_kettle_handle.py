"""Photo-informed approximate kettle revision. Preserves all previous generations.

140x25mm handle references are approximate user measurements. Catalog envelope
axes and unmeasured tube thickness/wave depth remain declared assumptions.
"""
import argparse, json, shutil, hashlib
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import trimesh
from scipy.spatial import ConvexHull

ap=argparse.ArgumentParser()
ap.add_argument('--out',type=Path,required=True)
a=ap.parse_args();out=a.out.resolve();out.mkdir(parents=True,exist_ok=False)
root=Path(__file__).resolve().parents[1]
shutil.copy2(__file__,out/Path(__file__).name)
old=root/'results/kettle-fit-20260919-001'
mesh=trimesh.load(old/'chaleira_visual.obj',process=False)
v=mesh.vertices.copy();faces=mesh.faces.copy();uv=mesh.visual.uv.copy()
# Remove the old handle outside the body's radial envelope before fitting body.
zgrid=np.linspace(0,.232,60)
radii=np.array([np.max(np.abs(v[np.abs(v[:,2]-z)<.006,1])) for z in zgrid])
radius=np.interp(v[:,2],zgrid,radii)
keep=np.all((v[:,0]<=radius+.0015)[faces],axis=1)
faces=faces[keep]
used=np.unique(faces)
remap=np.full(len(v),-1,dtype=int);remap[used]=np.arange(len(used));compact_faces=remap[faces]
scale=.159/np.ptp(v[used,1]);v[:,:2]*=scale
bodyhull=trimesh.Trimesh(v[used],compact_faces,process=False).convex_hull
equations=ConvexHull(bodyhull.vertices).equations
def back(z):
    e=equations[equations[:,0]>1e-7]
    return float(np.min((-e[:,3]-e[:,2]*z)/e[:,0]))

# Rounded C handle: 8mm section radius in XZ and 11mm half-thickness in Y.
# Horizontal connector inner faces lie at45 and185mm ->140mm clear height.
section_r=.008;half_y=.011
zlow=.037;zhigh=.193;corner=.018
points=[]
def relief_at(z):
    # Four smooth inner grip humps; their crests preserve nominal25mm gap.
    u=np.clip((z-.060)/.110,0,1)
    return .001*(1-np.cos(8*np.pi*u)) if .060<z<.170 else 0.
def rail(z):
    return back(z)+.025+section_r+relief_at(z)/2
xl=rail(zlow+corner)-corner
for x in np.linspace(back(zlow)-.004,xl,6):points.append([x,zlow])
for t in np.linspace(-np.pi/2,0,13)[1:]:points.append([xl+corner*np.cos(t),zlow+corner+corner*np.sin(t)])
for z in np.linspace(zlow+corner,zhigh-corner,73)[1:]:points.append([rail(z),z])
xh=rail(zhigh-corner)-corner
for t in np.linspace(0,np.pi/2,13)[1:]:points.append([xh+corner*np.cos(t),zhigh-corner+corner*np.sin(t)])
for x in np.linspace(xh,back(zhigh)-.004,6)[1:]:points.append([x,zhigh])
points=np.array(points);rings=[];angles=np.linspace(0,2*np.pi,24,endpoint=False)
for i,p in enumerate(points):
    tangent=points[min(i+1,len(points)-1)]-points[max(0,i-1)]
    tangent/=np.linalg.norm(tangent);normal=np.array([tangent[1],-tangent[0]])
    radius=section_r-relief_at(p[1])/2
    rings.append(np.array([[p[0]+radius*np.cos(t)*normal[0],half_y*np.sin(t),p[1]+radius*np.cos(t)*normal[1]] for t in angles]))
rings=np.array(rings);hv=rings.reshape(-1,3);hf=[];n=len(angles)
for i in range(len(rings)-1):
    for j in range(n):
        k=(j+1)%n;hf.extend([[i*n+j,(i+1)*n+j,(i+1)*n+k],[i*n+j,(i+1)*n+k,i*n+k]])
for j in range(1,n-1):hf.extend([[0,j+1,j],[(len(rings)-1)*n,(len(rings)-1)*n+j,(len(rings)-1)*n+j+1]])
handle=trimesh.Trimesh(hv,hf,process=False);handle.fix_normals();handle.export(out/'handle_visual.obj')
# Fit inferred213mm long envelope by reshaping only spout excess beyond body.
target_min=hv[:,0].max()-.213
rv=np.interp(v[:,2],zgrid,radii)*scale
excess=np.maximum(-v[:,0]-rv,0)
candidates=used[excess[used]>1e-8]
factor=np.min((v[candidates,0]-target_min)/excess[candidates])
v[:,0]-=factor*excess
with (out/'body_visual.obj').open('x') as f:
    for p in v[used]:f.write('v '+' '.join(f'{x:.9f}' for x in p)+'\n')
    for p in uv[used]:f.write('vt '+' '.join(f'{x:.9f}' for x in p)+'\n')
    for face in compact_faces+1:f.write('f '+' '.join(f'{i}/{i}' for i in face)+'\n')
shutil.copy2(old/'chaleira_tex.png',out/'body_texture.png')
body=trimesh.Trimesh(v[used],compact_faces,process=False).convex_hull
body.export(out/'body_collision.obj')
parts=[]
for i in range(len(rings)-1):
    part=trimesh.Trimesh(vertices=np.vstack([rings[i],rings[i+1]]),process=False).convex_hull
    name=f'handle_col{i:03d}';part.export(out/(name+'.obj'));parts.append((name,part))

source=root/'results/kettle-scene-20260919-001/scene.xml'
tree=ET.parse(source);xml=tree.getroot();assets=xml.find('asset')
for child in list(assets):
    if child.get('name','').startswith('chaleira'):assets.remove(child)
ET.SubElement(assets,'texture',name='chaleira_tex',type='2d',file=str(out/'body_texture.png'))
ET.SubElement(assets,'material',name='chaleira_mat',texture='chaleira_tex',specular='.3',shininess='.3')
for name,file in [('chaleira_body_visual','body_visual.obj'),('chaleira_body_collision','body_collision.obj'),('chaleira_handle_visual','handle_visual.obj')]+[(name,name+'.obj') for name,_ in parts]:
    ET.SubElement(assets,'mesh',name=name,file=str(out/file))
b=xml.find(".//body[@name='chaleira']")
for child in list(b):
    if child.tag in ('geom','inertial'):b.remove(child)
ET.SubElement(b,'geom',name='chaleira_body_visual',type='mesh',mesh='chaleira_body_visual',material='chaleira_mat',contype='0',conaffinity='0',density='0',group='1')
ET.SubElement(b,'geom',name='chaleira_handle_visual',type='mesh',mesh='chaleira_handle_visual',rgba='.035 .035 .04 1',contype='0',conaffinity='0',density='0',group='1')
ET.SubElement(b,'geom',name='chaleira_hot_body',type='mesh',mesh='chaleira_body_collision',mass='.88',group='3',rgba='.7 .7 .7 1',friction='1 .005 .0001',condim='4')
total=sum(p.volume for _,p in parts)
for name,part in parts:
    ET.SubElement(b,'geom',name=name,type='mesh',mesh=name,mass=str(.12*part.volume/total),group='3',rgba='.05 .05 .05 1',friction='1 .005 .0001',condim='4')
tree.write(out/'scene.xml')
allv=np.vstack([v[used],hv]);measure=np.ptp(allv,axis=0)*1000
report={'model':'gpt-6-astra','source_scene':str(source),'new_scene':str(out/'scene.xml'),
 'user_handle_references_mm':{'height':140,'width_at_grip_crests':25},
 'visual_envelope_xyz_mm':measure.tolist(),'handle_collision_parts':len(parts),
 'body_collision_geom':'chaleira_hot_body','permitted_grasp_geom_prefix':'handle_col',
 'mass_kg':1,'mass_distribution_assumed_kg':{'body':.88,'handle':.12},
 'assumptions':['EEK10 identification and213mm long-axis mapping are provisional.',
 '159mm transverse body envelope inferred from catalog; body diameters not independently measured.',
 'Handle140x25mm are approximate user measurements, not metrology.',
 'Section16x22mm, four2mm-depth grip reliefs, connector heights and curvature are photo-informed approximations.',
 'Retained body texture; cropped old handle. Body collision is solid convex hull and includes lid/spout as forbidden contact.',
 'No fluid simulation or physical handle-grasp validation in this generation.'],
 'measurement_reference':'assets/references/electrolux-handle-20260919/measurements.json'}
(out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
