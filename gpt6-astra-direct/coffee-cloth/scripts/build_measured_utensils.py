"""Fit confirmed lid knob and spoon drawing dimensions. Unmeasured spoon depth/density explicit."""
from pathlib import Path
import json,shutil,xml.etree.ElementTree as ET
import numpy as np,trimesh,mujoco
out=Path('results/utensil-dimensions-001');out.mkdir(exist_ok=False);shutil.copy2(__file__,out/Path(__file__).name);tree=ET.parse('results/prepared-layout-005/scene.xml');root=tree.getroot();assets=root.find('asset');lid=root.find('.//body[@name="tampa"]');spoon=root.find('.//body[@name="scoop"]');old_model=mujoco.MjModel.from_xml_path('results/prepared-layout-005/scene.xml')
# Preserve texture coordinates while shrinking only the confirmed knob region.
source=Path('assets/mesh/tampa/tampa_visual.obj');lines=source.read_text().splitlines();v=np.array([[float(x) for x in l.split()[1:4]] for l in lines if l.startswith('v ')]);top=v[v[:,2]>=.027];center=(top[:,:2].max(0)+top[:,:2].min(0))/2;scale=.020/(top[:,:2].max(0)-top[:,:2].min(0))
def fit_knob(vertices):
 vertices=np.array(vertices,copy=True);f=np.clip((vertices[:,2]-.026)/.001,0,1);vertices[:,:2]=vertices[:,:2]*(1-f[:,None])+((vertices[:,:2]-center)*scale+center)*f[:,None];return vertices
newv=fit_knob(v);it=iter(newv);newlines=['v '+' '.join(f'{x:.10f}' for x in next(it)) if l.startswith('v ') else l for l in lines];visual=(out/'lid-visual.obj').resolve();visual.write_text('\n'.join(newlines)+'\n');ET.SubElement(assets,'mesh',name='lid_dim_visual',file=str(visual));lid.find('geom[@name="tampa_vis"]').set('mesh','lid_dim_visual')
for i,g in enumerate([g for g in lid.findall('geom') if g.get('name','').startswith('tampa_slice_')]):
 original=assets.find(f"mesh[@name='{g.get('mesh')}']").get('file');part=trimesh.load(original,force='mesh',process=False);part.vertices=fit_knob(part.vertices);part=part.convex_hull;path=(out/f'lid-collision-{i:03d}.obj').resolve();part.export(path);name=f'lid_dim_col_{i:03d}';ET.SubElement(assets,'mesh',name=name,file=str(path));g.set('mesh',name)
# White spoon: elliptical paraboloid shell,30x25mm outer rim,120mm overall.
# Depth6mm, wall1mm and handle1.5mm/density1000kg/m3 are provisional assumptions.
a,b=.015,.0125;cx=-.045;depth=.006;thick=.001;verts=[];faces=[];N=64;K=16
for inner in [False,True]:
 for ir in range(K+1):
  rr=ir/K
  for j in range(N):
   t=2*np.pi*j/N;rx=a-thick if inner else a;ry=b-thick if inner else b;z=thick+(depth-thick)*rr**2 if inner else depth*rr**2;verts.append([cx+rx*rr*np.cos(t),ry*rr*np.sin(t),z])
stride=(K+1)*N
for surface in [0,1]:
 for ir in range(K):
  for j in range(N):
   k=surface*stride+ir*N+j;l=surface*stride+ir*N+(j+1)%N;quad=[k,l,l+N,k+N]
   if surface==0:quad=quad[::-1]
   faces.extend([[quad[0],quad[1],quad[2]],[quad[0],quad[2],quad[3]]])
for j in range(N):
 q=[K*N+j,K*N+(j+1)%N,stride+K*N+(j+1)%N,stride+K*N+j];faces.extend([[q[0],q[1],q[2]],[q[0],q[2],q[3]]])
bowl=trimesh.Trimesh(verts,faces,process=True);bowl.remove_unreferenced_vertices();trimesh.repair.fix_normals(bowl,multibody=True)
# Convex wall cells retain the bowl cavity.
parts=[]
for ir in range(5):
 for j in range(24):
  vv=[]
  for inner in [False,True]:
   for rr in [ir/5,(ir+1)/5]:
    for t in [j*2*np.pi/24,(j+1)*2*np.pi/24]:
     rx=a-thick if inner else a;ry=b-thick if inner else b;z=thick+(depth-thick)*rr**2 if inner else depth*rr**2;vv.append([cx+rx*rr*np.cos(t),ry*rr*np.sin(t),z])
  parts.append(trimesh.convex.convex_hull(np.array(vv)))
# Rounded tapered handle; intersects bowl neck by2mm.
xnodes=np.linspace(-.032,.060,25);handleparts=[]
def section(x):
 u=(x+.032)/.092;width=.0035+.0015*u
 if x>.055:width=.005*np.sqrt(max(0,1-((x-.055)/.005)**2))
 width=max(width,.00001);z=.005+0.0006*u;return width,z
for x0,x1 in zip(xnodes[:-1],xnodes[1:]):
 vv=[]
 for x in [x0,x1]:
  w,z=section(x)
  for yy in [-w,w]:
   for zz in [z-.00075,z+.00075]:vv.append([x,yy,zz])
 handleparts.append(trimesh.convex.convex_hull(np.array(vv)))
parts+=handleparts;visible=trimesh.util.concatenate([bowl,*handleparts]);vp=(out/'spoon-visual.obj').resolve();visible.export(vp)
for g in list(spoon.findall('geom')):spoon.remove(g)
for inertial in list(spoon.findall('inertial')):spoon.remove(inertial)
ET.SubElement(assets,'material',name='spoon_white_plastic',rgba='.92 .92 .90 1',specular='.25',shininess='.25');ET.SubElement(assets,'mesh',name='spoon_dim_visual',file=str(vp));ET.SubElement(spoon,'geom',name='scoop_vis',type='mesh',mesh='spoon_dim_visual',material='spoon_white_plastic',contype='0',conaffinity='0',group='1',density='0')
for i,p in enumerate(parts):
 name=f'scoop_dim_col_{i:03d}';path=(out/f'spoon-collision-{i:03d}.obj').resolve();p.export(path);ET.SubElement(assets,'mesh',name=name,file=str(path));ET.SubElement(spoon,'geom',name=name,type='mesh',mesh=name,group='3',rgba='0 0 0 0',friction='1 0.005 0.0001',condim='4',priority='1',solref='.008 1',density='1000')
scene=(out/'scene.xml').resolve();tree.write(scene);m=mujoco.MjModel.from_xml_path(str(scene));newtop=newv[newv[:,2]>=.027];metrics={'spoon_length_mm':float(np.ptp(visible.vertices[:,0])*1000),'spoon_bowl_length_mm':float(np.ptp(bowl.vertices[:,0])*1000),'spoon_bowl_width_mm':float(np.ptp(bowl.vertices[:,1])*1000),'lid_knob_width_xy_mm':(np.ptp(newtop[:,:2],axis=0)*1000).tolist(),'lid_knob_height_mm':8.,'spoon_mass_kg':float(m.body_mass[m.body('scoop').id]),'old_spoon_mass_kg':float(old_model.body_mass[old_model.body('scoop').id])};passed=abs(metrics['spoon_length_mm']-120)<.1 and abs(metrics['spoon_bowl_length_mm']-30)<.1 and abs(metrics['spoon_bowl_width_mm']-25)<.1 and max(abs(np.array(metrics['lid_knob_width_xy_mm'])-20))<.01
report={'model':'gpt-6-astra','scene':str(scene),'pass':bool(passed),'scope':'dimensional geometry validation only; revalidate physical grasps and stability','references':['assets/desenhos/PAB-003-colher.png','assets/desenhos/TAM-001-tampa-marrom.png'],'metrics':metrics,'assumptions':{'spoon_depth_mm':6,'spoon_wall_mm':1,'handle_thickness_mm':1.5,'plastic_density_kg_m3':1000,'mass_measured':False},'spoon_capacity_model_ml':float(np.pi*(a-thick)*(b-thick)*(depth-thick)/2*1e6),'collision_parts_spoon':len(parts),'lid_mass_inertia_preserved':True};(out/'report.json').write_text(json.dumps(report,indent=2));print(report)
