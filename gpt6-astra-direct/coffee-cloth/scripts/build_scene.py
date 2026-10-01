"""Diagnostic cloth dripper: native deformable mesh pinned to a fixed ring. gpt-6-astra."""
import copy,math,argparse
import xml.etree.ElementTree as ET
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];G1=ROOT.parent/'g1-cup-grasp'
ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--center',type=float,nargs=2,default=[.32,-.255]);ap.add_argument('--payload-kg',type=float,default=0.0);args=ap.parse_args()
if args.out.exists():raise FileExistsError(args.out)
tree=ET.parse(G1/'scene/scene_grasp.xml');root=tree.getroot();root.set('model','G1 cloth coffee diagnostic')
robot_path=args.out.with_name(args.out.stem+'-robot.xml').resolve()
if robot_path.exists():raise FileExistsError(robot_path)
robot=ET.parse(G1/'scene/g1_grasp.xml');robot.getroot().find('compiler').set('meshdir',str((G1/'scene/meshes').resolve()))
robot.write(robot_path,encoding='unicode')
root.find('include').set('file',str(robot_path))
w=root.find('worldbody');x,y=args.center;z=.90
stand=ET.SubElement(w,'body',name='coffee_stand',pos=f'{x} {y} .75')
ET.SubElement(stand,'geom',name='stand_base',type='box',size='.065 .065 .003',pos='0 0 .003',rgba='.2 .12 .055 1')
ET.SubElement(stand,'geom',name='stand_post',type='capsule',fromto='.062 0 .006 .062 0 .15',size='.006',rgba='.2 .12 .055 1')
ET.SubElement(stand,'geom',name='stand_arm',type='capsule',fromto='.062 0 .15 .04 0 .15',size='.004',rgba='.12 .12 .12 1')
ring=ET.SubElement(w,'body',name='filter_ring',pos=f'{x} {y} {z}')
N=16
for j in range(N):
 a,b=2*math.pi*j/N,2*math.pi*(j+1)/N
 ET.SubElement(ring,'geom',name=f'ring_{j}',type='capsule',fromto=f'{.042*math.cos(a)} {.042*math.sin(a)} 0 {.042*math.cos(b)} {.042*math.sin(b)} 0',size='.002',rgba='.16 .13 .1 1')
points=[];tris=[]
for k in range(4):
 r=.04-(.029*k/3);zz=-.065*k/3
 for j in range(N):points.extend([r*math.cos(2*math.pi*j/N),r*math.sin(2*math.pi*j/N),zz])
for k in range(3):
 for j in range(N):
  a=k*N+j;b=k*N+(j+1)%N;c=(k+1)*N+j;d=(k+1)*N+(j+1)%N;tris.extend([a,c,b,b,c,d])
points.extend([0,0,-.069]);center=len(points)//3-1
for j in range(N):tris.extend([3*N+j,center,3*N+(j+1)%N])
f=ET.SubElement(ring,'flexcomp',name='cloth_filter',type='direct',dim='2',point=' '.join(map(str,points)),element=' '.join(map(str,tris)),mass='.008',radius='.0007',rgba='.70 .53 .32 1')
ET.SubElement(f,'pin',id=' '.join(map(str,range(N))))
ET.SubElement(f,'edge',equality='true',damping='.01')
ET.SubElement(f,'contact',selfcollide='none',friction='.5 .005 .0001')
receiver=copy.deepcopy(w.find("body[@name='copo']"));receiver.set('name','receiver');receiver.set('pos',f'{x} {y} .758');receiver.set('quat','0.7071067811865476 0 0 0.7071067811865475')
for e in receiver.iter():
 if 'name' in e.attrib:e.set('name',e.get('name').replace('copo','receiver'))
w.append(receiver)
if args.payload_kg:
 height=args.payload_kg/(1000*math.pi*.037**2)
 if not 0<height<=.086:raise ValueError('Water-equivalent ballast does not fit')
 ET.SubElement(w.find("body[@name='copo']"),'geom',name='diagnostic_payload',type='cylinder',size=f'.037 {height/2}',pos=f'0 0 {.006+height/2}',mass=str(args.payload_kg),rgba='0 0 0 0',contype='0',conaffinity='0')
ET.SubElement(w,'camera',name='coffee_closeup',pos='.55 -.85 1.16',xyaxes='.94 .34 0 -.12 .34 .93',fovy='36')
args.out.parent.mkdir(parents=True,exist_ok=True);tree.write(args.out,encoding='unicode')
print(args.out)
