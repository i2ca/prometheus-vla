"""Record current scoop geometry and support pose, not a real-object measurement."""
from pathlib import Path
import json,shutil,numpy as np,trimesh,mujoco,cv2
out=Path('results/scoop-geometry-001');out.mkdir(exist_ok=False);shutil.copy2(__file__,out/Path(__file__).name);source=Path('results/lid-place-002');r=json.loads((source/'report.json').read_text());m=mujoco.MjModel.from_xml_path(r['scene']);d=mujoco.MjData(m);mujoco.mj_setState(m,d,np.load(source/'continuation.npz')['integration'],mujoco.mjtState.mjSTATE_INTEGRATION);mujoco.mj_forward(m,d);mesh=trimesh.load('assets/mesh/scoop/scoop_visual.obj',force='mesh',process=False);sections=[]
for x in np.linspace(mesh.bounds[0,0]+.001,mesh.bounds[1,0]-.001,13):
 sec=mesh.section([1,0,0],[x,0,0]);sections.append({'x_m':float(x),'bounds':None if sec is None else [sec.vertices.min(0).tolist(),sec.vertices.max(0).tolist()]})
b=m.body('scoop').id;report={'model':'gpt-6-astra','source':str(source),'scene':r['scene'],'local_visual_bounds_m':mesh.bounds.tolist(),'sections':sections,'body_position_m':d.xpos[b].tolist(),'body_rotation':d.xmat[b].reshape(3,3).tolist(),'mass_kg':float(m.body_mass[b]),'note':'simulated visual geometry, not caliper measurement'};(out/'report.json').write_text(json.dumps(report,indent=2));ren=mujoco.Renderer(m,640,640)
for i,az in enumerate([0,90,180,270]):
 cam=mujoco.MjvCamera();cam.lookat[:]=d.xpos[b]+[0,0,.012];cam.distance=.24;cam.azimuth=az;cam.elevation=-45;ren.update_scene(d,camera=cam);cv2.imwrite(str(out/f'view-{i}.png'),cv2.cvtColor(ren.render(),cv2.COLOR_RGB2BGR))
ren.close();print(report)
