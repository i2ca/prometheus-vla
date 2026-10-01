"""Collision-audited offline approaches; never represents physical grasp success."""
import argparse,json,sys,shutil
from pathlib import Path
import numpy as np,mujoco
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'g1-cup-grasp/scripts'))
from g1_sim import G1Sim
from kinematics import ArmIK
ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--initial',type=Path,default=Path('results/arms-down-posture-003/initial-qpos.npy'));a=ap.parse_args()
a.out.mkdir(parents=True,exist_ok=False);shutil.copy2(__file__,a.out/Path(__file__).name)
src=json.loads(Path('results/handle-reach-001/report.json').read_text());sim=G1Sim(src['scene']);m,d=sim.m,sim.d
initial=np.load(a.initial) if a.initial else np.array(src['initial_qpos']);rows=[]
# Keep fingers, wrists and forearms at least 20mm from the table throughout.
table_id=m.geom('tampo').id
arm_clearance_geoms=[g for g in range(m.ngeom) if (m.geom_contype[g] or m.geom_conaffinity[g]) and any(m.body(int(m.geom_bodyid[g])).name.startswith(side+'_'+part) for side in ['left','right'] for part in ['hand','wrist','elbow'])]
TABLE_CLEARANCE_M=.020
def audit():
    mujoco.mj_forward(m,d);bad=set()
    for g in arm_clearance_geoms:
        gap=mujoco.mj_geomDistance(m,d,g,table_id,TABLE_CLEARANCE_M,None)
        if gap<TABLE_CLEARANCE_M:
            bad.add(('table_clearance_under_20mm',m.body(int(m.geom_bodyid[g])).name))
    for c in d.contact:
        if c.dist>=-0.0001:continue
        gs=[m.geom(int(g)).name or '' for g in [c.geom1,c.geom2]]
        bs=[m.body(int(m.geom_bodyid[g])).name or '' for g in [c.geom1,c.geom2]]
        robot=[b.startswith(('right_','left_','torso','pelvis','waist','head')) for b in bs]
        if not any(robot):continue
        if all(robot) or not any(g.startswith('handle_col') for g in gs) or c.dist<-.0007:bad.add(tuple(sorted(gs)))
    return sorted(bad)
for candidate in src['results']:
    if not candidate['endpoint_pass'] or abs(candidate['q'][0])>1:continue
    ik=ArmIK(sim,candidate['joint_names'],palm=candidate['side']+'_wrist_yaw_link');ha=np.array([m.jnt_qposadr[m.joint(n).id] for n in candidate['hand_joint_names']]);hq=np.array(candidate['hand_q']);end=np.array(candidate['q']);p=np.array(candidate['palm_target_m']);R=np.array(candidate['palm_rotation'])
    for direction in [[0,1,0],[0,-1,0],[-.5,1,0],[-.5,-1,0],[0,1,.5],[0,-1,.5]]:
      for factor in [1,.95,.9]:
          d.qpos[:]=initial;d.qpos[ha]=hq*factor;q=end.copy();path=[];failure=None
          for retreat in np.linspace(0,.12,61):
              target=p+np.array(direction)/np.linalg.norm(direction)*retreat
              q,e=ik.solve(target,R,q,reference=end,iterations=160);d.qpos[ik.qa]=q;bad=audit()
              if bad or e['position_error_m']>.0015 or e['orientation_error_rad']>.0175:
                  failure={'retreat_m':float(retreat),'contacts':bad,'errors':e};break
              path.append(d.qpos.copy())
          row={'direction':direction,'candidate_index':candidate['candidate_index'],'factor':factor,'retreat_samples':len(path),'failure':failure,'retreat_pass':failure is None}
          if failure is None:
              # Audit closing, including preshape and final shape at the endpoint.
              closing=[];d.qpos[ik.qa]=end
              for f in np.linspace(factor,1,31):
                  d.qpos[ha]=hq*f;bad=audit()
                  if bad:closing.append({'factor':float(f),'contacts':bad})
              row['closing_failures']=closing
              # Audit direct joint-space folded-to-pregrasp path, without claiming completeness.
              pre=path[-1];bad_first=None
              for t in np.linspace(0,1,201):
                  d.qpos[:]=initial*(1-t)+pre*t;bad=audit()
                  if bad:bad_first={'fraction':float(t),'contacts':bad};break
              row['folded_to_pregrasp_failure']=bad_first
              if not closing and bad_first is None:
                  row['approach_pass']=True
                  full=[initial*(1-t)+pre*t for t in np.linspace(0,1,201)]+list(reversed(path))
                  for f in np.linspace(factor,1,31):
                      state=full[-1].copy();state[ha]=hq*f;full.append(state)
                  fn=f"path-c{candidate['candidate_index']}-f{factor}-d{direction}.npz";np.savez_compressed(a.out/fn,qpos=np.array(full));row['path']=fn
          rows.append(row);print(json.dumps(row),flush=True)
(a.out/'report.json').write_text(json.dumps({'model':'gpt-6-astra','scope':'Offline sampled approach only, tolerance 0.1mm forbidden and 0.7mm handle penetration; no dynamic support tested','scene':src['scene'],'results':rows},indent=2))
# Render best reachable endpoint, even if paths fail.
c=next(r for r in src['results'] if r['endpoint_pass']);d.qpos[:]=initial
for n,v in zip(c['joint_names']+c['hand_joint_names'],c['q']+c['hand_q']):d.qpos[m.jnt_qposadr[m.joint(n).id]]=v
mujoco.mj_forward(m,d);renderer=mujoco.Renderer(m,720,960);cam=mujoco.MjvCamera();cam.lookat[:]=[.15,.1,.90];cam.distance=.65;cam.azimuth=140;cam.elevation=-15
renderer.update_scene(d,camera=cam)
from PIL import Image
Image.fromarray(renderer.render()).save(a.out/'endpoint.png');renderer.close()
