"""Offline left Dex3 handle pinch fitting in an isolated copy of actual hand geometry."""
import json,shutil,copy,sys,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np,mujoco
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'g1-cup-grasp/scripts'))
from g1_sim import HAND_JOINTS
out=Path('results/spoon-pinch-fit-005');out.mkdir(exist_ok=False);shutil.copy2(__file__,out/Path(__file__).name);source=Path('results/lid-place-003');r=json.loads((source/'report.json').read_text());assert r['pass'];full=mujoco.MjModel.from_xml_path(r['scene']);fd=mujoco.MjData(full);mujoco.mj_setState(full,fd,np.load(source/'continuation.npz')['integration'],mujoco.mjtState.mjSTATE_INTEGRATION);mujoco.mj_forward(full,fd);flat=out/'full-model.xml';mujoco.mj_saveLastXML(str(flat),full);tree=ET.parse(flat);root=ET.Element('mujoco',model='offline left hand fit')
for tag in ['compiler','option','default','asset']:
 for node in tree.getroot().findall(tag):root.append(copy.deepcopy(node))
world=ET.SubElement(root,'worldbody');hand=copy.deepcopy(tree.find('.//body[@name="left_wrist_yaw_link"]'));hand.set('pos','0 0 0');hand.set('quat','1 0 0 0')
for j in hand.findall('joint'):hand.remove(j)
ET.SubElement(hand,'freejoint',name='hand_pose');world.append(hand);spoon=copy.deepcopy(tree.find('.//body[@name="scoop"]'));b=full.body('scoop').id;spoon.set('pos',' '.join(map(str,fd.xpos[b])));spoon.set('quat',' '.join(map(str,fd.xquat[b])));world.append(spoon);world.append(copy.deepcopy(tree.find('.//body[@name="mesa"]')))
body_names={n.get('name') for n in hand.iter('body')};contact=ET.SubElement(root,'contact')
for e in tree.findall('.//contact/exclude'):
 if e.get('body1') in body_names and e.get('body2') in body_names:contact.append(copy.deepcopy(e))
scene=(out/'hand-scene.xml').resolve();ET.ElementTree(root).write(scene);m=mujoco.MjModel.from_xml_path(str(scene));d=mujoco.MjData(m);mujoco.mj_forward(m,d);base=d.qpos.copy();names=[n.replace('right_','left_') for n in HAND_JOINTS];ha=m.jnt_qposadr[[m.joint(n).id for n in names]];left=m.body('left_wrist_yaw_link').id;obj=m.body('scoop').id;table=m.geom('tampo').id;hg=[g for g in range(m.ngeom) if m.geom_contype[g] and m.body(int(m.geom_bodyid[g])).name.startswith('left_')];og=[g for g in range(m.ngeom) if m.geom_contype[g] and m.geom_bodyid[g]==obj];target_geoms=[g for g in og if .040<m.geom_pos[g,0]<.059];tips=[next(g for g in hg if m.body(int(m.geom_bodyid[g])).name=='left_hand_'+n+'_link') for n in ['thumb_2','index_1']];excluded={frozenset([e.get('body1'),e.get('body2')]) for e in contact};pairs=[]
for g in hg:
 for h in hg:
  if g>=h:continue
  bg,bh=map(int,[m.geom_bodyid[g],m.geom_bodyid[h]]);ng,nh=m.body(bg).name,m.body(bh).name
  if bg==bh or m.body_parentid[bg]==bh or m.body_parentid[bh]==bg or frozenset([ng,nh]) in excluded:continue
  if ng.startswith('left_hand') and nh.startswith('left_hand') and ng.split('_')[2]==nh.split('_')[2]:continue
  pairs.append((g,h))
hand0=np.array([.05,.48,1.15,-.95,-1.30,-1.5,-1.7]);jlim=m.jnt_range[[m.joint(n).id for n in names[:5]]];lo=np.r_[[-.02,.24,.74],[-.6,.25,-.6],jlim[:,0]];hi=np.r_[[.14,.42,1.0],[.6,1.25,.6],jlim[:,1]];records=[]
def evaluate(x,detail=False):
 d.qpos[:]=base;d.qpos[:3]=x[:3];mujoco.mju_mat2Quat(d.qpos[3:7],Rotation.from_rotvec(x[3:6]).as_matrix().ravel());d.qpos[ha]=hand0;d.qpos[ha[:5]]=x[6:];mujoco.mj_forward(m,d);dist=[];points=[];directions=[]
 for g in tips:
  options=[]
  for h in target_geoms:
   pts=np.zeros(6);gap=mujoco.mj_geomDistance(m,d,g,h,.5,pts);options.append((gap,pts))
  gap,pts=min(options,key=lambda a:a[0]);dist.append(gap);points.append((pts[3:]-d.xpos[obj])@d.xmat[obj].reshape(3,3));n=(pts[3:]-pts[:3])*np.sign(gap);directions.append(n/max(np.linalg.norm(n),1e-12))
 tablegaps=[mujoco.mj_geomDistance(m,d,g,table,.1,None) for g in hg];selfgaps=[mujoco.mj_geomDistance(m,d,g,h,.01,None) for g,h in pairs];objgaps=[mujoco.mj_geomDistance(m,d,g,h,.003,None) for g in hg for h in og];opp=float(np.dot(*directions));loc=[max(0,.043-p[0])+max(0,p[0]-.058) for p in points]
 if detail:return dist,points,min(tablegaps),min(selfgaps),min(objgaps),opp
 return np.r_[(np.array(dist)-.001)*1000,np.array(loc)*1000,np.minimum(0,np.array(tablegaps)-.010)*3000,np.minimum(0,np.array(selfgaps)-.001)*3000,np.minimum(0,np.array(objgaps))*3000,max(0,opp+.65)*10,(x[6:]-hand0[:5])*.05,x[3:6]*.02]
for seed in range(6):
 known=json.loads(Path('results/spoon-pinch-fit-003/report.json').read_text())['results'][4];oldR=np.array(known['palm_R']);objpt=d.xpos[obj]+d.xmat[obj].reshape(3,3)@np.array([.052,0,.0055]);oldlocal=oldR.T@(objpt-np.array(known['palm_goal_m']));newR=Rotation.from_rotvec([0,.30+.08*seed,0]).as_matrix()@oldR;newpos=objpt-newR@oldlocal;x=np.r_[newpos,Rotation.from_matrix(newR).as_rotvec(),known['hand'][:5]];fit=least_squares(evaluate,np.clip(x,lo+1e-9,hi-1e-9),bounds=(lo,hi),max_nfev=300,diff_step=1e-4);dist,points,tgap,sgap,ogap,opp=evaluate(fit.x,True);passed=max(abs(np.array(dist)-.001))<.001 and tgap>.0099 and sgap>.0008 and ogap>-.0001 and opp<-.5 and all(.042<p[0]<.059 for p in points);row={'seed':seed,'pass':bool(passed),'palm_goal_m':fit.x[:3].tolist(),'palm_R':Rotation.from_rotvec(fit.x[3:6]).as_matrix().tolist(),'hand':d.qpos[ha].tolist(),'contact_gap_m':dist,'object_contact_local':[p.tolist() for p in points],'table_clearance_m':float(tgap),'self_gap_m':float(sgap),'object_min_gap_m':float(ogap),'normal_dot':opp,'cost':float(fit.cost)};records.append(row);print(row,flush=True)
(out/'report.json').write_text(json.dumps({'model':'gpt-6-astra','source':str(source),'scene':r['scene'],'isolated_scene':str(scene),'scope':'offline hand geometry only; no whole robot reach or physical grasp validated','hand_names':names,'results':records},indent=2))
