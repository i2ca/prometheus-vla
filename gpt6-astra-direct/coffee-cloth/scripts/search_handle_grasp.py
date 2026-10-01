"""Bounded detached-hand static fit to handle; never an execution success claim."""
import argparse,copy,json,sys,shutil
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np,mujoco,cv2
from scipy.optimize import least_squares

ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
a.out.mkdir(parents=True,exist_ok=False);shutil.copy2(__file__,a.out/Path(__file__).name)
root=Path(__file__).resolve().parents[1]
robot=ET.parse(root/'results/kettle-scene-20260919-001/robot.xml').getroot()
scene=ET.parse(root/'scene/setup-luiz-handle-v2.xml').getroot()
xml=ET.Element('mujoco');xml.append(copy.deepcopy(robot.find('compiler')));xml.append(copy.deepcopy(robot.find('default')))
assets=ET.SubElement(xml,'asset')
for el in robot.find('asset'):assets.append(copy.deepcopy(el))
for el in scene.find('asset'):
    if el.get('name','').startswith(('chaleira','handle_col')):assets.append(copy.deepcopy(el))
world=ET.SubElement(xml,'worldbody')
hand=copy.deepcopy(robot.find(".//body[@name='right_wrist_yaw_link']"));hand.set('pos','0 0 0');hand.set('quat','1 0 0 0');hand.set('mocap','true')
for el in list(hand):
    if el.tag in ('joint','camera'):hand.remove(el)
world.append(hand)
kettle=copy.deepcopy(scene.find(".//body[@name='chaleira']"));kettle.set('pos','0 0 0');kettle.set('quat','1 0 0 0')
for el in list(kettle):
    if el.tag in ('joint','freejoint'):kettle.remove(el)
world.append(kettle);ET.SubElement(world,'light',pos='.2 -.5 .5',diffuse='.8 .8 .8')
ET.ElementTree(xml).write(a.out/'detached-hand.xml')
m=mujoco.MjModel.from_xml_path(str(a.out/'detached-hand.xml'));d=mujoco.MjData(m)
names=['right_hand_'+s+'_joint' for s in ['thumb_0','thumb_1','thumb_2','index_0','index_1','middle_0','middle_1']]
qa=[m.jnt_qposadr[m.joint(n).id] for n in names]
handgs=[g for g in range(m.ngeom) if m.geom_contype[g] and m.body(int(m.geom_bodyid[g])).name.startswith(('right_hand','right_wrist'))]
tips=[next(g for g in handgs if m.body(int(m.geom_bodyid[g])).name=='right_hand_'+s+'_link') for s in ['thumb_2','index_1','middle_1']]
handles=[g for g in range(m.ngeom) if (m.geom(g).name or '').startswith('handle_col')]
metal=m.geom('chaleira_hot_body').id
def fingers(s):return np.array([0,-min(.85*s,1.047),-1.2*s,.9*s,1.2*s,.9*s,1.2*s])
def distances(gs):return [min(mujoco.mj_geomDistance(m,d,g,h,.2,None) for h in handles) for g in gs]
results=[]
for yaw in [-120,-90,-60,0,60,90,120,180]:
    R=cv2.Rodrigues(np.array([0,0,np.deg2rad(yaw)]))[0];quat=np.zeros(4);mujoco.mju_mat2Quat(quat,R.ravel());d.mocap_quat[0]=quat
    for z in [.095,.12,.145]:
        for seed in [.6,1.1]:
            d.mocap_pos[0]=0;d.qpos[qa]=fingers(seed);mujoco.mj_forward(m,d)
            center=(d.geom_xpos[tips[0]]+(d.geom_xpos[tips[1]]+d.geom_xpos[tips[2]])/2)/2
            p=np.array([.111,0,z])-center
            def setstate(x):
                d.mocap_pos[0]=[x[0],x[1],p[2]];d.qpos[qa]=fingers(x[2]);mujoco.mj_forward(m,d)
            def residual(x):
                setstate(x)
                tipdist=np.array(distances(tips))
                hot=np.array([mujoco.mj_geomDistance(m,d,g,metal,.2,None) for g in handgs])
                other=np.array(distances(handgs))
                # Three fingertip contacts, metal clearance, no deep handle penetration.
                return np.r_[(tipdist-.0001)*1000,np.minimum(hot-.001,0)*5000,np.minimum(other+.0005,0)*5000]
            sol=least_squares(residual,[p[0],p[1],seed],bounds=([-.35,-.35,.15],[.35,.35,1.4]),max_nfev=70,diff_step=1e-4)
            setstate(sol.x);td=distances(tips);hd=min(mujoco.mj_geomDistance(m,d,g,metal,.2,None) for g in handgs);pen=min(distances(handgs))
            intra=[]
            for c in d.contact:
                if c.geom1 in handgs and c.geom2 in handgs and c.dist<-.0002:intra.append(float(c.dist))
            passed=bool(max(abs(x) for x in td)<.001 and hd>=.0008 and pen>=-.0007 and not intra)
            row={'yaw_deg':yaw,'grip_height_m':z,'seed':seed,'palm_position_kettle_frame_m':d.mocap_pos[0].tolist(),'palm_rotation_kettle_frame':R.tolist(),'hand_joint_names':names,'hand_q':d.qpos[qa].tolist(),'tip_handle_distances_mm':(np.array(td)*1000).tolist(),'hot_clearance_mm':hd*1000,'worst_handle_distance_mm':pen*1000,'intra_hand_penetrations_m':intra,'static_fit_pass':passed,'cost':float(sol.cost)}
            results.append(row)
    print('yaw',yaw,'fits',sum(r['static_fit_pass'] for r in results),flush=True)
results.sort(key=lambda r:(not r['static_fit_pass'],r['cost']))
(a.out/'candidates.json').write_text(json.dumps({'model':'gpt-6-astra','scope':'48 bounded detached-hand fits; full arm reach, approach and load not tested','candidates':results},indent=2))
print(json.dumps(results[0],indent=2))
