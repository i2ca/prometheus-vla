"""Live geometry to joint position targets; object poses are copied ONLY to scratch IK."""
import numpy as np,mujoco
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

class SpoonGraspServo:
 def __init__(self,m,d,qa,ha,bounds,tips,third_link,anatomy):
  self.m=m;self.d=mujoco.MjData(m);self.qa=np.r_[qa,ha];self.bounds=np.vstack([bounds,m.jnt_range[[m.joint(n).id for n in ['right_hand_thumb_0_joint','right_hand_thumb_1_joint','right_hand_thumb_2_joint','right_hand_index_0_joint','right_hand_index_1_joint','right_hand_middle_0_joint','right_hand_middle_1_joint']]]])
  self.tips=tips;self.anatomy=anatomy;self.spoon=m.body('scoop').id;self.palm=m.body('right_wrist_yaw_link').id;self.other=m.body('left_wrist_yaw_link').id;self.table=m.geom('tampo').id
  self.reference=d.qpos[self.qa].copy();self.p=d.xpos[self.palm].copy();self.R=d.xmat[self.palm].reshape(3,3).copy();self.op=d.xpos[self.other].copy()
  self.hand=[g for g in range(m.ngeom) if m.geom_contype[g] and m.body(int(m.geom_bodyid[g])).name.startswith(('right_hand','right_wrist'))]
  self.shaft=[[h for h in range(m.ngeom) if m.geom_contype[h] and m.geom_bodyid[h]==self.spoon and ((.020<m.geom_pos[h,0]<.038) if i==2 else (.040<m.geom_pos[h,0]<.059))] for i in range(3)]
  self.objadr=m.jnt_qposadr[m.body_jntadr[self.spoon]];self.object_reference=d.qpos[self.objadr:self.objadr+7].copy()
  self.pairs=[]
  for g in self.hand:
   for h in self.hand:
    bg,bh=int(m.geom_bodyid[g]),int(m.geom_bodyid[h])
    if g<h and bg!=bh and m.body_parentid[bg]!=bh and m.body_parentid[bh]!=bg:self.pairs.append((g,h))
  self.anchors=[];self.initial_gaps=[]
  for i,g in enumerate(tips):
   opts=[]
   for h in self.shaft[i]:
    pts=np.zeros(6);gap=mujoco.mj_geomDistance(m,d,g,h,.04,pts);opts.append((gap,pts))
   gap,pts=min(opts,key=lambda x:x[0]);self.initial_gaps.append(gap);self.anchors.append(d.xmat[self.spoon].reshape(3,3).T@(pts[3:]-d.xpos[self.spoon]))
 def advance(self,live,target,elapsed):
  m,z=self.m,self.d;z.qpos[:]=live.qpos;z.qpos[self.objadr:self.objadr+7]=self.object_reference;z.qvel[:]=0;prior=target[self.qa].copy();desired=np.maximum(-.0002,np.array(self.initial_gaps)*(1-elapsed/5))
  low=np.maximum(self.bounds[:,0],prior-.004);high=np.minimum(self.bounds[:,1],prior+.004)
  def fun(q):
   z.qpos[self.qa]=q;mujoco.mj_kinematics(m,z);mujoco.mj_comPos(m,z);terms=[];points=[];normals=[]
   for i,g in enumerate(self.tips):
    options=[]
    for h in self.shaft[i]:
     pts=np.zeros(6);gap=mujoco.mj_geomDistance(m,z,g,h,.04,pts);options.append((gap,pts))
    gap,pts=min(options,key=lambda x:x[0]);local=z.xmat[self.spoon].reshape(3,3).T@(pts[3:]-z.xpos[self.spoon]);terms.extend([(gap-desired[i])*2000]);terms.extend((local-self.anchors[i])*150);points.append(pts[3:].copy());nn=(pts[3:]-pts[:3])*np.sign(gap);normals.append(nn/max(np.linalg.norm(nn),1e-9))
   if len(points)==2:
    line=points[1]-points[0];local_line=z.xmat[self.spoon].reshape(3,3).T@line;unit=line/max(np.linalg.norm(line),1e-9);terms.extend([local_line[0]*2000,max(0,.85-normals[0]@unit)*10,max(0,.85+normals[1]@unit)*10])
   gaps=[min(0,mujoco.mj_geomDistance(m,z,g,h,.005,None)-.0005) for g,h in self.pairs]
   table=[min(0,mujoco.mj_geomDistance(m,z,g,self.table,.02,None)-.012) for g in self.hand]
   return np.r_[terms,np.array(gaps)*10000,np.array(table)*5000,(z.xpos[self.palm]-self.p)*30,Rotation.from_matrix(z.xmat[self.palm].reshape(3,3)@self.R.T).as_rotvec()*20,(z.xpos[self.other]-self.op)*500,(q-prior)*.2,(q-self.reference)*.02,self.anatomy.penalty(z)]
  fit=least_squares(fun,np.clip(prior,low+1e-10,high-1e-10),bounds=(low,high),max_nfev=12)
  return fit.x
