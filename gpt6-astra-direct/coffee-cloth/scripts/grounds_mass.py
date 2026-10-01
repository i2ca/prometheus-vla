"""Quasi-static pot/spoon payload coupling, preserving live qpos/qvel.

No discrete grains, pickup drag or transfer momentum. Filter dry-coffee payload
is passed through KettleWater.coffee_g to the existing LiquidMassCoupler.
"""
import mujoco,numpy as np
class GroundsMassCoupler:
 def __init__(self,model):
  self.m=model;self.scratch=mujoco.MjData(model);self.dry={};self.bvh={}
  for name in ['pote','scoop']:
   b=model.body(name).id;R=np.zeros(9);mujoco.mju_quat2Mat(R,model.body_iquat[b]);R=R.reshape(3,3);self.dry[b]=(float(model.body_mass[b]),model.body_ipos[b].copy(),R@np.diag(model.body_inertia[b])@R.T);start=int(model.body_bvhadr[b]);count=int(model.body_bvhnum[b]);self.bvh[b]=(start,model.bvh_aabb[start:start+count].copy(),model.body_ipos[b].copy(),R.copy())
 def apply(self,data,grounds):
  m=self.m;extra={'pote':(grounds.pot_g/1000,np.array([0,0,(grounds.pot_floor_m+grounds.bed_height_m)/2]),np.array([grounds.pot_inner_radius_m]*2+[(grounds.bed_height_m-grounds.pot_floor_m)/2])), 'scoop':(grounds.spoon_g/1000,np.array([-.045,0,.0035]),np.array([.014,.0115,.0025]))}
  for name,(load,cp,radii) in extra.items():
   b=m.body(name).id;dry,center0,I0=self.dry[b];mass=dry+load;center=(dry*center0+load*cp)/mass
   # Ellipsoidal payload approximation; all dimensions/assumptions documented.
   Ix=load/5*np.array([radii[1]**2+radii[2]**2,radii[0]**2+radii[2]**2,radii[0]**2+radii[1]**2]);inertia=I0+np.diag(Ix)
   for w,p in [(dry,center0),(load,cp)]:delta=p-center;inertia+=w*(np.dot(delta,delta)*np.eye(3)-np.outer(delta,delta))
   values,axes=np.linalg.eigh(inertia)
   if np.linalg.det(axes)<0:axes[:,0]*=-1
   m.body_mass[b]=mass;m.body_ipos[b]=center;m.body_inertia[b]=values;mujoco.mju_mat2Quat(m.body_iquat[b],axes.ravel());start,boxes,oldcenter,oldaxes=self.bvh[b];m.bvh_aabb[start:start+len(boxes),:3]=(boxes[:,:3]@oldaxes.T+oldcenter-center)@axes;m.bvh_aabb[start:start+len(boxes),3:]=boxes[:,3:]@np.abs(axes.T@oldaxes).T+1e-12
  mujoco.mj_setConst(m,self.scratch);mujoco.mj_forward(m,data)
