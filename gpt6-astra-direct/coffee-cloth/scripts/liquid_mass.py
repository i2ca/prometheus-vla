"""Quasi-static water mass/inertia coupling, with no pose or velocity edits.

MuJoCo model mass changes require mj_setConst. It runs on separate scratch data;
the live integrator state is left intact. Jet impact/momentum transfer and slosh
are not simulated by this reduced model.
Reference: https://mujoco.readthedocs.io/en/latest/programming/simulation.html
"""
import numpy as np
import mujoco


class LiquidMassCoupler:
    def __init__(self, model, dry_kettle_kg=.78):
        self.m=model;self.scratch=mujoco.MjData(model);self.dry={};self.original_bvh={}
        for name in ['chaleira','coador','copo']:
            b=model.body(name).id;mass=float(model.body_mass[b]);rotation=np.zeros(9);mujoco.mju_quat2Mat(rotation,model.body_iquat[b]);rotation=rotation.reshape(3,3)
            inertia=rotation@np.diag(model.body_inertia[b])@rotation.T
            start=int(model.body_bvhadr[b]);count=int(model.body_bvhnum[b]);self.original_bvh[b]=(start,model.bvh_aabb[start:start+count].copy(),model.body_ipos[b].copy(),rotation.copy())
            if name=='chaleira':inertia*=dry_kettle_kg/mass;mass=dry_kettle_kg
            self.dry[b]=(mass,model.body_ipos[b].copy(),inertia)

    @staticmethod
    def point_moments(points, mass, fraction, rotation):
        if mass<=0:return None
        heights=points@rotation[2];level=np.quantile(heights,np.clip(fraction,0,1));wet=points[heights<=level]
        center=wet.mean(axis=0);x=wet-center;cov=x.T@x/len(x);inertia=mass*(np.trace(cov)*np.eye(3)-cov)
        return mass,center,inertia

    def apply(self, data, water):
        m=self.m;extra={b:[] for b in self.dry}
        jar=m.body('chaleira').id;cup=m.body('copo').id;filt=m.body('coador').id
        extra[jar].append(self.point_moments(water.points,water.source_ml/1000,water.source_ml/water.capacity_ml,data.xmat[jar].reshape(3,3)))
        extra[cup].append(self.point_moments(water.cup_points,water.receiver_ml/1000,water.receiver_ml/water.cup_capacity_ml,data.xmat[cup].reshape(3,3)))
        if water.filter_ml>0:
            mass=water.filter_ml/1000;h=.07*(water.filter_ml/water.filter_capacity_ml)**(1/3);radius=.035*h/.07
            extra[filt].append((mass,np.array([0,0,.135+.75*h]),np.diag([mass*(3*radius**2/20+3*h*h/80)]*2+[mass*3*radius**2/10])))
        retained=(water.cloth_retained_ml+water.coffee_retained_ml+water.coffee_g)/1000
        if retained>0:extra[filt].append((retained,np.array([0.,0.,.15]),np.eye(3)*retained*.02**2/3))
        for b,items in extra.items():
            parts=[self.dry[b]]+[x for x in items if x is not None];mass=sum(x[0] for x in parts);center=sum(w*p for w,p,_ in parts)/mass;inertia=np.zeros((3,3))
            for w,p,I in parts:
                delta=p-center;inertia+=I+w*(np.dot(delta,delta)*np.eye(3)-np.outer(delta,delta))
            values,axes=np.linalg.eigh(inertia)
            if np.linalg.det(axes)<0:axes[:,0]*=-1
            m.body_mass[b]=mass;m.body_ipos[b]=center;m.body_inertia[b]=values;mujoco.mju_mat2Quat(m.body_iquat[b],axes.ravel())
            # Body BVH boxes are expressed in the inertial frame. mj_setConst
            # does not rebuild them; transform ORIGINAL boxes conservatively.
            start,boxes,old_center,old_axes=self.original_bvh[b]
            m.bvh_aabb[start:start+len(boxes),:3]=(boxes[:,:3]@old_axes.T+old_center-center)@axes
            m.bvh_aabb[start:start+len(boxes),3:]=boxes[:,3:]@np.abs(axes.T@old_axes).T+1e-12
        mujoco.mj_setConst(m,self.scratch)
        mujoco.mj_forward(m,data)
