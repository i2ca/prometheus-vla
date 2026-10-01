"""Allocate desired object wrench through current fingertip contacts and motors.

No forces are applied to the object. Returns motor torques. Contact geometry,
object pose and velocity are privileged simulator feedback, not camera estimates.
"""
import numpy as np
import mujoco
from scipy.optimize import linprog


def allocate(model,data,jar_body,base_tau,wrench,friction_margin=.85,min_finger_normal=8.):
    m,d=model,data;joints=m.actuator_trnid[:,0];va=m.jnt_dofadr[joints];W=[];T=[];groups=[];contacts=[]
    for ct in d.contact:
        if ct.dist>=0:continue
        gs=[int(ct.geom1),int(ct.geom2)];bs=[int(m.geom_bodyid[g]) for g in gs]
        if jar_body not in bs:continue
        ji=bs.index(jar_body);other=bs[1-ji];name=m.body(other).name
        if not name.startswith('left_hand') or not (m.geom(gs[ji]).name or '').startswith('handle_col'):continue
        normal=ct.frame[:3].copy()*(1 if ji==1 else -1);t1=ct.frame[3:6].copy();t2=np.cross(normal,t1);point=ct.pos.copy();jp=np.zeros((3,m.nv));jr=np.zeros((3,m.nv));mujoco.mj_jac(m,d,jp,jr,point,other);part=name.split('_')[2];contacts.append({'body':name,'point_m':point.tolist()})
        for spin in [-1,-.7071,0,.7071,1]:
            for angle in np.linspace(0,2*np.pi,12,endpoint=False):
                force=normal+friction_margin*ct.friction[0]*np.sqrt(1-spin**2)*(np.cos(angle)*t1+np.sin(angle)*t2);moment=friction_margin*ct.friction[2]*spin*normal
                W.append(np.r_[force,np.cross(point-d.xipos[jar_body],force)+moment]);T.append(jp[:,va].T@force+jr[:,va].T@moment);groups.append(part)
    if not W:return {'pass':False,'reason':'no handle contacts'}
    W=np.array(W).T;T=np.array(T).T;present=sorted(set(groups));N=np.array([[1. if part==g else 0. for g in groups] for part in present]);A=np.vstack([T,-T,-N]);b=np.r_[m.actuator_ctrlrange[:,1]-base_tau,base_tau-m.actuator_ctrlrange[:,0],np.full(len(present),-min_finger_normal)]
    fit=linprog(np.ones(W.shape[1]),A_ub=A,b_ub=b,A_eq=W,b_eq=wrench,bounds=(0,None),method='highs')
    if not fit.success:return {'pass':False,'reason':fit.message,'contact_count':len(contacts)}
    return {'pass':True,'motor_torque':base_tau+T@fit.x,'contact_normals_N':dict(zip(present,(N@fit.x).tolist())),'wrench_residual':float(np.linalg.norm(W@fit.x-wrench)),'contact_count':len(contacts)}
