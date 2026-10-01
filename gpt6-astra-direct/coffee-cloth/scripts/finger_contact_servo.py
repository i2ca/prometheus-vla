"""Small finger POSITION-command increments from live contact geometry.

No object state or external object force is written. The index's two joints are
solved together when its proximal and distal links provide two supports, so one
support does not close by unintentionally moving the other away.
"""
import numpy as np
import mujoco


def _bounded_delta(jacobian, desired, rate=1.):
    J=np.atleast_2d(jacobian);desired=np.atleast_1d(desired)
    delta=J.T@np.linalg.solve(J@J.T+np.eye(len(desired))*1e-8,desired)
    return delta*min(1.,rate/max(float(np.max(np.abs(delta))),1e-9))


def advance(model,data,hand_joints,tip_bodies,contacts,forces,reference,offset,
            third_link='index_0',tip_self_force=0.,force_targets=None,avoid_self_contact=False,axial_axis=None,transverse_axis=None,object_rotation_error=None):
    """contacts: three (gap, finger_point_world, inward_normal_world) tuples."""
    if len(contacts) not in (2,3):return offset.copy()
    dofs=model.jnt_dofadr[hand_joints];gradients=[];steps=[];point_jacobians=[];rotation_jacobians=[]
    for body,(gap,point,normal),force,target in zip(
            tip_bodies,contacts,forces,(force_targets if force_targets is not None else ([.30,.30] if len(contacts)==2 else [.30,.15,.15]))):
        jac=np.zeros((3,model.nv));rot=np.zeros_like(jac)
        mujoco.mj_jac(model,data,jac,rot,point,int(body))
        gradients.append(normal@jac[:,dofs]);point_jacobians.append(jac[:,dofs]);rotation_jacobians.append(rot[:,dofs])
        steps.append(float(np.clip((max(gap,0.)*.15 if force<.01 else 0.)+(target-force)*.00008,
                                   -.00008,.00008)))
    if len(contacts)==3 and tip_self_force>.1:steps[0]=-.00005
    delta=np.zeros(7)
    delta[:3]=_bounded_delta(gradients[0][:3],[steps[0]])
    if len(contacts)==2:
        # Preserve separation of the finger meshes while regulating the object contacts.
        rows=[gradients[0][:5],gradients[1][:5]];desired=list(steps)
        for direction in [axial_axis,transverse_axis]:
            if direction is None:continue
            axis=np.asarray(direction);error=float(axis@(contacts[0][1]-contacts[1][1]))
            rows.append((axis@(point_jacobians[0]-point_jacobians[1]))[:5]);desired.append(float(np.clip(-.15*error,-.00005,.00005)))
        if object_rotation_error is not None:
            # Request small physical fingertip motions that counter measured object
            # rotation. Tangential point motion plus contact-normal twist closes
            # the two-point grasp's missing rotational direction through motors.
            center=(contacts[0][1]+contacts[1][1])*.5
            err=np.asarray(object_rotation_error)
            rows=[4*r for r in rows];desired=[4*v for v in desired]
            for i,(_,point,n) in enumerate(contacts):
                tangent=np.eye(3)-np.outer(n,n)
                wanted=.01*np.cross(err,point-center)
                rows.extend((tangent@point_jacobians[i])[:,:5])
                desired.extend(np.clip(tangent@wanted,-.00003,.00003))
                rows.append((n@rotation_jacobians[i])[:5]*.01)
                desired.append(float(np.clip(.0001*(n@err),-.00003,.00003)))
        geoms=[[g for g in range(model.ngeom) if model.geom_contype[g] and model.geom_bodyid[g]==body] for body in tip_bodies]
        choices=[]
        for g in geoms[0]:
            for h in geoms[1]:
                points=np.zeros(6);gap=mujoco.mj_geomDistance(model,data,g,h,.003,points);choices.append((gap,points))
        if choices and (avoid_self_contact or tip_self_force>.3):
            gap,points=min(choices,key=lambda x:x[0])
            if gap<(.0005 if avoid_self_contact else .003):
                n=(points[3:]-points[:3])*np.sign(gap);n/=max(np.linalg.norm(n),1e-9)
                j1=np.zeros((3,model.nv));j2=np.zeros_like(j1);rot=np.zeros_like(j1)
                mujoco.mj_jac(model,data,j1,rot,points[:3],int(tip_bodies[0]))
                mujoco.mj_jac(model,data,j2,rot,points[3:],int(tip_bodies[1]))
                rows.append((n@(j2-j1)[:,dofs])[:5]);desired.append(float(np.clip((.0002-gap)*.2 if avoid_self_contact else max(0,tip_self_force-.5)*.00008,0,.00008)))
        delta[:5]=_bounded_delta(np.stack(rows),desired)
    elif third_link=='index_0':
        delta[3:5]=_bounded_delta(np.stack([gradients[1][3:5],
                                           gradients[2][3:5]]),steps[1:])
    else:
        delta[3:5]=_bounded_delta(gradients[1][3:5],[steps[1]])
        delta[5:7]=_bounded_delta(gradients[2][5:7],[steps[2]])
    # Apply one common rate scale: preserve simultaneous closure of all supports.
    delta*=min(1.,.001/max(float(np.max(np.abs(delta))),1e-9))
    limits=model.jnt_range[hand_joints]
    target=np.clip(reference+offset+delta,limits[:,0],limits[:,1])
    return target-reference
