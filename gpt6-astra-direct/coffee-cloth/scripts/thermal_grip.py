"""Motor-reference adjustment to keep finger surfaces clear of hot kettle metal.

Projects the distance gradient into the nearest handle-contact tangent direction.
This is not a contact bypass: execution still rejects every actual metal contact.
"""
import numpy as np
import mujoco


def adjust_fingers(model,data,hand_names,reference,origin,clearance=.005):
    m,d=model,data;hot=m.geom('chaleira_hot_body').id;handles=[g for g in range(m.ngeom) if (m.geom(g).name or '').startswith('handle_col')];updates=[]
    for part in ['index','middle','thumb']:
        indices=[i for i,n in enumerate(hand_names) if n.startswith('left_hand_'+part) and not n.endswith('thumb_0_joint')]
        joints=[m.joint(hand_names[i]).id for i in indices];va=m.jnt_dofadr[joints]
        geoms=[g for g in range(m.ngeom) if m.geom_contype[g] and m.body(int(m.geom_bodyid[g])).name.startswith('left_hand_'+part)]
        nearest=None
        for g in geoms:
            points=np.zeros(6);gap=mujoco.mj_geomDistance(m,d,g,hot,.02,points)
            if nearest is None or gap<nearest[0]:nearest=(gap,g,points)
        gap,g,points=nearest
        if gap>=clearance:continue
        n=(points[3:]-points[:3])*np.sign(gap);n/=max(1e-12,np.linalg.norm(n));jp=np.zeros((3,m.nv));jr=np.zeros((3,m.nv));mujoco.mj_jac(m,d,jp,jr,points[:3],int(m.geom_bodyid[g]));hot_gradient=-n@jp[:,va]
        # Preserve the closest plastic contact of this finger, including both phalanges.
        contact=None
        for fg in geoms:
            for h in handles:
                p=np.zeros(6);dist=mujoco.mj_geomDistance(m,d,fg,h,.02,p)
                if contact is None or dist<contact[0]:contact=(dist,fg,p)
        dist,fg,p=contact;n=(p[3:]-p[:3])*np.sign(dist);n/=max(1e-12,np.linalg.norm(n));mujoco.mj_jac(m,d,jp,jr,p[:3],int(m.geom_bodyid[fg]));handle_gradient=-n@jp[:,va]
        projector=np.eye(len(indices))-np.outer(handle_gradient,handle_gradient)/(np.dot(handle_gradient,handle_gradient)+1e-12);direction=projector@hot_gradient
        if np.dot(direction,direction)<1e-10:continue
        delta=direction*min(.0001,.2*(clearance-gap))/(np.dot(direction,direction)+1e-10);delta=np.clip(delta,-.003,.003)
        for idx,j,change in zip(indices,joints,delta):reference[idx]=np.clip(reference[idx]+change,max(m.jnt_range[j,0],origin[idx]-.25),min(m.jnt_range[j,1],origin[idx]+.25))
        updates.append({'finger':part,'hot_gap_m':float(gap),'reference_change_rad':delta.tolist()})
    return updates


def thermal_motor_torque(model,data,motor_dofs,clearance=.010,stiffness=4000.,include_arm=False):
    """Repulsive motor torque from measured gap; never an external object force.

    Uses only finger motors, with caller enforcing actuator torque saturation.
    The actual metal-contact rejection is unchanged.
    """
    m,d=model,data;hot=m.geom('chaleira_hot_body').id;torque=np.zeros(len(motor_dofs));minimum=.1
    for part in ['index','middle','thumb']:
        nearest=None
        for g in range(m.ngeom):
            if not m.geom_contype[g] or not m.body(int(m.geom_bodyid[g])).name.startswith('left_hand_'+part):continue
            p=np.zeros(6);gap=mujoco.mj_geomDistance(m,d,g,hot,.1,p)
            if nearest is None or gap<nearest[0]:nearest=(gap,g,p)
        gap,g,p=nearest;minimum=min(minimum,gap)
        if gap>=clearance:continue
        n=(p[3:]-p[:3])*np.sign(gap);n/=max(np.linalg.norm(n),1e-12)
        jp=np.zeros((3,m.nv));jr=np.zeros((3,m.nv));mujoco.mj_jac(m,d,jp,jr,p[:3],int(m.geom_bodyid[g]))
        mask=np.array([m.joint(m.actuator_trnid[i,0]).name.startswith('left_hand_'+part) or (include_arm and m.joint(m.actuator_trnid[i,0]).name.startswith(('left_shoulder','left_elbow','left_wrist'))) for i in range(m.nu)])
        torque+=(jp[:,motor_dofs].T@(-n*min(40.,stiffness*(clearance-gap))))*mask
    return torque,minimum
