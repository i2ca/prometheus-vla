"""Analytic derivatives of the existing carry IK objective.

Uses MuJoCo point/angular Jacobians. Active anatomical penalties retain finite
differences; no physical time step, controller gain, or constraint is changed.
"""
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


def skew(v):
    x, y, z = v
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def rotation_log_derivative(phi):
    theta = np.linalg.norm(phi)
    K = skew(phi)
    coefficient = 1/12 if theta < 1e-4 else (1-(theta/2)/np.tan(theta/2))/(theta*theta)
    return np.eye(3)-.5*K+coefficient*(K@K)


def make_jacobian(ik, local, local_R, target_R, other_body, orientation_weight,
                  avoid, anatomy):
    m, d = ik.m, ik.d
    dofs = m.jnt_dofadr[[m.joint(n).id for n in ik.joint_names]]
    palm = ik.body
    n = len(dofs)

    def point_jac(point, body):
        jp = np.zeros((3, m.nv))
        jr = np.zeros_like(jp)
        mujoco.mj_jac(m, d, jp, jr, point, int(body))
        return jp[:, dofs], jr[:, dofs]

    def jacobian(x):
        pp, rr = ik.fk(x)
        jp, jr = point_jac(pp+rr@local, palm)
        phi = Rotation.from_matrix(rr@local_R@target_R.T).as_rotvec()
        jo, _ = point_jac(d.xpos[other_body], other_body)
        gaps = []
        for g, h, margin in avoid:
            pts = np.zeros(6)
            distance = mujoco.mj_geomDistance(m, d, g, h, .03, pts)
            row = np.zeros(n)
            if distance < margin and abs(distance) > 1e-10:
                normal = (pts[3:]-pts[:3])*np.sign(distance)
                normal /= max(np.linalg.norm(normal), 1e-12)
                jg, _ = point_jac(pts[:3], m.geom_bodyid[g])
                jh, _ = point_jac(pts[3:], m.geom_bodyid[h])
                row = normal@(jh-jg)*5000
            gaps.append(row)
        anatomical = np.zeros((4, n))
        if np.any(anatomy.penalty(d)):
            eps = 1e-6
            for i in range(n):
                delta = np.zeros(n)
                delta[i] = eps
                ik.fk(x+delta)
                upper = anatomy.penalty(d)
                ik.fk(x-delta)
                lower = anatomy.penalty(d)
                anatomical[:, i] = (upper-lower)/(2*eps)
            ik.fk(x)
        return np.vstack([jp*1000,
                          rotation_log_derivative(phi)@jr*orientation_weight,
                          jo*100, np.eye(n)*2, np.eye(n)*.2,
                          np.array(gaps).reshape(-1, n), anatomical])
    return jacobian
