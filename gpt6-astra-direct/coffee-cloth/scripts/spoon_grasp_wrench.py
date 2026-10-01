"""Allocate fingertip MOTOR forces; never apply a wrench to the object.

Point forces alone give at most five independent wrench directions. Two axial
contact torques use the existing condim=4 torsional friction (5 mm coefficient).
They too are requested through finger motors; no external object torque is set.
Its forces are bounded and projected into an assumed mu=0.8 friction cone.
Actual MuJoCo contacts, motor limits and grasp acceptance remain authoritative.
"""
import numpy as np


def cross_matrix(v):
    x, y, z = v
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def allocate(points, normals, com, mass, gravity, torque, squeeze):
    points = np.asarray(points)
    normals = np.asarray(normals)
    axis = points[1] - points[0]
    axis /= max(np.linalg.norm(axis), 1e-9)
    baseline = np.r_[axis * squeeze, -axis * squeeze, 0., 0.]
    lever = .02
    A = np.vstack([np.hstack([np.eye(3), np.eye(3), np.zeros((3, 2))]),
                   np.hstack([*(cross_matrix(p - com) / lever for p in points), normals.T])])
    desired = np.r_[-mass * np.asarray(gravity), torque / lever]
    correction = np.linalg.lstsq(
        np.vstack([A, np.eye(8) * .03]),
        np.r_[desired - A @ baseline, np.zeros(8)], rcond=None)[0]
    solution = baseline + correction
    forces = solution[:6].reshape(2, 3)
    torques = solution[6:] * lever
    for i, n in enumerate(normals):
        fn = float(np.clip(forces[i] @ n, squeeze, squeeze * 1.5))
        tangent = forces[i] - (forces[i] @ n) * n
        tangent *= min(1., .8 * fn / max(np.linalg.norm(tangent), 1e-9))
        forces[i] = fn * n + tangent
        torques[i] = np.clip(torques[i], -.005 * fn, .005 * fn)
    return forces, torques
