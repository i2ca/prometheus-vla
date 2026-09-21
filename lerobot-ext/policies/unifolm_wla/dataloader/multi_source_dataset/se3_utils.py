import numpy as np
from scipy.spatial.transform import Rotation


def rpy_to_matrix(rpy: np.ndarray, order: str = "xyz") -> np.ndarray:
    """Convert RPY angles to rotation matrix.

    Args:
        rpy: (..., 3) RPY angles in radians
        order: Euler angle convention, e.g. "xyz"
    Returns:
        (..., 3, 3) rotation matrices
    """
    shape = rpy.shape[:-1]
    rpy_flat = rpy.reshape(-1, 3)
    R = Rotation.from_euler(order, rpy_flat).as_matrix()
    return R.reshape(*shape, 3, 3)


def quat_to_matrix(quat: np.ndarray, order: str = "xyzw") -> np.ndarray:
    """Convert quaternion to rotation matrix.

    Args:
        quat: (..., 4) quaternions
        order: quaternion component order, "xyzw" or "wxyz"
    Returns:
        (..., 3, 3) rotation matrices
    """
    shape = quat.shape[:-1]
    quat_flat = quat.reshape(-1, 4)
    if order == "wxyz":
        quat_flat = quat_flat[:, [1, 2, 3, 0]]  # wxyz -> xyzw for scipy
    R = Rotation.from_quat(quat_flat).as_matrix()
    return R.reshape(*shape, 3, 3)


def rotvec_to_matrix(rotvec: np.ndarray) -> np.ndarray:
    """Convert rotation vector (axis-angle) to rotation matrix.

    Args:
        rotvec: (..., 3) rotation vectors
    Returns:
        (..., 3, 3) rotation matrices
    """
    shape = rotvec.shape[:-1]
    rotvec_flat = rotvec.reshape(-1, 3)
    R = Rotation.from_rotvec(rotvec_flat).as_matrix()
    return R.reshape(*shape, 3, 3)


def matrix_to_rotvec(R: np.ndarray) -> np.ndarray:
    """Convert rotation matrix to rotation vector (axis-angle).

    Args:
        R: (..., 3, 3) rotation matrices
    Returns:
        (..., 3) rotation vectors
    """
    shape = R.shape[:-2]
    R_flat = R.reshape(-1, 3, 3)
    rotvec = Rotation.from_matrix(R_flat).as_rotvec()
    return rotvec.reshape(*shape, 3)


def matrix_to_rot6d(R: np.ndarray) -> np.ndarray:
    """Convert rotation matrix to 6D representation.

    Uses the first two matrix columns in column-major order:
    [R00, R10, R20, R01, R11, R21].
    """
    first_two_cols = R[..., :, :2]
    return np.swapaxes(first_two_cols, -1, -2).reshape(*R.shape[:-2], 6)


def pose_to_se3(xyz: np.ndarray, rot_matrix: np.ndarray) -> np.ndarray:
    """Build SE(3) homogeneous matrix from position and rotation.

    Args:
        xyz: (..., 3) positions
        rot_matrix: (..., 3, 3) rotation matrices
    Returns:
        (..., 4, 4) SE(3) matrices
    """
    shape = xyz.shape[:-1]
    T = np.zeros((*shape, 4, 4), dtype=np.float64)
    T[..., :3, :3] = rot_matrix
    T[..., :3, 3] = xyz
    T[..., 3, 3] = 1.0
    return T


def se3_inverse(T: np.ndarray) -> np.ndarray:
    """Compute inverse of SE(3) matrix efficiently.

    For SE(3): T_inv = [[R^T, -R^T @ t], [0, 1]]

    Args:
        T: (..., 4, 4) SE(3) matrices
    Returns:
        (..., 4, 4) inverse SE(3) matrices
    """
    R = T[..., :3, :3]
    t = T[..., :3, 3:]
    R_T = np.swapaxes(R, -2, -1)

    shape = T.shape[:-2]
    T_inv = np.zeros((*shape, 4, 4), dtype=T.dtype)
    T_inv[..., :3, :3] = R_T
    T_inv[..., :3, 3:] = -R_T @ t
    T_inv[..., 3, 3] = 1.0
    return T_inv


def se3_to_xyz_rotvec(T: np.ndarray) -> np.ndarray:
    """Extract xyz + rotvec from SE(3) matrix.

    Args:
        T: (..., 4, 4) SE(3) matrices
    Returns:
        (..., 6) [x, y, z, rx, ry, rz]
    """
    xyz = T[..., :3, 3]
    rotvec = matrix_to_rotvec(T[..., :3, :3])
    return np.concatenate([xyz, rotvec], axis=-1)


def pose_to_se3_from_format(pose: np.ndarray, ee_format: str) -> np.ndarray:
    """Convert a pose vector to SE(3) matrix based on format.

    Args:
        pose: (..., 6) for xyz_rpy/xyz_rvec or (..., 7) for xyz_quat
        ee_format: "xyz_rpy", "xyz_quat", or "xyz_rvec"
    Returns:
        (..., 4, 4) SE(3) matrices
    """
    xyz = pose[..., :3]
    if ee_format == "xyz_rpy":
        R = rpy_to_matrix(pose[..., 3:6])
    elif ee_format == "xyz_quat":
        R = quat_to_matrix(pose[..., 3:7])
    elif ee_format == "xyz_rvec":
        R = rotvec_to_matrix(pose[..., 3:6])
    else:
        raise ValueError(f"Unknown ee_format: {ee_format}")
    return pose_to_se3(xyz, R)


def pose_to_xyz_rotvec_from_format(pose: np.ndarray, ee_format: str) -> np.ndarray:
    """Convert a pose vector to xyz + rotvec based on source format."""
    T = pose_to_se3_from_format(pose, ee_format)
    return se3_to_xyz_rotvec(T)


def pose_to_xyz_rot6d_from_format(pose: np.ndarray, ee_format: str) -> np.ndarray:
    """Convert a pose vector to xyz + rotation-6D based on source format."""
    T = pose_to_se3_from_format(pose, ee_format)
    xyz = T[..., :3, 3]
    rot6d = matrix_to_rot6d(T[..., :3, :3])
    return np.concatenate([xyz, rot6d], axis=-1)


def compute_relative_actions(
    current_state_pose: np.ndarray,
    future_action_poses: np.ndarray,
    ee_format: str,
) -> np.ndarray:
    """Compute relative EE actions: T_rel_i = T_curr^(-1) @ T_future_i.

    Args:
        current_state_pose: (D,) current EE state pose
        future_action_poses: (N, D) future EE action poses
        ee_format: "xyz_rpy", "xyz_quat", or "xyz_rvec"
    Returns:
        (N, 6) relative actions as [dx, dy, dz, drx, dry, drz] (xyz + rotvec)
    """
    T_curr = pose_to_se3_from_format(current_state_pose, ee_format)  # (4, 4)
    T_future = pose_to_se3_from_format(future_action_poses, ee_format)  # (N, 4, 4)
    T_curr_inv = se3_inverse(T_curr)  # (4, 4)
    T_rel = T_curr_inv @ T_future  # (N, 4, 4) via broadcasting
    return se3_to_xyz_rotvec(T_rel)  # (N, 6)
