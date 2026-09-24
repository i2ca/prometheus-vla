"""Maps dataset-specific actions/states to unified representations."""

import numpy as np

from .config import DatasetSourceConfig
from .se3_utils import pose_to_xyz_rot6d_from_format, pose_to_xyz_rotvec_from_format

# Unified action layout: 54 dimensions
# [0:6]   left_xyz_rotvec
# [6:7]   left_gripper
# [7:13]  left_fig6d
# [13:19] right_xyz_rotvec
# [19:20] right_gripper
# [20:26] right_fig6d
# [26:29] waist_joint
# [29:32] torso_joint
# [32:34] base_vx_vy
# [34:35] base_vw
# [35:41] base_rotvec  (6 dims: relative xyz [35:38] + axis-angle [38:41])
# [41:42] height
# [42:48] left_leg_joint
# [48:54] right_leg_joint

UNIFIED_DIM = 54
STATE_DIM = 60

# Action slice definitions
SLICES = {
    "left_xyz_rotvec": slice(0, 6),
    "left_gripper": slice(6, 7),
    "left_fig6d": slice(7, 13),
    "right_xyz_rotvec": slice(13, 19),
    "right_gripper": slice(19, 20),
    "right_fig6d": slice(20, 26),
    "waist_joint": slice(26, 29),
    "torso_joint": slice(29, 32),
    "base_vx_vy": slice(32, 34),
    "base_vw": slice(34, 35),
    "base_rotvec": slice(35, 41),   # EXPANDED: relative xyz [35:38] + axis-angle [38:41]
    "height": slice(41, 42),
    "left_leg_joint": slice(42, 48),
    "right_leg_joint": slice(48, 54),
}

# State layout: 60 dimensions
# EE state uses xyz + rotation-6D, while action keeps xyz + rotvec.
STATE_SLICES = {
    "left_xyz_rot6d": slice(0, 9),
    "left_gripper": slice(9, 10),
    "left_fig6d": slice(10, 16),
    "right_xyz_rot6d": slice(16, 25),
    "right_gripper": slice(25, 26),
    "right_fig6d": slice(26, 32),
    "waist_joint": slice(32, 35),
    "torso_joint": slice(35, 38),
    "base_vx_vy": slice(38, 40),
    "base_vw": slice(40, 41),
    "base_rotvec": slice(41, 47),
    "height": slice(47, 48),
    "left_leg_joint": slice(48, 54),
    "right_leg_joint": slice(54, 60),
}


def _fill_and_mask(
    unified: np.ndarray,
    mask: np.ndarray,
    slot: slice,
    values: np.ndarray,
):
    """Fill values into unified vector at given slot and set mask.

    If values has fewer elements than the slot, the remainder is zero-padded.
    """
    n = slot.stop - slot.start
    v = np.asarray(values, dtype=np.float32).ravel()
    if len(v) < n:
        padded = np.zeros(n, dtype=np.float32)
        padded[:len(v)] = v
        v = padded
    unified[slot] = v[:n]
    mask[slot] = True


def map_single_left_action(
    rel_ee: np.ndarray,
    gripper: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Map single-arm (→left) relative action to unified representation.

    Args:
        rel_ee: (6,) relative xyz+rotvec
        gripper: (1,) gripper value
    Returns:
        (unified[54], mask[54])
    """
    unified = np.zeros(UNIFIED_DIM, dtype=np.float32)
    mask = np.zeros(UNIFIED_DIM, dtype=bool)
    _fill_and_mask(unified, mask, SLICES["left_xyz_rotvec"], rel_ee)
    _fill_and_mask(unified, mask, SLICES["left_gripper"], gripper)
    return unified, mask

def map_action_chunk(
    config: DatasetSourceConfig,
    rel_ee_chunk: np.ndarray | None,
    left_rel_ee_chunk: np.ndarray | None,
    right_rel_ee_chunk: np.ndarray | None,
    gripper_chunk: np.ndarray | None,
    left_gripper_chunk: np.ndarray | None,
    right_gripper_chunk: np.ndarray | None,
    base_command_chunk: np.ndarray | None = None,
    torso_joint_chunk: np.ndarray | None = None,
    waist_joint_chunk: np.ndarray | None = None,
    left_leg_chunk: np.ndarray | None = None,
    right_leg_chunk: np.ndarray | None = None,
    left_fig6d_chunk: np.ndarray | None = None,
    right_fig6d_chunk: np.ndarray | None = None,
    base_pose_rel_chunk: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Map a full action chunk (T timesteps) to unified representation.

    Args:
        config: dataset source config
        *_chunk: (T, D) arrays for each action component, or None if not available
    Returns:
        actions: (T, 54) unified action chunk
        mask: (54,) bool mask (same for all timesteps)
    """
    if config.arm_type == "single_left":
        T = rel_ee_chunk.shape[0]
        actions = np.zeros((T, UNIFIED_DIM), dtype=np.float32)
        mask = np.zeros(UNIFIED_DIM, dtype=bool)
        actions[:, SLICES["left_xyz_rotvec"]] = rel_ee_chunk[:, :6]
        mask[SLICES["left_xyz_rotvec"]] = True
        if gripper_chunk is not None:
            actions[:, SLICES["left_gripper"]] = gripper_chunk[:, :1]
            mask[SLICES["left_gripper"]] = True
        return actions, mask

    elif config.arm_type in ("dual", "dual_with_legs"):
        T = left_rel_ee_chunk.shape[0]
        actions = np.zeros((T, UNIFIED_DIM), dtype=np.float32)
        mask = np.zeros(UNIFIED_DIM, dtype=bool)

        actions[:, SLICES["left_xyz_rotvec"]] = left_rel_ee_chunk[:, :6]
        mask[SLICES["left_xyz_rotvec"]] = True
        if left_gripper_chunk is not None:
            actions[:, SLICES["left_gripper"]] = left_gripper_chunk[:, :1]
            mask[SLICES["left_gripper"]] = True

        actions[:, SLICES["right_xyz_rotvec"]] = right_rel_ee_chunk[:, :6]
        mask[SLICES["right_xyz_rotvec"]] = True
        if right_gripper_chunk is not None:
            actions[:, SLICES["right_gripper"]] = right_gripper_chunk[:, :1]
            mask[SLICES["right_gripper"]] = True

        if base_command_chunk is not None and config.base_command_dims is not None:
            dims = config.base_command_dims
            if "vx" in dims and "vy" in dims:
                actions[:, SLICES["base_vx_vy"]] = base_command_chunk[:, [dims["vx"], dims["vy"]]]
                mask[SLICES["base_vx_vy"]] = True
            vw_key = "vyaw" if "vyaw" in dims else ("vw" if "vw" in dims else None)
            if vw_key:
                actions[:, SLICES["base_vw"]] = base_command_chunk[:, dims[vw_key]:dims[vw_key]+1]
                mask[SLICES["base_vw"]] = True
            if "height" in dims:
                actions[:, SLICES["height"]] = base_command_chunk[:, dims["height"]:dims["height"]+1]
                mask[SLICES["height"]] = True

        for chunk, slot in [
            (torso_joint_chunk, "torso_joint"),
            (waist_joint_chunk, "waist_joint"),
            (left_leg_chunk, "left_leg_joint"),
            (right_leg_chunk, "right_leg_joint"),
        ]:
            if chunk is not None:
                s = SLICES[slot]
                n = s.stop - s.start
                actions[:, s] = chunk[:, :n]
                mask[s] = True

        if left_fig6d_chunk is not None:
            actions[:, SLICES["left_fig6d"]] = left_fig6d_chunk[:, :6]
            mask[SLICES["left_fig6d"]] = True
        if right_fig6d_chunk is not None:
            actions[:, SLICES["right_fig6d"]] = right_fig6d_chunk[:, :6]
            mask[SLICES["right_fig6d"]] = True
        if base_pose_rel_chunk is not None:
            actions[:, SLICES["base_rotvec"]] = base_pose_rel_chunk[:, :6]
            mask[SLICES["base_rotvec"]] = True

        return actions, mask

    else:
        raise ValueError(f"Unknown arm_type: {config.arm_type}")


def map_state(
    config: DatasetSourceConfig,
    state_dict: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Map current state to unified representation.

    States use a 60-dim layout. EE poses are absolute xyz + rot6d.

    Args:
        config: dataset source config
        state_dict: maps semantic key (from StateKeysConfig field names) to value arrays
    Returns:
        state: (60,) unified state
        mask: (60,) bool mask
    """
    unified = np.zeros(STATE_DIM, dtype=np.float32)
    mask = np.zeros(STATE_DIM, dtype=bool)

    if config.arm_type == "single_left":
        if "ee_pose" in state_dict:
            pose = pose_to_xyz_rot6d_from_format(state_dict["ee_pose"], config.ee_format)
            _fill_and_mask(unified, mask, STATE_SLICES["left_xyz_rot6d"], pose)
        if "gripper" in state_dict:
            _fill_and_mask(unified, mask, STATE_SLICES["left_gripper"], state_dict["gripper"])

    elif config.arm_type in ("dual", "dual_with_legs"):
        if "left_ee_pose" in state_dict:
            pose = pose_to_xyz_rot6d_from_format(state_dict["left_ee_pose"], config.ee_format)
            _fill_and_mask(unified, mask, STATE_SLICES["left_xyz_rot6d"], pose)
        if "left_gripper" in state_dict:
            _fill_and_mask(unified, mask, STATE_SLICES["left_gripper"], state_dict["left_gripper"])
        if "right_ee_pose" in state_dict:
            pose = pose_to_xyz_rot6d_from_format(state_dict["right_ee_pose"], config.ee_format)
            _fill_and_mask(unified, mask, STATE_SLICES["right_xyz_rot6d"], pose)
        if "right_gripper" in state_dict:
            _fill_and_mask(unified, mask, STATE_SLICES["right_gripper"], state_dict["right_gripper"])
        if "left_fig6d" in state_dict:
            _fill_and_mask(unified, mask, STATE_SLICES["left_fig6d"], state_dict["left_fig6d"])
        if "right_fig6d" in state_dict:
            _fill_and_mask(unified, mask, STATE_SLICES["right_fig6d"], state_dict["right_fig6d"])

        if "base_rot" in state_dict:
            # gravity_xyz(3) + omega_xyz(3); replaces the old base_pose-derived state fill
            # (never active). The action side still predicts relative base pose from
            # state_base_pose/action_base_pose separately -- see _compute_relative_base_pose.
            _fill_and_mask(unified, mask, STATE_SLICES["base_rotvec"], state_dict["base_rot"])
        if "torso_joint" in state_dict:
            _fill_and_mask(unified, mask, STATE_SLICES["torso_joint"], state_dict["torso_joint"])
        if "waist_joint" in state_dict:
            _fill_and_mask(unified, mask, STATE_SLICES["waist_joint"], state_dict["waist_joint"])
        if "left_leg" in state_dict:
            _fill_and_mask(unified, mask, STATE_SLICES["left_leg_joint"], state_dict["left_leg"])
        if "right_leg" in state_dict:
            _fill_and_mask(unified, mask, STATE_SLICES["right_leg_joint"], state_dict["right_leg"])

    return unified, mask
