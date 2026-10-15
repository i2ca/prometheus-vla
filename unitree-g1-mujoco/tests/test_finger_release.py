"""Finger opening with the actual scene's inertia, friction and configured soft gains."""
from pathlib import Path
import mujoco
import numpy as np


def test_both_middle_fingers_can_close_then_reopen_with_soft_gains():
    assets = Path(__file__).resolve().parents[1] / 'assets'
    for side, sign in [('left', -1), ('right', 1)]:
        model = mujoco.MjModel.from_xml_path(str(assets / 'scene_43dof.xml'))
        model.opt.timestep = .004
        model.opt.gravity[:] = 0
        # Isolate joint friction/control from objects and robot self-contact.
        model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
        joint = model.joint(f'{side}_hand_middle_1_joint').id
        qadr = int(model.jnt_qposadr[joint])
        vadr = int(model.jnt_dofadr[joint])
        actuator = int(np.flatnonzero(model.actuator_trnid[:, 0] == joint)[0])
        data = mujoco.MjData(model)
        for target in (sign * 1.3, 0.):
            for _ in range(1250):
                q, velocity = data.qpos[qadr], data.qvel[vadr]
                data.qpos[:] = model.qpos0
                data.qvel[:] = 0
                data.qpos[qadr], data.qvel[vadr] = q, velocity
                data.ctrl[actuator] = .8 * (target - q) - .2 * velocity
                mujoco.mj_step(model, data)
            assert abs(data.qpos[qadr] - target) < .05, (side, target, data.qpos[qadr])
