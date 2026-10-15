"""Recovery behavior without DDS, a viewer, or external APIs."""
import sys
from pathlib import Path
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sim.workspace_recovery import WorkspaceRecovery


def scene():
    model = mujoco.MjModel.from_xml_path(str(ROOT / "assets/scene_43dof.xml"))
    return model, mujoco.MjData(model)


def test_all_props_recover_with_zero_velocity_and_force():
    model, data = scene()
    recovery = WorkspaceRecovery(model)
    robot_pose = data.qpos[:50].copy()
    for name, body, qadr, vadr, home, _ in recovery.props:
        data.qpos[qadr + 2] = 0.1
        data.qvel[vadr:vadr + 6] = 2
        data.qfrc_applied[vadr:vadr + 6] = 3
        data.xfrc_applied[body] = 4
        assert recovery.update(data) == [name]
        assert np.allclose(data.qpos[qadr:qadr + 7], home)
        assert np.allclose(data.qvel[vadr:vadr + 6], 0)
        assert np.allclose(data.qfrc_applied[vadr:vadr + 6], 0)
        assert np.allclose(data.xfrc_applied[body], 0)
    assert np.allclose(data.qpos[:50], robot_pose)


def test_lifted_and_tabletop_objects_are_not_reset():
    model, data = scene()
    recovery = WorkspaceRecovery(model)
    for _, _, qadr, _, _, _ in recovery.props:
        data.qpos[qadr + 2] += 0.3
    pose = data.qpos.copy()
    assert recovery.update(data) == []
    assert np.array_equal(data.qpos, pose)


def test_disabled_recovery_leaves_fallen_object_untouched():
    model, data = scene()
    qadr = model.jnt_qposadr[model.joint("junta_livre_bloco").id]
    data.qpos[qadr + 2] = 0.1
    assert WorkspaceRecovery(model, enabled=False).update(data) == []
    assert data.qpos[qadr + 2] == 0.1


def test_occupied_home_uses_clear_nearby_slot():
    model, data = scene()
    recovery = WorkspaceRecovery(model)
    cup = next(p for p in recovery.props if p[0] == "objeto_customizado")
    strainer = next(p for p in recovery.props if p[0] == "coador")
    data.qpos[cup[2]:cup[2] + 3] = strainer[4][:3]
    data.qpos[strainer[2] + 2] = 0.1
    assert recovery.update(data) == ["coador"]
    assert not np.allclose(data.qpos[strainer[2]:strainer[2] + 2], strainer[4][:2])
    assert np.allclose(data.qpos[cup[2]:cup[2] + 3], strainer[4][:3])


def test_physical_fall_is_recovered_without_resetting_robot():
    model, data = scene()
    model.opt.timestep = 0.004
    recovery = WorkspaceRecovery(model)
    prop = next(p for p in recovery.props if p[0] == "coador")
    data.qpos[prop[2]] = 1.2  # beyond the back edge
    recovered = []
    for _ in range(200):
        data.qpos[:50] = model.qpos0[:50]
        data.qvel[:49] = 0
        mujoco.mj_step(model, data)
        recovered.extend(recovery.update(data))
    assert recovered == ["coador"]
    assert abs(data.qpos[prop[2] + 2] - 0.75) < 0.003
    assert np.isfinite(data.qpos).all()


def test_no_clear_slot_preserves_object_and_retries_later():
    model, data = scene()
    recovery = WorkspaceRecovery(model)
    prop = next(p for p in recovery.props if p[0] == "coador")
    data.qpos[prop[2] + 2] = 0.1
    pose = data.qpos.copy()
    size = recovery.half_size.copy()
    recovery.half_size[:] = 0  # no valid candidate can fit
    assert recovery.update(data) == []
    assert np.array_equal(data.qpos, pose)
    recovery.half_size[:] = size
    assert recovery.update(data) == []  # throttle retries when all slots are blocked
    data.time = 0.5
    assert recovery.update(data) == ["coador"]
