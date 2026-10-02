"""Smoke tests for the MuJoCo scenes shipped with the simulator."""

import json
from pathlib import Path

import mujoco
import numpy as np


ASSETS = Path(__file__).resolve().parents[1] / "assets"


def load_scene(name: str) -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_path(str(ASSETS / name))


def names(model: mujoco.MjModel, object_type: mujoco.mjtObj, count: int) -> set[str]:
    return {
        mujoco.mj_id2name(model, object_type, index)
        for index in range(count)
        if mujoco.mj_id2name(model, object_type, index)
    }


def test_default_astra_scene_preserves_simulator_contract() -> None:
    model = load_scene("scene_43dof.xml")

    joint_names = names(model, mujoco.mjtObj.mjOBJ_JOINT, model.njnt)
    body_names = names(model, mujoco.mjtObj.mjOBJ_BODY, model.nbody)
    geom_names = names(model, mujoco.mjtObj.mjOBJ_GEOM, model.ngeom)
    camera_names = names(model, mujoco.mjtObj.mjOBJ_CAMERA, model.ncam)

    assert model.nu == 43
    assert "floating_base_joint" in joint_names
    assert "junta_livre_bloco" in joint_names
    assert {"mesa", "objeto_customizado", "torso_link"} <= body_names
    assert {"geometria_mesa", "geometria_bloco"} <= geom_names
    assert {
        "head_camera",
        "head_camera_depth",
        "left_wrist_camera",
        "right_wrist_camera",
        "view_left",
        "view_center",
        "side_view",
        "global_view",
    } <= camera_names


def test_all_scenes_compile_and_step_without_nan() -> None:
    for scene in (
        "scene_43dof.xml",
        "scene_astra_gonogo.xml",
        "scene_43dof_smart_ia_legacy.xml",
    ):
        model = load_scene(scene)
        data = mujoco.MjData(model)
        data.qpos[:] = model.qpos0
        for _ in range(25):
            mujoco.mj_step(model, data)
        assert np.isfinite(data.qpos).all(), scene
        assert np.isfinite(data.qvel).all(), scene


def test_astra_metadata_is_readable() -> None:
    metadata = ASSETS / "astra_grasp"
    markers = json.loads((metadata / "markers.json").read_text())
    cup_vertices = np.load(metadata / "cup_vertices.npy", allow_pickle=False)

    assert markers
    assert cup_vertices.ndim == 2
    assert cup_vertices.shape[1] == 3


def test_coffee_objects_settle_and_respond_to_force() -> None:
    model = load_scene("scene_43dof.xml")
    legacy = load_scene("scene_43dof_smart_ia_legacy.xml")
    model.opt.timestep = 0.004
    data = mujoco.MjData(model)
    # Additional free joints must not shift the robot's DDS indices.
    for index in range(44):
        assert model.joint(index).name == legacy.joint(index).name
        assert model.jnt_qposadr[index] == legacy.jnt_qposadr[index]
    assert model.nu == legacy.nu == 43

    def step():
        # Hold the robot still to isolate prop gravity/contact from teleop.
        data.qpos[:50] = model.qpos0[:50]
        data.qvel[:49] = 0
        mujoco.mj_step(model, data)

    for _ in range(750):
        step()
    for name in ("chaleira", "coador", "pote", "tampa", "scoop"):
        joint = model.joint(f"workspace_{name}_free").id
        qadr = model.jnt_qposadr[joint]
        vadr = model.jnt_dofadr[joint]
        assert model.body(name).mass[0] > 0
        assert abs(data.qpos[qadr + 2] - 0.75) < 0.002, name
        assert np.linalg.norm(data.qvel[vadr:vadr + 6]) < 0.01, name

    scoop_body = model.body("scoop").id
    scoop_joint = model.joint("workspace_scoop_free").id
    qadr = model.jnt_qposadr[scoop_joint]
    start_x = data.qpos[qadr]
    data.xfrc_applied[scoop_body, 0] = 1.0
    for _ in range(20):
        step()
    assert data.qpos[qadr] > start_x + 0.005
    data.xfrc_applied[:] = 0
    data.qpos[qadr + 2] += 0.1
    lifted_z = data.qpos[qadr + 2]
    for _ in range(100):
        step()
    assert data.qpos[qadr + 2] < lifted_z - 0.05
    assert np.isfinite(data.qpos).all()
    mujoco.mj_resetData(model, data)
    assert np.allclose(data.qpos, model.qpos0)
