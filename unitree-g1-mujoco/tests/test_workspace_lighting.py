"""Headless rendering regression: small head motion must not flicker the table.
Run with MUJOCO_GL=egl on a headless machine.
"""
from pathlib import Path
import mujoco
import numpy as np


def test_small_camera_motion_does_not_flicker_empty_tabletop():
    assets = Path(__file__).resolve().parents[1] / "assets"
    model = mujoco.MjModel.from_xml_path(str(assets / "scene_43dof.xml"))
    data = mujoco.MjData(model)
    camera = model.camera("head_camera").id
    origin = model.cam_pos[camera].copy()
    patches = []
    with mujoco.Renderer(model, height=480, width=640) as renderer:
        for offset in (0., 0.0002, -0.0002, 0.):
            model.cam_pos[camera] = origin + [offset, 0, 0]
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera="head_camera")
            frame = renderer.render()
            patches.append(frame[:100, 190:355].astype(float))
    # Same matte patch must stay spatially smooth and temporally stable.
    assert np.std(patches[0], axis=(0, 1)).mean() < 1.0
    assert max(np.abs(patch - patches[0]).mean() for patch in patches[1:]) < 0.5
