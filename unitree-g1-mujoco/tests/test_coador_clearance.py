"""The actual cup can enter the stand, but solid parts still collide."""
from pathlib import Path
import mujoco
import numpy as np

ASSETS = Path(__file__).resolve().parents[1] / "assets"


def setup():
    model = mujoco.MjModel.from_xml_path(str(ASSETS / "scene_43dof.xml"))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    cup_qadr = model.jnt_qposadr[model.joint("junta_livre_bloco").id]
    stand = model.body("coador").id
    return model, data, cup_qadr, stand


def cup_stand_contacts(model, data, stand):
    cup = model.body("objeto_customizado").id
    return [c for c in data.contact
            if {int(model.geom_bodyid[c.geom1]), int(model.geom_bodyid[c.geom2])} == {cup, stand}]


def test_cup_has_clear_front_insertion_path():
    model, data, qadr, stand = setup()
    rotation = data.xmat[stand].reshape(3, 3).copy()
    origin = data.xpos[stand].copy()
    assert (rotation @ np.array([-1., 0., 0.]))[0] > 0.99  # stem at the back
    for x in np.linspace(0.14, -0.009, 40):
        data.qpos[qadr:qadr + 3] = origin + rotation @ [x, 0.001, 0.0035]
        mujoco.mj_forward(model, data)
        assert not [c for c in cup_stand_contacts(model, data, stand) if c.dist < -1e-5], x


def test_cup_settles_on_stand_below_funnel():
    model, data, qadr, stand = setup()
    model.opt.timestep = 0.004
    data.qpos[qadr:qadr + 3] = data.xpos[stand] + data.xmat[stand].reshape(3, 3) @ [-0.009, 0.001, 0.004]
    for _ in range(750):
        data.qpos[:50] = model.qpos0[:50]
        data.qvel[:49] = 0
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)
    assert abs(data.qpos[qadr + 2] - 0.7533) < 0.003
    contacts = cup_stand_contacts(model, data, stand)
    assert contacts
    for contact in contacts:
        stand_geom = contact.geom1 if model.geom_bodyid[contact.geom1] == stand else contact.geom2
        assert model.geom(stand_geom).name == "workspace_coador_collision"
    assert np.isfinite(data.qpos).all()


def test_rear_stem_still_blocks_cup():
    model, data, qadr, stand = setup()
    data.qpos[qadr:qadr + 3] = data.xpos[stand] + data.xmat[stand].reshape(3, 3) @ [-0.0638, 0.0005, 0.04]
    mujoco.mj_forward(model, data)
    stem = model.geom("workspace_coador_stem").id
    assert any(stem in (c.geom1, c.geom2) and c.dist < 0
               for c in cup_stand_contacts(model, data, stand))
