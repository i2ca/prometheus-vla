"""Local training assistance: return fallen coffee props to a clear table slot."""
import mujoco
import numpy as np


class WorkspaceRecovery:
    def __init__(self, model, enabled=True):
        self.model = model
        self.enabled = enabled
        self.props = []
        self.next_attempt = {}
        self.last_time = 0.0
        table = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "geometria_mesa")
        self.table_geom = table
        if not enabled or table < 0:
            return
        reference = mujoco.MjData(model)
        mujoco.mj_forward(model, reference)
        self.table_top = reference.geom_xpos[table, 2] + model.geom_size[table, 2]
        self.center = reference.geom_xpos[table, :2].copy()
        self.half_size = model.geom_size[table, :2].copy()
        for name, joint_name in (
            ("objeto_customizado", "junta_livre_bloco"),
            *((name, f"workspace_{name}_free") for name in ("chaleira", "coador", "pote", "tampa", "scoop")),
        ):
            joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint < 0 or model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_FREE:
                continue
            body = int(model.jnt_bodyid[joint])
            qadr, vadr = int(model.jnt_qposadr[joint]), int(model.jnt_dofadr[joint])
            # Include the whole approximate collision footprint in table clearance.
            geoms = [g for g in range(model.ngeom) if model.geom_bodyid[g] == body
                     and (model.geom_contype[g] or model.geom_conaffinity[g])]
            footprint = np.zeros(2)
            for geom in geoms:
                kind = model.geom_type[geom]
                if kind == mujoco.mjtGeom.mjGEOM_BOX:
                    half = model.geom_size[geom].copy()
                elif kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
                    half = np.array([model.geom_size[geom, 0]] * 2 + [model.geom_size[geom, 1]])
                elif kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
                    half = np.array([model.geom_size[geom, 0]] * 2
                                    + [model.geom_size[geom, 1] + model.geom_size[geom, 0]])
                else:
                    half = np.full(3, model.geom_rbound[geom])
                extent = np.abs(reference.geom_xmat[geom].reshape(3, 3)) @ half
                offset = np.abs(reference.geom_xpos[geom, :2] - reference.xpos[body, :2])
                footprint = np.maximum(footprint, offset + extent[:2])
            self.props.append((name, body, qadr, vadr, model.qpos0[qadr:qadr + 7].copy(), footprint))

    def update(self, data):
        if not self.props:
            return []
        if data.time < self.last_time:
            self.next_attempt.clear()
        self.last_time = data.time
        recovered = []
        for name, body, qadr, vadr, home, footprint in self.props:
            # A lifted object is unaffected. Recover only once its base has fallen
            # 15 cm below the tabletop, rather than waiting for floor impact.
            if data.qpos[qadr + 2] >= self.table_top - 0.15:
                continue
            if data.time < self.next_attempt.get(name, 0.0):
                continue
            fallen_pose = data.qpos[qadr:qadr + 7].copy()
            offsets = [(0.0, 0.0)]
            for distance in (0.08, 0.16, 0.24):
                offsets.extend((dx, dy) for dx, dy in (
                    (distance, 0), (0, distance), (0, -distance), (-distance, 0),
                    (distance, distance), (distance, -distance),
                    (-distance, distance), (-distance, -distance)))
            placed = False
            for dx, dy in offsets:
                pose = home.copy()
                pose[:2] += (dx, dy)
                if np.any(np.abs(pose[:2] - self.center) + footprint + 0.01 > self.half_size + 1e-8):
                    continue
                data.qpos[qadr:qadr + 7] = pose
                mujoco.mj_forward(self.model, data)
                blocked = any(
                    contact.dist < -1e-5
                    and body in (self.model.geom_bodyid[contact.geom1], self.model.geom_bodyid[contact.geom2])
                    and self.table_geom not in (contact.geom1, contact.geom2)
                    for contact in data.contact
                )
                if blocked:
                    continue
                data.qvel[vadr:vadr + 6] = 0
                data.qacc_warmstart[vadr:vadr + 6] = 0
                data.qfrc_applied[vadr:vadr + 6] = 0
                data.xfrc_applied[body] = 0
                recovered.append(name)
                self.next_attempt.pop(name, None)
                placed = True
                break
            if not placed:
                # Never spawn inside a hand/object; avoid expensive searches at 250 Hz.
                data.qpos[qadr:qadr + 7] = fallen_pose
                self.next_attempt[name] = data.time + 0.5
            mujoco.mj_forward(self.model, data)
        return recovered
