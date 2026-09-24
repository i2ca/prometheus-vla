"""Seven-joint palm IK. All trial FK operates on scratch data, never on live physics."""
import cv2
import mujoco
import numpy as np
from g1_sim import ARM_JOINTS, HAND_JOINTS

PALM = "right_wrist_yaw_link"


class ArmIK:
    def __init__(self, sim, joint_names=None, palm=PALM):
        """joint_names: juntas que a IK move (padrao: as 7 do braco direito; pode incluir a cintura antes delas).
        palm: corpo cuja pose a IK resolve. O default e o punho direito, e passar o esquerdo permite
        usar o outro braco sem tocar em nada que ja esta validado."""
        self.sim, self.m = sim, sim.m
        self.d = mujoco.MjData(self.m)
        self.joint_names = list(joint_names) if joint_names else list(ARM_JOINTS)
        self.joints = [self.m.joint(n).id for n in self.joint_names]
        self.qa = self.m.jnt_qposadr[self.joints]
        self.va = self.m.jnt_dofadr[self.joints]
        self.bounds = self.m.jnt_range[self.joints]
        self.body = self.m.body(palm).id
        self.weights = None   # pesos por junta no passo da IK (ex.: cintura mais lenta que o braco)
        self.nullspace_gain = 0.03

    def fk(self, q):
        self.d.qpos[:] = self.sim.d.qpos
        self.d.qpos[self.qa] = q
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_comPos(self.m, self.d)
        return self.d.xpos[self.body].copy(), self.d.xmat[self.body].reshape(3, 3).copy()

    def solve(self, target, rotation, q, reference=None, iterations=120, max_step=None):
        """max_step: limite (rad) da variacao total de cada junta nesta chamada, i.e. por quadro de controle."""
        q0 = np.clip(np.asarray(q, float), self.bounds[:, 0], self.bounds[:, 1])
        q = q0.copy()
        reference = q.copy() if reference is None else np.asarray(reference)
        jp, jr = np.zeros((3, self.m.nv)), np.zeros((3, self.m.nv))
        for it in range(iterations):
            p, r = self.fk(q)
            ep = np.asarray(target) - p
            er = cv2.Rodrigues(np.asarray(rotation) @ r.T)[0].ravel()
            if np.linalg.norm(ep) < 0.0003 and np.linalg.norm(er) < 0.003:
                break
            mujoco.mj_jac(self.m, self.d, jp, jr, p, self.body)
            j = np.vstack((jp[:, self.va], 0.15 * jr[:, self.va]))
            inv = j.T @ np.linalg.solve(j @ j.T + 0.002**2 * np.eye(6), np.eye(6))
            delta = inv @ np.r_[ep, 0.15 * er]
            delta += self.nullspace_gain * (np.eye(len(q)) - inv @ j) @ (reference - q)
            if self.weights is not None:
                delta *= self.weights
            delta *= min(1.0, 0.10 / max(np.max(np.abs(delta)), 1e-12))
            q = np.clip(q + delta, self.bounds[:, 0], self.bounds[:, 1])
            if max_step is not None:
                q = q0 + np.clip(q - q0, -max_step, max_step)
        p, r = self.fk(q)
        return q, {"position_error_m": float(np.linalg.norm(np.asarray(target) - p)),
                   "orientation_error_rad": float(np.linalg.norm(cv2.Rodrigues(np.asarray(rotation) @ r.T)[0])),
                   "iterations": it + 1}

    def reference_grasp(self, arm, hand, frame, offset, gain=1.5):
        p, r = self.fk(arm[frame])
        opened = hand[0]
        closed = opened + gain * (hand[-1] - opened)
        for name, value in zip(HAND_JOINTS, opened + 0.6 * (closed - opened)):
            self.d.qpos[self.m.jnt_qposadr[self.m.joint(name).id]] = value
        mujoco.mj_kinematics(self.m, self.d)
        tip = lambda name: self.d.xpos[self.m.body("right_hand_" + name + "_link").id]
        center = (tip("thumb_2") + (tip("index_1") + tip("middle_1")) / 2) / 2
        cup = center + np.asarray(offset)
        return {"palm_position": p, "palm_rotation": r,
                "cup_position": cup.copy(), "palm_to_cup_translation": r.T @ (cup - p),
                "palm_to_center_translation": r.T @ (center - p),
                "palm_to_cup_rotation": r.T, "open": opened, "close": closed}
