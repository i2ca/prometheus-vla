"""Base compartilhada: modelo, mapas por nome, controle PD em torque e utilidades de contato.

Os atuadores do XML sao motores de torque; o PD abaixo reproduz o controle de posicao
do robo real (kp/kd por junta, torque saturado no limite do motor)."""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
import json
import numpy as np
import mujoco

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ARM_JOINTS = ["right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
              "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint"]
# ordem da semente (dataset): polegar 0,1,2, indicador 0,1, medio 0,1. O XML lista polegar, medio, indicador.
HAND_JOINTS = ["right_hand_thumb_0_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint",
               "right_hand_index_0_joint", "right_hand_index_1_joint",
               "right_hand_middle_0_joint", "right_hand_middle_1_joint"]
ARM_KP = np.array([80, 80, 80, 80, 40, 40, 40], float)
ARM_KD = np.array([3, 3, 3, 0.3, 1.5, 1.5, 1.5], float)
# fecho firme: valores fora do range sao saturados pelo limite da junta
CLOSE_TGT = np.array([0.0, -1.5, -1.5, 1.5, 1.5, 1.5, 1.5])


class G1Sim:
    def __init__(self, scene="scene_gonogo.xml", cup_body="copo"):
        self.m = mujoco.MjModel.from_xml_path(os.path.join(ROOT, "scene", scene))
        self.d = mujoco.MjData(self.m)
        m = self.m
        self.act_joint = {}          # nome da junta -> id do atuador
        for i in range(m.nu):
            self.act_joint[m.joint(m.actuator_trnid[i, 0]).name] = i
        self.kp = np.zeros(m.nu); self.kd = np.zeros(m.nu); self.q_des = np.zeros(m.nu)
        self.qadr = np.array([m.jnt_qposadr[m.actuator_trnid[i, 0]] for i in range(m.nu)])
        self.vadr = np.array([m.jnt_dofadr[m.actuator_trnid[i, 0]] for i in range(m.nu)])
        self.tau_max = m.actuator_ctrlrange[:, 1].copy()
        # ganhos padrao: pernas/tronco rigidos (base ja e fixa, so mantem postura), bracos e maos como o robo
        for jn, i in self.act_joint.items():
            if "hand" in jn:
                self.kp[i], self.kd[i] = 1.5, 0.1
            elif "right_" in jn and any(k in jn for k in ("shoulder", "elbow", "wrist")):
                k = ARM_JOINTS.index(jn); self.kp[i], self.kd[i] = ARM_KP[k], ARM_KD[k]
            elif "left_" in jn and any(k in jn for k in ("shoulder", "elbow", "wrist")):
                self.kp[i], self.kd[i] = 60, 2
            else:
                self.kp[i], self.kd[i] = 200, 5
        self.cup_body = m.body(cup_body).id
        self.cup_jnt = None
        for j in range(m.njnt):
            if m.jnt_bodyid[j] == self.cup_body and m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                self.cup_jnt = j
        self.cup_qadr = int(m.jnt_qposadr[self.cup_jnt]); self.cup_dadr = int(m.jnt_dofadr[self.cup_jnt])
        self.hand_bodies = [b for b in range(m.nbody) if m.body(b).name.startswith("right_hand")]
        mujoco.mj_forward(m, self.d)

    # ---- alvos ----
    def set_targets(self, joint_names, q, kp=None, kd=None):
        for jn, qi in zip(joint_names, q):
            i = self.act_joint[jn]; self.q_des[i] = qi
            if kp is not None: self.kp[i] = kp
            if kd is not None: self.kd[i] = kd

    def q(self, joint_names):
        return np.array([self.d.qpos[self.qadr[self.act_joint[j]]] for j in joint_names])

    def step(self, n=1):
        m, d = self.m, self.d
        for _ in range(n):
            tau = self.kp * (self.q_des - d.qpos[self.qadr]) - self.kd * d.qvel[self.vadr]
            d.ctrl[:] = np.clip(tau, -self.tau_max, self.tau_max)
            mujoco.mj_step(m, d)

    # ---- copo ----
    def place_cup(self, pos, quat=(1, 0, 0, 0)):
        d = self.d; a = self.cup_qadr
        d.qpos[a:a+3] = pos; d.qpos[a+3:a+7] = quat; d.qvel[self.cup_dadr:self.cup_dadr+6] = 0
        mujoco.mj_forward(self.m, d)

    def cup_pos(self):
        return self.d.xpos[self.cup_body].copy()

    def body_pos(self, name):
        return self.d.xpos[self.m.body(name).id].copy()

    def grasp_center(self):
        opp = (self.body_pos("right_hand_index_1_link") + self.body_pos("right_hand_middle_1_link")) / 2
        return (self.body_pos("right_hand_thumb_2_link") + opp) / 2

    def finger_contacts(self):
        """Nomes dos elos da mao direita em contato com o copo neste instante."""
        m, d = self.m, self.d; names = set()
        for i in range(d.ncon):
            c = d.contact[i]; b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
            if self.cup_body in (b1, b2):
                o = b2 if b1 == self.cup_body else b1
                if o in self.hand_bodies:
                    names.add(m.body(o).name.replace("right_hand_", "").replace("_link", ""))
        return names

    def cup_table_contact(self):
        m, d = self.m, self.d; tampo = m.geom("tampo").id
        for i in range(d.ncon):
            c = d.contact[i]
            if tampo in (c.geom1, c.geom2) and self.cup_body in (int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])):
                return True
        return False

    def warnings(self):
        return int(self.d.warning.number.sum())


def load_seed(ep=0):
    f = np.load(os.path.join(ROOT, "seed", "demo_reference_rightarm_hand.npz"), allow_pickle=True)
    T = int(f["lengths"][ep]); seq = f["ref_state"][ep, :T]
    return seq[:, :7].copy(), seq[:, 7:14].copy(), int(f["fps"])
