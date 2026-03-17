import logging
import pygame
import numpy as np
from dataclasses import dataclass
from typing import Any

from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.teleoperators.config import TeleoperatorConfig
from lerobot.processor import RobotAction

# Importa o solver IK e os nomes corretos
from .utils.g1_arm_ik import G1_29_ArmIK
from robot.unitree_g1.g1_utils import LEFT_HAND_JOINT_NAMES, RIGHT_HAND_JOINT_NAMES

logger = logging.getLogger(__name__)

ARM_JOINT_NAMES = [
    "kLeftShoulderPitch.q", "kLeftShoulderRoll.q", "kLeftShoulderYaw.q", "kLeftElbow.q", 
    "kLeftWristRoll.q", "kLeftWristPitch.q", "kLeftWristyaw.q",  
    "kRightShoulderPitch.q", "kRightShoulderRoll.q", "kRightShoulderYaw.q", "kRightElbow.q", 
    "kRightWristRoll.q", "kRightWristPitch.q", "kRightWristYaw.q"
]

LEFT_HAND_NAMES = [f"{name}.q" for name in LEFT_HAND_JOINT_NAMES]
RIGHT_HAND_NAMES = [f"{name}.q" for name in RIGHT_HAND_JOINT_NAMES]

LEFT_HAND_CLOSED_TARGETS =  [0.0,  1.5,  1.5, -1.5, -1.5, -1.5, -1.5]
RIGHT_HAND_CLOSED_TARGETS = [0.0, -1.5, -1.5,  1.5,  1.5,  1.5,  1.5]

@TeleoperatorConfig.register_subclass("keyboard_g1_arm_v2")
@dataclass
class KeyboardG1ArmV2Config(TeleoperatorConfig):
    speed: float = 0.005      # Velocidade do braço (XYZ)
    rot_speed: float = 0.03   # Velocidade dos motores do pulso (radianos)
    fps: int = 60

class KeyboardG1ArmV2(Teleoperator):
    config_class = KeyboardG1ArmV2Config
    name = "keyboard_g1_arm_v2" 

    def __init__(self, config: KeyboardG1ArmV2Config):
        super().__init__(config)
        self.config = config
        self._is_connected = False
        
        self.ik_solver = G1_29_ArmIK(visualization=False)
        self.current_q = np.zeros(14)
        self.wrist_initialized = False 

        # Estado independente para cada dedo (0.0 = Aberto, 1.0 = Fechado)
        self.left_fingers = {"thumb": 0.0, "index": 0.0, "middle": 0.0}
        self.right_fingers = {"thumb": 0.0, "index": 0.0, "middle": 0.0}

        self.left_pose = np.eye(4)
        self.right_pose = np.eye(4)
        self.left_pose[:3, 3] = [0.3,  0.2, 0.32] 
        self.right_pose[:3, 3] = [0.3, -0.2, 0.32]

        self.left_wrist_q = np.zeros(3)  # [Roll, Pitch, Yaw]
        self.right_wrist_q = np.zeros(3) # [Roll, Pitch, Yaw]

    def connect(self, calibrate: bool = True) -> None:
        if self._is_connected: return
        pygame.init()
        pygame.display.set_mode((400, 300))
        pygame.display.set_caption("Controle Híbrido G1 (Dedos Independentes)")
        logger.info("Controle Ativo. Mão Esq: Z, X, C. Mão Dir: M, N, B. Use SHIFT para abrir.")
        self._is_connected = True

    def disconnect(self) -> None:
        if self._is_connected:
            pygame.quit()
            self._is_connected = False

    @property
    def is_connected(self) -> bool: return self._is_connected
    @property
    def is_calibrated(self) -> bool: return True
    def calibrate(self) -> None: pass
    def configure(self) -> None: pass
    
    @property
    def action_features(self) -> dict:
        return {name: float for name in ARM_JOINT_NAMES + LEFT_HAND_NAMES + RIGHT_HAND_NAMES}

    @property
    def feedback_features(self) -> dict:
        return {name: float for name in ARM_JOINT_NAMES}

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        for i, name in enumerate(ARM_JOINT_NAMES):
            if name in feedback:
                self.current_q[i] = feedback[name]
                
                if not self.wrist_initialized:
                    if name == "kLeftWristRoll.q": self.left_wrist_q[0] = feedback[name]
                    if name == "kLeftWristPitch.q": self.left_wrist_q[1] = feedback[name]
                    if name == "kLeftWristyaw.q": self.left_wrist_q[2] = feedback[name]
                    if name == "kRightWristRoll.q": self.right_wrist_q[0] = feedback[name]
                    if name == "kRightWristPitch.q": self.right_wrist_q[1] = feedback[name]
                    if name == "kRightWristYaw.q": self.right_wrist_q[2] = feedback[name]
                    
        if "kLeftWristyaw.q" in feedback:
            self.wrist_initialized = True

    def get_action(self) -> RobotAction:
        if not self._is_connected:
             raise ConnectionError("Teleoperador por Teclado não está conectado.")
        
        pygame.event.pump()
        keys = pygame.key.get_pressed()
        vel = self.config.speed
        rot = self.config.rot_speed

        l_shift = keys[pygame.K_LSHIFT]
        r_shift = keys[pygame.K_RSHIFT]

        # ================= CONTROLE BRAÇO ESQUERDO =================
        if l_shift:
            if keys[pygame.K_w]: self.left_wrist_q[1] += rot  
            if keys[pygame.K_s]: self.left_wrist_q[1] -= rot  
            if keys[pygame.K_a]: self.left_wrist_q[2] += rot  
            if keys[pygame.K_d]: self.left_wrist_q[2] -= rot  
            if keys[pygame.K_q]: self.left_wrist_q[0] -= rot  
            if keys[pygame.K_e]: self.left_wrist_q[0] += rot  
        else:
            if keys[pygame.K_w]: self.left_pose[0, 3] += vel  
            if keys[pygame.K_s]: self.left_pose[0, 3] -= vel  
            if keys[pygame.K_a]: self.left_pose[1, 3] += vel  
            if keys[pygame.K_d]: self.left_pose[1, 3] -= vel  
            if keys[pygame.K_q]: self.left_pose[2, 3] += vel  
            if keys[pygame.K_e]: self.left_pose[2, 3] -= vel  

        # ================= CONTROLE BRAÇO DIREITO =================
        if r_shift:
            if keys[pygame.K_i]: self.right_wrist_q[1] += rot
            if keys[pygame.K_k]: self.right_wrist_q[1] -= rot
            if keys[pygame.K_j]: self.right_wrist_q[2] += rot 
            if keys[pygame.K_l]: self.right_wrist_q[2] -= rot
            if keys[pygame.K_u]: self.right_wrist_q[0] -= rot 
            if keys[pygame.K_o]: self.right_wrist_q[0] += rot
        else:
            if keys[pygame.K_i]: self.right_pose[0, 3] += vel 
            if keys[pygame.K_k]: self.right_pose[0, 3] -= vel 
            if keys[pygame.K_j]: self.right_pose[1, 3] += vel 
            if keys[pygame.K_l]: self.right_pose[1, 3] -= vel 
            if keys[pygame.K_u]: self.right_pose[2, 3] += vel 
            if keys[pygame.K_o]: self.right_pose[2, 3] -= vel 

        # ================= RESOLUÇÃO HÍBRIDA =================
        ik_current_q = self.current_q.copy()
        ik_current_q[4:7] = 0.0   
        ik_current_q[11:14] = 0.0 

        sol_q, _ = self.ik_solver.solve_ik(
            left_wrist=self.left_pose, 
            right_wrist=self.right_pose,
            current_arm_q=ik_current_q 
        )
        
        robot_action = {}
        for i, name in enumerate(ARM_JOINT_NAMES):
            if name == "kLeftWristRoll.q":    robot_action[name] = float(self.left_wrist_q[0])
            elif name == "kLeftWristPitch.q": robot_action[name] = float(self.left_wrist_q[1])
            elif name == "kLeftWristyaw.q":   robot_action[name] = float(self.left_wrist_q[2])
            elif name == "kRightWristRoll.q": robot_action[name] = float(self.right_wrist_q[0])
            elif name == "kRightWristPitch.q":robot_action[name] = float(self.right_wrist_q[1])
            elif name == "kRightWristYaw.q":  robot_action[name] = float(self.right_wrist_q[2])
            else:
                robot_action[name] = float(sol_q[i])

        # ================= MÃOS (Dedos Independentes) =================
        hand_speed = vel * 10 
        
        # --- Mão Esquerda ---
        # Z (Polegar)
        if keys[pygame.K_z]:
            if l_shift: self.left_fingers["thumb"] = max(self.left_fingers["thumb"] - hand_speed, 0.0)
            else:       self.left_fingers["thumb"] = min(self.left_fingers["thumb"] + hand_speed, 1.0)
        # X (Indicador)
        if keys[pygame.K_x]:
            if l_shift: self.left_fingers["index"] = max(self.left_fingers["index"] - hand_speed, 0.0)
            else:       self.left_fingers["index"] = min(self.left_fingers["index"] + hand_speed, 1.0)
        # C (Médio)
        if keys[pygame.K_c]:
            if l_shift: self.left_fingers["middle"] = max(self.left_fingers["middle"] - hand_speed, 0.0)
            else:       self.left_fingers["middle"] = min(self.left_fingers["middle"] + hand_speed, 1.0)

        # --- Mão Direita ---
        # M (Polegar)
        if keys[pygame.K_m]:
            if r_shift: self.right_fingers["thumb"] = max(self.right_fingers["thumb"] - hand_speed, 0.0)
            else:       self.right_fingers["thumb"] = min(self.right_fingers["thumb"] + hand_speed, 1.0)
        # N (Indicador)
        if keys[pygame.K_n]:
            if r_shift: self.right_fingers["index"] = max(self.right_fingers["index"] - hand_speed, 0.0)
            else:       self.right_fingers["index"] = min(self.right_fingers["index"] + hand_speed, 1.0)
        # B (Médio)
        if keys[pygame.K_b]:
            if r_shift: self.right_fingers["middle"] = max(self.right_fingers["middle"] - hand_speed, 0.0)
            else:       self.right_fingers["middle"] = min(self.right_fingers["middle"] + hand_speed, 1.0)

        # Vetor de multiplicação dos dedos da mão esquerda (0,1,2=Polegar | 3,4=Médio | 5,6=Indicador)
        left_factors = [
            self.left_fingers["thumb"], self.left_fingers["thumb"], self.left_fingers["thumb"],
            self.left_fingers["middle"], self.left_fingers["middle"],
            self.left_fingers["index"], self.left_fingers["index"]
        ]

        # Vetor de multiplicação dos dedos da mão direita (0,1,2=Polegar | 3,4=Indicador | 5,6=Médio)
        right_factors = [
            self.right_fingers["thumb"], self.right_fingers["thumb"], self.right_fingers["thumb"],
            self.right_fingers["index"], self.right_fingers["index"],
            self.right_fingers["middle"], self.right_fingers["middle"]
        ]

        for i, name in enumerate(LEFT_HAND_NAMES):
            robot_action[name] = float(left_factors[i] * LEFT_HAND_CLOSED_TARGETS[i])
        for i, name in enumerate(RIGHT_HAND_NAMES):
            robot_action[name] = float(right_factors[i] * RIGHT_HAND_CLOSED_TARGETS[i])

        return robot_action