#!/usr/bin/env python
"""
Unitree G1 Dex3 Robot - Unified DDS Architecture

This implementation uses the same DDS threading pattern as the body/arm code,
eliminating the multiprocessing complexity of the original Dex3_1_Controller.
"""

from dataclasses import dataclass, field
import logging
import threading
import time
import numpy as np
from functools import cached_property
import os
import sys
import subprocess

from .unitree_g1 import UnitreeG1, UnitreeG1Config
from lerobot.robots.config import RobotConfig
from .g1_utils import (
    Dex3_1_Left_JointIndex, 
    Dex3_1_Right_JointIndex,
    Dex3_1_Left_PressureTemperatureSensors,
    Dex3_1_Right_PressureTemperatureSensors,
    sensor_index,
    Dex3_Num_Motors,
    DEX3_LEFT_LOWER_LIMITS,
    DEX3_LEFT_UPPER_LIMITS,
    DEX3_RIGHT_LOWER_LIMITS,
    DEX3_RIGHT_UPPER_LIMITS,
    kTopicDex3LeftCommand,
    kTopicDex3RightCommand,
    kTopicDex3LeftState,
    kTopicDex3RightState,
    LEFT_HAND_JOINT_NAMES,
    RIGHT_HAND_JOINT_NAMES,
)

from lerobot.processor import RobotAction, RobotObservation

logger = logging.getLogger(__name__)


@dataclass
class HandMotorState:
    """State of a single hand motor."""
    q: float = 0.0  # position


@dataclass
class HandState:
    """State of a single hand (7 motors for Dex3-1)."""
    motor_state: list[HandMotorState] = field(
        default_factory=lambda: [HandMotorState() for _ in range(Dex3_Num_Motors)]
    )
    # Adicionando 27 sensores de pressão e 27 de temperatura por mão
    pressure: np.ndarray = field(default_factory=lambda: np.zeros(33, dtype=np.float32))
    temperature: np.ndarray = field(default_factory=lambda: np.zeros(33, dtype=np.float32))


@RobotConfig.register_subclass("unitree_g1_dex3")
@dataclass
class UnitreeG1Dex3Config(UnitreeG1Config):
    """Configuration for Unitree G1 with Dex3-1 hands."""
    hand_kp: float = 1.5  # Position gain for hand motors
    hand_kd: float = 0.2  # Damping gain for hand motors
    hand_control_dt: float = 0.01  # 100 Hz control loop
    
    def __post_init__(self):
        # LÓGICA DE RESOLUÇÃO DINÂMICA
        if self.is_simulation:
            self.robot_ip = "127.0.0.1"
            # Simulação: Leve e rápida para não gargalar a GPU/CPU
            cam_width = 320
            cam_height = 240
            cam2_width = 1280
            cam2_height = 720
        else:
            # Hardware Real: Resolução máxima da Intel RealSense
            cam_width = 640
            cam_height = 480
            cam2_width = 1280
            cam2_height = 720
            
        # Adiciona as câmeras ZMQ ao LeRobot usando as variáveis dinâmicas
        if not self.cameras:
            from lerobot.cameras.zmq.configuration_zmq import ZMQCameraConfig
            
            self.cameras = {
                # A NOSSA ÚNICA CÂMERA RGB (HD para o VR e para a IA)
                "head_camera": ZMQCameraConfig(
                    server_address=self.robot_ip, port=5555, camera_name="head_camera", width=cam2_width, height=cam2_height
                ),
                
                # AS 3 LENTES TÉCNICAS (Baixa Resolução para o processamento ser imediato)
                "d435i_depth": ZMQCameraConfig(
                    server_address=self.robot_ip, port=5555, camera_name="d435i_depth", width=cam_width, height=cam_height
                ),
                "d435i_ir_left": ZMQCameraConfig(
                    server_address=self.robot_ip, port=5555, camera_name="d435i_ir_left", width=cam_width, height=cam_height
                ),
                "d435i_ir_right": ZMQCameraConfig(
                    server_address=self.robot_ip, port=5555, camera_name="d435i_ir_right", width=cam_width, height=cam_height
                )
            }


class UnitreeG1Dex3(UnitreeG1):
    """
    Unitree G1 Robot with Dex3-1 Dexterous Hands.
    
    Uses the same DDS threading architecture as the body motors:
    - A background thread subscribes to hand state at 100Hz
    - Hand commands are published directly via DDS
    """
    config_class = UnitreeG1Dex3Config
    name = "unitree_g1_dex3"

    def __init__(self, config: UnitreeG1Dex3Config):
        super().__init__(config)
        
        # Hand state (similar to _lowstate for body)
        self._left_hand_state: HandState | None = None
        self._right_hand_state: HandState | None = None
        
        # Threading control
        self._hand_shutdown_event = threading.Event()
        self._hand_subscribe_thread: threading.Thread | None = None
        
        # DDS publishers/subscribers (initialized in connect)
        self._left_hand_cmd_pub = None
        self._right_hand_cmd_pub = None
        self._left_hand_state_sub = None
        self._right_hand_state_sub = None
        
        # Command messages (initialized in connect)
        self._left_hand_msg = None
        self._right_hand_msg = None
        
        # Use joint name constants from g1_utils
        self.left_hand_joint_names = LEFT_HAND_JOINT_NAMES
        self.right_hand_joint_names = RIGHT_HAND_JOINT_NAMES

    def _subscribe_hand_state(self):
        """
        Background thread that polls hand state via DDS at ~100Hz.
        Similar to _subscribe_motor_state() in UnitreeG1.
        """
        while not self._hand_shutdown_event.is_set():
            start_time = time.time()
            
            # Read left hand state
            left_msg = self._left_hand_state_sub.Read()
            if left_msg is not None:
                left_state = HandState()
                for idx, joint_id in enumerate(Dex3_1_Left_JointIndex):
                    left_state.motor_state[idx].q = left_msg.motor_state[joint_id].q
                #self._left_hand_state = left_state

                # --- LÓGICA DE SENSORES EXTRAÍDA DO robot_hand_unitree.py ---
                idx = 0
                has_sensors = len(left_msg.press_sensor_state) > 0
                for area_idx, id in enumerate(Dex3_1_Left_PressureTemperatureSensors):
                    for sensor_idx in sensor_index[id]:
                        # Proteção: verifica se o simulador/robô mandou os sensores
                        if has_sensors and area_idx < len(left_msg.press_sensor_state):
                            try:
                                left_state.pressure[idx] = left_msg.press_sensor_state[area_idx].pressure[sensor_idx]
                                left_state.temperature[idx] = left_msg.press_sensor_state[area_idx].temperature[sensor_idx]
                            except IndexError:
                                pass # Ignora se o array interno do sensor for menor que o esperado
                        idx += 1
                
                self._left_hand_state = left_state
            
            # Read right hand state
            right_msg = self._right_hand_state_sub.Read()
            if right_msg is not None:
                right_state = HandState()
                for idx, joint_id in enumerate(Dex3_1_Right_JointIndex):
                    right_state.motor_state[idx].q = right_msg.motor_state[joint_id].q
                #self._right_hand_state = right_state

                # --- LÓGICA DE SENSORES EXTRAÍDA DO robot_hand_unitree.py ---
                idx = 0
                has_sensors = len(right_msg.press_sensor_state) > 0
                for area_idx, id in enumerate(Dex3_1_Right_PressureTemperatureSensors):
                    for sensor_idx in sensor_index[id]:
                        if has_sensors and area_idx < len(right_msg.press_sensor_state):
                            try:
                                right_state.pressure[idx] = right_msg.press_sensor_state[area_idx].pressure[sensor_idx]
                                right_state.temperature[idx] = right_msg.press_sensor_state[area_idx].temperature[sensor_idx]
                            except IndexError:
                                pass
                        idx += 1
                
                self._right_hand_state = right_state
            
            # Maintain control rate
            elapsed = time.time() - start_time
            sleep_time = max(0, self.config.hand_control_dt - elapsed)
            time.sleep(sleep_time)

    def connect(self, calibrate: bool = True) -> None:
        """Connect to robot body and hands."""
        # Connect body first
        super().connect(calibrate=calibrate)
    
        
        # Skip hand connection in simulation mode
        if self.config.is_simulation:
            logger.info("Simulation mode: Iniciando a ponte da mão")

            # --- ADICIONE/AJUSTE ESTA PARTE ---
            print(f"[UnitreeG1] Forçando conexão ZMQ com a ponte em {self.config.robot_ip}")
            from .unitree_sdk2_socket import ChannelFactoryInitialize

            # Isso 'liga' os sockets internos do LeRobot
            ChannelFactoryInitialize(0, self.config.robot_ip) 
            try:
                # Descobre o caminho absoluto do diretório onde este arquivo está
                current_dir = os.path.dirname(os.path.abspath(__file__))
                # Monta o caminho exato para a pasta robot_control/ponte_mao.py
                ponte_path = os.path.join(current_dir, "robot_control", "ponte_mao.py")
                
                logger.info(f"[UnitreeG1] Iniciando ponte ZMQ-DDS em segundo plano: {ponte_path}")
                # Inicia o processo com o mesmo executável Python que o LeRobot está usando
                self._ponte_process = subprocess.Popen([sys.executable, ponte_path])
                logger.info(f"[UnitreeG1] Ponte iniciada com PID: {self._ponte_process.pid}")
                
                # Aguarda 1 segundinho para dar tempo da ponte abrir as portas ZMQ antes do código avançar
                time.sleep(1.0)
            except Exception as e:
                logger.error(f"[UnitreeG1] Falha ao iniciar a ponte automaticamente: {e}") 
        
        # Use ZMQ-based hand communication (SDK-free, no circular import issues)
        # The server on the robot bridges hand DDS topics to ZMQ sockets
        from .unitree_sdk2_socket import (
            ChannelSubscriber,
            ChannelPublisher,
            HandCmdMsg,
            kTopicDex3LeftCommand,
            kTopicDex3RightCommand,
            kTopicDex3LeftState,
            kTopicDex3RightState,
        )
        
        # Initialize hand state subscribers (ZMQ-based)
        self._left_hand_state_sub = ChannelSubscriber(kTopicDex3LeftState, None)
        self._left_hand_state_sub.Init()
        self._right_hand_state_sub = ChannelSubscriber(kTopicDex3RightState, None)
        self._right_hand_state_sub.Init()
        
        # Initialize hand command publishers (ZMQ-based)
        self._left_hand_cmd_pub = ChannelPublisher(kTopicDex3LeftCommand, None)
        self._left_hand_cmd_pub.Init()
        self._right_hand_cmd_pub = ChannelPublisher(kTopicDex3RightCommand, None)
        self._right_hand_cmd_pub.Init()
        
        # Initialize command messages with default gains
        self._left_hand_msg = HandCmdMsg()
        self._right_hand_msg = HandCmdMsg()
        
        kp = self.config.hand_kp
        kd = self.config.hand_kd
        
        for joint_id in Dex3_1_Left_JointIndex:
            # Mode byte: bits 0-3 = id, bits 4-6 = status (0x01 = enabled), bit 7 = timeout
            mode = (joint_id & 0x0F) | (0x01 << 4)
            self._left_hand_msg.motor_cmd[joint_id].mode = mode
            self._left_hand_msg.motor_cmd[joint_id].kp = kp
            self._left_hand_msg.motor_cmd[joint_id].kd = kd
            
        for joint_id in Dex3_1_Right_JointIndex:
            mode = (joint_id & 0x0F) | (0x01 << 4)
            self._right_hand_msg.motor_cmd[joint_id].mode = mode
            self._right_hand_msg.motor_cmd[joint_id].kp = kp
            self._right_hand_msg.motor_cmd[joint_id].kd = kd
        
        # Start hand state subscription thread
        self._hand_subscribe_thread = threading.Thread(
            target=self._subscribe_hand_state, 
            daemon=True,
            name="Dex3HandStateSubscriber"
        )
        self._hand_subscribe_thread.start()
        
        # Wait for first hand state
        timeout = 3.0
        start = time.time()
        while self._left_hand_state is None or self._right_hand_state is None:
            if time.time() - start > timeout:
                logger.warning("Timeout waiting for Dex3 hand state. Hands may not be connected.")
                break
            time.sleep(0.01)
        
        if self._left_hand_state is not None and self._right_hand_state is not None:
            logger.info("Connected to Dex3 Hands via ZMQ.")
        else:
            logger.warning("Dex3 Hands not fully connected - hand state unavailable.")

    def disconnect(self):
        """Disconnect from robot body and hands."""
        # Signal hand thread to stop
        self._hand_shutdown_event.set()
        
        # Wait for hand thread to finish
        if self._hand_subscribe_thread is not None:
            self._hand_subscribe_thread.join(timeout=2.0)
            if self._hand_subscribe_thread.is_alive():
                logger.warning("Hand subscribe thread did not stop cleanly")
        
        # Disconnect body
        super().disconnect()

    @cached_property
    def action_features(self) -> dict[str, type]:
        """Define action space: body joints (based on control_mode) + hand joints.
        
        - full_body mode: 29 body + 14 hand = 43 joints
        - upper_body mode: 14 arm + 14 hand = 28 joints
        """
        features = super().action_features
        for name in self.left_hand_joint_names:
            features[f"{name}.q"] = float
        for name in self.right_hand_joint_names:
            features[f"{name}.q"] = float
        return features

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        """Define observation space: body joints (based on control_mode) + hand joints.
        
        - full_body mode: 29 body + 14 hand = 43 joints
        - upper_body mode: 14 arm + 14 hand = 28 joints
        """
        features = super().observation_features
        for name in self.left_hand_joint_names:
            features[f"{name}.q"] = float
        for name in self.right_hand_joint_names:
            features[f"{name}.q"] = float

        for i in range(33):
            features[f"left_hand_pressure_{i}"] = float
            features[f"right_hand_pressure_{i}"] = float

        return features

    def get_observation(self) -> RobotObservation:
        """Get observation including hand joint positions."""
        obs = super().get_observation()
        
        # Add left hand state (default to 0.0 if hands not connected)
        for i, name in enumerate(self.left_hand_joint_names):
            if self._left_hand_state is not None:
                obs[f"{name}.q"] = float(self._left_hand_state.motor_state[i].q)
            else:
                obs[f"{name}.q"] = 0.0  # Default when hands not available
        
        # Add right hand state (default to 0.0 if hands not connected)
        for i, name in enumerate(self.right_hand_joint_names):
            if self._right_hand_state is not None:
                obs[f"{name}.q"] = float(self._right_hand_state.motor_state[i].q)
            else:
                obs[f"{name}.q"] = 0.0  # Default when hands not available

        # Add left hand sensors
        #if self._left_hand_state is not None:
        #    obs["left_hand_pressure"] = self._left_hand_state.pressure
        #    obs["left_hand_temperature"] = self._left_hand_state.temperature
        #else:
        #    obs["left_hand_pressure"] = np.zeros(33, dtype=np.float32)
        #    obs["left_hand_temperature"] = np.zeros(33, dtype=np.float32)
        #    
        # Add right hand sensors
        #if self._right_hand_state is not None:
        #    obs["right_hand_pressure"] = self._right_hand_state.pressure
        #    obs["right_hand_temperature"] = self._right_hand_state.temperature
        #else:
        #    obs["right_hand_pressure"] = np.zeros(33, dtype=np.float32)
        #    obs["right_hand_temperature"] = np.zeros(33, dtype=np.float32)

        # Injeção das pressões desmembradas
        left_pressure = self._left_hand_state.pressure if self._left_hand_state is not None else np.zeros(33)
        right_pressure = self._right_hand_state.pressure if self._right_hand_state is not None else np.zeros(33)

        for i in range(33):
            obs[f"left_hand_pressure_{i}"] = float(left_pressure[i])
            obs[f"right_hand_pressure_{i}"] = float(right_pressure[i])
        
        # ==========================================================
        # DEBUG TEMPORÁRIO: Só printa se houver alguma pressão!
        # ==========================================================
        max_l = np.max(left_pressure)
        max_r = np.max(right_pressure)
        if max_l > 0.01 or max_r > 0.01:
            print(f"👉 TATO DETECTADO! Pressão Esq: {max_l:.2f} | Pressão Dir: {max_r:.2f}")
        # ==========================================================
        return obs

        return obs

    def send_action(self, action: RobotAction) -> RobotAction:
        """Send action to robot including hand commands."""
        # Send body action
        super().send_action(action)
        
        # Check if we have hand publishers
        if self._left_hand_cmd_pub is None or self._right_hand_cmd_pub is None:
            return action
        
        # Check if action contains hand commands
        first_joint = self.left_hand_joint_names[0]
        if f"{first_joint}.q" not in action:
            return action
        
        # Extract and clamp left hand targets
        left_q = np.zeros(Dex3_Num_Motors)
        for i, name in enumerate(self.left_hand_joint_names):
            left_q[i] = action.get(f"{name}.q", 0.0)
        left_q = np.clip(left_q, DEX3_LEFT_LOWER_LIMITS, DEX3_LEFT_UPPER_LIMITS)
        
        # Extract and clamp right hand targets
        right_q = np.zeros(Dex3_Num_Motors)
        for i, name in enumerate(self.right_hand_joint_names):
            right_q[i] = action.get(f"{name}.q", 0.0)
        right_q = np.clip(right_q, DEX3_RIGHT_LOWER_LIMITS, DEX3_RIGHT_UPPER_LIMITS)
        
        # Update command messages
        for idx, joint_id in enumerate(Dex3_1_Left_JointIndex):
            self._left_hand_msg.motor_cmd[joint_id].q = left_q[idx]
        for idx, joint_id in enumerate(Dex3_1_Right_JointIndex):
            self._right_hand_msg.motor_cmd[joint_id].q = right_q[idx]
        
        # Publish commands
        self._left_hand_cmd_pub.Write(self._left_hand_msg)
        self._right_hand_cmd_pub.Write(self._right_hand_msg)
        
        return action