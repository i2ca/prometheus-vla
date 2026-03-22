import zmq
import time
import cv2
import threading
import logging
import numpy as np
from dataclasses import dataclass
from typing import Any

from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.teleoperators.config import TeleoperatorConfig
from lerobot.processor import RobotAction

# Imports do ecossistema Unitree (ajuste os caminhos conforme sua pasta)
from televuer import TeleVuerWrapper
from .utils.sensor_utils import SensorClient, ImageUtils
from teleop.robot_control.robot_arm_ik import G1_29_ArmIK
from teleop.robot_control.hand_retargeting import HandRetargeting, HandType

logger = logging.getLogger(__name__)

@TeleoperatorConfig.register_subclass("xr_g1_arm")
@dataclass
class XRG1ArmConfig(TeleoperatorConfig):
    img_server_ip: str = "127.0.0.1"
    #img_server_ip: str = "127.0.0.1"
    is_simulation: bool = True
    input_mode: str = "hand"       # 'hand' ou 'controller'
    display_mode: str = "immersive" # 'immersive', 'ego', 'pass-through'
    ee_type: str = "dex3"
    zmq: bool = True
    webrtc: bool = False

class XRG1Arm(Teleoperator):
    config_class = XRG1ArmConfig
    name = "xr_g1_arm"

    def __init__(self, config: XRG1ArmConfig):
        super().__init__(config)
        self.config = config
        self._is_connected = False

        if not self.config.is_simulation:
            self.config.img_server_ip = "192.168.123.164"
        
        # Carrega os nomes das juntas do seu robô no LeRobot
        from robot.unitree_g1.g1_utils import G1_29_JointIndex, LEFT_HAND_JOINT_NAMES, RIGHT_HAND_JOINT_NAMES
        self._left_hand_names = LEFT_HAND_JOINT_NAMES
        self._right_hand_names = RIGHT_HAND_JOINT_NAMES
        
        self.body_joints = {f"{motor.name}.q": 0.0 for motor in G1_29_JointIndex}
        self.hand_joints = {f"{name}.q": 0.0 for name in self._left_hand_names + self._right_hand_names}

        # Inicializa o Wrapper do VR e o Solver IK
        self.tv_wrapper = None
        self.arm_ik = None
        self.hand_retargeter = None

        # Estado atual do robô (necessário como "semente" para o cálculo do IK)
        self.current_arm_q = np.zeros(14)
        self.current_arm_dq = np.zeros(14)

        # NOVAS VARIÁVEIS DE SEGURANÇA
        self.vr_started = False
        self.start_time = None
        self.countdown_done = False

    def connect(self, calibrate: bool = True) -> None:
        if self._is_connected:
            return

        logger.info("Iniciando conexão com o Vuer VR...")
        
        # 1. Inicia o servidor WebRTC / WebSocket do Vuer
        self.tv_wrapper = TeleVuerWrapper(
            use_hand_tracking=(self.config.input_mode == "hand"),
            binocular=False,
            img_shape=(720, 1280),
            display_mode=self.config.display_mode,
            zmq=self.config.zmq,
            webrtc=self.config.webrtc,
            webrtc_url=f"https://{self.config.img_server_ip}:60000/offer",
        )
        
        # 2. Inicia o Solver de Cinemática Inversa dos braços
        logger.info("Carregando URDF e IK do Braço G1_29...")
        self.arm_ik = G1_29_ArmIK()

        # 3. Inicia o Retargeting Das Mãos
        if self.config.ee_type == "dex3":
            logger.info("Iniciando algoritmo de Retargeting para Dex3...")
            self.hand_retargeter = HandRetargeting(HandType.UNITREE_DEX3)

        self._is_connected = True

        # NOVA PARTE: Usando o SensorClient que já funciona!
        if self.config.zmq:
            logger.info(f"Conectando ao feed de vídeo ZMQ via SensorClient em {self.config.img_server_ip}:5555...")
            self.sensor_client = SensorClient()
            self.sensor_client.start_client(server_ip=self.config.img_server_ip, port=5555)
            
            # Inicia uma thread em background para receber as imagens
            self.video_thread = threading.Thread(target=self._receive_video_feed, daemon=True)
            self.video_thread.start()

        logger.info("VR Teleoperator Conectado! Visite o link do Vuer no navegador do headset.")

    def _receive_video_feed(self):
        while self._is_connected:
            try:
                # Recebe a mensagem estruturada usando o client oficial do seu sim
                data = self.sensor_client.receive_message()
                
                if not data:
                    time.sleep(0.005)
                    continue

                # Extrai a imagem do dicionário (lidando com os dois formatos que seu script suporta)
                img_data = None
                cam_name = "head_camera"  # Nome padrão, mas vamos buscar dinamicamente se falhar
                
                if "images" in data and cam_name in data["images"]:
                    img_data = data["images"][cam_name]
                elif cam_name in data:
                    img_data = data[cam_name]
                else:
                    # Pega a primeira câmera que encontrar no dicionário
                    keys = [k for k in data.keys() if k not in ["timestamps", "images"]]
                    if keys:
                        img_data = data[keys[0]]

                # Se achou a imagem, decodifica usando o ImageUtils
                if img_data is not None:
                    if isinstance(img_data, str):
                        img = ImageUtils.decode_image(img_data)
                    else:
                        img = img_data  # Assume que já é numpy array

                    if img is not None and isinstance(img, np.ndarray):
                        # Converte de BGR para RGB
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        
                        # REDIMENSIONAMENTO DE SEGURANÇA: Garante HD exato pro Vuer
                        img_rgb = cv2.resize(img_rgb, (1280, 720)) 
                        
                        # Envia para o VR
                        self.tv_wrapper.render_to_xr(img_rgb)

            except Exception as e:
                logger.error(f"Erro no feed de vídeo ZMQ: {e}")
                time.sleep(0.01) # Pausa curta para não floodar o log em caso de erro contínuo

    def disconnect(self) -> None:
        if self._is_connected:
            self._is_connected = False
            
            # Aguarda a thread de vídeo encerrar
            if hasattr(self, 'video_thread') and self.video_thread.is_alive():
                self.video_thread.join(timeout=1.0)
            
            if self.tv_wrapper:
                self.tv_wrapper.close()
                
            # Fecha o client corretamente
            if hasattr(self, 'sensor_client'):
                self.sensor_client.stop_client()

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    @property
    def action_features(self) -> dict:
        features = {}
        for key in self.body_joints.keys():
            features[key] = float
        for key in self.hand_joints.keys():
            features[key] = float
        return features

    @property
    def feedback_features(self) -> dict:
        return {"q": np.ndarray}

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        if "q" in feedback:
            self.current_arm_q = feedback["q"][:14] 

    def get_action(self) -> RobotAction:
        if not self._is_connected:
             raise ConnectionError("XR Teleoperator não está conectado.")

        # 1. Pega os dados do Headset VR
        tele_data = self.tv_wrapper.get_tele_data()

        # --- LÓGICA DE SEGURANÇA CORRIGIDA ---
        
        # Ignoramos a cabeça (head_pose) pois a biblioteca cria uma "falsa" ao carregar a página.
        # A forma 100% garantida é verificar se as mãos/controles estão sendo rastreados,
        # o que só acontece APÓS você clicar em "Enter VR" no óculos.
        has_right_hand = np.any(tele_data.right_hand_pos != 0.0)
        has_left_hand = np.any(tele_data.left_hand_pos != 0.0)

        # Considera que o VR iniciou se qualquer uma das mãos apareceu
        active_session = has_right_hand or has_left_hand

        if not active_session:
            if self.vr_started:
                logger.warning("Rastreamento de mãos não detectado (VR inativo). Mantendo posição.")
                self.vr_started = False
                self.countdown_done = False
            
            # Retorna a posição atual para manter o robô imóvel
            return {**self.body_joints, **self.hand_joints}

        # Inicia contagem se detectou as mãos no VR
        if active_session and not self.vr_started:
            logger.info(">>> MÃOS DETECTADAS NO VR! AGUARDANDO 5 SEGUNDOS PARA INICIAR...")
            self.vr_started = True
            self.start_time = time.time()

        if self.vr_started and not self.countdown_done:
            elapsed = time.time() - self.start_time
            if elapsed < 5.0:
                if int(elapsed * 10) % 10 == 0: 
                    print(f"--- ESTABILIZANDO ROBÔ: {5 - int(elapsed)}s ---", end='\r')
                return {**self.body_joints, **self.hand_joints}
            else:
                logger.info("\n>>> SISTEMA LIBERADO! MOVIMENTANDO G1...")
                self.countdown_done = True
        
        # --- FIM DA LÓGICA DE SEGURANÇA ---


        # 2. Calcula IK dos Braços (retorna 14 ângulos)
        sol_q, _ = self.arm_ik.solve_ik(
            tele_data.left_wrist_pose, 
            tele_data.right_wrist_pose, 
            self.current_arm_q, 
            self.current_arm_dq
        )

        # Mapeia os 14 ângulos para o dicionário do LeRobot
        # Esquerdo (índices 0 a 6)
        self.body_joints["kLeftShoulderPitch.q"] = sol_q[0]
        self.body_joints["kLeftShoulderRoll.q"]  = sol_q[1]
        self.body_joints["kLeftShoulderYaw.q"]   = sol_q[2]
        self.body_joints["kLeftElbow.q"]         = sol_q[3]
        self.body_joints["kLeftWristRoll.q"]     = sol_q[4]
        self.body_joints["kLeftWristPitch.q"]    = sol_q[5]
        self.body_joints["kLeftWristYaw.q"]      = sol_q[6]

        # Direito (índices 7 a 13)
        self.body_joints["kRightShoulderPitch.q"] = sol_q[7]
        self.body_joints["kRightShoulderRoll.q"]  = sol_q[8]
        self.body_joints["kRightShoulderYaw.q"]   = sol_q[9]
        self.body_joints["kRightElbow.q"]         = sol_q[10]
        self.body_joints["kRightWristRoll.q"]     = sol_q[11]
        self.body_joints["kRightWristPitch.q"]    = sol_q[12]
        self.body_joints["kRightWristYaw.q"]      = sol_q[13]

        # 3. Calcula o Retargeting das Mãos (Dedos)
        if self.config.ee_type == "dex3" and self.config.input_mode == "hand":
            
            # CORREÇÃO 2: Formatação e cálculo correto dos vetores das mãos (Conforme Dex3_1_Controller)
            left_hand_data = tele_data.left_hand_pos.reshape(25, 3)
            right_hand_data = tele_data.right_hand_pos.reshape(25, 3)
            
            # Só calcula se a mão foi detectada no frame atual (evita crash na primeira iteração)
            if not np.all(right_hand_data == 0.0) and not np.all(left_hand_data[4] == np.array([-1.13, 0.3, 0.15])):
                
                # Subtrai as posições 3D usando a tabela de índices da Unitree
                ref_left_value = left_hand_data[self.hand_retargeter.left_indices[1,:]] - left_hand_data[self.hand_retargeter.left_indices[0,:]]
                ref_right_value = right_hand_data[self.hand_retargeter.right_indices[1,:]] - right_hand_data[self.hand_retargeter.right_indices[0,:]]

                # Executa o retargeting e remapeia para a ordem certa dos motores da mão
                left_hand_q = self.hand_retargeter.left_retargeting.retarget(ref_left_value)[self.hand_retargeter.left_dex_retargeting_to_hardware]
                right_hand_q = self.hand_retargeter.right_retargeting.retarget(ref_right_value)[self.hand_retargeter.right_dex_retargeting_to_hardware]

                # --- AJUSTE DE PINÇA (OFFSET) ---
                # Valor em radianos para forçar o fechamento. Você terá que testar se no seu robô
                # o sentido de fechar é somar (+) ou subtrair (-). Comece com 0.15 ou -0.15.
                OFFSET_ESQUEDA = 0.17 
                OFFSET_DIREITA = 0.25

                # Mão Esquerda (Ordem: Thumb 0,1,2, Middle 3,4, Index 5,6)
                left_hand_q[5] -= OFFSET_ESQUEDA # Base do indicador esquerdo
                left_hand_q[6] -= OFFSET_ESQUEDA # Ponta do indicador esquerdo

                # Mão Direita (Ordem diferente! Thumb 0,1,2, Index 3,4, Middle 5,6)
                right_hand_q[3] += OFFSET_DIREITA # Base do indicador direito
                right_hand_q[4] += OFFSET_DIREITA # Ponta do indicador direito
                # --------------------------------

                # Aplica as juntas calculadas no dicionário do LeRobot
                for i, name in enumerate(self._left_hand_names):
                    self.hand_joints[f"{name}.q"] = left_hand_q[i]

                for i, name in enumerate(self._right_hand_names):
                    self.hand_joints[f"{name}.q"] = right_hand_q[i]

        # Concatena os dicionários e retorna a Ação Final
        action_data = {**self.body_joints, **self.hand_joints}
        return action_data