import os
import sys
import time
import torch
import numpy as np
from safetensors.torch import load_file

# Importa o seu robô e as ferramentas do LeRobot
from robot.unitree_g1.unitree_g1_dex3 import UnitreeG1Dex3
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.configs.policies import PreTrainedConfig
from train.act_d_injector import inject_act_d, depth_to_pointcloud

def load_mutant_policy(checkpoint_dir, device):
    print(f"⏳ Carregando ACT-D da pasta: {checkpoint_dir}")
    
    # 1. Carrega apenas as configurações estruturais originais
    config_path = os.path.join(checkpoint_dir, "config.json")
    config = PreTrainedConfig.from_pretrained(checkpoint_dir)
    
    # 2. Instancia a Política Pura (Base ACT)
    policy = ACTPolicy(config)
    
    # 3. Aplica a Mutação (Cria a PointNet e a Camada de Pressão na memória)
    inject_act_d(policy, device)
    
    # 4. Carrega os Pesos Treinados ignorando conflitos estritos (strict=False)
    # Isso permite que os pesos da PointNet entrem nas variáveis que acabamos de injetar
    model_file = os.path.join(checkpoint_dir, "model.safetensors")
    state_dict = load_file(model_file)
    policy.load_state_dict(state_dict, strict=False)
    
    # 5. Patch na função Select Action (Para inferência em tempo real)
    policy.original_select_action = policy.select_action
    
    def patched_select_action(self, batch):
        # Rouba as modalidades extras para a ResNet não bugar
        depth_tensor = batch.pop("observation.images.head_camera_depth")
        left_pressure = batch.pop("observation.left_hand_pressure")
        right_pressure = batch.pop("observation.right_hand_pressure")
        
        # Processa a Nuvem de Pontos e a Pressão
        pc = depth_to_pointcloud(depth_tensor, self.camera_intrinsics)
        features_3d = self.pointnet(pc)
        full_pressure = torch.cat([left_pressure, right_pressure], dim=1)
        features_pressure = self.pressure_proj(full_pressure)
        
        # Hack do Projetor: Soma a visão 3D e Tato aos motores
        original_proj = self.input_proj_env_state
        def temp_proj(env_state):
            return original_proj(env_state) + features_3d + features_pressure
        self.input_proj_env_state = temp_proj
        
        # Roda a inferência original limpa
        action = self.original_select_action(batch)
        
        # Restaura o sistema
        self.input_proj_env_state = original_proj
        return action

    # Aplica o patch
    import types
    policy.select_action = types.MethodType(patched_select_action, policy)
    
    policy.eval()
    policy.to(device)
    print("✅ ACT-D Carregado e Pronto!")
    return policy

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # === 1. CONFIGURAÇÕES ===
    # Substitua 'last' pelo número do checkpoint ideal (ex: 'checkpoint_025000') se o último estiver com overfitting
    checkpoint_dir = "train_output/act_d_push_cup/checkpoints/last/pretrained_model" 
    
    # === 2. INICIALIZA A REDE NEURAL ===
    policy = load_mutant_policy(checkpoint_dir, device)
    
    # === 3. INICIALIZA O ROBÔ ===
    print("⏳ Conectando ao Unitree G1...")
    # Ajuste os IPs conforme a sua configuração de teleoperação
    robot = UnitreeG1Dex3(
        ip="192.168.123.164", # IP do G1 na sua rede ZeroTier/ZMQ
        control_mode="upper_body",
        use_hands=True
    )
    robot.connect()
    print("✅ Robô Conectado!")
    
    print("\n🚀 INICIANDO INFERÊNCIA EM 3 SEGUNDOS...")
    time.sleep(3)
    
    # === 4. O LOOP DE CONTROLE (O Cérebro em Ação) ===
    try:
        while True:
            start_t = time.perf_counter()
            
            # A. Lê os sensores do mundo real
            obs = robot.get_observation()
            
            # B. Formata os dados para o formato PyTorch [Batch, ...]
            batch = {}
            for key, value in obs.items():
                if isinstance(value, np.ndarray):
                    tensor = torch.from_numpy(value).float().to(device)
                else:
                    tensor = value.clone().to(device)
                batch[key] = tensor.unsqueeze(0) # Adiciona a dimensão do Batch=1
            
            # C. A IA pensa e prevê a próxima ação
            with torch.inference_mode(), torch.autocast(device_type=device.type):
                action = policy.select_action(batch)
            
            # D. Extrai o vetor de ação e envia para os motores
            action_numpy = action.squeeze(0).cpu().numpy()
            robot.send_action(action_numpy)
            
            # Controle de frequência (O ACT geralmente roda a 50Hz)
            elapsed = time.perf_counter() - start_t
            sleep_time = max(0, 0.02 - elapsed) # 0.02s = 50Hz
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n🛑 Inferência interrompida pelo usuário.")
    finally:
        robot.disconnect()
        print("🔌 Motores desativados. Sistema seguro.")

if __name__ == "__main__":
    main()