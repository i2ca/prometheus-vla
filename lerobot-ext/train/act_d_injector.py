import types
import torch
import torch.nn as nn
from train.depth_encoder import PointNetEncoder, depth_to_pointcloud

class FusedProjector(nn.Module):
    def __init__(self, original_proj):
        super().__init__()
        self.original_proj = original_proj
        self.features_3d = 0
        self.features_pressure = 0

    def forward(self, robot_state):
        state_token = self.original_proj(robot_state)
        return state_token + self.features_3d + self.features_pressure

def inject_act_d(policy, device):
    print("\n💉 [INJEÇÃO ACT-D]: Ativando Bypass Geométrico 3D e Fusão Tátil (Pressão Dex3)...")

    # 1. Pega o projetor original dos motores do LeRobot
    original_proj = policy.model.encoder_robot_state_input_proj
    hidden_dim = original_proj.out_features

    # 2. Cria as nossas novas redes (Visão 3D e Pressão)
    policy.pointnet = PointNetEncoder(output_dim=hidden_dim).to(device)
    policy.camera_intrinsics = {'fx': 600.0, 'fy': 600.0, 'cx': 320.0, 'cy': 240.0}
    
    policy.pressure_proj = nn.Sequential(
        nn.Linear(66, 256),
        nn.ReLU(),
        nn.Linear(256, hidden_dim)
    ).to(device)

    # 3. Cria a nossa "Camada Mutante" e substitui a original DE VEZ
    fused_layer = FusedProjector(original_proj)
    policy.model.encoder_robot_state_input_proj = fused_layer

    # Guarda o método forward original da política inteira
    policy.original_forward = policy.forward

    # 4. A Função Interceptadora Suprema
    def patched_forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        
        # --- A. ROUBO DE DADOS ---
        depth_tensor = batch.pop("observation.images.head_camera_depth", None)
        left_pressure = batch.pop("observation.left_hand_pressure", None)
        right_pressure = batch.pop("observation.right_hand_pressure", None)
        
        # --- B. PROCESSAMENTO MUTAÇÃO ---
        f3d = 0
        fpress = 0

        if depth_tensor is not None:
            pc = depth_to_pointcloud(depth_tensor, self.camera_intrinsics)
            f3d = self.pointnet(pc)

        if left_pressure is not None and right_pressure is not None:
            full_pressure = torch.cat([left_pressure, right_pressure], dim=1) 
            fpress = self.pressure_proj(full_pressure)

        # --- C. FUSÃO ---
        self.model.encoder_robot_state_input_proj.features_3d = f3d
        self.model.encoder_robot_state_input_proj.features_pressure = fpress
        
        # =================================================================
        # O SEGREDO MÁGICO: Esconde o Depth do LeRobot por 1 milissegundo
        # Isso ataca a raiz do problema (input_features) temporariamente
        # =================================================================
        hidden_depth_config = None
        if "observation.images.head_camera_depth" in self.config.input_features:
            hidden_depth_config = self.config.input_features.pop("observation.images.head_camera_depth")
        
        # --- D. EXECUÇÃO ORIGINAL ---
        # Agora o LeRobot processa o RGB sem procurar pelo Depth no batch!
        output = self.original_forward(batch)
        
        # =================================================================
        # RESTAURA A CONFIGURAÇÃO ANTES QUE OS LOGS PERCEBAM
        # =================================================================
        if hidden_depth_config is not None:
            self.config.input_features["observation.images.head_camera_depth"] = hidden_depth_config
        
        # --- E. DEVOLVE OS DADOS PARA O BATCH ---
        if depth_tensor is not None:
            batch["observation.images.head_camera_depth"] = depth_tensor
        if left_pressure is not None:
            batch["observation.left_hand_pressure"] = left_pressure
        if right_pressure is not None:
            batch["observation.right_hand_pressure"] = right_pressure
        
        # Reseta as features na camada falsa para a próxima iteração
        self.model.encoder_robot_state_input_proj.features_3d = 0
        self.model.encoder_robot_state_input_proj.features_pressure = 0
        
        return output

    # Aplica o Monkey Patch final no método forward
    policy.forward = types.MethodType(patched_forward, policy)
    print("✅ [INJEÇÃO ACT-D]: Concluída com Sucesso! Visão 3D e Tato operacionais.\n")