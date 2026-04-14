import types
import torch
import torch.nn as nn
from depth_encoder import PointNetEncoder, depth_to_pointcloud

def inject_act_d(policy, device):
    print("\n💉 [INJEÇÃO ACT-D]: Ativando Bypass Geométrico 3D e Fusão Tátil (Pressão Dex3)...")

    # 1. Descobrimos o tamanho do "Cérebro" do Transformer dinamicamente (geralmente 512)
    hidden_dim = policy.input_proj_env_state.out_features

    # 2. Acopla a PointNet (Visão 3D) na Memória da Política
    policy.pointnet = PointNetEncoder(output_dim=hidden_dim).to(device)
    policy.camera_intrinsics = {'fx': 600.0, 'fy': 600.0, 'cx': 320.0, 'cy': 240.0}

    # 3. Acopla a Camada Tátil (Pressão) na Memória da Política
    # 33 sensores na esquerda + 33 na direita = 66 entradas.
    policy.pressure_proj = nn.Sequential(
        nn.Linear(66, 256),
        nn.ReLU(),
        nn.Linear(256, hidden_dim)
    ).to(device)

    # Guarda o método forward original
    policy.original_forward = policy.forward

    # 4. A Função Intercetadora Suprema
    # 4. A Função Intercetadora Suprema (AGORA DINÂMICA)
    def patched_forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        
        # --- A. EXTRAÇÃO SEGURA (Prevenção de KeyError) ---
        # Usa None como valor padrão caso as chaves estejam comentadas no YAML
        depth_tensor = batch.pop("observation.images.head_camera_depth", None)
        left_pressure = batch.pop("observation.left_hand_pressure", None)
        right_pressure = batch.pop("observation.right_hand_pressure", None)
        
        # Inicia as features extras como 0 (neutras)
        features_3d = 0
        features_pressure = 0

        # --- B. PROCESSAMENTO 3D (Só roda se o depth existir) ---
        if depth_tensor is not None:
            pc = depth_to_pointcloud(depth_tensor, self.camera_intrinsics)
            features_3d = self.pointnet(pc) # Saída: [Batch, hidden_dim]

        # --- C. PROCESSAMENTO TÁTIL (Só roda se a pressão existir) ---
        if left_pressure is not None and right_pressure is not None:
            full_pressure = torch.cat([left_pressure, right_pressure], dim=1) 
            features_pressure = self.pressure_proj(full_pressure) # Saída: [Batch, hidden_dim]

        # --- D. HACK DA FUSÃO MULTIMODAL ---
        original_proj = self.input_proj_env_state
        
        def patched_proj(env_state):
            state_token = original_proj(env_state) # Token original (Motores)
            
            # Se a IA for cega pro 3D e Tato (comentados no YAML), features serão 0
            # state_token + 0 + 0 = state_token normal!
            fused_token = state_token + features_3d + features_pressure 
            return fused_token
            
        self.input_proj_env_state = patched_proj
        
        # --- E. EXECUÇÃO DO FORWARD ORIGINAL ---
        output = self.original_forward(batch)
        
        # Devolvemos as chaves ao batch apenas se elas existirem
        if depth_tensor is not None:
            batch["observation.images.head_camera_depth"] = depth_tensor
        if left_pressure is not None:
            batch["observation.left_hand_pressure"] = left_pressure
        if right_pressure is not None:
            batch["observation.right_hand_pressure"] = right_pressure
        
        # Restauramos o projetor original
        self.input_proj_env_state = original_proj
        
        return output

    # Aplica o Monkey Patch final no objeto
    policy.forward = types.MethodType(patched_forward, policy)
    print("✅ [INJEÇÃO ACT-D]: Concluída com Sucesso! Visão 3D e Tato operacionais.\n")