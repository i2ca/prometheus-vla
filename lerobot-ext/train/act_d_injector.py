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
    def patched_forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        
        # --- A. EXTRAÇÃO DA PROFUNDIDADE E DA PRESSÃO ---
        # Fazemos '.pop()' para esconder estes dados da rede ResNet original
        depth_tensor = batch.pop("observation.images.head_camera_depth")
        left_pressure = batch.pop("observation.left_hand_pressure")
        right_pressure = batch.pop("observation.right_hand_pressure")
        
        # --- B. PROCESSAMENTO 3D (PointNet) ---
        pc = depth_to_pointcloud(depth_tensor, self.camera_intrinsics)
        features_3d = self.pointnet(pc) # Saída: [Batch, hidden_dim]

        # --- C. PROCESSAMENTO TÁTIL (Pressão) ---
        # Junta as duas mãos num único vetor de 66 posições
        full_pressure = torch.cat([left_pressure, right_pressure], dim=1) 
        features_pressure = self.pressure_proj(full_pressure) # Saída: [Batch, hidden_dim]

        # --- D. HACK DA FUSÃO MULTIMODAL ---
        original_proj = self.input_proj_env_state
        
        # Temporariamente sobrescrevemos o projetor linear do estado dos motores
        def patched_proj(env_state):
            state_token = original_proj(env_state) # Token original (Motores)
            
            # 🧠 FUSÃO SUPREMA: 
            # Somamos a perceção geométrica (3D) e a perceção tátil (Pressão)
            # à propriocepção do robô (Motores).
            fused_token = state_token + features_3d + features_pressure 
            return fused_token
            
        self.input_proj_env_state = patched_proj
        
        # --- E. EXECUÇÃO DO FORWARD ORIGINAL ---
        output = self.original_forward(batch)
        
        # Devolvemos as chaves ao batch para não quebrar o cálculo de logs e estatísticas
        batch["observation.images.head_camera_depth"] = depth_tensor
        batch["observation.left_hand_pressure"] = left_pressure
        batch["observation.right_hand_pressure"] = right_pressure
        
        # Restauramos o projetor original para a próxima iteração
        self.input_proj_env_state = original_proj
        
        return output

    # Aplica o Monkey Patch final no objeto
    policy.forward = types.MethodType(patched_forward, policy)
    print("✅ [INJEÇÃO ACT-D]: Concluída com Sucesso! Visão 3D e Tato operacionais.\n")