import torch
import torch.nn as nn
import torch.nn.functional as F

class PointNetEncoder(nn.Module):
    """Codificador 3D para extrair features globais da Nuvem de Pontos (ACT-D)"""
    def __init__(self, output_dim=512):
        super().__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, output_dim)

    def forward(self, x):
        # x shape: [Batch, 3, Num_Points]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        # Max Pooling (Invariância espacial)
        x = torch.max(x, 2, keepdim=True)[0].view(-1, 1024)
        x = F.relu(self.fc1(x))
        return self.fc2(x) # [Batch, output_dim]

def depth_to_pointcloud(depth_tensor, intrinsics, num_points=1024):
    """Reverte o Hack ZMQ e projeta os pixels no espaço 3D real"""
    B, C, H, W = depth_tensor.shape
    device = depth_tensor.device

    # 1. Cria a malha de pixels
    grid_y, grid_x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    grid_x = grid_x.float().unsqueeze(0).expand(B, -1, -1)
    grid_y = grid_y.float().unsqueeze(0).expand(B, -1, -1)

    # 2. REVERSÃO MATEMÁTICA DO SEU HACK ZMQ (Recupera metros reais!)
    # O tensor chega entre 0 e 1. Como 1.0 = 2000mm (2 metros), multiplicamos por 2.0.
    z = depth_tensor[:, 0, :, :] * 2.0 
    
    # 3. Projeção Pinhole 3D
    fx, fy, cx, cy = intrinsics['fx'], intrinsics['fy'], intrinsics['cx'], intrinsics['cy']
    x = (grid_x - cx) * z / fx
    y = (grid_y - cy) * z / fy

    # 4. Agrupa e Amostra (Sampling de 1024 pontos para a PointNet não explodir a VRAM)
    point_cloud = torch.stack((x, y, z), dim=1).view(B, 3, -1)
    
    sampled_pcs = []
    for b in range(B):
        pc = point_cloud[b]
        valid_mask = pc[2, :] > 0.05 # Ignora ruído na lente (< 5cm)
        valid_pc = pc[:, valid_mask]
        
        if valid_pc.shape[1] > num_points:
            indices = torch.randperm(valid_pc.shape[1], device=device)[:num_points]
            sampled_pcs.append(valid_pc[:, indices])
        else:
            pad = torch.zeros((3, num_points - valid_pc.shape[1]), device=device)
            sampled_pcs.append(torch.cat([valid_pc, pad], dim=1))

    return torch.stack(sampled_pcs)