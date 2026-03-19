import cv2
import numpy as np
import matplotlib.pyplot as plt
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# CONFIGURAÇÃO: Caminho local confirmado pelo seu comando 'tree'
REPO_ID = "Mrwlker/g1_pick_kettle_custom"
ROOT_PATH = "meu_dataset"

print("🔍 Carregando dataset...")
# O LeRobotDataset vai tentar usar pyav automaticamente se o torchcodec falhar
dataset = LeRobotDataset(REPO_ID, root=ROOT_PATH)

# Escolha um frame onde você sabe que houve interação física (ex: frame 300)
frame_idx = 300 
if frame_idx >= len(dataset):
    frame_idx = len(dataset) - 1

frame_data = dataset[frame_idx]

print(f"\n=== RELATÓRIO DO DATASET PROMETHEUS ===")
print(f"📊 Total de frames: {len(dataset)}")
print(f"📌 Chaves gravadas: {list(frame_data.keys())}")

# --- VALIDAÇÃO DOS MOTORES E TATO ---
state = frame_data["observation.state"].numpy()
print(f"📏 Tamanho do vetor 'state': {len(state)} colunas")

# Cálculo dos índices baseado na nossa injeção de 109 colunas:
# 29 (corpo) + 14 (mãos) + 33 (pressão esq) + 33 (pressão dir) = 109
if len(state) >= 109:
    # Fatiamento dos dados táteis injetados no final do vetor
    tato_esq = state[-66:-33]
    tato_dir = state[-33:]
    
    print(f"\n--- Diagnóstico de Tato (Frame {frame_idx}) ---")
    print(f"✋ Mão Esquerda: Max {np.max(tato_esq):.2f} | Média {np.mean(tato_esq):.2f}")
    print(f"✋ Mão Direita:  Max {np.max(tato_dir):.2f} | Média {np.mean(tato_dir):.2f}")

    # Criando gráfico de barras para ver o reconhecimento de cada sensor
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.bar(range(33), tato_esq, color='blue', alpha=0.7)
    plt.title("Sensores Mão Esquerda")
    plt.ylabel("Pressão")
    
    plt.subplot(1, 2, 2)
    plt.bar(range(33), tato_dir, color='red', alpha=0.7)
    plt.title("Sensores Mão Direita")
    
    plt.suptitle(f"Reconhecimento Tátil - Frame {frame_idx}")
    print("📈 Gerando gráfico de barras dos sensores... (Feche o gráfico para ver a imagem)")
    plt.show()
else:
    print("\n⚠️ Aviso: O vetor de estado não contém as 109 colunas esperadas para o tato.")

# --- VISUALIZAÇÃO DA CÂMERA (BACKEND CV2/PYAV) ---
chave_cam = "observation.images.cam_rgb_high"
if chave_cam in frame_data:
    # Converte o Tensor (C, H, W) para formato OpenCV (H, W, C)
    img_rgb = frame_data[chave_cam].permute(1, 2, 0).numpy()
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    
    print(f"\n🖼️ Exibindo visão do robô: {chave_cam}")
    cv2.putText(img_bgr, f"Frame: {frame_idx}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow("Visao do Robo - Prometheus VLA", img_bgr)
    print("⌨️  Pressione QUALQUER TECLA na janela da imagem para encerrar.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print(f"\n❌ Erro: Chave de imagem '{chave_cam}' não encontrada.")