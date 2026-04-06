"""
Script para testar a IA treinada no robô físico ou no Simulador MuJoCo.
"""
import os
import sys
import time
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import collections

# Registra os módulos do G1
try:
    import robot.unitree_g1
except ImportError as e:
    print(f"Erro ao carregar módulos: {e}")
    sys.exit(1)

from lerobot.policies.act.modeling_act import ACTPolicy
from robot.unitree_g1.unitree_g1_dex3 import UnitreeG1Dex3, UnitreeG1Dex3Config

def parse_args():
    parser = argparse.ArgumentParser(description="Inicia a inferência da Rede Neural no Unitree G1.")
    
    # 1. Argumento Obrigatório: Caminho do Modelo
    parser.add_argument(
        "--checkpoint_path", 
        type=str, 
        required=True,
        help="Caminho exato para a pasta 'pretrained_model' do seu treinamento."
    )
    
    # 2. Argumento Opcional: Flag de Simulação
    parser.add_argument(
        "--sim", 
        action="store_true", 
        help="Se ativado, roda no simulador MuJoCo. Se omitido, conecta no robô FÍSICO."
    )
    
    # 3. Argumento Opcional: IP do Robô
    parser.add_argument(
        "--robot_ip", 
        type=str, 
        default="127.0.0.1", 
        help="Endereço IP do robô ou da ponte ZMQ (Padrão: 127.0.0.1)"
    )
    
    return parser.parse_args()

def main():
    # Lê os argumentos passados no terminal
    args = parse_args()
    CHECKPOINT_PATH = args.checkpoint_path

    if not os.path.exists(CHECKPOINT_PATH):
        print(f"\n[ERRO] Checkpoint não encontrado: {CHECKPOINT_PATH}")
        print("Dica: Lembre-se de colocar a subpasta '/pretrained_model' no final do caminho.")
        sys.exit(1)

    # Define os avisos baseados na flag --sim
    modo_texto = "Simulação (MuJoCo)" if args.sim else "ROBÔ FÍSICO REAL (CUIDADO!)"
    print(f"\n[1] Iniciando o sistema no modo: {modo_texto}")
    
    # Repassa as configurações dinâmicas para a classe do G1
    config = UnitreeG1Dex3Config(is_simulation=args.sim, robot_ip=args.robot_ip)
    robot = UnitreeG1Dex3(config)
    robot.connect()

    print(f"[2] Carregando a Rede Neural ACT do diretório fornecido...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        policy = ACTPolicy.from_pretrained(CHECKPOINT_PATH)
        policy.to(device)
        policy.eval()
    except Exception as e:
        print(f"\n[ERRO] Falha ao carregar os pesos da IA: {e}")
        robot.disconnect()
        sys.exit(1)

    print("\n[3] IA Pronta! A Rede Neural assumiu o controle. (Pressione Ctrl+C para parada de emergência)\n")
    
    historico_chunks = []

    try:
        while True:
            start_time = time.time()

            # A. Lê as câmeras e posição dos motores
            obs = robot.get_observation()
            
            # B. Prepara os dados para o formato PyTorch (Tradução para o LeRobot)
            torch_obs = {}
            
            # 1. CÂMERAS DINÂMICAS: Lê apenas as câmeras que estão descomentadas no unitree_g1_dex3.py
            for cam_name in robot.cameras.keys(): 
                if cam_name in obs:
                    # O LeRobot espera formato (C, H, W) e valores entre 0.0 e 1.0
                    img = torch.from_numpy(obs[cam_name]).permute(2, 0, 1).float() / 255.0
                    # Adiciona o prefixo exato que a rede neural espera
                    torch_obs[f"observation.images.{cam_name}"] = img.unsqueeze(0).to(device)

            # 2. ESTADO DINÂMICO: Pega exatamente a lista de sensores ativos (pressão, temperatura, motores)
            # Como você varre observation_features.keys(), ele se adapta sozinho se você esconder o braço direito!
            state_keys = [k for k in robot.observation_features.keys() if "camera" not in k]
            
            # Puxa os valores do dicionário de observação na mesma ordem
            state_values = [float(obs[k]) for k in state_keys]
            
            # Converte para o tensor 1D que a Inteligência Artificial consome
            torch_obs["observation.state"] = torch.tensor(state_values, dtype=torch.float32).unsqueeze(0).to(device)

            # C. A IA analisa a imagem e decide os ângulos
            with torch.inference_mode():
                action = policy.select_action(torch_obs)

                # 👈 NOVA LINHA: Roubamos os 100 passos do futuro que a IA previu!
                raw_chunk = policy.predict_action_chunk(torch_obs)[0].cpu().numpy()
                historico_chunks.append(raw_chunk)
                
                if isinstance(action, dict):
                    action = action["action"]
                    
                action_array = action.squeeze().cpu().numpy()
                
                if len(action_array.shape) > 1:
                    action_array = action_array[0] 

            # D. Devolve os valores para o dicionário do robô
            action_keys = list(robot.action_features.keys())
            action_dict = {}
            for i, joint_name in enumerate(action_keys):
                action_dict[joint_name] = float(action_array[i])

            # E. Envia para o robô se mexer
            robot.send_action(action_dict)

            # Mantém um limite de ~30 FPS
            elapsed = time.time() - start_time
            time.sleep(max(0, 0.03 - elapsed))

    except KeyboardInterrupt:
        print("\n[INFO] Parada solicitada pelo usuário. Desligando...")
    finally:
        robot.disconnect()
        print("\n📊 Robô seguro! Gerando Gráfico de Incerteza do ACT...")
        
        if len(historico_chunks) > 0:
            plt.figure(figsize=(12, 6))
            
            # Escolha qual motor você quer investigar. 
            # 0 = Pitch do Ombro Esquerdo (Geralmente o que mais flutua no braço)
            junta_alvo = 0 
            
            # Plota todas as sobreposições de futuro
            for t, chunk in enumerate(historico_chunks):
                # chunk tem shape [100, 28] (100 passos no futuro, 28 juntas)
                eixo_tempo = range(t, t + len(chunk))
                # Usamos alpha=0.05 para deixar as linhas transparentes. 
                # Onde elas concordam, a cor fica forte. Onde discordam, fica uma nuvem borrada.
                plt.plot(eixo_tempo, chunk[:, junta_alvo], color='blue', alpha=0.05)
                
            plt.title("Visão Interna da IA: Incerteza do Braço Esquerdo")
            plt.xlabel("Passos de Tempo (Eixo X = Tempo contínuo)")
            plt.ylabel("Ângulo Desejado do Motor (Radianos)")
            plt.grid(True)
            
            print("Pressione X na janela do gráfico para encerrar o programa totalmente.")
            plt.show()

if __name__ == "__main__":
    main()