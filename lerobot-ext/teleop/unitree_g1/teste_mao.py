import time
from televuer import VuerTeleop, VuerTeleopConfig
import numpy as np

# Ignorando a formatação de print científico do numpy para ficar mais limpo
np.set_printoptions(suppress=True, precision=3)

def main():
    print("Configurando Teleoperador...")
    # Configuração base (sem webrtc/ngrok, focando no adb localhost)
    config = VuerTeleopConfig(
        binocular=False,
        use_hand_tracking=True,
        img_shape=(480, 640, 3),
        # Se precisar do dummy de memória compartilhada para não dar erro:
        img_shm_name="dummy_img_shm" 
    )

    teleop = VuerTeleop(config)

    print("\nIniciando servidor Vuer...")
    print("-> Coloque o Quest 3 e acesse: http://localhost:8012")
    teleop.connect()

    print("\nAguardando conexão do WebXR... (Pressione Ctrl+C para sair)\n")
    
    try:
        while True:
            # Puxa o dicionário de ações gerado pelo seu teleoperador
            action = teleop.get_action()

            # Pega apenas a coordenada do pulso (índice 0) das matrizes (25, 3)
            left_wrist = action["left_hand_pos"][0]
            right_wrist = action["right_hand_pos"][0]
            
            # Pega o valor da pinça (dedão e indicador)
            left_pinch = action.get("left_pinch_value", 0.0)

            # Imprime no console (sobrescrevendo a linha para ficar mais limpo)
            print(f"\rMão Esq [X: {left_wrist[0]:.2f}, Y: {left_wrist[1]:.2f}, Z: {left_wrist[2]:.2f}] | Pinça Esq: {left_pinch:05.1f}   ", end="")
            
            # Roda a 10Hz (10x por segundo) para não travar o terminal
            time.sleep(0.1) 
            
    except KeyboardInterrupt:
        print("\n\nEncerrando servidor...")
        teleop.disconnect()
        print("Finalizado com sucesso.")

if __name__ == "__main__":
    main()