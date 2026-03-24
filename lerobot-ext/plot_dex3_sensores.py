import time
import threading
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# =========================================================
# IMPORTAÇÕES DA SUA PONTE ZMQ (Em vez do SDK da Unitree)
# =========================================================

from robot.unitree_g1.unitree_sdk2_socket import (
    ChannelFactoryInitialize, 
    ChannelSubscriber, 
    kTopicDex3LeftState, 
    kTopicDex3RightState
)

from robot.unitree_g1.g1_utils import (
    Dex3_1_Left_PressureTemperatureSensors,
    Dex3_1_Right_PressureTemperatureSensors,
    sensor_index
)

NUM_SENSORS = 33

left_pressure = np.zeros(NUM_SENSORS)
left_temperature = np.zeros(NUM_SENSORS)
right_pressure = np.zeros(NUM_SENSORS)
right_temperature = np.zeros(NUM_SENSORS)

def thread_escuta_dados():
    """Thread isolada que consome os dados do ZMQ a 100Hz"""
    global left_pressure, left_temperature, right_pressure, right_temperature
    
    # Inicializa a ponte ZMQ conectando ao localhost (ou IP do robô)
    print("⏳ Conectando à ponte ZMQ...")
    ChannelFactoryInitialize(0, "127.0.0.1")
    
    sub_left = ChannelSubscriber(kTopicDex3LeftState, None)
    sub_left.Init()
    
    sub_right = ChannelSubscriber(kTopicDex3RightState, None)
    sub_right.Init()

    print("📡 Conectado! Escutando dados do ZMQ...")

    while True:
        # --- MÃO ESQUERDA ---
        l_msg = sub_left.Read()
        if l_msg is not None and len(l_msg.press_sensor_state) > 0:
            idx = 0
            for area_idx, id in enumerate(Dex3_1_Left_PressureTemperatureSensors):
                for s_idx in sensor_index[id]:
                    if area_idx < len(l_msg.press_sensor_state):
                        try:
                            left_pressure[idx] = l_msg.press_sensor_state[area_idx].pressure[s_idx]
                            left_temperature[idx] = l_msg.press_sensor_state[area_idx].temperature[s_idx]
                        except IndexError:
                            pass
                    idx += 1

        # --- MÃO DIREITA ---
        r_msg = sub_right.Read()
        if r_msg is not None and len(r_msg.press_sensor_state) > 0:
            idx = 0
            for area_idx, id in enumerate(Dex3_1_Right_PressureTemperatureSensors):
                for s_idx in sensor_index[id]:
                    if area_idx < len(r_msg.press_sensor_state):
                        try:
                            right_pressure[idx] = r_msg.press_sensor_state[area_idx].pressure[s_idx]
                            right_temperature[idx] = r_msg.press_sensor_state[area_idx].temperature[s_idx]
                        except IndexError:
                            pass
                    idx += 1
        
        # Mantém a leitura a 100Hz
        time.sleep(0.01)

def iniciar_dashboard():
    # Inicia a thread que abastece as arrays de forma isolada
    t = threading.Thread(target=thread_escuta_dados, daemon=True)
    t.start()
    
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig.canvas.manager.set_window_title('Painel Tátil - Unitree G1 Dex3')
    
    x_pos = np.arange(NUM_SENSORS)
    
    # === GRÁFICOS DE PRESSÃO ===
    axs[0, 0].set_title('Mão Esquerda - Pressão')
    axs[0, 0].set_ylim(0, 1.0)  # Começa com teto baixo para toques leves
    axs[0, 0].set_ylabel('Força')
    bars_lp = axs[0, 0].bar(x_pos, left_pressure, color='blue')
    
    axs[0, 1].set_title('Mão Direita - Pressão')
    axs[0, 1].set_ylim(0, 1.0)
    bars_rp = axs[0, 1].bar(x_pos, right_pressure, color='blue')
    
    # === GRÁFICOS DE TEMPERATURA ===
    axs[1, 0].set_title('Mão Esquerda - Temperatura (°C)')
    axs[1, 0].set_ylim(20, 30) 
    axs[1, 0].set_ylabel('Graus °C')
    bars_lt = axs[1, 0].bar(x_pos, left_temperature, color='red')
    
    axs[1, 1].set_title('Mão Direita - Temperatura (°C)')
    axs[1, 1].set_ylim(20, 30)
    bars_rt = axs[1, 1].bar(x_pos, right_temperature, color='red')

    def update_plot(frame):
        # ESCALA DINÂMICA: Ajusta o eixo Y sozinho se a pressão for forte!
        max_press = max(1.0, np.max(left_pressure), np.max(right_pressure))
        axs[0, 0].set_ylim(0, max_press * 1.1)
        axs[0, 1].set_ylim(0, max_press * 1.1)

        for bar, val in zip(bars_lp, left_pressure): bar.set_height(val)
        for bar, val in zip(bars_rp, right_pressure): bar.set_height(val)
        for bar, val in zip(bars_lt, left_temperature): bar.set_height(val)
        for bar, val in zip(bars_rt, right_temperature): bar.set_height(val)
            
        return bars_lp.patches + bars_rp.patches + bars_lt.patches + bars_rt.patches

    ani = FuncAnimation(fig, update_plot, interval=100, blit=False, cache_frame_data=False)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    iniciar_dashboard()