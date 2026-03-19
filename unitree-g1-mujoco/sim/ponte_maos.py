import time
import zmq
import json
import threading
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandCmd_, HandState_

# Configurações de Portas (Igual ao robô real)
HANDSTATE_PORT = 6002 # Onde a ponte PUBLICA o estado (ZMQ PUB)
HANDCMD_PORT = 6003   # Onde a ponte RECEBE comandos (ZMQ PULL)

class PonteZmqDds:
    def __init__(self, interface="lo"):
        # 1. Inicializa DDS (Fala com o MuJoCo)
        ChannelFactoryInitialize(0, interface)
        self.dds_pub = ChannelPublisher("rt/dex3/left/cmd", HandCmd_)
        self.dds_pub.Init()
        self.dds_sub = ChannelSubscriber("rt/dex3/left/state", HandState_)
        self.dds_sub.Init(self.dds_callback, 1)

        # 2. Inicializa ZMQ (Fala com o LeRobot)
        self.ctx = zmq.Context()
        
        # Socket para ENVIAR estado para o LeRobot (PUB)
        self.zmq_pub = self.ctx.socket(zmq.PUB)
        self.zmq_pub.bind(f"tcp://*: {HANDSTATE_PORT}") # BIND para abrir a porta
        
        # Socket para RECEBER comandos do LeRobot (PULL)
        self.zmq_pull = self.ctx.socket(zmq.PULL)
        self.zmq_pull.bind(f"tcp://*: {HANDCMD_PORT}") # BIND para abrir a porta
        
        self.last_dds_msg = None
        print(f"[*] Ponte Ativa! Portas ZMQ {HANDSTATE_PORT} e {HANDCMD_PORT} abertas.")

    def dds_callback(self, msg):
        """Recebe do MuJoCo (DDS) e manda pro LeRobot (ZMQ)"""
        # Converte a mensagem DDS para um dicionário simples (formato que o LeRobot entende)
        data = {
            "topic": "rt/dex3/left/state",
            "data": {
                "motor_state": [{"q": m.q, "dq": m.dq} for m in msg.motor_state]
            }
        }
        self.zmq_pub.send_string(json.dumps(data))

    def rodar_receptor_comandos(self):
        """Recebe do LeRobot (ZMQ) e manda pro MuJoCo (DDS)"""
        while True:
            try:
                # O LeRobot manda um JSON via PUSH
                msg_json = self.zmq_pull.recv_string()
                payload = json.loads(msg_json)
                
                # Cria a mensagem DDS nativa para o simulador
                cmd_dds = HandCmd_()
                for i, m_data in enumerate(payload['data']['motor_cmd']):
                    cmd_dds.motor_cmd[i].q = m_data['q']
                    cmd_dds.motor_cmd[i].kp = m_data.get('kp', 0.0)
                    cmd_dds.motor_cmd[i].kd = m_data.get('kd', 0.0)
                
                self.dds_pub.Write(cmd_dds)
            except Exception as e:
                print(f"Erro na ponte: {e}")

if __name__ == "__main__":
    ponte = PonteZmqDds(interface="lo") # Use "lo" para local ou a interface do seu DDS
    t = threading.Thread(target=ponte.rodar_receptor_comandos, daemon=True)
    t.start()
    
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt:
        print("Desligando ponte...")