import time, zmq, json, threading
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandCmd_, HandState_
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__HandCmd_ as HandCmd_default

class PonteG1Completa:
    def __init__(self):
        # 1. DDS Setup (Fala com MuJoCo)
        ChannelFactoryInitialize(0, "lo")
        
        # Publishers (Ponte -> MuJoCo)
        self.dds_left_pub = ChannelPublisher("rt/dex3/left/cmd", HandCmd_)
        self.dds_right_pub = ChannelPublisher("rt/dex3/right/cmd", HandCmd_)
        self.dds_left_pub.Init(); self.dds_right_pub.Init()

        # Subscribers (MuJoCo -> Ponte)
        self.dds_left_sub = ChannelSubscriber("rt/dex3/left/state", HandState_)
        self.dds_right_sub = ChannelSubscriber("rt/dex3/right/state", HandState_)
        self.dds_left_sub.Init(self.callback_left_state, 1)
        self.dds_right_sub.Init(self.callback_right_state, 1)

        # 2. ZMQ Setup (Fala com LeRobot)
        self.ctx = zmq.Context()
        # Porta 6003: Recebe comandos do LeRobot
        self.zmq_pull = self.ctx.socket(zmq.PULL)
        self.zmq_pull.bind("tcp://127.0.0.1:6003")
        # Porta 6002: Envia estado para o LeRobot
        self.zmq_pub = self.ctx.socket(zmq.PUB)
        self.zmq_pub.bind("tcp://127.0.0.1:6002")
        
        print("[OK] Ponte G1 Ativa! Esperando LeRobot em 127.0.0.1...")

    def callback_left_state(self, msg):
        self._enviar_zmq("rt/dex3/left/state", msg)

    def callback_right_state(self, msg):
        self._enviar_zmq("rt/dex3/right/state", msg)

    def _enviar_zmq(self, topic, msg):
        """Converte DDS para JSON e manda pro LeRobot"""
        data = {
            "topic": topic,
            "data": {"motor_state": [{"q": m.q, "dq": m.dq} for m in msg.motor_state]}
        }
        self.zmq_pub.send_string(json.dumps(data))

    def rodar_loop_comandos(self):
        """Recebe do LeRobot e injeta no MuJoCo"""
        while True:
            try:
                msg_json = self.zmq_pull.recv_string()
                payload = json.loads(msg_json)
                topic = payload.get("topic", "")
                
                cmd_dds = HandCmd_default()
                for i, m_data in enumerate(payload['data']['motor_cmd']):
                    if i < len(cmd_dds.motor_cmd):
                        cmd_dds.motor_cmd[i].q = float(m_data['q'])
                        cmd_dds.motor_cmd[i].kp = float(m_data.get('kp', 20.0))
                        cmd_dds.motor_cmd[i].kd = float(m_data.get('kd', 1.0))
                        cmd_dds.motor_cmd[i].mode = 1

                if "left" in topic:
                    self.dds_left_pub.Write(cmd_dds)
                else:
                    self.dds_right_pub.Write(cmd_dds)
            except Exception as e:
                print(f"Erro: {e}")

if __name__ == "__main__":
    p = PonteG1Completa()
    threading.Thread(target=p.rodar_loop_comandos, daemon=True).start()
    while True: time.sleep(1)