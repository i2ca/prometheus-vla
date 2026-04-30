#!/usr/bin/env python3

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
DDS-to-ZMQ bridge server for Unitree G1 robot with Dex3 hands.
"""

import argparse
import base64
import contextlib
import json
import threading
import time
from typing import Any

import zmq
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_, unitree_hg_msg_dds__HandCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as hg_LowCmd, LowState_ as hg_LowState
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandCmd_, HandState_
from unitree_sdk2py.utils.crc import CRC

from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient

class MotionSwitcher:
    def __init__(self):
        self.msc = MotionSwitcherClient()
        self.msc.SetTimeout(1.0)
        self.msc.Init()

    def Enter_Debug_Mode(self):
        try:
            status, result = self.msc.CheckMode()
            while result['name']:
                self.msc.ReleaseMode()
                status, result = self.msc.CheckMode()
                time.sleep(1)
            return status, result
        except Exception as e:
            return None, None
    
    def Exit_Debug_Mode(self):
        try:
            status, result = self.msc.SelectMode(nameOrAlias='ai')
            return status, result
        except Exception as e:
            return None, None


kTopicLowCommand_Debug = "rt/lowcmd"
kTopicLowState = "rt/lowstate"
kTopicDex3LeftCommand = "rt/dex3/left/cmd"
kTopicDex3RightCommand = "rt/dex3/right/cmd"
kTopicDex3LeftState = "rt/dex3/left/state"
kTopicDex3RightState = "rt/dex3/right/state"

LOWCMD_PORT = 6000
LOWSTATE_PORT = 6001
HANDSTATE_PORT = 6002
HANDCMD_PORT = 6003

NUM_MOTORS = 35
NUM_HAND_MOTORS = 7


def lowstate_to_dict(msg: hg_LowState) -> dict[str, Any]:
    motor_states = []
    for i in range(NUM_MOTORS):
        temp = msg.motor_state[i].temperature
        avg_temp = float(sum(temp) / len(temp)) if isinstance(temp, list) else float(temp)
        motor_states.append({
            "q": float(msg.motor_state[i].q),
            "dq": float(msg.motor_state[i].dq),
            "tau_est": float(msg.motor_state[i].tau_est),
            "temperature": avg_temp,
        })
    return {
        "motor_state": motor_states,
        "imu_state": {
            "quaternion": [float(x) for x in msg.imu_state.quaternion],
            "gyroscope": [float(x) for x in msg.imu_state.gyroscope],
            "accelerometer": [float(x) for x in msg.imu_state.accelerometer],
            "rpy": [float(x) for x in msg.imu_state.rpy],
            "temperature": float(msg.imu_state.temperature),
        },
        "wireless_remote": base64.b64encode(bytes(msg.wireless_remote)).decode("ascii"),
        "mode_machine": int(msg.mode_machine),
    }

def handstate_to_dict(msg: HandState_, side: str) -> dict[str, Any]:
    motor_states = []
    for i in range(NUM_HAND_MOTORS):
        motor_states.append({
            "q": float(msg.motor_state[i].q),
            "dq": float(msg.motor_state[i].dq),
            "tau_est": float(msg.motor_state[i].tau_est),
        })

    press_sensors = []
    if hasattr(msg, 'press_sensor_state'):
        for p in msg.press_sensor_state:
            press_sensors.append({
                "pressure": list(p.pressure),
                "temperature": list(p.temperature)
            })

    return {
        "side": side,
        "motor_state": motor_states,
        "press_sensor_state": press_sensors,
    }


def dict_to_lowcmd(data: dict[str, Any], enable_legs: bool) -> hg_LowCmd:
    cmd = unitree_hg_msg_dds__LowCmd_()
    cmd.mode_pr = data.get("mode_pr", 0)
    cmd.mode_machine = data.get("mode_machine", 0)

    # ==========================================================
    # A MÁGICA DO CONTROLE HÍBRIDO (PRIORIDADE OVERRIDE)
    # ==========================================================
    if enable_legs:
        # Se a IA está rodando, precisamos forçar que o LowCmd do LeRobot 
        # tenha prioridade absoluta (1) em cima dos braços para não "travar".
        cmd.mode_pr = 1 

    for i, motor_data in enumerate(data.get("motor_cmd", [])):
        if enable_legs and i < 15:
            # MODO HÍBRIDO: Zera completamente pernas/cintura. A IA (que está viva) assume 100%.
            cmd.motor_cmd[i].mode = 0
            cmd.motor_cmd[i].q = 0.0
            cmd.motor_cmd[i].dq = 0.0
            cmd.motor_cmd[i].kp = 0.0
            cmd.motor_cmd[i].kd = 0.0
            cmd.motor_cmd[i].tau = 0.0
        else:
            # MODO HÍBRIDO E BRUTO: Passa os comandos do LeRobot. 
            # (Com mode_pr = 1, a IA será silenciada apenas nos braços).
            cmd.motor_cmd[i].mode = motor_data.get("mode", 0)
            cmd.motor_cmd[i].q = motor_data.get("q", 0.0)
            cmd.motor_cmd[i].dq = motor_data.get("dq", 0.0)
            cmd.motor_cmd[i].kp = motor_data.get("kp", 0.0)
            cmd.motor_cmd[i].kd = motor_data.get("kd", 0.0)
            cmd.motor_cmd[i].tau = motor_data.get("tau", 0.0)

    return cmd


def dict_to_handcmd(data: dict[str, Any]) -> HandCmd_:
    cmd = unitree_hg_msg_dds__HandCmd_()
    for i, motor_data in enumerate(data.get("motor_cmd", [])):
        cmd.motor_cmd[i].mode = motor_data.get("mode", 0)
        cmd.motor_cmd[i].q = motor_data.get("q", 0.0)
        cmd.motor_cmd[i].dq = motor_data.get("dq", 0.0)
        cmd.motor_cmd[i].kp = motor_data.get("kp", 0.0)
        cmd.motor_cmd[i].kd = motor_data.get("kd", 0.0)
        cmd.motor_cmd[i].tau = motor_data.get("tau", 0.0)
    return cmd


def state_forward_loop(lowstate_sub, lowstate_sock, state_period, shutdown_event):
    last_state_time = 0.0
    while not shutdown_event.is_set():
        msg = lowstate_sub.Read()
        if msg is None: continue
        now = time.time()
        if now - last_state_time >= state_period:
            state_dict = lowstate_to_dict(msg)
            payload = json.dumps({"topic": kTopicLowState, "data": state_dict}).encode("utf-8")
            try: lowstate_sock.send(payload, zmq.NOBLOCK)
            except (zmq.Again, zmq.error.ContextTerminated): pass
            last_state_time = now

def handstate_forward_loop(left_sub, right_sub, handstate_sock, state_period, shutdown_event):
    last_left_time = 0.0
    last_right_time = 0.0
    while not shutdown_event.is_set():
        now = time.time()
        msg_left = left_sub.Read()
        if msg_left is not None and (now - last_left_time >= state_period):
            state_dict = handstate_to_dict(msg_left, "left")
            payload = json.dumps({"topic": kTopicDex3LeftState, "data": state_dict}).encode("utf-8")
            try: handstate_sock.send(payload, zmq.NOBLOCK)
            except (zmq.Again, zmq.error.ContextTerminated): pass
            last_left_time = now
        
        msg_right = right_sub.Read()
        if msg_right is not None and (now - last_right_time >= state_period):
            state_dict = handstate_to_dict(msg_right, "right")
            payload = json.dumps({"topic": kTopicDex3RightState, "data": state_dict}).encode("utf-8")
            try: handstate_sock.send(payload, zmq.NOBLOCK)
            except (zmq.Again, zmq.error.ContextTerminated): pass
            last_right_time = now
        time.sleep(0.001)

def cmd_forward_loop(lowcmd_sock, lowcmd_pub_debug, crc, enable_legs):
    while True:
        try: payload = lowcmd_sock.recv()
        except zmq.ContextTerminated: break
        msg_dict = json.loads(payload.decode("utf-8"))
        cmd = dict_to_lowcmd(msg_dict.get("data", {}), enable_legs)
        cmd.crc = crc.Crc(cmd)
        if msg_dict.get("topic", "") == kTopicLowCommand_Debug:
            lowcmd_pub_debug.Write(cmd)

def handcmd_forward_loop(handcmd_sock, left_pub, right_pub, shutdown_event):
    while not shutdown_event.is_set():
        try: payload = handcmd_sock.recv(zmq.NOBLOCK)
        except zmq.Again:
            time.sleep(0.001)
            continue
        except zmq.ContextTerminated: break
        
        msg_dict = json.loads(payload.decode("utf-8"))
        cmd = dict_to_handcmd(msg_dict.get("data", {}))
        topic = msg_dict.get("topic", "")
        if topic == kTopicDex3LeftCommand: left_pub.Write(cmd)
        elif topic == kTopicDex3RightCommand: right_pub.Write(cmd)


def main():
    parser = argparse.ArgumentParser(description="ZMQ Bridge para Unitree G1")
    parser.add_argument('--enable-legs', action='store_true', help='Ativa o auto-equilíbrio (locomoção) com controle de braços via Override.')
    args = parser.parse_args()

    ChannelFactoryInitialize(0)
    ms = MotionSwitcher()

    if not args.enable_legs:
        print("[AVISO] DEFAULT MODO BRUTO: Desligando locomoção interna (O robô vai cair se não estiver pendurado).")
        ms.Enter_Debug_Mode()
        print("[SUCESSO] IA desligada. Robô 100% dependente do LeRobot.")
    else:
        print("[INFO] '--enable-legs' ATIVADO: O robô continuará se equilibrando nas pernas (IA Ligada).")
        print("[INFO] ZMQ rodará com Prioridade de Sobrescrita (mode_pr = 1) para dominar os braços.")
        # NÃO CHAMAMOS O DEBUG MODE AQUI. A IA CONTINUA CUIDANDO DAS PERNAS!

    crc = CRC()

    lowcmd_pub_debug = ChannelPublisher(kTopicLowCommand_Debug, hg_LowCmd)
    lowcmd_pub_debug.Init()
    lowstate_sub = ChannelSubscriber(kTopicLowState, hg_LowState)
    lowstate_sub.Init()

    left_hand_cmd_pub = ChannelPublisher(kTopicDex3LeftCommand, HandCmd_)
    left_hand_cmd_pub.Init()
    right_hand_cmd_pub = ChannelPublisher(kTopicDex3RightCommand, HandCmd_)
    right_hand_cmd_pub.Init()

    left_hand_state_sub = ChannelSubscriber(kTopicDex3LeftState, HandState_)
    left_hand_state_sub.Init()
    right_hand_state_sub = ChannelSubscriber(kTopicDex3RightState, HandState_)
    right_hand_state_sub.Init()

    ctx = zmq.Context.instance()
    lowcmd_sock = ctx.socket(zmq.PULL)
    lowcmd_sock.bind(f"tcp://0.0.0.0:{LOWCMD_PORT}")
    lowstate_sock = ctx.socket(zmq.PUB)
    lowstate_sock.bind(f"tcp://0.0.0.0:{LOWSTATE_PORT}")
    handstate_sock = ctx.socket(zmq.PUB)
    handstate_sock.bind(f"tcp://0.0.0.0:{HANDSTATE_PORT}")
    handcmd_sock = ctx.socket(zmq.PULL)
    handcmd_sock.bind(f"tcp://0.0.0.0:{HANDCMD_PORT}")

    shutdown_event = threading.Event()

    t_state = threading.Thread(target=state_forward_loop, args=(lowstate_sub, lowstate_sock, 0.002, shutdown_event))
    t_state.start()
    t_handstate = threading.Thread(target=handstate_forward_loop, args=(left_hand_state_sub, right_hand_state_sub, handstate_sock, 0.002, shutdown_event))
    t_handstate.start()
    t_handcmd = threading.Thread(target=handcmd_forward_loop, args=(handcmd_sock, left_hand_cmd_pub, right_hand_cmd_pub, shutdown_event))
    t_handcmd.start()

    print("\nBridge rodando e aguardando comandos do ZMQ...")

    try:
        cmd_forward_loop(lowcmd_sock, lowcmd_pub_debug, crc, args.enable_legs)
    except KeyboardInterrupt:
        print("\nDesligando a bridge...")
    finally:
        shutdown_event.set()
        ctx.term()
        t_state.join(timeout=2.0)
        t_handstate.join(timeout=2.0)
        t_handcmd.join(timeout=2.0)
        print("Finalizado.")

if __name__ == "__main__":
    main()